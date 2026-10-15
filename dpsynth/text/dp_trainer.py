# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General-purpose differentially private training for JAX models.

``DPTrainer`` is a ``DPMechanism`` that runs DP-SGD on an arbitrary JAX loss
function and parameter pytree. It wraps ``jax_privacy`` — which provides
per-example gradient clipping, noise addition, and privacy accounting — and
exposes the same three-phase configure → calibrate → run API as other dpsynth
mechanisms (``DPHistogram``, ``DPMarginals``, etc.).

**Design rationale.** The key design decision is framework-agnosticism:
``DPTrainer`` operates on pure JAX pytrees and callables, with no dependency on
Flax, NNX, or any model-specific concepts like LoRA or tokenization. This
preserves the full generality of ``jax_privacy`` (which itself is
framework-agnostic) while fitting into dpsynth's ``DPMechanism`` protocol for
uniform calibration and composition. Supervised fine-tuning of Flax NNX models
is supported as a special case: the caller splits the ``nnx.Module`` into
trainable and frozen state, closes over the frozen state in the loss function,
and passes the trainable pytree to ``DPTrainer``. The higher-level
``DPTextTrainer`` handles this NNX plumbing, along with Gemma model loading,
LoRA application, tokenization, and checkpoint management — keeping this module
focused on the DP-SGD training loop itself.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
import dataclasses
import json
import math
import typing
from typing import Any

from absl import logging
import dp_accounting
from dpsynth import api
import jax
from jax_privacy import execution_plan
from jax_privacy import noise_addition
from jax_privacy import training
import numpy as np
import optax


class TrainingTask(abc.ABC):
  """Non-private specification of model parameter initialization and loss."""

  @abc.abstractmethod
  def init_params(self) -> training.Params:
    """Returns the initial trainable parameter pytree."""

  @abc.abstractmethod
  def loss_fn(
      self,
      params: training.Params,
      data: training.Batch,
      prng: jax.Array,
  ) -> tuple[jax.Array, Any]:
    """Computes ``(loss, aux)`` for a batch."""

  def preprocess_fn(self, record: Any) -> training.Batch:
    """Transforms a single input record into an example pytree."""
    return record

  def postprocess_fn(self, params: training.Params) -> Any:
    """Transforms the final trained ``Params`` pytree into the output result."""
    return params


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DPTrainerConfig(api.MechanismConfig):
  """General-purpose differentially private training via DP-SGD.

  Wraps ``jax_privacy.DPTrainer`` in the ``DPMechanism`` protocol, providing
  calibration, privacy accounting, and a clean ``(rng, data)`` call signature.

  The caller is responsible for preparing ``init_params`` and ``loss_fn``. For
  Flax NNX models, this means splitting the model into trainable and frozen
  state and closing over the frozen state in the loss function. For pure JAX
  code, the params and loss function can be used directly.

  Mechanism can be configured via `jax_privacy.execution_plan.BandMFConfig`.
  To configure a basic 4-epoch DP-SGD, use:

    >>> config = execution_plan.BandMFConfig.default(
    ...   num_bands=1, iterations=500, expected_participations=4, l2_clip_norm=1
    ... )

  Attributes:
    task: ``TrainingTask`` specifying parameter initialization and loss.
    mechanism_config: ``BandMFConfig`` specifying the DP-SGD mechanism.
    optimizer: Optax gradient transformation.
    microbatch_size: Optional microbatch size for per-example gradient clipping.
    performance_flags: Optional ``PerformanceFlags`` for microbatching, SPMD,
      compute dtype, and noise generation.
    callback: Optional ``(step, state, aux) -> None`` called after each step.
  """

  task: TrainingTask
  mechanism_config: execution_plan.ExecutionPlanConfig
  optimizer: optax.GradientTransformation = dataclasses.field(
      default_factory=lambda: optax.adamw(1e-4)
  )
  microbatch_size: int | None = None
  performance_flags: execution_plan.PerformanceFlags | None = None
  callback: training.CallbackFn | None = None

  def configure(self, _=None, *, budget, delta=0.0) -> DPTrainer:
    """Returns a copy with noise calibrated to the zCDP budget.

    Uses a loose upper bound ignoring subsampling amplification:
    ``sigma = sqrt(T / (2 * budget))``. The ``dp_event`` property returns the
    full event with amplification for tight downstream accounting.

    Args:
      budget: Dummy calibration budget (roughly on a zCDP rho scale).
      delta: Unused. Accepted for interface compatibility.

    Returns:
      A new ``DPTrainer`` with calibrated ``config.noise_multiplier``.
    """
    if isinstance(self.mechanism_config, execution_plan.NonPrivateConfig):
      return DPTrainer(config=self)
    cfg = typing.cast(execution_plan.BandMFConfig, self.mechanism_config)
    num_bands = len(cfg.strategy)  # pyrefly: ignore[bad-argument-type]
    rounds = math.ceil(cfg.iterations / num_bands)
    noise_multiplier = math.sqrt(rounds / (2.0 * budget))
    calibrated_config = dataclasses.replace(
        cfg,
        noise_multiplier=noise_multiplier,
    )
    return DPTrainer(
        config=dataclasses.replace(self, mechanism_config=calibrated_config)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class DPTrainer(api.CalibratedMechanism):
  """Calibrated differentially private trainer via DP-SGD."""

  config: DPTrainerConfig

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """The DpEvent characterizing the privacy cost of DP-SGD training."""
    return self._make_plan().dp_event

  def _make_plan(self) -> execution_plan.DPExecutionPlan:
    return self.config.mechanism_config.make(
        performance_flags=self.config.performance_flags
    )

  def __call__(self, rng: int, data: Sequence[Any]) -> Any:
    """Runs DP-SGD training on a sequence of records.

    Args:
      rng: Random seed for batch selection and noise generation.
      data: Sequence of input records. ``config.task.preprocess_fn`` is applied
        to each record independently and stacked into an array tree along axis
        0.

    Returns:
      The result of ``config.task.postprocess_fn(state.params)``.
    """
    cfg = typing.cast(typing.Any, self.config.mechanism_config)
    d = dataclasses.asdict(cfg)
    if hasattr(cfg, 'strategy') and cfg.strategy is not None:
      d['strategy'] = cfg.strategy.tolist()  # JSON/numpy hack.
    logging.info('DPTrainer config:\n%s', json.dumps(d, indent=2))

    task = self.config.task
    examples = [task.preprocess_fn(record) for record in data]
    batch_data = jax.tree.map(lambda *xs: np.stack(xs), *examples)

    init_params = task.init_params()
    perf = self.config.performance_flags or execution_plan.PerformanceFlags()
    perf = dataclasses.replace(
        perf,
        noise_seed=rng,
        microbatch_size=self.config.microbatch_size or perf.microbatch_size,
    )
    mesh = jax.sharding.get_mesh()
    if not mesh.empty and 'replica' in mesh.axis_names:
      perf = dataclasses.replace(
          perf,
          batch_axis_name='replica',
          intermediate_strategy=noise_addition.SupportedStrategies.ZERO,
          param_specs=jax.tree.map(lambda x: x.sharding, init_params),
      )

    dp_trainer = training.DPTrainer(
        config=self.config.mechanism_config,
        performance_flags=perf,
        loss_fn=task.loss_fn,
        optimizer=self.config.optimizer,
    )
    state = dp_trainer.fit(
        batch_data,
        init_params,
        callback=self.config.callback,
        rng_or_seed=rng,
    )
    return task.postprocess_fn(state.params)
