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

Usage::

  trainer = DPTrainer.default(
      init_params=my_params,
      loss_fn=my_loss,
      iterations=100,
      batch_size=8,
      num_examples=1000,
  ).configure(budget=0.5)

  final_state = trainer(rng=42, data=dataset)
"""

from __future__ import annotations

import dataclasses
import json
import math
import typing

from absl import logging
import dp_accounting
from dpsynth import api
from jax_privacy import execution_plan
from jax_privacy import training
import optax


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DPTrainer(api.DPMechanism):
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
    mechanism_config: ``BandMFConfig`` specifying the DP-SGD mechanism.
    init_params: Initial trainable parameter pytree.
    loss_fn: ``(params, batch, prng) -> scalar`` loss function. Must be
      compatible with ``jax.grad`` w.r.t. the first argument.
    optimizer: Optax gradient transformation.
    performance_flags: Optional ``PerformanceFlags`` for microbatching, SPMD,
      compute dtype, and noise generation.
    callback: Optional ``(step, state, aux) -> None`` called after each step.
  """

  init_params: training.Params
  loss_fn: training.LossFn
  mechanism_config: execution_plan.ExecutionPlanConfig
  optimizer: optax.GradientTransformation
  performance_flags: execution_plan.PerformanceFlags | None = None
  callback: training.CallbackFn | None = None

  def configure(self, _=None, *, budget, delta=0.0):
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
      return self
    cfg = typing.cast(execution_plan.BandMFConfig, self.mechanism_config)
    num_bands = len(cfg.strategy)  # pyrefly: ignore[bad-argument-type]
    rounds = math.ceil(cfg.iterations / num_bands)
    noise_multiplier = math.sqrt(rounds / (2.0 * budget))
    calibrated_config = dataclasses.replace(
        cfg,
        noise_multiplier=noise_multiplier,
    )
    return dataclasses.replace(self, mechanism_config=calibrated_config)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """The DpEvent characterizing the privacy cost of DP-SGD training."""
    if (
        hasattr(self.mechanism_config, 'noise_multiplier')
        and self.mechanism_config.noise_multiplier is None
    ):
      raise ValueError('noise_multiplier is not set. Call calibrate() first.')
    return self._make_plan().dp_event

  def _make_plan(self) -> execution_plan.DPExecutionPlan:
    return self.mechanism_config.make(performance_flags=self.performance_flags)

  def __call__(self, rng: int, data: training.Batch) -> training.TrainingState:
    """Runs DP-SGD training on the given dataset.

    Args:
      rng: Random seed for batch selection and noise generation.
      data: Training data as a dict of JAX arrays. The first axis of each array
        is the example axis.

    Returns:
      Final ``TrainingState`` containing the trained parameters.
    """
    if (
        hasattr(self.mechanism_config, 'noise_multiplier')
        and self.mechanism_config.noise_multiplier is None
    ):
      raise ValueError('noise_multiplier is not set. Call calibrate() first.')

    cfg = typing.cast(typing.Any, self.mechanism_config)
    d = dataclasses.asdict(cfg)
    if hasattr(self.mechanism_config, 'strategy') and cfg.strategy is not None:
      d['strategy'] = cfg.strategy.tolist()  # JSON/numpy hack.
    logging.info('DPTrainer config:\n%s', json.dumps(d, indent=2))

    dp_trainer = training.DPTrainer(
        config=self.mechanism_config,
        performance_flags=(
            self.performance_flags or execution_plan.PerformanceFlags()
        ),
        loss_fn=self.loss_fn,
        optimizer=self.optimizer,
    )
    return dp_trainer.fit(
        data,
        self.init_params,
        callback=self.callback,
        rng_or_seed=rng,
    )
