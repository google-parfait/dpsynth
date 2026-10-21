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

  final_state = trainer(rng=np.random.default_rng(42), data=dataset)
"""

from __future__ import annotations

import dataclasses
import json
import math
import typing

from absl import logging
import dp_accounting
from dpsynth import _checkpoint
from dpsynth import api
from dpsynth.text import callbacks
import grain.python as grain
import jax
from jax_privacy import auditing
from jax_privacy import execution_plan
from jax_privacy import training
import numpy as np
import optax


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingResult:
  """Result of running ``DPTrainer``.

  Attributes:
    params: Final trained parameter pytree.
    auditor: Optional ``CanaryScoreAuditor`` when ``canaries`` are provided.
  """

  params: training.Params
  auditor: auditing.CanaryScoreAuditor | None = None


def _setup_canaries(
    data: optax.ArrayTree | grain.MapDataset,
    canaries: optax.ArrayTree,
    rng: np.random.Generator,
) -> tuple[
    optax.ArrayTree | grain.MapDataset, optax.ArrayTree, optax.ArrayTree
]:
  """Randomly splits canaries 50/50 and appends held-in canaries to data."""
  # Permute the candidate pool using rng and split 50/50 into held-in and
  # held-out sets so membership assignment is exchangeable by construction.
  n = jax.tree.leaves(canaries)[0].shape[0]
  if n < 2:
    raise ValueError(f'canaries must have >= 2 examples, got {n}.')
  perm = rng.permutation(n)
  in_idx, out_idx = np.split(perm, 2)
  in_canaries = jax.tree.map(lambda x: x[in_idx], canaries)
  out_canaries = jax.tree.map(lambda x: x[out_idx], canaries)
  if isinstance(data, grain.MapDataset):
    get_row = lambda i: jax.tree.map(lambda x: x[i], in_canaries)
    in_data = grain.MapDataset.range(len(in_idx)).map(get_row)
    data = grain.MapDataset.concatenate([data, in_data])
  else:
    fn = lambda a, b: np.concatenate([a, b], axis=0)
    data = jax.tree.map(fn, data, in_canaries)
  return data, in_canaries, out_canaries


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

  Note: Step-level metrics logged when ``log_every`` is set (such as loss,
  batch size, and gradient norms) are un-noised and not accounted for in the
  formal ``(epsilon, delta)`` privacy guarantee. Leave ``log_every=None`` when
  strict end-to-end DP accounting of all outputs is required.

  Attributes:
    mechanism_config: ``BandMFConfig`` specifying the DP-SGD mechanism.
    init_params: Initial trainable parameter pytree.
    loss_fn: ``(params, batch, prng) -> scalar`` loss function. Must be
      compatible with ``jax.grad`` w.r.t. the first argument.
    optimizer: Optax gradient transformation.
    performance_flags: Optional ``PerformanceFlags`` for microbatching, SPMD,
      compute dtype, and noise generation.
    log_every: Optional step interval for logging un-noised training metrics.
    checkpoint_every: Step interval for saving Orbax training checkpoints when
      running inside a ``dpsynth.checkpoint(...)`` context.
  """

  init_params: training.Params
  loss_fn: training.LossFn
  mechanism_config: execution_plan.ExecutionPlanConfig
  optimizer: optax.GradientTransformation
  performance_flags: execution_plan.PerformanceFlags | None = None
  log_every: int | None = None
  checkpoint_every: int = 100

  def __post_init__(self):
    if self.log_every and self.checkpoint_every % self.log_every != 0:
      raise ValueError('checkpoint_every must be a multiple of log_every.')

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
      return DPTrainer(self)
    cfg = typing.cast(execution_plan.BandMFConfig, self.mechanism_config)
    num_bands = len(cfg.strategy)  # pyrefly: ignore[bad-argument-type]
    rounds = math.ceil(cfg.iterations / num_bands)
    noise_multiplier = math.sqrt(rounds / (2.0 * budget))
    calibrated_config = dataclasses.replace(
        cfg,
        noise_multiplier=noise_multiplier,
    )
    cfg = dataclasses.replace(self, mechanism_config=calibrated_config)
    return DPTrainer(cfg)


@dataclasses.dataclass(frozen=True, slots=True)
class DPTrainer(api.CalibratedMechanism):
  """Calibrated differentially private trainer via DP-SGD."""

  config: DPTrainerConfig

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """The DpEvent characterizing the privacy cost of DP-SGD training."""
    return self.config.mechanism_config.make().dp_event

  def __call__(
      self,
      rng: np.random.Generator,
      data: optax.ArrayTree | grain.MapDataset,
      *,
      metrics_fn: callbacks.MetricsFn | None = None,
      canaries: optax.ArrayTree | None = None,
  ) -> TrainingResult:
    """Runs DP-SGD training on the given dataset.

    Args:
      rng: A numpy random number generator.
      data: Training data as a dict of JAX arrays or a ``grain.MapDataset``.
        When passing arrays, the first axis of each array is the example axis.
      metrics_fn: Optional ``(step, metrics) -> None`` callback invoked every
        ``log_every`` steps with scalar summary metrics.
      canaries: Optional public canary candidate batch with the same PyTree
        structure as ``data`` elements. Canaries are considered public data
        (their per-example loss scores are returned in ``auditor``). When
        provided, half are randomly assigned as held-in (appended to ``data``)
        and half as held-out for auditing.

    Returns:
      A ``TrainingResult`` containing the trained parameters and optional
      canary score auditor.
    """
    cfg = typing.cast(typing.Any, self.config.mechanism_config)
    d = dataclasses.asdict(cfg)
    if hasattr(cfg, 'strategy') and cfg.strategy is not None:
      d['strategy'] = [float(x) for x in cfg.strategy]
    logging.info('DPTrainer config:\n%s', json.dumps(d, indent=2))

    in_canaries = out_canaries = {}
    if canaries is not None:
      data, in_canaries, out_canaries = _setup_canaries(data, canaries, rng)

    perf = self.config.performance_flags or execution_plan.PerformanceFlags()
    if perf.noise_seed is None:
      perf = dataclasses.replace(perf, noise_seed=int(rng.integers(2**31)))

    trainer = training.DPTrainer(
        config=self.config.mechanism_config,
        performance_flags=perf,
        loss_fn=self.config.loss_fn,
        optimizer=self.config.optimizer,
    )
    callback = callbacks.DefaultCallbackFn(self.config.log_every, metrics_fn)
    with _checkpoint.orbax_session(
        trainer.init,
        self.config.init_params,
        checkpoint_every=self.config.checkpoint_every,
        callback=callback,
    ) as (initial_state, step_callback):
      state = trainer.fit(
          data,
          initial_state,
          callback=step_callback,
          rng_or_seed=rng,
      )

    auditor = None
    if canaries is not None:
      auditor = training.audit(
          self.config.loss_fn,
          state.params,
          in_canaries,
          out_canaries,
          self.config.init_params,
          microbatch_size=perf.microbatch_size,
          prng_key=int(rng.integers(2**31)),
      )
    return TrainingResult(params=state.params, auditor=auditor)
