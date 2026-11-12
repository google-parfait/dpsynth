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

``DPTrainerConfig`` and ``DPTrainer`` wrap ``jax_privacy`` in DPSynth's
configure / calibrate / run lifecycle, operating on arbitrary JAX parameter
pytrees and loss functions without depending on Flax or model-specific logic.

Example usage::

  trainer = DPTrainerConfig(
      init_params=my_params,
      loss_fn=my_loss_fn,
      iterations=100,
      epochs=1.0,
  ).configure(budget=0.5)

  result = trainer(rng=np.random.default_rng(42), data=dataset)
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
from dpsynth.training import _callbacks as callbacks
from dpsynth.training import _optimizers
import grain.python as grain
import jax
from jax.experimental import multihost_utils
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
    repeats: int = 1,
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
  in_idx, out_idx = np.array_split(perm, 2)
  in_canaries = jax.tree.map(lambda x: x[in_idx], canaries)
  out_canaries = jax.tree.map(lambda x: x[out_idx], canaries)
  if isinstance(data, grain.MapDataset):
    get_row = lambda i: jax.tree.map(lambda x: x[i], in_canaries)
    in_data = grain.MapDataset.range(len(in_idx)).map(get_row)
    data = grain.MapDataset.concatenate([data] + [in_data] * repeats)
  else:
    fn = lambda a, b: np.concatenate([a] + [b] * repeats, axis=0)
    data = jax.tree.map(fn, data, in_canaries)
  return data, in_canaries, out_canaries


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DPTrainerConfig(api.MechanismConfig):
  """General-purpose differentially private training via DP-SGD.

  Wraps ``DPTrainer`` in the ``DPMechanism`` protocol, providing calibration,
  privacy accounting, and a clean ``(rng, data)`` call signature.

  The caller is responsible for preparing ``init_params`` and ``loss_fn``. For
  Flax NNX models, this means splitting the model into trainable and frozen
  state and closing over the frozen state in the loss function. For pure JAX
  code, the params and loss function can be used directly.

  Note: Step-level metrics logged when ``log_every`` is set (such as loss,
  batch size, and gradient norms) are un-noised and not accounted for in the
  formal ``(epsilon, delta)`` privacy guarantee. Leave ``log_every=None`` when
  strict end-to-end DP accounting of all outputs is required.

  Attributes:
    init_params: Initial trainable parameter pytree.
    loss_fn: ``(params, batch, prng) -> (scalar_loss, aux_dict)`` loss function.
    iterations: Total number of training iterations.
    epochs: Expected number of times each example participates across training.
    num_bands: Number of bands in the BandMF strategy matrix.
    clipping_norm: Maximum L2 norm of per-example gradients.
    learning_rate: Learning rate for the default optimizer. To use a custom
      optimizer, call ``dataclasses.replace(mechanism, optimizer=...)`` on the
      calibrated mechanism returned by ``configure`` or ``dpsynth.calibrate``.
    microbatch_size: Physical microbatch size for per-example gradient clipping.
    log_every: Optional step interval for logging un-noised training metrics.
    checkpoint_every: Step interval for saving mid-training Orbax checkpoints.
    canary_repeats: Held-in canary repetitions for empirical group privacy.
  """

  init_params: training.Params
  loss_fn: training.LossFn
  iterations: int
  epochs: float
  num_bands: int = 1
  clipping_norm: float = 1.0
  learning_rate: float = 1e-3
  microbatch_size: int | None = None
  log_every: int | None = None
  checkpoint_every: int = 100
  canary_repeats: int = 1

  def __post_init__(self):
    if self.log_every and self.checkpoint_every % self.log_every != 0:
      raise ValueError('checkpoint_every must be a multiple of log_every.')
    if self.canary_repeats < 1:
      raise ValueError('canary_repeats must be >= 1.')

  def configure(self, _=None, *, budget, delta=0.0) -> DPTrainer:
    """Returns a ``DPTrainer`` with noise calibrated to the zCDP budget.

    Uses a loose upper bound ignoring subsampling amplification:
    ``sigma = sqrt(T / (2 * budget))``. The ``dp_event`` property returns the
    full event with amplification for tight downstream accounting.

    Args:
      budget: Dummy calibration budget (roughly on a zCDP rho scale).
      delta: Unused. Accepted for interface compatibility.

    Returns:
      A new ``DPTrainer`` with calibrated ``mechanism_config.noise_multiplier``.
    """
    del delta
    rounds = math.ceil(self.iterations / self.num_bands)
    noise_multiplier = math.sqrt(rounds / (2.0 * budget))
    mechanism_config = execution_plan.BandMFConfig.default(
        num_bands=self.num_bands,
        iterations=self.iterations,
        expected_participations=self.epochs,
        l2_clip_norm=self.clipping_norm,
        noise_multiplier=noise_multiplier,
    )
    perf = execution_plan.PerformanceFlags(microbatch_size=self.microbatch_size)
    return DPTrainer(
        init_params=self.init_params,
        loss_fn=self.loss_fn,
        mechanism_config=mechanism_config,
        optimizer=_optimizers.denoised_adamw(self.learning_rate),
        performance_flags=perf,
        log_every=self.log_every,
        checkpoint_every=self.checkpoint_every,
        canary_repeats=self.canary_repeats,
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DPTrainer(api.CalibratedMechanism):
  """Calibrated differentially private trainer via DP-SGD."""

  init_params: training.Params
  loss_fn: training.LossFn
  mechanism_config: execution_plan.ExecutionPlanConfig
  optimizer: optax.GradientTransformation
  performance_flags: execution_plan.PerformanceFlags | None = None
  log_every: int | None = None
  checkpoint_every: int = 100
  canary_repeats: int = 1

  def __post_init__(self):
    if self.log_every and self.checkpoint_every % self.log_every != 0:
      raise ValueError('checkpoint_every must be a multiple of log_every.')
    if self.canary_repeats < 1:
      raise ValueError('canary_repeats must be >= 1.')
    if getattr(self.mechanism_config, 'noise_multiplier', 0.0) is None:
      raise ValueError('noise_multiplier is not set on mechanism_config.')

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """The DpEvent characterizing the privacy cost of DP-SGD training."""
    return self.mechanism_config.make().dp_event

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
      data: Training dataset as either (1) an ``ArrayTree`` (e.g. dict or tuple
        of arrays) where all leaves share leading example dimension ``(N, ...)``
        or (2) a ``grain.MapDataset`` whose elements ``data[i]`` are single
        unbatched example PyTrees of shape ``(...)``. In both cases, ``loss_fn``
        always receives a batched PyTree with leading axis ``(B, ...)``.
      metrics_fn: Optional ``(step, metrics) -> None`` callback invoked every
        ``log_every`` steps with scalar summary metrics.
      canaries: Optional batched ``ArrayTree`` of public canary candidates with
        a leading dimension of at least 2 examples (``N >= 2``) and the same
        per-example PyTree structure and leaf shapes as ``data``. Half are
        randomly assigned as held-in (appended to ``data``, potentially with
        repetition) and half as held-out for empirical privacy auditing in
        ``TrainingResult.auditor``.

    Returns:
      A ``TrainingResult`` containing the trained parameters and optional
      canary score auditor.
    """
    cfg = typing.cast(typing.Any, self.mechanism_config)
    d = dataclasses.asdict(cfg)
    if hasattr(cfg, 'strategy') and cfg.strategy is not None:
      d['strategy'] = [float(x) for x in cfg.strategy]
    logging.info('DPTrainer config:\n%s', json.dumps(d, indent=2))

    if jax.process_index() == 0:
      seed = _checkpoint.get_or_compute(
          'rng_seed', lambda r: r.integers(2**31), rng=rng
      )
    else:
      seed = 0  # Dummy placeholder; overwritten by broadcast_one_to_all below.
    seed = multihost_utils.broadcast_one_to_all(np.asarray(seed, np.int32))
    # Seed a deterministic PRNG from rng so multi-host and resumed runs replay
    # identical batch-selection and noise streams even for seedless CSPRNGs.
    rng = np.random.default_rng(int(seed))

    in_canaries = out_canaries = {}
    if canaries is not None:
      data, in_canaries, out_canaries = _setup_canaries(
          data, canaries, rng, self.canary_repeats
      )

    perf = self.performance_flags or execution_plan.PerformanceFlags()
    if perf.noise_seed is None:
      perf = dataclasses.replace(perf, noise_seed=int(rng.integers(2**31)))

    trainer = training.DPTrainer(
        config=self.mechanism_config,
        performance_flags=perf,
        loss_fn=self.loss_fn,
        optimizer=self.optimizer,
    )
    callback = callbacks.DefaultCallbackFn(self.log_every, metrics_fn)
    # Copy init_params since DPTrainer.fit donates state buffers.
    init_params = jax.tree.map(jax.numpy.copy, self.init_params)
    with _checkpoint.orbax_session(
        trainer.init,
        init_params,
        checkpoint_every=self.checkpoint_every,
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
          self.loss_fn,
          state.params,
          in_canaries,
          out_canaries,
          self.init_params,
          microbatch_size=perf.microbatch_size,
          prng_key=int(rng.integers(2**31)),
      )
    return TrainingResult(params=state.params, auditor=auditor)
