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

"""Differentially private fine-tuning for Gemma language models.

``DPFineTuner`` is a ``DPMechanism`` that takes raw text strings and handles
everything end-to-end: tokenization, model loading, LoRA application, and
DP-SGD training via ``DPTrainer``.

Example usage::

  fine_tuner = DPFineTunerConfig(
      model='gemma4_e2b_it',
      iterations=100,
      epochs=1.0,
  ).configure(budget=0.5)

  data = [("What is 2+2?", "4"), ("Capital of France?", "Paris")]
  result = fine_tuner(rng=np.random.default_rng(42), data=data)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
import math

import dp_accounting
from dpsynth import api
from dpsynth.training import _callbacks as callbacks
from dpsynth.training import _gemma as model
from dpsynth.training import _sequence_model
from dpsynth.training import _trainer as dp_trainer
import grain.python as grain
import jax
import jax_privacy
from jax_privacy import execution_plan
from jax_privacy import noise_addition
import numpy as np
import optax

SequenceModel = _sequence_model.SequenceModel
SequenceModelLoader = Callable[[], SequenceModel]


def _tokenize_examples(examples, tok_fn, pad_fn):
  """Tokenizes and pads a non-empty sequence of examples into a stacked PyTree."""
  records = [pad_fn(tok_fn(ex)) for ex in examples]
  return jax.tree.map(lambda *xs: np.stack(xs), *records)


@dataclasses.dataclass(frozen=True)
class FineTuneResult:
  """Result of running ``DPFineTuner``.

  Private training state (noise state, optimizer state) is intentionally
  excluded.

  Attributes:
    model: ``SequenceModel`` with trained parameters.
    auditor: Optional ``CanaryScoreAuditor`` when ``canaries`` are provided.
    samples: Optional ``(prompt, response)`` pairs from ``target_prompts``.
  """

  model: _sequence_model.SequenceModel
  auditor: jax_privacy.auditing.CanaryScoreAuditor | None = None
  samples: list[tuple[str, str]] | None = None


@dataclasses.dataclass(frozen=True)
class DPFineTunerConfig(api.MechanismConfig):
  """Differentially private fine-tuning of Gemma models via DP-SGD.

  A ``DPMechanism`` that wraps ``DPTrainer`` with tokenization, model loading,
  and LoRA handling. All configuration is specified at construction time;
  ``__call__`` takes ``(rng, data)`` where ``data`` is a sequence or
  ``grain.MapDataset`` of ``(prompt, response)`` string pairs.

  Note: Step-level metrics logged when ``log_every`` is set (such as loss,
  batch size, and gradient norms) are un-noised and not accounted for in the
  formal ``(epsilon, delta)`` privacy guarantee. Leave ``log_every=None`` when
  strict end-to-end DP accounting of all outputs is required.

  Attributes:
    model: Gemma model preset name.
    iterations: Total number of training iterations.
    epochs: Expected number of times each example participates across training.
    num_bands: Number of bands in the BandMF strategy matrix.
    clipping_norm: Maximum L2 norm of per-example gradients.
    lora_rank: Rank of the LoRA adapter matrices.
    max_seq_length: Maximum token sequence length (including special tokens).
    learning_rate: Learning rate for the default optimizer. To use a custom
      optimizer, call ``dataclasses.replace(mechanism, optimizer=...)`` on the
      calibrated mechanism returned by ``configure`` or ``dpsynth.calibrate``.
    microbatch_size: Physical microbatch size for per-example gradient clipping.
    log_every: Optional step interval for logging un-noised training metrics.
    checkpoint_every: Step interval for saving mid-training Orbax checkpoints.
  """

  model: model.ModelName
  iterations: int
  epochs: float
  num_bands: int = 1
  clipping_norm: float = 1.0
  lora_rank: int = 16
  max_seq_length: int = 512
  learning_rate: float = 1e-3
  microbatch_size: int | None = None
  log_every: int | None = None
  checkpoint_every: int = 500

  def __post_init__(self):
    model.validate_model_name(self.model)
    if self.log_every and self.checkpoint_every % self.log_every != 0:
      raise ValueError('checkpoint_every must be a multiple of log_every.')

  def configure(self, _=None, *, budget, delta=0.0) -> DPFineTuner:
    """Returns a copy with noise_multiplier calibrated to the zCDP budget.

    Sets the noise_multiplier to satisfy ``budget`` under a **loose upper
    bound** that ignores Poisson subsampling amplification: the unamplified
    composition of ``T`` Gaussian mechanisms with noise_multiplier ``sigma``
    has zCDP cost ``T / (2 * sigma**2)``, so we set
    ``sigma = sqrt(T / (2 * budget))``.

    This is deliberately conservative. The ``dp_event`` property returns the
    **full** event including subsampling amplification, so downstream callers
    using ``dp_accounting`` (e.g. ``calibrate_dp_mechanism`` over a
    heterogeneous composition) will get tight PLD-based accounting from the
    raw events -- not from this loose zCDP bound.

    Args:
      budget: The privacy budget.
      delta: Unused. Accepted for interface compatibility.

    Returns:
      A new ``DPFineTuner`` with calibrated
      ``mechanism_config.noise_multiplier``.
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
    # Defer loading until __call__ so calibration binary search does not
    # instantiate weights and loading runs inside the caller's device mesh.
    model_fn = functools.partial(
        model.load_gemma_sequence_model, self.model, lora_rank=self.lora_rank
    )
    return DPFineTuner(
        model=model_fn,
        mechanism_config=mechanism_config,
        optimizer=optax.adamw(self.learning_rate),
        max_seq_length=self.max_seq_length,
        microbatch_size=self.microbatch_size,
        log_every=self.log_every,
        checkpoint_every=self.checkpoint_every,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class DPFineTuner(api.CalibratedMechanism):
  """Calibrated differentially private fine-tuner for Gemma models via DP-SGD."""

  model: SequenceModel | SequenceModelLoader
  mechanism_config: execution_plan.ExecutionPlanConfig
  optimizer: optax.GradientTransformation
  max_seq_length: int = 512
  microbatch_size: int | None = None
  log_every: int | None = None
  checkpoint_every: int = 500

  def __post_init__(self):
    if self.log_every and self.checkpoint_every % self.log_every != 0:
      raise ValueError('checkpoint_every must be a multiple of log_every.')
    if getattr(self.mechanism_config, 'noise_multiplier', 0.0) is None:
      raise ValueError('noise_multiplier is not set on mechanism_config.')

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """The DpEvent characterizing the privacy cost of DP-SGD training."""
    return self.mechanism_config.make().dp_event

  def __call__(
      self,
      rng: np.random.Generator,
      data: Sequence[tuple[str, str]] | grain.MapDataset,
      *,
      metrics_fn: callbacks.MetricsFn | None = None,
      canaries: Sequence[tuple[str, str]] | None = None,
      target_prompts: Sequence[str] | None = None,
  ) -> FineTuneResult:
    """Tokenizes text, loads the model, and runs DP-SGD fine-tuning.

    When ``canaries`` are provided, they are treated as public data and split
    50/50 at random into held-in examples (appended to ``data``) and held-out
    examples for empirical privacy auditing.

    When ``target_prompts`` are provided, ``GemmaSampler`` generates synthetic
    responses at the end of training using default sampling settings. This is
    primarily intended to spot-check synthetic data quality; for custom sampling
    configuration or large-scale generation, run offline sampling on the
    fine-tuned model instead.

    Args:
      rng: A numpy random number generator.
      data: Sequence or ``grain.MapDataset`` of ``(prompt, response)`` string
        pairs. Only ``response`` tokens contribute to the training loss; for
        unconditional text generation, pair each document with a constant
        instruction prompt (e.g., ``("Write a research abstract.", doc)``).
      metrics_fn: Optional ``(step, metrics) -> None`` callback invoked every
        ``log_every`` steps with scalar summary metrics.
      canaries: Optional public ``(prompt, response)`` pairs for DP auditing.
      target_prompts: Optional prompts to sample from for output spot-checking.

    Returns:
      A ``FineTuneResult`` with the trained model and optional canary score
      auditor and sampled ``(prompt, response)`` pairs. Private training state
      (noise, optimizer) is not exposed.
    """
    seq_model = (
        self.model if isinstance(self.model, SequenceModel) else self.model()
    )

    # 1-to-1 per-record mapping preserves record-level differential privacy.
    tok_fn = seq_model.tokenize_fn
    pad_fn = functools.partial(
        _sequence_model.pad_and_truncate, length=self.max_seq_length
    )
    canary_data = (
        _tokenize_examples(canaries, tok_fn, pad_fn) if canaries else None
    )

    if isinstance(data, grain.MapDataset):
      dataset = data.map(tok_fn).map(pad_fn)
    else:
      dataset = _tokenize_examples(data, tok_fn, pad_fn)

    trainable_params = seq_model.params
    mesh = jax.sharding.get_mesh()
    if not mesh.empty and 'replica' in mesh.axis_names:
      perf = execution_plan.PerformanceFlags(
          microbatch_size=self.microbatch_size,
          batch_axis_name='replica',
          intermediate_strategy=noise_addition.SupportedStrategies.ZERO,
          param_specs=jax.tree.map(lambda x: x.sharding, trainable_params),
      )
    else:
      perf = execution_plan.PerformanceFlags(
          microbatch_size=self.microbatch_size,
      )

    trainer = dp_trainer.DPTrainer(
        init_params=trainable_params,
        loss_fn=seq_model.loss_fn,
        mechanism_config=self.mechanism_config,
        optimizer=self.optimizer,
        performance_flags=perf,
        log_every=self.log_every,
        checkpoint_every=self.checkpoint_every,
    )
    del trainable_params
    state = trainer(rng, dataset, metrics_fn=metrics_fn, canaries=canary_data)

    trained_model = dataclasses.replace(seq_model, params=state.params)

    samples = None
    if target_prompts is not None and trained_model.sample_fn is not None:
      responses = trained_model.sample_fn(
          state.params, target_prompts, self.max_seq_length
      )
      samples = list(zip(target_prompts, responses, strict=True))

    return FineTuneResult(
        model=trained_model, auditor=state.auditor, samples=samples
    )
