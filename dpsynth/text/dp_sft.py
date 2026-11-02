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

  config = jax_privacy.execution_plan.BandMFConfig.default(
      num_bands=1, iterations=100, expected_participations=1.0,
  )
  fine_tuner = DPFineTunerConfig(
      model='gemma4_e2b_it',
      mechanism_config=config,
  ).configure(budget=0.5)

  data = [("What is 2+2?", "4"), ("Capital of France?", "Paris")]
  result = fine_tuner(rng=np.random.default_rng(42), data=data)

For the underlying DP-SGD implementation, see ``dp_trainer``.
For model loading and loss construction, see ``model``.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import math
import os

import dp_accounting
from dpsynth import api
from dpsynth.text import callbacks
from dpsynth.text import dp_trainer
from dpsynth.text import model
from flax import nnx
import grain.python as grain
import jax
import jax.numpy as jnp
import jax_privacy
from jax_privacy import execution_plan
from jax_privacy import noise_addition
import numpy as np
import optax


def _tokenize_examples(examples, tok_fn):
  """Tokenizes a non-empty sequence of examples into a stacked PyTree."""
  records = [tok_fn(ex) for ex in examples]
  return jax.tree.map(lambda *xs: np.stack(xs), *records)


@dataclasses.dataclass(frozen=True)
class FineTuneResult:
  """Result of running ``DPFineTuner``.

  Private training state (noise state, optimizer state) is intentionally
  excluded.

  Attributes:
    model: The LoRA-wrapped model with trained parameters merged in.
    auditor: Optional ``CanaryScoreAuditor`` when ``canaries`` are provided.
    samples: Optional ``(prompt, response)`` pairs from ``target_prompts``.
  """

  model: nnx.Module
  auditor: jax_privacy.auditing.CanaryScoreAuditor | None = None
  samples: list[tuple[str, str]] | None = None

  def save(self, path: str | os.PathLike[str]) -> None:
    """Saves trainable LoRA parameters from ``self.model`` to ``path``."""
    model.save_lora_params(self.model, path)


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
    mechanism_config: ``BandMFConfig`` specifying the DP-SGD mechanism.
    lora_rank: Rank of the LoRA adapter matrices.
    max_seq_length: Maximum token sequence length (including special tokens).
    optimizer: Optax gradient transformation for parameter updates.
    microbatch_size: Physical microbatch size for per-example gradient clipping.
    log_every: Optional step interval for logging un-noised training metrics.
    checkpoint_every: Step interval for saving mid-training Orbax checkpoints.
  """

  model: model.ModelName
  mechanism_config: execution_plan.BandMFConfig
  lora_rank: int = 16
  max_seq_length: int = 512
  optimizer: optax.GradientTransformation = dataclasses.field(
      default_factory=lambda: optax.adamw(1e-3)
  )
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
    num_bands = len(self.mechanism_config.strategy)  # pyrefly: ignore[bad-argument-type]
    rounds = math.ceil(self.mechanism_config.iterations / num_bands)
    noise_multiplier = math.sqrt(rounds / (2.0 * budget))
    calibrated_config = dataclasses.replace(
        self.mechanism_config,
        noise_multiplier=noise_multiplier,
    )
    cfg = dataclasses.replace(self, mechanism_config=calibrated_config)
    return DPFineTuner(cfg)


@dataclasses.dataclass(frozen=True)
class DPFineTuner(api.CalibratedMechanism):
  """Calibrated differentially private fine-tuner for Gemma models via DP-SGD."""

  config: DPFineTunerConfig

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """The DpEvent characterizing the privacy cost of DP-SGD training."""
    return self.config.mechanism_config.make().dp_event

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
        pairs.
      metrics_fn: Optional ``(step, metrics) -> None`` callback invoked every
        ``log_every`` steps with scalar summary metrics.
      canaries: Optional public ``(prompt, response)`` pairs for DP auditing.
      target_prompts: Optional prompts to sample from for output spot-checking.

    Returns:
      A ``FineTuneResult`` with the trained model and optional canary score
      auditor and sampled ``(prompt, response)`` pairs. Private training state
      (noise, optimizer) is not exposed.
    """
    # 1-to-1 per-record mapping preserves record-level differential privacy.
    tok_fn = functools.partial(
        model.tokenize_example, max_seq_length=self.config.max_seq_length
    )
    canary_data = _tokenize_examples(canaries, tok_fn) if canaries else None

    if isinstance(data, grain.MapDataset):
      dataset = data.map(tok_fn)
    else:
      dataset = _tokenize_examples(data, tok_fn)

    lora_config = model.LoraConfig(rank=self.config.lora_rank)
    loaded_model = model.load_gemma(
        self.config.model,
        lora_config,
    )
    graphdef, trainable_params, frozen_params = nnx.split(
        loaded_model, nnx.LoRAParam, ...
    )
    del loaded_model

    def loss_fn(params, data, prng):
      del prng  # Deterministic forward pass (no dropout during DP training).
      bf16_params = optax.tree.cast(params, jnp.bfloat16)
      merged = nnx.merge(graphdef, bf16_params, frozen_params)
      return model.sft_loss_fn(merged, data)

    mesh = jax.sharding.get_mesh()
    if not mesh.empty and 'replica' in mesh.axis_names:
      perf = execution_plan.PerformanceFlags(
          microbatch_size=self.config.microbatch_size,
          batch_axis_name='replica',
          intermediate_strategy=noise_addition.SupportedStrategies.ZERO,
          param_specs=jax.tree.map(lambda x: x.sharding, trainable_params),
      )
    else:
      perf = execution_plan.PerformanceFlags(
          microbatch_size=self.config.microbatch_size,
      )

    # Keep master LoRA weights in float32 so small updates do not underflow.
    trainer_config = dp_trainer.DPTrainerConfig(
        mechanism_config=self.config.mechanism_config,
        init_params=optax.tree.cast(trainable_params, jnp.float32),
        loss_fn=loss_fn,
        optimizer=self.config.optimizer,
        performance_flags=perf,
        log_every=self.config.log_every,
        checkpoint_every=self.config.checkpoint_every,
    )
    del trainable_params
    trainer = dp_trainer.DPTrainer(trainer_config)
    state = trainer(rng, dataset, metrics_fn=metrics_fn, canaries=canary_data)

    bf16_params = optax.tree.cast(state.params, jnp.bfloat16)
    merged = nnx.merge(graphdef, bf16_params, frozen_params)

    samples = None
    if target_prompts is not None:
      max_len = self.config.max_seq_length
      sampler = model.GemmaSampler(merged, max_seq_length=max_len)
      samples = list(zip(target_prompts, sampler(target_prompts), strict=True))

    return FineTuneResult(model=merged, auditor=state.auditor, samples=samples)
