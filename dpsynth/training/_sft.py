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

"""Differentially private fine-tuning for language models.

``DPFineTuner`` is a ``DPMechanism`` that takes raw text strings and handles
everything end-to-end: tokenization, model loading, LoRA application, and
DP-SGD training via ``DPTrainer``. Gemma presets are configured with
``dpsynth.training.gemma.DPFineTunerConfig``.

Example usage::

  fine_tuner = dpsynth.training.gemma.DPFineTunerConfig(
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

import dp_accounting
from dpsynth.core import _mechanism as api
from dpsynth.training import _callbacks as callbacks
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class DPFineTuner(api.CalibratedMechanism):
  """Calibrated differentially private fine-tuner for Gemma models via DP-SGD."""

  model: SequenceModel | SequenceModelLoader
  mechanism_config: execution_plan.ExecutionPlanConfig
  optimizer: optax.GradientTransformation
  max_seq_length: int = 512
  loss_normalization: _sequence_model.LossNormalization = 'per_example_tokens'
  microbatch_size: int | None = None
  log_every: int | None = None
  checkpoint_every: int = 500
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
      data: Sequence[tuple[str, str]] | grain.MapDataset,
      *,
      metrics_fn: callbacks.MetricsFn | None = None,
      canaries: Sequence[tuple[str, str]] | None = None,
      target_prompts: Sequence[str] | None = None,
  ) -> FineTuneResult:
    """Tokenizes text, loads the model, and runs DP-SGD fine-tuning.

    When ``canaries`` are provided, they are treated as public data and split
    50/50 at random into held-in examples (appended to ``data``, potentially
    with repetition) and held-out examples for empirical privacy auditing.

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
        loss_fn=functools.partial(
            seq_model.loss_fn, loss_normalization=self.loss_normalization
        ),
        mechanism_config=self.mechanism_config,
        optimizer=self.optimizer,
        performance_flags=perf,
        log_every=self.log_every,
        checkpoint_every=self.checkpoint_every,
        canary_repeats=self.canary_repeats,
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
