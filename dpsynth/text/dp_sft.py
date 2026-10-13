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

  config = execution_plan.BandMFConfig.default(
      num_bands=1, iterations=100, expected_participations=1.0,
  )
  fine_tuner = DPFineTuner(
      model_variant=model.GemmaModel.default('gemma4_e2b_it'),
      mechanism_config=config,
  ).configure(budget=0.5)

  data = [("What is 2+2?", "4"), ("Capital of France?", "Paris")]
  final_state = fine_tuner(rng=42, data=data)

For the underlying DP-SGD implementation, see ``dp_trainer``.
For model loading and loss construction, see ``model``.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math

import dp_accounting
from dpsynth import api
from dpsynth.text import dp_trainer
from dpsynth.text import model
from flax import nnx
import jax.numpy as jnp
from jax_privacy import execution_plan
import optax


@dataclasses.dataclass(frozen=True)
class FineTuneResult:
  """Result of running ``DPFineTuner``.

  Private training state (noise state, optimizer state) is intentionally
  excluded.

  Attributes:
    model: The LoRA-wrapped model with trained parameters merged in.
  """

  model: nnx.Module


@dataclasses.dataclass(frozen=True)
class DPFineTuner(api.DPMechanism):
  """Differentially private fine-tuning of Gemma models via DP-SGD.

  A ``DPMechanism`` that wraps ``DPTrainer`` with tokenization, model loading,
  and LoRA handling. All configuration is specified at construction time;
  ``__call__`` takes ``(rng, data)`` where ``data`` is a sequence of text
  strings.
  """

  model_variant: model.GemmaModel
  mechanism_config: execution_plan.BandMFConfig
  lora_rank: int = 16
  max_seq_length: int = 512
  optimizer: optax.GradientTransformation = dataclasses.field(
      default_factory=lambda: optax.adamw(1e-4)
  )
  performance_flags: execution_plan.PerformanceFlags | None = None

  def configure(self, _=None, *, budget, delta=0.0):
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
    return dataclasses.replace(self, mechanism_config=calibrated_config)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """The DpEvent characterizing the privacy cost of DP-SGD training."""
    if self.mechanism_config.noise_multiplier is None:
      raise ValueError('noise_multiplier is not set. Call calibrate() first.')
    return self.mechanism_config.make(
        performance_flags=self.performance_flags
    ).dp_event

  def __call__(
      self,
      rng: int,
      data: Sequence[tuple[str, str]],
  ) -> FineTuneResult:
    """Tokenizes text, loads the model, and runs DP-SGD fine-tuning.

    Args:
      rng: Random seed for batch selection and noise generation.
      data: Sequence of ``(prompt, response)`` string pairs.

    Returns:
      A ``FineTuneResult`` with the trained model and merged LoRA parameters.
      Private training state (noise, optimizer) is not exposed.
    """
    dataset = model.tokenize_texts(
        data,
        max_seq_length=self.max_seq_length,
    )

    lora_config = model.LoraConfig(rank=self.lora_rank)
    loaded_model = model.load_gemma(
        self.model_variant,
        lora_config,
    )
    graphdef, trainable_params, frozen_params = nnx.split(
        loaded_model, nnx.LoRAParam, ...
    )

    def loss_fn(params, data, prng):
      del prng  # Deterministic forward pass (no dropout during DP training).
      bf16_params = optax.tree.cast(params, jnp.bfloat16)
      merged = nnx.merge(graphdef, bf16_params, frozen_params)
      return model.sft_loss_fn(merged, data)

    trainer = dp_trainer.DPTrainer(
        mechanism_config=self.mechanism_config,
        init_params=optax.tree.cast(trainable_params, jnp.float32),
        loss_fn=loss_fn,
        optimizer=self.optimizer,
        performance_flags=self.performance_flags,
    )
    state = trainer(rng=rng, data=dataset)

    bf16_params = optax.tree.cast(state.params, jnp.bfloat16)
    merged = nnx.merge(graphdef, bf16_params, frozen_params)
    return FineTuneResult(model=merged)
