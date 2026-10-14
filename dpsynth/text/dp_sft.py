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

"""Supervised fine-tuning task for Gemma language models.

``GemmaSFT`` is a ``TrainingTask`` that pairs with ``DPTrainerConfig`` to run
differentially private fine-tuning on raw ``(prompt, response)`` text pairs,
handling per-record tokenization, lazy model loading, LoRA application, and
parameter merging.

Example usage::

  config = jax_privacy.execution_plan.BandMFConfig.default(
      num_bands=1, iterations=100, expected_participations=1.0,
  )
  fine_tuner = dp_trainer.DPTrainerConfig(
      task=GemmaSFT(
          model_variant=model.GemmaModel.default('gemma4_e2b_it'),
      ),
      mechanism_config=config,
  ).configure(budget=0.5)

  data = [("What is 2+2?", "4"), ("Capital of France?", "Paris")]
  result = fine_tuner(rng=42, data=data)

For the underlying DP-SGD implementation, see ``dp_trainer``.
For model loading and loss construction, see ``model``.
"""

from __future__ import annotations

import dataclasses
import functools

from dpsynth.text import dp_trainer
from dpsynth.text import model
from flax import nnx
import jax.numpy as jnp
import optax


@dataclasses.dataclass(frozen=True)
class FineTuneResult:
  """Result of running ``GemmaSFT`` with ``DPTrainer``.

  Private training state (noise state, optimizer state) is intentionally
  excluded.

  Attributes:
    model: The LoRA-wrapped model with trained parameters merged in.
  """

  model: nnx.Module


@dataclasses.dataclass(frozen=True, kw_only=True)
class GemmaSFT(dp_trainer.TrainingTask):
  """Supervised fine-tuning ``TrainingTask`` for Gemma models with LoRA.

  Attributes:
    model_variant: Which Gemma variant to load.
    lora_rank: LoRA adapter rank.
    max_seq_length: Maximum tokenized sequence length.
  """

  model_variant: model.GemmaModel
  lora_rank: int = 16
  max_seq_length: int = 512

  # Lazily load and split the Gemma model on first access so constructing or
  # calibrating DPTrainerConfig does not load checkpoint weights.
  @functools.cached_property
  def _split_model(self):
    lora_config = model.LoraConfig(rank=self.lora_rank)
    loaded_model = model.load_gemma(self.model_variant, lora_config)
    return nnx.split(loaded_model, nnx.LoRAParam, ...)

  def init_params(self):
    """Returns initial trainable LoRA parameters cast to float32."""
    _, trainable_params, _ = self._split_model
    return optax.tree.cast(trainable_params, jnp.float32)

  def loss_fn(self, params, data, prng):
    """Merges trainable LoRA parameters into the model and computes SFT loss."""
    del prng  # Deterministic forward pass (no dropout during DP training).
    graphdef, _, frozen_params = self._split_model
    bf16_params = optax.tree.cast(params, jnp.bfloat16)
    merged = nnx.merge(graphdef, bf16_params, frozen_params)
    return model.sft_loss_fn(merged, data)

  def preprocess_fn(self, record):
    """Tokenizes a single ``(prompt, response)`` pair."""
    return model.tokenize_example(record, max_seq_length=self.max_seq_length)

  def postprocess_fn(self, params) -> FineTuneResult:
    """Merges final trained LoRA parameters into the model."""
    graphdef, _, frozen_params = self._split_model
    bf16_params = optax.tree.cast(params, jnp.bfloat16)
    merged = nnx.merge(graphdef, bf16_params, frozen_params)
    return FineTuneResult(model=merged)
