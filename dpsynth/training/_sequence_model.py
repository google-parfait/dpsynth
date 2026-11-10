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

"""Sequence model interface and next-token-prediction loss for SFT."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax_privacy import training
import numpy as np
import optax

Params = optax.ArrayTree
TokenBatch = jax.Array
Logits = jax.Array
Prompt = str
Response = str
MaxSeqLength = int
# Dict with 'input_tokens' and 'loss_mask' int32 arrays.
TokenizedExample = dict[str, np.typing.NDArray[np.int32]]

ForwardFn = Callable[[Params, TokenBatch], Logits]
TokenizeFn = Callable[[tuple[Prompt, Response]], TokenizedExample]
SampleFn = Callable[[Params, Sequence[Prompt], MaxSeqLength], list[Response]]


def pad_and_truncate(tree: Any, length: int) -> Any:
  """Truncates or right-pads 1D array leaves with zeros to ``length``."""
  pad = lambda x: np.pad(x[:length], (0, max(0, length - x.size)))
  return jax.tree.map(pad, tree)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SequenceModel:
  """Differentiable sequence model for ``DPFineTuner``.

  Attributes:
    params: Trainable parameter pytree.
    forward_fn: Differentiable ``(params, tokens) -> logits`` function mapping
      trainable parameters and ``int32`` token IDs ``[B, L]`` to vocabulary
      logits ``[B, L, V]``.
    tokenize_fn: ``example -> tokenized_dict`` tokenizer mapping a ``(prompt,
      response)`` pair to unpadded 1D host NumPy ``int32`` arrays.
    sample_fn: Optional ``(params, prompts, max_seq_length) -> responses``
      function for generating responses from trained parameters.
  """

  params: Params
  forward_fn: ForwardFn
  tokenize_fn: TokenizeFn
  sample_fn: SampleFn | None = None

  def loss_fn(
      self,
      params: training.Params,
      data: Any,
      prng: jax.Array,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Cross-entropy next-token-prediction loss compatible with ``DPTrainer``."""
    del prng  # Deterministic forward pass (no dropout during DP training).
    input_tokens = data['input_tokens']
    loss_mask = data['loss_mask']
    logits = self.forward_fn(params, input_tokens)[:, :-1, :]
    targets = input_tokens[:, 1:]
    mask = loss_mask[:, 1:]
    pt_losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    loss = jnp.sum(pt_losses * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    return loss, {'loss': loss}
