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
from typing import Any, Literal

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

# How each example's summed next-token loss is normalized. Only positions where
# ``loss_mask`` is set (response tokens, not prompt or padding) contribute to
# the loss; "loss-bearing tokens" below means those positions.
#
# - ``'per_example_tokens'``: Divide by the example's number of loss-bearing
#   tokens. Even without clipping, every example has equal weight regardless of
#   length.
# - ``'sequence_length'``: Divide by the padded sequence length. Without
#   clipping, every loss-bearing token has equal weight regardless of which
#   example it is in, as in non-private training; tokens in examples that are
#   clipped may have lower effective weight. Per-example losses and gradients
#   shrink by the example's fraction of loss-bearing tokens, so a smaller
#   clipping norm may be appropriate.
LossNormalization = Literal['per_example_tokens', 'sequence_length']


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
      *,
      loss_normalization: LossNormalization = 'per_example_tokens',
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Cross-entropy next-token-prediction loss compatible with ``DPTrainer``.

    Args:
      params: Trainable parameter pytree.
      data: Dict with ``'input_tokens'`` and ``'loss_mask'`` (int32 ``[B, L]``).
      prng: Unused random key.
      loss_normalization: How the summed loss is normalized (see
        ``LossNormalization``).

    Returns:
      ``(loss, aux)`` where ``aux`` contains ``'loss'``.
    """
    del prng  # Deterministic forward pass (no dropout during DP training).
    input_tokens = data['input_tokens']
    loss_mask = data['loss_mask']
    logits = self.forward_fn(params, input_tokens)[:, :-1, :]
    targets = input_tokens[:, 1:]
    mask = loss_mask[:, 1:]
    pt_losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    match loss_normalization:
      case 'per_example_tokens':
        num_tokens = jnp.maximum(jnp.sum(mask), 1.0)
      case 'sequence_length':
        num_tokens = mask.size
      case _:
        raise ValueError(f'Unknown loss_normalization: {loss_normalization!r}')
    loss = jnp.sum(pt_losses * mask) / num_tokens
    return loss, {'loss': loss}
