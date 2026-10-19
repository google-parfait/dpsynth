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

"""Gemma model loading and SFT loss function."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
import itertools
import time
from typing import Any, Literal

from absl import logging
from etils import epath
from flax import nnx
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint as ocp
import qwix
import tunix
from tunix.models.gemma4 import model as gemma4_model
from tunix.models.gemma4 import params as gemma4_params

# Supported model names for GemmaModel.default().
ModelName = Literal[
    'gemma4_e2b_it',
    'gemma4_e4b_it',
    'gemma4_31b_it',
]

_DEFAULT_LORA_TARGETS = (
    r'.*q_einsum|.*k_einsum|.*kv_einsum|.*qkv_einsum|.*attn_vec_einsum'
    r'|.*gate_proj|.*up_proj|.*down_proj'
)

# gemma4_params only defines checkpoint constants for E2B and E4B variants.
_GEMMA4_31B_IT = 'gs://gemma-data/checkpoints/gemma4-31b-it'

_DEFAULTS: dict[str, tuple[Callable[..., Any], str]] = {
    'gemma4_e2b_it': (
        gemma4_model.ModelConfig.gemma4_e2b,
        gemma4_params.GEMMA4_E2B_IT,
    ),
    'gemma4_e4b_it': (
        gemma4_model.ModelConfig.gemma4_e4b,
        gemma4_params.GEMMA4_E4B_IT,
    ),
    'gemma4_31b_it': (
        gemma4_model.ModelConfig.gemma4_31b,
        _GEMMA4_31B_IT,
    ),
}


def make_mesh(tp: int = 1) -> jax.sharding.Mesh:
  """Builds a 3D ('replica', 'fsdp'=1, 'tp'=tp) device mesh for Gemma 4."""
  n = jax.device_count()
  if n % tp != 0:
    raise ValueError(f'device_count {n} must be divisible by tp={tp}.')
  shape = (n // tp, 1, tp)
  try:
    devices = mesh_utils.create_device_mesh(
        shape, allow_split_physical_axes=True
    )
  except (ValueError, NotImplementedError, RuntimeError, AssertionError):
    devices = np.asarray(jax.devices()).reshape(shape)
  return jax.sharding.Mesh(devices, axis_names=('replica', 'fsdp', 'tp'))


@dataclasses.dataclass(frozen=True)
class GemmaModel:
  """Specification for a Gemma model variant."""

  name: ModelName
  checkpoint_path: str

  @classmethod
  def default(cls, name: ModelName) -> GemmaModel:
    """Constructs a GemmaModel from a preset name."""
    if name not in _DEFAULTS:
      raise ValueError(f'Unknown model {name!r}. Options: {list(_DEFAULTS)}')
    _, checkpoint_path = _DEFAULTS[name]
    return cls(name=name, checkpoint_path=checkpoint_path)


@dataclasses.dataclass(frozen=True)
class LoraConfig:
  """Configuration for LoRA adaptation."""

  rank: int = 16
  alpha: float | None = None
  targets: str = _DEFAULT_LORA_TARGETS
  weight_qtype: str | None = None


def load_gemma(
    model_variant: GemmaModel,
    lora_config: LoraConfig,
    *,
    checkpoint_path: epath.PathLike | None = None,
    mesh: jax.sharding.Mesh | None = None,
) -> nnx.Module:
  """Loads a Gemma model with LoRA adapters.

  Args:
    model_variant: Which Gemma variant to load.
    lora_config: LoRA adapter configuration.
    checkpoint_path: Checkpoint to restore from. If None, loads pretrained base
      weights from ``model_variant.checkpoint_path`` and initializes fresh LoRA
      adapters. If a checkpoint is provided, restores LoRA params.
    mesh: Optional sharding mesh to constrain parameters across devices.

  Returns:
    The LoRA-wrapped ``nnx.Module``.
  """
  if mesh is None and not jax.sharding.get_mesh().empty:
    mesh = jax.sharding.get_mesh()
  config_fn, _ = _DEFAULTS[model_variant.name]
  # Work around a Tunix/Qwix limitation: RematConfig.BLOCK wraps blocks in
  # nnx.remat(graph_updates=False) and jax.checkpoint, which masks qwix_rngs
  # and discards LoRAParam attribute mutations during the init forward pass
  # (see tunix/cli/utils/model.py). Enable RematConfig.BLOCK after _init_lora.
  cfg = dataclasses.replace(
      config_fn(),
      dtype=jnp.bfloat16,
      remat_config=gemma4_model.RematConfig.NONE,
  )
  loaded = gemma4_params.create_model_from_checkpoint(
      model_variant.checkpoint_path,
      cfg,
      mesh=mesh,
  )
  alpha = (
      lora_config.alpha
      if lora_config.alpha is not None
      else float(lora_config.rank)
  )
  lora_provider = qwix.LoraProvider(
      module_path=lora_config.targets,
      rank=lora_config.rank,
      alpha=alpha,
      weight_qtype=lora_config.weight_qtype,
  )
  warm_tokens = jnp.ones((1, 1), dtype=jnp.int32)
  warm_mask = jnp.ones((1, 1, 1), dtype=jnp.bool_)
  # Skip eager init so multi-host shards are traced abstractly under nnx.jit.
  lora_model = qwix.apply_lora_to_model(
      loaded,
      lora_provider,
      warm_tokens,
      attention_mask=warm_mask,
      rngs=nnx.Rngs(0),
      skip_nnx_init=True,
  )
  assert isinstance(lora_model, nnx.Module)
  lora_model.set_attributes(qwix_rngs=nnx.Rngs(0))

  @nnx.jit
  def _init_lora(m: nnx.Module, tok: jax.Array, msk: jax.Array) -> nnx.Module:
    logits, _ = m(tok, attention_mask=msk)
    _ = jnp.sum(logits)
    if mesh is not None:
      state = nnx.state(m, nnx.LoRAParam)
      pspecs = nnx.get_partition_spec(state)
      nnx.update(m, jax.lax.with_sharding_constraint(state, pspecs))
    return m

  lora_model = _init_lora(lora_model, warm_tokens, warm_mask)
  lora_model.set_attributes(qwix_rngs=None)
  lora_model.config.remat_config = gemma4_model.RematConfig.BLOCK
  if checkpoint_path is not None:
    load_lora_params(lora_model, checkpoint_path)

  lora_state = nnx.state(lora_model, nnx.LoRAParam)
  num_lora = sum(x.size for x in jax.tree.leaves(lora_state))
  total = sum(x.size for x in jax.tree.leaves(nnx.state(lora_model)))
  num_base = total - num_lora
  logging.info(
      'Loaded Gemma model w/ LoRA (rank=%d): %d lora (%.4f%%), %d base',
      lora_config.rank,
      num_lora,
      100.0 * num_lora / (num_lora + num_base),
      num_base,
  )
  return lora_model


def save_lora_params(lora_model: nnx.Module, path: epath.PathLike) -> None:
  """Saves trainable LoRA parameters from ``lora_model`` to ``path``."""
  lora_state = nnx.state(lora_model, nnx.LoRAParam)
  checkpointer = ocp.StandardCheckpointer()
  checkpointer.save(epath.Path(path), lora_state, force=True)
  checkpointer.wait_until_finished()


def load_lora_params(lora_model: nnx.Module, path: epath.PathLike) -> None:
  """Restores trainable LoRA parameters into ``lora_model`` in-place."""
  abstract_lora = nnx.state(lora_model, nnx.LoRAParam)
  checkpointer = ocp.StandardCheckpointer()
  restored = checkpointer.restore(epath.Path(path), abstract_lora)
  nnx.update(lora_model, restored)


def sft_loss_fn(
    module: nnx.Module,
    data: dict[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Cross-entropy next-token-prediction loss for supervised fine-tuning.

  Args:
    module: LoRA-wrapped Gemma model.
    data: Dict with ``'input_tokens'`` and ``'loss_mask'`` (int32 ``[B, L]``).

  Returns:
    ``(loss, aux)`` where ``aux`` contains ``'loss'``.
  """
  input_tokens = data['input_tokens']
  loss_mask = data['loss_mask']

  pad_mask = input_tokens != 0
  positions = jnp.cumsum(pad_mask, axis=-1) - (
      jnp.cumsum(pad_mask, axis=-1) >= 1
  )
  seq_len = input_tokens.shape[-1]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  attention_mask = pad_mask[..., None, :] * causal_mask[None, :]

  logits, _ = module(input_tokens, positions, None, attention_mask)
  logits = logits[:, :-1, :]
  targets = input_tokens[:, 1:]
  mask = loss_mask[:, 1:]

  pt_losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
  loss = jnp.sum(pt_losses * mask) / jnp.maximum(jnp.sum(mask), 1.0)
  return loss, {'loss': loss}


def format_prompt(prompt: str) -> str:
  """Formats a prompt string with the Gemma user and model turn tags."""
  # Universal chat prompt format expected by Gemma instruction-tuned models.
  return f'<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n'


def format_response(response: str) -> str:
  """Formats a model response string with the end-of-turn tag."""
  return f'{response}<end_of_turn>'


@functools.cache
def _get_tokenizer() -> Any:
  return gemma4_params.create_tokenizer()


def tokenize_example(
    example: tuple[str, str],
    max_seq_length: int,
) -> dict[str, np.ndarray]:
  """Tokenizes a single (prompt, response) pair for supervised fine-tuning.

  Prompt tokens are masked out (``loss_mask=0``) so only the response
  contributes to the training loss. Turn formatting follows the Gemma
  dialog template (forked from ``gemma/gm/data/_tasks.py``).

  Args:
    example: A ``(prompt, response)`` string pair.
    max_seq_length: Maximum sequence length (including special tokens).

  Returns:
    Dict with ``'input_tokens'`` and ``'loss_mask'`` (int32 ``[L]``).
  """
  tokenizer = _get_tokenizer()
  prompt, response = example
  tokens = np.zeros((max_seq_length,), dtype=np.int32)
  mask = np.zeros((max_seq_length,), dtype=np.int32)

  # Embed turn tags as strings so SentencePiece handles tokenization
  # boundaries correctly (encoding pieces separately can shift BPE merges).
  prompt_str = format_prompt(prompt)
  response_str = format_response(response)
  prompt_ids = [tokenizer.bos_id()] + tokenizer.EncodeAsIds(prompt_str)
  response_ids = tokenizer.EncodeAsIds(response_str) + [tokenizer.eos_id()]

  ids = prompt_ids + response_ids
  length = min(len(ids), max_seq_length)
  tokens[:length] = ids[:length]
  # Mask: 0 for prompt, 1 for response.
  resp_start = min(len(prompt_ids), length)
  mask[resp_start:length] = 1

  return {'input_tokens': tokens, 'loss_mask': mask}


class GemmaSampler:
  """Batched inference sampler for generating synthetic text."""

  def __init__(
      self,
      *,
      model: nnx.Module,
      model_variant: GemmaModel,
      max_seq_length: int,
      temperature: float,
  ):
    config_fn, _ = _DEFAULTS[model_variant.name]
    cfg = config_fn()
    cache_config = tunix.CacheConfig(
        cache_size=max_seq_length,
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads or cfg.num_heads,
        head_dim=cfg.head_dim,
    )
    self._max_seq_length = max_seq_length
    self._temperature = temperature
    self._sampler = tunix.Sampler(
        transformer=model,
        tokenizer=gemma4_params.create_tokenizer(),
        cache_config=cache_config,
    )
    self._eos_tokens = [
        self._sampler.tokenizer.eos_id(),
        *self._sampler.tokenizer.encode('<end_of_turn>'),
    ]

  def __call__(
      self,
      prompts: Sequence[str],
      *,
      rng: int = 0,
      batch_size: int = 32,
  ) -> list[str]:
    """Formats prompts, batches them across devices, and samples responses.

    Args:
      prompts: Sequence of prompt instruction strings.
      rng: Base random seed for sampling (default 0). The random seed is set per
        batch (``rng + batch_idx``), so fixing the seed and changing the
        ``batch_size`` will change the sampled outputs.
      batch_size: Inference batch size (default 32).

    Returns:
      List of generated response strings corresponding to each prompt.
    """
    formatted = [format_prompt(p) for p in prompts]

    results: list[str] = []
    for i, batch_items in enumerate(itertools.batched(formatted, batch_size)):
      cur_size = len(batch_items)
      batch = list(batch_items) + [batch_items[-1]] * (batch_size - cur_size)
      t0 = time.perf_counter()
      out = self._sampler(
          input_strings=batch,
          max_generation_steps=self._max_seq_length // 2,
          temperature=self._temperature,
          top_p=1.0 if self._temperature > 0 else None,
          eos_tokens=self._eos_tokens,
          seed=rng + i,
      )
      elapsed = time.perf_counter() - t0
      logging.info(
          'Batch %d: %d samples in %.2fs (%.2f samples/s)',
          i + 1,
          cur_size,
          elapsed,
          cur_size / elapsed if elapsed > 0 else 0.0,
      )
      results.extend([str(r) for r in out.text[:cur_size]])
    return results
