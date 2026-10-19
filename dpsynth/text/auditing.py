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

"""Canary insertion and empirical privacy auditing for LoRA fine-tuned models.

Provides dataset mixing before training (``mix_canaries``), in-JAX
log-likelihood
ratio canary scoring against the zeroed-LoRA base model (``score_canaries``),
and
empirical privacy auditing via ``jax_privacy.auditing.CanaryScoreAuditor``
(``audit_scores`` and ``audit_canaries``).
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools

from dpsynth.text import model
from flax import nnx
import grain.python as grain
import jax
import jax.numpy as jnp
import jax_privacy
import numpy as np
import optax


@dataclasses.dataclass(frozen=True)
class EmpiricalAuditResult:
  """Empirical privacy audit metrics from held-in vs. held-out canary scores.

  Attributes:
    epsilon_one_run_fdp: Lower bound on (epsilon, delta)-DP under f-DP.
    epsilon_one_run: Lower bound on (epsilon, delta)-DP in one training run.
    epsilon_clopper_pearson: Clopper-Pearson lower bound on (epsilon, delta)-DP.
    epsilon_raw_counts: Uncalibrated empirical epsilon estimate ln(TPR / FPR).
    auroc: Area under the membership-inference ROC curve.
    max_accuracy: Maximum classification accuracy across score thresholds.
    tpr_at_fpr_01_percent: True positive rate at 0.1% false positive rate.
    delta: Target delta used for empirical epsilon bounds.
    significance: Significance level alpha (1 - confidence) for lower bounds.
    num_held_in: Number of held-in canaries evaluated.
    num_held_out: Number of held-out canaries evaluated.
  """

  epsilon_one_run_fdp: float
  epsilon_one_run: float
  epsilon_clopper_pearson: float
  epsilon_raw_counts: float
  auroc: float
  max_accuracy: float
  tpr_at_fpr_01_percent: float
  delta: float
  significance: float = 0.05
  num_held_in: int = 0
  num_held_out: int = 0


def mix_canaries(
    train_data: Sequence[tuple[str, str]] | grain.MapDataset,
    held_in_canaries: Sequence[tuple[str, str]] | grain.MapDataset,
    *,
    repetitions: int = 1,
) -> list[tuple[str, str]] | grain.MapDataset:
  """Mixes ``repetitions`` copies of ``held_in_canaries`` into ``train_data``."""
  if repetitions < 1:
    raise ValueError(f'repetitions must be >= 1, got {repetitions}.')
  is_map = isinstance(train_data, grain.MapDataset) or isinstance(
      held_in_canaries, grain.MapDataset
  )
  if is_map:
    train_ds = (
        train_data
        if isinstance(train_data, grain.MapDataset)
        else grain.MapDataset.source(list(train_data))
    )
    canary_ds = (
        held_in_canaries
        if isinstance(held_in_canaries, grain.MapDataset)
        else grain.MapDataset.source(list(held_in_canaries))
    )
    return grain.MapDataset.concatenate(
        [train_ds, canary_ds.repeat(repetitions)]
    )
  return list(train_data) + list(held_in_canaries) * repetitions


def _example_nll(module: nnx.Module, data: dict[str, jax.Array]) -> jax.Array:
  """Computes per-example response negative log-likelihoods of shape [B]."""
  input_tokens = data['input_tokens']
  loss_mask = data['loss_mask']
  pad_mask = input_tokens != 0
  csum = jnp.cumsum(pad_mask, axis=-1)
  positions = csum - (csum >= 1)
  seq_len = input_tokens.shape[-1]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  attention_mask = pad_mask[..., None, :] * causal_mask[None, :]
  logits, _ = module(input_tokens, positions, None, attention_mask)
  pt_losses = optax.softmax_cross_entropy_with_integer_labels(
      logits[:, :-1, :], input_tokens[:, 1:]
  )
  return jnp.sum(pt_losses * loss_mask[:, 1:], axis=-1)


def score_canaries(
    trained_model: nnx.Module,
    canaries: Sequence[tuple[str, str]] | grain.MapDataset,
    *,
    max_seq_length: int = 512,
    batch_size: int = 32,
) -> np.ndarray:
  """Scores canaries by log-likelihood ratio against the zero-LoRA base model.

  Computes ``log p_trained(y | x) - log p_base(y | x)`` for each ``(prompt,
  response)`` pair without loading a second checkpoint by zeroing out the
  ``nnx.LoRAParam`` adapters for the reference pass.

  Args:
    trained_model: Fine-tuned LoRA model.
    canaries: Sequence or ``grain.MapDataset`` of ``(prompt, response)`` pairs.
    max_seq_length: Maximum token sequence length.
    batch_size: Number of examples per forward pass batch.

  Returns:
    1D float32 NumPy array of log-likelihood ratio scores (higher = more likely
    held-in).
  """
  if batch_size < 1:
    raise ValueError(f'batch_size must be >= 1, got {batch_size}.')
  tok_fn = functools.partial(
      model.tokenize_example, max_seq_length=max_seq_length
  )
  records = (
      list(canaries.map(tok_fn))
      if isinstance(canaries, grain.MapDataset)
      else [tok_fn(ex) for ex in canaries]
  )
  n = len(records)
  if n == 0:
    return np.empty(0, dtype=np.float32)

  tokens = np.stack([r['input_tokens'] for r in records])
  masks = np.stack([r['loss_mask'] for r in records])
  pad = (-n) % batch_size
  if pad:
    tokens = np.pad(tokens, ((0, pad), (0, 0)))
    masks = np.pad(masks, ((0, pad), (0, 0)))

  graphdef, lora_params, base_params = nnx.split(
      trained_model, nnx.LoRAParam, ...
  )
  zero_lora = jax.tree.map(jnp.zeros_like, lora_params)

  @jax.jit
  def _score_batch(lora_p, zero_p, base_p, batch):
    trained_nll = _example_nll(nnx.merge(graphdef, lora_p, base_p), batch)
    base_nll = _example_nll(nnx.merge(graphdef, zero_p, base_p), batch)
    return base_nll - trained_nll

  chunks = []
  for start in range(0, len(tokens), batch_size):
    batch = {
        'input_tokens': jnp.asarray(tokens[start : start + batch_size]),
        'loss_mask': jnp.asarray(masks[start : start + batch_size]),
    }
    chunks.append(
        np.asarray(
            _score_batch(lora_params, zero_lora, base_params, batch),
            dtype=np.float32,
        )
    )
  return np.concatenate(chunks)[:n]


def audit_scores(
    held_in_scores: np.typing.ArrayLike,
    held_out_scores: np.typing.ArrayLike,
    *,
    delta: float,
    significance: float = 0.05,
    min_count: int = 50,
    threshold_strategy: jax_privacy.auditing.ThresholdStrategy | None = None,
) -> EmpiricalAuditResult:
  """Computes empirical privacy metrics from held-in and held-out canary scores."""
  in_arr = np.asarray(held_in_scores, dtype=np.float64)
  out_arr = np.asarray(held_out_scores, dtype=np.float64)
  strategy = threshold_strategy or jax_privacy.auditing.Split(seed=0)
  auditor = jax_privacy.auditing.CanaryScoreAuditor(in_arr, out_arr)
  eps_fdp = auditor.epsilon_one_run_fdp(
      significance, delta, threshold_strategy=strategy
  )
  eps_one_run = auditor.epsilon_one_run(
      significance, delta, threshold_strategy=strategy
  )
  eps_cp = auditor.epsilon_clopper_pearson(
      significance, delta, threshold_strategy=strategy
  )
  eps_raw = auditor.epsilon_raw_counts(min_count=min_count, delta=delta)
  return EmpiricalAuditResult(
      epsilon_one_run_fdp=float(eps_fdp),
      epsilon_one_run=float(eps_one_run),
      epsilon_clopper_pearson=float(eps_cp),
      epsilon_raw_counts=float(eps_raw),
      auroc=float(auditor.attack_auroc()),
      max_accuracy=float(auditor.max_accuracy()),
      tpr_at_fpr_01_percent=float(auditor.tpr_at_given_fpr(0.001)),
      delta=delta,
      significance=significance,
      num_held_in=int(in_arr.size),
      num_held_out=int(out_arr.size),
  )


def audit_canaries(
    trained_model: nnx.Module,
    held_in_canaries: Sequence[tuple[str, str]] | grain.MapDataset,
    held_out_canaries: Sequence[tuple[str, str]] | grain.MapDataset,
    *,
    delta: float,
    significance: float = 0.05,
    min_count: int = 50,
    max_seq_length: int = 512,
    batch_size: int = 32,
    threshold_strategy: jax_privacy.auditing.ThresholdStrategy | None = None,
) -> EmpiricalAuditResult:
  """Scores held-in and held-out canaries and returns an ``EmpiricalAuditResult``."""
  in_scores = score_canaries(
      trained_model,
      held_in_canaries,
      max_seq_length=max_seq_length,
      batch_size=batch_size,
  )
  out_scores = score_canaries(
      trained_model,
      held_out_canaries,
      max_seq_length=max_seq_length,
      batch_size=batch_size,
  )
  return audit_scores(
      in_scores,
      out_scores,
      delta=delta,
      significance=significance,
      min_count=min_count,
      threshold_strategy=threshold_strategy,
  )
