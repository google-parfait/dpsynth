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

"""Step-level training callbacks and metric aggregation for DPTrainer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import time
import typing
from typing import TypeAlias

from absl import logging
import jax
import jax_privacy
import numpy as np

PerExampleAux: TypeAlias = jax_privacy.training.PerExampleAux
TrainingState: TypeAlias = jax_privacy.training.TrainingState
MetricsFn: TypeAlias = Callable[[int, Mapping[str, float]], None]
ReduceFn: TypeAlias = Callable[[PerExampleAux], Mapping[str, float]]


# Replicate sharded arrays across hosts before transferring to CPU so
# jax.device_get can read the full global array on every process.
def _replicate(x: typing.Any) -> typing.Any:
  if isinstance(x, jax.Array) and not x.is_fully_replicated:
    mesh = getattr(x.sharding, 'mesh')
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return jax.device_put(x, sharding)
  return x


def summarize_aux(aux: PerExampleAux) -> dict[str, float]:
  """Computes exact (non-DP) summary metrics from per-example outputs."""
  n = len(aux.values)
  metrics = {
      'loss': np.mean(aux.values) if n else np.nan,
      'batch_size': n,
  }
  for path, val in jax.tree_util.tree_leaves_with_path(aux.aux):
    key = jax.tree_util.keystr(path, simple=True, separator='/') or 'aux'
    metrics[key] = np.mean(val) if n else np.nan
  if aux.grad_norms is not None and n:
    norms = aux.grad_norms
    metrics['grad_norm_mean'] = np.mean(norms)
    metrics['grad_norm_min'] = np.min(norms)
    metrics['grad_norm_median'] = np.median(norms)
    metrics['grad_norm_max'] = np.max(norms)
  return {k: float(v) for k, v in metrics.items()}


@dataclasses.dataclass
class DefaultCallbackFn:
  """Buffers per-step outputs and periodically logs metrics."""

  log_every: int | None = None
  metrics_fn: MetricsFn | None = None
  reduce_fn: ReduceFn = summarize_aux
  _aux_buffer: list[PerExampleAux] = dataclasses.field(default_factory=list)
  _last_time: float | None = None

  def __call__(self, step: int, state: TrainingState, aux: PerExampleAux):
    del state
    if self.log_every is None:
      return
    if self._last_time is None:
      jax.block_until_ready(aux)
      self._last_time = time.monotonic()
    self._aux_buffer.append(aux)
    if step % self.log_every != 0:
      return

    buf = jax.device_get(jax.tree.map(_replicate, self._aux_buffer))
    self._aux_buffer.clear()
    now = time.monotonic()
    num_steps = len(buf)
    combined = jax.tree.map(lambda *xs: np.concatenate(xs), *buf)
    valid = ~np.isnan(combined.values)
    combined = jax.tree.map(lambda x: x[valid], combined)

    metrics = dict(self.reduce_fn(combined))
    metrics['batch_size'] /= num_steps
    metrics['step_time_sec'] = (now - self._last_time) / num_steps

    logging.info('Step %d: %s', step, metrics)
    if self.metrics_fn is not None:
      self.metrics_fn(step, metrics)
    self._last_time = time.monotonic()
