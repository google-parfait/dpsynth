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

"""Checkpointing support for dpsynth intermediate computations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
import contextvars
from typing import Any, TypeVar

from absl import logging
from dpsynth.local_mode import initialization
from etils import epath
import jax
import mbi

T = TypeVar("T")
PathType = epath.PathLike

_ACTIVE_CHECKPOINT_DIR: contextvars.ContextVar[epath.Path | None] = (
    contextvars.ContextVar("_ACTIVE_CHECKPOINT_DIR", default=None)
)

# Register here to avoid taking a JAX dependency in initialization.py.
jax.tree_util.register_dataclass(
    initialization.CategoricalMeasurement,
    data_fields=["noisy_counts"],
    meta_fields=["categorical_attribute", "stddev"],
)
jax.tree_util.register_dataclass(
    initialization.OpenSetMeasurement,
    data_fields=["noisy_counts"],
    meta_fields=["categorical_attribute", "stddev"],
)
jax.tree_util.register_dataclass(
    initialization.NumericalMeasurement,
    data_fields=["bin_edges", "noisy_counts"],
    meta_fields=["categorical_attribute", "stddev"],
)


@contextlib.contextmanager
def checkpoint(
    checkpoint_dir: str | PathType | None,
) -> Iterator[epath.Path | None]:
  """Context manager enabling intermediate checkpointing for dpsynth mechanisms.

  When active, mechanisms that support checkpointing (such as TabularMechanism
  and SWIFT) check whether stage outputs already exist on disk under
  ``checkpoint_dir``. If so, the computation is skipped and the checkpoint is
  loaded. If not, the computation is run and saved to ``checkpoint_dir`` as an
  ``.npz`` archive via ``mbi.save``, which supports PyTrees of numpy arrays
  with static metadata.

  Args:
    checkpoint_dir: Directory path where checkpoints should be stored/loaded. If
      None, checkpointing is disabled and computations run in-memory.

  Yields:
    The resolved path of the checkpoint directory, or None if it is disabled.
  """
  path = epath.Path(checkpoint_dir) if checkpoint_dir is not None else None
  if path is not None:
    path.mkdir(parents=True, exist_ok=True)
  token = _ACTIVE_CHECKPOINT_DIR.set(path)
  try:
    yield path
  finally:
    _ACTIVE_CHECKPOINT_DIR.reset(token)


def get_active_checkpoint_dir() -> epath.Path | None:
  """Returns the active checkpoint directory, or None if not set."""
  return _ACTIVE_CHECKPOINT_DIR.get()


def get_or_compute(
    name: str,
    compute_fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
  """Returns the checkpointed value if available, or computes and saves it.

  Args:
    name: The stage name (e.g. 'column_measurements', 'discrete_data').
    compute_fn: Callable that produces the intermediate value.
    *args: Positional arguments for ``compute_fn``.
    **kwargs: Keyword arguments for ``compute_fn``.

  Returns:
    The loaded or computed value.
  """
  dir_path = get_active_checkpoint_dir()
  if dir_path is None:
    return compute_fn(*args, **kwargs)

  path = dir_path / f"{name}.npz"
  if path.exists():
    logging.info("[DPSynth Checkpoint]: Loading %s from %s", name, path)
    return mbi.load(path)

  result = compute_fn(*args, **kwargs)
  mbi.save(result, path)
  logging.info("[DPSynth Checkpoint]: Saved %s to %s", name, path)
  return result
