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
from typing import Any, TypeAlias, TypeVar

from absl import logging
from dpsynth.statistical import initialization
from etils import epath
import jax
from jax.experimental import multihost_utils
import mbi
import numpy as np

T = TypeVar("T")
PathType = epath.PathLike
CallbackFn: TypeAlias = Callable[[int, Any, Any], None]

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
    run_id: str | None,
    *,
    cleanup: bool = True,
) -> Iterator[epath.Path | None]:
  """Context manager enabling intermediate checkpointing for dpsynth mechanisms.

  Persists intermediate mechanism state under ``checkpoint_dir / run_id`` so
  preempted runs can resume, and deletes ``checkpoint_dir / run_id`` upon normal
  exit when ``cleanup=True``.

  Note:
    Treat ``checkpoint_dir`` as private working storage, as intermediate
    checkpoints are not differentially private. Checkpoints are not validated
    against the mechanism configuration, privacy budget, or input dataset:
    reusing checkpoints across different configurations, budgets, or datasets,
    or invoking a mechanism multiple times in one ``checkpoint`` block, breaks
    differential privacy guarantees. Pass a ``run_id`` that stays constant
    across preemptions of a single run and differs across distinct runs.

  Args:
    checkpoint_dir: Base directory for checkpoints, or None to disable.
    run_id: Run identifier appended as ``checkpoint_dir / run_id`` to isolate
      runs. Required whenever ``checkpoint_dir`` is not None.
    cleanup: Whether to delete ``checkpoint_dir / run_id`` upon normal exit.

  Yields:
    The resolved path of ``checkpoint_dir / run_id``, or None if disabled.
  """
  path = None
  if checkpoint_dir is not None:
    if not run_id:
      raise ValueError("run_id is required when checkpoint_dir is set.")
    path = epath.Path(checkpoint_dir) / run_id
    if not path.is_absolute():
      path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
  token = _ACTIVE_CHECKPOINT_DIR.set(path)
  try:
    yield path
    if path is not None and cleanup:
      if jax.process_count() > 1:
        multihost_utils.sync_global_devices("dpsynth_checkpoint_cleanup")
      if jax.process_index() == 0:
        path.rmtree(missing_ok=True)
  finally:
    _ACTIVE_CHECKPOINT_DIR.reset(token)


def get_active_checkpoint_dir() -> epath.Path | None:
  """Returns the active checkpoint directory, or None if not set."""
  return _ACTIVE_CHECKPOINT_DIR.get()


def get_or_compute(
    name: str,
    compute_fn: Callable[..., T],
    *args: Any,
    rng: np.random.Generator | None,
    **kwargs: Any,
) -> T:
  """Returns the checkpointed value if available, or computes and saves it.

  Args:
    name: The stage name (e.g. 'column_measurements', 'discrete_data').
    compute_fn: Callable that produces the intermediate value.
    *args: Positional arguments for ``compute_fn``.
    rng: Random number generator for randomized stages, or None for
      deterministic stages.
    **kwargs: Keyword arguments for ``compute_fn``.

  Returns:
    The loaded or computed value.
  """
  if rng is not None:
    # Generator.spawn derives a statistically independent child BitGenerator
    # stream via SeedSequence hashing while advancing the parent's spawn counter
    # by 1, isolating the caller's RNG from how many draws the stage consumes
    # and ensuring identical RNG progression on cache hits and misses.
    args = (rng.spawn(1)[0], *args)
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


# jax_privacy's BandMF allocates (bands - 1, *param.shape) state buffers, which
# produces 0-size arrays when bands=1 that Orbax StandardSave rejects.
def _pack_zero_size(tree: Any) -> Any:
  return jax.tree.map(lambda x: 0 if getattr(x, "size", 1) == 0 else x, tree)


def _unpack_zero_size(target: Any, restored: Any) -> Any:
  return jax.tree.map(
      lambda t, r: t if getattr(t, "size", 1) == 0 else r, target, restored
  )


@contextlib.contextmanager
def orbax_session(
    init_fn: Callable[[Any], Any],
    init_params: Any,
    *,
    checkpoint_every: int = 100,
    callback: CallbackFn | None = None,
) -> Iterator[tuple[Any, CallbackFn | None]]:
  """Manages periodic Orbax step-level checkpointing and restoration.

  When no ``dpsynth.checkpoint(...)`` context is active, this is a no-op that
  yields ``(init_fn(init_params), callback)`` without touching disk.

  When a checkpoint directory is active, this context manager opens an
  ``orbax.checkpoint.CheckpointManager`` under ``<checkpoint_dir>/train_state``
  and manages three phases around the training loop:

  1. **On entry (initialize or restore)**: Evaluates ``init_fn(init_params)`` to
     construct an initial ``TrainingState``, and if a checkpoint already exists,
     restores the latest checkpoint into it.
  2. **During training (periodic save)**: Wraps ``callback`` in a step hook that
     saves the current ``TrainingState`` every ``checkpoint_every`` steps
     (retaining only the most recent checkpoint) before forwarding
     ``(step, state, aux)`` to the user's ``callback``.
  3. **On exit (final save & flush)**: If training executed at least one step
     and the final step was not already saved at a ``checkpoint_every``
     boundary, force-saves the final ``TrainingState``, then waits for all
     background Orbax writes to complete before closing the manager.

  Args:
    init_fn: Callable (typically ``trainer.init``) that maps ``init_params`` to
      a freshly initialized ``TrainingState`` PyTree.
    init_params: Initial parameter PyTree passed to ``init_fn``.
    checkpoint_every: Step interval between periodic checkpoints.
    callback: Optional user step callback ``(step, state, aux) -> None`` to
      invoke after each step's checkpoint hook.

  Yields:
    A ``(initial_state, step_callback)`` pair to pass directly to
    ``trainer.fit(data, initial_state, callback=step_callback, ...)``, where
    ``initial_state`` is an initialized or restored ``TrainingState``.
  """
  ckpt_dir = get_active_checkpoint_dir()
  if ckpt_dir is None:
    yield init_fn(init_params), callback
    return

  from orbax import checkpoint as ocp  # pylint: disable=g-import-not-at-top

  options = ocp.CheckpointManagerOptions(
      should_save_fn=lambda step, _: step > 0 and step % checkpoint_every == 0,
      max_to_keep=1,
      cleanup_tmp_directories=True,
  )
  with ocp.CheckpointManager(ckpt_dir / "train_state", options=options) as mgr:
    initial_state = init_fn(init_params)
    latest_step = mgr.latest_step()
    if latest_step is not None:
      logging.info("[DPSynth Checkpoint]: Restoring step %d", latest_step)
      restored = mgr.restore(
          latest_step,
          args=ocp.args.StandardRestore(_pack_zero_size(initial_state)),
      )
      initial_state = _unpack_zero_size(initial_state, restored)

    last_state = None

    def _step_callback(step: int, state: Any, aux: Any):
      nonlocal last_state
      last_state = state
      if mgr.should_save(step):
        mgr.save(step, args=ocp.args.StandardSave(_pack_zero_size(state)))
      if callback is not None:
        callback(step, state, aux)

    yield initial_state, _step_callback
    if last_state is not None:
      mgr.wait_until_finished()
      if mgr.latest_step() != int(last_state.step):
        save_args = ocp.args.StandardSave(_pack_zero_size(last_state))
        mgr.save(int(last_state.step), args=save_args, force=True)
