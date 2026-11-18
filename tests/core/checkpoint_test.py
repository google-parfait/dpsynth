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

"""Unit tests for dpsynth.checkpoint."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import dpsynth
from dpsynth.core import _checkpoint
from dpsynth.statistical import initialization
from etils import epath
import mbi
import numpy as np


class _PreemptedError(Exception):
  pass


class CheckpointTest(parameterized.TestCase):

  def test_context_manager_scoping(self):
    self.assertIs(dpsynth.checkpoint, _checkpoint.checkpoint)
    self.assertIsNone(_checkpoint.get_active_checkpoint_dir())

    temp_dir = self.create_tempdir().full_path
    with _checkpoint.checkpoint(temp_dir, "run_1") as active_dir:
      self.assertEqual(active_dir, epath.Path(temp_dir) / "run_1")
      self.assertEqual(_checkpoint.get_active_checkpoint_dir(), active_dir)
    self.assertFalse((epath.Path(temp_dir) / "run_1").exists())
    self.assertTrue(epath.Path(temp_dir).exists())

    self.assertIsNone(_checkpoint.get_active_checkpoint_dir())
    with _checkpoint.checkpoint(os.path.relpath(temp_dir), "run_2") as rel_dir:
      self.assertEqual(rel_dir, epath.Path(temp_dir) / "run_2")

  @parameterized.parameters(None, "")
  def test_missing_run_id_raises(self, bad_run_id):
    temp_dir = self.create_tempdir().full_path
    with self.assertRaises(ValueError):
      with _checkpoint.checkpoint(temp_dir, bad_run_id):
        pass

  def test_run_id_isolation_and_cleanup(self):
    temp_dir = self.create_tempdir().full_path
    run1_dir = epath.Path(temp_dir) / "run_1"
    run2_dir = epath.Path(temp_dir) / "run_2"

    def fail():
      raise _PreemptedError()

    with self.assertRaises(_PreemptedError):
      with _checkpoint.checkpoint(temp_dir, "run_1"):
        _checkpoint.get_or_compute("s1", lambda: np.array([1, 2]), rng=None)
        _checkpoint.get_or_compute("s2", fail, rng=None)
    self.assertTrue((run1_dir / "s1.npz").exists())

    with _checkpoint.checkpoint(temp_dir, "run_2", cleanup=False):
      val2 = _checkpoint.get_or_compute("s1", lambda: np.array([3]), rng=None)
      np.testing.assert_array_equal(val2, [3])
    self.assertTrue(run2_dir.exists())

    with _checkpoint.checkpoint(temp_dir, "run_1"):
      val1 = _checkpoint.get_or_compute("s1", lambda: np.array([9]), rng=None)
      val2 = _checkpoint.get_or_compute("s2", lambda: np.array([5]), rng=None)
      np.testing.assert_array_equal(val1, [1, 2])
      np.testing.assert_array_equal(val2, [5])
    self.assertFalse(run1_dir.exists())
    self.assertTrue(run2_dir.exists())

  def test_context_manager_none_disabled(self):
    with _checkpoint.checkpoint(None, None) as active_dir:
      self.assertIsNone(active_dir)
      self.assertIsNone(_checkpoint.get_active_checkpoint_dir())
    with _checkpoint.checkpoint(None, "run_1") as active_dir:
      self.assertIsNone(active_dir)

  def test_get_or_compute_disabled_runs_function(self):
    call_count = 0

    def compute():
      nonlocal call_count
      call_count += 1
      return 42

    result1 = _checkpoint.get_or_compute("test_stage", compute, rng=None)
    result2 = _checkpoint.get_or_compute("test_stage", compute, rng=None)

    self.assertEqual(result1, 42)
    self.assertEqual(result2, 42)
    self.assertEqual(call_count, 2)

  def test_get_or_compute_saves_and_loads(self):
    temp_dir = self.create_tempdir().full_path
    with _checkpoint.checkpoint(temp_dir, "run_1"):
      call_count = 0

      def compute():
        nonlocal call_count
        call_count += 1
        return np.array([1.0, 2.0, 3.0])

      # First call: computes and saves.
      result1 = _checkpoint.get_or_compute("array_stage", compute, rng=None)
      self.assertEqual(call_count, 1)
      np.testing.assert_array_equal(result1, [1.0, 2.0, 3.0])

      # Second call: loads from checkpoint, skips compute.
      result2 = _checkpoint.get_or_compute("array_stage", compute, rng=None)
      self.assertEqual(call_count, 1)
      np.testing.assert_array_equal(result2, [1.0, 2.0, 3.0])

  def test_get_or_compute_spawns_rng(self):
    temp_dir = self.create_tempdir().full_path
    with _checkpoint.checkpoint(temp_dir, "run_1"):
      rng1 = np.random.default_rng(42)
      _checkpoint.get_or_compute(
          "rng_stage", lambda r: r.normal(size=5), rng=rng1
      )
      draw_after_compute = rng1.normal()

      rng2 = np.random.default_rng(42)
      _checkpoint.get_or_compute(
          "rng_stage", lambda r: r.normal(size=5), rng=rng2
      )
      draw_after_load = rng2.normal()

      self.assertEqual(draw_after_compute, draw_after_load)

  def test_column_measurements_serialization(self):
    temp_dir = self.create_tempdir().full_path
    cat_attr = dpsynth.CategoricalAttribute(possible_values=("a", "b", "c"))

    num_meas = initialization.NumericalMeasurement(
        categorical_attribute=cat_attr,
        bin_edges=np.array([0.0, 1.0, 2.0]),
        noisy_counts=np.array([10.0, 20.0]),
        stddev=1.5,
    )
    cat_meas = initialization.CategoricalMeasurement(
        categorical_attribute=cat_attr,
        noisy_counts=np.array([5.0, 15.0, 25.0]),
        stddev=2.0,
    )
    open_meas = initialization.OpenSetMeasurement(
        categorical_attribute=cat_attr,
        noisy_counts=np.array([1.0, 2.0]),
        stddev=0.5,
    )
    total_meas = mbi.LinearMeasurement(
        noisy_measurement=np.array([100.0]),
        clique=(),
        stddev=1.0,
    )

    bundle = (
        total_meas,
        {"num": num_meas, "cat": cat_meas, "open": open_meas},
    )

    path = epath.Path(temp_dir) / "measurements.npz"
    mbi.save(bundle, path)
    self.assertTrue(path.exists())

    loaded_total, loaded_dict = mbi.load(path)

    self.assertEqual(loaded_total.clique, ())
    np.testing.assert_allclose(loaded_total.noisy_measurement, [100.0])
    self.assertEqual(loaded_total.stddev, 1.0)

    # Numerical
    self.assertEqual(
        loaded_dict["num"].categorical_attribute,
        cat_attr,
    )
    np.testing.assert_array_equal(
        loaded_dict["num"].bin_edges,
        [0.0, 1.0, 2.0],
    )
    np.testing.assert_array_equal(
        loaded_dict["num"].noisy_counts,
        [10.0, 20.0],
    )
    self.assertEqual(loaded_dict["num"].stddev, 1.5)

    # Categorical
    self.assertEqual(
        loaded_dict["cat"].categorical_attribute,
        cat_attr,
    )
    np.testing.assert_array_equal(
        loaded_dict["cat"].noisy_counts,
        [5.0, 15.0, 25.0],
    )
    self.assertEqual(loaded_dict["cat"].stddev, 2.0)

    # OpenSet
    self.assertEqual(
        loaded_dict["open"].categorical_attribute,
        cat_attr,
    )
    np.testing.assert_array_equal(
        loaded_dict["open"].noisy_counts,
        [1.0, 2.0],
    )
    self.assertEqual(loaded_dict["open"].stddev, 0.5)

  def test_orbax_session_initializes_state(self):
    init_fn = lambda p: {"step": np.int32(0), "params": p}
    with _checkpoint.orbax_session(init_fn, {"w": np.ones(2)}) as (state, cb):
      self.assertIsNone(cb)
      self.assertEqual(state["step"], 0)
      np.testing.assert_array_equal(state["params"]["w"], [1.0, 1.0])

    temp_dir = self.create_tempdir().full_path
    with _checkpoint.checkpoint(temp_dir, "run_1"):
      with _checkpoint.orbax_session(init_fn, {"w": np.ones(2)}) as (state, cb):
        self.assertIsNotNone(cb)
        self.assertEqual(state["step"], 0)
        np.testing.assert_array_equal(state["params"]["w"], [1.0, 1.0])


if __name__ == "__main__":
  absltest.main()
