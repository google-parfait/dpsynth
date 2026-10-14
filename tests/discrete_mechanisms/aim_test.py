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

from absl.testing import absltest
from dpsynth.discrete_mechanisms import aim
from dpsynth.discrete_mechanisms import common
import mbi
import numpy as np


def _make_correlated_dataset(rng, n=1000):
  domain = mbi.Domain(["a", "b", "c"], [3, 3, 3])
  a = rng.integers(0, 3, size=n)
  b = np.where(rng.random(n) < 0.75, a, rng.integers(0, 3, size=n))
  c = (a + b + rng.integers(0, 2, size=n)) % 3
  return mbi.Dataset({"a": a, "b": b, "c": c}, domain)


def _normalized_l1(data, model, clique):
  expected = data.project(clique).datavector()
  actual = model.project(clique).datavector()
  expected /= expected.sum()
  actual /= actual.sum()
  return np.abs(expected - actual).sum() / 2.0


def _correlated_workload_mechanism_baseline_errors(
    config, baseline_config, workload, budget=5.0
):
  rng = np.random.default_rng(0)
  data = _make_correlated_dataset(rng)

  mechanism_result = config.configure(budget=budget)(rng, data)
  baseline_result = baseline_config.configure(budget=budget)(rng, data)

  mechanism_error = np.mean([
      _normalized_l1(data, mechanism_result.model, clique)
      for clique in workload
  ])
  baseline_error = np.mean([
      _normalized_l1(data, baseline_result.model, clique) for clique in workload
  ])
  return mechanism_error, baseline_error


class AIMTest(absltest.TestCase):

  def test_fits_one_way_marginals_with_aim(self):
    data = mbi.Dataset.synthetic(mbi.Domain(["a", "b", "c"], [3, 4, 5]), N=1000)
    workload = [("a",), ("b",), ("c",)]
    config = aim.AIMConfig(max_rounds=4, pgm_iters=500)

    calibrated = config.configure(budget=10000, workload=workload)
    result = calibrated(np.random.default_rng(0), data)

    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertNotEmpty(result.measurements)
    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = result.model.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=1)

  def test_default_configuration_values(self):
    config = aim.AIMConfig()
    self.assertEqual(config.pgm_iters, 1000)


if __name__ == "__main__":
  absltest.main()
