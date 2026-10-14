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

import itertools
import pickle

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth import transformations
from dpsynth.discrete_mechanisms import common
import mbi
import numpy as np


def assert_serializable(obj):
  pickle.dumps(obj)


class CommonTest(parameterized.TestCase):

  def test_measure_marginals_with_noise(self):
    data = mbi.Dataset.synthetic(mbi.Domain(["a", "b", "c"], [3, 4, 5]), N=1000)
    marginal_queries = [("a",), ("b",), ("c",)]
    measurements = common.measure_marginals_with_noise(
        np.random.default_rng(0), data, marginal_queries, gdp_sigma=1.0
    )
    self.assertLen(measurements, 3)
    for m in measurements:
      self.assertLen(m.clique, 1)

  def test_compressed_measurement(self):
    answer = np.array([5, 10, 15, 20, 25, 30])
    measurement = mbi.LinearMeasurement(answer, ("a",), 1.0)

    size, transform = transformations.create_rare_value_merging_transformation(
        np.array([True, False, True, False, True, False])
    )
    compressed = common.compressed_measurement(measurement, size, transform)
    self.assertEqual(compressed.clique, ("a",))
    self.assertLen(compressed.noisy_measurement, 4)
    assert_serializable(compressed)

  def test_get_domain_compression_transformations(self):
    answer = np.array([5, 10, 15, 20, 25, 30])
    measurement = mbi.LinearMeasurement(answer, ("a",), stddev=4.0)
    compressed_domain = common.get_domain_compression_transformations(
        [measurement]
    )[0]
    self.assertEqual(compressed_domain, mbi.Domain.fromdict({"a": 5}))

  def test_supporting_cliques(self):
    domain = mbi.Domain(["a", "b", "c", "d"], [3, 3, 3, 100])
    cliques = common.supporting_cliques(domain, workload=None)
    self.assertCountEqual(
        cliques, list(itertools.combinations(domain.attributes, 3))
    )
    # Default workload with two columns.
    two_column_domain = mbi.Domain(["a", "b"], [3, 3])
    cliques = common.supporting_cliques(two_column_domain, workload=None)
    self.assertCountEqual(cliques, [("a", "b")])
    # List workload.
    cliques = common.supporting_cliques(domain, [("a", "b"), ("c", "d")])
    self.assertCountEqual(cliques, [("a", "b"), ("c", "d")])
    # Dict workload uses keys.
    cliques = common.supporting_cliques(domain, {("a", "b"): 1.0})
    self.assertCountEqual(cliques, [("a", "b")])
    # Filters by max_marginal_size.
    cliques = common.supporting_cliques(
        domain, [("a", "b"), ("a", "d")], max_marginal_size=50
    )
    self.assertIn(("a", "b"), cliques)
    self.assertNotIn(("a", "d"), cliques)
    # Workload with lists instead of tuples.
    cliques = common.supporting_cliques(domain, [["a", "b"], ["c", "d"]])
    self.assertCountEqual(cliques, [("a", "b"), ("c", "d")])
    for cl in cliques:
      self.assertIsInstance(cl, tuple)

  def test_compiled_workload_with_lists(self):
    domain = mbi.Domain(["a", "b"], [3, 3])
    workload = common.compiled_workload(domain, [["a", "b"]])
    self.assertIn(("a", "b"), workload)
    for cl in workload.keys():
      self.assertIsInstance(cl, tuple)

  def test_compiled_workload_default_with_two_columns(self):
    domain = mbi.Domain(["a", "b"], [3, 3])
    workload = common.compiled_workload(domain, None)
    self.assertCountEqual(
        workload.keys(),
        [("a",), ("b",), ("a", "b")],
    )

  def test_precompute_marginals_standard_and_jax(self):
    domain = mbi.Domain(["a", "b"], [3, 4])
    data = mbi.Dataset.synthetic(domain, N=100)
    cliques = [("a",), ("a", "b")]

    result_standard = common.precompute_marginals(data, cliques, use_jax=False)
    self.assertIsInstance(result_standard, mbi.CliqueVector)
    self.assertEqual(result_standard.domain, domain)

    result_jax = common.precompute_marginals(data, cliques, use_jax=True)
    self.assertIsInstance(result_jax, mbi.CliqueVector)
    self.assertEqual(result_jax.domain, domain)

    for cl in cliques:
      np.testing.assert_allclose(
          result_standard[cl].datavector(), result_jax[cl].datavector()
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="valid",
          kwargs=dict(valid=np.array([[0, 0], [0, 1], [1, 1]])),
      ),
      dict(
          testcase_name="invalid",
          kwargs=dict(invalid=np.array([[0, 2], [1, 0], [1, 2]])),
      ),
      dict(
          testcase_name="mapping",
          kwargs=dict(mapping=np.array([0, 1])),
      ),
  )
  def test_relax_constraints_relaxes_only_contradicted_cells(self, kwargs):
    dom = mbi.Domain(("a", "b"), (2, 3))
    c = mbi.Constraint(dom, **kwargs)
    counts = np.zeros((2, 3))
    counts[0, 2] = 11.0  # > 5.0 * 2.0 -> relaxed
    counts[1, 0] = 9.0  # <= 5.0 * 2.0 -> stays forbidden
    m = mbi.LinearMeasurement(counts.ravel(), clique=("a", "b"), stddev=2.0)

    (relaxed_c,) = common.relax_constraints([c], [m], threshold=5.0)
    vals = np.asarray(relaxed_c.potential.values)
    self.assertEqual(vals[0, 2], 0.0)
    self.assertEqual(vals[1, 0], -np.inf)
    self.assertEqual(vals[1, 2], -np.inf)
    self.assertFalse(relaxed_c.is_deterministic)

  def test_relax_constraints_none_or_unmeasured_unchanged(self):
    dom = mbi.Domain(("a", "b"), (2, 3))
    c = mbi.Constraint(dom, invalid=np.array([[0, 2]]))
    m_ab = mbi.LinearMeasurement(np.full(6, 100.0), ("a", "b"), stddev=1.0)
    m_a = mbi.LinearMeasurement(np.full(2, 100.0), ("a",), stddev=1.0)

    (res_none,) = common.relax_constraints([c], [m_ab], threshold=None)
    self.assertIs(res_none, c)
    (res_unmeasured,) = common.relax_constraints([c], [m_a], threshold=5.0)
    self.assertIs(res_unmeasured, c)

  def test_relax_constraints_transposed_clique_and_all_cells(self):
    dom = mbi.Domain(("a", "b"), (2, 3))
    c = mbi.Constraint(dom, invalid=np.array([[0, 2], [1, 0]]))
    counts_ba = np.zeros((3, 2))
    counts_ba[2, 0] = 10.0  # (0, 2) in ('a', 'b')
    m1 = mbi.LinearMeasurement(counts_ba.ravel(), ("b", "a"), stddev=2.0)
    (c1,) = common.relax_constraints([c], [m1], threshold=3.0)
    np.testing.assert_array_equal(
        np.isneginf(c1.potential.values),
        [[False, False, False], [True, False, False]],
    )

    counts_ba[0, 1] = 10.0  # (1, 0) in ('a', 'b')
    m2 = mbi.LinearMeasurement(counts_ba.ravel(), ("b", "a"), stddev=2.0)
    (c2,) = common.relax_constraints([c1], [m2], threshold=3.0)
    np.testing.assert_array_equal(c2.potential.values, np.zeros((2, 3)))

  @parameterized.parameters(0, 1, 2)
  def test_random_3way_relaxation(self, seed):
    rng = np.random.default_rng(seed)
    dom = mbi.Domain(("a", "b", "c"), (2, 3, 4))
    mask = rng.random(dom.shape) < 0.4
    c = mbi.Constraint(dom, invalid=np.argwhere(mask))

    counts_cab = rng.normal(loc=5.0, scale=5.0, size=(4, 2, 3))
    m = mbi.LinearMeasurement(counts_cab.ravel(), ("c", "a", "b"), stddev=1.5)

    (relaxed_c,) = common.relax_constraints([c], [m], threshold=4.0)
    counts_abc = np.moveaxis(counts_cab, (0, 1, 2), (2, 0, 1))
    expected_invalid = mask & ~(counts_abc > 4.0 * 1.5)
    np.testing.assert_array_equal(
        np.isneginf(relaxed_c.potential.values), expected_invalid
    )


if __name__ == "__main__":
  absltest.main()
