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
from dpsynth import transformations
from dpsynth.discrete_mechanisms import common
import mbi
import numpy as np


def assert_serializable(obj):
  pickle.dumps(obj)


class CommonTest(absltest.TestCase):

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
    # Configurable workload degree.
    cliques = common.supporting_cliques(domain, workload=None, degree=2)
    self.assertCountEqual(
        cliques, list(itertools.combinations(domain.attributes, 2))
    )
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

  def test_compiled_workload_degree(self):
    domain = mbi.Domain(["a", "b", "c"], [3, 3, 3])
    workload = common.compiled_workload(domain, None, degree=2)
    self.assertCountEqual(
        workload,
        [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c")],
    )

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


if __name__ == "__main__":
  absltest.main()
