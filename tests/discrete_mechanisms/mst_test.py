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
import dp_accounting
from dpsynth.discrete_mechanisms import common
from dpsynth.discrete_mechanisms import mst
import mbi
import numpy as np


class MSTTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)

  def test_dp_maximum_spanning_tree_infinite_rho(self):
    weights = {
        ('A', 'B'): 1.0,
        ('A', 'C'): 2.0,
        ('A', 'D'): 3.0,
        ('B', 'C'): 4.0,
        ('B', 'D'): 5.0,
        ('C', 'D'): 6.0,
    }

    expected_mst_edges = {
        frozenset({'C', 'D'}),
        frozenset({'B', 'D'}),
        frozenset({'A', 'D'}),
    }

    actual_edges_list = mst.dp_maximum_spanning_tree(
        np.random.default_rng(0), weights, zcdp_rho=100
    )

    actual_mst_edges = {frozenset(edge) for edge in actual_edges_list}

    self.assertEqual(actual_mst_edges, expected_mst_edges)

  def test_dp_maximum_spanning_tree_infinite_eps(self):
    weights = {
        ('A', 'B'): 1.0,
        ('A', 'C'): 2.0,
        ('A', 'D'): 3.0,
        ('B', 'C'): 4.0,
        ('B', 'D'): 5.0,
        ('C', 'D'): 6.0,
    }

    expected_mst_edges = {
        frozenset({'C', 'D'}),
        frozenset({'B', 'D'}),
        frozenset({'A', 'D'}),
    }

    actual_edges_list = mst.dp_maximum_spanning_tree(
        np.random.default_rng(0), weights, exponential_mechanism_epsilon=100
    )

    actual_mst_edges = {frozenset(edge) for edge in actual_edges_list}

    self.assertEqual(actual_mst_edges, expected_mst_edges)

  def test_fits_one_way_marginals(self):
    """MST + externally-supplied 1-ways should recover all one-way marginals."""
    data = mbi.Dataset.synthetic(mbi.Domain(['a', 'b', 'c'], [3, 4, 5]), N=1000)

    calibrated = mst.MSTConfig(pgm_iters=500).configure(budget=10000)

    # One-way marginals are now supplied externally by the synthesizer layer.
    # Simulate that by constructing them here with near-zero noise.
    initial_measurements = []
    for col in data.domain:
      true_marginal = data.project([col]).datavector()
      initial_measurements.append(
          mbi.LinearMeasurement(
              noisy_measurement=true_marginal,
              clique=(col,),
              stddev=1e-6,
              query=mbi.DatavectorQuery(use_for_total_estimation=True),
          )
      )

    result = calibrated(
        np.random.default_rng(0),
        data,
        initial_measurements=initial_measurements,
    )

    self.assertIsInstance(result, common.DiscreteMechanismResult)
    # 3 externally-supplied one-ways + 2 MST-selected pairwise = 5 total.
    self.assertLen(result.measurements, 2 * len(data.domain) - 1)
    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = result.model.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=1)

  def test_dp_event_returns_zcdp(self):
    config = mst.MSTConfig().configure(budget=1.0)
    event = config.dp_event
    self.assertIsInstance(event, dp_accounting.ZCDpEvent)


if __name__ == '__main__':
  absltest.main()
