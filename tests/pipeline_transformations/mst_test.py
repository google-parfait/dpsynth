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

"""Tests for MST pipeline transformations."""

from unittest import mock

from absl.testing import absltest
from dpsynth.pipeline_transformations import marginals_computations
from dpsynth.pipeline_transformations import mst
import mbi
import numpy as np
import pipeline_dp


class MstTest(absltest.TestCase):

  def test_compute_graph_weights_3_attributes(self):
    # Create test data
    one_way_dp_marginals = [[
        mbi.LinearMeasurement(np.array([2, 8]), (0,)),
        mbi.LinearMeasurement(np.array([4, 6]), (1,)),
        mbi.LinearMeasurement(np.array([-1, 4, 7]), (2,)),
    ]]
    two_way_marginals = [
        # ravel() to convert to 1d vector, which is expected by the function.
        ((0, 1), np.array([[1, 3], [3, 3]]).ravel()),
        ((0, 2), np.array([[2, 2, 2], [2, 2, 2]]).ravel()),
        ((1, 2), np.array([[1, 1, 1], [1, 1, 1]]).ravel()),
    ]
    backend = pipeline_dp.LocalBackend()

    # Compute graph weights
    weights = marginals_computations.compute_errors(
        backend, one_way_dp_marginals, two_way_marginals
    )

    weights = list(weights)[0]

    self.assertAlmostEqual(weights[(0, 1)], 4)
    self.assertAlmostEqual(weights[(0, 2)], 11.6)
    self.assertAlmostEqual(weights[(1, 2)], 10.0)

  def test_filter_marginals_with_mst(self):
    marginals = [
        ((0, 1), np.array([1, 2, 3])),
        ((0, 2), np.array([4, 5, 6])),
        ((1, 2), np.array([7, 8, 9])),
    ]
    mst_edges = [[(0, 1), (1, 2)]]

    result = list(
        mst._filter_marginals_with_mst(
            pipeline_dp.LocalBackend(), marginals, mst_edges
        )
    )

    self.assertEqual(result[0][0], (0, 1))
    np.testing.assert_array_equal(result[0][1], np.array([1, 2, 3]))
    self.assertEqual(result[1][0], (1, 2))
    np.testing.assert_array_equal(result[1][1], np.array([7, 8, 9]))

  @mock.patch.object(mst.mst, "dp_maximum_spanning_tree", autospec=True)
  @mock.patch.object(marginals_computations, "compute_errors", autospec=True)
  def test_get_dp_maximum_spanning_tree(
      self, mock_compute_graph_weights, mock_dp_mst
  ):
    # Arrange
    backend = pipeline_dp.LocalBackend()
    accountant = pipeline_dp.PLDBudgetAccountant(
        total_epsilon=1.0, total_delta=1e-8
    )

    num_attributes = 3
    one_way_dp_marginals = [
        mbi.LinearMeasurement(noisy_measurement=np.array([10.0]), clique=(0,)),
        mbi.LinearMeasurement(noisy_measurement=np.array([10.0]), clique=(1,)),
        mbi.LinearMeasurement(noisy_measurement=np.array([10.0]), clique=(2,)),
    ]

    two_way_marginals = [
        (((0, 1), np.array([[1, 2], [3, 4]]).ravel()),),
        (((0, 2), np.array([[5, 6], [7, 8]]).ravel()),),
        (((1, 2), np.array([[9, 0], [1, 2]]).ravel()),),
    ]

    # Weights are chosen so that the MST is almost always is (0, 1) and (1, 2).
    mock_weights_data = {(0, 1): 10.0, (0, 2): 5.0, (1, 2): 0.0}
    mock_compute_graph_weights.return_value = [mock_weights_data]
    mock_dp_mst.return_value = [("0", "1"), ("0", "2")]

    # Act
    result_mst = mst._select_dp_maximum_spanning_tree(
        backend,
        accountant,
        one_way_dp_marginals,
        two_way_marginals,
        num_attributes,
    )
    accountant.compute_budgets()
    result_mst_list = list(result_mst)

    # Assert
    expected_mst_int_edges = [(0, 1), (0, 2)]
    self.assertLen(result_mst_list, 1)
    self.assertListEqual(sorted(result_mst_list[0]), expected_mst_int_edges)

  @mock.patch.object(mst.mst, "dp_maximum_spanning_tree", autospec=True)
  @mock.patch.object(marginals_computations, "compute_errors", autospec=True)
  def test_select_dp_maximum_spanning_tree_check_epsilon(
      self, mock_compute_graph_weights, mock_dp_mst
  ):
    # Arrange
    backend = pipeline_dp.LocalBackend()
    accountant = pipeline_dp.PLDBudgetAccountant(
        total_epsilon=1.0, total_delta=1e-8
    )

    num_attributes = 3
    one_way_dp_marginals = [
        mbi.LinearMeasurement(noisy_measurement=np.array([10.0]), clique=(0,)),
        mbi.LinearMeasurement(noisy_measurement=np.array([10.0]), clique=(1,)),
        mbi.LinearMeasurement(noisy_measurement=np.array([10.0]), clique=(2,)),
    ]

    two_way_marginals = [
        (((0, 1), np.array([[1, 2], [3, 4]])),),
        (((0, 2), np.array([[5, 6], [7, 8]])),),
        (((1, 2), np.array([[9, 0], [1, 2]])),),
    ]

    mock_weights_data = {(0, 1): 10.0, (0, 2): 5.0, (1, 2): 0.0}
    mock_compute_graph_weights.return_value = [mock_weights_data]

    selected_edges = [("0", "1"), ("1", "2")]
    mock_dp_mst.return_value = selected_edges

    # Act
    result_mst = mst._select_dp_maximum_spanning_tree(
        backend,
        accountant,
        one_way_dp_marginals,
        two_way_marginals,
        num_attributes,
    )
    accountant.compute_budgets()
    result_mst_list = list(result_mst)

    # Assert
    self.assertListEqual(result_mst_list[0], [(0, 1), (1, 2)])
    mock_dp_mst.assert_called_once()
    args, kwargs = mock_dp_mst.call_args

    # Check positional arguments (rng, weights)
    self.assertLen(args, 2)
    self.assertIsInstance(args[0], np.random.Generator)
    self.assertEqual(
        args[1], {("0", "1"): 10.0, ("0", "2"): 5.0, ("1", "2"): 0.0}
    )

    # Check keyword arguments (epsilon)
    # PLD is used, so the epsilon is slightly less than 1.0/2.
    self.assertAlmostEqual(
        kwargs["exponential_mechanism_epsilon"], 0.49999946, delta=1e-8
    )

  def test_fit_model(self):
    backend = pipeline_dp.LocalBackend()
    num_attributes = 2
    compressed_data = [(0, 1), (0, 1), (1, 0), (1, 1)]
    compressed_domain = [mbi.Domain(attributes=([0, 1]), shape=(2, 2))]

    # Mock one-way DP marginals.
    # For attribute 0: counts for value 0 and 1
    one_way_m0 = mbi.LinearMeasurement(
        noisy_measurement=np.array([2.0, 2.0]), clique=(0,)
    )
    # For attribute 1: counts for value 0 and 1
    one_way_m1 = mbi.LinearMeasurement(
        noisy_measurement=np.array([1.0, 3.0]), clique=(1,)
    )
    compressed_one_way_marginals = [[one_way_m0, one_way_m1]]

    budget_accountant = pipeline_dp.PLDBudgetAccountant(1.0, 1e-5)
    dp_engine = pipeline_dp.DPEngine(
        budget_accountant,
        backend,
    )

    result = mst.fit_model(
        backend,
        budget_accountant,
        dp_engine,
        num_attributes,
        compressed_data,
        compressed_one_way_marginals,
        compressed_domain,
    )
    budget_accountant.compute_budgets()

    result_list = list(result)
    self.assertLen(result_list, 1)
    fitted_model = result_list[0]

    self.assertIsInstance(fitted_model, mbi.MarkovRandomField)
    self.assertEqual(fitted_model.domain.shape, (2, 2))
    self.assertEqual(tuple(fitted_model.potentials.cliques), ((0, 1),))


if __name__ == "__main__":
  absltest.main()
