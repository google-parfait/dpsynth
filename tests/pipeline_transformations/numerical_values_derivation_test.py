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
from dpsynth.core import domain
from dpsynth.pipeline_transformations import numerical_values_derivation
import pipeline_dp


class DeriveNumericalValuesTest(absltest.TestCase):
  _EPSILON = 1e8
  _DELTA = 1e-10

  def test_compute_dp_quantiles_simple_quantiles(self):
    # field data: 0, 1, 2, .. 10.
    input_data = [{"field": i, "unused_field": -10} for i in range(0, 11)]

    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=self._EPSILON, total_delta=self._DELTA
    )

    backend = pipeline_dp.LocalBackend()
    dp_engine = pipeline_dp.DPEngine(accountant, backend)
    key_to_attr_dict = {
        "field": domain.NumericalAttribute(min_value=0, max_value=10),
    }

    quantiles = numerical_values_derivation._compute_dp_quantiles(
        input_data=input_data,
        backend=backend,
        dp_engine=dp_engine,
        key_to_attr=[key_to_attr_dict],
        num_attributes=1,
        num_quantile_buckets=4,  # 3 boundaries
    )

    accountant.compute_budgets()
    quantiles_list = list(quantiles)

    self.assertLen(quantiles_list, 1)
    self.assertEqual(quantiles_list[0][0], "field")

    # Ideal quantiles for uniform data in range [0, 10]: 2.5, 5.0, 7.5
    expected_quantiles = [2.5, 5.0, 7.5]

    # DP quantiles are noisy, so we use a large delta to test for approximate
    # correctness.
    self.assertSequenceAlmostEqual(
        quantiles_list[0][1], expected_quantiles, delta=10
    )

  def test_compute_dp_quantiles_non_zero_minima(self):
    # feat1 data: 10, 11, 12, .. 19.
    # F2 data: 50, 60, 70, .. 140.
    input_data = [
        {"feat1": i + 10, "feat2": (i * 10) + 50} for i in range(0, 10)
    ]

    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=self._EPSILON, total_delta=self._DELTA
    )

    backend = pipeline_dp.LocalBackend()
    dp_engine = pipeline_dp.DPEngine(accountant, backend)

    key_to_attr_dict = {
        "feat1": domain.NumericalAttribute(min_value=10, max_value=20),
        "feat2": domain.NumericalAttribute(min_value=50, max_value=150),
    }
    quantiles = numerical_values_derivation._compute_dp_quantiles(
        input_data=input_data,
        backend=backend,
        dp_engine=dp_engine,
        key_to_attr=[key_to_attr_dict],
        num_attributes=2,
        num_quantile_buckets=5,  # 4 boundaries
    )

    accountant.compute_budgets()
    quantiles_dict = dict(list(quantiles))

    self.assertLen(quantiles_dict, 2)
    self.assertIn("feat1", quantiles_dict)
    self.assertIn("feat2", quantiles_dict)

    self.assertLen(
        quantiles_dict["feat1"], 4
    )  # Expect 4 boundaries (20th, 40th, 60th, 80th) percentiles

    # feat1: Denormalized from [0.2, 0.4, 0.6, 0.8] to [10, 20] range yields
    # q_orig = q_scaled * 10 + 10
    expected_quantiles_feat1 = [12.0, 14.0, 16.0, 18.0]
    # DP quantiles are noisy, so we use a large delta to test for approximate
    # correctness.
    self.assertSequenceAlmostEqual(
        quantiles_dict["feat1"], expected_quantiles_feat1, delta=10
    )

    self.assertLen(quantiles_dict["feat2"], 4)

    # F2: Denormalized from [0.2, 0.4, 0.6, 0.8] to [50, 150] range yields
    # q_orig = q_scaled * 100 + 50
    expected_quantiles_f2 = [70.0, 90.0, 110.0, 130.0]
    # DP quantiles are noisy, so we use a large delta to test for approximate
    # correctness.
    self.assertSequenceAlmostEqual(
        quantiles_dict["feat2"], expected_quantiles_f2, delta=20
    )

  def test_compute_dp_quantiles_none_values(self):
    # field data: 10, 20, 30.
    input_data = [
        {"field": 10},
        {"field": None},
        {"field": 20},
        {"field": None},
        {"field": 30},
    ]

    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=self._EPSILON, total_delta=self._DELTA
    )

    backend = pipeline_dp.LocalBackend()
    dp_engine = pipeline_dp.DPEngine(accountant, backend)

    key_to_attr_dict = {
        "field": domain.NumericalAttribute(min_value=0, max_value=40),
    }

    quantiles = numerical_values_derivation._compute_dp_quantiles(
        input_data=input_data,
        backend=backend,
        dp_engine=dp_engine,
        key_to_attr=[key_to_attr_dict],
        num_attributes=1,
        num_quantile_buckets=2,  # 1 boundary (50th percentile)
    )

    accountant.compute_budgets()

    quantiles_list = list(quantiles)

    # Denormalized = 0.5 * 40 + 0
    expected_quantiles = [20]
    self.assertEqual(quantiles_list[0][0], "field")
    self.assertLen(quantiles_list[0][1], 1)
    # DP quantiles are noisy, so we use a large delta to test for approximate
    # correctness.
    self.assertSequenceAlmostEqual(
        quantiles_list[0][1], expected_quantiles, delta=20.0
    )

  def test_derive_numerical_attributes(self):
    # feat1 data: 0, 1, 2, .. 9
    # F2 data: 10, 20, .. 100
    input_data = [{"feat1": i, "feat2": (i + 1) * 10} for i in range(10)]

    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=self._EPSILON, total_delta=self._DELTA
    )
    backend = pipeline_dp.LocalBackend()
    dp_engine = pipeline_dp.DPEngine(accountant, backend)

    attribute_keys = ["feat1", "feat2"]
    num_buckets = 5  # 4 boundaries (20th, 40th, 60th, 80th percentiles)

    derived_attrs = numerical_values_derivation.derive_numerical_attributes(
        input_data=input_data,
        backend=backend,
        dp_engine=dp_engine,
        attribute_keys_to_derive=attribute_keys,
        num_quantile_buckets=num_buckets,
    )

    accountant.compute_budgets()
    derived_attrs_dict = {o.key: o for o in derived_attrs}

    self.assertLen(derived_attrs_dict, 2)
    self.assertIn("feat1", derived_attrs_dict)
    self.assertIn("feat2", derived_attrs_dict)

    # Assert feat1 (Range [0, 9])
    feat1_output = derived_attrs_dict["feat1"]
    self.assertEqual(feat1_output.attribute.min_value, 0)
    self.assertEqual(feat1_output.attribute.max_value, 9)
    self.assertLen(feat1_output.quantiles, 4)
    expected_quantiles_feat1 = [1.8, 3.6, 5.4, 7.2]
    self.assertSequenceAlmostEqual(
        feat1_output.quantiles, expected_quantiles_feat1, delta=10
    )

    # Assert F2 (Range [10, 100])
    f2_output = derived_attrs_dict["feat2"]
    self.assertEqual(f2_output.attribute.min_value, 10)
    self.assertEqual(f2_output.attribute.max_value, 100)
    self.assertLen(f2_output.quantiles, 4)

    expected_quantiles_f2 = [28.0, 46.0, 64.0, 82.0]
    self.assertSequenceAlmostEqual(
        f2_output.quantiles, expected_quantiles_f2, delta=10
    )

  def test_derive_numerical_attributes_empty_input(self):
    input_data = []
    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=self._EPSILON, total_delta=self._DELTA
    )
    backend = pipeline_dp.LocalBackend()
    dp_engine = pipeline_dp.DPEngine(accountant, backend)

    derived_attrs = numerical_values_derivation.derive_numerical_attributes(
        input_data=input_data,
        backend=backend,
        dp_engine=dp_engine,
        attribute_keys_to_derive=["field"],
        num_quantile_buckets=3,
    )
    accountant.compute_budgets()
    derived_attrs_list = list(derived_attrs)
    self.assertEmpty(derived_attrs_list)

  def test_derive_numerical_attributes_constant_val(self):
    input_data = [{"field": 10} for _ in range(10)]
    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=self._EPSILON, total_delta=self._DELTA
    )
    backend = pipeline_dp.LocalBackend()
    dp_engine = pipeline_dp.DPEngine(accountant, backend)

    derived_attrs = numerical_values_derivation.derive_numerical_attributes(
        input_data=input_data,
        backend=backend,
        dp_engine=dp_engine,
        attribute_keys_to_derive=["field"],
        num_quantile_buckets=4,
    )

    accountant.compute_budgets()
    derived_attrs_list = list(derived_attrs)

    self.assertLen(derived_attrs_list, 1)
    output = derived_attrs_list[0]
    self.assertEqual(output.key, "field")
    self.assertEqual(output.attribute.min_value, 10)
    # Constant column: derivation pads max_value to min_value + 1.
    self.assertEqual(output.attribute.max_value, 11.0)
    self.assertLen(output.quantiles, 3)
    self.assertSequenceAlmostEqual(
        output.quantiles, [10.0, 10.0, 10.0], delta=1.0
    )

  def test_derive_numerical_attributes_no_attributes_to_derive(self):
    input_data = [{"feat1": i, "feat2": (i + 1) * 10} for i in range(10)]
    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=self._EPSILON, total_delta=self._DELTA
    )
    backend = pipeline_dp.LocalBackend()
    dp_engine = pipeline_dp.DPEngine(accountant, backend)
    derived_attrs = numerical_values_derivation.derive_numerical_attributes(
        input_data=input_data,
        backend=backend,
        dp_engine=dp_engine,
        attribute_keys_to_derive=[],
        num_quantile_buckets=5,
    )
    accountant.compute_budgets()
    self.assertIsNone(derived_attrs)


if __name__ == "__main__":
  absltest.main()
