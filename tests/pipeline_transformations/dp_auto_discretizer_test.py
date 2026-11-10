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
from dpsynth.pipeline_transformations import dp_auto_discretizer
import pipeline_dp


class DpAutoDiscretizerTest(absltest.TestCase):

  def test_create_transformations_via_dp_quantiles_output_as_expected(self):
    input_data = [
        {
            "value1": i,
            "value2": i * 2,
            "unused_field": -10,
        }
        for i in range(0, 100)
    ]

    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=1e8, total_delta=0
    )

    backend = pipeline_dp.LocalBackend()
    engine = pipeline_dp.DPEngine(accountant, backend)

    transformations = (
        dp_auto_discretizer.create_transformations_via_dp_quantiles(
            pcol=input_data,
            engine=engine,
            backend=backend,
            field_name_to_attribute={
                "value1": domain.NumericalAttribute(min_value=0, max_value=100),
                "value2": (
                    domain.NumericalAttribute(min_value=-15, max_value=200)
                ),
            },
            num_quanitle_buckets=4,
        )
    )

    accountant.compute_budgets()

    transformations = list(transformations)

    self.assertLen(transformations, 2)
    self.assertSameElements(
        [t[0] for t in transformations], ["value1", "value2"]
    )
    self.assertLen(transformations[0][1].possible_values, 4)
    self.assertLen(transformations[1][1].possible_values, 4)

    value1_transformation = [t[2] for t in transformations if t[0] == "value1"][
        0
    ]
    value1_intervals = [
        t[1].possible_values for t in transformations if t[0] == "value1"
    ][0]
    self.assertEqual(value1_transformation.transform(12), value1_intervals[0])
    self.assertEqual(value1_transformation.transform(35), value1_intervals[1])

  def test_quantile_output_as_expected(self):
    input_data = [
        {
            "value1": i,
            "value2": i * 2,
            "unused_field": -10,
        }
        for i in range(0, 100)
    ]

    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=1e8, total_delta=0
    )

    backend = pipeline_dp.LocalBackend()
    engine = pipeline_dp.DPEngine(accountant, backend)

    quantiles = dp_auto_discretizer._quantiles(
        pcol=input_data,
        engine=engine,
        backend=backend,
        field_name_to_attribute={
            "value1": domain.NumericalAttribute(min_value=0, max_value=100),
            "value2": domain.NumericalAttribute(min_value=-15, max_value=200),
        },
        num_quanitle_buckets=4,
    )

    accountant.compute_budgets()

    quantiles = list(quantiles)

    self.assertLen(quantiles, 2)
    self.assertLen(quantiles[0][1], 3)
    self.assertLen(quantiles[1][1], 3)
    value1_percentiles = [q[1] for q in quantiles if q[0] == "value1"][0]
    self.assertSequenceAlmostEqual(
        value1_percentiles,
        [25, 50, 75],
        delta=10,
    )
    value2_percentiles = [q[1] for q in quantiles if q[0] == "value2"][0]
    self.assertSequenceAlmostEqual(
        value2_percentiles,
        [50, 100, 150],
        delta=10,
    )

  def test_quantile_values_outside_range(self):
    input_data = [
        {
            "test": i,
        }
        for i in range(-100, 0)
    ]

    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=1e8, total_delta=0
    )

    backend = pipeline_dp.LocalBackend()
    engine = pipeline_dp.DPEngine(accountant, backend)

    quantiles = dp_auto_discretizer._quantiles(
        pcol=input_data,
        engine=engine,
        backend=backend,
        field_name_to_attribute={
            "test": domain.NumericalAttribute(min_value=0, max_value=100),
        },
        num_quanitle_buckets=4,
    )

    accountant.compute_budgets()

    quantiles = list(quantiles)

    self.assertLen(quantiles, 1)
    self.assertLen(quantiles[0][1], 3)
    self.assertSequenceAlmostEqual(
        quantiles[0][1],
        [0, 0, 0],
        delta=0.1,
    )


if __name__ == "__main__":
  absltest.main()
