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

from typing import Any
from unittest import mock

from absl.testing import absltest
from dpsynth.core import domain
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import dataset_encoding
from dpsynth.pipeline_transformations import numerical_values_derivation
import numpy as np
import pipeline_dp


class FakeDataRecordConverter(dataset_descriptor.DataRecordConverter):
  """Fake data record converter for testing."""

  def to_tuple(self, record: Any) -> tuple[Any, ...]:
    raise NotImplementedError("to_tuple is not implemented")

  def from_tuple(self, record: tuple[Any, ...]) -> Any:
    raise NotImplementedError("from_tuple is not implemented")


class DatasetEncodingTest(absltest.TestCase):

  @mock.patch(
      "dpsynth.pipeline_transformations.numerical_values_derivation.derive_numerical_attributes"
  )
  def test_encode_decode_mixed(self, mock_derive_numerical):
    backend = pipeline_dp.LocalBackend()
    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=5.0, total_delta=1e-10
    )
    dp_engine = pipeline_dp.DPEngine(accountant, backend)
    descriptors = dataset_descriptor.DatasetDescriptor(
        attributes=[
            dataset_descriptor.AttributeDescriptor(
                name="col1", data_type=dataset_descriptor.DataType.INT
            ),
            # already known possible values, no need to derive.
            dataset_descriptor.AttributeDescriptor(
                name="col2",
                data_type=dataset_descriptor.DataType.ENUM,
                categorical_attribute=domain.CategoricalAttribute(
                    possible_values=[0, 1]
                ),
            ),
            # numerical attribute
            dataset_descriptor.AttributeDescriptor(
                name="col3", data_type=dataset_descriptor.DataType.FLOAT
            ),
        ],
        data_record_converter=FakeDataRecordConverter(),
    )

    # Mock output of derive_numerical_attributes for col3
    mock_quantiles = (3.0, 7.0)
    mock_numerical_attribute = domain.NumericalAttribute(
        min_value=0.0, max_value=12.0, clip_to_range=False
    )
    mock_numerical_output = backend.to_collection(
        [
            numerical_values_derivation.NumericalAttributeOutput(
                key=2,
                attribute=mock_numerical_attribute,
                quantiles=mock_quantiles,
            )
        ],
        backend,
        "MockNumericalOutput",
    )
    mock_derive_numerical.return_value = mock_numerical_output

    # The first 100 rows are from 100 privacy units, so the will be chosen.
    # The last row is from different privacy unit, so it will be dropped.
    data = [("A", 1, 10.0), (3, 0, 5.5), (True, 1, 2.1)] * 100 + [("C", 0, 3.3)]
    # bins indices:
    # 0=(-inf, 0.0], 1=(0.0, 3.0], 2=(3.0, 7.0], 3=(7.0, 12.0], 4=(12.0, inf]
    expected_encoded_data = [(2, 1, 3), (1, 0, 2), (3, 1, 1)] * 100 + [
        (0, 0, 2)
    ]

    num_quantiles = 3
    encoded_data, output_descriptor = dataset_encoding.encode_dataset(
        data, backend, dp_engine, descriptors, num_quantiles
    )
    accountant.compute_budgets()
    encoded_data = list(encoded_data)
    output_descriptor = list(output_descriptor)[0]
    self.assertEqual(
        output_descriptor.attributes[0].categorical_attribute.possible_values,
        ["<OOD>", "3", "A", "True"],
    )
    self.assertEqual(
        output_descriptor.attributes[1].categorical_attribute.possible_values,
        [0, 1],
    )
    # Col3 is numerical
    self.assertIsNone(output_descriptor.attributes[2].categorical_attribute)
    self.assertEqual(
        output_descriptor.attributes[2].numerical_attribute,
        mock_numerical_attribute,
    )
    self.assertEqual(output_descriptor.attributes[2].quantiles, mock_quantiles)

    self.assertEqual(encoded_data, expected_encoded_data)

    # Verify that decoding gives back the original data, except rare values that
    # were not selected with DP partition selection.
    decoded_data = dataset_encoding.decode_dataset(
        encoded_data, [output_descriptor], backend
    )
    decoded_data = list(decoded_data)

    # bin index 1 -> (0.0, 3.0] -> 1.5
    # bin index 2 -> (3.0, 7.0] -> 5.0
    # bin index 3 -> (7.0, 12.0] -> 9.5
    expected_decoded_data = [
        ("A", 1, np.float64(9.5)),
        ("3", 0, np.float64(5.0)),
        ("True", 1, np.float64(1.5)),
    ] * 100 + [("<OOD>", 0, np.float64(5.0))]

    self.assertEqual(decoded_data, expected_decoded_data)

    mock_derive_numerical.assert_called_once_with(
        data, backend, dp_engine, [2], num_quantiles
    )

  def test_get_indices_to_discretisize(self):
    descriptors = dataset_descriptor.DatasetDescriptor(
        attributes=[
            dataset_descriptor.AttributeDescriptor(
                name="col1", data_type=dataset_descriptor.DataType.INT
            ),
            dataset_descriptor.AttributeDescriptor(
                name="col2",
                data_type=dataset_descriptor.DataType.ENUM,
                categorical_attribute=domain.CategoricalAttribute(
                    possible_values=[0, 1]
                ),
            ),
            dataset_descriptor.AttributeDescriptor(
                name="col3", data_type=dataset_descriptor.DataType.FLOAT
            ),
            dataset_descriptor.AttributeDescriptor(
                name="col4",
                data_type=dataset_descriptor.DataType.FLOAT,
                numerical_attribute=domain.NumericalAttribute(
                    min_value=0.0, max_value=10.0
                ),
            ),
        ],
        data_record_converter=FakeDataRecordConverter(),
    )

    (
        categorical_indices,
        numerical_indices,
    ) = dataset_encoding.get_indices_to_discretisize(descriptors)

    self.assertEqual(categorical_indices, [0])
    self.assertEqual(numerical_indices, [2, 3])


if __name__ == "__main__":
  absltest.main()
