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

from absl.testing import absltest
from dpsynth.core import domain
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import dataset_compression
import pipeline_dp


class FakeDataRecordConverter(dataset_descriptor.DataRecordConverter):
  """Fake data record converter for testing."""

  def to_tuple(self, record: Any) -> tuple[Any, ...]:
    raise NotImplementedError("to_tuple is not implemented")

  def from_tuple(self, record: tuple[Any, ...]) -> Any:
    raise NotImplementedError("from_tuple is not implemented")


class DatasetCompressionTest(absltest.TestCase):

  def test_compress_uncompress(self):
    # This test has flakiness rate of ~0.001, as a missing values can be as
    # non rare chosen: 2*P(0 + N(0, sigma^2) >= 3*sigma)
    # Input data: 2 columns:
    #  - first column 0, 1 are rare values, 2, 3, 4 not rare
    #  - second column 1, 2 are rare values, 0, 3, 4 not rare
    input_data = [(2, 0), (4, 3), (3, 4)] * 1000
    expected_compressed_data = [(0, 0), (2, 1), (1, 2)] * 1000
    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=100.0, total_delta=1e-10
    )
    backend = pipeline_dp.LocalBackend()
    dp_engine = pipeline_dp.DPEngine(accountant, backend)

    descriptor = dataset_descriptor.DatasetDescriptor(
        attributes=[
            dataset_descriptor.AttributeDescriptor(
                name="attr1",
                data_type=dataset_descriptor.DataType.INT,
                categorical_attribute=domain.CategoricalAttribute(
                    possible_values=[0, 1, 2, 3, 4]
                ),
            ),
            dataset_descriptor.AttributeDescriptor(
                name="attr1",
                data_type=dataset_descriptor.DataType.INT,
                categorical_attribute=domain.CategoricalAttribute(
                    possible_values=[0, 1, 2, 3, 4]
                ),
            ),
        ],
        data_record_converter=FakeDataRecordConverter(),
    )
    compressed_data, updated_descriptor = dataset_compression.compress_dataset(
        input_data, backend, dp_engine, [descriptor], num_attributes=2
    )
    accountant.compute_budgets()

    compressed_data = list(compressed_data)
    updated_descriptor = list(updated_descriptor)[0]

    self.assertEqual(updated_descriptor.attributes[0].compressed_size, 4)
    self.assertEqual(updated_descriptor.attributes[1].compressed_size, 4)
    self.assertEqual(compressed_data, expected_compressed_data)

    # Check that the uncompressed data is the same as the input data.
    uncompressed_data = list(
        dataset_compression.uncompress_dataset(
            compressed_data, backend, [updated_descriptor]
        )
    )
    self.assertEqual(uncompressed_data, input_data)


if __name__ == "__main__":
  absltest.main()
