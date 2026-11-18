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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth import transformations
from dpsynth.core import domain
from dpsynth.dataset_descriptors import dataset_descriptor
import mbi
import numpy as np


class AttributeDescriptorTest(parameterized.TestCase):

  def test_encoded_size_with_categorical(self):
    attr_desc = dataset_descriptor.AttributeDescriptor(
        name="test_attribute",
        data_type=dataset_descriptor.DataType.INT,
        categorical_attribute=domain.CategoricalAttribute(
            possible_values=[1, 2, 3]
        ),
    )
    self.assertEqual(attr_desc.encoded_size, 3)

  def test_encoded_size_without_categorical(self):
    attr_desc = dataset_descriptor.AttributeDescriptor(
        name="test_attribute",
        data_type=dataset_descriptor.DataType.INT,
    )
    with self.assertRaises(ValueError):
      attr_desc.encoded_size  # pylint: disable=pointless-statement

  @parameterized.named_parameters(
      (
          "no_rare_values",
          np.array([5.0, 6.5, 9.0]),
          3,
      ),  # rare_values_mask=[False, False, False]]
      (
          "one_rare_value",
          np.array([1.0, 10.0, 7.6]),
          3,
      ),  # rare_values_mask=[True, False, False]]
      (
          "two_rare_values",
          np.array([-0.5, 2.1, 100.0]),
          2,
      ),  # rare_values_mask=[True, True, False]]
      (
          "all_rare_values",
          np.array([1.0, 0.5, -0.3]),
          1,
      ),  # rare_values_mask=[True, True, True]]
  )
  def test_compressed_size(self, noised_counts, expected_size):
    measurement = mbi.LinearMeasurement(
        noisy_measurement=noised_counts,
        clique=(0,),
        stddev=1.0,
    )
    attr_desc = dataset_descriptor.AttributeDescriptor(
        name="test_attribute",
        data_type=dataset_descriptor.DataType.INT,
        measurement=measurement,
    )
    self.assertEqual(attr_desc.compressed_size, expected_size)

  def test_encoding_transform(self):
    categorical_attribute = domain.CategoricalAttribute(
        possible_values=["A", "B", "C"]
    )
    attr_desc = dataset_descriptor.AttributeDescriptor(
        name="test_attribute",
        data_type=dataset_descriptor.DataType.ENUM,
        categorical_attribute=categorical_attribute,
    )
    self.assertIsInstance(
        attr_desc.encoding_transform, transformations.DataTransformation
    )
    self.assertEqual(attr_desc.encoding_transform("B"), 1)

  def test_compress_transform(self):
    attr_desc = dataset_descriptor.AttributeDescriptor(
        name="test_attribute",
        data_type=dataset_descriptor.DataType.INT,
        measurement=mbi.LinearMeasurement(
            noisy_measurement=np.array(
                [1.0, 5.0, 3.0]
            ),  # rare_values_mask=[True, False, False]]
            clique=(0,),
            stddev=1.0,
        ),
    )
    self.assertIsInstance(
        attr_desc.compress_transform, transformations.DataTransformation
    )
    self.assertEqual(attr_desc.compress_transform(0), 2)
    self.assertEqual(attr_desc.compress_transform(1), 0)
    self.assertEqual(attr_desc.compress_transform(2), 1)

  def test_getstate(self):
    attr_desc = dataset_descriptor.AttributeDescriptor(
        name="test_attribute",
        data_type=dataset_descriptor.DataType.INT,
        categorical_attribute=domain.CategoricalAttribute(
            possible_values=[1, 2, 3]
        ),
    )
    attr_desc._encoding_transform = mock.Mock()
    attr_desc._compress_transform = mock.Mock()

    state = attr_desc.__getstate__()
    self.assertNotIn("encoding_transform", state)
    self.assertNotIn("compress_transform", state)
    self.assertEqual(state["name"], "test_attribute")
    self.assertEqual(state["data_type"], dataset_descriptor.DataType.INT)
    self.assertEqual(
        state["categorical_attribute"],
        domain.CategoricalAttribute(possible_values=[1, 2, 3]),
    )


class MockDataRecordConverter(dataset_descriptor.DataRecordConverter):

  def to_tuple(self, record):
    return tuple(record)

  def from_tuple(self, record, not_used_proto_object=None):
    return record


class DatasetDescriptorTest(absltest.TestCase):

  def test_encoded_shape(self):
    attributes = [
        dataset_descriptor.AttributeDescriptor(
            name="attr1",
            data_type=dataset_descriptor.DataType.INT,
            categorical_attribute=domain.CategoricalAttribute(
                possible_values=[1, 2, 3]
            ),
        ),
        dataset_descriptor.AttributeDescriptor(
            name="attr2",
            data_type=dataset_descriptor.DataType.ENUM,
            categorical_attribute=domain.CategoricalAttribute(
                possible_values=["A", "B"]
            ),
        ),
    ]
    desc = dataset_descriptor.DatasetDescriptor(
        attributes=attributes, data_record_converter=MockDataRecordConverter()
    )
    self.assertEqual(desc.encoded_shape, (3, 2))

  def test_compressed_shape(self):
    attributes = [
        dataset_descriptor.AttributeDescriptor(
            name="attr1",
            data_type=dataset_descriptor.DataType.INT,
            measurement=mbi.LinearMeasurement(
                noisy_measurement=np.array(
                    [1.0, 5.0, 4.0]
                ),  # rare_values_mask=[True, False, False]]
                clique=(0,),
                stddev=1.0,
            ),
        ),
        dataset_descriptor.AttributeDescriptor(
            name="attr2",
            data_type=dataset_descriptor.DataType.ENUM,
            measurement=mbi.LinearMeasurement(
                noisy_measurement=np.array(
                    [1.0, 2.0, 5.0]
                ),  # rare_values_mask=[True, True, False]]
                clique=(1,),
                stddev=1.0,
            ),
        ),
    ]
    desc = dataset_descriptor.DatasetDescriptor(
        attributes=attributes, data_record_converter=MockDataRecordConverter()
    )
    self.assertEqual(desc.compressed_shape, (3, 2))

  def test_encode(self):
    attributes = [
        dataset_descriptor.AttributeDescriptor(
            name="attr1",
            data_type=dataset_descriptor.DataType.INT,
            categorical_attribute=domain.CategoricalAttribute(
                possible_values=[1, 2, 3]
            ),
        ),
        dataset_descriptor.AttributeDescriptor(
            name="attr2",
            data_type=dataset_descriptor.DataType.ENUM,
            categorical_attribute=domain.CategoricalAttribute(
                possible_values=["A", "B"]
            ),
        ),
    ]
    desc = dataset_descriptor.DatasetDescriptor(
        attributes=attributes, data_record_converter=MockDataRecordConverter()
    )
    self.assertEqual(desc.encode((2, "B")), (1, 1))

  def test_deencode(self):
    attributes = [
        dataset_descriptor.AttributeDescriptor(
            name="attr1",
            data_type=dataset_descriptor.DataType.INT,
            categorical_attribute=domain.CategoricalAttribute(
                possible_values=[1, 2, 3]
            ),
        ),
        dataset_descriptor.AttributeDescriptor(
            name="attr2",
            data_type=dataset_descriptor.DataType.ENUM,
            categorical_attribute=domain.CategoricalAttribute(
                possible_values=["A", "B"]
            ),
        ),
    ]
    desc = dataset_descriptor.DatasetDescriptor(
        attributes=attributes, data_record_converter=MockDataRecordConverter()
    )
    self.assertEqual(desc.decode((1, 1)), (2, "B"))

  def test_compress(self):
    attributes = [
        dataset_descriptor.AttributeDescriptor(
            name="attr1",
            data_type=dataset_descriptor.DataType.INT,
            measurement=mbi.LinearMeasurement(
                noisy_measurement=np.array(
                    [1.0, 5.0, 4.0]
                ),  # rare_values_mask=[True, False, False]]
                clique=(0,),
                stddev=1.0,
            ),
        ),
        dataset_descriptor.AttributeDescriptor(
            name="attr2",
            data_type=dataset_descriptor.DataType.ENUM,
            measurement=mbi.LinearMeasurement(
                noisy_measurement=np.array(
                    [1.0, 2.0, 3.0]
                ),  # rare_values_mask=[True, True, False]]
                clique=(1,),
                stddev=1.0,
            ),
        ),
    ]
    desc = dataset_descriptor.DatasetDescriptor(
        attributes=attributes, data_record_converter=MockDataRecordConverter()
    )
    self.assertEqual(desc.compress((0, 1)), (2, 1))
    self.assertEqual(desc.compress((1, 2)), (0, 0))

  def test_uncompress(self):
    attributes = [
        dataset_descriptor.AttributeDescriptor(
            name="attr1",
            data_type=dataset_descriptor.DataType.INT,
            measurement=mbi.LinearMeasurement(
                noisy_measurement=np.array(
                    [1.0, 7.0, 10.0]
                ),  # rare_values_mask=[True, False, False]]
                clique=(0,),
                stddev=1.0,
            ),
        ),
        dataset_descriptor.AttributeDescriptor(
            name="attr2",
            data_type=dataset_descriptor.DataType.ENUM,
            measurement=mbi.LinearMeasurement(
                noisy_measurement=np.array(
                    [1.0, 2.0, 10.0]
                ),  # rare_values_mask=[True, True, False]]
                clique=(1,),
                stddev=1.0,
            ),
        ),
    ]
    desc = dataset_descriptor.DatasetDescriptor(
        attributes=attributes, data_record_converter=MockDataRecordConverter()
    )
    self.assertEqual(desc.uncompress((1, 0)), (2, 2))
    self.assertIn(desc.uncompress((2, 1)), [(0, 0), (0, 1)])

  def test_encoded_domain(self):
    attributes = [
        dataset_descriptor.AttributeDescriptor(
            name="attr1",
            data_type=dataset_descriptor.DataType.INT,
            categorical_attribute=domain.CategoricalAttribute(
                possible_values=[1, 2, 3]
            ),
        ),
        dataset_descriptor.AttributeDescriptor(
            name="attr2",
            data_type=dataset_descriptor.DataType.ENUM,
            categorical_attribute=domain.CategoricalAttribute(
                possible_values=["A", "B"]
            ),
        ),
    ]
    desc = dataset_descriptor.DatasetDescriptor(
        attributes=attributes, data_record_converter=MockDataRecordConverter()
    )
    self.assertEqual(desc.encoded_domain.shape, (3, 2))
    self.assertEqual(desc.encoded_domain.attributes, (0, 1))

  def test_compressed_domain(self):
    attributes = [
        dataset_descriptor.AttributeDescriptor(
            name="attr1",
            data_type=dataset_descriptor.DataType.INT,
            measurement=mbi.LinearMeasurement(
                noisy_measurement=np.array(
                    [1.0, 5.0, 4.0]
                ),  # rare_values_mask=[True, False, False]]
                clique=(0,),
                stddev=1.0,
            ),
        ),
        dataset_descriptor.AttributeDescriptor(
            name="attr2",
            data_type=dataset_descriptor.DataType.ENUM,
            measurement=mbi.LinearMeasurement(
                noisy_measurement=np.array(
                    [1.0, 2.0, 5.0]
                ),  # rare_values_mask=[True, True, False]]
                clique=(1,),
                stddev=1.0,
            ),
        ),
    ]
    desc = dataset_descriptor.DatasetDescriptor(
        attributes=attributes, data_record_converter=MockDataRecordConverter()
    )
    self.assertEqual(desc.compressed_domain.shape, (3, 2))
    self.assertEqual(desc.compressed_domain.attributes, (0, 1))

  def test_all_attributes_initialized(self):
    attributes = [
        dataset_descriptor.AttributeDescriptor(
            name="attr1",
            data_type=dataset_descriptor.DataType.INT,
            categorical_attribute=domain.CategoricalAttribute(
                possible_values=[1, 2, 3]
            ),
        ),
        dataset_descriptor.AttributeDescriptor(
            name="attr2",
            data_type=dataset_descriptor.DataType.ENUM,
        ),
    ]
    desc = dataset_descriptor.DatasetDescriptor(
        attributes=attributes, data_record_converter=MockDataRecordConverter()
    )
    self.assertFalse(desc.all_attributes_initialized)
    desc.attributes[1].categorical_attribute = domain.CategoricalAttribute(
        possible_values=["A", "B"]
    )
    self.assertTrue(desc.all_attributes_initialized)

  def test_get_attribute_names(self):
    attr_descriptor1 = dataset_descriptor.AttributeDescriptor(
        name="test_attr1",
        data_type=dataset_descriptor.DataType.INT,
    )
    attr_descriptor2 = dataset_descriptor.AttributeDescriptor(
        name="test_attr2",
        data_type=dataset_descriptor.DataType.STR,
    )
    data_descriptor = dataset_descriptor.DatasetDescriptor(
        attributes=[attr_descriptor1, attr_descriptor2],
        data_record_converter=mock.MagicMock(),
    )
    self.assertEqual(
        data_descriptor.attribute_names, ("test_attr1", "test_attr2")
    )

  def test_update_from_domain_specification(self):
    attributes = [
        dataset_descriptor.AttributeDescriptor(
            name="attr1",
            data_type=dataset_descriptor.DataType.INT,
        ),
        dataset_descriptor.AttributeDescriptor(
            name="attr2",
            data_type=dataset_descriptor.DataType.INT,
        ),
    ]
    desc = dataset_descriptor.DatasetDescriptor(
        attributes=attributes, data_record_converter=MockDataRecordConverter()
    )
    domain_spec = {
        "attr1": domain.CategoricalAttribute(possible_values=[1, 2]),
        "attr2": domain.NumericalAttribute(min_value=0.0, max_value=10.0),
    }
    desc.update_from_domain_specification(domain_spec)
    self.assertEqual(
        desc.attributes[0].categorical_attribute,
        domain.CategoricalAttribute(possible_values=[1, 2]),
    )
    self.assertEqual(
        desc.attributes[1].numerical_attribute,
        domain.NumericalAttribute(min_value=0.0, max_value=10.0),
    )


if __name__ == "__main__":
  absltest.main()
