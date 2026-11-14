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

import collections
import os

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
import chex
from dpsynth import domain
from dpsynth.dataset_descriptors import csv_descriptor
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import input_output
from dpsynth.pipeline_transformations import types
import jax.numpy as jnp
import mbi
import numpy as np
import pytest
tf = pytest.importorskip("tensorflow")


DATA_RECORD_CONVERTER = csv_descriptor.CSVConverter(
    attributes=collections.OrderedDict([
        ('int_attr', dataset_descriptor.DataType.INT),
        ('str_attr', dataset_descriptor.DataType.STR),
    ])
)


class InputOutputTest(absltest.TestCase):
  """Tests for the input_output module."""

  def setUp(self):
    """Sets up the test environment."""
    super().setUp()

    def _linear_measurement_eq(self, other):
      """Compares two LinearMeasurement objects for equality."""
      if not isinstance(other, mbi.LinearMeasurement):
        return NotImplemented
      return (
          self.clique == other.clique
          and self.stddev == other.stddev
          and self.query == other.query
          and jnp.array_equal(self.noisy_measurement, other.noisy_measurement)
      )

    mbi.LinearMeasurement.__eq__ = _linear_measurement_eq

    def csv_converter_eq(a, b):
      if not isinstance(b, csv_descriptor.CSVConverter):
        return NotImplemented
      return a._attributes == b._attributes

    csv_descriptor.CSVConverter.__eq__ = csv_converter_eq

  def test_save_csv_data(self):
    """Tests the save_csv_data function."""
    # Create a temporary directory for the output file.
    temp_dir = self.create_tempdir().full_path
    output_path = os.path.join(temp_dir, 'output.csv')
    data = [(1, 'test_string1'), (2, 'test_string2')]

    input_output.save_csv(data, output_path, ('test_int', 'test_string'))
    with open(output_path, 'r') as f:
      read_data = f.readlines()
    self.assertEqual(
        read_data,
        [
            'test_int,test_string\n',
            '1,test_string1\n',
            '2,test_string2\n',
        ],
    )

  def test_save_tfrecord_data_local(self):
    temp_dir = self.create_tempdir().full_path
    output_path = os.path.join(temp_dir, 'output.tfrecord')

    sample_data = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'test_int': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[1])
                    ),
                    'test_string': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b'test_string1'])
                    ),
                }
            )
        ),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'test_int': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[2])
                    ),
                    'test_string': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b'test_string2'])
                    ),
                }
            )
        ),
    ]

    input_output.save_data_local(
        output_path,
        sample_data,
        types.DataFormat.TFRECORD,
        ('test_int', 'test_string'),
    )

    dataset = tf.data.TFRecordDataset(output_path)
    read_data = [
        tf.train.Example.FromString(record.numpy()) for record in dataset
    ]

    self.assertEqual(read_data, sample_data)

  def test_save_load_model_local(self):
    descriptor = dataset_descriptor.DatasetDescriptor(
        attributes=[
            dataset_descriptor.AttributeDescriptor(
                name='int_attr',
                data_type=dataset_descriptor.DataType.INT,
                categorical_attribute=domain.CategoricalAttribute(
                    possible_values=[1, 2]
                ),
            ),
        ],
        data_record_converter=DATA_RECORD_CONVERTER,
    )
    domain_mrf = mbi.Domain(attributes=(0,), shape=(2,))
    clique = (0,)
    factor = mbi.Factor(domain=domain_mrf, values=np.array([1, 2]))
    clique_vector = mbi.CliqueVector(
        domain_mrf,
        [clique],
        {clique: factor},
    )
    model = mbi.MarkovRandomField(
        potentials=clique_vector, marginals=clique_vector, total=10
    )
    temp_dir = self.create_tempdir().full_path
    model_path = os.path.join(temp_dir, 'model.pkl')

    input_output.save_model_local(model_path, model, descriptor)

    loaded_model, loaded_descriptor = input_output.load_model_local(model_path)
    chex.assert_trees_all_equal(model, loaded_model)
    self.assertEqual(descriptor, loaded_descriptor)

  def test_save_load_model_pipeline(self):
    descriptor = dataset_descriptor.DatasetDescriptor(
        attributes=[
            dataset_descriptor.AttributeDescriptor(
                name='int_attr',
                data_type=dataset_descriptor.DataType.INT,
                categorical_attribute=domain.CategoricalAttribute(
                    possible_values=[1, 2]
                ),
            ),
        ],
        data_record_converter=DATA_RECORD_CONVERTER,
    )
    domain_mrf = mbi.Domain(attributes=(0,), shape=(2,))
    clique = (0,)
    factor = mbi.Factor(domain=domain_mrf, values=np.array([1, 2]))
    clique_vector = mbi.CliqueVector(
        domain_mrf,
        [clique],
        {clique: factor},
    )
    model = mbi.MarkovRandomField(
        potentials=clique_vector, marginals=clique_vector, total=10
    )
    temp_dir = self.create_tempdir().full_path
    model_path = os.path.join(temp_dir, 'model_pipeline')

    with beam.Pipeline() as pipeline:
      models_pc = pipeline | 'CreateModels' >> beam.Create([model])
      descriptors_pc = pipeline | 'CreateDescriptors' >> beam.Create(
          [descriptor]
      )
      input_output.save_model_pipeline(models_pc, descriptors_pc, model_path)

    with beam.Pipeline() as pipeline:
      loaded_models, loaded_descriptors = input_output.load_model_pipeline(
          pipeline, model_path
      )
      beam_testing_util.assert_that(
          loaded_models | 'MapModel' >> beam.Map(lambda m: m.total),
          beam_testing_util.equal_to([10]),
          label='CheckModels',
      )
      beam_testing_util.assert_that(
          loaded_descriptors
          | 'MapDesc' >> beam.Map(lambda d: len(d.attributes)),
          beam_testing_util.equal_to([1]),
          label='CheckDescriptors',
      )


if __name__ == '__main__':
  absltest.main()
