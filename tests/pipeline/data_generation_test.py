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
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as beam_test_util
import chex
from dpsynth.dataset_descriptors import csv_descriptor
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline import _data_generation as data_generation
from dpsynth.pipeline_transformations import types
import jax.numpy as jnp
import mbi
import pandas as pd
import pipeline_dp
import tensorflow as tf


class NoOpRecordConverter(dataset_descriptor.DataRecordConverter):
  """No-op data record converter for testing."""

  def to_tuple(self, record: tuple[Any, ...]) -> tuple[Any, ...]:
    return record

  def from_tuple(
      self, record: tuple[Any, ...], proto_object: Any | None = None
  ) -> tuple[Any, ...]:
    return record


class DataGenerationTest(parameterized.TestCase):

  def setUp(self):
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

  def _assert_proto_count_equal(self, actual, expected):
    self.assertEqual(len(actual), len(expected))
    actual_copy = list(actual)
    for exp in expected:
      found = False
      for act in actual_copy:
        if act == exp:
          actual_copy.remove(act)
          found = True
          break
      if not found:
        self.fail(f"Expected proto not found in actual: {exp}")

  def test_infer_backend(self):
    self.assertIsInstance(
        data_generation._infer_backend([1, 2]), pipeline_dp.LocalBackend
    )
    with test_pipeline.TestPipeline() as p:
      self.assertIsInstance(
          data_generation._infer_backend(p | "Create" >> beam.Create([1, 2])),
          pipeline_dp.BeamBackend,
      )

  def test_convert_tuples_to_records_with_coll_descriptor_no_one_element(self):
    data = [(1,), (2,)]
    descriptor = dataset_descriptor.DatasetDescriptor(
        attributes=[
            dataset_descriptor.AttributeDescriptor(
                name="attr1",
                data_type=dataset_descriptor.DataType.INT,
            ),
        ],
        data_record_converter=NoOpRecordConverter(),
    )
    backend = pipeline_dp.LocalBackend()
    descriptor_col = backend.to_collection([descriptor], [], "desc")
    records = list(
        data_generation._convert_tuples_to_records(
            data,
            descriptor_col,
            backend,
            one_element_of_input_data=None,
        )
    )
    self.assertEqual(records, data)

  def test_convert_tuples_to_records_with_coll_descriptor_with_one_element(
      self,
  ):
    data = [(1,), (2,)]
    descriptor = dataset_descriptor.DatasetDescriptor(
        attributes=[
            dataset_descriptor.AttributeDescriptor(
                name="attr1",
                data_type=dataset_descriptor.DataType.INT,
            ),
        ],
        data_record_converter=NoOpRecordConverter(),
    )
    backend = pipeline_dp.LocalBackend()
    descriptor_col = backend.to_collection([descriptor], [], "desc")
    records = list(
        data_generation._convert_tuples_to_records(
            data,
            descriptor_col,
            backend,
            one_element_of_input_data=[1],
        )
    )
    self.assertEqual(records, data)

  @parameterized.parameters(
      (types.DataFormat.CSV, [(1,), (2,)]),
      (
          types.DataFormat.TFRECORD,
          [
              tf.train.Example(
                  features=tf.train.Features(
                      feature={
                          "attr1": tf.train.Feature(
                              int64_list=tf.train.Int64List(value=[1])
                          )
                      }
                  )
              ),
              tf.train.Example(
                  features=tf.train.Features(
                      feature={
                          "attr1": tf.train.Feature(
                              int64_list=tf.train.Int64List(value=[2])
                          )
                      }
                  )
              ),
          ],
      ),
  )
  def test_format_output_data(self, output_format, expected_output):
    descriptor = dataset_descriptor.DatasetDescriptor(
        attributes=[
            dataset_descriptor.AttributeDescriptor(
                name="attr1",
                data_type=dataset_descriptor.DataType.INT,
            ),
        ],
        data_record_converter=NoOpRecordConverter(),
    )
    config = data_generation.DataGenerationConfig(
        epsilon=1.0,
        delta=1e-6,
        mechanism=data_generation.Mechanism.MST,
        dataset_descriptor=descriptor,
        data_format=types.DataFormat.CSV,
        output_format=output_format,
    )
    input_data = [(1,), (2,)]

    backend = pipeline_dp.LocalBackend()
    formatted_data = list(
        data_generation._format_output_data(
            generated_synthetic_data=input_data,
            input_data=input_data,
            output_format=config.output_format or config.data_format,
            backend=backend,
            descriptor_col=backend.to_collection([descriptor], [], "desc"),
        )
    )

    if output_format == types.DataFormat.TFRECORD:
      # Compare serialized representations for TFRecord
      self._assert_proto_count_equal(formatted_data, expected_output)
    else:
      self.assertEqual(formatted_data, expected_output)

  @parameterized.parameters(
      data_generation.Mechanism.INDEPENDENT,
      data_generation.Mechanism.MST,
  )
  def test_generate_with_local_backend_e2e(self, mechanism):
    descriptor = dataset_descriptor.DatasetDescriptor(
        attributes=[
            dataset_descriptor.AttributeDescriptor(
                name="attr1",
                data_type=dataset_descriptor.DataType.INT,
            ),
            dataset_descriptor.AttributeDescriptor(
                name="attr2",
                data_type=dataset_descriptor.DataType.INT,
            ),
        ],
        data_record_converter=NoOpRecordConverter(),
    )
    descriptor.attributes[0].categorical_attribute = (
        dataset_descriptor.domain.CategoricalAttribute(
            possible_values=list(range(100))
        )
    )
    descriptor.attributes[1].categorical_attribute = (
        dataset_descriptor.domain.CategoricalAttribute(
            possible_values=list(range(100))
        )
    )
    config = data_generation.DataGenerationConfig(
        epsilon=1.0,
        delta=1e-6,
        mechanism=mechanism,
        dataset_descriptor=descriptor,
        data_format=types.DataFormat.CSV,
        num_out_records=100,
    )
    input_data = []
    for i in range(100):
      for j in range(100):
        input_data.append((i, j))
    output_data, _, _ = data_generation.generate_and_return_model(
        input_data, config
    )
    self.assertLen(list(output_data), 100)

  def test_generate_with_diagnostic_info(self):
    descriptor = dataset_descriptor.DatasetDescriptor(
        attributes=[
            dataset_descriptor.AttributeDescriptor(
                name="attr1",
                data_type=dataset_descriptor.DataType.INT,
            ),
            dataset_descriptor.AttributeDescriptor(
                name="attr2",
                data_type=dataset_descriptor.DataType.INT,
            ),
        ],
        data_record_converter=NoOpRecordConverter(),
    )
    # Mock domains for attributes to verify compressed size collection
    descriptor.attributes[0].categorical_attribute = (
        dataset_descriptor.domain.CategoricalAttribute(
            possible_values=[0, 1, 2]
        )
    )
    descriptor.attributes[1].categorical_attribute = (
        dataset_descriptor.domain.CategoricalAttribute(possible_values=[0, 1])
    )

    config = data_generation.DataGenerationConfig(
        epsilon=1.0,
        delta=1e-6,
        mechanism=data_generation.Mechanism.INDEPENDENT,
        dataset_descriptor=descriptor,
        data_format=types.DataFormat.CSV,
        num_out_records=10,
    )
    input_data = []
    for i in range(10):
      input_data.append((i % 3, i % 2))

    additional_output = data_generation.AdditionalOutput()
    output_data, _, _ = data_generation.generate_and_return_model(
        input_data, config, additional_output=additional_output
    )
    list(output_data)

    self.assertIsNotNone(additional_output.diagnostic_info)
    diagnostic_info_list = list(additional_output.diagnostic_info)
    self.assertLen(diagnostic_info_list, 1)
    diagnostic_info = diagnostic_info_list[0]

    self.assertEqual(diagnostic_info.epsilon, 1.0)
    self.assertEqual(diagnostic_info.delta, 1e-6)
    self.assertEqual(
        diagnostic_info.mechanism, data_generation.Mechanism.INDEPENDENT
    )
    self.assertEqual(diagnostic_info.attribute_names, ["attr1", "attr2"])
    # Independent mechanism treats all values as frequent enough if not
    # filtering, but let's be safe and expect +1 or just check if it's >=.
    # Actually, for small epsilon it might compress everything to "Other" or
    # keep them. With epsilon=1.0 and 10 records, threshold might be small.
    # Let's just check they are present.
    self.assertLen(diagnostic_info.compressed_attribute_sizes, 2)

  def test_generate_from_model_e2e(self):
    descriptor = dataset_descriptor.DatasetDescriptor(
        attributes=[
            dataset_descriptor.AttributeDescriptor(
                name="attr1",
                data_type=dataset_descriptor.DataType.INT,
            ),
        ],
        data_record_converter=NoOpRecordConverter(),
    )
    descriptor.attributes[0].categorical_attribute = (
        dataset_descriptor.domain.CategoricalAttribute(
            possible_values=[0, 1, 2]
        )
    )
    domain_mrf = mbi.Domain(attributes=(0,), shape=(3,))
    clique = (0,)
    factor = mbi.Factor(domain=domain_mrf, values=jnp.array([1.0, 1.0, 1.0]))
    clique_vector = mbi.CliqueVector(
        domain_mrf,
        [clique],
        {clique: factor},
    )
    model = mbi.MarkovRandomField(
        potentials=clique_vector, marginals=clique_vector, total=10.0
    )
    config = data_generation.DataGenerationConfig(
        epsilon=1.0,
        delta=1e-6,
        mechanism=data_generation.Mechanism.MST,
        dataset_descriptor=descriptor,
        data_format=types.DataFormat.CSV,
        output_format=types.DataFormat.CSV,
    )
    backend = pipeline_dp.LocalBackend()
    models = backend.to_collection([model], [], "model")
    descriptors = backend.to_collection([descriptor], [], "desc")

    with mock.patch.object(
        dataset_descriptor.DatasetDescriptor,
        "compressed_domain",
        new_callable=mock.PropertyMock,
        return_value=domain_mrf,
    ), mock.patch.object(
        target=dataset_descriptor.DatasetDescriptor,
        attribute="uncompress",
        side_effect=lambda x: x,
    ):
      output_data = data_generation.generate_from_model(
          models, descriptors, 10, config.output_format or config.data_format
      )
      self.assertLen(list(output_data), 10)

  def test_generate_from_model_tfrecord_e2e(self):
    descriptor = dataset_descriptor.DatasetDescriptor(
        attributes=[
            dataset_descriptor.AttributeDescriptor(
                name="attr1",
                data_type=dataset_descriptor.DataType.INT,
            ),
        ],
        data_record_converter=NoOpRecordConverter(),
    )
    descriptor.attributes[0].categorical_attribute = (
        dataset_descriptor.domain.CategoricalAttribute(
            possible_values=[0, 1, 2]
        )
    )
    domain_mrf = mbi.Domain(attributes=(0,), shape=(3,))
    clique = (0,)
    factor = mbi.Factor(domain=domain_mrf, values=jnp.array([1.0, 1.0, 1.0]))
    clique_vector = mbi.CliqueVector(
        domain_mrf,
        [clique],
        {clique: factor},
    )
    model = mbi.MarkovRandomField(
        potentials=clique_vector, marginals=clique_vector, total=10.0
    )
    config = data_generation.DataGenerationConfig(
        epsilon=1.0,
        delta=1e-6,
        mechanism=data_generation.Mechanism.MST,
        dataset_descriptor=descriptor,
        data_format=types.DataFormat.CSV,
        output_format=types.DataFormat.TFRECORD,
    )
    backend = pipeline_dp.LocalBackend()
    models = backend.to_collection([model], [], "model")
    descriptors = backend.to_collection([descriptor], [], "desc")

    with mock.patch.object(
        dataset_descriptor.DatasetDescriptor,
        "compressed_domain",
        new_callable=mock.PropertyMock,
        return_value=domain_mrf,
    ), mock.patch.object(
        target=dataset_descriptor.DatasetDescriptor,
        attribute="uncompress",
        side_effect=lambda x: x,
    ):
      output_data = data_generation.generate_from_model(
          models, descriptors, 10, config.output_format or config.data_format
      )
      self.assertLen(list(output_data), 10)
      self.assertIsInstance(list(output_data)[0], tf.train.Example)

  @parameterized.parameters(
      (data_generation.Mechanism.INDEPENDENT, [], [], 1),
      (data_generation.Mechanism.MST, [], [], 2),
      (data_generation.Mechanism.INDEPENDENT, [0], [], 2),
      (data_generation.Mechanism.MST, [0], [], 3),
      (data_generation.Mechanism.INDEPENDENT, [], [0], 2),
      (data_generation.Mechanism.MST, [], [0], 3),
      (data_generation.Mechanism.INDEPENDENT, [0], [1], 3),
      (data_generation.Mechanism.MST, [0], [1], 4),
  )
  def test_get_num_pipeline_dp_aggregations(
      self,
      mechanism,
      categorical_indices,
      numerical_indices,
      expected_aggregations,
  ):
    with mock.patch(
        "dpsynth.pipeline_transformations.dataset_encoding.get_indices_to_discretisize"
    ) as mock_get_indices:
      mock_get_indices.return_value = (categorical_indices, numerical_indices)

      descriptor = dataset_descriptor.DatasetDescriptor(
          attributes=[],
          data_record_converter=NoOpRecordConverter(),
      )

      self.assertEqual(
          data_generation._get_num_pipeline_dp_aggregations(
              mechanism, descriptor
          ),
          expected_aggregations,
      )


if __name__ == "__main__":
  absltest.main()
