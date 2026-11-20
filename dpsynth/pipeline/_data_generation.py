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

"""Library that implements data pipeline for data generation."""

from collections.abc import Iterable
import copy
import dataclasses
import enum
from typing import Any

import apache_beam as beam
from dpsynth.dataset_descriptors import creating_data_recorder_converter
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import aim
from dpsynth.pipeline_transformations import dataset_compression
from dpsynth.pipeline_transformations import dataset_encoding
from dpsynth.pipeline_transformations import diagnostic_info as diagnostic_info_lib
from dpsynth.pipeline_transformations import independent_mechanism
from dpsynth.pipeline_transformations import model as model_lib
from dpsynth.pipeline_transformations import mst
from dpsynth.pipeline_transformations import swift
from dpsynth.pipeline_transformations import types
from google.protobuf import message
import mbi
import pipeline_dp


class Mechanism(enum.StrEnum):
  """Mechanism to use for data generation.

  Attributes:
    INDEPENDENT (str): Measures 1-way marginals using the Gaussian mechanism and
      generates data from each column independently.
    MST (str): Minimum spanning tree. It selects and measures a subset of 2-way
      marginal queries whose attribute pairs form a spanning tree. The spanning
      tree is chosen based on the characteristics of the data in a private
      manner using the exponential mechanism. Read
      https://arxiv.org/abs/2108.04978 for more details.
    AIM (str): Adaptive iterative mechanism. It selects and measures a subset of
      low-dimensional marginals iteratively, based on the workload, the privacy
      budget, efficiency considerations, and the characteristics of the data.
      Read https://arxiv.org/abs/2201.12677 for more details.
  """

  INDEPENDENT = 'independent'
  MST = 'mst'
  AIM = 'aim'
  SWIFT = 'swift'


@dataclasses.dataclass(frozen=True)
class DataGenerationConfig:
  """Configuration for data generation.

  Attributes:
    epsilon: The epsilon privacy parameter.
    delta: The delta privacy parameter.
    mechanism: The mechanism to use for data generation.
    dataset_descriptor: The descriptor of the dataset.
    data_format: The format of the input data.
    aim_parameters: Parameters for the AIM mechanism. Should only be set if
      `mechanism` is Mechanism.AIM.
    swift_parameters: Parameters for the SWIFT mechanism. Should only be set if
      `mechanism` is Mechanism.SWIFT.
    num_out_records: The number of synthetic records to generate. If None, the
      anonmized with DP number of records in the input data is used.
    output_format: The format of the output synthetic data. If None, it defaults
      to `data_format`.
  """

  epsilon: float
  delta: float
  mechanism: Mechanism
  dataset_descriptor: dataset_descriptor.DatasetDescriptor
  data_format: types.DataFormat
  aim_parameters: aim.AIMParameters | None = None
  swift_parameters: swift.SwiftParameters | None = None
  num_out_records: int | None = None
  output_format: types.DataFormat | None = None

  def __post_init__(self):
    if self.output_format is None:
      object.__setattr__(self, 'output_format', self.data_format)

    if self.mechanism != Mechanism.AIM and self.aim_parameters is not None:
      raise ValueError('aim_parameters can only be set for AIM mechanism.')
    if self.mechanism != Mechanism.SWIFT and self.swift_parameters is not None:
      raise ValueError('swift_parameters can only be set for SWIFT mechanism.')

    if self.mechanism == Mechanism.AIM and self.aim_parameters is None:
      object.__setattr__(self, 'aim_parameters', aim.AIMParameters())
    if self.mechanism == Mechanism.SWIFT and self.swift_parameters is None:
      object.__setattr__(self, 'swift_parameters', swift.SwiftParameters())


@dataclasses.dataclass
class AdditionalOutput:
  """Additional output of the data generation pipeline."""

  diagnostic_info: (
      types.Collection[diagnostic_info_lib.DiagnosticInformation] | None
  ) = None


def _infer_backend(
    input_data: types.Collection[Any],
) -> pipeline_dp.PipelineBackend:
  """Creates a pipeline backend based on the type of `data`."""
  if isinstance(input_data, beam.PCollection):
    return pipeline_dp.BeamBackend()
  return pipeline_dp.LocalBackend()


def generate(
    input_data: types.Collection[Any],
    config: DataGenerationConfig,
    backend: pipeline_dp.PipelineBackend | None = None,
    additional_output: AdditionalOutput | None = None,
) -> types.Collection[Any]:
  """Creates a pipeline to generate synthetic data.

  Args:
    input_data: The data to generate synthetic data for. It can be either an
      iterable (for in-process run) or a PCollection (for Python Beam run).
    config: The configuration for data generation.
    backend: The backend to use for data generation. If None, the backend is
      automatically selected based on the type of `data`.
    additional_output: Additional output to populate diagnostic info.

  Returns:
    Synthetic data. PCollection if `data` is a PCollection, otherwise a
    list.
  """
  synthetic_data, _, _ = generate_and_return_model(
      input_data, config, backend, additional_output
  )
  return synthetic_data


def generate_from_model(
    model: types.Collection[mbi.MarkovRandomField],
    descriptor: types.Collection[dataset_descriptor.DatasetDescriptor],
    num_out_records: int,
    output_format: types.DataFormat,
    backend: pipeline_dp.PipelineBackend | None = None,
    proto_type: type[message.Message] | None = None,
) -> types.Collection[Any]:
  """Generates synthetic data from a trained model.

  Args:
    model: A singleton collection with the trained model.
    descriptor: A singleton collection with the dataset descriptor.
    num_out_records: Number of synthetic output records to generate.
    output_format: The format of the output synthetic data.
    backend: The backend to use for data generation.
    proto_type: unused for now.

  Returns:
    Synthetic data. PCollection if `model` is a PCollection,
    otherwise a list.
  """

  if backend is None:
    backend = _infer_backend(model)

  compressed_domain = backend.map(
      descriptor, lambda d: d.compressed_domain, 'GetDomain'
  )

  compressed_generated_synthetic_data = model_lib.generate_synthetic_data(
      backend,
      model,
      compressed_domain,
      num_out_records,
  )

  decoded_data = _uncompress_and_decode(
      compressed_generated_synthetic_data,
      descriptor,
      backend,
  )

  return _format_output_data(
      decoded_data,
      input_data=None,
      output_format=output_format,
      backend=backend,
      descriptor_col=descriptor,
      proto_type=proto_type,
  )


def _uncompress_and_decode(
    compressed_generated_synthetic_data: types.Collection[tuple[int, ...]],
    descriptor: types.Collection[dataset_descriptor.DatasetDescriptor],
    backend: pipeline_dp.PipelineBackend,
) -> types.Collection[Any]:
  """Uncompresses and decodes generated synthetic data."""
  encoded_generated_synthetic_data = backend.map_with_side_inputs(
      compressed_generated_synthetic_data,
      lambda row, desc: desc.uncompress(row),
      [descriptor],
      'Uncompress generated data',
  )

  return backend.map_with_side_inputs(
      encoded_generated_synthetic_data,
      lambda row, desc: desc.decode(row),
      [descriptor],
      'Decode generated data',
  )


def generate_and_return_model(
    input_data: types.Collection[Any],
    config: DataGenerationConfig,
    backend: pipeline_dp.PipelineBackend | None = None,
    additional_output: AdditionalOutput | None = None,
) -> tuple[
    types.Collection[Any],
    types.Collection[mbi.MarkovRandomField],
    types.Collection[dataset_descriptor.DatasetDescriptor],
]:
  """Creates a pipeline to generate synthetic data and returns the model.

  Args:
    input_data: The data to generate synthetic data for. It can be either an
      iterable (for in-process run) or a PCollection (for Python Beam run).
    config: The configuration for data generation.
    backend: The backend to use for data generation. If None, the backend is
      automatically selected based on the type of `data`.
    additional_output: Additional output to populate diagnostic info.

  Returns:
    A tuple containing:
    - Synthetic data. PCollection if `data` is a PCollection, otherwise a
    list.
    - A singleton collection containing the trained mbi.MarkovRandomField model.
    - A singleton collection containing the DatasetDescriptor for converting
      between original format and the format model supports.
  """
  if backend is None:
    backend = _infer_backend(input_data)

  data_tuples = _convert_records_to_tuples(
      input_data, config.dataset_descriptor, backend
  )

  num_pipeline_dp_aggregations = _get_num_pipeline_dp_aggregations(
      config.mechanism, config.dataset_descriptor
  )

  budget_accountant = pipeline_dp.PLDBudgetAccountant(
      config.epsilon,
      config.delta,
      num_aggregations=num_pipeline_dp_aggregations,
  )
  dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

  # Encode the dataset.
  encoded_data, encoded_descriptor = dataset_encoding.encode_dataset(
      data_tuples, backend, dp_engine, config.dataset_descriptor
  )

  diagnostic_info = diagnostic_info_lib.DiagnosticInformation(
      epsilon=config.epsilon,
      delta=config.delta,
      mechanism=config.mechanism,
      attribute_names=[a.name for a in config.dataset_descriptor.attributes],
  )
  # diagnostic_info_collection: singleton collection of DiagnosticInformation
  diagnostic_info_collection = backend.to_collection(
      [diagnostic_info], data_tuples, 'Create DiagnosticInfo'
  )

  # Compress the dataset.
  num_attributes = len(config.dataset_descriptor.attributes)
  compressed_data, compressed_descriptor = dataset_compression.compress_dataset(
      encoded_data, backend, dp_engine, encoded_descriptor, num_attributes
  )

  compressed_one_way_marginals = backend.map(
      compressed_descriptor,
      lambda x: x.compressed_measurements(),
      'CompressedMeasurements',
  )  # singleton collection of (mbi.LinearMeasurement,...)

  compressed_domain = backend.map(
      compressed_descriptor, lambda x: x.compressed_domain, 'GetDomain'
  )

  def add_compressed_sizes(
      diagnostic_info: diagnostic_info_lib.DiagnosticInformation,
      descriptor: dataset_descriptor.DatasetDescriptor,
  ) -> diagnostic_info_lib.DiagnosticInformation:
    diagnostic_info.compressed_attribute_sizes.extend(
        [a.compressed_size for a in descriptor.attributes]
    )
    return diagnostic_info

  diagnostic_info_collection = backend.map_with_side_inputs(
      diagnostic_info_collection,
      add_compressed_sizes,
      [compressed_descriptor],
      'Add Compressed Sizes',
  )
  if additional_output is not None:
    additional_output.diagnostic_info = diagnostic_info_collection

  match config.mechanism:
    case Mechanism.INDEPENDENT:
      model_collection = independent_mechanism.fit_model(
          backend,
          compressed_descriptor,
      )
    case Mechanism.MST:
      model_collection = mst.fit_model(
          backend,
          budget_accountant,
          dp_engine,
          num_attributes,
          compressed_data,
          compressed_one_way_marginals,
          compressed_domain,
          additional_output=additional_output,
      )
    case Mechanism.AIM:
      model_collection = aim.fit_model(
          backend,
          budget_accountant,
          compressed_data,
          compressed_descriptor,
          config.aim_parameters or aim.AIMParameters(),
          additional_output=additional_output,
      )
    case Mechanism.SWIFT:
      model_collection = swift.fit_model(
          backend,
          budget_accountant,
          compressed_data,
          compressed_descriptor,
          config.swift_parameters or swift.SwiftParameters(),
          additional_output=additional_output,
      )
    case _:
      raise ValueError(f'Unsupported mechanism: {config.mechanism}')
  # model_collection: singleton collection of mbi.MarkovRandomField

  budget_accountant.compute_budgets()

  if (
      additional_output is not None
      and additional_output.diagnostic_info is not None
  ):

    def add_dp_operations_fn(
        diag_info: diagnostic_info_lib.DiagnosticInformation,
    ) -> diagnostic_info_lib.DiagnosticInformation:
      diag_info = copy.deepcopy(diag_info)  # Beam doesn't like modifying inputs
      for spec in budget_accountant.mechanism_specs:
        dp_operation = diagnostic_info_lib.DPOperation(
            name=spec.name,
            mechanism_type=str(spec.mechanism_type),
            count=spec.count,
        )
        if spec._eps is not None:  # pylint: disable=protected-access
          dp_operation.epsilon = spec.eps
        if spec._delta is not None:  # pylint: disable=protected-access
          dp_operation.delta = spec.delta
        if spec.standard_deviation_is_set:
          dp_operation.sigma = spec.noise_standard_deviation
        diag_info.dp_operations.append(dp_operation)
      return diag_info

    additional_output.diagnostic_info = backend.map(
        additional_output.diagnostic_info,
        add_dp_operations_fn,
        'Add DP Operations',
    )

  # Generate synthetic data.
  compressed_generated_synthetic_data = model_lib.generate_synthetic_data(
      backend,
      model_collection,
      compressed_domain,
      config.num_out_records,
  )  # (tuple[int, ...])

  decoded_data = _uncompress_and_decode(
      compressed_generated_synthetic_data,
      compressed_descriptor,
      backend,
  )

  output_data = _format_output_data(
      decoded_data,
      input_data=input_data,
      output_format=config.output_format or config.data_format,
      backend=backend,
      descriptor_col=compressed_descriptor,
  )
  return output_data, model_collection, compressed_descriptor


def _format_output_data(
    generated_synthetic_data: (
        beam.PCollection[tuple[Any, ...]] | Iterable[tuple[Any, ...]]
    ),
    input_data: beam.PCollection[Any] | Iterable[Any] | None,
    output_format: types.DataFormat,
    backend: pipeline_dp.PipelineBackend,
    descriptor_col: (
        types.Collection[dataset_descriptor.DatasetDescriptor] | None
    ) = None,
    proto_type: type[message.Message] | None = None,
) -> beam.PCollection[Any] | Iterable[Any]:
  """Formats the output data according to the config."""
  if output_format == types.DataFormat.CSV:
    return generated_synthetic_data

  if descriptor_col is None:
    raise ValueError('descriptor_col is required for non-CSV output.')

  needs_proto = output_format in [
  ]

  one_element_of_input_data = None
  if needs_proto:
    if proto_type is not None:
      test_proto = proto_type()
      one_element_of_input_data = backend.to_collection(
          [test_proto], [], 'CreateTestProto'
      )
    elif input_data is not None:
      one_element_of_input_data = (
          input_data
          | 'Get One Element' >> beam.combiners.Sample.FixedSizeGlobally(1)
          | 'RemoveList' >> beam.Map(lambda x: x[0])
      )
    else:
      raise ValueError(
          f'Either input_data or proto_type is required for {output_format}'
          ' output format.'
      )

  if one_element_of_input_data is None:
    return backend.map_with_side_inputs(
        generated_synthetic_data,
        lambda row, desc: creating_data_recorder_converter.create_data_record_converter(
            desc, output_format, proto_type
        ).from_tuple(
            row
        ),
        [descriptor_col],
        f'ConvertTuplesTo{output_format}',
    )
  else:
    return backend.map_with_side_inputs(
        generated_synthetic_data,
        lambda row, desc, proto_elem: creating_data_recorder_converter.create_data_record_converter(
            desc, output_format, proto_type or type(proto_elem)
        ).from_tuple(
            row, proto_elem
        ),
        [descriptor_col, one_element_of_input_data],
        f'ConvertTuplesTo{output_format}',
    )


def _get_num_pipeline_dp_aggregations(
    mechanism: Mechanism, descriptor: dataset_descriptor.DatasetDescriptor
) -> int:
  """Computes the number of Pipeline DP aggregations."""
  # We need to count only PipelineDP aggregations (i.e. calls of DPEngine.*
  # methods). For example, Exponential mechanism is not PipelineDP operation.
  # This is needed for budget annotations in the pipeline. It does not
  # affect the privacy analysis nor any operations in the pipeline.
  match mechanism:
    case Mechanism.INDEPENDENT:
      num_aggregations = 1  # one way marginals
    case Mechanism.MST:
      num_aggregations = 2  # one way marginals and two way marginals
    case Mechanism.AIM:
      num_aggregations = 1  # one way marginals
    case Mechanism.SWIFT:
      num_aggregations = 1  # one way marginals
    case _:
      raise ValueError(f'Unsupported mechanism: {mechanism}')

  categorical_indices, numerical_indices = (
      dataset_encoding.get_indices_to_discretisize(descriptor)
  )

  if categorical_indices:
    num_aggregations += 1
  if numerical_indices:
    num_aggregations += 1

  return num_aggregations


def _convert_records_to_tuples(
    data: Iterable[Any] | beam.PCollection[Any],
    descriptor: dataset_descriptor.DatasetDescriptor,
    backend: pipeline_dp.PipelineBackend,
) -> beam.PCollection[tuple[Any, ...]] | Iterable[tuple[Any, ...]]:
  """Converts records to tuples."""
  record_converter = descriptor.data_record_converter
  return backend.map(data, record_converter.to_tuple, 'ConvertRecordsToTuples')


def _convert_tuples_to_records(
    data: beam.PCollection[tuple[Any, ...]] | Iterable[tuple[Any, ...]],
    descriptor: types.Collection[dataset_descriptor.DatasetDescriptor],
    backend: pipeline_dp.PipelineBackend,
    one_element_of_input_data: types.Collection[Any] | None,
) -> beam.PCollection[Any] | Iterable[Any]:
  """Converts tuples to records."""
  if one_element_of_input_data is None:
    return backend.map_with_side_inputs(
        data,
        lambda t, d: d.data_record_converter.from_tuple(t),
        [descriptor],
        'ConvertTuplesToRecords',
    )
  else:
    return backend.map_with_side_inputs(
        data,
        lambda t, d, e: d.data_record_converter.from_tuple(t, e),
        [descriptor, one_element_of_input_data],
        'ConvertTuplesToRecords',
    )
