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

"""Main program to launch synthetic data generation Beam jobs."""

from collections.abc import Mapping
from typing import Any

from absl import app
from absl import flags
import apache_beam as beam
import dpsynth
from dpsynth import data_generation
from dpsynth.dataset_descriptors import csv_descriptor
from dpsynth.dataset_descriptors import proto_descriptors
from dpsynth.dataset_descriptors import tfrecord_descriptor
from examples import _proto_class_flag
from dpsynth.pipeline_transformations import aim
from dpsynth.pipeline_transformations import input_output
from dpsynth.pipeline_transformations import types
import pipeline_dp


_DATASET_PATH = flags.DEFINE_string(
    'dataset',
    None,
    'Path to the dataset to generate synthetic data for.',
    required=True,
)

_DOMAIN_FILE = flags.DEFINE_string(
    'domain_file',
    None,
    'Path to the domain specification YAML file containing the data scheme.',
    required=False,
)

_EPSILON = flags.DEFINE_float(
    'epsilon',
    None,
    'Epsilon for differential privacy.',
    required=True,
)

_DELTA = flags.DEFINE_float(
    'delta',
    None,
    'Delta for differential privacy.',
    required=True,
)

_DATA_FORMAT = flags.DEFINE_enum_class(
    name='data_format',
    default=None,
    enum_class=types.DataFormat,
    help='Format of the dataset.',
    required=True,
)

_OUTPUT_FORMAT = flags.DEFINE_enum_class(
    name='output_format',
    default=None,
    enum_class=types.DataFormat,
    help=(
        'Format of the output dataset. If None, it will be the same as'
        ' data_format.'
    ),
    required=False,
)

_USE_BEAM = flags.DEFINE_boolean(
    'use_beam',
    True,
    'Whether to use Beam for data generation. If False local data'
    ' generation is used.',
    required=False,
)

_MECHANISM = flags.DEFINE_enum_class(
    'mechanism',
    data_generation.Mechanism.MST,
    data_generation.Mechanism,
    'Mechanism to use: MST (default). AIM is not supported yet.'
    ' See `data_generation.Mechanism` enum definition for more details.',
)

_ATTRIBUTES: flags.FlagHolder[list[str] | None] = flags.DEFINE_list(
    'attributes',
    None,
    'Comma separated attributes which should be used for generation. If None'
    'all attributes will be used.',
    required=False,
)

_NUM_OUT_RECORDS = flags.DEFINE_integer(
    'num_out_records',
    None,
    'Number of output records. If None, the output dataset of the size '
    'approximate to the input dataset is generated',
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    'Path to the output file.',
    required=True,
)

_DIAGNOSTIC_INFORMATION_PATH = flags.DEFINE_string(
    'diagnostic_information_path',
    None,
    'Path to the diagnostic information file. If specified, diagnostic'
    ' information is generated and saved to this path.',
    required=False,
)

_AIM_ROUNDS = flags.DEFINE_integer(
    'aim_rounds',
    100,
    'Number of rounds for AIM. Only used if mechanism is AIM.',
)

_AIM_PGM_ITERS = flags.DEFINE_integer(
    'aim_pgm_iters',
    1_000,
    'Number of PGM iterations for AIM. Only used if mechanism is AIM.',
)

_AIM_MAX_MODEL_SIZE = flags.DEFINE_integer(
    'aim_max_model_size',
    100,
    'Maximum size of the graphical model in megabytes. Only used if mechanism'
    ' is AIM.',
)

_MODEL_SAVE_PATH = flags.DEFINE_string(
    'model_save_path',
    None,
    'Path to save the generated model and DatasetDescriptor.',
    required=False,
)


def register_flag_validators():
  """Registers flag validators."""


def get_config() -> data_generation.DataGenerationConfig:
  """Gets the data generation configuration.

  Returns:
    A DataGenerationConfig object.

  Raises:
    NotImplementedError: If the data format is not supported.
  """
  if _DATA_FORMAT.value == types.DataFormat.TFRECORD:
    descriptor = tfrecord_descriptor.get_dataset_descriptor_for_tfrecord(
        tfrecord_descriptor.read_tfrecords_sample(_DATASET_PATH.value),
        attributes=_ATTRIBUTES.value,
    )
  elif _DATA_FORMAT.value == types.DataFormat.CSV:
    descriptor = csv_descriptor.get_dataset_descriptor_for_csv(
        csv_descriptor.read_csv_sample(_DATASET_PATH.value), _ATTRIBUTES.value
    )
  else:
    raise NotImplementedError(
        'Dataset descriptor is not supported for data format: '
        f'{_DATA_FORMAT.value}'
    )

  if _DOMAIN_FILE.value:
    domain_spec = dpsynth.from_yaml_file(_DOMAIN_FILE.value)
    descriptor.update_from_domain_specification(domain_spec)

  aim_parameters = (
      _get_aim_parameters()
      if _MECHANISM.value == data_generation.Mechanism.AIM
      else None
  )
  return data_generation.DataGenerationConfig(
      epsilon=_EPSILON.value,
      delta=_DELTA.value,
      mechanism=_MECHANISM.value,
      num_out_records=_NUM_OUT_RECORDS.value,
      data_format=_DATA_FORMAT.value,
      dataset_descriptor=descriptor,
      aim_parameters=aim_parameters,
      output_format=_OUTPUT_FORMAT.value,
  )


def _get_aim_parameters() -> aim.AIMParameters:
  return aim.AIMParameters(
      rounds=_AIM_ROUNDS.value,
      pgm_iters=_AIM_PGM_ITERS.value,
      max_model_size=_AIM_MAX_MODEL_SIZE.value,
  )


def main_beam(config: data_generation.DataGenerationConfig):
  """Runs data generation on Beam."""
  pipeline_runner = None

  with beam.Pipeline(runner=pipeline_runner) as pipeline:
    backend = pipeline_dp.BeamBackend()
    data = input_output.load_data_for_beam(
        pipeline,
        _DATASET_PATH.value,
        _DATA_FORMAT.value,
    )
    additional_output = data_generation.AdditionalOutput()
    synthetic_data, model_collection, descriptor = (
        data_generation.generate_and_return_model(
            input_data=data,
            config=config,
            backend=backend,
            additional_output=additional_output,
        )
    )
    if _DIAGNOSTIC_INFORMATION_PATH.value:
      input_output.save_diagnostic_info_pipeline(
          additional_output.diagnostic_info,
          _DIAGNOSTIC_INFORMATION_PATH.value,
      )
    attributes = config.dataset_descriptor.attribute_names
    if _MODEL_SAVE_PATH.value:
      input_output.save_model_pipeline(
          model_collection,
          descriptor,
          _MODEL_SAVE_PATH.value,
      )
    input_output.save_beam_data(
        synthetic_data,
        _OUTPUT_PATH.value,
        config.output_format,
        attributes,
    )


def main_local(config: data_generation.DataGenerationConfig):
  """Runs data generation locally."""
  backend = pipeline_dp.LocalBackend()
  data = input_output.load_data_local(_DATASET_PATH.value, _DATA_FORMAT.value)
  additional_output = data_generation.AdditionalOutput()
  synthetic_data, model_collection, descriptor = (
      data_generation.generate_and_return_model(
          input_data=data,
          config=config,
          backend=backend,
          additional_output=additional_output,
      )
  )

  attributes = config.dataset_descriptor.attribute_names
  if _MODEL_SAVE_PATH.value:
    input_output.save_model_local(
        _MODEL_SAVE_PATH.value, list(model_collection)[0], list(descriptor)[0]
    )
  input_output.save_data_local(
      _OUTPUT_PATH.value, synthetic_data, config.output_format, attributes  # pyrefly: ignore[bad-argument-type]
  )

  if _DIAGNOSTIC_INFORMATION_PATH.value:
    diagnostic_info = list(additional_output.diagnostic_info)[0]  # pyrefly: ignore[bad-argument-type]
    input_output.save_diagnostic_info_local(
        _DIAGNOSTIC_INFORMATION_PATH.value, diagnostic_info
    )


def main(_):
  config = get_config()

  if _USE_BEAM.value:
    main_beam(config)
  else:
    main_local(config)


if __name__ == '__main__':
  register_flag_validators()
  app.run(main)
