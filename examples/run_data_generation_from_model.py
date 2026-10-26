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

"""Main program to generate synthetic data from a saved model.

This binary supports generating synthetic data from a saved model
and saving the output as CSV or TFRecord.
"""

from absl import app
from absl import flags
import apache_beam as beam
from dpsynth import data_generation
from dpsynth.pipeline_transformations import input_output
from dpsynth.pipeline_transformations import types
import pipeline_dp

_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    None,
    'Path to the saved synthetic model.',
    required=True,
)

_NUM_OUT_RECORDS = flags.DEFINE_integer(
    'num_out_records',
    None,
    'Number of output records to generate. Required.',
    required=True,
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    'Path to the output file.',
    required=True,
)

_OUTPUT_FORMAT = flags.DEFINE_enum_class(
    name='output_format',
    default=None,
    enum_class=types.DataFormat,
    help='Format of the output dataset.',
    required=True,
)


def main(_) -> None:
  """Runs data generation using Beam."""

  pipeline_runner = None

  with beam.Pipeline(runner=pipeline_runner) as pipeline:
    backend = pipeline_dp.BeamBackend()

    header_str = None
    models, descriptors = input_output.load_model_pipeline(
        pipeline, _MODEL_PATH.value
    )

    generated_synthetic_data = data_generation.generate_from_model(
        model=models,
        descriptor=descriptors,
        num_out_records=_NUM_OUT_RECORDS.value,
        output_format=_OUTPUT_FORMAT.value,
        backend=backend,
    )

    input_output.save_beam_data(
        data=generated_synthetic_data,
        path=_OUTPUT_PATH.value,
        data_format=_OUTPUT_FORMAT.value,
        attributes=tuple(header_str.split(',') if header_str else ()),
    )


if __name__ == '__main__':
  app.run(main)
