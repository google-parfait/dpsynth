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

r"""Main program to launch synthetic data generation jobs.

python3 bin/main.py \
  --dataset=/path/to/dataset.csv \
  --domain=/path/to/domain.yaml \
  --epsilon=1.0 \
  --delta=1e-8 \
  --mechanism=aim \
  --output_path=/path/to/output.csv \
  --alsologtostderr
"""

from absl import app
from absl import flags
import dpsynth
from bin import _read_csv_args
import numpy as np
import pandas as pd

_DATASET_PATH = flags.DEFINE_string(
    'dataset',
    None,
    'Path to the dataset to generate synthetic data for.',
    required=True,
)

_DOMAIN_PATH = flags.DEFINE_string(
    'domain',
    None,
    'Path to the domain file.',
    required=True,
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

_MECHANISM = flags.DEFINE_enum(
    'mechanism',
    'mst',
    ['mst', 'aim', 'independent'],
    'Mechanism to use.',
)

_SEED = flags.DEFINE_integer(
    'seed',
    0,
    'Seed for random number generation.',
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    'Path to the output file.',
    required=True,
)

_CSV_FIELD_SEPARATOR = flags.DEFINE_enum(
    'field_separator',
    None,
    ['tab', 'pipe', 'comma', 'semicolon', 'space'],
    'Field separator for the input CSV.',
)

_CSV_COLUMN_NAMES = flags.DEFINE_list(
    'column_names',
    None,
    'Column names to use when the input CSV does not have a header.',
)

_CSV_COLUMN_COUNT = flags.DEFINE_integer(
    'column_count',
    None,
    'Number of columns when the input CSV does not have a header.',
)


def main(_):
  np.random.seed(_SEED.value)
  read_csv_kwargs = _read_csv_args.ReadCsvArgs(
      field_separator=_CSV_FIELD_SEPARATOR.value,
      column_names=_CSV_COLUMN_NAMES.value,
      column_count=_CSV_COLUMN_COUNT.value,
  ).to_read_csv_kwargs()
  df = pd.read_csv(_DATASET_PATH.value, **read_csv_kwargs)
  attribute_domains = dpsynth.domain.from_yaml_file(_DOMAIN_PATH.value)

  match _MECHANISM.value:
    case 'mst':
      mechanism_config = dpsynth.MSTConfig()
    case 'aim':
      mechanism_config = dpsynth.AIMConfig()
    case 'independent':
      mechanism_config = dpsynth.IndependentConfig()
    case _:
      raise ValueError(f'Unknown mechanism: {_MECHANISM.value}')

  mechanism = dpsynth.TabularConfig(
      discrete_mechanism=mechanism_config,
  ).calibrate(attribute_domains, epsilon=_EPSILON.value, delta=_DELTA.value)
  result = mechanism(np.random.default_rng(_SEED.value), df)  # pyrefly: ignore[bad-argument-type]
  result.synthetic_data.to_csv(_OUTPUT_PATH.value, index=False)


if __name__ == '__main__':
  app.run(main)
