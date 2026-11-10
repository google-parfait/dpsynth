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

"""Downloads and prepares the UCI Adult dataset for DPSynth examples."""

from __future__ import annotations

from absl import logging
import pandas as pd

# Base URL for the UCI Adult dataset files.
_UCI_ADULT_BASE = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
)

# Output CSV file path for the cleaned Adult dataset.
_OUTPUT_PATH = 'adult.csv'

# Column names for the UCI Adult dataset.
_ADULT_COLUMNS = (
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income>50K',
)


def main() -> None:
  logging.set_verbosity(logging.INFO)
  train_df = pd.read_csv(
      f'{_UCI_ADULT_BASE}/adult.data',
      names=_ADULT_COLUMNS,
      skipinitialspace=True,
  )
  test_df = pd.read_csv(
      f'{_UCI_ADULT_BASE}/adult.test',
      skiprows=1,
      names=_ADULT_COLUMNS,
      skipinitialspace=True,
  )
  df = pd.concat([train_df, test_df]).dropna(how='all')
  df['income>50K'] = df['income>50K'].astype(str).str.rstrip('.')

  df.to_csv(_OUTPUT_PATH, index=False)
  logging.info('Saved %d Adult rows to %s', len(df), _OUTPUT_PATH)


if __name__ == '__main__':
  main()
