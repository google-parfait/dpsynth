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

"""Downloads and prepares standard benchmark datasets for DPSynth examples."""

from __future__ import annotations

import dataclasses
from typing import Literal

from absl import logging
from etils import epath
import pandas as pd
import simple_parsing

_UCI_ADULT_BASE = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
)
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


@dataclasses.dataclass(frozen=True)
class Config:
  """Command-line configuration for downloading example benchmark datasets.

  Attributes:
    dataset: Benchmark dataset to download and prepare.
    output_dir: Directory where prepared dataset files are saved.
  """

  dataset: Literal['adult'] = 'adult'
  output_dir: str = 'datasets'


def prepare_adult(
    output_dir: str | epath.Path = 'datasets',
    *,
    train_source: str | epath.Path = f'{_UCI_ADULT_BASE}/adult.data',
    test_source: str | epath.Path = f'{_UCI_ADULT_BASE}/adult.test',
) -> epath.Path:
  """Downloads and cleans the UCI Adult dataset."""
  out_dir = epath.Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  train_df = pd.read_csv(
      str(train_source), names=_ADULT_COLUMNS, skipinitialspace=True
  )
  test_df = pd.read_csv(
      str(test_source), skiprows=1, names=_ADULT_COLUMNS, skipinitialspace=True
  )
  df = pd.concat([train_df, test_df]).dropna(how='all')
  df['income>50K'] = df['income>50K'].astype(str).str.rstrip('.')

  csv_path = out_dir / 'adult.csv'
  df.to_csv(csv_path, index=False)
  logging.info('Saved %d Adult rows to %s', len(df), csv_path)
  return csv_path


def run(cfg: Config) -> epath.Path:
  """Runs the requested dataset download and preparation."""
  if cfg.dataset == 'adult':
    return prepare_adult(cfg.output_dir)
  raise ValueError(f'Unsupported dataset: {cfg.dataset}.')


def main() -> None:
  logging.set_verbosity(logging.INFO)
  run(simple_parsing.parse(Config))


if __name__ == '__main__':
  main()
