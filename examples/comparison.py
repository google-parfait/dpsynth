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

"""Script to compare real and synthetic data."""

import dataclasses
import json
from typing import Any

from absl import app
from absl import flags
from absl import logging
import dpsynth
import pandas as pd
import sdmetrics

import pathlib


PathType = pathlib.Path
QualityReport = sdmetrics.reports.single_table.QualityReport
DiagnosticsReport = sdmetrics.reports.single_table.DiagnosticReport


@dataclasses.dataclass(frozen=True)
class CompareGroupByColumns:
  """Columns to cross compare."""

  categorical_columns: list[str]
  numerical_columns: list[str]

  def __post_init__(self):
    if (self.categorical_columns and not self.numerical_columns) or (
        not self.categorical_columns and self.numerical_columns
    ):
      raise ValueError(
          'Either both categorical_columns and numerical_columns must be set,'
          ' or neither.'
          f' Categorical columns: {", ".join(self.categorical_columns)}, '
          f' Numerical columns: {", ".join(self.numerical_columns)}',
      )


_REAL_DATA_PATH = flags.DEFINE_string(
    'real_data_path',
    'adult.csv',
    'Path to the real data to compare with.',
)

_SYNTHETIC_DATA_PATH = flags.DEFINE_string(
    'synthetic_data_path',
    'adult_synthetic.csv',
    'Path to the synthetic data to compare with.',
)

_DOMAIN_PATH = flags.DEFINE_string(
    'domain_path',
    'adult_domain.yaml',
    'Path to the domain file.',
)

_OUTPUT_REPORT_DIR = flags.DEFINE_string(
    'output_report_dir',
    None,
    'Path to the output report file.',
)

_COLUMNS_TO_COMPARE = flags.DEFINE_list(
    'columns_to_compare',
    ['age'],
    'Columns to compare.',
)


_CROSS_COMPARE_CATEGORICAL_COLUMNS = flags.DEFINE_list(
    'cross_compare_categorical_columns',
    ['sex', 'race'],
    'Categorical columns to group by for cross-column comparison.',
)

_CROSS_COMPARE_NUMERICAL_COLUMNS = flags.DEFINE_list(
    'cross_compare_numerical_columns',
    ['age'],
    'Numerical columns whose grouped means are compared.',
)


_MISSING_VALUE = flags.DEFINE_integer(
    'missing_value',
    -1,
    'Missing value to use for the histograms.',
)


def _pd_to_list(row: pd.Series, is_header: bool = False) -> list[str]:
  """Converts a pandas row to a Markdown-formatted list.

  Args:
    row (pd.Series): the pandas row to convert.
    is_header (bool): whether the row is the table header.

  Returns:
    A list of strings containing the row contents in markdown format.
  """
  result = ['|']
  result += [str(cell) + '|' for cell in row]
  result += ['\n']
  # add an "underline" row if this is the table header
  if is_header:
    result += ['|']
    result += ['---|' for _ in row]
    result += ['\n']
  return result


def _pd_dataframe_to_markdown(df: pd.DataFrame) -> str:
  """Converts a pandas dataframe to a Markdown-formatted table.

  Args:
    df (pd.DataFrame): the pandas dataframe to convert.

  Returns:
    A string containing the Markdown-formatted table representation
    of the input DataFrame.
  """
  # Create the Markdown table header
  result = _pd_to_list(df.columns, is_header=True)
  # Handle newline characters in header
  result = [field.replace('\\n', '\n') for field in result]
  # Add rows to the Markdown table
  for _, row in df.iterrows():
    result += _pd_to_list(row)
  return ''.join(result)


def _validate_column(
    column: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
):
  if column not in real_data.columns:
    raise ValueError(f'Column {column} not found in real data.')
  if column not in synthetic_data.columns:
    raise ValueError(f'Column {column} not found in synthetic data.')


def _compare_histograms(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    column_names: list[str] | None = None,
) -> None:
  """Compare the histograms of the given column."""
  if column_names is None:
    column_names = real_data.columns

  for column in column_names:
    _validate_column(column, real_data, synthetic_data)

  for column in column_names:
    comparison_df = pd.DataFrame({
        'real': real_data[column],
        'synthetic': synthetic_data[column],
    }).fillna(_MISSING_VALUE.value)
    print(f'Histogram for column {column}:')
    print(_pd_dataframe_to_markdown(comparison_df))


def _compare_histograms_cross_columns(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    compare_by_columns: CompareGroupByColumns,
) -> None:
  """Compare the histograms of the given column."""
  for column in compare_by_columns.categorical_columns:
    _validate_column(column, real_data, synthetic_data)
  for column in compare_by_columns.numerical_columns:
    _validate_column(column, real_data, synthetic_data)

  group_by_columns = compare_by_columns.categorical_columns
  for agg_column in compare_by_columns.numerical_columns:
    comparison_df = pd.DataFrame({
        'real': real_data.groupby(group_by_columns)[agg_column].mean(),
        'synthetic': (
            synthetic_data.groupby(group_by_columns)[agg_column].mean()
        ),
    }).fillna(_MISSING_VALUE.value)
    print(
        f'Cross-column histogram for columns {", ".join(group_by_columns)} and '
        f'{agg_column} mean:'
    )
    print(_pd_dataframe_to_markdown(comparison_df))


def _create_metadata_from_domain_yaml(
    domain_path: PathType,
) -> dict[str, Any]:
  """Creates a metadata from the domain YAML file."""
  dom = dpsynth.from_yaml_file(domain_path)
  metadata = {}
  cols = {}
  metadata['primary_key'] = ''
  for name, attr in dom.items():
    if isinstance(attr, dpsynth.NumericalAttribute):
      cols[name] = {'sdtype': 'numerical'}
    elif isinstance(attr, dpsynth.CategoricalAttribute):
      cols[name] = {'sdtype': 'categorical'}
    else:
      raise ValueError('Unknown attribute type: {type(attr)}')

  metadata['columns'] = cols
  return metadata


def _create_quality_report(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    metadata: dict[str, Any],
  ) -> QualityReport:
  """Creates a quality report."""
  print('Quality report:')
  quality_report = QualityReport()
  quality_report.generate(real_data, synthetic_data, metadata)
  if not quality_report.is_generated:
    raise ValueError('Failed to generate quality report.')

  return quality_report


def _create_diagnostics_report(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    metadata: dict[str, Any],
  ) -> DiagnosticsReport:
  """Creates a diagnostics report."""
  print('Diagnostics report:')
  diagnostics_report = DiagnosticsReport()
  diagnostics_report.generate(real_data, synthetic_data, metadata)
  if not diagnostics_report.is_generated:
    raise ValueError('Failed to generate diagnostics report.')

  return diagnostics_report


def main(_) -> None:
  real_data_path = PathType(_REAL_DATA_PATH.value)
  synthetic_data_path = PathType(_SYNTHETIC_DATA_PATH.value)
  domain_path = PathType(_DOMAIN_PATH.value)

  if not real_data_path.exists():
    raise ValueError(f'Real data path {real_data_path} does not exist.')
  if not synthetic_data_path.exists():
    raise ValueError(
        f'Synthetic data path {synthetic_data_path} does not exist.'
    )
  if not domain_path.exists():
    raise ValueError(f'Domain path {domain_path} does not exist.')

  real_data = pd.read_csv(real_data_path)
  synthetic_data = pd.read_csv(synthetic_data_path)
  sdmetrics_metadata = _create_metadata_from_domain_yaml(domain_path)
  columns_to_compare = _COLUMNS_TO_COMPARE.value
  columns_to_cross_compare = CompareGroupByColumns(
      categorical_columns=_CROSS_COMPARE_CATEGORICAL_COLUMNS.value or [],
      numerical_columns=_CROSS_COMPARE_NUMERICAL_COLUMNS.value or [],
  )

  # Compare histograms of the given columns.
  _compare_histograms(real_data, synthetic_data, columns_to_compare)

  # Compare grouped histograms of the given columns and aggregated column.
  _compare_histograms_cross_columns(
      real_data,
      synthetic_data,
      columns_to_cross_compare,
  )

  # Use SDMetrics to compare the data and generate reports.
  quality_report = _create_quality_report(
      real_data, synthetic_data, sdmetrics_metadata
  )
  diagnostics_report = _create_diagnostics_report(
      real_data, synthetic_data, sdmetrics_metadata
  )

  if _OUTPUT_REPORT_DIR.value:
    output_report_dir = PathType(_OUTPUT_REPORT_DIR.value)
    if not output_report_dir.exists():
      logging.info('Creating output report directory: %s', output_report_dir)
      output_report_dir.mkdir(parents=True, exist_ok=True)

    quality_report_path = output_report_dir / 'quality_report.txt'
    diagnostics_report_path = output_report_dir / 'diagnostics_report.txt'
    quality_report.save(str(quality_report_path))
    diagnostics_report.save(str(diagnostics_report_path))

  else:
    print(f'Quality report: {quality_report.get_info()}')
    print(f'Quality report Score: {quality_report.get_score()}')
    print(f'Diagnostics report: {diagnostics_report.get_info()}')
    print(f'Diagnostics report Score: {diagnostics_report.get_score()}')


if __name__ == '__main__':
  app.run(main)
