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

"""Arguments for reading CSV files."""

from collections.abc import Sequence
import dataclasses
from typing import Any


_FIELD_SEPARATOR_MAP = {
    'tab': '\t',
    'pipe': '|',
    'comma': ',',
    'semicolon': ';',
    'space': ' ',
}

_ALLOWED_FIELD_SEPARATORS = _FIELD_SEPARATOR_MAP.keys()

# Use this as the help string for flags of type ReadCsvArgs.
FLAG_HELP = (
    'Additional arguments for reading CSV files.\nThe field_separator can be'
    f' one of {", ".join(_ALLOWED_FIELD_SEPARATORS)}. You can skip setting the'
    ' field_separator if you are using a comma-separated file.\nSet the'
    ' column_names OR column_count if the CSV file does not have a header. Set'
    ' only one of column_names and column_count, setting both will cause an'
    ' error.'
)


@dataclasses.dataclass(frozen=True)
class ReadCsvArgs:
  """Args for reading CSV files."""

  field_separator: str | None = None
  column_names: list[str] | None = None
  column_count: int | None = None

  def __post_init__(self):
    if self.column_names and self.column_count:
      raise ValueError(
          'Both column_names and column_count can not be set at the same time.'
      )

    if (
        self.field_separator
        and self.field_separator not in _FIELD_SEPARATOR_MAP
    ):
      raise ValueError(
          f'Field separator {self.field_separator} is not supported. Valid'
          f' values are: {", ".join(_ALLOWED_FIELD_SEPARATORS)}.'
      )

  @property
  def _column_names_to_use(self) -> Sequence[str | int] | None:
    """Returns the column names to use for reading the CSV file."""
    if self.column_names:
      return self.column_names
    if self.column_count:
      return list(range(self.column_count))

    return None

  def to_read_csv_kwargs(self) -> dict[str, Any]:
    """Converts ReadCsvArgs to a dict of kwargs for pd.read_csv."""
    kwargs = {}
    if self.field_separator:
      kwargs['sep'] = _FIELD_SEPARATOR_MAP[self.field_separator]

    column_names = self._column_names_to_use
    if column_names:
      kwargs['names'] = column_names
      kwargs['header'] = None
    return kwargs
