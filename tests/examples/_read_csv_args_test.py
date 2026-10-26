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

from absl.testing import absltest
from absl.testing import parameterized
from examples import _read_csv_args


class ReadCsvArgsTest(parameterized.TestCase):

  def test_column_names_and_column_count_both_set_raises_error(self):
    with self.assertRaises(ValueError):
      _read_csv_args.ReadCsvArgs(column_names=['a', 'b'], column_count=2)

  def test_to_read_csv_kwargs_with_column_names(self):
    args = _read_csv_args.ReadCsvArgs(
        field_separator='tab', column_names=['a', 'b']
    )
    self.assertEqual(
        args.to_read_csv_kwargs(),
        {'sep': '\t', 'names': ['a', 'b'], 'header': None},
    )

  def test_to_read_csv_kwargs_with_column_count(self):
    args = _read_csv_args.ReadCsvArgs(field_separator='pipe', column_count=10)
    self.assertEqual(
        args.to_read_csv_kwargs(),
        {'sep': '|', 'names': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'header': None},
    )

  def test_to_read_csv_kwargs_with_no_args(self):
    args = _read_csv_args.ReadCsvArgs()
    self.assertEqual(args.to_read_csv_kwargs(), {})

  def test_to_read_csv_kwargs_with_field_separator_only(self):
    args = _read_csv_args.ReadCsvArgs(field_separator='comma')
    self.assertEqual(args.to_read_csv_kwargs(), {'sep': ','})

  @parameterized.named_parameters(
      ('tab', 'tab', '\t'),
      ('pipe', 'pipe', '|'),
      ('comma', 'comma', ','),
      ('semicolon', 'semicolon', ';'),
      ('space', 'space', ' '),
  )
  def test_to_read_csv_kwargs_with_field_separator(
      self, field_separator, expected_sep
  ):
    args = _read_csv_args.ReadCsvArgs(field_separator=field_separator)
    self.assertEqual(args.to_read_csv_kwargs(), {'sep': expected_sep})

  def test_invalid_field_separator_raises_error(self):
    with self.assertRaises(ValueError):
      _read_csv_args.ReadCsvArgs(field_separator='invalid', column_count=2)


if __name__ == '__main__':
  absltest.main()
