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

"""Tests for examples/datasets/download.py."""

from __future__ import annotations

from absl.testing import absltest
from examples.datasets import download
from etils import epath
import pandas as pd


class DownloadTest(absltest.TestCase):

  def test_prepare_adult_cleans_whitespace_and_trailing_dots(self):
    tmp_dir = epath.Path(self.create_tempdir().full_path)
    train_file = tmp_dir / 'adult.data'
    test_file = tmp_dir / 'adult.test'
    row = (
        '39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical,'
        ' Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K\n'
    )
    train_file.write_text(row)
    test_file.write_text(
        '|1x3 Cross validator\n' + row.replace('<=50K', '>50K.')
    )

    out_dir = tmp_dir / 'out'
    csv_path = download.prepare_adult(
        out_dir, train_source=train_file, test_source=test_file
    )
    df = pd.read_csv(csv_path)
    self.assertLen(df, 2)
    self.assertEqual(list(df['income>50K']), ['<=50K', '>50K'])
    self.assertEqual(list(df['workclass']), ['State-gov', 'State-gov'])


if __name__ == '__main__':
  absltest.main()
