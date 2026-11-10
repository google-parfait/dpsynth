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

"""Downloads and prepares the UCI MSNBC clickstream dataset for DPSynth examples."""

from __future__ import annotations

import gzip
import pathlib
import urllib.request

from absl import logging
import pandas as pd

# URL for the UCI MSNBC clickstream sequence file.
_UCI_MSNBC_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/msnbc-mld/msnbc990928.seq.gz'

# Output CSV file path for the prepared MSNBC sequence table.
_OUTPUT_PATH = pathlib.Path(__file__).parent / 'msnbc.csv'

# Minimum session length to retain and fixed sequence length to pad/truncate to.
_MIN_LENGTH = 5
_SEQ_LENGTH = 16

# Category vocabulary: index 0 is '<END>', and 1..17 are the UCI categories.
_CATEGORIES = (
    '<END>',
    'frontpage',
    'news',
    'tech',
    'local',
    'opinion',
    'on-air',
    'misc',
    'weather',
    'msn-news',
    'health',
    'living',
    'business',
    'msn-sports',
    'sports',
    'summary',
    'bbs',
    'travel',
)


def main() -> None:
  logging.set_verbosity(logging.INFO)
  with urllib.request.urlopen(_UCI_MSNBC_URL) as resp:
    raw_text = gzip.decompress(resp.read()).decode('utf-8')

  rows = []
  for line in raw_text.splitlines():
    line = line.strip()
    if not line or line.startswith('%') or line.startswith('frontpage'):
      continue
    tokens = [int(tok) for tok in line.split()]
    if len(tokens) < _MIN_LENGTH:
      continue
    padded = (tokens[:_SEQ_LENGTH] + [0] * _SEQ_LENGTH)[:_SEQ_LENGTH]
    rows.append([_CATEGORIES[idx] for idx in padded])

  columns = [f'step_{i:02d}' for i in range(1, _SEQ_LENGTH + 1)]
  df = pd.DataFrame(rows, columns=columns)
  df.to_csv(_OUTPUT_PATH, index=False)
  logging.info('Saved %d MSNBC rows to %s', len(df), _OUTPUT_PATH)


if __name__ == '__main__':
  main()
