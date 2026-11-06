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

"""Downloads the WildChat text and pre-annotated feature datasets.

Derived from allenai/WildChat-1M (Zhao et al., 2024), licensed under the Open
Data Commons Attribution License (ODC-By v1.0).
"""

from __future__ import annotations

import pathlib
from urllib import request

from absl import logging
import datasets
import pandas as pd

_DATASET = 'allenai/WildChat-1M'
_FEATURES_URL = (
    'https://github.com/google-parfait/dpsynth/releases/download/datasets'
    '/wildchat_features.parquet'
)
_TEXT_OUTPUT_PATH = pathlib.Path(__file__).parent / 'wildchat.parquet'
_FEATURES_OUTPUT_PATH = (
    pathlib.Path(__file__).parent / 'wildchat_features.parquet'
)


def _format_conversation(turns: list[dict[str, str]]) -> str:
  """Formats role/content conversation turns as a dialogue string."""
  return '\n\n'.join(f"{t['role']}: {t['content']}" for t in turns)


def main() -> None:
  logging.set_verbosity(logging.INFO)
  data = datasets.load_dataset(_DATASET, split='train')
  texts = [_format_conversation(c) for c in data['conversation']]
  df = pd.DataFrame({'text': texts})
  df.to_parquet(_TEXT_OUTPUT_PATH, index=False)
  logging.info('Saved %d conversations to %s', len(df), _TEXT_OUTPUT_PATH)

  request.urlretrieve(_FEATURES_URL, _FEATURES_OUTPUT_PATH)
  logging.info('Saved pre-annotated features to %s', _FEATURES_OUTPUT_PATH)


if __name__ == '__main__':
  main()
