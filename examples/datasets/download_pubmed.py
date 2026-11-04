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

"""Downloads and prepares the PubMed abstracts dataset for DPSynth examples."""

from __future__ import annotations

import json
import pathlib

from absl import logging
import datasets

# HuggingFace dataset identifier for PubMed scientific abstracts.
_DATASET = 'ccdv/pubmed-summarization'

# Output JSON file path for the cleaned list of abstract strings.
_OUTPUT_PATH = pathlib.Path(__file__).parent / 'pubmed.json'


def main() -> None:
  logging.set_verbosity(logging.INFO)
  data = datasets.load_dataset(_DATASET, split='train', trust_remote_code=True)
  abstracts = [t.strip() for t in data['abstract'] if t and t.strip()]

  with _OUTPUT_PATH.open('w') as f:
    json.dump(abstracts, f)
  logging.info('Saved %d PubMed abstracts to %s', len(abstracts), _OUTPUT_PATH)


if __name__ == '__main__':
  main()
