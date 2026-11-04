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

r"""Generates synthetic PubMed canary documents for empirical privacy auditing.

Follows the synthetic canary construction of Mitchell et al. (arXiv:2606.10481)
by sampling realistic in-domain documents with unique details from an LLM at
elevated temperature and saving the resulting list of strings as JSON for
``examples/finetune_text.py``. Mitchell et al. report that intermediate
temperatures of 0.8 to 1.5 provide strong membership inference attack signal.
While this script uses the Gemini API for convenience, generating canaries
directly from the target pretrained base model is often preferred when auditing
model fine-tuning.
"""

from __future__ import annotations

from concurrent import futures
import json

from absl import logging
from google import genai
from google.genai import errors

# Output JSON file path for the generated canary strings.
_OUTPUT_PATH = 'pubmed_canaries.json'

# Number of canaries (must be even for seen/unseen split in DPFineTuner).
_NUM_CANARIES = 100

# Minimum word count required for a generated abstract to be kept as a canary.
_MIN_TOKENS = 60

# Number of concurrent worker threads for Gemini API requests.
_MAX_WORKERS = 20

# Sampling temperature (Mitchell et al. recommend 0.8 to 1.5 for MI auditing).
_TEMPERATURE = 1.5

# Gemma 4 model identifier used for generation via the Gemini API.
_MODEL = 'gemma-4-26b-a4b-it'

# Prompt instructing the model to generate a single realistic PubMed abstract.
_PROMPT = (
    'Generate a realistic biomedical research paper abstract of about 120 '
    'words. Include specific clinical or experimental details (such as cohort '
    'characteristics, interventions, molecular targets, sample sizes, and '
    'statistical outcomes) so the abstract is likely to be unique. Do not '
    'include a title or any meta-commentary. Respond only with the abstract '
    'body, nothing else. Please avoid placeholders for names, compounds, and '
    'places: use only realistic names.'
)


def main() -> None:
  # Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment by default.
  client = genai.Client()
  api_errors = 0
  short_count = 0

  def _generate(_: int) -> str:
    nonlocal api_errors, short_count
    while True:
      try:
        response = client.models.generate_content(
            model=_MODEL,
            contents=_PROMPT,
            config={'temperature': _TEMPERATURE},
        )
      except errors.APIError as e:
        api_errors += 1
        logging.warning('Gemini APIError (%d): %s', api_errors, e)
        if api_errors > 2 * _NUM_CANARIES:
          raise
        continue
      text = (response.text or '').strip()
      if len(text.split()) >= _MIN_TOKENS:
        return text
      short_count += 1

  with futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
    canaries = list(pool.map(_generate, range(_NUM_CANARIES)))

  logging.info('Discarded %d short generations.', short_count)
  with open(_OUTPUT_PATH, 'w') as f:
    json.dump(canaries, f, indent=2)


if __name__ == '__main__':
  main()
