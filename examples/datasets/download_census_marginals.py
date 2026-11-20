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

"""Downloads pre-computed 1940 US Census marginals in CSV and CliqueVector form.

Downloads two compact exact summary representations of the 54 GB raw full-count
1940 US Census extract (131,902,398 rows x 135 discretized columns):
- ``census_marginals/*.csv`` (2.2 MB zip): 114 low-treewidth string-labeled
  marginal CSV tables for ``examples/postprocess_marginals.py``.
- ``census_1940_clique_vector.npz`` (415 MB): An ``int32`` ``mbi.CliqueVector``
  archive (with category and bin labels on ``mbi.Domain.labels``) containing all
  2-way marginals, all 3-way marginals with at most 10,000 cells, all 3-way
  marginals pairing at least two core demographic attributes (``age``, ``sex``,
  ``race``, ``hispan``, ``marst``, ``educ``, ``empstat``, ``incwage``,
  ``urban``, ``statefip``) with any third attribute, and curated 4-way and 5-way
  marginals for ``examples/discrete_synthesis.py``.

Derived from public-domain 1940 US Census data collected by the U.S. Census
Bureau and harmonized by IPUMS USA (Ruggles et al., IPUMS USA: Version 15.0,
Minneapolis, MN: IPUMS, 2024, https://doi.org/10.18128/D010.V15.0). Consistent
with IPUMS Terms of Use, only aggregate summary cross-tabulations are
distributed; no individual-level microdata is included.
"""

from __future__ import annotations

import io
import pathlib
from urllib import request
import zipfile

from absl import logging

_RELEASE_BASE = (
    'https://github.com/google-parfait/dpsynth/releases/download/datasets'
)
_CSV_MARGINALS_URL = f'{_RELEASE_BASE}/census_1940_marginals.zip'
_CLIQUE_VECTOR_URL = f'{_RELEASE_BASE}/census_1940_clique_vector.npz'

_DATASETS_DIR = pathlib.Path(__file__).parent
_CSV_OUTPUT_DIR = _DATASETS_DIR / 'census_marginals'
_CLIQUE_VECTOR_PATH = _DATASETS_DIR / 'census_1940_clique_vector.npz'


def main() -> None:
  logging.set_verbosity(logging.INFO)
  _CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  for old_csv in _CSV_OUTPUT_DIR.glob('*.csv'):
    old_csv.unlink()

  with request.urlopen(_CSV_MARGINALS_URL) as response:
    payload = response.read()
  with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(_CSV_OUTPUT_DIR)

  num_tables = sum(1 for _ in _CSV_OUTPUT_DIR.glob('*.csv'))
  logging.info('Extracted %d tables to %s', num_tables, _CSV_OUTPUT_DIR)

  request.urlretrieve(_CLIQUE_VECTOR_URL, _CLIQUE_VECTOR_PATH)
  logging.info('Saved CliqueVector archive to %s', _CLIQUE_VECTOR_PATH)


if __name__ == '__main__':
  main()
