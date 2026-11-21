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

"""Integration test for California Census relational differential privacy synthesis."""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
import dpsynth
from dpsynth.relational import domain as rel_domain
from dpsynth.relational import synthesizer as rel_synth
from etils import epath
import jax
import numpy as np
import pandas as pd

jax.config.update('jax_enable_compilation_cache', False)


def _find_domain_path() -> epath.Path:
  """Resolves the path to domain.yaml across package and open-source layouts."""
  local_path = epath.Path(__file__).parent / 'domain.yaml'
  if local_path.exists():
    return local_path
  repo_root_path = (
      epath.resource_path('dpsynth').parent
      / 'examples/relational/california_census/domain.yaml'
  )
  if repo_root_path.exists():
    return repo_root_path
  pkg_path = (
      epath.resource_path('dpsynth')
      / 'examples/relational/california_census/domain.yaml'
  )
  if pkg_path.exists():
    return pkg_path
  cwd_path = epath.Path('examples/relational/california_census/domain.yaml')
  if cwd_path.exists():
    return cwd_path
  return local_path


def _generate_mock_california_data(
    num_households: int = 50,
    rng: np.random.Generator | None = None,
) -> dict[str, pd.DataFrame]:
  """Generates mock California census DataFrames adhering to domain.yaml."""
  if rng is None:
    rng = np.random.default_rng(42)

  # 1. Generate Household records
  h_ids = np.arange(1, num_households + 1)
  household_df = pd.DataFrame({
      'HOUSEHOLD': h_ids,
      'FARM': rng.integers(0, 2, size=num_households),
      'OWNERSHP': rng.integers(0, 3, size=num_households),
      'ACREHOUS': rng.integers(0, 3, size=num_households),
      'TAXINCL': rng.integers(0, 3, size=num_households),
      'PROPINSR': rng.integers(0, 60, size=num_households),
      'COSTELEC': rng.integers(0, 100, size=num_households),
      'VALUEH': rng.integers(0, 100, size=num_households),
      'ROOMS': rng.integers(0, 10, size=num_households),
      'PLUMBING': rng.integers(0, 3, size=num_households),
      'PUMA': rng.integers(0, 233, size=num_households),
  })

  # 2. Generate variable-sized child individual records
  # (1 to 4 individuals per household)
  group_sizes = rng.integers(1, 5, size=num_households)
  ind_households = np.repeat(h_ids, group_sizes)
  num_individuals = len(ind_households)

  individual_df = pd.DataFrame({
      'HOUSEHOLD': ind_households,
      'RELATE': rng.integers(0, 8, size=num_individuals),
      'SEX': rng.integers(0, 2, size=num_individuals),
      'AGE': rng.integers(0, 86, size=num_individuals),
      'MARST': rng.integers(0, 6, size=num_individuals),
      'RACE': rng.integers(0, 9, size=num_individuals),
      'CITIZEN': rng.integers(0, 4, size=num_individuals),
      'SPEAKENG': rng.integers(0, 6, size=num_individuals),
      'SCHOOL': rng.integers(0, 3, size=num_individuals),
      'EDUC': rng.integers(0, 11, size=num_individuals),
      'GRADEATT': rng.integers(0, 8, size=num_individuals),
      'SCHLTYPE': rng.integers(0, 4, size=num_individuals),
      'EMPSTAT': rng.integers(0, 4, size=num_individuals),
      'CLASSWKR': rng.integers(0, 3, size=num_individuals),
      'INCTOT': rng.integers(0, 100, size=num_individuals),
      'DISABWRK': rng.integers(0, 3, size=num_individuals),
  })

  return {'household': household_df, 'individual': individual_df}


class CaliforniaCensusSynthesisIntegrationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    domain_path = _find_domain_path()
    self.table_domains, self.foreign_keys = rel_domain.from_yaml_file(
        str(domain_path)
    )

  def test_california_census_pipeline_e2e_mst(self):
    rng = np.random.default_rng(12345)
    data = _generate_mock_california_data(num_households=50, rng=rng)

    config = rel_synth.MultiTableConfig(
        foreign_keys=self.foreign_keys,
        discrete_mechanism=dpsynth.MSTConfig(pgm_iters=10),
        num_permutation_slots=1,
        exploration_strategy='empty_token',
        numerical_bins=2,
    )
    calibrated_mechanism = config.calibrate(
        self.table_domains, epsilon=3.2, delta=1e-6
    )

    # Synthesize
    result = calibrated_mechanism(rng=rng, data=data)

    self.assertIsInstance(result, rel_synth.MultiDataGenerationResult)
    self.assertIn('household', result.synthetic_tables)
    self.assertIn('individual', result.synthetic_tables)

    synth_h = result.synthetic_tables['household']
    synth_i = result.synthetic_tables['individual']

    # Non-empty tables
    self.assertNotEmpty(synth_h)
    self.assertNotEmpty(synth_i)

    # Verify column presence
    self.assertCountEqual(
        synth_h.columns,
        [
            'HOUSEHOLD',
            'FARM',
            'OWNERSHP',
            'ACREHOUS',
            'TAXINCL',
            'PROPINSR',
            'COSTELEC',
            'VALUEH',
            'ROOMS',
            'PLUMBING',
            'PUMA',
        ],
    )
    self.assertCountEqual(
        synth_i.columns,
        [
            'HOUSEHOLD',
            'RELATE',
            'SEX',
            'AGE',
            'MARST',
            'RACE',
            'CITIZEN',
            'SPEAKENG',
            'SCHOOL',
            'EDUC',
            'GRADEATT',
            'SCHLTYPE',
            'EMPSTAT',
            'CLASSWKR',
            'INCTOT',
            'DISABWRK',
        ],
    )

    # Verify relational integrity: No orphaned children
    h_pks = set(synth_h['HOUSEHOLD'])
    i_fks = set(synth_i['HOUSEHOLD'])
    self.assertTrue(
        i_fks.issubset(h_pks), 'Found orphaned individual foreign keys!'
    )

    # Verify group size capacity: max children per parent <= 8
    counts_per_h = synth_i['HOUSEHOLD'].value_counts()
    self.assertTrue(
        (counts_per_h <= 8).all(),
        f'Max children exceeded: {counts_per_h.max()}',
    )


if __name__ == '__main__':
  absltest.main()
