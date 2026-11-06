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

"""Unit tests for California Census relational domain configuration."""

from __future__ import annotations
from absl.testing import absltest
from dpsynth.core import domain
from dpsynth.relational import domain as rel_domain
from etils import epath


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


class CaliforniaCensusDomainTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.domain_path = _find_domain_path()

  def test_load_california_census_domain(self):
    table_domains, foreign_keys = rel_domain.from_yaml_file(
        str(self.domain_path)
    )

    # Validate table set
    self.assertCountEqual(
        list(table_domains.keys()), ['household', 'individual']
    )

    # Validate household attributes
    household_schema = table_domains['household']
    self.assertLen(household_schema, 10)
    self.assertIsInstance(household_schema['FARM'], domain.CategoricalAttribute)
    self.assertEqual(household_schema['FARM'].size, 2)
    self.assertIsInstance(
        household_schema['PROPINSR'], domain.NumericalAttribute
    )
    self.assertEqual(household_schema['PROPINSR'].min_value, 0.0)
    self.assertEqual(household_schema['PROPINSR'].max_value, 59.0)

    # Validate individual attributes
    individual_schema = table_domains['individual']
    self.assertLen(individual_schema, 15)
    self.assertIsInstance(
        individual_schema['RELATE'], domain.CategoricalAttribute
    )
    self.assertEqual(individual_schema['RELATE'].size, 8)
    self.assertIsInstance(individual_schema['AGE'], domain.NumericalAttribute)
    self.assertEqual(individual_schema['AGE'].min_value, 0.0)
    self.assertEqual(individual_schema['AGE'].max_value, 85.0)

    # Validate foreign keys
    self.assertLen(foreign_keys, 1)
    fk = foreign_keys[0]
    self.assertEqual(fk.parent_table, 'household')
    self.assertEqual(fk.parent_primary_key, 'HOUSEHOLD')
    self.assertEqual(fk.child_table, 'individual')
    self.assertEqual(fk.child_foreign_key, 'HOUSEHOLD')
    self.assertEqual(fk.max_children_per_parent, 8)

    # Validate topological hierarchy
    hierarchy = rel_domain.topological_sort_hierarchy(
        list(table_domains.keys()), foreign_keys
    )
    self.assertEqual(
        hierarchy,
        [
            (0, 'household', None),
            (1, 'individual', fk),
        ],
    )


if __name__ == '__main__':
  absltest.main()
