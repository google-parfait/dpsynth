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

"""Unit tests for dpsynth.relational.domain."""

import textwrap

from absl.testing import absltest
import dpsynth
from dpsynth.relational import domain


class DomainTest(absltest.TestCase):

  def test_foreign_key_relation_initialization(self):
    fk = dpsynth.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='household_id',
        child_table='persons',
        child_foreign_key='household_id',
        max_children_per_parent=5,
    )
    self.assertEqual(fk.parent_table, 'households')
    self.assertEqual(fk.parent_primary_key, 'household_id')
    self.assertEqual(fk.child_table, 'persons')
    self.assertEqual(fk.child_foreign_key, 'household_id')
    self.assertEqual(fk.max_children_per_parent, 5)

  def test_foreign_key_relation_invalid_capacity(self):
    with self.assertRaisesRegex(
        ValueError, 'max_children_per_parent must be >= 1'
    ):
      dpsynth.ForeignKeyRelation(
          parent_table='households',
          parent_primary_key='household_id',
          child_table='persons',
          child_foreign_key='household_id',
          max_children_per_parent=0,
      )

  def test_topological_sort_linear_chain(self):
    tables = ['households', 'persons', 'activities']
    fk1 = dpsynth.ForeignKeyRelation('households', 'hid', 'persons', 'hid', 3)
    fk2 = dpsynth.ForeignKeyRelation('persons', 'pid', 'activities', 'pid', 2)
    order = domain.topological_sort_hierarchy(tables, [fk1, fk2])
    self.assertEqual(
        order,
        [
            (0, 'households', None),
            (1, 'persons', fk1),
            (2, 'activities', fk2),
        ],
    )

  def test_topological_sort_branching(self):
    tables = ['households', 'persons', 'vehicles']
    fk_p = dpsynth.ForeignKeyRelation('households', 'hid', 'persons', 'hid', 3)
    fk_v = dpsynth.ForeignKeyRelation('households', 'hid', 'vehicles', 'hid', 2)
    order = domain.topological_sort_hierarchy(tables, [fk_p, fk_v])
    self.assertEqual(order[0], (0, 'households', None))
    self.assertCountEqual(
        order[1:], [(1, 'persons', fk_p), (1, 'vehicles', fk_v)]
    )

  def test_topological_sort_forest(self):
    tables = ['households', 'persons', 'companies', 'departments']
    fk_h = dpsynth.ForeignKeyRelation('households', 'hid', 'persons', 'hid', 3)
    fk_c = dpsynth.ForeignKeyRelation(
        'companies', 'cid', 'departments', 'cid', 10
    )
    order = domain.topological_sort_hierarchy(tables, [fk_h, fk_c])
    depths = {t: d for d, t, _ in order}
    self.assertEqual(depths['households'], 0)
    self.assertEqual(depths['companies'], 0)
    self.assertEqual(depths['persons'], 1)
    self.assertEqual(depths['departments'], 1)

  def test_topological_sort_single_table_and_empty(self):
    self.assertEqual(
        domain.topological_sort_hierarchy(['households'], []),
        [(0, 'households', None)],
    )
    self.assertEqual(domain.topological_sort_hierarchy([], []), [])

  def test_topological_sort_cycle_raises_error(self):
    tables = ['a', 'b']
    fk1 = dpsynth.ForeignKeyRelation('a', 'id', 'b', 'id', 1)
    fk2 = dpsynth.ForeignKeyRelation('b', 'id', 'a', 'id', 1)
    with self.assertRaisesRegex(ValueError, 'Cycle detected'):
      domain.topological_sort_hierarchy(tables, [fk1, fk2])

  def test_topological_sort_self_cycle_raises_error(self):
    tables = ['a']
    fk = dpsynth.ForeignKeyRelation('a', 'id', 'a', 'id', 1)
    with self.assertRaisesRegex(ValueError, 'Self-referential cycle'):
      domain.topological_sort_hierarchy(tables, [fk])

  def test_topological_sort_multiple_parents_raises_error(self):
    tables = ['p1', 'p2', 'c']
    fk1 = dpsynth.ForeignKeyRelation('p1', 'id', 'c', 'p1_id', 1)
    fk2 = dpsynth.ForeignKeyRelation('p2', 'id', 'c', 'p2_id', 1)
    with self.assertRaisesRegex(ValueError, 'multiple parents'):
      domain.topological_sort_hierarchy(tables, [fk1, fk2])

  def test_topological_sort_unknown_table_raises_error(self):
    tables = ['households']
    fk = dpsynth.ForeignKeyRelation('households', 'hid', 'unknown', 'hid', 1)
    with self.assertRaisesRegex(ValueError, 'unknown table'):
      domain.topological_sort_hierarchy(tables, [fk])

  def test_from_dict_valid_3tier_schema(self):
    config = {
        'tables': {
            'households': {
                'income': {
                    'type': 'NumericalAttribute',
                    'min_value': 0.0,
                    'max_value': 200000.0,
                },
                'region': {
                    'type': 'CategoricalAttribute',
                    'possible_values': ['Urban', 'Rural'],
                },
            },
            'persons': {
                'age': {
                    'type': 'NumericalAttribute',
                    'min_value': 0,
                    'max_value': 100,
                    'dtype': 'int',
                },
                'gender': {
                    'type': 'CategoricalAttribute',
                    'possible_values': ['M', 'F'],
                },
            },
        },
        'foreign_keys': [{
            'parent_table': 'households',
            'parent_primary_key': 'household_id',
            'child_table': 'persons',
            'child_foreign_key': 'household_id',
            'max_children_per_parent': 3,
        }],
    }
    table_domains, fks = domain.from_dict(config)
    self.assertIn('households', table_domains)
    self.assertIn('persons', table_domains)
    self.assertIsInstance(
        table_domains['households']['income'], dpsynth.NumericalAttribute
    )
    self.assertIsInstance(
        table_domains['households']['region'],
        dpsynth.CategoricalAttribute,
    )
    self.assertLen(fks, 1)
    self.assertEqual(fks[0].parent_table, 'households')
    self.assertEqual(fks[0].parent_primary_key, 'household_id')
    self.assertEqual(fks[0].child_table, 'persons')
    self.assertEqual(fks[0].child_foreign_key, 'household_id')
    self.assertEqual(fks[0].max_children_per_parent, 3)

  def test_from_dict_missing_tables_block_raises_error(self):
    with self.assertRaisesRegex(ValueError, "'tables' block missing"):
      domain.from_dict({})

  def test_from_dict_missing_type_field_raises_error(self):
    config = {
        'tables': {
            'households': {'income': {'min_value': 0.0, 'max_value': 200000.0}}
        }
    }
    with self.assertRaisesRegex(ValueError, "missing required 'type' field"):
      domain.from_dict(config)

  def test_from_dict_unknown_type_field_raises_error(self):
    config = {
        'tables': {
            'households': {
                'income': {
                    'type': 'InvalidType',
                    'min_value': 0.0,
                    'max_value': 100.0,
                }
            }
        }
    }
    with self.assertRaisesRegex(
        ValueError, "Unknown attribute type 'InvalidType'"
    ):
      domain.from_dict(config)

  def test_from_dict_invalid_table_schema_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'must be a mapping'):
      domain.from_dict({'tables': {'households': 'invalid'}})

  def test_from_dict_invalid_attribute_spec_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'Invalid attribute specification'):
      domain.from_dict({'tables': {'households': {'income': 123}}})

  def test_from_dict_invalid_foreign_key_spec_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'Invalid foreign key'):
      domain.from_dict({'tables': {'h': {}}, 'foreign_keys': [123]})

  def test_from_yaml_file_roundtrip(self):
    yaml_content = textwrap.dedent("""\
        tables:
          households:
            income:
              type: NumericalAttribute
              min_value: 0.0
              max_value: 100000.0
            region:
              type: CategoricalAttribute
              possible_values: ["Urban", "Rural"]
          persons:
            age:
              type: NumericalAttribute
              min_value: 0
              max_value: 100
              dtype: int
            gender:
              type: CategoricalAttribute
              possible_values: ["M", "F"]
        foreign_keys:
          - parent_table: households
            parent_primary_key: hid
            child_table: persons
            child_foreign_key: hid
            max_children_per_parent: 4
        """)
    tmp_path = self.create_tempfile(content=yaml_content).full_path
    table_domains, fks = domain.from_yaml_file(tmp_path)
    self.assertIn('households', table_domains)
    self.assertIn('persons', table_domains)
    self.assertIsInstance(
        table_domains['persons']['age'], dpsynth.NumericalAttribute
    )
    self.assertIsInstance(
        table_domains['persons']['gender'], dpsynth.CategoricalAttribute
    )
    self.assertLen(fks, 1)
    self.assertEqual(fks[0].parent_table, 'households')
    self.assertEqual(fks[0].parent_primary_key, 'hid')
    self.assertEqual(fks[0].child_table, 'persons')
    self.assertEqual(fks[0].child_foreign_key, 'hid')
    self.assertEqual(fks[0].max_children_per_parent, 4)

  def test_to_dict_and_roundtrip(self):
    table_domains = {
        'households': {
            'income': dpsynth.NumericalAttribute(
                min_value=0.0, max_value=200000.0, dtype='float'
            ),
            'region': dpsynth.CategoricalAttribute(
                possible_values=['Urban', 'Rural']
            ),
        },
        'persons': {
            'age': dpsynth.NumericalAttribute(
                min_value=0, max_value=100, dtype='int'
            ),
            'gender': dpsynth.CategoricalAttribute(possible_values=['M', 'F']),
        },
    }
    fks = [
        dpsynth.ForeignKeyRelation(
            parent_table='households',
            parent_primary_key='hid',
            child_table='persons',
            child_foreign_key='hid',
            max_children_per_parent=3,
        )
    ]
    serialized = domain.to_dict(table_domains, fks)
    self.assertIn('tables', serialized)
    self.assertIn('foreign_keys', serialized)
    self.assertLen(serialized['foreign_keys'], 1)
    self.assertEqual(
        serialized['tables']['households']['income']['type'],
        'NumericalAttribute',
    )

    # Roundtrip verification
    rt_domains, rt_fks = domain.from_dict(serialized)
    self.assertEqual(
        rt_domains['households']['income'].min_value,
        table_domains['households']['income'].min_value,
    )
    self.assertEqual(
        rt_domains['persons']['gender'].possible_values,
        table_domains['persons']['gender'].possible_values,
    )
    self.assertEqual(rt_fks, fks)

  def test_to_yaml_file_and_roundtrip(self):
    table_domains = {
        'households': {
            'income': dpsynth.NumericalAttribute(
                min_value=0.0, max_value=150000.0
            ),
            'region': dpsynth.CategoricalAttribute(
                possible_values=['Urban', 'Suburban']
            ),
        },
        'persons': {
            'age': dpsynth.NumericalAttribute(
                min_value=0, max_value=120, dtype='int'
            ),
        },
    }
    fks = [
        dpsynth.ForeignKeyRelation(
            parent_table='households',
            parent_primary_key='household_id',
            child_table='persons',
            child_foreign_key='household_id',
            max_children_per_parent=5,
        )
    ]
    tmp_path = self.create_tempfile().full_path
    domain.to_yaml_file(table_domains, fks, tmp_path)

    rt_domains, rt_fks = domain.from_yaml_file(tmp_path)
    self.assertIn('households', rt_domains)
    self.assertIn('persons', rt_domains)
    self.assertEqual(rt_domains['households']['income'].max_value, 150000.0)
    self.assertEqual(rt_fks, fks)

  def test_to_dict_without_foreign_keys(self):
    table_domains = {
        'single_table': {
            'col_a': dpsynth.CategoricalAttribute(possible_values=['x', 'y'])
        }
    }
    serialized = domain.to_dict(table_domains)
    self.assertIn('tables', serialized)
    self.assertNotIn('foreign_keys', serialized)
    rt_domains, rt_fks = domain.from_dict(serialized)
    self.assertIn('single_table', rt_domains)
    self.assertEmpty(rt_fks)


if __name__ == '__main__':
  absltest.main()
