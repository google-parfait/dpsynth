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

"""Unit tests for dpsynth.relational.synthesizer."""

import math
import unittest.mock

from absl.testing import absltest
import dp_accounting
from dpsynth import api
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth import relational as rel_pkg
from dpsynth.local_mode import initialization
from dpsynth.relational import domain as rel_domain
from dpsynth.relational import post_processing
from dpsynth.relational import synthesizer
from dpsynth.relational import transformations
import mbi
import numpy as np
import pandas as pd


class SynthesizerTest(absltest.TestCase):

  def test_classes_inherit_correct_api_abstractions(self):
    self.assertTrue(
        issubclass(synthesizer.MultiTableConfig, api.MechanismConfig)
    )
    self.assertTrue(
        issubclass(synthesizer.MultiTableMechanism, api.CalibratedMechanism)
    )

  def test_create_table_initializers_success(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(min_value=0.0, max_value=100.0),
            'region': domain.CategoricalAttribute(possible_values=['U', 'R']),
            'tags': domain.OpenSetCategoricalAttribute(),
        },
        'Person': {
            'age': domain.NumericalAttribute(min_value=0, max_value=100),
            'gender': domain.CategoricalAttribute(possible_values=['M', 'F']),
        },
    }
    inits = synthesizer._create_table_initializers(domains, numerical_bins=16)

    self.assertIn('Household', inits)
    self.assertIn('Person', inits)
    self.assertIsInstance(
        inits['Household']['income'], initialization.NumericalInitializerConfig
    )
    self.assertEqual(inits['Household']['income'].num_partitions, 16)
    self.assertIsInstance(
        inits['Household']['region'],
        initialization.CategoricalInitializerConfig,
    )
    self.assertIsInstance(
        inits['Household']['tags'], initialization.OpenSetInitializerConfig
    )
    self.assertIsInstance(
        inits['Person']['age'], initialization.NumericalInitializerConfig
    )
    self.assertIsInstance(
        inits['Person']['gender'], initialization.CategoricalInitializerConfig
    )

  def test_create_table_initializers_unsupported_type_raises(self):
    domains = {
        'Household': {
            'text': domain.FreeFormTextAttribute(),
        }
    }
    with self.assertRaisesRegex(ValueError, 'Unsupported attribute type'):
      synthesizer._create_table_initializers(domains, numerical_bins=16)

  def test_compute_table_col_deltas_with_open_set(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(min_value=0.0, max_value=100.0),
            'tags': domain.OpenSetCategoricalAttribute(),
        },
        'Person': {
            'hobbies': domain.OpenSetCategoricalAttribute(),
            'gender': domain.CategoricalAttribute(possible_values=['M', 'F']),
        },
    }
    deltas = synthesizer._compute_table_col_deltas(domains, delta=1e-4)
    # 2 open-set columns -> each gets 5e-5.
    self.assertAlmostEqual(deltas['Household']['tags'], 5e-5)
    self.assertEqual(deltas['Household']['income'], 0.0)
    self.assertAlmostEqual(deltas['Person']['hobbies'], 5e-5)
    self.assertEqual(deltas['Person']['gender'], 0.0)

  def test_compute_table_col_deltas_no_open_set(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(min_value=0.0, max_value=100.0),
            'region': domain.CategoricalAttribute(possible_values=['U', 'R']),
        },
    }
    deltas = synthesizer._compute_table_col_deltas(domains, delta=0.0)
    self.assertEqual(deltas['Household']['income'], 0.0)
    self.assertEqual(deltas['Household']['region'], 0.0)

  def test_compute_table_col_deltas_missing_delta_raises(self):
    domains = {
        'Household': {
            'tags': domain.OpenSetCategoricalAttribute(),
        },
    }
    with self.assertRaisesRegex(ValueError, 'delta must be positive'):
      synthesizer._compute_table_col_deltas(domains, delta=0.0)

  def test_dp_event_composition(self):
    # Setup calibrated initializers for 2 tables.
    household_inits = {
        'income': (
            initialization.NumericalInitializerConfig(
                num_partitions=16,
            ).configure(
                domain.NumericalAttribute(min_value=0, max_value=100),
                budget=0.01,
            )
        ),
        'region': (
            initialization.CategoricalInitializerConfig().configure(
                domain.CategoricalAttribute(possible_values=['U', 'R']),
                budget=0.01,
            )
        ),
    }
    person_inits = {
        'age': (
            initialization.NumericalInitializerConfig(
                num_partitions=16,
            ).configure(
                domain.NumericalAttribute(min_value=0, max_value=100),
                budget=0.01,
            )
        ),
    }
    calibrated_initializers = {
        'Household': household_inits,
        'Person': person_inits,
    }

    # Setup calibrated discrete mechanism for link Household -> Person.
    calibrated_discrete = discrete_mechanisms.AIMConfig().configure(budget=0.1)
    calibrated_discrete_mechanisms = {
        'Household->Person': calibrated_discrete,
    }

    mech = synthesizer.MultiTableMechanism(
        domains={},
        foreign_keys=(),
        calibrated_discrete_mechanisms=calibrated_discrete_mechanisms,
        calibrated_initializers=calibrated_initializers,
        total_count_sigma=5.0,
        link_sensitivities={'Household->Person': 1},
    )

    event = mech.dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    # Expected: 3 column initializers
    # + 1 root Gaussian count + 1 discrete mechanism = 5 events.
    self.assertLen(event.events, 5)
    # Verify root total count event is Gaussian with noise_multiplier=5.0.
    gaussian_events = [
        e
        for e in event.events
        if isinstance(e, dp_accounting.GaussianDpEvent)
        and e.noise_multiplier == 5.0
    ]
    self.assertLen(gaussian_events, 1)

  def test_compute_link_sensitivities_3_tier(self):
    tables = ['Household', 'Person', 'Activity']
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,  # s_1 = 3
        ),
        rel_domain.ForeignKeyRelation(
            parent_table='Person',
            parent_primary_key='pid',
            child_table='Activity',
            child_foreign_key='pid',
            max_children_per_parent=2,  # s_2 = 2
        ),
    ]
    hierarchy = rel_domain.topological_sort_hierarchy(tables, foreign_keys)

    sensitivities = synthesizer._compute_link_sensitivities(hierarchy)
    # Link 1 (Household -> Person): Delta_1 = 1
    self.assertEqual(sensitivities['Household->Person'], 1)
    # Link 2 (Person -> Activity): Delta_2 = s_1 = 3
    self.assertEqual(sensitivities['Person->Activity'], 3)

  def test_compute_link_sensitivities_branching(self):
    tables = ['Household', 'Person', 'Vehicle']
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Vehicle',
            child_foreign_key='hid',
            max_children_per_parent=2,
        ),
    ]
    hierarchy = rel_domain.topological_sort_hierarchy(tables, foreign_keys)

    sensitivities = synthesizer._compute_link_sensitivities(hierarchy)
    self.assertEqual(sensitivities['Household->Person'], 1)
    self.assertEqual(sensitivities['Household->Vehicle'], 1)

  def test_compute_link_sensitivities_no_links(self):
    tables = ['Household', 'Logs']
    hierarchy = rel_domain.topological_sort_hierarchy(tables, foreign_keys=())
    sensitivities = synthesizer._compute_link_sensitivities(hierarchy)
    self.assertEmpty(sensitivities)

  def test_configure_single_table_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'requires at least two tables in schema'
    ):
      config = synthesizer.MultiTableConfig(foreign_keys=())
      config.configure(
          schema={'Household': {'income': domain.NumericalAttribute(0, 100)}},
          budget=0.5,
      )

  def test_configure_empty_foreign_keys_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'requires at least one foreign key relationship'
    ):
      config = synthesizer.MultiTableConfig(foreign_keys=())
      config.configure(
          schema={
              'Household': {'income': domain.NumericalAttribute(0, 100)},
              'Person': {'age': domain.NumericalAttribute(0, 100)},
          },
          budget=0.5,
      )

  def test_configure_end_to_end_3_tier(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
            'region': domain.CategoricalAttribute(['U', 'R']),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
            'gender': domain.CategoricalAttribute(['M', 'F']),
        },
        'Activity': {
            'amount': domain.NumericalAttribute(0, 50),
        },
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
        rel_domain.ForeignKeyRelation(
            parent_table='Person',
            parent_primary_key='pid',
            child_table='Activity',
            child_foreign_key='pid',
            max_children_per_parent=2,
        ),
    ]
    config = synthesizer.MultiTableConfig(
        foreign_keys=foreign_keys,
        init_budget_fraction=0.1,
    )

    # 5 total cols + 1 root count = 6 initializers.
    # init_rho = 0.1 * 0.6 = 0.06 => per_col_rho = 0.01.
    # total_count_sigma = sqrt(0.5 / 0.01) = sqrt(50).
    # discrete_rho = 0.6 - 0.06 = 0.54 => per_link_rho = 0.27 across 2 links.
    mech = config.configure(domains, budget=0.6)

    self.assertIsInstance(mech, synthesizer.MultiTableMechanism)
    self.assertAlmostEqual(mech.total_count_sigma, math.sqrt(50.0))
    self.assertLen(mech.calibrated_initializers, 3)
    self.assertLen(mech.calibrated_discrete_mechanisms, 2)
    self.assertIn('Household->Person', mech.calibrated_discrete_mechanisms)
    self.assertIn('Person->Activity', mech.calibrated_discrete_mechanisms)

    # Verify composed DpEvent has
    # 5 col inits + 1 root Gaussian + 2 discrete links = 8 events.
    event = mech.dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    self.assertLen(event.events, 8)
    expected_link_event = config.discrete_mechanism.configure(
        budget=0.27
    ).dp_event
    self.assertEqual(event.events[-2], expected_link_event)
    self.assertEqual(event.events[-1], expected_link_event)

  def test_calibrate_end_to_end(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
            'region': domain.CategoricalAttribute(['U', 'R']),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
        },
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
    ]
    config = synthesizer.MultiTableConfig(
        foreign_keys=foreign_keys,
    )
    # Calibrate solves for optimal zcdp_rho using PLD / RDP accountant.
    mech = config.calibrate(domains, epsilon=1.0, delta=1e-5)
    self.assertIsInstance(mech, synthesizer.MultiTableMechanism)
    self.assertGreater(mech.total_count_sigma, 0.0)
    self.assertLen(mech.calibrated_discrete_mechanisms, 1)

  def test_configure_hyperparameter_validations(self):
    domains = {
        'Household': {'income': domain.NumericalAttribute(0, 100)},
        'Person': {'age': domain.NumericalAttribute(0, 100)},
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
    ]

    with self.subTest('negative_budget'):
      config = synthesizer.MultiTableConfig(foreign_keys=foreign_keys)
      with self.assertRaisesRegex(ValueError, 'budget must be positive'):
        config.configure(domains, budget=-0.1)

    with self.subTest('invalid_init_budget_fraction_above_one'):
      with self.assertRaisesRegex(
          ValueError, 'init_budget_fraction must be strictly in'
      ):
        synthesizer.MultiTableConfig(
            foreign_keys=foreign_keys,
            init_budget_fraction=1.5,
        )

    with self.subTest('invalid_init_budget_fraction_zero'):
      with self.assertRaisesRegex(
          ValueError, 'init_budget_fraction must be strictly in'
      ):
        synthesizer.MultiTableConfig(
            foreign_keys=foreign_keys,
            init_budget_fraction=0.0,
        )

    with self.subTest('invalid_numerical_bins'):
      with self.assertRaisesRegex(ValueError, 'numerical_bins must be >= 1'):
        synthesizer.MultiTableConfig(
            foreign_keys=foreign_keys,
            numerical_bins=0,
        )

    with self.subTest('invalid_num_permutation_slots'):
      with self.assertRaisesRegex(
          ValueError, 'num_permutation_slots must be >= 1'
      ):
        synthesizer.MultiTableConfig(
            foreign_keys=foreign_keys,
            num_permutation_slots=0,
        )

    with self.subTest('invalid_exploration_strategy'):
      with self.assertRaisesRegex(
          ValueError, 'Unsupported exploration_strategy'
      ):
        synthesizer.MultiTableConfig(
            foreign_keys=foreign_keys,
            exploration_strategy='unsupported_strategy',
        )

    with self.subTest('invalid_discrete_mechanism'):
      with self.assertRaisesRegex(
          ValueError,
          'discrete_mechanism must be an instance of MechanismConfig',
      ):
        synthesizer.MultiTableConfig(
            foreign_keys=foreign_keys,
            discrete_mechanism='not_a_config',  # pyrefly: ignore[bad-argument-type]
        )

    with self.subTest('dot_in_table_name'):
      with self.assertRaisesRegex(ValueError, "must not contain '.'"):
        config = synthesizer.MultiTableConfig(
            foreign_keys=[
                rel_domain.ForeignKeyRelation(
                    parent_table='House.hold',
                    parent_primary_key='hid',
                    child_table='Person',
                    child_foreign_key='hid',
                    max_children_per_parent=2,
                )
            ],
        )
        config.configure(
            schema={
                'House.hold': {'income': domain.NumericalAttribute(0, 100)},
                'Person': {'age': domain.NumericalAttribute(0, 100)},
            },
            budget=0.5,
        )

    with self.subTest('dot_in_column_name'):
      with self.assertRaisesRegex(ValueError, "must not contain '.'"):
        config = synthesizer.MultiTableConfig(foreign_keys=foreign_keys)
        config.configure(
            schema={
                'Household': {'inc.ome': domain.NumericalAttribute(0, 100)},
                'Person': {'age': domain.NumericalAttribute(0, 100)},
            },
            budget=0.5,
        )

    with self.subTest('reserved_column_name_group_size'):
      with self.assertRaisesRegex(
          ValueError, 'reserved for relational exploration'
      ):
        config = synthesizer.MultiTableConfig(foreign_keys=foreign_keys)
        config.configure(
            schema={
                'Household': {'group_size': domain.NumericalAttribute(0, 100)},
                'Person': {'age': domain.NumericalAttribute(0, 100)},
            },
            budget=0.5,
        )

    with self.subTest('reserved_column_name_slot_prefix'):
      with self.assertRaisesRegex(ValueError, 'reserved for permutation slots'):
        config = synthesizer.MultiTableConfig(foreign_keys=foreign_keys)
        config.configure(
            schema={
                'Household': {'slot_1': domain.NumericalAttribute(0, 100)},
                'Person': {'age': domain.NumericalAttribute(0, 100)},
            },
            budget=0.5,
        )

    with self.subTest('empty_table_schema'):
      with self.assertRaisesRegex(ValueError, 'schema cannot be empty'):
        config = synthesizer.MultiTableConfig(foreign_keys=foreign_keys)
        config.configure(
            schema={
                'Household': {},
                'Person': {'age': domain.NumericalAttribute(0, 100)},
            },
            budget=0.5,
        )

    with self.subTest('unsupported_attribute_type'):
      with self.assertRaisesRegex(ValueError, 'unsupported attribute type'):
        config = synthesizer.MultiTableConfig(foreign_keys=foreign_keys)
        config.configure(
            schema={
                'Household': {'text': domain.FreeFormTextAttribute()},
                'Person': {'age': domain.NumericalAttribute(0, 100)},
            },
            budget=0.5,
        )

    with self.subTest('multi_root_forest_raises'):
      with self.assertRaisesRegex(ValueError, 'expects a single root table'):
        config = synthesizer.MultiTableConfig(foreign_keys=foreign_keys)
        config.configure(
            schema={
                'Household': {'income': domain.NumericalAttribute(0, 100)},
                'Person': {'age': domain.NumericalAttribute(0, 100)},
                'Unlinked': {'type': domain.CategoricalAttribute(['A', 'B'])},
            },
            budget=0.5,
        )

    with self.subTest('pk_in_domain_schema_raises'):
      with self.assertRaisesRegex(ValueError, 'must not be in schema'):
        config = synthesizer.MultiTableConfig(foreign_keys=foreign_keys)
        config.configure(
            schema={
                'Household': {
                    'income': domain.NumericalAttribute(0, 100),
                    'hid': domain.CategoricalAttribute(['H1', 'H2']),
                },
                'Person': {'age': domain.NumericalAttribute(0, 100)},
            },
            budget=0.5,
        )

    with self.subTest('fk_in_domain_schema_raises'):
      with self.assertRaisesRegex(ValueError, 'must not be in schema'):
        config = synthesizer.MultiTableConfig(foreign_keys=foreign_keys)
        config.configure(
            schema={
                'Household': {'income': domain.NumericalAttribute(0, 100)},
                'Person': {
                    'age': domain.NumericalAttribute(0, 100),
                    'hid': domain.CategoricalAttribute(['H1', 'H2']),
                },
            },
            budget=0.5,
        )

    with self.subTest('custom_initializers_mismatched_tables'):
      with self.assertRaisesRegex(ValueError, 'do not match schema tables'):
        config = synthesizer.MultiTableConfig(
            foreign_keys=foreign_keys,
            initializers={'Household': {}},
        )
        config.configure(schema=domains, budget=0.5)

    with self.subTest('custom_initializers_mismatched_columns'):
      mock_cfg = unittest.mock.MagicMock(spec=api.MechanismConfig)
      with self.assertRaisesRegex(ValueError, 'do not match schema columns'):
        config = synthesizer.MultiTableConfig(
            foreign_keys=foreign_keys,
            initializers={
                'Household': {'wrong_col': mock_cfg},
                'Person': {'age': mock_cfg},
            },
        )
        config.configure(schema=domains, budget=0.5)

  def test_validate_input_table_columns_success(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
            'region': domain.CategoricalAttribute(['U', 'R']),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
        },
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
    ]
    table_columns = {
        'Household': ['hid', 'income', 'region'],
        'Person': ['pid', 'hid', 'age'],
    }
    # Should complete without error.
    synthesizer._validate_input_table_columns(
        domains, foreign_keys, table_columns
    )

  def test_validate_input_table_columns_raises(self):
    domains = {
        'Household': {'income': domain.NumericalAttribute(0, 100)},
        'Person': {'age': domain.NumericalAttribute(0, 100)},
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
    ]

    with self.subTest('missing_table'):
      table_columns = {'Household': ['hid', 'income']}
      with self.assertRaisesRegex(ValueError, 'Table .* not found'):
        synthesizer._validate_input_table_columns(
            domains, foreign_keys, table_columns
        )

    with self.subTest('missing_schema_column'):
      table_columns = {
          'Household': ['hid'],  # missing 'income'
          'Person': ['hid', 'age'],
      }
      with self.assertRaisesRegex(ValueError, 'Column .* not found in table'):
        synthesizer._validate_input_table_columns(
            domains, foreign_keys, table_columns
        )

    with self.subTest('missing_parent_primary_key'):
      table_columns = {
          'Household': ['income'],  # missing 'hid'
          'Person': ['hid', 'age'],
      }
      with self.assertRaisesRegex(
          ValueError, 'Parent primary key column .* not found'
      ):
        synthesizer._validate_input_table_columns(
            domains, foreign_keys, table_columns
        )

    with self.subTest('missing_child_foreign_key'):
      table_columns = {
          'Household': ['hid', 'income'],
          'Person': ['age'],  # missing 'hid'
      }
      with self.assertRaisesRegex(
          ValueError, 'Child foreign key column .* not found'
      ):
        synthesizer._validate_input_table_columns(
            domains, foreign_keys, table_columns
        )

  def test_preprocess_weighted_tables(self):
    tables = {
        'Household': pd.DataFrame({
            'hid': ['H1', 'H2'],
            'income': [50.0, 75.0],
        }),
        'Person': pd.DataFrame({
            'pid': ['P1', 'P2', 'P3', 'P4', 'P5'],
            'hid': ['H1', 'H1', 'H1', 'H2', 'Orphan'],
            'age': [10, 20, 30, 40, 50],
        }),
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=2,  # H1 has 3 persons -> 1 truncated
        ),
    ]
    hierarchy = rel_domain.topological_sort_hierarchy(
        list(tables.keys()), foreign_keys
    )
    rng = np.random.default_rng(42)

    filtered_tables, filtered_weights = synthesizer._preprocess_weighted_tables(
        tables, hierarchy, rng=rng
    )

    # Root table retains all 2 rows with weight 1.0 each.
    self.assertLen(filtered_tables['Household'], 2)
    np.testing.assert_allclose(filtered_weights['Household'], [1.0, 1.0])

    # Person table had 5 rows:
    # 2 kept from H1 (weight 0.5 each), 1 kept from H2 (weight 1.0),
    # 1 truncated from H1 (w=0.0 filtered out), 1 orphan (w=0.0 filtered out).
    self.assertLen(filtered_tables['Person'], 3)
    np.testing.assert_allclose(filtered_weights['Person'], [0.5, 0.5, 1.0])
    self.assertAlmostEqual(filtered_weights['Person'].sum(), 2.0)

  def test_measure_root_total_count(self):
    rng = np.random.default_rng(123)
    # Zero noise test
    total, measurement = synthesizer._measure_root_total_count(
        rng,
        root_record_count=100,
        total_count_sigma=0.0,
    )
    self.assertEqual(total, 100.0)
    self.assertEqual(measurement.clique, ())
    self.assertEqual(measurement.stddev, 0.0)
    np.testing.assert_allclose(measurement.noisy_measurement, [100.0])

    # Noisy test
    total_noisy, measurement_noisy = synthesizer._measure_root_total_count(
        rng,
        root_record_count=100,
        total_count_sigma=5.0,
    )
    self.assertGreaterEqual(total_noisy, 1.0)
    self.assertEqual(measurement_noisy.clique, ())
    self.assertEqual(measurement_noisy.stddev, 5.0)

    # Negative noisy value clips to 1.0
    total_clipped, measurement_clipped = synthesizer._measure_root_total_count(
        rng,
        root_record_count=0,
        total_count_sigma=0.0,
    )
    self.assertEqual(total_clipped, 1.0)
    np.testing.assert_allclose(measurement_clipped.noisy_measurement, [1.0])

  def test_run_single_col_initializer(self):
    rng = np.random.default_rng(42)

    # 1. Numerical initializer on weighted data
    num_init = initialization.NumericalInitializerConfig(
        num_partitions=8,
    ).configure(
        domain.NumericalAttribute(min_value=0.0, max_value=100.0),
        budget=0.5,
    )
    data_num = np.array([10.0, 20.0, 30.0, 80.0, 90.0])
    weights_num = np.array([0.5, 0.5, 1.0, 0.5, 0.5])
    res_num = synthesizer._run_single_col_initializer(
        num_init, rng, data_num, weights_num
    )
    self.assertIsInstance(res_num, initialization.NumericalMeasurement)
    self.assertIsNotNone(res_num.bin_edges)
    self.assertIsNotNone(res_num.categorical_attribute)
    self.assertIsNotNone(res_num.noisy_counts)

    # 2. Categorical initializer on weighted data
    cat_init = initialization.CategoricalInitializerConfig().configure(
        domain.CategoricalAttribute(possible_values=['M', 'F']),
        budget=0.5,
    )
    data_cat = np.array(['M', 'M', 'F', 'F'])
    weights_cat = np.array([0.5, 0.5, 1.0, 1.0])
    res_cat = synthesizer._run_single_col_initializer(
        cat_init, rng, data_cat, weights_cat
    )
    self.assertIsInstance(res_cat, initialization.CategoricalMeasurement)
    self.assertEqual(res_cat.categorical_attribute.size, 2)
    self.assertIsNotNone(res_cat.noisy_counts)

    # 3. Open-set initializer on weighted strings (including mixed-type data)
    open_init = initialization.OpenSetInitializerConfig(
        min_count=1,
    ).configure(
        domain.OpenSetCategoricalAttribute(),
        budget=0.5,
        delta=1e-3,
    )
    data_open = np.array(['sport', 123, 'music'] * 50, dtype=object)
    weights_open = np.ones(len(data_open))
    res_open = synthesizer._run_single_col_initializer(
        open_init, rng, data_open, weights_open
    )
    self.assertIsInstance(res_open, initialization.OpenSetMeasurement)
    self.assertIn('sport', res_open.categorical_attribute.possible_values)
    self.assertIn('123', res_open.categorical_attribute.possible_values)

    # 4. Unsupported initializer raises
    with self.assertRaisesRegex(ValueError, 'Unsupported initializer type'):
      synthesizer._run_single_col_initializer(
          object(), rng, data_num, weights_num  # pyrefly: ignore[bad-argument-type]
      )

  def test_run_table_initializers(self):
    rng = np.random.default_rng(42)
    calibrated_inits = {
        'Household': {
            'income': (
                initialization.NumericalInitializerConfig(
                    num_partitions=8,
                ).configure(
                    domain.NumericalAttribute(min_value=0.0, max_value=100.0),
                    budget=0.1,
                )
            ),
            'region': (
                initialization.CategoricalInitializerConfig().configure(
                    domain.CategoricalAttribute(possible_values=['U', 'R']),
                    budget=0.1,
                )
            ),
        },
        'Person': {
            'age': (
                initialization.NumericalInitializerConfig(
                    num_partitions=8,
                ).configure(
                    domain.NumericalAttribute(min_value=0, max_value=100),
                    budget=0.1,
                )
            ),
        },
    }
    tables = {
        'Household': pd.DataFrame({
            'income': [20.0, 80.0],
            'region': ['U', 'R'],
        }),
        'Person': pd.DataFrame({
            'age': [15, 30, 45],
        }),
    }
    weights = {
        'Household': np.array([1.0, 1.0]),
        'Person': np.array([0.5, 0.5, 1.0]),
    }

    results = synthesizer._run_table_initializers(
        calibrated_inits,
        rng=rng,
        tables=tables,
        weights=weights,
    )

    self.assertIn('Household', results)
    self.assertIn('Person', results)
    self.assertIn('income', results['Household'])
    self.assertIn('region', results['Household'])
    self.assertIn('age', results['Person'])

    self.assertIsInstance(
        results['Household']['income'], initialization.ColumnMeasurement
    )
    self.assertIsInstance(
        results['Household']['region'], initialization.ColumnMeasurement
    )
    self.assertIsInstance(
        results['Person']['age'], initialization.ColumnMeasurement
    )

  def test_encode_and_compress_tables(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
            'region': domain.CategoricalAttribute(['U', 'R']),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
        },
    }
    table_measurements = {
        'Household': {
            'income': initialization.NumericalMeasurement(
                categorical_attribute=domain.CategoricalAttribute([
                    '0',
                    '1',
                    '2',
                    '3',
                ]),
                bin_edges=np.array([25.0, 50.0, 75.0]),
                noisy_counts=np.array([5.0, 5.0, 5.0, 5.0]),
                stddev=1.0,
            ),
            'region': initialization.CategoricalMeasurement(
                categorical_attribute=domain.CategoricalAttribute(['U', 'R']),
                noisy_counts=np.array([10.0, 10.0]),
                stddev=1.0,
            ),
        },
        'Person': {
            'age': initialization.NumericalMeasurement(
                categorical_attribute=domain.CategoricalAttribute(['0', '1']),
                bin_edges=np.array([50.0]),
                noisy_counts=np.array([10.0, 10.0]),
                stddev=1.0,
            ),
        },
    }
    tables = {
        'Household': pd.DataFrame({
            'income': [20.0, 80.0],
            'region': ['U', 'R'],
        }),
        'Person': pd.DataFrame({
            'age': [30, 60],
        }),
    }
    weights = {
        'Household': np.array([1.0, 1.0]),
        'Person': np.array([0.5, 0.5]),
    }

    codecs, datasets, mappings, one_ways = (
        synthesizer._encode_and_compress_tables(
            domains,
            table_measurements,
            tables,
            weights,
            compress_columns=True,
        )
    )

    self.assertIn('Household', codecs)
    self.assertIn('Person', codecs)
    self.assertIn('Household', datasets)
    self.assertIn('Person', datasets)
    self.assertIn('Household', mappings)
    self.assertIn('Person', mappings)

    self.assertIsInstance(datasets['Household'], mbi.Dataset)
    self.assertIsInstance(datasets['Person'], mbi.Dataset)
    np.testing.assert_allclose(datasets['Household'].weights, [1.0, 1.0])
    np.testing.assert_allclose(datasets['Person'].weights, [0.5, 0.5])

    self.assertIn('Household', one_ways)
    self.assertIn('Person', one_ways)
    self.assertNotEmpty(one_ways['Household'])
    self.assertNotEmpty(one_ways['Person'])

  def test_run_table_preprocessing_end_to_end(self):
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
            'region': domain.CategoricalAttribute(['U', 'R']),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
            'gender': domain.CategoricalAttribute(['M', 'F']),
        },
    }
    foreign_keys = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=3,
        ),
    ]
    config = synthesizer.MultiTableConfig(
        foreign_keys=foreign_keys,
        init_budget_fraction=0.2,
    )
    mech = config.configure(domains, budget=0.5)
    rng = np.random.default_rng(42)

    data = {
        'Household': pd.DataFrame({
            'hid': ['H1', 'H2'],
            'income': [30.0, 80.0],
            'region': ['U', 'R'],
        }),
        'Person': pd.DataFrame({
            'pid': ['P1', 'P2', 'P3', 'P4'],
            'hid': ['H1', 'H1', 'H2', 'Orphan'],
            'age': [10, 20, 40, 90],
            'gender': ['M', 'F', 'F', 'M'],
        }),
    }

    preprocessed = synthesizer._run_table_preprocessing(
        mech, rng=rng, data=data
    )

    self.assertIsInstance(preprocessed, synthesizer.PreprocessedTables)
    self.assertIn('Household', preprocessed.compressed_datasets)
    self.assertIn('Person', preprocessed.compressed_datasets)

    # Active records: 2 households, 3 valid persons (1 orphan stripped).
    self.assertEqual(preprocessed.compressed_datasets['Household'].records, 2)
    self.assertEqual(preprocessed.compressed_datasets['Person'].records, 3)

    # Verify keys are captured in table_keys.
    self.assertIn('hid', preprocessed.table_keys['Household'])
    self.assertIn('hid', preprocessed.table_keys['Person'])
    self.assertIn('pid', preprocessed.table_keys['Person'])
    self.assertLen(preprocessed.table_keys['Household']['hid'], 2)
    self.assertLen(preprocessed.table_keys['Person']['hid'], 3)

    # Verify root total count and measurement.
    self.assertGreaterEqual(preprocessed.noisy_root_total, 1.0)
    self.assertEqual(preprocessed.root_total_measurement.clique, ())

    # Verify codecs and one-ways.
    self.assertIn('Household', preprocessed.column_codecs)
    self.assertIn('Person', preprocessed.column_codecs)
    self.assertIn('Household', preprocessed.one_way_measurements)
    self.assertIn('Person', preprocessed.one_way_measurements)

  def test_synthesized_link_result_dataclass(self):
    parent_dom = mbi.Domain(('income',), (2,))
    child_dom = mbi.Domain(('amt',), (2,))
    parent_ds = mbi.Dataset({'income': np.array([0])}, parent_dom)
    child_ds = mbi.Dataset({'amt': np.array([1])}, child_dom)
    parent_idx = np.array([0], dtype=np.int64)

    link_res = synthesizer.SynthesizedLinkResult(
        unstacked_child_dataset=child_ds,
        parent_row_indices=parent_idx,
        synth_parent_dataset=parent_ds,
    )

    self.assertEqual(link_res.unstacked_child_dataset.records, 1)
    self.assertIsNotNone(link_res.synth_parent_dataset)
    self.assertEqual(link_res.synth_parent_dataset.records, 1)
    np.testing.assert_array_equal(link_res.parent_row_indices, [0])
    self.assertIsNone(link_res.discrete_mechanism_result)

  def test_fit_and_sample_wide_link_mrf_zero_rows(self):
    wide_domain = mbi.Domain(('income', 'slot_1.age'), (3, 4))
    res = synthesizer._fit_and_sample_wide_link_mrf(
        wide_domain=wide_domain,
        wide_measurements=(),
        wide_constraints=(),
        num_rows=0,
    )
    self.assertEqual(res.records, 0)
    self.assertEqual(res.domain.attributes, ('income', 'slot_1.age'))

  def test_fit_and_sample_wide_link_mrf_with_constraints(self):
    # Wide domain: 1 parent attribute, 1 slot with 2 attributes
    #   (age [3 bins], gender [2 bins])
    child_domain = mbi.Domain(('age', 'gender'), (3, 2))
    parent_domain = mbi.Domain(('income',), (2,))
    wide_domain = transformations.build_exploration_domain(
        parent_domain=parent_domain,
        child_domain=child_domain,
        max_group_size=1,
        num_permutation_slots=1,
        strategy='empty_token',
    )
    # Constraints: monolithic locking on slot 1
    constraints = post_processing.create_slot_linear_chain_constraints(
        child_domain=child_domain,
        num_permutation_slots=1,
    )
    # Measurements: 1-way for parent
    # & child features + 2-way clique for constraints
    measurements = [
        mbi.LinearMeasurement(np.array([10.0, 10.0]), ('income',), stddev=1.0),
        mbi.LinearMeasurement(
            np.array([5.0, 5.0, 5.0, 5.0]), ('slot_1.age',), stddev=1.0
        ),
        mbi.LinearMeasurement(
            np.array([8.0, 7.0, 5.0]), ('slot_1.gender',), stddev=1.0
        ),
        mbi.LinearMeasurement(
            np.full(12, 2.0), ('slot_1.age', 'slot_1.gender'), stddev=1.0
        ),
    ]
    sampled = synthesizer._fit_and_sample_wide_link_mrf(
        wide_domain=wide_domain,
        wide_measurements=measurements,
        wide_constraints=constraints,
        num_rows=20,
        iters=20,
    )
    self.assertEqual(sampled.records, 20)
    # Check that mixed states are never sampled: age==3 iff gender==2 (<EMPTY>)
    ages = sampled.data['slot_1.age']
    genders = sampled.data['slot_1.gender']
    for a, g in zip(ages, genders):
      if a == 3:
        self.assertEqual(g, 2)
      if g == 2:
        self.assertEqual(a, 3)

  def test_synthesize_relational_link_level_1(self):
    rng = np.random.default_rng(42)
    parent_dom = mbi.Domain(('income',), (2,))
    child_dom = mbi.Domain(('age',), (3,))
    parent_ds = mbi.Dataset({'income': np.array([0, 1] * 10)}, parent_dom)
    child_ds = mbi.Dataset({'age': np.array([0, 1, 2, 0] * 10)}, child_dom)
    parent_pks = [f'H{i}' for i in range(20)]
    child_fks = [f'H{i // 2}' for i in range(40)]
    fk = rel_domain.ForeignKeyRelation(
        parent_table='Household',
        parent_primary_key='hid',
        child_table='Person',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    discrete_mech = discrete_mechanisms.AIMConfig(
        pgm_iters=10, max_rounds=2
    ).configure(budget=0.5)

    res = synthesizer._synthesize_relational_link(
        parent_dataset=parent_ds,
        child_dataset=child_ds,
        parent_primary_keys=parent_pks,
        child_foreign_keys=child_fks,
        fk_relation=fk,
        discrete_mechanism=discrete_mech,
        num_permutation_slots=2,
        strategy='empty_token',
        rng=rng,
        synth_parents=None,
        noisy_root_total=20.0,
    )

    self.assertIsInstance(res, synthesizer.SynthesizedLinkResult)
    self.assertIsNotNone(res.synth_parent_dataset)
    self.assertEqual(res.synth_parent_dataset.records, 20)
    self.assertEqual(res.unstacked_child_dataset.domain.attributes, ('age',))
    self.assertIsNotNone(res.discrete_mechanism_result)
    self.assertGreaterEqual(len(res.parent_row_indices), 0)
    if len(res.parent_row_indices) > 0:
      self.assertTrue(np.all(res.parent_row_indices < 20))

  def test_synthesize_relational_link_downstream_copula(self):
    rng = np.random.default_rng(42)
    parent_dom = mbi.Domain(('age',), (3,))
    child_dom = mbi.Domain(('amt',), (2,))
    parent_ds = mbi.Dataset({'age': np.array([0, 1, 2])}, parent_dom)
    child_ds = mbi.Dataset({'amt': np.array([0, 1, 1])}, child_dom)
    parent_pks = ['P1', 'P2', 'P3']
    child_fks = ['P1', 'P2', 'P2']
    fk = rel_domain.ForeignKeyRelation(
        parent_table='Person',
        parent_primary_key='pid',
        child_table='Activity',
        child_foreign_key='pid',
        max_children_per_parent=2,
    )
    discrete_mech = discrete_mechanisms.AIMConfig(
        pgm_iters=10, max_rounds=2
    ).configure(budget=0.5)

    # Simulated synth_parents from Level 1
    synth_parents = mbi.Dataset({'age': np.array([2, 0, 1])}, parent_dom)

    res = synthesizer._synthesize_relational_link(
        parent_dataset=parent_ds,
        child_dataset=child_ds,
        parent_primary_keys=parent_pks,
        child_foreign_keys=child_fks,
        fk_relation=fk,
        discrete_mechanism=discrete_mech,
        num_permutation_slots=2,
        strategy='empty_token',
        rng=rng,
        synth_parents=synth_parents,
    )

    self.assertIsInstance(res, synthesizer.SynthesizedLinkResult)
    self.assertIsNone(res.synth_parent_dataset)
    self.assertEqual(res.unstacked_child_dataset.domain.attributes, ('amt',))
    self.assertIsNotNone(res.discrete_mechanism_result)
    self.assertTrue(np.all(res.parent_row_indices < 3))

  def test_synthesize_relational_hierarchy_3_tier(self):
    rng = np.random.default_rng(42)
    data = {
        'Household': pd.DataFrame(
            {'hid': ['H1', 'H2'], 'income': [30.0, 80.0]}
        ),
        'Person': pd.DataFrame({
            'pid': ['P1', 'P2', 'P3'],
            'hid': ['H1', 'H1', 'H2'],
            'age': [25, 30, 45],
        }),
        'Activity': pd.DataFrame({
            'aid': ['A1', 'A2', 'A3', 'A4'],
            'pid': ['P1', 'P2', 'P2', 'P3'],
            'amt': [100.0, 200.0, 150.0, 300.0],
        }),
    }
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
        },
        'Activity': {
            'amt': domain.NumericalAttribute(0, 500),
        },
    }
    fks = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=2,
        ),
        rel_domain.ForeignKeyRelation(
            parent_table='Person',
            parent_primary_key='pid',
            child_table='Activity',
            child_foreign_key='pid',
            max_children_per_parent=2,
        ),
    ]
    hierarchy = rel_domain.topological_sort_hierarchy(
        tables=list(domains.keys()), foreign_keys=fks
    )
    cfg = synthesizer.MultiTableConfig(
        foreign_keys=fks,
        discrete_mechanism=discrete_mechanisms.AIMConfig(
            pgm_iters=10, max_rounds=2
        ),
        num_permutation_slots=2,
    )
    mech = cfg.configure(domains, budget=0.5)

    preprocessed = synthesizer._run_table_preprocessing(
        mechanism=mech,
        rng=rng,
        data=data,
    )

    synth_datasets, parent_mappings, discrete_results = (
        synthesizer._synthesize_relational_hierarchy(
            mechanism=mech,
            preprocessed=preprocessed,
            hierarchy=hierarchy,
            rng=rng,
        )
    )

    self.assertIn('Household', synth_datasets)
    self.assertIn('Person', synth_datasets)
    self.assertIn('Activity', synth_datasets)
    self.assertGreater(synth_datasets['Household'].records, 0)
    self.assertIn('Person', parent_mappings)
    self.assertIn('Activity', parent_mappings)
    if synth_datasets['Person'].records > 0:
      self.assertTrue(
          np.all(
              parent_mappings['Person'] < synth_datasets['Household'].records
          )
      )
    if synth_datasets['Activity'].records > 0:
      self.assertTrue(
          np.all(parent_mappings['Activity'] < synth_datasets['Person'].records)
      )
    self.assertIn('Household->Person', discrete_results)
    self.assertIn('Person->Activity', discrete_results)

  def test_synthesize_relational_hierarchy_branching(self):
    rng = np.random.default_rng(42)
    data = {
        'Household': pd.DataFrame(
            {'hid': ['H1', 'H2'], 'income': [30.0, 80.0]}
        ),
        'Person': pd.DataFrame({
            'pid': ['P1', 'P2', 'P3'],
            'hid': ['H1', 'H1', 'H2'],
            'age': [25, 30, 45],
        }),
        'Vehicle': pd.DataFrame({
            'vid': ['V1', 'V2'],
            'hid': ['H1', 'H2'],
            'type': ['sedan', 'suv'],
        }),
    }
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(0, 100),
        },
        'Person': {
            'age': domain.NumericalAttribute(0, 100),
        },
        'Vehicle': {
            'type': domain.CategoricalAttribute(['sedan', 'suv']),
        },
    }
    fks = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=2,
        ),
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Vehicle',
            child_foreign_key='hid',
            max_children_per_parent=2,
        ),
    ]
    hierarchy = rel_domain.topological_sort_hierarchy(
        tables=list(domains.keys()), foreign_keys=fks
    )
    cfg = synthesizer.MultiTableConfig(
        foreign_keys=fks,
        discrete_mechanism=discrete_mechanisms.AIMConfig(
            pgm_iters=10, max_rounds=2
        ),
        num_permutation_slots=2,
    )
    mech = cfg.configure(domains, budget=0.5)

    preprocessed = synthesizer._run_table_preprocessing(
        mechanism=mech,
        rng=rng,
        data=data,
    )

    synth_datasets, parent_mappings, discrete_results = (
        synthesizer._synthesize_relational_hierarchy(
            mechanism=mech,
            preprocessed=preprocessed,
            hierarchy=hierarchy,
            rng=rng,
        )
    )

    self.assertIn('Household', synth_datasets)
    self.assertIn('Person', synth_datasets)
    self.assertIn('Vehicle', synth_datasets)
    if synth_datasets['Person'].records > 0:
      self.assertTrue(
          np.all(
              parent_mappings['Person'] < synth_datasets['Household'].records
          )
      )
    if synth_datasets['Vehicle'].records > 0:
      self.assertTrue(
          np.all(
              parent_mappings['Vehicle'] < synth_datasets['Household'].records
          )
      )
    self.assertIn('Household->Person', discrete_results)
    self.assertIn('Household->Vehicle', discrete_results)

  def test_decompress_synthetic_datasets(self):
    # Table 1: compressed domain (original size 4, compressed to size 2
    # via [0, 0, 1, 1])
    orig_dom_h = mbi.Domain(('region',), (4,))
    comp_dom_h = mbi.Domain(('region',), (2,))
    comp_ds_h = mbi.Dataset({'region': np.array([0, 1, 0, 1])}, comp_dom_h)

    # Table 2: uncompressed domain
    dom_p = mbi.Domain(('age',), (5,))
    ds_p = mbi.Dataset({'age': np.array([0, 2, 4])}, dom_p)

    synth_datasets = {'Household': comp_ds_h, 'Person': ds_p}
    compression_mappings = {
        'Household': {'region': np.array([0, 0, 1, 1], dtype=np.int64)},
        'Person': {},
    }

    decompressed = synthesizer._decompress_synthetic_datasets(
        synth_datasets, compression_mappings
    )

    self.assertIn('Household', decompressed)
    self.assertIn('Person', decompressed)
    # Household should have restored original domain
    self.assertEqual(decompressed['Household'].domain, orig_dom_h)
    self.assertEqual(decompressed['Household'].records, 4)
    # Values mapped from 0 should be in {0, 1}, and from 1 should be in {2, 3}
    h_vals = decompressed['Household'].data['region']
    self.assertIn(h_vals[0], (0, 1))
    self.assertIn(h_vals[1], (2, 3))
    self.assertIn(h_vals[2], (0, 1))
    self.assertIn(h_vals[3], (2, 3))
    # Person should be unchanged
    self.assertEqual(decompressed['Person'].domain['age'], 5)
    np.testing.assert_array_equal(
        decompressed['Person'].data['age'], np.array([0, 2, 4])
    )

  def test_decode_synthetic_tables(self):
    rng = np.random.default_rng(42)
    domains = {
        'Household': {
            'income': domain.NumericalAttribute(min_value=0.0, max_value=100.0),
            'region': domain.CategoricalAttribute(
                possible_values=['Urban', 'Rural']
            ),
        },
        'Person': {
            'age': domain.NumericalAttribute(min_value=0, max_value=100),
            'gender': domain.CategoricalAttribute(possible_values=['M', 'F']),
        },
    }
    # Mock column measurements for initializers
    h_measurements = {
        'income': initialization.NumericalMeasurement(
            categorical_attribute=domain.CategoricalAttribute(
                possible_values=['[0.0, 50.0)', '[50.0, 100.0]']
            ),
            bin_edges=np.array([50.0]),
        ),
        'region': initialization.CategoricalMeasurement(
            categorical_attribute=domain.CategoricalAttribute(
                possible_values=['Urban', 'Rural']
            ),
            noisy_counts=np.array([10.0, 10.0]),
            stddev=1.0,
        ),
    }
    p_measurements = {
        'age': initialization.NumericalMeasurement(
            categorical_attribute=domain.CategoricalAttribute(
                possible_values=['[0, 50)', '[50, 100]']
            ),
            bin_edges=np.array([50.0]),
        ),
        'gender': initialization.CategoricalMeasurement(
            categorical_attribute=domain.CategoricalAttribute(
                possible_values=['M', 'F']
            ),
            noisy_counts=np.array([10.0, 10.0]),
            stddev=1.0,
        ),
    }
    codecs = {
        'Household': (
            synthesizer.data_generation_v3.TabularCodec.from_measurements(
                h_measurements, domains['Household']
            )
        ),
        'Person': synthesizer.data_generation_v3.TabularCodec.from_measurements(
            p_measurements, domains['Person']
        ),
    }
    decompressed_datasets = {
        'Household': mbi.Dataset(
            {'income': np.array([0, 1]), 'region': np.array([0, 1])},
            codecs['Household'].mbi_domain,
        ),
        'Person': mbi.Dataset(
            {
                'age': np.array([0, 1, 0]),
                'gender': np.array([1, 0, 1]),
            },
            codecs['Person'].mbi_domain,
        ),
    }

    decoded_tables = synthesizer._decode_synthetic_tables(
        decompressed_datasets=decompressed_datasets,
        column_codecs=codecs,
        domains=domains,
        rng=rng,
    )

    self.assertIn('Household', decoded_tables)
    self.assertIn('Person', decoded_tables)
    h_df = decoded_tables['Household']
    p_df = decoded_tables['Person']
    self.assertEqual(list(h_df.columns), ['income', 'region'])
    self.assertEqual(list(p_df.columns), ['age', 'gender'])
    self.assertLen(h_df, 2)
    self.assertLen(p_df, 3)
    # Check numerical dequantization bounds
    self.assertTrue(np.all((h_df['income'] >= 0.0) & (h_df['income'] <= 100.0)))
    self.assertTrue(np.all((p_df['age'] >= 0) & (p_df['age'] <= 100)))
    # Check categorical decoding strings
    self.assertEqual(list(h_df['region']), ['Urban', 'Rural'])
    self.assertEqual(list(p_df['gender']), ['F', 'M', 'F'])

  def test_assign_relational_keys_3_tier(self):
    tables = {
        'Household': pd.DataFrame({'income': [50.0, 80.0]}),
        'Person': pd.DataFrame({'age': [25, 30, 45]}),
        'Activity': pd.DataFrame({'amt': [10.0, 20.0, 30.0, 40.0]}),
    }
    fks = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=2,
        ),
        rel_domain.ForeignKeyRelation(
            parent_table='Person',
            parent_primary_key='pid',
            child_table='Activity',
            child_foreign_key='pid',
            max_children_per_parent=2,
        ),
    ]
    parent_mappings = {
        'Person': np.array([0, 0, 1], dtype=np.int64),
        'Activity': np.array([0, 1, 1, 2], dtype=np.int64),
    }
    hierarchy = [
        (0, 'Household', None),
        (1, 'Person', fks[0]),
        (2, 'Activity', fks[1]),
    ]

    linked = synthesizer._assign_relational_keys(
        tables=tables,
        foreign_keys=fks,
        parent_mappings=parent_mappings,
        hierarchy=hierarchy,
    )

    self.assertIn('hid', linked['Household'].columns)
    np.testing.assert_array_equal(linked['Household']['hid'], np.array([0, 1]))

    self.assertIn('hid', linked['Person'].columns)
    self.assertIn('pid', linked['Person'].columns)
    np.testing.assert_array_equal(linked['Person']['hid'], np.array([0, 0, 1]))
    np.testing.assert_array_equal(linked['Person']['pid'], np.array([0, 1, 2]))

    self.assertIn('pid', linked['Activity'].columns)
    np.testing.assert_array_equal(
        linked['Activity']['pid'], np.array([0, 1, 1, 2])
    )

  def test_assign_relational_keys_branching_and_empty_child(self):
    tables = {
        'Household': pd.DataFrame({'income': [50.0, 80.0]}),
        'Person': pd.DataFrame({'age': [25, 30]}),
        'Vehicle': pd.DataFrame({'type': []}),
    }
    fks = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=2,
        ),
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Vehicle',
            child_foreign_key='hid',
            max_children_per_parent=2,
        ),
    ]
    parent_mappings = {
        'Person': np.array([0, 1], dtype=np.int64),
        'Vehicle': np.empty(0, dtype=np.int64),
    }
    hierarchy = [
        (0, 'Household', None),
        (1, 'Person', fks[0]),
        (1, 'Vehicle', fks[1]),
    ]

    linked = synthesizer._assign_relational_keys(
        tables=tables,
        foreign_keys=fks,
        parent_mappings=parent_mappings,
        hierarchy=hierarchy,
    )

    np.testing.assert_array_equal(linked['Household']['hid'], np.array([0, 1]))
    np.testing.assert_array_equal(linked['Person']['hid'], np.array([0, 1]))
    self.assertEmpty(linked['Vehicle'])
    self.assertIn('hid', linked['Vehicle'].columns)
    self.assertEmpty(linked['Vehicle']['hid'])

  def test_multi_table_mechanism_call_end_to_end(self):
    rng = np.random.default_rng(42)
    h_df = pd.DataFrame({
        'hid': [0, 1, 2, 3],
        'income': [20.0, 50.0, 80.0, 100.0],
        'region': ['Urban', 'Rural', 'Urban', 'Rural'],
    })
    p_df = pd.DataFrame({
        'hid': [0, 0, 1, 2, 2, 3],
        'pid': [10, 11, 12, 13, 14, 15],
        'age': [15, 45, 30, 20, 55, 60],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    })
    data = {'Household': h_df, 'Person': p_df}

    domains = {
        'Household': {
            'income': domain.NumericalAttribute(min_value=0.0, max_value=100.0),
            'region': domain.CategoricalAttribute(
                possible_values=['Urban', 'Rural']
            ),
        },
        'Person': {
            'age': domain.NumericalAttribute(min_value=0, max_value=100),
            'gender': domain.CategoricalAttribute(possible_values=['M', 'F']),
        },
    }
    fks = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=2,
        )
    ]

    cfg = synthesizer.MultiTableConfig(
        foreign_keys=fks,
        discrete_mechanism=discrete_mechanisms.AIMConfig(
            max_rounds=2,
            pgm_iters=50,
        ),
        numerical_bins=4,
        init_budget_fraction=0.2,
    )
    mechanism = cfg.configure(
        domains,
        budget=1.0,
    )

    result = mechanism(rng=rng, data=data)

    self.assertIsInstance(result, synthesizer.MultiDataGenerationResult)
    self.assertIn('Household', result.synthetic_tables)
    self.assertIn('Person', result.synthetic_tables)

    synth_h = result.synthetic_tables['Household']
    synth_p = result.synthetic_tables['Person']

    # Referential integrity check
    self.assertIn('hid', synth_h.columns)
    self.assertIn('hid', synth_p.columns)
    self.assertTrue(set(synth_p['hid']).issubset(set(synth_h['hid'])))

    # Column existence and bounds
    self.assertIn('income', synth_h.columns)
    self.assertIn('region', synth_h.columns)
    self.assertIn('age', synth_p.columns)
    self.assertIn('gender', synth_p.columns)

    self.assertTrue(
        np.all((synth_h['income'] >= 0.0) & (synth_h['income'] <= 100.0))
    )
    self.assertTrue(np.all((synth_p['age'] >= 0) & (synth_p['age'] <= 100)))
    self.assertTrue(set(synth_h['region']).issubset({'Urban', 'Rural'}))
    self.assertTrue(set(synth_p['gender']).issubset({'M', 'F'}))
    self.assertIn('Household->Person', result.discrete_mechanism_results)

  def test_multi_table_mechanism_call_size_sliced(self):
    rng = np.random.default_rng(42)
    h_df = pd.DataFrame({
        'hid': [0, 1, 2, 3],
        'income': [20.0, 50.0, 80.0, 100.0],
        'region': ['Urban', 'Rural', 'Urban', 'Rural'],
    })
    p_df = pd.DataFrame({
        'hid': [0, 0, 1, 2, 2, 3],
        'pid': [10, 11, 12, 13, 14, 15],
        'age': [15, 45, 30, 20, 55, 60],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    })
    data = {'Household': h_df, 'Person': p_df}

    domains = {
        'Household': {
            'income': domain.NumericalAttribute(min_value=0.0, max_value=100.0),
            'region': domain.CategoricalAttribute(
                possible_values=['Urban', 'Rural']
            ),
        },
        'Person': {
            'age': domain.NumericalAttribute(min_value=0, max_value=100),
            'gender': domain.CategoricalAttribute(possible_values=['M', 'F']),
        },
    }
    fks = [
        rel_domain.ForeignKeyRelation(
            parent_table='Household',
            parent_primary_key='hid',
            child_table='Person',
            child_foreign_key='hid',
            max_children_per_parent=2,
        )
    ]

    cfg = synthesizer.MultiTableConfig(
        foreign_keys=fks,
        discrete_mechanism=discrete_mechanisms.AIMConfig(
            max_rounds=2,
            pgm_iters=50,
        ),
        numerical_bins=4,
        init_budget_fraction=0.2,
        exploration_strategy='size_sliced',
    )
    mechanism = cfg.configure(
        domains,
        budget=1.0,
    )

    result = mechanism(rng=rng, data=data)

    self.assertIsInstance(result, synthesizer.MultiDataGenerationResult)
    self.assertIn('Household', result.synthetic_tables)
    self.assertIn('Person', result.synthetic_tables)

    synth_h = result.synthetic_tables['Household']
    synth_p = result.synthetic_tables['Person']

    self.assertIn('hid', synth_h.columns)
    self.assertIn('hid', synth_p.columns)
    self.assertTrue(set(synth_p['hid']).issubset(set(synth_h['hid'])))
    self.assertIn('Household->Person', result.discrete_mechanism_results)

  def test_public_api_exports(self):
    self.assertIs(rel_pkg.MultiTableConfig, synthesizer.MultiTableConfig)
    self.assertIs(rel_pkg.MultiTableMechanism, synthesizer.MultiTableMechanism)
    self.assertIs(
        rel_pkg.MultiDataGenerationResult, synthesizer.MultiDataGenerationResult
    )
    self.assertIs(rel_pkg.ForeignKeyRelation, rel_domain.ForeignKeyRelation)


if __name__ == '__main__':
  absltest.main()
