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

"""Unit tests for cattrs-based serialize.py in DPSynth."""

import os
from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
import dpsynth
import mbi
import yaml


class SerializeTest(parameterized.TestCase):

  def test_mst_config_roundtrip_and_defaults_omitted(self):
    config = dpsynth.MSTConfig(
        pgm_iters=2500,
        select_budget_fraction=0.75,
    )
    yaml_str = dpsynth.to_yaml(config)
    # Verify default maximum_marginal_size is omitted from YAML.
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'MSTConfig')
    self.assertEqual(raw_dict['pgm_iters'], 2500)
    self.assertEqual(raw_dict['select_budget_fraction'], 0.75)
    self.assertNotIn('maximum_marginal_size', raw_dict)

    # Roundtrip check
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_aim_config_roundtrip(self):
    config = dpsynth.AIMConfig(
        max_rounds=25,
        max_model_size=50,
        max_marginal_size=2e6,
        anneal_factor=3.0,
        select_budget_fraction=0.2,
        pgm_iters=500,
    )
    yaml_str = dpsynth.to_yaml(config)
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_swift_config_roundtrip(self):
    config = dpsynth.SWIFTConfig(
        max_clique_size=5e6,
        max_marginal_size=2e6,
        pgm_iters=8000,
        select_budget_frac=0.15,
    )
    yaml_str = dpsynth.to_yaml(config)
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_independent_config_roundtrip(self):
    config = dpsynth.IndependentConfig(pgm_iters=3000)
    yaml_str = dpsynth.to_yaml(config)
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_direct_config_roundtrip(self):
    config = dpsynth.DirectConfig(
        pgm_iters=4000,
        prespecified_marginal_queries=[('a', 'b'), ('c',)],
    )
    yaml_str = dpsynth.to_yaml(config)
    self.assertIn('- [a, b]\n', yaml_str)
    self.assertIn('- [c]\n', yaml_str)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertNotIn('estimator', raw_dict)
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_direct_config_with_estimator_roundtrip(self):
    config = dpsynth.DirectConfig(
        pgm_iters=1500,
        prespecified_marginal_queries=[('a', 'b')],
        estimator=mbi.estimation.UniversalAcceleratedMethod(max_iter_search=50),
    )
    yaml_str = dpsynth.to_yaml(config)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(
        raw_dict['estimator'],
        {'type': 'UniversalAcceleratedMethod', 'max_iter_search': 50},
    )
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_discrete_config_roundtrip(self):
    config = dpsynth.DiscreteConfig(
        mechanism=dpsynth.AIMConfig(pgm_iters=400),
        compress_columns=['col1', 'col2'],
        one_way_budget_fraction=0.2,
    )
    yaml_str = dpsynth.to_yaml(config)
    self.assertIn('- col1\n- col2\n', yaml_str)
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_tabular_config_roundtrip_and_defaults_omitted(self):
    config = dpsynth.TabularConfig(
        discrete_mechanism=dpsynth.MSTConfig(pgm_iters=1500),
        numerical_bins=64,
        init_budget_fraction=0.15,
    )
    yaml_str = dpsynth.to_yaml(config)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'TabularConfig')
    self.assertEqual(raw_dict['numerical_bins'], 64)
    # Default domains=None and cross_attribute_constraints=() are omitted:
    self.assertNotIn('domains', raw_dict)
    self.assertNotIn('cross_attribute_constraints', raw_dict)
    self.assertNotIn('compress_columns', raw_dict)

    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_tabular_config_with_compress_columns(self):
    config = dpsynth.TabularConfig(
        compress_columns=True,
    )
    yaml_str = dpsynth.to_yaml(config)
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, config)
    self.assertTrue(loaded.compress_columns)

  def test_tabular_config_with_legacy_domains(self):
    config = dpsynth.TabularConfig(
        domains={
            'age': dpsynth.NumericalAttribute(min_value=0, max_value=120),
            'gender': dpsynth.CategoricalAttribute(possible_values=['M', 'F']),
            'notes': dpsynth.FreeFormTextAttribute(max_tokens=100),
            'state': dpsynth.OpenSetCategoricalAttribute(),
        },
        discrete_mechanism=dpsynth.MSTConfig(pgm_iters=1500),
        numerical_bins=64,
    )
    yaml_str = dpsynth.to_yaml(config)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertIn('domains', raw_dict)
    self.assertNotIn('clip_to_range', raw_dict['domains']['age'])
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_initializer_configs_roundtrip(self):
    num_init = dpsynth.NumericalInitializerConfig(
        num_partitions=16,
    )
    yaml_str = dpsynth.to_yaml(num_init)
    self.assertEqual(dpsynth.from_yaml(yaml_str), num_init)

    cat_init = dpsynth.CategoricalInitializerConfig()
    yaml_str = dpsynth.to_yaml(cat_init)
    self.assertEqual(dpsynth.from_yaml(yaml_str), cat_init)

    open_init = dpsynth.OpenSetInitializerConfig(min_count=5)
    yaml_str = dpsynth.to_yaml(open_init)
    self.assertEqual(dpsynth.from_yaml(yaml_str), open_init)

  def test_multitable_config_roundtrip(self):
    config = dpsynth.MultiTableConfig(
        discrete_mechanism=dpsynth.MSTConfig(pgm_iters=500),
    )
    yaml_str = dpsynth.to_yaml(config)
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_schema_roundtrip(self):
    schema = dpsynth.Schema({
        'age': dpsynth.NumericalAttribute(min_value=0, max_value=120),
        'cat': dpsynth.CategoricalAttribute(possible_values=['A', 'B']),
    })
    yaml_str = dpsynth.to_yaml(schema)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'Schema')
    # Default constraints=() should be omitted:
    self.assertNotIn('constraints', raw_dict)

    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, schema)

  def test_domain_dict_roundtrip(self):
    domains = {
        'age': dpsynth.NumericalAttribute(
            min_value=17, max_value=90, dtype='int'
        ),
        'cat': dpsynth.CategoricalAttribute(possible_values=['A', 'B']),
        'city': dpsynth.OpenSetCategoricalAttribute(),
    }
    yaml_str = dpsynth.to_yaml(domains)
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, domains)

  def test_file_io_roundtrip(self):
    config = dpsynth.TabularConfig(
        discrete_mechanism=dpsynth.MSTConfig(pgm_iters=1000),
    )
    tmp_path = os.path.join(self.create_tempdir().full_path, 'config.yaml')
    dpsynth.to_yaml(config, tmp_path)
    loaded = dpsynth.from_yaml(tmp_path)
    self.assertEqual(loaded, config)

  def test_dpsynth_top_level_helpers(self):
    config = dpsynth.MSTConfig(pgm_iters=1234)
    yaml_str = dpsynth.to_yaml(config)
    self.assertEqual(dpsynth.from_yaml(yaml_str), config)

    tmp_path = os.path.join(self.create_tempdir().full_path, 'top_level.yaml')
    dpsynth.to_yaml(config, tmp_path)
    self.assertEqual(dpsynth.from_yaml(tmp_path), config)

  def test_zcdp_event_roundtrip(self):
    event = dp_accounting.ZCDpEvent(rho=0.5)
    yaml_str = dpsynth.to_yaml(event)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'ZCDpEvent')
    self.assertEqual(raw_dict['rho'], 0.5)

    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, event)

  def test_gaussian_dp_event_roundtrip(self):
    event = dp_accounting.GaussianDpEvent(noise_multiplier=1.25)
    yaml_str = dpsynth.to_yaml(event)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'GaussianDpEvent')
    self.assertEqual(raw_dict['noise_multiplier'], 1.25)

    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, event)

  def test_composed_dp_event_roundtrip(self):
    event = dp_accounting.ComposedDpEvent([
        dp_accounting.ZCDpEvent(rho=0.1),
        dp_accounting.dp_event.EpsilonDeltaDpEvent(epsilon=0.0, delta=1e-5),
    ])
    yaml_str = dpsynth.to_yaml(event)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'ComposedDpEvent')
    self.assertEqual(raw_dict['events'][0]['type'], 'ZCDpEvent')
    self.assertEqual(raw_dict['events'][1]['type'], 'EpsilonDeltaDpEvent')

    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, event)

  def test_self_composed_dp_event_roundtrip(self):
    event = dp_accounting.SelfComposedDpEvent(
        event=dp_accounting.GaussianDpEvent(noise_multiplier=1.5), count=10
    )
    yaml_str = dpsynth.to_yaml(event)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'SelfComposedDpEvent')
    self.assertEqual(raw_dict['event']['type'], 'GaussianDpEvent')
    self.assertEqual(raw_dict['count'], 10)

    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, event)

  def test_mechanism_dp_event_roundtrip(self):
    dom = {'col': dpsynth.CategoricalAttribute(possible_values=['a', 'b'])}
    config = dpsynth.TabularConfig(
        discrete_mechanism=dpsynth.MSTConfig(pgm_iters=100)
    )
    mech = config.calibrate(dom, epsilon=1.0, delta=1e-5)
    event = mech.dp_event
    yaml_str = dpsynth.to_yaml(event)
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, event)

  def test_unknown_type_raises(self):
    with self.assertRaises(ValueError):
      dpsynth.from_yaml('type: NonExistentConfig\n')


if __name__ == '__main__':
  absltest.main()
