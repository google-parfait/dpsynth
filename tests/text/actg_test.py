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

"""Unit tests for dpsynth.text.actg."""

from unittest import mock

from absl.testing import absltest
import dp_accounting
import dpsynth
from dpsynth import domain
from dpsynth.discrete_mechanisms import independent
from dpsynth.text import actg
from dpsynth.text import dp_sft
from dpsynth.text import model
from flax import nnx
import jax_privacy
import numpy as np
import pandas as pd


def _mixed_schema(multi_text: bool = False) -> domain.Schema:
  """Builds a mixed tabular + text Schema for testing."""
  attrs: dict[str, domain.AttributeType] = {
      'topic': domain.CategoricalAttribute(possible_values=['AI', 'Bio']),
      'score': domain.NumericalAttribute(
          min_value=0.0, max_value=10.0, bin_edges=[5.0]
      ),
      'abstract': domain.FreeFormTextAttribute(
          max_tokens=128,
          description='Short research abstract.',
          formatting='Plain paragraph.',
      ),
  }
  if multi_text:
    attrs['title'] = domain.FreeFormTextAttribute(
        max_tokens=32, description='Concise paper title.'
    )
  return domain.Schema(attrs)


def _fine_tuner_config() -> dp_sft.DPFineTunerConfig:
  """Builds a lightweight DPFineTunerConfig for testing."""
  band_cfg = jax_privacy.execution_plan.BandMFConfig.default(
      num_bands=1, iterations=10, expected_participations=1.0
  )
  return dp_sft.DPFineTunerConfig(mechanism_config=band_cfg, learning_rate=3e-4)


class ACTGTest(absltest.TestCase):

  def test_config_validation(self):
    with self.assertRaises(ValueError):
      actg.ACTGConfig(tabular_budget_fraction=0.0)
    with self.assertRaises(ValueError):
      actg.ACTGConfig(tabular_budget_fraction=1.5)
    with self.assertRaises(ValueError):
      actg.ACTGConfig(sampling_temperature=-0.1)
    with self.assertRaises(ValueError):
      actg.ACTGConfig(sampling_batch_size=0)

  def test_schema_validation(self):
    cfg = actg.ACTGConfig()
    with self.assertRaises(ValueError):
      cfg.configure(None, budget=1.0)
    cat = domain.CategoricalAttribute(possible_values=['AI', 'Bio'])
    with self.assertRaises(ValueError):
      cfg.configure(domain.Schema({'topic': cat}), budget=1.0)
    only_text = domain.Schema({'abstract': domain.FreeFormTextAttribute()})
    with self.assertRaises(ValueError):
      cfg.configure(only_text, budget=1.0)

  def test_configure_and_calibrate_jointly(self):
    schema = _mixed_schema()
    tab_cfg = dpsynth.TabularConfig(
        discrete_mechanism=independent.IndependentConfig(pgm_iters=50)
    )
    cfg = actg.ACTGConfig(
        tabular_config=tab_cfg,
        fine_tuner_config=_fine_tuner_config(),
        tabular_budget_fraction=0.2,
    )
    mech = dpsynth.calibrate(cfg, schema, epsilon=2.0, delta=1e-5)
    self.assertIsInstance(mech, actg.ACTGMechanism)
    self.assertIsNotNone(mech.fine_tuner)
    assert mech.fine_tuner is not None
    sigma = mech.fine_tuner.config.mechanism_config.noise_multiplier
    self.assertIsNotNone(sigma)
    self.assertGreater(sigma, 0.0)
    self.assertIsInstance(mech.dp_event, dp_accounting.ComposedDpEvent)
    self.assertEqual(list(mech.tabular_mechanism.schema), ['topic', 'score'])
    self.assertEqual(list(mech.text_attributes), ['abstract'])

    accountant = dp_accounting.pld.PLDAccountant()
    accountant.compose(mech.dp_event)
    self.assertLessEqual(accountant.get_epsilon(1e-5), 2.0 + 1e-2)

  def test_zero_shot_mode_skips_fine_tuner(self):
    cfg = actg.ACTGConfig(tabular_budget_fraction=1.0)
    mech = cfg.configure(_mixed_schema(), budget=0.5)
    self.assertIsNone(mech.fine_tuner)
    self.assertEqual(mech.dp_event, mech.tabular_mechanism.dp_event)

  def test_single_text_prompt_and_response_formatting(self):
    mech = actg.ACTGConfig().configure(_mixed_schema(), budget=1.0)
    row = {'topic': 'AI', 'score': np.float64(7.5), 'abstract': 'Synth text.'}
    prompt, response = mech.format_example(row)
    self.assertIn('{"topic": "AI", "score": 7.5}', prompt)
    self.assertIn('Description: Short research abstract.', prompt)
    self.assertIn('Formatting: Plain paragraph.', prompt)
    self.assertEqual(response, 'Synth text.')
    parsed = mech.parse_response('  Generated output. \n')
    self.assertEqual(parsed, {'abstract': 'Generated output.'})

  def test_multi_text_prompt_and_response_formatting(self):
    schema = _mixed_schema(multi_text=True)
    mech = actg.ACTGConfig().configure(schema, budget=1.0)
    row = {
        'topic': 'Bio',
        'score': 3.0,
        'abstract': 'Cell study.',
        'title': 'Cells',
    }
    prompt, response = mech.format_example(row)
    expected_line = '- abstract: Short research abstract. Plain paragraph.'
    self.assertIn(expected_line, prompt)
    self.assertIn('- title: Concise paper title.', prompt)
    parsed = mech.parse_response(f'```json\n{response}\n```')
    self.assertEqual(parsed, {'abstract': 'Cell study.', 'title': 'Cells'})
    fallback = mech.parse_response('not valid json')
    self.assertEqual(fallback, {'abstract': '', 'title': ''})

  def test_end_to_end_call_with_mock_gemma(self):
    schema = _mixed_schema(multi_text=True)
    tab_cfg = dpsynth.TabularConfig(
        discrete_mechanism=independent.IndependentConfig(pgm_iters=20)
    )
    cfg = actg.ACTGConfig(
        tabular_config=tab_cfg,
        fine_tuner_config=_fine_tuner_config(),
        tabular_budget_fraction=0.25,
        sampling_batch_size=2,
    )
    mech = cfg.configure(schema, budget=1.0)
    df = pd.DataFrame({
        'topic': ['AI', 'Bio', 'AI', 'Bio'],
        'score': [2.0, 8.0, 3.0, 7.0],
        'abstract': ['a1', 'a2', 'a3', 'a4'],
        'title': ['t1', 't2', 't3', 't4'],
    })
    dummy_model = nnx.Module()
    fake_ft_result = dp_sft.FineTuneResult(model=dummy_model)
    fake_sampler = mock.Mock(
        return_value=[
            '{"abstract": "syn_a1", "title": "syn_t1"}',
            '{"abstract": "syn_a2", "title": "syn_t2"}',
            '{"abstract": "syn_a3", "title": "syn_t3"}',
        ]
    )
    with (
        mock.patch.object(
            dp_sft.DPFineTuner, '__call__', return_value=fake_ft_result
        ) as mock_ft,
        mock.patch.object(
            model, 'GemmaSampler', return_value=fake_sampler
        ) as mock_sampler_cls,
    ):
      result = mech(np.random.default_rng(0), df, num_rows=3)

    mock_ft.assert_called_once()
    sft_examples = mock_ft.call_args.args[1]
    self.assertLen(sft_examples, 4)
    mock_sampler_cls.assert_called_once_with(
        dummy_model, max_seq_length=512, temperature=1.0
    )
    syn = result.synthetic_data
    self.assertEqual(list(syn.columns), ['topic', 'score', 'abstract', 'title'])
    self.assertLen(syn, 3)
    self.assertEqual(list(syn['abstract']), ['syn_a1', 'syn_a2', 'syn_a3'])
    self.assertEqual(list(syn['title']), ['syn_t1', 'syn_t2', 'syn_t3'])
    self.assertIs(result.fine_tune_result, fake_ft_result)

  def test_zero_shot_call_loads_base_gemma(self):
    schema = _mixed_schema(multi_text=False)
    tab_cfg = dpsynth.TabularConfig(
        discrete_mechanism=independent.IndependentConfig(pgm_iters=20)
    )
    cfg = actg.ACTGConfig(tabular_config=tab_cfg, tabular_budget_fraction=1.0)
    mech = cfg.configure(schema, budget=1.0)
    df = pd.DataFrame({
        'topic': ['AI', 'Bio'],
        'score': [2.0, 8.0],
        'abstract': ['a1', 'a2'],
    })
    dummy_model = nnx.Module()
    fake_sampler = mock.Mock(return_value=['zero_shot_1', 'zero_shot_2'])
    with (
        mock.patch.object(
            model, 'load_gemma', return_value=dummy_model
        ) as mock_load,
        mock.patch.object(model, 'GemmaSampler', return_value=fake_sampler),
    ):
      result = mech(np.random.default_rng(0), df, num_rows=2)

    lora_cfg = model.LoraConfig(rank=16)
    mock_load.assert_called_once_with('gemma4_e2b_it', lora_cfg)
    self.assertIsNone(result.fine_tune_result)
    syn_abstracts = list(result.synthetic_data['abstract'])
    self.assertEqual(syn_abstracts, ['zero_shot_1', 'zero_shot_2'])

  def test_yaml_roundtrip(self):
    cfg = actg.ACTGConfig(
        tabular_config=dpsynth.TabularConfig(
            discrete_mechanism=independent.IndependentConfig(pgm_iters=100),
            numerical_bins=16,
        ),
        fine_tuner_config=_fine_tuner_config(),
        tabular_budget_fraction=0.2,
        sampling_temperature=0.8,
    )
    yaml_str = dpsynth.to_yaml(cfg)
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertIsInstance(loaded, actg.ACTGConfig)
    self.assertEqual(loaded.tabular_config, cfg.tabular_config)
    self.assertEqual(loaded.tabular_budget_fraction, 0.2)
    self.assertEqual(loaded.sampling_temperature, 0.8)
    self.assertEqual(loaded.fine_tuner_config.model, 'gemma4_e2b_it')
    self.assertEqual(loaded.fine_tuner_config.learning_rate, 3e-4)
    ft_mech = loaded.fine_tuner_config.mechanism_config
    orig_mech = cfg.fine_tuner_config.mechanism_config
    self.assertEqual(ft_mech.iterations, orig_mech.iterations)
    loaded_strat = np.asarray(ft_mech.strategy, dtype=float)
    expected_strat = np.asarray(orig_mech.strategy, dtype=float)
    np.testing.assert_allclose(loaded_strat, expected_strat)
    calibrated = loaded.configure(_mixed_schema(), budget=0.5)
    self.assertIsNotNone(calibrated.dp_event)


if __name__ == '__main__':
  absltest.main()
