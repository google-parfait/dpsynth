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

from __future__ import annotations

import functools
import os

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
import dpsynth
from dpsynth import constraints
from dpsynth import data_generation_v3
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth.discrete_mechanisms import aim
from dpsynth.discrete_mechanisms.independent import IndependentConfig
import mbi
import numpy as np
import pandas as pd

TabularConfig = functools.partial(
    data_generation_v3.TabularConfig,
    discrete_mechanism=discrete_mechanisms.MSTConfig(pgm_iters=100),
)


def _make_discrete_data(rng, n=1000):
  domains = mbi.Domain(['a', 'b', 'c'], [3, 3, 3])
  a = rng.integers(0, 3, size=n)
  b = np.where(rng.random(n) < 0.75, a, rng.integers(0, 3, size=n))
  c = (a + b + rng.integers(0, 2, size=n)) % 3
  return mbi.Dataset({'a': a, 'b': b, 'c': c}, domains)


def _make_mixed_data(rng, n=1000):
  domains = {
      'a': domain.CategoricalAttribute(possible_values=[0, 1, 2]),
      'b': domain.CategoricalAttribute(possible_values=[0, 1, 2]),
      'c': domain.NumericalAttribute(
          min_value=0.0, max_value=1.0, dtype='float'
      ),
  }
  a = rng.integers(0, 3, size=n)
  b = np.where(rng.random(n) < 0.75, a, rng.integers(0, 3, size=n))
  c = np.clip((a + b) / 4.0 + rng.normal(0.0, 0.1, size=n), 0.0, 1.0)
  return pd.DataFrame({'a': a, 'b': b, 'c': c}), domains


def _normalized_l1(data, model, clique):
  expected = data.project(clique).datavector().astype(float)
  actual = model.project(clique).datavector().astype(float)
  expected /= expected.sum()
  actual /= actual.sum()
  return np.abs(expected - actual).sum() / 2.0


def _discrete_workload_mechanism_baseline_errors(
    config, baseline_config, workload, budget=5.0
):
  rng = np.random.default_rng(0)
  data = _make_discrete_data(rng)

  mechanism_result = config.configure(budget=budget)(rng, data)
  baseline_result = baseline_config.configure(budget=budget)(rng, data)

  mechanism_error = np.mean([
      _normalized_l1(data, mechanism_result.model, clique)
      for clique in workload
  ])
  baseline_error = np.mean([
      _normalized_l1(data, baseline_result.model, clique) for clique in workload
  ])
  return mechanism_error, baseline_error


def _mixed_workload_mechanism_baseline_errors(
    config, baseline_config, workload, budget=5.0, numerical_bins=16
):
  rng = np.random.default_rng(0)
  data, domains = _make_mixed_data(rng, n=1000)

  mechanism_synth = TabularConfig(
      discrete_mechanism=config,
      numerical_bins=numerical_bins,
  )
  baseline_synth = TabularConfig(
      discrete_mechanism=baseline_config,
      numerical_bins=numerical_bins,
  )

  mechanism_result = mechanism_synth.configure(domains, budget=budget)(
      rng, data
  )
  baseline_result = baseline_synth.configure(domains, budget=budget)(rng, data)

  mechanism_error = np.mean([
      _normalized_l1(
          mechanism_result.codec.encode(data),
          mechanism_result.discrete_mechanism_result.model,
          clique,
      )
      for clique in workload
  ])
  baseline_error = np.mean([
      _normalized_l1(
          baseline_result.codec.encode(data),
          baseline_result.discrete_mechanism_result.model,
          clique,
      )
      for clique in workload
  ])
  return mechanism_error, baseline_error


class DataGenerationV3Test(parameterized.TestCase):

  def test_end_to_end_categorical(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
        'B': domain.CategoricalAttribute(
            possible_values=['x', 'y', 'z'], out_of_domain_index=0
        ),
    }
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
    rng = np.random.default_rng(0)
    calibrated = TabularConfig().configure(domains, budget=100.0)
    synthetic_df = calibrated(rng, df).synthetic_data
    self.assertIsInstance(synthetic_df, pd.DataFrame)
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])

  def test_end_to_end_numerical(self):
    domains = {
        'A': domain.NumericalAttribute(min_value=0, max_value=10),
        'B': domain.NumericalAttribute(min_value=-10, max_value=10),
    }
    df = pd.DataFrame({'A': [5, 5, 0], 'B': [5, -10, -5]}, dtype=float)
    rng = np.random.default_rng(0)
    calibrated = TabularConfig().configure(domains, budget=100.0)
    synthetic_df = calibrated(rng, df).synthetic_data
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])
    for col, attr in domains.items():
      self.assertTrue(
          synthetic_df[col].between(attr.min_value, attr.max_value).all()
      )

  def test_end_to_end_mixed_domain(self):
    domains = {
        'A': domain.OpenSetCategoricalAttribute(),
        'B': domain.NumericalAttribute(min_value=0, max_value=10),
    }
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1.0, 5.0, 10.0]})
    rng = np.random.default_rng(0)
    calibrated = TabularConfig().configure(domains, budget=100.0, delta=1e-5)
    synthetic_df = calibrated(rng, df).synthetic_data
    self.assertIsInstance(synthetic_df, pd.DataFrame)
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])

  @parameterized.named_parameters(
      ('large_epsilon', 100, 0.1),
      ('small_epsilon', 1e-5, 1e-6),
  )
  def test_end_to_end_with_epsilon_delta(self, epsilon: float, delta: float):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
        'B': domain.CategoricalAttribute(
            possible_values=['x', 'y', 'z'], out_of_domain_index=0
        ),
    }
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
    rng = np.random.default_rng(0)
    calibrated = TabularConfig().calibrate(
        domains, epsilon=epsilon, delta=delta
    )
    result = calibrated(rng, df)
    synthetic_df = result.synthetic_data
    self.assertIsInstance(synthetic_df, pd.DataFrame)
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])

  def test_raises_on_freeform_text_attribute(self):
    domains = {
        'A': domain.CategoricalAttribute(possible_values=['a', 'b']),
        'text': domain.FreeFormTextAttribute(max_tokens=128),
    }
    v3 = TabularConfig()
    with self.assertRaises(Exception):
      v3.configure(domains, budget=1.0)

  def test_raises_when_not_calibrated(self):
    df = pd.DataFrame({'A': ['a', 'b', 'c']})
    rng = np.random.default_rng(0)
    v3 = TabularConfig()
    with self.assertRaises(Exception):
      v3(rng, df)

  def test_dp_event_returns_composed_event(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
    }
    calibrated = TabularConfig().configure(domains, budget=100.0)
    self.assertIsInstance(calibrated.dp_event, dp_accounting.ComposedDpEvent)

  def test_calibrate_domain_positional_only(self):
    domains = {
        'A': domain.CategoricalAttribute(possible_values=['a', 'b', 'c']),
    }
    config = TabularConfig()
    # Passing domain positionally works via both method and top-level function.
    with self.assertWarnsRegex(
        DeprecationWarning,
        'MechanismConfig.calibrate is deprecated. Use dpsynth.calibrate'
        ' instead.',
    ):
      calibrated = config.calibrate(domains, epsilon=1.0, delta=1e-5)
    self.assertIsNotNone(calibrated)
    calibrated_top = dpsynth.calibrate(config, domains, epsilon=1.0, delta=1e-5)
    self.assertEqual(calibrated.dp_event, calibrated_top.dp_event)
    # Passing domain as keyword argument raises TypeError.
    with self.assertRaises(TypeError):
      config.calibrate(domain=domains, epsilon=1.0, delta=1e-5)  # pyrefly: ignore[unexpected-keyword]
    with self.assertRaises(TypeError):
      dpsynth.calibrate(config, domain=domains, epsilon=1.0, delta=1e-5)  # pyrefly: ignore[unexpected-keyword]

  def test_calibrate_delta_split(self):
    domains = {
        'A': domain.OpenSetCategoricalAttribute(),
        'B': domain.OpenSetCategoricalAttribute(),
        'C': domain.CategoricalAttribute(possible_values=['x', 'y']),
    }
    config = TabularConfig()
    calibrated = dpsynth.calibrate(config, domains, epsilon=1.0, delta=1e-5)
    self.assertAlmostEqual(calibrated.initializers['A'].delta, 2.5e-6)
    self.assertAlmostEqual(calibrated.initializers['B'].delta, 2.5e-6)

    custom = dpsynth.calibrate(
        config, domains, epsilon=1.0, delta=1e-5, delta_split=0.2
    )
    self.assertAlmostEqual(custom.initializers['A'].delta, 1e-6)
    self.assertAlmostEqual(custom.initializers['B'].delta, 1e-6)

    for bad_split in (0.0, 1.0, -0.1, 1.5):
      with self.assertRaisesRegex(ValueError, 'delta_split must be in'):
        dpsynth.calibrate(
            config, domains, epsilon=1.0, delta=1e-5, delta_split=bad_split
        )

  @parameterized.product(
      sentinel=[np.nan, None],
      clip_to_range=[True, False],
      dtype=['float', 'int'],
  )
  def test_nan_numerical_column(self, sentinel, clip_to_range, dtype):
    """Regression test: NaN/None/pd.NA numerical columns should not crash."""
    max_value = 100 if dtype == 'int' else 10
    domains = {
        'A': domain.NumericalAttribute(
            min_value=0,
            max_value=max_value,
            clip_to_range=clip_to_range,
            dtype=dtype,
        ),
        'B': domain.CategoricalAttribute(possible_values=['x', 'y']),
    }
    df = pd.DataFrame({
        'A': [sentinel, sentinel, sentinel],
        'B': ['x', 'y', 'x'],
    })
    rng = np.random.default_rng(0)
    calibrated = TabularConfig().configure(domains, budget=100.0)
    result = calibrated(rng, df)
    self.assertIsInstance(result.synthetic_data, pd.DataFrame)
    self.assertListEqual(result.synthetic_data.columns.tolist(), ['A', 'B'])

  def test_heterogeneous_input_dataframe_with_none(self):
    """Verifies that DataFrames containing None, NaN, and mixed types run cleanly."""
    domains = {
        'cat_str': domain.CategoricalAttribute(possible_values=['a', 'b']),
        'cat_int': domain.CategoricalAttribute(possible_values=[1, 2, 3]),
        'num': domain.NumericalAttribute(min_value=0.0, max_value=100.0),
        'openset': domain.OpenSetCategoricalAttribute(),
    }
    df = pd.DataFrame({
        'cat_str': ['a', None, 'b', np.nan, 'other'],
        'cat_int': [1, None, 2, np.nan, 999],
        'num': [10.5, None, 50.0, np.nan, '75.0'],
        'openset': ['alpha', None, 'beta', np.nan, 42],
    })
    rng = np.random.default_rng(0)
    calibrated = TabularConfig().configure(domains, budget=100.0, delta=1e-5)
    result = calibrated(rng, df)
    self.assertIsInstance(result.synthetic_data, pd.DataFrame)
    self.assertListEqual(
        result.synthetic_data.columns.tolist(),
        ['cat_str', 'cat_int', 'num', 'openset'],
    )

  def test_discrete_workload_regression_with_aim(self):
    workload = [('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c'), ('b', 'c')]
    config = aim.AIMConfig(workload=workload, max_rounds=4, pgm_iters=500)
    baseline_config = IndependentConfig(pgm_iters=500)
    mechanism_error, baseline_error = (
        _discrete_workload_mechanism_baseline_errors(
            config, baseline_config, workload
        )
    )
    self.assertLess(mechanism_error, 0.05 * baseline_error)

  def test_mixed_workload_regression_with_aim(self):
    workload = [('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c'), ('b', 'c')]
    config = aim.AIMConfig(workload=workload, max_rounds=4, pgm_iters=500)
    baseline_config = IndependentConfig(pgm_iters=500)
    mechanism_error, baseline_error = _mixed_workload_mechanism_baseline_errors(
        config, baseline_config, workload
    )
    self.assertLess(mechanism_error, 0.05 * baseline_error)

  def test_empty_dataset(self):
    """Tests that DPSynth works without crashing on empty datasets, and outputs noisy rows."""
    domains = {
        'A': domain.CategoricalAttribute(possible_values=['a', 'b', 'c']),
        'B': domain.NumericalAttribute(min_value=0, max_value=10),
    }
    df = pd.DataFrame(columns=['A', 'B'])
    rng = np.random.default_rng(0)
    calibrated = TabularConfig().configure(domains, budget=100.0)
    result = calibrated(rng, df)

    self.assertIsInstance(result.synthetic_data, pd.DataFrame)
    self.assertListEqual(result.synthetic_data.columns.tolist(), ['A', 'B'])
    # The true count is 0, but DPSynth always outputs at least one row.
    self.assertLen(result.synthetic_data, 1)

  def test_checkpointing(self):
    temp_dir = self.create_tempdir().full_path
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
        'B': domain.NumericalAttribute(min_value=0.0, max_value=10.0),
    }
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1.0, 5.0, 10.0]})
    rng = np.random.default_rng(0)
    calibrated = TabularConfig().configure(domains, budget=100.0)

    with dpsynth.checkpoint(temp_dir):
      result1 = calibrated(rng, df)
    self.assertIsInstance(result1.synthetic_data, pd.DataFrame)

    col_meas_path = os.path.join(temp_dir, 'column_measurements.npz')
    disc_data_path = os.path.join(temp_dir, 'discrete_data.npz')
    precomp_path = os.path.join(temp_dir, 'precomputed_marginals.npz')
    self.assertTrue(os.path.exists(col_meas_path))
    self.assertTrue(os.path.exists(disc_data_path))
    self.assertTrue(os.path.exists(precomp_path))

    with dpsynth.checkpoint(temp_dir):
      result2 = calibrated(rng, df)

    self.assertIsInstance(result2.synthetic_data, pd.DataFrame)
    self.assertListEqual(result2.synthetic_data.columns.tolist(), ['A', 'B'])

  def test_numerical_epsilon_ratio_plumbing(self):
    domains = {'A': domain.NumericalAttribute(min_value=0, max_value=10)}
    config = TabularConfig(numerical_epsilon_ratio=1.414)
    calibrated = config.configure(domains, budget=10.0)
    init_mech = calibrated.initializers['A']
    self.assertEqual(init_mech.config.epsilon_ratio, 1.414)

  def test_poisson_calibrate_with_categorical_domains_and_gdp_mech(self):
    domains = {
        'A': domain.CategoricalAttribute(possible_values=['a', 'b', 'c']),
        'B': domain.CategoricalAttribute(possible_values=['x', 'y', 'z']),
    }
    config = TabularConfig(
        discrete_mechanism=discrete_mechanisms.IndependentConfig(),
    )
    mechanism = config.calibrate(
        domains,
        epsilon=1.0,
        delta=1e-3,
        poisson_sampling_prob=0.1,
    )
    self.assertIsNotNone(mechanism)

  def test_poisson_calibrate_with_mixed_domains(self):
    domains = {
        'A': domain.CategoricalAttribute(possible_values=['a', 'b', 'c']),
        'B': domain.NumericalAttribute(min_value=0, max_value=10),
        'C': domain.OpenSetCategoricalAttribute(),
    }
    config = TabularConfig()
    mechanism = config.calibrate(
        domains,
        epsilon=1.0,
        delta=1e-6,
        poisson_sampling_prob=0.1,
    )
    self.assertIsNotNone(mechanism)

  def test_configure_infinite_zcdp_rho(self):
    domains = {
        'A': domain.CategoricalAttribute(possible_values=['a', 'b', 'c']),
        'B': domain.NumericalAttribute(min_value=0, max_value=10),
    }
    config = TabularConfig()
    mechanism = config.configure(domains, budget=np.inf)
    self.assertIsNotNone(mechanism)

  def test_configure_with_schema(self):
    domains = {
        'A': domain.CategoricalAttribute(possible_values=['a', 'b', 'c']),
        'B': domain.CategoricalAttribute(possible_values=['x', 'y', 'z']),
    }
    preset = TabularConfig()
    calibrated = preset.configure(domains, budget=100.0)
    self.assertIsInstance(calibrated, data_generation_v3.TabularMechanism)

  def test_configure_schema_overrides_domains(self):
    old = {'A': domain.CategoricalAttribute(possible_values=['a', 'b'])}
    new = {
        'X': domain.CategoricalAttribute(possible_values=['x', 'y', 'z']),
        'Y': domain.CategoricalAttribute(possible_values=['1', '2']),
    }
    config = TabularConfig(domains=old)
    calibrated = config.configure(new, budget=100.0)
    self.assertSetEqual(set(calibrated.schema.keys()), {'X', 'Y'})

  def test_configure_no_schema_no_domains_raises(self):
    config = TabularConfig()
    with self.assertRaisesRegex(ValueError, 'No schema provided'):
      config.configure(budget=100.0)

  def test_configure_schema_with_constraints(self):
    domains = {
        'A': domain.CategoricalAttribute(possible_values=['a', 'b']),
    }
    mock_constraint = object()
    schema = domain.Schema(domains, constraints=[mock_constraint])
    config = TabularConfig()
    calibrated = config.configure(schema, budget=100.0)
    self.assertEqual(calibrated.schema.constraints, (mock_constraint,))

  def test_tabular_synthesizer_deprecated(self):
    with self.assertWarnsRegex(
        DeprecationWarning,
        'TabularSynthesizer is deprecated. Use TabularConfig for configuration '
        'and TabularMechanism for the calibrated runnable mechanism.',
    ):
      data_generation_v3.TabularSynthesizer(domains={})

  def test_compress_columns(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['common1', 'common2', 'rare1', 'rare2', 'rare3'],
            out_of_domain_index=0,
        ),
        'B': domain.CategoricalAttribute(
            possible_values=['x', 'y'], out_of_domain_index=0
        ),
    }
    df = pd.DataFrame({
        'A': ['common1'] * 500 + ['common2'] * 500,
        'B': ['x'] * 500 + ['y'] * 500,
    })
    rng = np.random.default_rng(0)
    calibrated = TabularConfig(compress_columns=True).configure(
        domains, budget=10000.0
    )
    result = calibrated(rng, df)
    self.assertIsInstance(result.synthetic_data, pd.DataFrame)
    self.assertListEqual(result.synthetic_data.columns.tolist(), ['A', 'B'])
    self.assertIn('A', result.discrete_mechanism_result.mappings)

  def test_compress_columns_respects_constraints(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['common1', 'common2', 'rare1', 'rare2', 'rare3'],
            out_of_domain_index=0,
        ),
        'B': domain.CategoricalAttribute(
            possible_values=['x', 'y'], out_of_domain_index=0
        ),
    }
    df = pd.DataFrame({
        'A': ['common1'] * 500 + ['common2'] * 500,
        'B': ['x'] * 500 + ['y'] * 500,
    })
    constraint = constraints.Constraint(
        attribute_names=('A', 'B'),
        attribute_domains=(
            domains['A'],
            domains['B'],
        ),
        impossible_combinations=[('rare1', 'x')],
    )
    schema = domain.Schema(domains, constraints=[constraint])
    rng = np.random.default_rng(0)
    calibrated = TabularConfig(compress_columns=True).configure(
        schema, budget=10000.0
    )
    result = calibrated(rng, df)
    self.assertNotIn('A', result.discrete_mechanism_result.mappings)

  def test_use_jax_for_generation_and_bincount(self):
    domains = {
        'A': domain.CategoricalAttribute(possible_values=['a1', 'a2']),
        'B': domain.CategoricalAttribute(possible_values=['b1', 'b2']),
    }
    df = pd.DataFrame({
        'A': ['a1', 'a2'] * 10,
        'B': ['b1', 'b2'] * 10,
    })
    rng = np.random.default_rng(0)
    calibrated = TabularConfig(
        use_jax_for_bincount=True,
        use_jax_for_generation=True,
    ).configure(domains, budget=100.0)
    result = calibrated(rng, df)
    self.assertIsInstance(result.synthetic_data, pd.DataFrame)
    self.assertListEqual(result.synthetic_data.columns.tolist(), ['A', 'B'])

  def test_mbi_callbacks_logging_configured(self):
    if not hasattr(mbi, 'callbacks') or not hasattr(
        mbi.callbacks, 'set_log_fn'
    ):
      self.skipTest(
          'mbi.callbacks.set_log_fn not supported in this mbi version'
      )

    with self.assertLogs(level='INFO') as logs:
      mbi.callbacks.log('test', 'message', sep=' | ')
    self.assertTrue(any('test | message' in output for output in logs.output))


class GroupSizeTest(parameterized.TestCase):
  """Tests the ``group_size`` parameter of ``dpsynth.calibrate`` end to end."""

  def _categorical_domains(self):
    return {
        'A': domain.CategoricalAttribute(['a', 'b', 'c']),
        'B': domain.CategoricalAttribute(['x', 'y', 'z']),
    }

  def test_calibrate_scales_noise_with_group_size(self):
    k = 5
    config = TabularConfig()
    domains = self._categorical_domains()
    cal_base = dpsynth.calibrate(
        config, domains, epsilon=1.0, delta=1e-5, group_size=1
    )
    cal_scaled = dpsynth.calibrate(
        config, domains, epsilon=1.0, delta=1e-5, group_size=k
    )
    self.assertAlmostEqual(
        cal_scaled.initializers['A'].sigma,
        k * cal_base.initializers['A'].sigma,
        places=4,
    )
    lifted = dpsynth.with_group_size(cal_scaled.initializers['A'].dp_event, k)
    self.assertAlmostEqual(
        lifted.noise_multiplier,
        cal_base.initializers['A'].dp_event.noise_multiplier,
        places=4,
    )

  def test_end_to_end_with_group_size(self):
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1.0, 5.0, 10.0]})
    domains = {
        'A': domain.CategoricalAttribute(['a', 'b', 'c']),
        'B': domain.NumericalAttribute(0.0, 10.0),
    }
    config = TabularConfig()
    calibrated = dpsynth.calibrate(
        config,
        domains,
        epsilon=10.0,
        delta=1e-5,
        group_size=3,
    )
    synthetic_df = calibrated(np.random.default_rng(0), df).synthetic_data
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])

  def test_open_set_with_group_size_greater_than_one_raises(self):
    domains = {'A': domain.OpenSetCategoricalAttribute()}
    config = TabularConfig()
    with self.assertRaises(dp_accounting.UnsupportedEventError):
      dpsynth.calibrate(config, domains, epsilon=1.0, delta=1e-5, group_size=3)

  @parameterized.named_parameters(('zero', 0), ('negative', -3))
  def test_invalid_group_size_raises(self, k):
    config = TabularConfig()
    with self.assertRaises(ValueError):
      _ = dpsynth.calibrate(
          config,
          self._categorical_domains(),
          epsilon=1.0,
          delta=1e-5,
          group_size=k,
      )


if __name__ == '__main__':
  absltest.main()
