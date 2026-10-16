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

"""Tests for DiscreteMechanism wrapper: 1-way bootstrapping and compression."""

from unittest import mock
from absl.testing import absltest
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
from dpsynth.discrete_mechanisms import discrete
from dpsynth.discrete_mechanisms import mst
import mbi
import numpy as np

DiscreteConfig = discrete.DiscreteConfig
DiscreteMechanism = discrete.DiscreteMechanism
MSTConfig = mst.MSTConfig


class DiscreteConfigTest(absltest.TestCase):

  def test_configure_splits_budget(self):
    config = DiscreteConfig(
        mechanism=MSTConfig(pgm_iters=500),
        one_way_budget_fraction=0.25,
    )
    synth = config.configure(budget=100.0)
    self.assertIsInstance(synth, DiscreteMechanism)
    self.assertAlmostEqual(
        synth.one_way_gdp_budget,
        accounting.zcdp_to_gdp(25.0),
    )

  def test_configure_zero_one_way_fraction(self):
    config = DiscreteConfig(
        mechanism=MSTConfig(pgm_iters=500),
        one_way_budget_fraction=0.0,
    )
    synth = config.configure(budget=100.0)
    self.assertAlmostEqual(synth.one_way_gdp_budget, 0.0)


class DiscreteMechanismTest(absltest.TestCase):

  def test_full_pipeline(self):
    domain = mbi.Domain(['a', 'b', 'c'], [3, 4, 5])
    data = mbi.Dataset.synthetic(domain, N=500)
    rng = np.random.default_rng(42)

    config = DiscreteConfig(
        mechanism=MSTConfig(pgm_iters=500),
    )
    synth = config.configure(budget=10000)
    result = synth(rng, data)

    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertEqual(result.synthetic_data.domain, domain)

  def test_with_initial_measurements_skips_one_way(self):
    domain = mbi.Domain(['a', 'b', 'c'], [3, 4, 5])
    data = mbi.Dataset.synthetic(domain, N=500)
    rng = np.random.default_rng(42)

    measurements = common.measure_marginals_with_noise(
        rng, data, [('a',), ('b',), ('c',)], gdp_sigma=1.0
    )
    config = DiscreteConfig(
        mechanism=MSTConfig(pgm_iters=500),
    )
    synth = config.configure(budget=10000)
    result = synth(rng, data, initial_measurements=measurements)

    self.assertIsInstance(result, common.DiscreteMechanismResult)

  def test_compression_restores_domain(self):
    domain = mbi.Domain(['a', 'b', 'c'], [10, 4, 5])
    rng = np.random.default_rng(0)
    df = {col: rng.integers(0, domain[col], size=1000) for col in domain}
    df['a'] = rng.choice(3, size=1000)
    data = mbi.Dataset(df, domain)

    config = DiscreteConfig(
        mechanism=MSTConfig(pgm_iters=500),
        compress_columns=True,
    )
    synth = config.configure(budget=10000)
    result = synth(rng, data)

    self.assertEqual(result.synthetic_data.domain, domain)

  def test_dp_event_composes_one_way_and_inner(self):
    config = DiscreteConfig(
        mechanism=MSTConfig(pgm_iters=500),
        one_way_budget_fraction=0.25,
    )
    synth = config.configure(budget=100.0)
    event = synth.dp_event
    self.assertIsNotNone(event)

  def test_calibrate_works(self):
    config = DiscreteConfig(
        mechanism=MSTConfig(pgm_iters=500),
    )
    domain = mbi.Domain(['a', 'b'], [3, 4])
    data = mbi.Dataset.synthetic(domain, N=200)
    rng = np.random.default_rng(0)

    calibrated = config.calibrate(epsilon=1.0, delta=1e-5)
    result = calibrated(rng, data)
    self.assertIsInstance(result, common.DiscreteMechanismResult)

  def test_converts_dataset_to_clique_vector(self):
    config = DiscreteConfig(mechanism=MSTConfig(pgm_iters=500))
    synth = config.configure(budget=100.0)
    domain = mbi.Domain(['a', 'b'], [3, 4])
    data = mbi.Dataset.synthetic(domain, N=20)
    with mock.patch.object(
        mst.MST, '__call__', side_effect=synth.base_mechanism.__call__
    ) as mock_call:
      synth(np.random.default_rng(0), data)
      self.assertIsInstance(mock_call.call_args.args[1], mbi.CliqueVector)

  def test_use_jax_for_generation(self):
    config = DiscreteConfig(
        mechanism=MSTConfig(pgm_iters=100),
        use_jax_for_generation=True,
    )
    domain = mbi.Domain(['a', 'b'], [3, 4])
    data = mbi.Dataset.synthetic(domain, N=200)
    rng = np.random.default_rng(0)
    synth = config.configure(budget=100.0)
    result = synth(rng, data)
    self.assertEqual(result.synthetic_data.domain, domain)
    self.assertEqual(result.synthetic_data.records, 200)

  def test_use_jax_for_bincount_and_generation(self):
    config = DiscreteConfig(
        mechanism=MSTConfig(pgm_iters=100),
        use_jax_for_bincount=True,
        use_jax_for_generation=True,
    )
    domain = mbi.Domain(['a', 'b'], [3, 4])
    data = mbi.Dataset.synthetic(domain, N=200)
    rng = np.random.default_rng(0)
    synth = config.configure(budget=100.0)
    result = synth(rng, data)
    self.assertEqual(result.synthetic_data.domain, domain)
    self.assertEqual(result.synthetic_data.records, 200)


if __name__ == '__main__':
  absltest.main()
