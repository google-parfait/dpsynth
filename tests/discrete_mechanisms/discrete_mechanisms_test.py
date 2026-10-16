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

"""Property tests shared across all discrete mechanisms.

Note: This mechanism is not intended to be called directly. It should typically
be used within `DiscreteMechanism` or `TabularSynthesizer`. Users who call it
directly will miss out on features like 1-way measurement selection and domain
compression.
"""

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth.discrete_mechanisms import aim
from dpsynth.discrete_mechanisms import common
from dpsynth.discrete_mechanisms import direct
from dpsynth.discrete_mechanisms import discrete
from dpsynth.discrete_mechanisms import independent
from dpsynth.discrete_mechanisms import mst
from dpsynth.discrete_mechanisms import swift
import mbi
import numpy as np

_ZCDP_RHO = 10000
_WORKLOAD = [('a', 'b'), ('b', 'c'), ('a',), ('b',), ('c',)]

_MECHANISMS = {
    'AIM': aim.AIMConfig(workload=_WORKLOAD, max_rounds=4, pgm_iters=500),
    'MST': mst.MSTConfig(pgm_iters=500),
    'SWIFT': swift.SWIFTConfig(workload=_WORKLOAD, pgm_iters=500),
    'Independent': independent.IndependentConfig(),
    'Direct': direct.DirectConfig(
        prespecified_marginal_queries=_WORKLOAD, pgm_iters=500
    ),
}


def _make_skewed_dataset(rng):
  """Creates a dataset where column 'a' concentrates in 3 of 10 bins."""
  domain = mbi.Domain(['a', 'b', 'c'], [10, 4, 5])
  df = {col: rng.integers(0, domain[col], size=1000) for col in domain}
  df['a'] = rng.choice(3, size=1000)  # Only bins 0-2 populated.
  return mbi.Dataset(df, domain)


class SupportingCliquesSufficiencyTest(parameterized.TestCase):
  """Checks that supporting_cliques are sufficient for each mechanism."""

  @parameterized.named_parameters(*_MECHANISMS.items())
  def test_mechanism_runs_on_precomputed_marginals(self, mechanism):
    domain = mbi.Domain(['a', 'b', 'c', 'd'], [3, 4, 5, 6])
    data = mbi.Dataset.synthetic(domain, N=500)
    rng = np.random.default_rng(42)

    calibrated = mechanism.configure(budget=_ZCDP_RHO)
    cliques = mechanism.supporting_cliques(domain)

    precomputed = common.precompute_marginals(data, cliques)

    result = calibrated(rng, precomputed)
    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertIsNotNone(result.model)


class CompressionPropertyTest(parameterized.TestCase):
  """Tests that compression restores the original domain via synthesizer."""

  @parameterized.named_parameters(*_MECHANISMS.items())
  def test_compression_restores_domain(self, config):
    synth_config = discrete.DiscreteConfig(
        mechanism=config,
        compress_columns=True,
    )
    rng = np.random.default_rng(0)
    data = _make_skewed_dataset(rng)
    original_domain = data.domain

    result = synth_config.configure(budget=_ZCDP_RHO)(rng, data)

    self.assertEqual(result.synthetic_data.domain, original_domain)

  @parameterized.named_parameters(*_MECHANISMS.items())
  def test_compression_with_initial_measurements(self, config):
    synth_config = discrete.DiscreteConfig(
        mechanism=config,
        compress_columns=True,
    )
    rng = np.random.default_rng(0)
    data = _make_skewed_dataset(rng)
    original_domain = data.domain
    initial_measurements = common.measure_marginals_with_noise(
        rng, data, [('a',), ('b',)], gdp_sigma=1.0
    )

    mechanism = synth_config.configure(budget=_ZCDP_RHO)
    result = mechanism(rng, data, initial_measurements=initial_measurements)

    self.assertEqual(result.synthetic_data.domain, original_domain)
    self.assertNotEmpty(result.mappings)


class CalibrationTest(parameterized.TestCase):
  """Tests that calibration works across mechanisms."""

  @parameterized.named_parameters(_MECHANISMS.items())
  def test_low_epsilon_calibration(self, mechanism):
    if isinstance(mechanism, independent.IndependentConfig):
      return
    rng = np.random.default_rng(0)
    data = _make_skewed_dataset(rng)
    result = mechanism.calibrate(epsilon=1e-3, delta=1e-5)(rng, data)
    self.assertIsInstance(result, common.DiscreteMechanismResult)


if __name__ == '__main__':
  absltest.main()
