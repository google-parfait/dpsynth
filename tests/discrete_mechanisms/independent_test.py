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

from absl.testing import absltest
from dpsynth.discrete_mechanisms import common
from dpsynth.discrete_mechanisms import independent
import mbi
import numpy as np


class IndependentTest(absltest.TestCase):

  def test_fits_one_way_marginals(self):
    """Independent with externally-supplied 1-ways should recover marginals."""
    data = mbi.Dataset.synthetic(mbi.Domain(['a', 'b', 'c'], [3, 4, 5]), N=1000)

    config = independent.IndependentConfig()

    # One-way marginals are now supplied externally by the synthesizer layer.
    # Simulate that by constructing them here with near-zero noise.
    initial_measurements = []
    for col in data.domain:
      true_marginal = data.project([col]).datavector()
      initial_measurements.append(
          mbi.LinearMeasurement(
              noisy_measurement=true_marginal,
              clique=(col,),
              stddev=1e-6,
              query=mbi.DatavectorQuery(use_for_total_estimation=True),
          )
      )

    result = config.configure(budget=10000)(
        np.random.default_rng(0),
        data,
        initial_measurements=initial_measurements,
    )

    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertLen(result.measurements, len(data.domain))
    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = result.model.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=0.1)

  def test_skips_duplicate_cliques_from_initial_measurements(self):
    """Independent should not re-measure pre-measured cliques."""
    data = mbi.Dataset.synthetic(mbi.Domain(['a', 'b', 'c'], [3, 4, 5]), N=100)
    # Pre-measure column 'a'.
    marginal_a = data.project(('a',)).datavector()
    initial = [mbi.LinearMeasurement(marginal_a, ('a',), stddev=1.0)]

    config = independent.IndependentConfig()
    # This should not raise 'Cliques must be unique'.
    model = config.configure(budget=100.0)(
        np.random.default_rng(0), data, initial_measurements=initial
    )
    self.assertIsNotNone(model)


if __name__ == '__main__':
  absltest.main()
