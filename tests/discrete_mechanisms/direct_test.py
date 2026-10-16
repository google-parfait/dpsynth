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
from dpsynth.discrete_mechanisms import direct
import mbi
import numpy as np


class DirectTest(absltest.TestCase):

  def test_fits_one_way_marginals(self):
    data = mbi.Dataset.synthetic(mbi.Domain(['a', 'b', 'c'], [3, 4, 5]), N=1000)

    prespecified_queries = [('a', 'b'), ('a', 'c'), ('b', 'c')]
    config = direct.DirectConfig(
        prespecified_marginal_queries=prespecified_queries,
        pgm_iters=500,
    )
    result = config.configure(budget=10000)(np.random.default_rng(0), data)

    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertLen(result.measurements, len(prespecified_queries))
    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = result.model.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=1)

  def test_calibrate_gives_expected_gdp_budget(self):
    calibrated = direct.DirectConfig().calibrate(epsilon=1.0, delta=1e-5)
    self.assertAlmostEqual(calibrated.gdp_budget, 0.07185134, places=6)

  def test_custom_estimator(self):
    data = mbi.Dataset.synthetic(mbi.Domain(['a', 'b', 'c'], [3, 4, 5]), N=1000)
    prespecified_queries = [('a', 'b'), ('a', 'c'), ('b', 'c')]
    config = direct.DirectConfig(
        prespecified_marginal_queries=prespecified_queries,
        estimator=mbi.estimation.InteriorGradient(),
        pgm_iters=500,
    )
    result = config.configure(budget=10000)(np.random.default_rng(0), data)

    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = result.model.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=1)

  def test_marginal_oracle_propagated_to_estimator(self):
    data = mbi.Dataset.synthetic(mbi.Domain(['a', 'b'], [2, 3]), N=100)
    calls = []

    def tracking_oracle(potentials, total=1, constraints=()):
      calls.append(total)
      return mbi.marginal_oracles.brute_force_marginals(
          potentials, total, constraints=constraints
      )

    config = direct.DirectConfig(
        prespecified_marginal_queries=[('a', 'b')],
        estimator=mbi.estimation.InteriorGradient(),
        marginal_oracle=tracking_oracle,
        pgm_iters=10,
    )
    config.configure(budget=100)(np.random.default_rng(0), data)
    self.assertNotEmpty(calls)


if __name__ == '__main__':
  absltest.main()
