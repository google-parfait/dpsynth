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
from absl.testing import parameterized
from dpsynth.discrete_mechanisms import accounting
import numpy as np


class BoundedRangeAccountingTest(parameterized.TestCase):
  """Tests comparing the GDP and zCDP guarantees of a bounded range mechanism."""

  @parameterized.named_parameters(
      ("small_eps", 0.01),
      ("moderate_eps", 1.0),
      ("large_eps", 2.0),
      ("very_large_eps", 10.0),
  )
  def test_zcdp_bounded_range_optimal_rho_is_at_most_generic_bound(self, exp_eps):
    rho = accounting.zcdp_bounded_range_optimal_rho(exp_eps)
    self.assertGreater(rho, 0.0)
    self.assertLessEqual(rho, exp_eps**2 / 8)

  @parameterized.named_parameters(
      ("small_eps", 0.01),
      ("moderate_eps", 1.0),
      ("large_eps", 2.0),
      ("very_large_eps", 10.0),
  )
  def test_tv_matches_gdp_and_is_tighter_than_zcdp(self, exp_eps):
    # Tight TV bound for bounded range.
    eta = np.tanh(exp_eps / 4)
    rho = accounting.zcdp_bounded_range_optimal_rho(exp_eps)
    mu = np.sqrt(accounting.gdp_budget_bounded_range(exp_eps))

    delta_zcdp = accounting.zcdp_delta(rho, eps=0.0)
    delta_gdp = accounting.gdp_delta(mu, eps=0.0)

    np.testing.assert_allclose(delta_gdp, eta, rtol=1e-6)
    self.assertGreater(delta_zcdp, eta)

  @parameterized.named_parameters(
      ("small_eps", 0.01),
      ("moderate_eps", 1.0),
      ("large_eps", 2.0),
      ("very_large_eps", 10.0),
  )
  def test_zcdp_is_tighter_than_gdp_for_bounded_range_at_edges(self, exp_eps):
    # The zCDP analysis of a single bounded range mechanism is tighter than the
    # GDP analysis for low delta. Use eps = 2 * exp_eps so that both deltas are
    # small but still representable.
    eps = 2 * exp_eps
    rho = accounting.zcdp_bounded_range_optimal_rho(exp_eps)
    mu = np.sqrt(accounting.gdp_budget_bounded_range(exp_eps))

    delta_zcdp = accounting.zcdp_delta(rho, eps=eps)
    delta_gdp = accounting.gdp_delta(mu, eps=eps)

    self.assertGreater(delta_zcdp, 0.0)
    self.assertLess(delta_zcdp, delta_gdp)


if __name__ == "__main__":
  absltest.main()
