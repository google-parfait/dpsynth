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

"""Privacy accounting helpers for zCDP, GDP, and (epsilon, delta)-DP.

The zCDP <> ADP conversion follows Section 2.3 of "The Discrete Gaussian for
Differential Privacy" (https://arxiv.org/abs/2004.00010) and was adapted from
https://github.com/IBM/discrete-gaussian-differential-privacy/.

The GDP helpers follow "Gaussian Differential Privacy"
(https://arxiv.org/abs/1905.02383).

Note: this file is subject to be deprecated in the near future, in favor of
using the dp_accounting library.
"""

import math

import scipy.special
import scipy.stats


def zcdp_delta(rho: float, eps: float) -> float:
  """Return the minimum value of delta such that rho-zCDP implies (epsilon, delta)-DP."""
  assert rho >= 0
  assert eps >= 0
  if rho == 0:
    return 0

  amin = 1.01
  amax = (eps + 1) / (2 * rho) + 2
  alpha = math.nan
  while amax - amin > 1e-10:
    alpha = (amin + amax) / 2
    derivative = (2 * alpha - 1) * rho - eps + math.log1p(-1.0 / alpha)
    if derivative < 0:
      amin = alpha
    else:
      amax = alpha
  delta = math.exp(
      (alpha - 1) * (alpha * rho - eps) + alpha * math.log1p(-1 / alpha)
  ) / (alpha - 1.0)
  return min(delta, 1.0)


def zcdp_eps(rho: float, delta: float) -> float:
  """Return the minimum value of epsilon such that rho-zCDP implies (epsilon, delta)-DP."""
  assert rho >= 0
  assert delta > 0
  if delta >= 1 or rho == 0:
    return 0.0
  epsmin = 0.0
  epsmax = rho + 2 * math.sqrt(rho * math.log(1 / delta))
  while epsmax - epsmin > 1e-10:
    eps = (epsmin + epsmax) / 2
    if zcdp_delta(rho, eps) <= delta:
      epsmax = eps
    else:
      epsmin = eps
  return epsmax


def zcdp_rho(eps: float, delta: float) -> float:
  """Return the maximum value of rho such that rho-zCDP implies (epsilon, delta)-DP."""
  assert eps >= 0
  assert delta > 0
  if delta >= 1:
    return 0.0
  rhomin = 0.0
  rhomax = eps + 1
  while rhomax - rhomin > 1e-10:
    rho = (rhomin + rhomax) / 2
    if zcdp_delta(rho, eps) <= delta:
      rhomin = rho
    else:
      rhomax = rho
  return rhomin


def zcdp_gaussian_sigma(rho: float) -> float:
  """Minimum sigma such that the Gaussian mechanism satisfies rho-zCDP."""
  # rho = 0.5 / sigma^2
  return math.sqrt(0.5 / rho)


def zcdp_exponential_nu(rho: float) -> float:
  """Maximum nu such that the exponential mechanism satisfies rho-zCDP."""
  # rho = 1/8 * nu^2
  return math.sqrt(8 * rho)


def zcdp_bounded_range_optimal_rho(nu: float) -> float:
  """Return the tight zCDP parameter rho of a bounded range mechanism.

  A mechanism with bounded range parameter nu (e.g. the exponential mechanism
  with parameter nu) satisfies rho-zCDP for
  rho = nu / (exp(nu) - 1) + log((exp(nu) - 1) / nu) - 1.
  This improves on the generic bound rho = nu^2 / 8 used by
  `zcdp_exponential_nu`. See https://arxiv.org/abs/2510.25746.

  Args:
    nu: The bounded range parameter of the mechanism.
  """
  assert nu >= 0
  if nu == 0:
    return 0.0
  expm1_nu = math.expm1(nu)
  return nu / expm1_nu + math.log(expm1_nu / nu) - 1.0


def gdp_gaussian_sigma(budget: float) -> float:
  """Return the Gaussian mechanism sigma that satisfies sqrt(budget)-GDP."""
  return math.sqrt(1.0 / budget)


def gdp_budget_bounded_range(nu: float) -> float:
  """Return the squared GDP parameter mu^2 of a bounded range mechanism.

  A mechanism with bounded range parameter nu satisfies mu-GDP for
  mu = -2 * Phi^{-1}(1 / (exp(nu / 2) + 1)).

  Args:
    nu: The bounded range parameter of the mechanism.
  """
  assert nu >= 0
  mu = -2.0 * scipy.stats.norm.ppf(1.0 / (math.exp(nu / 2.0) + 1.0))
  return mu**2


def gdp_bounded_range_nu(budget: float) -> float:
  """Return the largest bounded range parameter nu that satisfies sqrt(budget)-GDP.

  This is the inverse of `gdp_budget_bounded_range`, given by nu = 2 * L(mu)
  with L(t) = log(Phi(t / 2) / Phi(-t / 2)).

  Args:
    musq: The GDP budget mu^2 of the mechanism.
  """
  assert budget >= 0
  mu = math.sqrt(budget)
  return 2.0 * (scipy.special.log_ndtr(mu / 2.0) - scipy.special.log_ndtr(-mu / 2.0))


def gdp_exponential_nu(budget: float) -> float:
  """Return the exponential mechanism nu that satisfies sqrt(budget)-GDP."""
  return gdp_bounded_range_nu(budget)


def gdp_delta(mu: float, eps: float) -> float:
  """Return the minimum delta such that mu-GDP implies (epsilon, delta)-DP.

  See Dong, Roth, and Su, "Gaussian Differential Privacy"
  (https://arxiv.org/abs/1905.02383).

  Args:
    mu: The GDP parameter.
    eps: The epsilon of the target (epsilon, delta)-DP guarantee.
  """
  assert mu >= 0
  assert eps >= 0
  if mu == 0:
    return 0.0
  return scipy.stats.norm.cdf(-eps / mu + mu / 2) - math.exp(eps) * (
      scipy.stats.norm.cdf(-eps / mu - mu / 2)
  )


def zcdp_to_gdp(rho: float) -> float:
  """Return the largest GDP budget (mu^2) such that mu-GDP implies rho-zCDP."""
  # rho = 0.5 / sigma^2 = 0.5 * budget
  return 2 * rho
