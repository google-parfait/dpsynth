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

"""Implementation of the direct mechanism."""

from collections.abc import Sequence
import dataclasses
from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
import mbi
import numpy as np


@dataclasses.dataclass(frozen=True, kw_only=True)
class DirectConfig(api.MechanismConfig):
  """Config for the direct mechanism that measures prespecified marginals."""

  def configure(self, _=None, *, budget, delta=0):
    return Direct(
        config=self,
        gdp_budget=accounting.zcdp_to_gdp(budget),
    )

  estimator: mbi.Estimator = mbi.estimation.MirrorDescent()
  marginal_oracle: mbi.MarginalOracle | None = None
  pgm_iters: int = 5000
  prespecified_marginal_queries: list[tuple[str, ...]] = dataclasses.field(
      default_factory=list
  )

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns the prespecified marginal queries."""
    del domain  # Unused.
    return list(self.prespecified_marginal_queries)


@dataclasses.dataclass(frozen=True)
class Direct(api.CalibratedMechanism):
  """Calibrated direct mechanism instance."""

  config: DirectConfig
  gdp_budget: float

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DP event."""
    return dp_accounting.GaussianDpEvent(
        accounting.gdp_gaussian_sigma(self.gdp_budget)
    )

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: Sequence[mbi.LinearMeasurement] = (),
      constraints: Sequence[mbi.Constraint] = (),
  ) -> common.DiscreteMechanismResult:
    """Selects, measures, estimates, and generates in the compressed domain."""
    common.validate_initial_measurements(initial_measurements)
    phase_times = {}
    selected = list(self.config.prespecified_marginal_queries)
    all_cliques = [m.clique for m in initial_measurements] + list(selected)

    summary = mbi.summarize(data.domain, all_cliques)
    logging.info('[%s]:\n%s', type(self).__name__, summary)

    # Kick off async AOT compilation of the estimator while we measure.
    estimator = self.config.estimator
    oracle = estimator.marginal_oracle or self.config.marginal_oracle
    estimator = dataclasses.replace(estimator, marginal_oracle=oracle)
    pgm_future = estimator.precompile(
        data.domain,
        list(initial_measurements),
        extra_cliques=list(selected),  # pyrefly: ignore[bad-argument-type]
        constraints=constraints,
    )
    new_measurements = common.measure_marginals_with_noise(
        rng=rng,
        data=data,  # pyrefly: ignore[bad-argument-type]
        marginal_queries=selected,
        gdp_sigma=accounting.gdp_gaussian_sigma(self.gdp_budget),
    )
    measurements = list(initial_measurements) + new_measurements

    with common.timed(phase_times, 'estimation'):
      pgm_future.result()
      model = estimator.estimate(
          data.domain,
          measurements,
          iters=self.config.pgm_iters,
          callback_fn=mbi.callbacks.default(measurements, data.domain),
          constraints=constraints,
      )

    return common.DiscreteMechanismResult(
        measurements=measurements,
        model=model,
        diagnostics=common.clique_stats(model),
    )
