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

"""This mechanism independently estimates data from initial measurements."""

from collections.abc import Sequence
import dataclasses
import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import common
import mbi
import numpy as np


@dataclasses.dataclass(frozen=True, kw_only=True)
class IndependentConfig(api.MechanismConfig):
  """Independent config that doesn't select or measure any marginals."""

  pgm_iters: int = 5000

  def configure(self, _=None, *, budget, delta=0.0, workload=None):
    del budget, delta, workload
    return Independent(config=self)

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns the one-way marginals this mechanism expects to process."""
    return [(a,) for a in domain.attributes]


@dataclasses.dataclass(frozen=True)
class Independent(api.CalibratedMechanism):
  """Calibrated independent mechanism instance."""

  config: IndependentConfig

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns the one-way marginals this mechanism expects to process."""
    return self.config.supporting_cliques(domain)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns a zero-cost DP event (no new measurements)."""
    return dp_accounting.NoOpDpEvent()

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: Sequence[mbi.LinearMeasurement] = (),
      constraints: Sequence[mbi.Constraint] = (),
  ) -> common.DiscreteMechanismResult:
    """Estimates and generates from initial measurements."""
    common.validate_initial_measurements(initial_measurements)
    phase_times = {}

    # Kick off async AOT compilation of the estimator
    estimator = mbi.estimation.MirrorDescent(None)
    measurements = list(initial_measurements)
    with common.timed(phase_times, 'estimation'):
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
