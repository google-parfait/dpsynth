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

"""Implementation of the Adaptive+Iterative Mechanism (AIM)."""

from collections.abc import Iterable, Mapping
from collections.abc import Sequence
import dataclasses
from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
from dpsynth.local_mode import primitives
import jax.numpy as jnp
import mbi
import mbi.junction_tree
import numpy as np


def _filter_candidates(
    candidates: Mapping[mbi.Clique, float],
    model: mbi.MarkovRandomField,
    size_limit: float,
) -> Mapping[mbi.Clique, float]:
  """Filters the given candidates that lead to tractable graphical models.

  Args:
    candidates: The candidate marginal queries.
    model: The current graphical model.
    size_limit: The size limit in megabytes for the new graphical model, if a
      given candidate is selected.

  Returns:
    A collection of new candidates that pass the size_limit filter.
  """
  ans = {}
  free_cliques = common.downward_closure(model.cliques)
  domain = model.domain
  for cl in candidates:
    cliques = [*model.cliques, cl]
    cond1 = (
        mbi.junction_tree.hypothetical_model_size(domain, cliques) <= size_limit
    )
    cond2 = cl in free_cliques
    if cond1 or cond2:
      ans[cl] = candidates[cl]
  return ans


def _worst_approximated(
    rng: np.random.Generator,
    candidates: Mapping[mbi.Clique, float],
    data: mbi.Dataset | mbi.CliqueVector,
    estimates: mbi.CliqueVector,
    eps: float,
    sigma: float,
    domain: mbi.Domain,
) -> mbi.Clique:
  """Returns the worst approximated candidate in the given candidates."""
  errors = {}
  for cl in candidates:
    wgt = candidates[cl]
    diff = data.project(cl).datavector() - estimates[cl].datavector()
    bias = jnp.sqrt(2 / jnp.pi) * sigma * domain.size(cl)
    errors[cl] = wgt * (jnp.linalg.norm(diff, ord=1) - bias)

  max_sensitivity = max(
      candidates.values()
  )  # if all weights are 0, could be a problem
  keys, values = list(errors.keys()), np.array(list(errors.values()))
  idx = primitives.exponential_mechanism(
      rng, values, epsilon=eps, sensitivity=max_sensitivity, monotonic=False
  )
  return keys[idx]


@dataclasses.dataclass(frozen=True)
class AIMConfig(api.MechanismConfig):
  """Configuration for the AIM mechanism.

  Details are described in the paper:
  [AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic
  Data](https://arxiv.org/abs/2201.12677). This mechanism is a competitive
  algorithm within the broader SELECT-MEASURE-GENERATE paradigm. It is an
  MWEM-style algorithm (Multiplicative Weights + Exponential Mechanism), that
  iteratively improves the estimate of the data distribution by selecting
  marginal queries that are poorly approximated by the current model. It is a
  scalable algorithm that can handle high-dimensional datasets, but it can be
  time consuming to run (hours). The runtime/utility trade-off can be controlled
  by the max_model_size parameter. For quick experimentation, we recommend
  setting max_model_size = 1, for production use cases, we recommend setting
  max_model_size >= 80.

  Attributes:
    workload: A collection of marginal queries (and weights) the synthetic data
      should be tailored to.
    max_rounds: The maximum number of rounds to run the mechanism.
    max_model_size: The maximum size of the graphical model in megabytes.
      Controls the utility/runtime trade-off.
    max_marginal_size: The maximum size of a marginal query to consider.
    degree: The maximum degree of the automatically constructed workload.
      Ignored when an explicit workload is provided.
    anneal_factor: The factor by which to anneal the privacy.
    select_budget_fraction: The fraction of the total budget to use for
      selecting two-way marginal queries.
  """

  workload: Mapping[mbi.Clique, float] | Iterable[mbi.Clique] | None = None
  max_rounds: int | None = None
  max_model_size: int = 80
  max_marginal_size: float = 1e6
  degree: int = 3
  anneal_factor: float = 4.0
  select_budget_fraction: float = 0.1
  pgm_iters: int = 1000
  marginal_oracle: mbi.MarginalOracle | None = None

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns the workload cliques filtered by max_marginal_size."""
    return common.supporting_cliques(
        domain, self.workload, self.max_marginal_size, degree=self.degree
    )

  def configure(self, _=None, *, budget, delta=0):
    return AIM(
        config=self,
        zcdp_rho=budget,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class AIM(api.CalibratedMechanism):
  """Calibrated AIM instance."""

  config: AIMConfig
  zcdp_rho: float

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DP event for the AIM mechanism."""
    return dp_accounting.ZCDpEvent(self.zcdp_rho)

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: Sequence[mbi.LinearMeasurement] | None = None,
      constraints: Sequence[mbi.Constraint] = (),
  ) -> common.DiscreteMechanismResult:
    common.validate_initial_measurements(initial_measurements)
    measurements = list(initial_measurements) if initial_measurements else []
    phase_times = {}
    logging.info('[AIM]: Starting Mechanism.')
    zcdp_rho = self.zcdp_rho
    terminate = False
    rho_remaining = self.zcdp_rho
    max_rounds = self.config.max_rounds or 16 * len(data.domain)
    rho_per_round = self.zcdp_rho / max_rounds

    #########################################################################
    # Compile workload into candidate measurements.                         #
    #########################################################################
    candidates = common.compiled_workload(
        data.domain,
        self.config.workload,
        self.config.max_marginal_size,
        degree=self.config.degree,
    )

    estimator = mbi.estimation.MirrorDescent(self.config.marginal_oracle)
    model = estimator.estimate(
        data.domain,
        measurements,
        iters=self.config.pgm_iters,
        constraints=constraints,
    )
    assert isinstance(model, mbi.MarkovRandomField)

    t = 0
    while not terminate:
      t += 1
      if rho_remaining < 2 * rho_per_round:
        logging.info('[AIM] Final round, Using all remaining privacy budget.')
        rho_per_round = rho_remaining
        terminate = True

      ########################################################################
      # Select a marginal query worst approximated by the current model.     #
      ########################################################################
      with common.timed(phase_times, 'selection'):
        rho_remaining -= rho_per_round
        fraction = self.config.select_budget_fraction
        sigma = accounting.zcdp_gaussian_sigma((1 - fraction) * rho_per_round)
        epsilon = accounting.zcdp_exponential_eps(fraction * rho_per_round)
        size_limit = (
            self.config.max_model_size * (zcdp_rho - rho_remaining) / zcdp_rho
        )
        small_candidates = _filter_candidates(candidates, model, size_limit)

        estimates = mbi.marginal_oracles.bulk_variable_elimination(
            model.potentials, list(small_candidates), total=model.total  # pyrefly: ignore[bad-argument-type]
        )
        marginal_query = _worst_approximated(
            rng,
            small_candidates,
            data,
            estimates,
            epsilon,
            sigma,
            data.domain,
        )

      summary = mbi.summarize(
          data.domain, [m.clique for m in measurements] + [marginal_query]
      )
      logging.info(
          '[AIM] Round %d, Budget used: %.4f, Measuring: %s, Candidates: %d,'
          ' cliques: %d, treewidth: %d, memory: %d bytes',
          t,
          (zcdp_rho - rho_remaining) / zcdp_rho,
          marginal_query,
          len(small_candidates),
          summary.num_cliques,
          summary.treewidth,
          summary.memory_bytes,
      )

      ######################################################################
      # Measure the marginal query privately using the Gaussian mechanism. #
      ######################################################################
      with common.timed(phase_times, 'measurement'):
        measurement = common.measure_marginals_with_noise(
            rng,
            data,  # pyrefly: ignore[bad-argument-type]
            [marginal_query],  # pyrefly: ignore[bad-argument-type]
            sigma,
        )[0]
        measurements.append(measurement)
        old_estimate = model.project(marginal_query).datavector()

      #####################################################
      # Estimate the data distribution using Private-PGM. #
      #####################################################
      with common.timed(phase_times, 'estimation'):
        callback_fn = mbi.callbacks.default(measurements, data.domain)
        model = estimator.estimate(
            data.domain,
            measurements,
            warm_start=model,
            iters=self.config.pgm_iters,
            callback_fn=callback_fn,
            constraints=constraints,
        )
        assert isinstance(model, mbi.MarkovRandomField)

      new_estimate = model.project(marginal_query).datavector()

      ##########################################
      # Anneal epsilon and sigma if necessary. #
      ##########################################
      threshold = sigma * np.sqrt(2 / np.pi) * data.domain.size(marginal_query)
      if np.linalg.norm(new_estimate - old_estimate, ord=1) <= threshold:
        # No useful information at this noise level, increase budget per round.
        rho_per_round *= self.config.anneal_factor
        fraction = self.config.select_budget_fraction
        sigma = accounting.zcdp_gaussian_sigma((1 - fraction) * rho_per_round)
        logging.info('[AIM] Reducing sigma: %.1f', sigma)

    return common.DiscreteMechanismResult(
        measurements=measurements,
        model=model,
        diagnostics=common.clique_stats(model),
    )
