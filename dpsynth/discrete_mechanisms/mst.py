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

"""Implementation of the Maximum Spanning Tree mechanism."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import itertools
import typing

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
from dpsynth.local_mode import primitives
import mbi
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import DisjointSet  # pylint: disable=g-importing-member


def dp_maximum_spanning_tree(
    rng: np.random.Generator,
    weights: dict[tuple[str, str], float],
    zcdp_rho: float | None = None,
    exponential_mechanism_epsilon: float | None = None,
    initial_marginal_queries: Sequence[tuple[str, str]] = (),
    sensitivity: float = 1.0,
) -> list[tuple[str, str]]:
  """Computes an approximate maximum spanning tree with differential privacy.

  This is a differentially-private version of Kruskal's algorithm, where the
  best edge in each round is selected privately by the exponential mechanism.

  The differential privacy guarantees:
    1. zcdp_rho-zCDP if zcdp_rho is given.
    2. otherwise, it has the same privacy guarantees as the len(weights)-1
     Exponential Mechanism with parameter exponential_mechanism_epsilon.

  It is assumed that the weights are obtained from sensitivity ``sensitivity``
  functions of the data (i.e., L1 norm between true and estimated marginal,
  scaled by the maximum number of records a single user contributes).

  Args:
    rng: A numpy random number generator.
    weights: A dictionary mapping pairs of attributes to the sensitivity 1
      measure of correlation between them.
    zcdp_rho: the zCDP budget to use for this mechanism.
    exponential_mechanism_epsilon: The epsilon parameter for the exponential
      mechanism. If None, the value is computed from zcdp_rho.
    initial_marginal_queries: The list of initial attribute pairs to include in
      the tree.
    sensitivity: The sensitivity of the quality scores in ``weights``.

  Returns:
    A list of attribute pairs that constitute an approximate maximum spanning
    tree.
  """
  if (zcdp_rho is None) == (exponential_mechanism_epsilon is None):
    raise ValueError(
        'zcdp_rho or exponential_mechanism_epsilon must be set, but not both.'
    )
  tree = nx.Graph()
  attributes = set()
  for key in weights.keys():
    for attribute in key:
      attributes.add(attribute)
  tree.add_nodes_from(attributes)
  ds = DisjointSet(attributes)

  for e in initial_marginal_queries:
    tree.add_edge(*e)
    ds.merge(*e)

  candidates = list(weights.keys())
  r = len(list(nx.connected_components(tree)))
  if exponential_mechanism_epsilon is None:
    assert zcdp_rho is not None
    exponential_mechanism_epsilon = np.sqrt(8 * zcdp_rho / max(r - 1, 1))
  for _ in range(r - 1):
    candidates = [e for e in candidates if not ds.connected(*e)]
    wgts = np.array([weights[e] for e in candidates])
    idx = primitives.exponential_mechanism(
        rng,
        wgts,
        epsilon=exponential_mechanism_epsilon,
        sensitivity=sensitivity,
    )
    e = candidates[idx]
    tree.add_edge(*e)
    ds.merge(*e)

  return list(tree.edges)


def _select_two_way_marginal_queries(
    rng: np.random.Generator,
    data: mbi.Dataset | mbi.CliqueVector,
    zcdp_rho: float,
    one_way_measurements: list[mbi.LinearMeasurement],
    initial_marginal_queries: Sequence[tuple[str, ...]] = (),
    maximum_marginal_size: int = 10_000_000,
    pgm_iters: int = 2500,
) -> list[tuple[str, str]]:
  """Selects a set of two-way marginal queries with DP to form a spanning tree.

  This mechanism satisfies rho-zCDP.

  Args:
    rng: A numpy random number generator.
    data: The sensitive dataset to use to determine the quality scores of each
      two-way marginal query.
    zcdp_rho: The zCDP privacy parameter.
    one_way_measurements: The initial one-way measurements already made.
    initial_marginal_queries: The list of cliques to start with.
    maximum_marginal_size: The maximum size of a marginal query.
    pgm_iters: The maximum number of mirror descent iterations.

  Returns:
    A list of two-way marginal queries over highly correlated attributes.
  """

  independent_model = mbi.estimation.MirrorDescent().estimate(
      data.domain, list(one_way_measurements), iters=min(2500, pgm_iters)
  )
  independent_model = typing.cast(mbi.MarkovRandomField, independent_model)

  # Construct a complete graph where nodes=attributes and weight of edge
  # (a, b) is a sensitivity 1 measure of correlation between a and b.
  candidates = [
      cl
      for cl in itertools.combinations(data.domain.attributes, 2)
      if data.domain.size(cl) <= maximum_marginal_size
  ]
  logging.info('[MST]: Computing Quality Scores')
  weights = common.compute_independence_errors(
      data, independent_model, candidates  # pyrefly: ignore[bad-argument-type]
  )

  return dp_maximum_spanning_tree(
      rng=rng,
      weights=weights,  # pyrefly: ignore[bad-argument-type]
      zcdp_rho=zcdp_rho,
      initial_marginal_queries=initial_marginal_queries,  # pyrefly: ignore[bad-argument-type]
      sensitivity=1,
  )


@dataclasses.dataclass(frozen=True)
class MSTConfig(api.MechanismConfig):
  """Configuration for the maximum spanning tree mechanism.

  Details are described in the paper:
  [Winning the NIST Contest: A scalable and general approach to differentially
  private synthetic data](https://arxiv.org/abs/2108.04978)

  Attributes:
    select_budget_fraction: The fraction of the remaining budget (after one-way
      measurements) to use for selecting two-way marginal queries.
    maximum_marginal_size: The maximum size of a marginal query.
  """

  marginal_oracle: mbi.MarginalOracle | None = None
  pgm_iters: int = 5000

  select_budget_fraction: float = 1 / 2
  maximum_marginal_size: int = 10_000_000

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns all pairwise marginals within the size limit."""
    return common.supporting_cliques(
        domain,
        itertools.combinations(domain.attributes, 2),
        self.maximum_marginal_size,
    )

  def configure(self, _=None, *, budget, delta=0, workload=None):
    del delta, workload
    return MST(
        config=self,
        zcdp_rho=budget,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class MST(api.CalibratedMechanism):
  """Calibrated MST instance."""

  config: MSTConfig
  zcdp_rho: float

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns all pairwise marginals within the size limit."""
    return self.config.supporting_cliques(domain)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DP event for the MST mechanism."""
    # exponential mechanisms and (d-1) Gaussian mechanisms.
    return dp_accounting.ZCDpEvent(self.zcdp_rho)

  def _select(self, rng, data, measurements, phase_times):
    with common.timed(phase_times, 'selection'):
      return _select_two_way_marginal_queries(
          rng,
          data,
          self.zcdp_rho * self.config.select_budget_fraction,
          measurements,
          maximum_marginal_size=self.config.maximum_marginal_size,
          pgm_iters=self.config.pgm_iters,
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
    selected = self._select(rng, data, initial_measurements, phase_times)
    all_cliques = [m.clique for m in initial_measurements] + list(selected)

    logging.info(mbi.summarize(data.domain, all_cliques))

    # Kick off async AOT compilation of the estimator while we measure.
    estimator = mbi.estimation.MirrorDescent(self.config.marginal_oracle)
    pgm_future = estimator.precompile(
        data.domain,
        list(initial_measurements),
        extra_cliques=list(selected),  # pyrefly: ignore[bad-argument-type]
        constraints=constraints,
    )

    with common.timed(phase_times, 'measurement'):
      select_rho = self.zcdp_rho * self.config.select_budget_fraction
      sigma = accounting.zcdp_gaussian_sigma(self.zcdp_rho - select_rho)
      new_measurements = common.measure_marginals_with_noise(
          rng=rng,
          data=data,  # pyrefly: ignore[bad-argument-type]
          marginal_queries=selected,
          gdp_sigma=sigma,
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
      assert isinstance(model, mbi.MarkovRandomField)

    return common.DiscreteMechanismResult(
        measurements=measurements,
        model=model,
        diagnostics=common.clique_stats(model),
    )
