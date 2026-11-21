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

"""Implementation of the SWIFT mechanism."""

import dataclasses
import functools
from typing import Any

from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import diagnostic_info
from dpsynth.pipeline_transformations import marginals_computations
from dpsynth.pipeline_transformations import types
from dpsynth.statistical.base import accounting
from dpsynth.statistical.base import common
from dpsynth.statistical.base import swift
import mbi
import networkx as nx
import numpy as np
import pipeline_dp


@dataclasses.dataclass(frozen=True)
class SwiftParameters:
  """Parameters for SWIFT."""

  max_clique_size: float = 1e9
  max_marginal_size: float = 1e7
  pgm_iters: int = 5_000
  select_budget_frac: float = 0.1


def fit_model(
    backend: pipeline_dp.PipelineBackend,
    budget_accountant: pipeline_dp.BudgetAccountant,
    data: types.Collection[tuple[int, ...]],
    descriptor: types.Collection[dataset_descriptor.DatasetDescriptor],
    parameters: SwiftParameters,
    workload: list[types.Clique] | None = None,
    additional_output: Any | None = None,
) -> types.Collection[mbi.MarkovRandomField]:
  """Fits the model."""

  # 1. Generate workload.
  domain = backend.map(descriptor, lambda x: x.compressed_domain, 'Get domain')

  def compile_workload_fn(dom):
    return common.compiled_workload(dom, workload, parameters.max_marginal_size)  # pyrefly: ignore[bad-argument-type]

  candidates = backend.map(domain, compile_workload_fn, 'Compile workload')
  # candidates: singleton collection of dict[mbi.Clique, float]

  candidate_cliques = backend.map(
      candidates, lambda d: list(d.keys()), 'Get candidate cliques'
  )
  # candidate_cliques: singleton collection of list[mbi.Clique]

  # 2. Compute exact marginals.
  exact_marginals = marginals_computations.compute_exact_marginals(
      backend, data, candidate_cliques, domain
  )

  # 3. Compute initial errors.
  one_way_dp_marginals = backend.map(
      descriptor,
      lambda desc: list(desc.compressed_measurements()),
      'Get 1D marginals',
  )

  errors = marginals_computations.compute_errors(
      backend, one_way_dp_marginals, exact_marginals
  )
  # errors: singleton collection of dict[mbi.Clique, float]

  # 4. Select queries.
  select_spec = budget_accountant.request_budget(
      pipeline_dp.budget_accounting.MechanismType.GAUSSIAN,
      name='Swift Select Queries',
      weight=parameters.select_budget_frac,
  )
  measure_spec = budget_accountant.request_budget(
      pipeline_dp.budget_accounting.MechanismType.GAUSSIAN,
      name='Swift Measure Marginals',
      weight=1 - parameters.select_budget_frac,
  )
  errors = backend.map(
      errors,
      lambda errors_dict: _add_noise_to_errors(errors_dict, select_spec),
      'Add noise to Swift errors',
  )

  def select_queries_fn(
      errors_dict, candidates_dict, domain_obj
  ) -> tuple[dict[mbi.Clique, float], nx.Graph]:
    """Selects queries using SWIFT algorithm."""
    # `measure_spec` corresponds to the Gaussian mechanism that specifies the
    # budget for marginal measurements. Convert its noise stddev to GDP budget.
    gdp_budget = 1.0 / measure_spec.noise_standard_deviation**2
    return swift.select_queries(
        errors_dict,
        candidates_dict,
        domain_obj,
        parameters.max_clique_size,
        gdp_budget,
    )
    # return selected, jtree

  selected_and_tree = backend.map_with_side_inputs(
      errors,
      select_queries_fn,
      [candidates, domain],
      'Select Queries',
  )

  selected_queries = backend.map(
      selected_and_tree, lambda x: x[0], 'Get selected queries'
  )
  selected_cliques = backend.map(
      selected_and_tree, lambda x: list(x[0].keys()), 'Get selected queries'
  )
  jtree = backend.map(selected_and_tree, lambda x: x[1], 'Get junction tree')

  # 6. Measure selected marginals (add noise).
  def filter_selected_marginals(exact_marginal, selected):
    clique, _ = exact_marginal
    return clique in selected

  selected_exact_marginals = backend.filter_with_side_inputs(
      exact_marginals,
      filter_selected_marginals,
      [selected_queries],
      'Filter selected marginals',
  )
  # selected_exact_marginals: (Clique, np.ndarray)

  noised_selected_marginals = backend.map_with_side_inputs(
      selected_exact_marginals,
      _add_noise_fn,
      [selected_queries],
      'Add noise to selected marginals',
  )

  noised_selected_marginals = backend.to_list(
      noised_selected_marginals, 'To List'
  )

  if (
      additional_output is not None
      and additional_output.diagnostic_info is not None
  ):
    errors_singleton = backend.map(
        errors, lambda d: [(k, v) for k, v in d.items()], 'Errors to List'
    )
    additional_output.diagnostic_info = diagnostic_info.update_diagnostic_info(
        backend,
        additional_output.diagnostic_info,
        errors_singleton,
        selected_cliques,
        'Update Diagnostic Info (SWIFT)',
    )

  # 7. Estimate final model.
  def fit_model_fn(measurements_list, jtree_obj, domain_obj):
    closed_oracle = functools.partial(
        mbi.marginal_oracles.message_passing_stable, jtree=jtree_obj
    )
    return mbi.estimation.MirrorDescent(
        marginal_oracle=closed_oracle,  # pyrefly: ignore[bad-argument-type]
    ).estimate(
        domain_obj,
        measurements_list,
        iters=parameters.pgm_iters,
    )

  return backend.map_with_side_inputs(
      noised_selected_marginals,
      fit_model_fn,
      [jtree, domain],
      'Fit Final Model',
  )


def _add_noise_fn(
    clique_marginal: tuple[mbi.Clique, np.ndarray],
    selected_dict: dict[mbi.Clique, float],
) -> mbi.LinearMeasurement:
  """Adds noise to a marginal."""
  clique, marginal = clique_marginal
  budget = selected_dict[clique]

  sigma = accounting.gdp_gaussian_sigma(budget)

  spec = pipeline_dp.budget_accounting.MechanismSpec(
      mechanism_type=pipeline_dp.budget_accounting.MechanismType.GAUSSIAN,
      name='Swift Measure Marginal',
  )
  spec.set_noise_standard_deviation(sigma)
  sensitivities = pipeline_dp.dp_computations.Sensitivities(l2=1.0)
  mechanism = pipeline_dp.dp_computations.create_additive_mechanism(
      spec, sensitivities
  )
  noised_marginal = mechanism.add_noise(marginal)
  return mbi.LinearMeasurement(noised_marginal, clique, sigma)  # pyrefly: ignore[bad-argument-type]


def _add_noise_to_errors(
    errors_dict: dict[mbi.Clique, float],
    spec: pipeline_dp.budget_accounting.MechanismSpec,
) -> dict[mbi.Clique, float]:
  """Adds noise to selection errors."""
  if not errors_dict:
    return {}
  cliques = list(errors_dict)
  errors = np.array([errors_dict[clique] for clique in cliques])
  sensitivities = pipeline_dp.dp_computations.Sensitivities(
      l2=np.sqrt(len(cliques))
  )
  mechanism = pipeline_dp.dp_computations.create_additive_mechanism(
      spec, sensitivities
  )
  noised_errors = mechanism.add_noise(errors)
  return dict(zip(cliques, noised_errors))  # pyrefly: ignore[bad-argument-type]
