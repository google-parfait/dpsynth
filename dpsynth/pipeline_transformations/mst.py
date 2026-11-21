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

"""Pipeline transformations for MST mechanism."""

import itertools
from typing import Any, TypeAlias

from dpsynth.pipeline_transformations import diagnostic_info
from dpsynth.pipeline_transformations import marginals_computations
from dpsynth.pipeline_transformations import model
from dpsynth.pipeline_transformations import types
from dpsynth.statistical.base import mst
import mbi
import numpy as np
import pipeline_dp

Edge: TypeAlias = tuple[int, int]


# One-way Marginals are assumed to be fit in memory, so they are passed as a
# singleton collection of lists.
OneMayDPMarginals = types.Collection[list[mbi.LinearMeasurement]]

# Two-way marginals are not assumed to be fit in memory, so they are passed as a
# collection, where each element corresponds to a pair of attributes.
TwoWayMarginals = types.Collection[tuple[Edge, np.ndarray]]

# MST is a singleton collection of a list of edges.
MST = types.Collection[list[Edge]]


def fit_model(
    backend: pipeline_dp.PipelineBackend,
    budget_accountant: pipeline_dp.BudgetAccountant,
    dp_engine: pipeline_dp.DPEngine,
    num_attributes: int,
    compressed_data: types.Collection[tuple[int, ...]],
    compressed_one_way_marginals: OneMayDPMarginals,
    compressed_domain: types.Collection[mbi.Domain],
    additional_output: Any | None = None,
) -> types.Collection[mbi.MarkovRandomField]:
  """Fit model using the MST mechanism."""
  # 1. Compute 2-way marginals.
  marginal_queries = list(itertools.combinations(range(num_attributes), 2))
  marginal_queries = backend.to_collection(
      [marginal_queries], compressed_data, "ToPCollection"
  )
  two_way_marginals = marginals_computations.compute_exact_marginals(
      backend, compressed_data, marginal_queries, compressed_domain
  )  # (clique, np.ndarray)

  # 2. Select the MST.
  spanning_tree = _select_dp_maximum_spanning_tree(
      backend,
      budget_accountant,
      compressed_one_way_marginals,
      two_way_marginals,
      num_attributes,
      additional_output,
  )  # singleton collection of (tuple[int, int],)

  # 3. Filter and add noise to the 2-way marginals to the MST.
  mst_marginals = _filter_marginals_with_mst(
      backend, two_way_marginals, spanning_tree
  )  # (tuple[int, int], np.ndarray)

  mst_dp_marginals = marginals_computations.add_dp_noise_to_marginals(
      backend,
      dp_engine,
      mst_marginals,
      num_attributes,
  )  # (mbi.LinearMeasurement,)

  # 4. Fit the model.
  marginals = marginals_computations.combine_marginals(
      backend, compressed_one_way_marginals, mst_dp_marginals
  )  # singleton collection of (mbi.LinearMeasurement,...)

  return model.fit_model(
      backend,
      marginals,
      compressed_domain,
  )  # singleton collection of (mbi.MarkovRandomField,)


def _filter_marginals_with_mst(
    backend: pipeline_dp.PipelineBackend,
    marginals: TwoWayMarginals,
    spanning_tree: MST,
) -> TwoWayMarginals:
  """Returns the marginals which are in the MST."""

  mst_set = backend.map(spanning_tree, set, "MST to Set")

  def filter_by_mst_fn(
      two_way_marginal: tuple[Edge, np.ndarray], tree_edges: set[Edge]
  ) -> bool:
    attr1, attr2 = two_way_marginal[0]
    return (attr1, attr2) in tree_edges or (attr2, attr1) in tree_edges

  return backend.filter_with_side_inputs(
      marginals, filter_by_mst_fn, [mst_set], "Filter By MST"
  )


def _select_dp_maximum_spanning_tree(
    backend: pipeline_dp.PipelineBackend,
    accountant: pipeline_dp.BudgetAccountant,
    one_way_dp_marginals: OneMayDPMarginals,
    two_way_marginals: TwoWayMarginals,
    num_attributes: int,
    additional_output: Any | None = None,
) -> MST:
  """Returns the edges of the MST.

  The output is DP, the privacy budget for all DP operations are requested from
  the accountant.

  Args:
    backend: The backend to use for running the pipeline operations.
    accountant: The accountant to use for requesting budget.
    one_way_dp_marginals: One-way marginals computed with DP for each attribute.
    two_way_marginals: Two-way DP marginals for each pair of attributes,
      computed w/o DP.
    num_attributes: The number of attributes in the dataset.
    additional_output: Additional output to populate diagnostic info.

  Returns:
    A singleton collection of a list of edges of the MST.
  """
  weights = marginals_computations.compute_errors(
      backend, one_way_dp_marginals, two_way_marginals
  )

  # Laplace mechanism is used as dominating mechanism for Exponential mechanism.
  # is supported by dp_accounting library.
  budget = accountant.request_budget(
      pipeline_dp.budget_accounting.MechanismType.LAPLACE,
      count=num_attributes - 1,
      name="MST Exponential Mechanism",
  )

  def get_mst_fn(weights: dict[Edge, float]) -> list[Edge]:
    # dp_maximum_spanning_tree takes a dictionary with keys as strings, so we
    # convert the keys to strings and back.
    weights_str = {(str(k[0]), str(k[1])): v for k, v in weights.items()}

    epsilon = _get_eps_from_laplace_noise_std(budget.noise_standard_deviation)
    spanning_tree = mst.dp_maximum_spanning_tree(
        np.random.default_rng(),
        weights_str,
        exponential_mechanism_epsilon=epsilon,
    )

    def convert_edge(a: str, b: str) -> tuple[int, int]:
      a_int, b_int = int(a), int(b)
      # Sort the vertices to make sure that the output is deterministic.
      return (a_int, b_int) if a_int < b_int else (b_int, a_int)

    return [convert_edge(a, b) for a, b in spanning_tree]

  spanning_tree = backend.map(weights, get_mst_fn, "Get MST")

  if (
      additional_output is not None
      and additional_output.diagnostic_info is not None
  ):
    errors_singleton = backend.map(
        weights, lambda d: [(k, v) for k, v in d.items()], "Weights to List"
    )
    additional_output.diagnostic_info = diagnostic_info.update_diagnostic_info(
        backend,
        additional_output.diagnostic_info,
        errors_singleton,
        spanning_tree,
        "Update Diagnostic Info (MST)",
    )

  return spanning_tree


def _get_eps_from_laplace_noise_std(laplace_noise_std: float) -> float:
  """Returns the epsilon of Laplace mechanism from the standard deviation."""
  # We need this because we use Laplace mechanism as dominating mechanism for
  # Exponential mechanism. PLD returns the standard deviation of the Laplace
  # mechanism, but for Exponential mechanism we need the epsilon. So get
  # epsilon for the Laplace mechanism.
  laplace_b = laplace_noise_std / np.sqrt(2)
  return 1 / laplace_b
