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

import copy
import dataclasses
import itertools
from typing import Any, TypeAlias, TypeVar

from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.local_mode import primitives
from dpsynth.pipeline_transformations import diagnostic_info
from dpsynth.pipeline_transformations import independent_mechanism
from dpsynth.pipeline_transformations import marginals_computations
from dpsynth.pipeline_transformations import types
import jax
import jax.numpy as jnp
import mbi
import numpy as np
import pipeline_dp

Clique: TypeAlias = tuple[int, ...]
MarginalQuery: TypeAlias = tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class AIMParameters:
  """Parameters for AIM."""

  # attributes.
  rounds: int = 100
  pgm_iters: int = 1000
  max_model_size: int = 500


def fit_model(
    backend: pipeline_dp.PipelineBackend,
    budget_accountant: pipeline_dp.BudgetAccountant,
    data: types.Collection[tuple[int, ...]],
    descriptor: types.Collection[dataset_descriptor.DatasetDescriptor],
    parameters: AIMParameters,
    workload: list[MarginalQuery] | None = None,
    additional_output: Any | None = None,
) -> types.Collection[mbi.MarkovRandomField]:
  """Fits the model."""

  # Generate workload.
  domain = backend.map(descriptor, lambda x: x.compressed_domain, 'Get domain')
  if workload is None:
    workload = backend.map(
        domain,
        _generate_workload,
        'Generate workload',
    )
  else:
    workload = backend.to_collection([workload], data, 'Create Workload')
  # workload: singleton collection of list of marginals.

  marginals = marginals_computations.compute_exact_marginals(
      backend, data, workload, domain
  )
  # (clique, np.ndarray)

  exponential_spec, gaussian_spec = _get_dp_parameters(
      budget_accountant, parameters.rounds
  )

  # Extract 1d LinearMeasurements and create the initial model.
  measurements = backend.map(
      descriptor,
      lambda x: list(x.compressed_measurements()),
      'Extract Measurements',
  )
  # measurements: singleton of list[LinearMeasurements]

  model = independent_mechanism.fit_model(backend, descriptor)
  # model: singleton (mbi.MarkovRandomField,)

  selected_marginals = backend.to_collection([[]], measurements, 'Create empty')
  # singleton (list[Clique])

  for _ in range(parameters.rounds):
    new_measurement, selected_marginals = _find_worst_approximated_marginal(
        backend,
        marginals,
        model,
        selected_marginals,
        gaussian_spec,
        exponential_spec,
        parameters.max_model_size,
        additional_output,
    )
    # new_measurement: singleton (Clique, np.ndarray)

    measurements = backend.map_with_side_inputs(
        measurements,
        _extend_list_fn,
        [new_measurement],
        'Extend Measurements with new noised marginal',
    )
    # singleton (list[LinearMeasurements])

    model = backend.map_with_side_inputs(
        model,
        lambda model, measurements: _create_new_model(
            model, measurements, parameters.pgm_iters
        ),
        [measurements],
        'create new model',
        resource_hints={'worker_cpu': 32},
    )

  return model


def _generate_workload(domain: mbi.Domain) -> list[MarginalQuery]:
  def tuple_to_int(t: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(int(x) for x in t)

  return [
      tuple_to_int(cl)  # pyrefly: ignore[bad-argument-type]
      for cl in itertools.combinations(domain, 3)
      if domain.size(cl) <= 1e6
  ]


def _get_dp_parameters(
    accountant: pipeline_dp.BudgetAccountant, rounds: int
) -> tuple[
    pipeline_dp.budget_accounting.MechanismSpec,
    pipeline_dp.budget_accounting.MechanismSpec,
]:
  """Returns the DP parameters for the given budget and number of rounds."""
  # Laplace mechanism is used as a dominating mechanism for the exponential
  # mechanism.
  exponential_budget = accountant.request_budget(
      pipeline_dp.budget_accounting.MechanismType.LAPLACE,
      count=rounds,
      name='AIM Exponential Mechanism',
  )
  gaussian_budget = accountant.request_budget(
      pipeline_dp.budget_accounting.MechanismType.GAUSSIAN,
      count=rounds,
      name='AIM Gaussian Mechanism',
  )
  return exponential_budget, gaussian_budget


def _find_worst_approximated_marginal(
    backend: pipeline_dp.PipelineBackend,
    marginals: types.Collection[tuple[Clique, np.ndarray]],
    model: types.Collection[mbi.MarkovRandomField],
    selected_marginals: types.Collection[list[mbi.LinearMeasurement]],
    gaussian_spec: pipeline_dp.budget_accounting.MechanismSpec,
    exponential_spec: pipeline_dp.budget_accounting.MechanismSpec,
    max_model_size: int,
    additional_output: Any | None = None,
) -> tuple[
    types.Collection[mbi.LinearMeasurement],
    types.Collection[list[Clique]],
]:
  """Finds the worst approximated marginal by model.

  Args:
    backend: The backend to perform pipeline operations.
    marginals: The marginals candidates to consider.
    model: The current model.
    selected_marginals: The marginals already selected.
    gaussian_spec: The Gaussian mechanism spec.
    exponential_spec: The exponential mechanism spec.
    max_model_size: The maximum model size.
    additional_output: Additional output to populate diagnostic info.

  Returns:
    A tuple of the worst approximated marginal and the updated selected
    marginals.
  """

  def filter_fn(m: tuple[Clique, np.ndarray], model, selected_marginals):
    clique, _ = m
    if clique in selected_marginals:
      return False
    new_cliques = [*model.cliques, clique]
    return (
        mbi.junction_tree.hypothetical_model_size(model.domain, new_cliques)  # pyrefly: ignore[bad-argument-type]
        <= max_model_size
    )

  filtered_marginals = backend.filter_with_side_inputs(
      marginals,
      filter_fn,
      [model, selected_marginals],
      'filter to leave only small marginals',
  )
  # (Clique, np.ndarray)

  errors = backend.map_with_side_inputs(
      filtered_marginals,
      lambda clique_marginal, model: _compute_error(
          clique_marginal, model, gaussian_spec
      ),
      [model],
      'compute errors',
      resource_hints={'desired_worker_machines': 200},
  )
  # (Clique, error)

  errors_singleton = backend.to_list(errors, 'ToList')
  # singleton (list[tuple[Clique, float]])

  worst_approximated = backend.map(
      errors_singleton,
      lambda x: _select_worst_approximated(
          np.random.default_rng(), x, exponential_spec
      ),
      'Get worst approximated',
  )
  # singleton (Clique,)

  worst_approximated_marginal = backend.filter_with_side_inputs(
      marginals,
      lambda x, clique: x[0] == clique,
      [worst_approximated],
      'Leave only worst approximated',
  )
  # singleton (tuple[int, ...], np.ndarray)

  noised_worst_approximated_marginal = backend.map(
      worst_approximated_marginal,
      lambda marginal: _add_dp_noise(marginal, gaussian_spec),
      'Get LinearMeasurement for worst approximate',
  )
  # singleton mbi.LinearMeasurement

  selected_marginals = backend.map_with_side_inputs(
      selected_marginals,
      _extend_list_fn,
      [worst_approximated],
      'Extend',
  )
  # singleton (list[Clique])

  if (
      additional_output is not None
      and additional_output.diagnostic_info is not None
  ):
    worst_approximated_list = backend.map(
        worst_approximated, lambda x: [x], 'Worst Approximated to List'
    )
    additional_output.diagnostic_info = diagnostic_info.update_diagnostic_info(
        backend,
        additional_output.diagnostic_info,
        errors_singleton,
        worst_approximated_list,
        'Update Diagnostic Info',
    )

  return (
      noised_worst_approximated_marginal,
      selected_marginals,
  )


def _compute_error(
    clique_marginals: tuple[Clique, np.ndarray],
    model: mbi.MarkovRandomField,
    gaussian_spec: pipeline_dp.budget_accounting.MechanismSpec,
) -> tuple[Clique, float]:
  """Computes the error between the marginal and the model."""
  clique, marginal = clique_marginals
  jitted = jax.jit(
      mbi.marginal_oracles.variable_elimination, static_argnums=(1, 2, 3)
  )
  estimate_fn = jitted.lower(model.potentials, clique, model.total).compile()
  estimate = estimate_fn(model.potentials)
  diff = marginal.ravel() - estimate.datavector()
  sigma = gaussian_spec.noise_standard_deviation
  bias = jnp.sqrt(2 / jnp.pi) * sigma * marginal.size
  return clique, float(jnp.linalg.norm(diff, ord=1) - bias)


def _select_worst_approximated(
    rng: np.random.Generator,
    clique_errors: list[tuple[Clique, float]],
    exponential_spec: pipeline_dp.budget_accounting.MechanismSpec,
) -> Clique:
  """Returns the worst approximated candidate in the given errors."""
  errors = np.array([x[1] for x in clique_errors])
  exponential_eps = np.sqrt(2) / exponential_spec.noise_standard_deviation
  idx = primitives.exponential_mechanism(
      rng, errors, epsilon=exponential_eps, sensitivity=1.0, monotonic=True
  )
  return clique_errors[idx][0]


def _add_dp_noise(
    clique_marginal: tuple[Clique, np.ndarray],
    mechanism_spec: pipeline_dp.budget_accounting.MechanismSpec,
) -> mbi.LinearMeasurement:
  """Adds DP noise to the marginal."""
  clique, marginal = clique_marginal
  sensitivities = pipeline_dp.dp_computations.Sensitivities(l2=1.0)
  gaussian_mechanism = pipeline_dp.dp_computations.create_additive_mechanism(
      mechanism_spec, sensitivities
  )
  return mbi.LinearMeasurement(
      gaussian_mechanism.add_noise(marginal), clique, gaussian_mechanism.std  # pyrefly: ignore[bad-argument-type]
  )


T = TypeVar('T')


def _extend_list_fn(items: list[T], item: T) -> list[T]:
  """Extends the list with one item."""
  result = copy.copy(items)  # Copy since Beam does like to mutate the input.
  result.append(item)
  return result


def _create_new_model(
    model: mbi.MarkovRandomField,
    measurements: list[mbi.LinearMeasurement],
    pgm_iters: int,
) -> mbi.Model:
  """Adds measurements to the model and running mirror descent."""
  return mbi.estimation.MirrorDescent().estimate(
      model.domain,
      copy.copy(measurements),
      warm_start=model,
      iters=pgm_iters,
  )
