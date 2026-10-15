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

"""Common utility functions for synthetic data mechanisms."""

from collections.abc import Iterable, Mapping, Sequence
import contextlib
import dataclasses
import functools
import itertools
import time
from typing import TypeAlias

from absl import logging
from dpsynth import transformations
from dpsynth.local_mode import primitives
import mbi
import mbi.clique_utils
import mbi.junction_tree
import more_itertools
import numpy as np
import tqdm


def validate_initial_measurements(
    initial_measurements: Sequence[mbi.LinearMeasurement] | None,
) -> None:
  """Warns if the discrete mechanism is called without initial measurements."""
  if not initial_measurements:
    logging.warning(
        'This mechanism is not intended to be called directly, but through '
        'the higher-level APIs (TabularMechanism, DiscreteMechanism). It '
        'will run without error, but is missing 1-way marginal selection.'
    )


@dataclasses.dataclass
class MechanismDiagnostics:
  """Diagnostic info from a discrete mechanism run.

  Attributes:
    phase_times: Wall-clock time in seconds for each named phase.
    num_cliques: Number of cliques in the fitted model.
    max_clique_size: Size of the largest clique.
    total_clique_size: Sum of all clique sizes.
    max_jtree_node_size: Size of the largest junction tree node.
    total_jtree_size: Sum of all junction tree node sizes.
  """

  phase_times: dict[str, float] = dataclasses.field(default_factory=dict)
  num_cliques: int = 0
  max_clique_size: int = 0
  total_clique_size: int = 0
  max_jtree_node_size: int = 0
  total_jtree_size: int = 0


@contextlib.contextmanager
def timed(phase_times: dict[str, float], name: str):
  """Context manager that logs and records wall-clock time for a phase."""
  start = time.monotonic()
  yield
  elapsed = time.monotonic() - start
  phase_times[name] = phase_times.get(name, 0.0) + elapsed
  logging.info('[%s] %.2fs', name, elapsed)


def clique_stats(model: mbi.Model) -> MechanismDiagnostics:
  """Compute structural diagnostics from a fitted model and log them.

  Args:
    model: The fitted graphical model (must be a MarkovRandomField).

  Returns:
    A MechanismDiagnostics populated with clique and junction tree stats.

  Raises:
    TypeError: If model is not a MarkovRandomField.
  """
  if not isinstance(model, mbi.MarkovRandomField):
    raise TypeError(f'Expected MarkovRandomField, got {type(model).__name__}')
  domain = model.potentials.domain
  cliques = model.potentials.cliques
  sizes = [domain.size(c) for c in cliques]
  jtree, _ = mbi.junction_tree.make_junction_tree(domain, cliques)
  jtree_nodes = list(jtree.nodes)
  jtree_sizes = [domain.size(n) for n in jtree_nodes]
  diagnostics = MechanismDiagnostics(
      num_cliques=len(cliques),
      max_clique_size=max(sizes, default=0),
      total_clique_size=sum(sizes),
      max_jtree_node_size=max(jtree_sizes, default=0),
      total_jtree_size=sum(jtree_sizes),
  )
  logging.info(
      'Cliques: %d, max_size: %d, total_size: %d',
      diagnostics.num_cliques,
      diagnostics.max_clique_size,
      diagnostics.total_clique_size,
  )
  logging.info(
      'Junction tree: %d nodes, max_size: %d, total_size: %d',
      len(jtree_nodes),
      diagnostics.max_jtree_node_size,
      diagnostics.total_jtree_size,
  )
  return diagnostics


@dataclasses.dataclass
class DiscreteMechanismResult:
  """Result of running a discrete mechanism.

  Attributes:
    model: The estimated graphical model (Markov random field). When domain
      compression is enabled, this model is defined over the compressed domain.
    synthetic_data: A synthetic dataset whose marginals closely match the model.
      Always defined over the original (uncompressed) domain.
    measurements: The noisy marginal measurements made by the mechanism.
    diagnostics: Optional mechanism-specific diagnostic information.
    mappings: A dict mapping column names to integer mapping arrays used for
      domain compression. Empty when no compression was applied.
  """

  model: mbi.Model
  synthetic_data: mbi.Dataset | None = None
  measurements: list[mbi.LinearMeasurement] = dataclasses.field(
      default_factory=list
  )
  diagnostics: MechanismDiagnostics | None = None
  mappings: dict[str | int, np.ndarray] = dataclasses.field(
      default_factory=dict
  )


def precompute_marginals(
    data: mbi.Dataset,
    cliques: Sequence[mbi.Clique],
    *,
    use_jax: bool = False,
) -> mbi.CliqueVector:
  """Computes marginals over cliques from a Dataset, optionally using JAX."""
  if not cliques:
    return mbi.CliqueVector(data.domain, [], {})
  if use_jax:
    return mbi.extensions.precompute_marginals(
        data, cliques  # pyrefly: ignore[bad-argument-type]
    )
  return mbi.CliqueVector.from_projectable(
      data, cliques  # pyrefly: ignore[bad-argument-type]
  )


def generate_synthetic_data(
    model: mbi.Model,
    rng: np.random.Generator,
    *,
    use_jax: bool = False,
    rows: int | None = None,
) -> mbi.Dataset:
  """Generates synthetic data from a fitted model."""
  if rows is None:
    rows = max(1, int(round(model.total)))  # pyrefly: ignore[bad-argument-type]
  if use_jax and isinstance(model, mbi.MarkovRandomField):
    seed = int(rng.integers(0, 2**31 - 1))
    return mbi.extensions.synthetic_data(
        model, rows=rows, seed=seed  # pyrefly: ignore[bad-argument-type]
    )
  return model.synthetic_data(rows=rows)


def compression_mappings(
    one_way_measurements: list[mbi.LinearMeasurement],
    compress_columns: bool | Sequence[str | int] = False,
    constraints: Sequence[mbi.Constraint] = (),
) -> dict[str | int, np.ndarray]:
  """Computes mappings that merge rare domain values for compression."""
  if not compress_columns:
    return {}

  cols = (
      set(compress_columns)
      if isinstance(compress_columns, Sequence)
      else {m.clique[0] for m in one_way_measurements}
  )
  if constraints:
    constrained = set().union(*(c.domain.attributes for c in constraints))
    skipped = cols & constrained
    if skipped:
      logging.info(
          'Skipping compression for columns with constraints: %s', skipped
      )
    cols -= constrained

  mappings: dict[str | int, np.ndarray] = {}
  for m in one_way_measurements:
    col = m.clique[0]
    if col not in cols:
      continue
    mask = np.asarray(m.noisy_measurement) < 3 * m.stddev
    compressed_size = int((~mask).sum()) + 1
    mapping = np.where(mask, compressed_size - 1, np.cumsum(~mask) - 1)
    if compressed_size < len(mapping):
      mappings[col] = mapping
  return mappings


def measure_marginals_with_noise(
    rng: np.random.Generator,
    data: mbi.Dataset | mbi.CliqueVector,
    marginal_queries: Sequence[mbi.Clique],
    gdp_sigma: float,
    weights: np.ndarray | None = None,
) -> list[mbi.LinearMeasurement]:
  """Measures the given marginal queries with the Gaussian mechanism.

  Note that this function is a **single** instance of the Gaussian mechanism,
  even when multiple marginal queries are provided as input. The input gdp_sigma
  will be divided between the marginal queries evenly, unless weights are
  sigma is divided proportionally to the weights.

  Args:
    rng: A numpy random number generator.
    data: The sensitive dataset whose marginals are to be measured.
    marginal_queries: The list of marginal queries to measure, represented as a
      list of tuples of column names.
    gdp_sigma: The parameter of the Gaussian mechanism.
    weights: The weights to use for each marginal query. If None, use uniform
      weights.

  Returns:
    The list of LinearMeasurements.
  """
  if weights is None:
    weights = np.ones(len(marginal_queries))
  weights = np.array(weights) / np.linalg.norm(weights)
  if len(weights) != len(marginal_queries):
    raise ValueError(
        'The number of weights must be equal to the number of marginal queries.'
    )
  measurements = []
  for proj, wgt in zip(marginal_queries, weights):
    stddev = gdp_sigma / wgt
    x = np.asarray(data.project(proj).datavector())
    y = primitives.gaussian_mechanism(rng, x, sigma=stddev, l2_sensitivity=1.0)
    measurements.append(mbi.LinearMeasurement(y, proj, stddev))
  return measurements


def compressed_measurement(
    one_way_measurement: mbi.LinearMeasurement,
    size: int,
    transform_fn: transformations.DataTransformation[int, int],
) -> mbi.LinearMeasurement:
  """Returns a measurement defined over the compressed domain.

  Args:
    one_way_measurement: The measurement to compress.
    size: The size of the compressed domain.
    transform_fn: The domain compression transformation.

  Returns:
    A measurement defined over the compressed domain.
  """
  if len(one_way_measurement.clique) != 1:
    raise ValueError(
        'The measurement must be defined with respect to a one-way marginal,'
        f' got {one_way_measurement.clique}.'
    )
  y = np.asarray(one_way_measurement.noisy_measurement)
  mapping = np.array([transform_fn(i) for i in range(y.size)])
  y2 = np.bincount(mapping, weights=y, minlength=size)
  coefs = np.sqrt(np.bincount(mapping, minlength=size))
  return mbi.LinearMeasurement(
      y2 / coefs,
      one_way_measurement.clique,
      one_way_measurement.stddev,
      query=mbi.WeightedQuery(1.0 / coefs),
  )


def compression_transformation(
    measurement: mbi.LinearMeasurement,
) -> tuple[int, transformations.DataTransformation[int, int]]:
  """Returns a domain compression transformation for the given measurement."""
  mask = measurement.noisy_measurement < 3 * measurement.stddev  # pyrefly: ignore[unsupported-operation]
  size, transform_fn = transformations.create_rare_value_merging_transformation(
      mask  # pyrefly: ignore[bad-argument-type]
  )
  return size, transform_fn


def convert_to_total_measurement(
    measurements: list[mbi.LinearMeasurement],
) -> mbi.LinearMeasurement:
  """Converts a list of measurements to a measurement of total records."""
  # Note: This is a hack to get around the fact that
  # mbi.estimation.minimum_variance_unbiased_total does not work on compressed
  # measurements.
  total = mbi.estimation.minimum_variance_unbiased_total(measurements)
  return mbi.LinearMeasurement(
      noisy_measurement=total,  # pyrefly: ignore[bad-argument-type]
      clique=(),
      stddev=1.0,  # ideally we'd get this from minimum_variance_unbiased_total.
  )


def get_domain_compression_transformations(
    one_way_measurements: list[mbi.LinearMeasurement],
) -> tuple[
    mbi.Domain,
    list[mbi.LinearMeasurement],
    dict[str | int, transformations.DataTransformation[int, int]],
]:
  """Returns a new domain and transformations for compressing the domain.

  Args:
    one_way_measurements: List of one-way measurements over the original domain.

  Returns: A tuple of three elements:
    - The new (compressed) domain.
    - The list of measurements defined over the compressed domain.
    - A dictionary mapping each column of the original domain to a
      transformation that maps values in that column to values in the
      compressed domain.
  """
  column_transforms = {}
  sizes = {}
  new_measurements = []
  for measurement in one_way_measurements:
    col = measurement.clique[0]
    size, transform_fn = compression_transformation(measurement)
    sizes[col] = size
    column_transforms[col] = transform_fn
    new_measurements.append(
        compressed_measurement(measurement, size, transform_fn)
    )
  return mbi.Domain.fromdict(sizes), new_measurements, column_transforms


def downward_closure(
    marginal_queries: Iterable[mbi.Clique],
) -> Iterable[mbi.Clique]:
  """Returns the downward closure of the given marginal queries.

  Given a collection of sets, the downward closure is the set of all sets that
  are subsets of any of the given sets.

  Example Usage:
  >>> downward_closure([('a', 'b'), ('a', 'c')])
  [('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c')]

  Args:
    marginal_queries: The marginal queries to compute the downward closure of.

  Returns:
    The downward closure of the given marginal queries, without the empty tuple.
  """
  ans = set()
  for proj in marginal_queries:
    ans.update(more_itertools.powerset(proj))
  return list(sorted(ans - {()}, key=len))  # pyrefly: ignore[bad-return]


Workload: TypeAlias = Mapping[mbi.Clique, float]
Workload2: TypeAlias = Iterable[mbi.Clique]


def supporting_cliques(
    domain: mbi.Domain,
    workload: Workload | Workload2 | None,
    max_marginal_size: float = float('inf'),
    degree: int = 3,
) -> list[mbi.Clique]:
  """Returns the maximal workload cliques filtered by domain size.

  This function extracts a collection of top-level cliques from the workload,
  filtering out cliques whose domain exceeds ``max_marginal_size`` or that
  are redundant given other cliques. Any marginal query in the downward closure
  of the workload below ``max_marginal_size`` can be computed from the
  supporting marginals.

  Args:
    domain: The domain of the dataset.
    workload: A workload specification. Defaults to all three-way marginals, or
      the maximum possible degree for fewer than three columns.
    max_marginal_size: The maximum domain size of a clique to include.
    degree: The maximum degree of the automatically constructed workload.
      Ignored when an explicit workload is provided.

  Returns:
    A list of cliques from the workload whose domain size is within the limit.
  """
  if workload is None:
    degree = min(degree, len(domain.attributes))
    cliques = list(itertools.combinations(domain.attributes, degree))
  elif isinstance(workload, Mapping):
    cliques = [tuple(cl) for cl in workload.keys()]
  else:
    cliques = [tuple(cl) for cl in workload]
  cliques = mbi.clique_utils.downward_closure(cliques)
  cliques = [cl for cl in cliques if domain.size(cl) <= max_marginal_size]
  return mbi.clique_utils.maximal_subset(cliques)


def one_way_cliques(
    workload: Workload | Workload2 | None, domain: mbi.Domain
) -> list[mbi.Clique]:
  """Returns one-way cliques from the workload, or all columns if None."""
  if workload is None:
    return [(col,) for col in domain]
  return [tuple(cl) for cl in workload if len(cl) == 1]


def compiled_workload(
    domain: mbi.Domain,
    workload: Workload | Workload2 | None,
    max_marginal_size: float = float('inf'),
    degree: int = 3,
) -> Workload:
  """Compiles an input workload into a set of candidate measurements for AIM.

  Args:
    domain: The domain of the dataset.
    workload: A dictionary mapping marginal queries to weights representing the
      importance of each marginal query.
    max_marginal_size: The maximum size of a marginal query to consider.
    degree: The maximum degree of the automatically constructed workload.
      Ignored when an explicit workload is provided.

  Returns:
    A dictionary mapping marginal queries in the downward closure of the
    workload to weights representing the importance of each marginal query.
  """

  if workload is None:
    degree = min(degree, len(domain.attributes))
    workload = list(itertools.combinations(domain.attributes, degree))

  if not isinstance(workload, Mapping):
    workload = {tuple(cl): 1.0 for cl in workload}

  dc = list(downward_closure(workload.keys()))

  # Precompute per-attribute weights so we can score each candidate in
  # O(|candidate|) instead of O(|workload|).
  attr_weight: dict[str | int, float] = {}
  for workload_cl, weight in workload.items():
    for attr in workload_cl:
      attr_weight[attr] = attr_weight.get(attr, 0.0) + weight

  def score(cl):
    return sum(attr_weight.get(attr, 0.0) for attr in cl)

  return {cl: score(cl) for cl in dc if domain.size(cl) <= max_marginal_size}


def compute_independence_errors(
    data: mbi.Dataset | mbi.CliqueVector,
    model: mbi.MarkovRandomField,
    cliques: Sequence[mbi.Clique],
) -> dict[mbi.Clique, float]:
  """Computes L1 errors between actual marginals and the independence model."""
  total = float(model.total)  # pyrefly: ignore[bad-argument-type]

  # Pure numpy to avoid XLA recompilation: each distinct clique shape triggers
  # a separate compilation, which dominates runtime for large candidate sets.
  oneway = {
      a: np.asarray(model.project((a,)).datavector()) / total
      for a in data.domain
  }

  errors = {}
  for cl in tqdm.tqdm(cliques, desc='Computing independence errors'):
    estimate = functools.reduce(np.multiply.outer, (oneway[a] for a in cl))
    actual = np.asarray(data.project(cl).datavector(flatten=False))
    error = np.abs(total * estimate - actual).sum()
    errors[cl] = error
  return errors
