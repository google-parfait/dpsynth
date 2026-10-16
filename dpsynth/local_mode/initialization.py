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

"""Utilities for measuring and integer-encoding single columns."""

from __future__ import annotations

import dataclasses
import math

import dp_accounting
from dpsynth import api
from dpsynth import domain
from dpsynth.local_mode import primitives
from dpsynth.local_mode import vectorized_transformations as vtx
import numpy as np


def encode_to_grid(values, lower, upper, delta, attribute=None, **_):
  """Maps finite value(s) to quantile-grid index/indices in [0, grid_size - 1].

  Clipping to ``[lower, upper]`` folds out-of-grid values into the boundary bins
  and guarantees the returned index stays in range. Polymorphic over scalars and
  NumPy arrays; if ``attribute`` is provided, applies its ``standardize``
  semantics in a vectorized manner first.

  Args:
    values: A scalar or sequence/array of numerical values.
    lower: The inclusive lower bound of the candidate grid.
    upper: The inclusive upper bound of the candidate grid.
    delta: The spacing between adjacent grid points.
    attribute: Optional ``NumericalAttribute`` used to standardize ``values``.

  Returns:
    The nearest grid index (or array of indices) as ``np.int64``.
  """
  if attribute is not None:
    values = np.asarray(values, dtype=float)
    if attribute.clip_to_range:
      values = np.where(np.isnan(values), attribute.min_value, values)
    elif values.ndim > 0:
      values = values[
          (values >= attribute.min_value) & (values <= attribute.max_value)
      ]
    if attribute.dtype == 'int':
      values = np.round(values)
  clamped = np.clip(values, lower, upper)
  return np.round((clamped - lower) / delta).astype(np.int64)


@dataclasses.dataclass(frozen=True)
class NumericalMeasurement:
  """Measurement from a numerical initializer."""

  categorical_attribute: domain.CategoricalAttribute
  bin_edges: np.ndarray
  noisy_counts: np.ndarray | None = None
  stddev: float = np.nan


@dataclasses.dataclass(frozen=True)
class CategoricalMeasurement:
  """Measurement from a categorical initializer."""

  categorical_attribute: domain.CategoricalAttribute
  noisy_counts: np.ndarray
  stddev: float


@dataclasses.dataclass(frozen=True)
class OpenSetMeasurement:
  """Measurement from an open-set categorical initializer."""

  categorical_attribute: domain.CategoricalAttribute
  noisy_counts: np.ndarray
  stddev: float


ColumnMeasurement = (
    NumericalMeasurement | CategoricalMeasurement | OpenSetMeasurement
)


def compute_grid_spec(
    attribute: domain.NumericalAttribute,
    num_partitions: int,
    max_grid_size: int = 10_000_000,
) -> tuple[float, float, int]:
  """Returns (lower, upper, grid_size) for the quantile candidate grid."""
  min_value = float(attribute.min_value)
  if attribute.dtype == 'int':
    m = primitives.jitter_factor(num_partitions)
    budget = max(2, max_grid_size // m)
    int_range = int(attribute.max_value - attribute.min_value + 1)
    step = max(1, math.ceil(int_range / budget))
    gs = math.ceil(int_range / step)
    return min_value, min_value + (gs - 1) * step, gs

  return min_value, float(attribute.exclusive_max_value), max_grid_size


@dataclasses.dataclass(frozen=True, kw_only=True)
class NumericalInitializerConfig(api.MechanismConfig):
  """Configuration for initializing numerical attributes.

  Attributes:
    num_partitions: Number of partitions (must be a power of 2).
    max_grid_size: Maximum grid size for the histogram.
    epsilon_ratio: Ratio by which privacy budget epsilon increases at each
      deeper level of recursive bisection. Defaults to 1.0 (uniform budget split
      across levels). Setting to sqrt(2) approx 1.414 can provide minor accuracy
      gains on smooth continuous data.
    quantile_budget_fraction: Fraction of the zCDP budget allocated to the
      quantile tree; the remainder is allocated to the Gaussian mechanism for
      measuring discretized bin counts. Defaults to 0.5.
  """

  num_partitions: int
  max_grid_size: int = 10_000_000
  epsilon_ratio: float = 1.0
  quantile_budget_fraction: float = 0.5

  def __post_init__(self):
    if self.max_grid_size < 2:
      raise ValueError(f'max_grid_size must be >= 2, got {self.max_grid_size}.')
    if self.num_partitions >= self.max_grid_size:
      raise ValueError(f'{self.num_partitions=} >= {self.max_grid_size=}')
    if not 0.0 < self.quantile_budget_fraction < 1.0:
      raise ValueError(
          'quantile_budget_fraction must be in (0, 1), got'
          f' {self.quantile_budget_fraction}.'
      )

  def configure(self, attribute=None, *, budget, delta=0):
    assert attribute is not None

    levels = int(np.log2(self.num_partitions))
    if 2**levels != self.num_partitions:
      raise ValueError(f'{self.num_partitions=} must be a power of 2.')

    quantile_rho = budget * self.quantile_budget_fraction
    count_rho = budget * (1.0 - self.quantile_budget_fraction)
    rho_ratio = self.epsilon_ratio**2
    budget_weights = rho_ratio ** np.arange(levels)[::-1]
    rho_levels = quantile_rho * budget_weights / budget_weights.sum()
    eps = np.sqrt(8.0 * rho_levels)
    if attribute.bin_edges is not None:
      # Skip DP quantiles and allocate the full budget to bin counts.
      eps = np.array([])
      count_rho = budget
    sigma = math.sqrt(0.5 / count_rho)
    return NumericalInitializer(
        config=self,
        attribute=attribute,
        epsilon_levels=tuple(eps.tolist()),
        sigma=sigma,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class NumericalInitializer(api.CalibratedMechanism):
  """Calibrated mechanism for initializing numerical attributes."""

  config: NumericalInitializerConfig
  attribute: domain.NumericalAttribute
  epsilon_levels: tuple[float, ...]
  sigma: float

  @property
  def _num_levels(self) -> int:
    return int(np.log2(self.config.num_partitions))

  @property
  def grid_spec(self) -> tuple[float, float, int]:
    return compute_grid_spec(
        self.attribute, self.config.num_partitions, self.config.max_grid_size
    )

  @property
  def grid_size(self) -> int:
    return self.grid_spec[2]

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed privacy event for quantiles and bin counts."""
    events: list[dp_accounting.DpEvent] = [
        dp_accounting.ExponentialMechanismDpEvent(epsilon=float(eps))
        for eps in self.epsilon_levels
    ]
    events.append(dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma))
    return dp_accounting.ComposedDpEvent(events)

  def __call__(
      self,
      rng: np.random.Generator,
      data: np.ndarray,
  ) -> NumericalMeasurement:
    """Returns a NumericalMeasurement with the discretization and noisy counts."""
    counts = self._grid_histogram(data)
    ood_count = (
        float(len(data) - counts.sum())
        if not self.attribute.clip_to_range
        else 0.0
    )
    return self.from_summary(rng, counts, ood_count=ood_count)

  def _grid_histogram(self, data):
    """Returns the quantile candidate-grid histogram (length grid_size)."""
    lower, upper, gs = self.grid_spec
    delta = (upper - lower) / (gs - 1)
    indices = encode_to_grid(data, lower, upper, delta, self.attribute)
    return np.bincount(indices, minlength=gs)

  def from_summary(
      self,
      rng: np.random.Generator,
      counts: np.ndarray,
      *,
      ood_count: float = 0.0,
  ) -> NumericalMeasurement:
    """Returns a NumericalMeasurement from pre-aggregated histogram counts."""
    lower, upper, _ = self.grid_spec
    delta = (upper - lower) / max(1, np.asarray(counts).size - 1)
    if self.attribute.bin_edges is not None:
      # Skip DP quantiles and aggregate the grid histogram at bin_edges.
      raw_edges = self.attribute.bin_edges
    else:
      jitter_strategy = (
          'refine' if self.attribute.dtype == 'int' else 'symmetric'
      )
      indices = primitives.quantiles_from_histogram(
          rng,
          counts,
          epsilon_levels=np.asarray(self.epsilon_levels),
          jitter_strategy=jitter_strategy,
      )
      raw_edges = [lower + i * delta for i in indices]

    cm = edges_to_column_measurement(
        raw_edges=raw_edges,
        attribute=self.attribute,
    )
    splits = np.round((cm.bin_edges - lower) / delta).astype(np.int64) + 1
    segments = np.split(np.asarray(counts, dtype=float), splits)
    in_domain_counts = np.array([seg.sum() for seg in segments], dtype=float)
    if self.attribute.clip_to_range:
      bin_counts = in_domain_counts
    else:
      bin_counts = np.r_[ood_count, in_domain_counts]
    noisy = primitives.gaussian_mechanism(
        rng, bin_counts, sigma=self.sigma, l2_sensitivity=1.0
    )
    return dataclasses.replace(
        cm, noisy_counts=np.asarray(noisy), stddev=self.sigma
    )


def edges_to_column_measurement(
    raw_edges,
    attribute,
) -> NumericalMeasurement:
  """Converts raw quantile edges into a NumericalMeasurement.

  Handles edge deduplication, degenerate-bin removal, and categorical
  attribute construction.

  Args:
    raw_edges: Quantile edge values (unsorted duplicates are fine).
    attribute: The ``NumericalAttribute`` defining the data domain.

  Returns:
    A ``NumericalMeasurement`` with deduplicated bin edges.
  """
  raw_edges = np.asarray(raw_edges, dtype=float)
  bin_edges = np.unique(raw_edges)
  if len(bin_edges) > 0 and bin_edges[-1] >= attribute.max_value:
    bin_edges = bin_edges[:-1]
  cat_attr = vtx.categorical_attribute_from_edges(bin_edges, attribute)
  return NumericalMeasurement(cat_attr, bin_edges)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CategoricalInitializerConfig(api.MechanismConfig):
  """Configuration for initializing categorical attributes."""

  def configure(self, attribute=None, *, budget, delta=0):
    assert attribute is not None
    return CategoricalInitializer(
        config=self,
        attribute=attribute,
        sigma=math.sqrt(0.5 / budget),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CategoricalInitializer(api.CalibratedMechanism):
  """Calibrated mechanism for initializing categorical attributes."""

  config: CategoricalInitializerConfig
  attribute: domain.CategoricalAttribute
  sigma: float

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for this mechanism."""
    return dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> CategoricalMeasurement:
    """Returns a CategoricalMeasurement with the noisy histogram."""
    encoded = vtx.discrete_encode(data, self.attribute)
    counts = np.bincount(encoded, minlength=self.attribute.size)
    return self.from_summary(rng, counts)

  def from_summary(
      self, rng: np.random.Generator, counts: np.ndarray
  ) -> CategoricalMeasurement:
    """Returns a CategoricalMeasurement from pre-aggregated counts."""
    noisy = primitives.gaussian_mechanism(
        rng, counts, sigma=self.sigma, l2_sensitivity=1.0
    )
    return CategoricalMeasurement(
        self.attribute, noisy_counts=np.asarray(noisy), stddev=self.sigma
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class OpenSetInitializerConfig(api.MechanismConfig):
  """Configuration for initializing open-set categorical attributes."""

  min_count: int = 1

  def configure(self, attribute=None, *, budget, delta=0):
    assert attribute is not None
    return OpenSetInitializer(
        config=self,
        attribute=attribute,
        sigma=math.sqrt(0.5 / budget),
        delta=delta,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class OpenSetInitializer(api.CalibratedMechanism):
  """Calibrated mechanism for initializing open-set categorical attributes."""

  config: OpenSetInitializerConfig
  attribute: domain.OpenSetCategoricalAttribute
  sigma: float
  delta: float

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the privacy event including thresholding delta."""
    main_event = dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)
    failure_event = dp_accounting.dp_event.EpsilonDeltaDpEvent(0, self.delta)
    return dp_accounting.ComposedDpEvent([main_event, failure_event])

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> OpenSetMeasurement:
    """Returns a differentially private measurement of the given data."""
    data = np.asarray(data, dtype=str)
    unique_values, inverse = np.unique(data, return_inverse=True)
    counts = np.bincount(inverse)
    return self.from_summary(rng, unique_values, counts)

  def from_summary(
      self,
      rng: np.random.Generator,
      unique_values: np.ndarray,
      counts: np.ndarray,
  ) -> OpenSetMeasurement:
    """Returns an OpenSetMeasurement from pre-aggregated value counts."""
    selected_partitions, estimated_counts = primitives.gaussian_thresholding(
        rng,
        counts,
        sigma=self.sigma,
        delta=self.delta,
        l2_sensitivity=1.0,
        linf_sensitivity=1.0,
        min_count=self.config.min_count,
    )
    selected_values = np.array(
        [str(v) for v in unique_values[selected_partitions]]
    )

    if self.attribute.public_possible_values:
      pub = np.array(self.attribute.public_possible_values)
      selected_values, estimated_counts = primitives.ensure_public_partitions(
          rng,
          selected_values,
          estimated_counts,
          self.sigma,
          pub,
      )

    default = self.attribute.default_value
    possible_values = [default] + selected_values.tolist()
    cat_attr = domain.CategoricalAttribute(possible_values)

    return OpenSetMeasurement(
        cat_attr, noisy_counts=estimated_counts, stddev=self.sigma
    )
