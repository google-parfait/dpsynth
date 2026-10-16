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

"""Differentially private primitives for local mode synthetic data generation.

This module is intended to be the singular home for low-level DP building blocks
in dpsynth, making a potential future switch to a PyDP or OpenDP backend
simpler.
While not all DP code in dpsynth currently goes through this module, that is the
long-term goal (at least for tabular data).

Design Decisions:

1) Pre-Computed Aggregates: All functions expect pre-computed aggregates
   (counts, histograms, quality scores) rather than performing data aggregation
   themselves. Therefore, it is the responsibility of the caller to ensure the
   appropriate privacy assumptions are satisfied (primarily that each user
   contributes at most one record to one bucket).
2) Pure Functions: Primitives are pure functions that take NumPy inputs and
   return NumPy or Python outputs. State and DP accounting, when needed, are
   managed externally.

Privacy Characterizations:

- Exponential Mechanism (`exponential_mechanism`): Standard exponential
  mechanism for discrete selection given candidate quality scores.
- Quantiles (`quantiles_from_histogram`): Composition of exponential mechanisms
  via jittered recursive median bisection over a dense histogram.
- Gaussian Thresholding (`gaussian_thresholding`): Partition selection mechanism
  that adds Gaussian noise to counts and tests against a threshold bounding
  false positives for empty partitions at delta.
- Gaussian Mechanism (`gaussian_mechanism`): Standard Gaussian mechanism applied
  to input summary statistics (e.g., counts).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.stats

# ---------------------------------------------------------------------------
# Exponential Mechanism
# ---------------------------------------------------------------------------


def exponential_mechanism(
    rng: np.random.Generator,
    quality_scores: np.ndarray,
    *,
    epsilon: float,
    sensitivity: float,
    monotonic: bool = False,
) -> int:
  """Selects an index using the discrete exponential mechanism.

  Samples a candidate index with probability proportional to
  exp(coef * epsilon * quality_scores / sensitivity), where coef is 1.0
  if monotonic is True and 0.5 otherwise. This is implemented via the
  Gumbel-max trick (adding Gumbel noise with scale = sensitivity / (coef *
  epsilon)
  and returning argmax), which avoids computing normalizing constants and works
  gracefully with infinite epsilon (scale = 0).

  Args:
    rng: A numpy random number generator.
    quality_scores: 1D array of utility/quality scores for each candidate.
    epsilon: Privacy parameter epsilon. Must be non-negative.
    sensitivity: Upper bound on the quality score sensitivity. Must be positive.
    monotonic: Whether the score function is monotonic with respect to dataset
      modifications (sensitivity Delta u instead of 2 * Delta u). Defaults to
      False.

  Returns:
    The index of the selected candidate.

  Raises:
    ValueError: If epsilon < 0, sensitivity <= 0, or quality_scores is empty.
  """
  if epsilon < 0:
    raise ValueError(f'epsilon must be non-negative, got {epsilon}')
  if sensitivity <= 0:
    raise ValueError(f'sensitivity must be positive, got {sensitivity}')

  scores = np.asarray(quality_scores, dtype=float)
  if scores.size == 0:
    raise ValueError('quality_scores must not be empty.')

  if epsilon == 0:
    return int(rng.choice(scores.size))

  coef = 1.0 if monotonic else 0.5
  scale = sensitivity / (coef * epsilon)
  noise = rng.gumbel(scale=scale, size=scores.size)
  return int(np.argmax(scores + noise))


# ---------------------------------------------------------------------------
# DP Quantiles via Recursive Median Bisection
# ---------------------------------------------------------------------------


def _median_from_histogram(
    rng: np.random.Generator,
    counts: np.ndarray,
    epsilon: float,
) -> int:
  """Returns the index of a DP median within a dense histogram."""
  total_points = len(counts)
  if total_points == 0:
    return 0
  n = counts.sum()
  target = n / 2.0
  cumsum = np.cumsum(counts)

  # Score u(v) = -dist(target, [L_v, R_v]), sensitivity 1/2.
  left_ranks = np.r_[0, cumsum[:-1]]
  scores = -np.maximum(0, np.maximum(left_ranks - target, target - cumsum))

  return exponential_mechanism(
      rng=rng,
      quality_scores=scores,
      epsilon=epsilon,
      sensitivity=0.5,
      monotonic=False,
  )


def jitter_factor(num_partitions: int) -> int:
  """Returns a data-independent jitter resolution m from num_partitions."""
  # m >= num_partitions keeps each jittered cell below one partition's mass;
  # the 4x absorbs multinomial fluctuation.
  return max(1, 4 * num_partitions)


def quantiles_from_histogram(
    rng: np.random.Generator,
    counts: np.ndarray,
    epsilon_levels: np.ndarray,
    jitter_strategy: Literal['symmetric', 'refine'],
) -> list[int]:
  """DP quantile edge indices into ``counts`` via jittered median bisection.

  Operates purely in index space: it returns cell indices into ``counts`` and
  leaves the mapping from index to domain value to the caller.

  Tie handling via jitter:
  Recursive median bisection needs each record assigned to one side of every
  split independently. A "spike" of records tied on one grid cell breaks this: a
  whole-cell split sends all that mass to one side, biasing the quantiles and
  collapsing sub-ranges (dropping edges). We fix this by breaking ties directly
  in the histogram domain rather than over the raw data values -- each cell's
  count is redistributed to nearby cells as Multinomial(count, kernel) (one
  draw per non-empty cell), which is distributionally identical to independently
  perturbing each record and so needs no extra privacy budget. The ``refine``
  strategy uses a strictly-positive kernel over refined sub-cells (value-
  preserving); the ``symmetric`` strategy uses a symmetric kernel over
  neighboring grid cells.

  Args:
    rng: A numpy random number generator.
    counts: Dense 1D histogram counts.
    epsilon_levels: Per-level exponential mechanism epsilons, ordered from the
      deepest (finest) level to the shallowest (coarsest).
    jitter_strategy: Specifies the pre-processing jitter strategy, -
      'symmetric': jitter mass to +/- m//2 neighbors on the same grid. -
      'refine': jitter mass to m equivalent sub-cells.

  Returns:
    A sorted list of ``2 ** len(epsilon_levels) - 1`` cell indices.
  """
  counts = np.asarray(counts)
  m = jitter_factor(2 ** len(epsilon_levels))

  if jitter_strategy == 'refine':
    stride, offsets = m, np.arange(m)
  else:
    half = m // 2
    stride, offsets = 1, np.arange(-half, half + 1)

  # Scatter each cell's mass over its jittered targets: same law as perturbing
  # each record, so it breaks ties without spending extra privacy budget.
  num_cells = counts.size * stride
  nz = np.flatnonzero(counts)
  probas = np.full(offsets.size, 1.0 / offsets.size)
  split = rng.multinomial(counts[nz].astype(np.int64), probas)
  targets = np.clip(nz[:, None] * stride + offsets, 0, num_cells - 1)
  jittered = np.bincount(  # pyrefly: ignore[no-matching-overload]
      targets.flatten(), weights=split.flatten(), minlength=num_cells
  )

  def _rec(lo_idx, hi_idx, depth):
    if depth == 0:
      return []
    median_idx = lo_idx + _median_from_histogram(
        rng, jittered[lo_idx:hi_idx], epsilon_levels[depth - 1]
    )
    left = _rec(lo_idx, median_idx, depth - 1)
    right = _rec(median_idx, hi_idx, depth - 1)
    return left + [median_idx] + right

  result = _rec(0, jittered.size, len(epsilon_levels))
  if jitter_strategy == 'refine':
    result = [idx // m for idx in result]
  return result


# ---------------------------------------------------------------------------
# Partition Selection
# ---------------------------------------------------------------------------


def gaussian_thresholding(
    rng: np.random.Generator,
    counts: np.ndarray,
    *,
    sigma: float,
    delta: float,
    l2_sensitivity: float,
    linf_sensitivity: float,
    min_count: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
  """Selects partitions using Gaussian Thresholding (Weighted Gaussian).

  This implements Algorithm 2 from the DP-SIPS paper (Swanberg et al., 2023) on
  pre-aggregated partition counts. It is the simplest partition selection
  mechanism:

    1. Filter partitions with true count below ``min_count``.
    2. Add Gaussian noise with standard deviation ``l2_sensitivity * sigma``.
    3. Return indices of partitions whose noisy count exceeds a threshold chosen
       to bound the false-positive probability per empty partition at ``delta``.

  Under item-level DP each record is treated as a distinct user contributing to
  at most one partition, so ``l2_sensitivity = 1`` and ``linf_sensitivity = 1``.
  The threshold is ``T = min_count + sigma * Phi^{-1}(1 - delta)``, following
  the paper's formula with ``max_part = 1`` and a shift of ``(min_count - 1)``
  to account for the minimum count guarantee.

  When ``min_count > 1``, partitions with true count below ``min_count`` are
  pre-filtered and the threshold shifts up accordingly. The privacy guarantee is
  preserved: partitions where both neighboring datasets are above ``min_count``
  are covered by the Gaussian mechanism, and the boundary case (one dataset at
  ``min_count - 1``, the other at ``min_count``) is covered by the same additive
  ``delta``.

  When a single user can contribute multiple records across or within
  partitions, ``l2_sensitivity`` scales the Gaussian noise standard deviation
  (``stddev = l2_sensitivity * sigma``) and ``linf_sensitivity`` bounds the
  maximum change to any single partition's count between neighboring datasets,
  shifting the threshold base to ``min_count + linf_sensitivity - 1``.

  Args:
    rng: A numpy random number generator.
    counts: 1D array of pre-aggregated partition counts.
    sigma: The Gaussian noise multiplier (unscaled standard deviation).
    delta: Failure probability (false positive bound per empty partition).
    l2_sensitivity: L2 sensitivity of the count vector across partitions.
    linf_sensitivity: L-infinity sensitivity (maximum change to any single
      partition's count between neighboring datasets).
    min_count: Minimum true count for a partition to be eligible. Partitions
      with fewer occurrences in the data are never returned. Must be >= 1.

  Returns:
    A tuple ``(selected_indices, noisy_counts)`` where ``selected_indices`` is a
    1D integer array of indices into ``counts`` that passed the threshold, and
    ``noisy_counts`` is a 1D float array of their noisy counts.

  Raises:
    ValueError: If sigma < 0, delta not in (0, 1], l2_sensitivity <= 0,
      linf_sensitivity <= 0, or min_count < 1.
  """
  if sigma < 0:
    raise ValueError(f'{sigma=} must be non-negative.')
  if delta <= 0 or delta > 1:
    raise ValueError(f'{delta=} must be in (0, 1].')
  if l2_sensitivity <= 0 or linf_sensitivity <= 0:
    raise ValueError(f'{l2_sensitivity=} and {linf_sensitivity=} must be > 0.')
  if min_count < 1:
    raise ValueError(f'{min_count=} must be >= 1.')

  counts = np.asarray(counts, dtype=float)

  # Filter partitions below the minimum count before adding noise.
  eligible_idx = np.flatnonzero(counts >= min_count)
  noisy_counts = np.asarray(
      gaussian_mechanism(
          rng, counts[eligible_idx], sigma=sigma, l2_sensitivity=l2_sensitivity
      )
  )

  # A partition that is a candidate here (count >= min_count) but ineligible in
  # a neighboring dataset (count <= min_count - 1) drives the per-partition
  # false-positive budget `delta`. Between neighboring datasets: (i) the noise
  # std is stddev = l2_sensitivity * sigma, and (ii) such a partition's true
  # count can reach (min_count - 1) + linf_sensitivity -- the neighbor sits just
  # under the eligibility cutoff at min_count - 1 and the differing user adds up
  # to linf_sensitivity records to it. Bounding
  #   Pr[(min_count - 1 + linf_sensitivity) + N(0, stddev^2) >= T] <= delta
  # gives T = (min_count + linf_sensitivity - 1) + stddev * ppf(1 - delta).
  stddev = l2_sensitivity * sigma
  base = float(linf_sensitivity + min_count - 1)
  threshold = base + stddev * scipy.stats.norm.ppf(1.0 - delta)
  passed = noisy_counts >= threshold
  # eligible_idx is sorted (np.flatnonzero), so the output order is
  # deterministic.
  return eligible_idx[passed], noisy_counts[passed]


def ensure_public_partitions(
    rng: np.random.Generator,
    selected: np.ndarray,
    counts: np.ndarray,
    stddev: float,
    public: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Ensures public partition IDs appear in the selected set.

  For any public ID not already in ``selected``, appends it with a noisy
  count drawn from N(0, stddev²), consistent with the Gaussian mechanism
  applied to an empty partition.

  Args:
    rng: A numpy random number generator.
    selected: 1D array of already-selected partition IDs.
    counts: 1D array of noisy counts parallel to ``selected``.
    stddev: Gaussian noise standard deviation used by the mechanism.
    public: 1D array of public partition IDs to guarantee.

  Returns:
    A (selected, counts) tuple with missing public partitions appended.
  """
  missing_mask = ~np.isin(public, selected)
  missing = public[missing_mask]
  if missing.size == 0:
    return selected, counts
  noise = np.asarray(
      gaussian_mechanism(
          rng, np.zeros(missing.size), sigma=stddev, l2_sensitivity=1.0
      )
  )
  all_selected = np.concatenate([selected, missing])
  all_counts = np.concatenate([counts, noise])
  # Sort by partition key to ensure deterministic order and avoid leaking
  # which partitions were missing.
  order = np.argsort(all_selected)
  return all_selected[order], all_counts[order]


# ---------------------------------------------------------------------------
# Gaussian Mechanism
# ---------------------------------------------------------------------------


def gaussian_mechanism(
    rng: np.random.Generator,
    counts: np.ndarray | float | int,
    *,
    sigma: float,
    l2_sensitivity: float,
) -> float | np.ndarray:
  """Adds Gaussian noise to scalar, 1D array, or multi-dimensional array counts.

  Args:
    rng: A numpy random number generator.
    counts: The true count(s). Can be a scalar, 1D array, or multi-dimensional
      array.
    sigma: The Gaussian noise multiplier (unscaled standard deviation).
    l2_sensitivity: The L2 sensitivity of `counts`. Scales the noise standard
      deviation as `l2_sensitivity * sigma`.

  Returns:
    The noisy count(s) with the same shape as `counts`.

  Raises:
    ValueError: If sigma < 0 or l2_sensitivity <= 0.
  """
  if sigma < 0:
    raise ValueError(f'sigma must be non-negative, got {sigma}')
  if l2_sensitivity <= 0:
    raise ValueError(f'l2_sensitivity must be positive, got {l2_sensitivity}')
  stddev = l2_sensitivity * sigma

  if isinstance(counts, (int, float, np.generic)):
    return float(counts) + float(rng.normal(scale=stddev))

  counts_arr = np.asarray(counts, dtype=float)
  return counts_arr + rng.normal(scale=stddev, size=counts_arr.shape)
