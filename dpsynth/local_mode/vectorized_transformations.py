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

"""Vectorized transformations for the local_mode package.

This is a vectorized fork of ``dpsynth/transformations.py``, optimized for
single-machine (numpy-based) environments.  Functions operate on 1-D numpy
arrays rather than scalar values, yielding significant speedups by replacing
per-element Python loops with bulk numpy operations.

This is an internal API: all functions accept ``np.ndarray`` directly and
perform no input coercion or shape validation.  Callers are responsible for
providing arrays of the correct dtype and dimensionality.

Covers:
  * Discrete encoding / decoding (categorical <-> integer index).
  * Discretization / undiscretization (numerical <-> bin index).
  * Rare-value merging / unmerging (domain compression).
"""

from __future__ import annotations

from dpsynth import domain
import numpy as np


def discrete_encode(
    data: np.ndarray,
    attribute_domain: domain.CategoricalAttribute,
) -> np.ndarray:
  """Maps categorical values to integer indices in ``attribute_domain``.

  Out-of-domain values are mapped to ``attribute_domain.out_of_domain_index``.

  Args:
    data: 1-D array of categorical values.  Must be a homogeneous dtype.
    attribute_domain: The categorical attribute defining the encoding.

  Returns:
    A 1-D integer array of indices into ``attribute_domain.possible_values``.
  """
  lookup = attribute_domain.lookup
  default = attribute_domain.out_of_domain_index
  # Loop over unique values only (typically ≪ len(data) for categoricals),
  # then remap via pure numpy fancy indexing.  Normalizing to str avoids
  # mixed-type comparison errors in np.unique on numpy 2.x (e.g. NaN + str).
  data = np.asarray(data, dtype=str)
  uniq, inv = np.unique(data, return_inverse=True)
  uniq_encoded = np.array([lookup.get(v, default) for v in uniq], dtype=int)
  return uniq_encoded[inv]


def discrete_decode(
    encoded: np.ndarray,
    attribute_domain: domain.CategoricalAttribute,
) -> np.ndarray:
  """Maps integer indices back to categorical values.

  Args:
    encoded: 1-D integer array of indices into attribute_domain.possible_values.
    attribute_domain: The categorical attribute defining the decoding.

  Returns:
    A 1-D object-dtype array of categorical values.
  """
  values = np.array(attribute_domain.possible_values, dtype=object)
  return values[encoded]


def _validate_bin_edges(bin_edges, attribute_domain):
  """Validates bin_edges against the attribute domain."""
  min_, max_ = attribute_domain.min_value, attribute_domain.max_value
  if bin_edges.size == 0:
    return  # 1-bin: entire domain is a single bucket; nothing to validate.
  if bin_edges[0] < min_ or bin_edges[-1] >= max_:
    raise ValueError(f'{bin_edges=} must be within the range [{min_}, {max_}].')
  if np.any(np.diff(bin_edges) <= 0):
    raise ValueError(f'{bin_edges=} must be monotonically increasing.')


def categorical_attribute_from_edges(
    bin_edges: np.ndarray,
    attribute_domain: domain.NumericalAttribute,
) -> domain.CategoricalAttribute:
  """Builds a CategoricalAttribute with interval-string possible_values.

  Args:
    bin_edges: Sorted inner bin edges.
    attribute_domain: The NumericalAttribute describing the data domain.

  Returns:
    A CategoricalAttribute whose possible_values are interval strings.
  """
  min_, max_ = attribute_domain.exclusive_min_value, attribute_domain.max_value
  full_edges = np.r_[min_, bin_edges, max_]
  if attribute_domain.dtype == 'int':
    e = full_edges.astype(int)
    intervals = [f'[{l+1}, {r}]' for l, r in zip(e[:-1], e[1:])]
  else:
    intervals = [f'({l}, {r}]' for l, r in zip(full_edges[:-1], full_edges[1:])]
  if not attribute_domain.clip_to_range:
    intervals = ['OUT_OF_DOMAIN'] + intervals
  return domain.CategoricalAttribute(intervals)  # pyrefly: ignore[bad-argument-count]


def discretize(
    data: np.ndarray,
    bin_edges: np.ndarray,
    attribute_domain: domain.NumericalAttribute,
) -> np.ndarray:
  """Maps numerical values to bin indices via ``np.searchsorted``.

  Bin intervals are right-closed: ``(left, right]``, matching the
  ``pandas.IntervalIndex`` convention.

  Args:
    data: 1-D array of numerical values.
    bin_edges: Sorted inner bin edges. Must be monotonically increasing and
      within ``[min_value, max_value)``.
    attribute_domain: The ``NumericalAttribute`` describing the data domain.

  Returns:
    A 1-D integer array of 0-based bin indices.  When
    ``attribute_domain.clip_to_range`` is ``False``, index 0 represents the
    out-of-domain (``'<OOD>'``) bin and in-domain bins start at 1.
  """
  min_, max_ = attribute_domain.min_value, attribute_domain.max_value
  # Coerce to float64 so object-typed arrays (e.g. containing '<OOD>') work
  # with np.clip / np.isnan without TypeError.
  data = np.asarray(data, dtype=float)
  _validate_bin_edges(bin_edges, attribute_domain)

  if bin_edges.size == 0:
    # Single bin covers the entire domain.
    return np.zeros(len(data), dtype=int)

  if attribute_domain.clip_to_range:
    # Out-of-domain values are clipped to the public range (including nan).
    standardized = np.clip(data, min_, max_)
    standardized = np.where(np.isnan(standardized), min_, standardized)
    return np.searchsorted(bin_edges, standardized, side='left')
  else:
    # Out-of-domain values are mapped to a special "OOD" bucket.
    ood_mask = np.isnan(data) | (data < min_) | (data > max_)
    indices = np.searchsorted(bin_edges, data, side='left')
    return np.where(ood_mask, 0, indices + 1)


def undiscretize(
    bin_indices: np.ndarray,
    bin_edges: np.ndarray,
    attribute_domain: domain.NumericalAttribute,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
  """Maps bin indices back to numerical value (inverse of :func:`discretize`.)

  Args:
    bin_indices: 1-D integer array of bin indices.
    bin_edges: The same sorted inner bin edges used during discretization.
    attribute_domain: The ``NumericalAttribute`` describing the data domain.
    rng: An optional numpy random generator.

  Returns:
    A 1-D array.  For ``'midpoint'`` and ``'sample'`` the dtype is float
    (or int when ``dtype == 'int'`` and all values are in-domain).  For
    ``'interval'`` the dtype is ``object`` (strings).  Out-of-domain bins
    (index 0 when ``clip_to_range`` is ``False``) map to
    ``attribute_domain.resolved_sentinel``.
  """
  rng = np.random.default_rng(rng)
  min_, max_ = attribute_domain.min_value, attribute_domain.max_value
  _validate_bin_edges(bin_edges, attribute_domain)

  if bin_edges.size == 0:
    # Single bin: return domain midpoint for all indices.
    mid = (min_ + max_) / 2.0
    result = np.full(len(bin_indices), mid)
    if attribute_domain.dtype == 'int' and attribute_domain.clip_to_range:
      result = np.ceil(result).astype(int)
    return result
  full_edges = np.r_[min_, bin_edges, max_]
  lefts, rights = full_edges[:-1], full_edges[1:]
  handling = attribute_domain.interval_handling
  sentinel = attribute_domain.resolved_sentinel

  if handling == 'interval':
    # First interval is closed on both sides; the rest are half-open.
    strs = [f'[{lefts[0]}, {rights[0]}]'] + [
        f'({l}, {r}]' for l, r in zip(lefts[1:], rights[1:])
    ]
    values = np.array(strs, dtype=str)
    if not attribute_domain.clip_to_range:
      values = np.r_[np.array(sentinel, dtype=str), values]
    return values[bin_indices]
  elif handling == 'sample':
    if not attribute_domain.clip_to_range:
      ood = bin_indices == 0
      idx = np.where(ood, 0, bin_indices - 1)
      result = np.where(ood, sentinel, rng.uniform(lefts[idx], rights[idx]))
    else:
      result = rng.uniform(lefts[bin_indices], rights[bin_indices])
  elif handling == 'midpoint':
    midpoints = (lefts + rights) / 2.0
    if not attribute_domain.clip_to_range:
      midpoints = np.r_[sentinel, midpoints]
    result = midpoints[bin_indices]
  else:
    raise ValueError(f'Unsupported interval_handling: {handling}')

  if attribute_domain.dtype == 'int':
    result = np.ceil(result)
    if attribute_domain.clip_to_range or not np.isnan(sentinel):
      result = result.astype(int)
  return result


def merge_rare_values(
    data: np.ndarray,
    rare_value_mask: np.ndarray,
) -> tuple[int, np.ndarray]:
  """Maps integer-encoded data to a compressed domain, merging rare values.

  Non-rare values are renumbered contiguously starting from 0; all rare values
  are mapped to the last index in the compressed domain.

  Args:
    data: 1-D integer array in the original domain.
    rare_value_mask: 1-D boolean array indicating which original-domain values
      are rare (``True`` means rare).

  Returns:
    A tuple ``(compressed_size, compressed_data)`` where *compressed_size* is
    the number of bins in the compressed domain and *compressed_data* is a 1-D
    integer array of the same length as *data*.
  """

  num_rare = int(rare_value_mask.sum())
  ncommon = rare_value_mask.size - num_rare
  compressed_size = ncommon + (1 if num_rare >= 1 else 0)
  mapping = np.where(
      rare_value_mask, compressed_size - 1, np.cumsum(~rare_value_mask) - 1
  )

  return compressed_size, mapping[data]


def unmerge_rare_values(
    data: np.ndarray,
    rare_value_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
  """Maps compressed-domain integers back, randomly restoring rare values.

  This is the inverse of :func:`merge_rare_values`. For the merged-rare bin,
  each element is randomly assigned to one of the original rare values.

  Args:
    data: 1-D integer array in the compressed domain.
    rare_value_mask: 1-D boolean array (see :func:`merge_rare_values`).
    rng: A numpy random number generator used for sampling rare values.

  Returns:
    A 1-D integer array in the original domain.
  """

  common_indices = np.flatnonzero(~rare_value_mask)
  result = np.r_[common_indices, -1][data]
  rare_mask = data == common_indices.size

  # If any rare values are present in `data`, restore them randomly
  if rare_mask.any():
    rare_indices = np.flatnonzero(rare_value_mask)
    result[rare_mask] = rng.choice(rare_indices, size=np.sum(rare_mask))

  return result
