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

"""Canonical representation and validation of marginal query workloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import itertools
import math
from typing import TypeAlias

from absl import logging


@dataclasses.dataclass(frozen=True)
class Workload:
  """Canonical representation of a marginal query workload.

  Attributes:
    cliques: Sequence of attribute tuples to preserve.
    weights: Optional positive weights for each clique (defaults to 1.0).
  """

  cliques: Sequence[tuple[str, ...]] = ()
  weights: Sequence[float] | None = None

  def __post_init__(self):
    if isinstance(self.cliques, (str, Mapping)) or not isinstance(
        self.cliques, Iterable
    ):
      raise ValueError(f'Invalid cliques: {self.cliques!r}.')
    cliques: list[tuple[str, ...]] = []
    for cl in self.cliques:
      if isinstance(cl, str) or not isinstance(cl, Sequence):
        raise ValueError(f'Clique must be a sequence of strings, got {cl!r}.')
      clique = tuple(cl)
      if not clique:
        raise ValueError('Workload cliques must be non-empty.')
      if not all(isinstance(col, str) for col in clique):
        raise ValueError(f'Clique elements must be strings, got {clique!r}.')
      if len(set(clique)) != len(clique):
        raise ValueError(f'Clique has duplicate attributes: {clique!r}.')
      cliques.append(clique)
    object.__setattr__(self, 'cliques', tuple(cliques))

    if self.weights is not None:
      weights = tuple(float(w) for w in self.weights)
      if len(weights) != len(cliques):
        raise ValueError(f'Length mismatch: {len(cliques)} != {len(weights)}.')
      for clique, w in zip(cliques, weights):
        if not math.isfinite(w) or w <= 0:
          raise ValueError(f'Weight for {clique!r} must be positive: {w}.')
      object.__setattr__(self, 'weights', weights)

  def as_dict(self) -> dict[tuple[str, ...], float]:
    """Returns a dictionary mapping each clique to its weight."""
    if self.weights is not None:
      return dict(zip(self.cliques, self.weights))
    return {cl: 1.0 for cl in self.cliques}

  @classmethod
  def all_kway(cls, attributes: Iterable[str], degree: int = 3) -> Workload:
    """Returns a workload containing all k-way marginals over attributes."""
    if degree < 1:
      raise ValueError(f'degree must be >= 1, got {degree}.')
    attrs = tuple(attributes)
    k = min(degree, len(attrs)) if attrs else degree
    return cls(list(itertools.combinations(attrs, k)))

  @classmethod
  def from_any(
      cls,
      workload: WorkloadInput | None,
      attributes: Iterable[str] | None = None,
  ) -> Workload | None:
    """Coerces a raw workload into a Workload and optionally validates it."""
    if workload is None:
      return None
    if isinstance(workload, Workload):
      resolved = workload
    elif isinstance(workload, Mapping):
      resolved = cls(list(workload.keys()), list(workload.values()))
    else:
      resolved = cls(list(workload))
    if attributes is not None:
      resolved.validate(attributes)
    return resolved

  def validate(self, attributes: Iterable[str]) -> None:
    """Validates that all workload attributes exist in the schema."""
    attr_set = set(attributes)
    workload_attrs = set().union(*self.cliques) if self.cliques else set()
    unknown = workload_attrs - attr_set
    if unknown:
      raise ValueError(f'Unknown workload attributes: {sorted(unknown)}.')
    uncovered = attr_set - workload_attrs
    if uncovered:
      logging.warning('Attributes not in workload: %s', sorted(uncovered))


WorkloadInput: TypeAlias = (
    Workload | Mapping[tuple[str, ...], float] | Iterable[tuple[str, ...]]
)
