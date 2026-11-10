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

"""Cross-attribute constraints for categorical data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from dpsynth import transformations
from dpsynth.core import domain
import mbi
import numpy as np


def _validate(c: Constraint) -> None:
  """Validate a Constraint's fields."""
  if len(c.attribute_names) != len(c.attribute_domains):
    raise ValueError(
        'attribute_names and attribute_domains must have the same length, got'
        f' {len(c.attribute_names)} != {len(c.attribute_domains)}.'
    )
  modes = (
      c.possible_combinations,
      c.impossible_combinations,
      c.functional_dependency,
  )
  n_set = sum(x is not None for x in modes)
  if n_set != 1:
    raise ValueError(
        "Specify exactly one of 'possible_combinations',"
        " 'impossible_combinations', or 'functional_dependency'."
    )
  if c.functional_dependency is not None and len(c.attribute_names) != 2:
    raise ValueError(
        'functional_dependency requires exactly 2 attributes (fine, coarse),'
        f' got {len(c.attribute_names)}.'
    )
  if c.functional_dependency is None:
    combos = c.possible_combinations or c.impossible_combinations
    n_attrs = len(c.attribute_names)
    for combo in combos:  # pyrefly: ignore[not-iterable]
      if len(combo) != n_attrs:
        raise ValueError(
            'Each combination must have length equal to the number of'
            f' attributes ({n_attrs}), got {len(combo)}.'
        )


@dataclasses.dataclass(frozen=True)
class Constraint:
  """A constraint on allowed value combinations across attributes.

  Mirrors :class:`mbi.Constraint` but accepts human-readable values from
  :class:`dpsynth.domain.CategoricalAttribute` instead of integer arrays.
  Exactly one of ``possible_combinations``, ``impossible_combinations``, or
  ``functional_dependency`` must be specified.

  Example Usage:

    >>> d1 = domain.CategoricalAttribute(['GameSuite', 'OfficePro', 'DevTool'])
    >>> d2 = domain.CategoricalAttribute(['Windows', 'Linux', 'MacOS'])
    >>> constraint = Constraint(
    ...     attribute_names=('Software', 'Operating System'),
    ...     attribute_domains=(d1, d2),
    ...     possible_combinations=[
    ...         ('GameSuite', 'Windows'),
    ...         ('OfficePro', 'Windows'),
    ...         ('OfficePro', 'MacOS'),
    ...         ('DevTool', 'Linux'),
    ...         ('DevTool', 'MacOS'),
    ...     ],
    ... )

  Attributes:
    attribute_names: Names of the constrained attributes.
    attribute_domains: Categorical domain for each attribute.
    possible_combinations: Allowed value combinations.
    impossible_combinations: Forbidden value combinations.
    functional_dependency: Dict mapping fine attribute values to coarse
      attribute values. Requires exactly two attributes.
  """

  attribute_names: tuple[str, ...]
  attribute_domains: tuple[domain.CategoricalAttribute, ...]
  possible_combinations: Sequence[tuple[Any, ...]] | None = None
  impossible_combinations: Sequence[tuple[Any, ...]] | None = None
  functional_dependency: Mapping[Any, Any] | None = None

  def __post_init__(self):
    _validate(self)

  def to_mbi(self) -> mbi.Constraint:
    """Convert to an mbi.Constraint."""
    shape = tuple(d.size for d in self.attribute_domains)
    mbi_domain = mbi.Domain(self.attribute_names, shape)
    encoders = [
        transformations.discrete_encoder(d) for d in self.attribute_domains
    ]

    if self.functional_dependency is not None:
      _, coarse_enc = encoders
      fine_values = self.attribute_domains[0].possible_values
      coarse_indices = [
          coarse_enc(self.functional_dependency[v]) for v in fine_values
      ]
      return mbi.Constraint(
          domain=mbi_domain, mapping=np.array(coarse_indices, dtype=np.int32)
      )

    combos = self.possible_combinations or self.impossible_combinations
    encoded = [[enc(v) for enc, v in zip(encoders, c)] for c in combos]  # pyrefly: ignore[not-iterable]
    indices = np.array(encoded, dtype=np.int32)
    if self.possible_combinations is not None:
      return mbi.Constraint(domain=mbi_domain, valid=indices)
    return mbi.Constraint(domain=mbi_domain, invalid=indices)
