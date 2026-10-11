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
"""Pandas dtype to dpsynth domain adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from absl import logging
from dpsynth import domain
import pandas as pd


def _infer_attribute(name, raw_dtype, bounds):
  """Infers a single attribute domain from a pandas dtype."""
  dtype = pd.api.types.pandas_dtype(raw_dtype)
  if isinstance(dtype, pd.CategoricalDtype):
    return domain.CategoricalAttribute(dtype.categories.tolist())
  if pd.api.types.is_bool_dtype(dtype):
    return domain.CategoricalAttribute([False, True])
  is_int = pd.api.types.is_integer_dtype(dtype)
  if is_int or pd.api.types.is_float_dtype(dtype):
    if name not in bounds:
      raise ValueError(f"Numerical bounds must be specified for '{name}'.")
    lo, hi = bounds[name]
    return domain.NumericalAttribute(lo, hi, dtype="int" if is_int else "float")
  if pd.api.types.is_string_dtype(dtype):
    return domain.OpenSetCategoricalAttribute()
  raise ValueError(f"Field '{name}' of dtype {dtype} is not supported.")


def infer_domain(
    dtypes: pd.Series | Mapping[str, Any],
    *,
    numerical_bounds: Mapping[str, tuple[float, float]] | None = None,
    ignore_unsupported_fields: bool = False,
) -> dict[str, domain.AttributeType]:
  """Infers attribute domains from pandas column dtypes.

  For ``pd.CategoricalDtype`` columns, ``dtype.categories`` must originate from
  a public schema rather than calling ``df[col].astype('category')`` on
  sensitive data without differential privacy.

  Args:
    dtypes: A ``df.dtypes`` ``pd.Series`` or mapping from column name to dtype
      specification.
    numerical_bounds: Mapping from column name to ``(min, max)`` bounds for
      numeric columns.
    ignore_unsupported_fields: Whether to skip unsupported or unbounded columns.

  Returns:
    Mapping from column name to ``domain.AttributeType``.

  Raises:
    TypeError: If ``dtypes`` is not a Series or Mapping.
    ValueError: If a column has an unsupported dtype or missing numerical bounds
      and ``ignore_unsupported_fields`` is False.
  """
  if not isinstance(dtypes, (pd.Series, Mapping)):
    raise TypeError(f"Expected a Series or Mapping, got {type(dtypes)}.")
  bounds = numerical_bounds or {}
  attributes: dict[str, domain.AttributeType] = {}
  for name, raw_dtype in dtypes.items():
    try:
      attributes[str(name)] = _infer_attribute(name, raw_dtype, bounds)
    except (TypeError, ValueError) as e:
      if not ignore_unsupported_fields:
        raise ValueError(str(e)) from e
      logging.info("Skipping field %s: %s", name, e)
  return attributes
