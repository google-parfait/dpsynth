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
"""Tests for pandas adapter."""

from __future__ import annotations

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth import domain
from dpsynth.adapters import pandas as pandas_adapter
import numpy as np
import pandas as pd

CategoricalAttribute = domain.CategoricalAttribute
NumericalAttribute = domain.NumericalAttribute
OpenSetCategoricalAttribute = domain.OpenSetCategoricalAttribute

_BOUNDS = {"user_id": (0.0, 100.0), "score": (0.0, 100.0)}


def _sample_dtypes() -> pd.Series:
  cats = pd.CategoricalDtype(["UNKNOWN", "PENDING", "ACTIVE", "COMPLETED"])
  return pd.DataFrame({
      "user_id": pd.Series([1, 2], dtype="int64"),
      "username": pd.Series(["alice", "bob"], dtype="string"),
      "status": pd.Series(["PENDING", "ACTIVE"], dtype=cats),
      "score": pd.Series([10.5, 95.0], dtype="float64"),
      "is_verified": pd.Series([True, False], dtype="bool"),
  }).dtypes


class PandasAdapterTest(parameterized.TestCase):

  def test_infer_domain(self):
    dtypes = _sample_dtypes()
    attrs = pandas_adapter.infer_domain(dtypes, numerical_bounds=_BOUNDS)
    cats = ["UNKNOWN", "PENDING", "ACTIVE", "COMPLETED"]
    self.assertEqual(attrs["user_id"], NumericalAttribute(0, 100, dtype="int"))
    self.assertEqual(attrs["username"], OpenSetCategoricalAttribute())
    self.assertEqual(attrs["status"], CategoricalAttribute(cats))
    self.assertEqual(attrs["score"], NumericalAttribute(0, 100, dtype="float"))
    self.assertEqual(attrs["is_verified"], CategoricalAttribute([False, True]))

  def test_input_flexibility(self):
    dtypes = _sample_dtypes()
    for x in [dtypes, dtypes.to_dict()]:
      attrs = pandas_adapter.infer_domain(x, numerical_bounds=_BOUNDS)
      self.assertLen(attrs, 5)
      self.assertEqual(attrs["username"], OpenSetCategoricalAttribute())

  @parameterized.parameters(
      (["a", "b", "c"], ["a", "b", "c"]),
      ([10, 20, 30], [10, 20, 30]),
      ([1.5, 2.5], [1.5, 2.5]),
      ([False, True], [False, True]),
  )
  def test_categorical_dtype_value_types(self, categories, expected):
    dtypes = {"col": pd.CategoricalDtype(categories=categories)}
    attrs = pandas_adapter.infer_domain(dtypes)
    self.assertEqual(attrs["col"], CategoricalAttribute(expected))

  @parameterized.parameters(
      ("int8", "int"),
      ("int32", "int"),
      ("int64", "int"),
      ("uint16", "int"),
      ("Int64", "int"),
      (int, "int"),
      (np.int32, "int"),
      ("float32", "float"),
      ("float64", "float"),
      ("Float64", "float"),
      (float, "float"),
      (np.float64, "float"),
  )
  def test_numeric_dtypes(self, raw_dtype, kind):
    dtypes, bounds = {"val": raw_dtype}, {"val": (-10.0, 10.0)}
    attrs = pandas_adapter.infer_domain(dtypes, numerical_bounds=bounds)
    self.assertEqual(attrs["val"], NumericalAttribute(-10, 10, dtype=kind))

  @parameterized.parameters("bool", "boolean", bool, np.bool_)
  def test_boolean_dtypes(self, raw_dtype):
    attrs = pandas_adapter.infer_domain({"flag": raw_dtype})
    self.assertEqual(attrs["flag"], CategoricalAttribute([False, True]))

  @parameterized.parameters("string", "str", "object", str, object)
  def test_string_dtypes(self, raw_dtype):
    attrs = pandas_adapter.infer_domain({"text": raw_dtype})
    self.assertEqual(attrs["text"], OpenSetCategoricalAttribute())

  def test_missing_bounds_raises_or_skips(self):
    dtypes = _sample_dtypes()
    with self.assertRaisesRegex(ValueError, "Numerical bounds"):
      pandas_adapter.infer_domain(dtypes)

    attrs = pandas_adapter.infer_domain(dtypes, ignore_unsupported_fields=True)
    self.assertEqual(list(attrs.keys()), ["username", "status", "is_verified"])

  @parameterized.parameters(
      ("datetime64[ns]",),
      ("timedelta64[ns]",),
      ("complex128",),
      ("not_a_valid_dtype",),
      (pd.CategoricalDtype(categories=[]),),
      (pd.CategoricalDtype(categories=pd.to_datetime(["2020-01-01"])),),
  )
  def test_unsupported_dtype_raises_or_skips(self, bad_dtype):
    with self.assertRaises(ValueError):
      pandas_adapter.infer_domain({"bad": bad_dtype})

    dtypes = {"bad": bad_dtype, "good": "string"}
    attrs = pandas_adapter.infer_domain(dtypes, ignore_unsupported_fields=True)
    self.assertEqual(attrs, {"good": OpenSetCategoricalAttribute()})

  @parameterized.parameters(
      (pd.DataFrame({"a": [1, 2]}),),
      ("not_a_series_or_mapping",),
  )
  def test_invalid_input_type_raises(self, bad_input: Any):
    with self.assertRaisesRegex(TypeError, "Expected a Series or Mapping"):
      pandas_adapter.infer_domain(bad_input)


if __name__ == "__main__":
  absltest.main()
