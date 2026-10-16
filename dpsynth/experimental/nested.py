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

"""Differentially private synthesizer for nested/typed tabular data.

Privacy model — neighboring relation and participation assumptions:

  Each individual contributes exactly one row to exactly one type-table.
  Two datasets are *neighbors* if they differ in the addition or removal
  of a single such row.  Because a row appears in only one type-table,
  the per-type detail models operate on disjoint partitions and satisfy
  parallel composition: the privacy cost of the detail level equals the
  cost of a single type, not the sum over types.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth import data_generation_v3
from dpsynth import discrete_mechanisms
from dpsynth import domain
import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class NestedSynthesisResult:
  """Synthetic models and per-type data resulting from DP measurement."""

  shared_result: data_generation_v3.DataGenerationResult
  detail_results: Mapping[str, data_generation_v3.DataGenerationResult]
  synthetic_data: Mapping[str, pd.DataFrame]


def _resample_to_size(rng, df, n):
  """Shuffles df and tiles it cyclically to exactly n rows."""
  m = len(df)
  # This should not happen, since DPSynth will always generate at least one row.
  assert m > 0, "input DataFrame cannot be empty"
  if m == n:
    return df.reset_index(drop=True)
  shuffled = df.sample(frac=1, random_state=int(rng.integers(2**31)))
  indices = np.arange(n) % m
  return shuffled.reset_index(drop=True).iloc[indices].reset_index(drop=True)


def _stitch_synthetic_output(
    rng,
    synthetic_shared,
    synthetic_details,
    type_vocabulary,
):
  """Stitches shared and per-type synthetic attributes into typed records."""
  # Note: This is post-processing of already DP-synthesized data.
  output: dict[str, pd.DataFrame] = {}
  for type_name in type_vocabulary:
    type_mask = synthetic_shared["_type"] == type_name
    type_shared = synthetic_shared.loc[type_mask].drop(columns=["_type"])
    n_synthetic = len(type_shared)
    if n_synthetic == 0:
      continue

    type_shared = type_shared.reset_index(drop=True)
    if type_name in synthetic_details:
      detail = synthetic_details[type_name]
      detail = _resample_to_size(rng, detail, n_synthetic)
      # Assumes conditional independence: P(detail | type, shared) ≈
      # P(detail | type). Could be tightened with shared alignment keys.
      output[type_name] = pd.concat([type_shared, detail], axis=1)
    else:
      output[type_name] = type_shared

  logging.info("[DPSynth Nested]: Generated records for %d types.", len(output))
  return output


@dataclasses.dataclass(frozen=True)
class NestedSchema:
  """Schema specification for nested tabular data.

  Attributes:
    shared_schema: Schema for columns shared across all record types.
    per_type_schemas: Mapping from record type name to type-specific schema.
  """

  shared_schema: domain.Schema
  per_type_schemas: Mapping[str, domain.Schema]

  @property
  def type_vocabulary(self) -> Sequence[str]:
    """Returns the list of type names from per_type_schemas keys."""
    return list(self.per_type_schemas.keys())


@dataclasses.dataclass(frozen=True)
class NestedTabularMechanism(api.CalibratedMechanism):
  """Calibrated mechanism for nested tabular data generation."""

  schema: NestedSchema
  type_vocabulary: Sequence[str]
  shared_synth: data_generation_v3.TabularMechanism
  detail_synths: Mapping[str, data_generation_v3.TabularMechanism]

  # Note: `detail_rho` must exactly match the zCDP guarantee of the individual
  # mechanisms in `detail_synths`. This is because NestedTabularMechanism
  # implements parallel privacy accounting across the detail_synths manually
  # utilizing this detail_rho float. Consequently, directly constructing this
  # mechanism manually (without going through the `configure(...)` API) is
  # potentially dangerous/incorrect if detail_rho is not perfectly aligned with
  # detail_synths.
  detail_rho: float
  detail_delta: float

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed DpEvent for the full mechanism."""
    # supports it, instead of falling back to a ZCDpEvent.
    events = [self.shared_synth.dp_event]
    if self.detail_rho is not None and self.detail_rho > 0:
      base = dp_accounting.ZCDpEvent(self.detail_rho)
      failure = dp_accounting.dp_event.EpsilonDeltaDpEvent(0, self.detail_delta)
      composed = dp_accounting.ComposedDpEvent([base, failure])
      events.append(base if self.detail_delta == 0 else composed)
    return dp_accounting.ComposedDpEvent(events)

  def __call__(
      self,
      rng: np.random.Generator,
      data: Mapping[str, pd.DataFrame],
  ) -> NestedSynthesisResult:
    """Runs the mechanism on per-type DataFrames, returns synthetic records."""
    shared_cols = list(self.schema.shared_schema.attributes)

    types = list(self.type_vocabulary)
    data = dict(data)
    for t in types:
      if t not in data or data[t].empty:
        columns = shared_cols + list(self.schema.per_type_schemas[t].attributes)
        data[t] = pd.DataFrame(columns=columns)

    shared_dfs = [
        data[t][shared_cols].assign(_type=t) for t in types if len(data[t]) > 0
    ]
    shared_df = (
        pd.concat(shared_dfs, ignore_index=True)
        if shared_dfs
        else pd.DataFrame(columns=shared_cols + ["_type"])
    )
    assert isinstance(shared_df, pd.DataFrame)
    shared_result = self.shared_synth(rng, shared_df)

    synthetic_details: dict[str, pd.DataFrame] = {}
    detail_results: dict[str, data_generation_v3.DataGenerationResult] = {}
    for type_name, synth in self.detail_synths.items():
      detail_cols = list(self.schema.per_type_schemas[type_name].attributes)
      sub_df = data[type_name].reindex(columns=detail_cols)
      res = synth(rng, sub_df)
      synthetic_details[type_name] = res.synthetic_data
      detail_results[type_name] = res

    stitched = _stitch_synthetic_output(
        rng=rng,
        synthetic_shared=shared_result.synthetic_data,
        synthetic_details=synthetic_details,
        type_vocabulary=self.type_vocabulary,
    )

    return NestedSynthesisResult(
        shared_result=shared_result,
        detail_results=detail_results,
        synthetic_data=stitched,
    )


@dataclasses.dataclass(frozen=True)
class NestedTabularConfig(api.MechanismConfig):
  """DP synthesizer for datasets with typed records.

  Each record has a type and type-specific attributes. Different types may
  have different schemas. The synthesizer learns:

  1. **Shared model (Model 1):** Joint distribution over record types and
     shared attributes using TabularConfig.
  2. **Per-type detail models (Model 2):** Independent TabularConfig
     per type for type-specific attributes, under parallel composition.

  Privacy cost is: sequential composition of Model 1 + Model 2. Model 2
  costs nothing extra under parallel composition (each type's data is
  disjoint), so only one representative type's cost is counted.

  Example:

  >>> import numpy as np
  >>> import pandas as pd
  >>> from dpsynth import domain
  >>> from dpsynth.experimental import nested
  >>> Cat = domain.CategoricalAttribute
  >>> schema = nested.NestedSchema(
  ...     shared_schema=domain.Schema({'platform': Cat(['web', 'mobile'])}),
  ...     per_type_schemas={
  ...         'click': domain.Schema({'element': Cat(['button', 'link'])}),
  ...         'purchase': domain.Schema({'amount': Cat(['low', 'high'])}),
  ...     },
  ... )
  >>> synth = nested.NestedTabularConfig()
  >>> calibrated = synth.configure(schema, budget=1.0)
  >>> rng = np.random.default_rng(42)
  >>> rows = {'platform': ['web', 'mobile'] * 20}
  >>> click_df = pd.DataFrame({**rows, 'element': ['button', 'link'] * 20})
  >>> purchase_df = pd.DataFrame({**rows, 'amount': ['low', 'high'] * 20})
  >>> data = {'click': click_df, 'purchase': purchase_df}
  >>> result = calibrated(rng, data)  # doctest: +SKIP
  >>> sorted(result.synthetic_data)  # doctest: +SKIP
  ['click', 'purchase']

  Attributes:
    shared_budget_fraction: Fraction of total rho for Model 1.
    shared_mechanism: Discrete mechanism for the shared model.
    detail_mechanism: Discrete mechanism for per-type models.
    init_budget_fraction: Within each TabularConfig, fraction for initializers.
  """

  shared_budget_fraction: float = 0.5
  shared_mechanism: api.MechanismConfig = dataclasses.field(
      default_factory=discrete_mechanisms.MSTConfig
  )
  detail_mechanism: api.MechanismConfig = dataclasses.field(
      default_factory=discrete_mechanisms.MSTConfig
  )
  init_budget_fraction: float = 0.1

  def configure(  # pyrefly: ignore[bad-override]
      self,
      schema: NestedSchema,
      *,
      budget: float,
      delta: float = 0.0,
  ) -> NestedTabularMechanism:
    """Returns a configured NestedTabularMechanism with the given budget."""
    if not isinstance(schema, NestedSchema):
      raise TypeError(f"Expected NestedSchema, got {type(schema).__name__}")

    # Additive zCDP split; each type gets full rho_detail
    # (parallel composition over disjoint type partitions).
    rho_shared = self.shared_budget_fraction * budget
    rho_detail = (1 - self.shared_budget_fraction) * budget
    delta_shared = delta * self.shared_budget_fraction
    delta_detail = delta * (1 - self.shared_budget_fraction)

    shared_domains = dict(schema.shared_schema.attributes)
    shared_domains["_type"] = domain.CategoricalAttribute(
        possible_values=schema.type_vocabulary
    )
    shared_config = data_generation_v3.TabularConfig(
        domains=shared_domains,
        discrete_mechanism=self.shared_mechanism,
        init_budget_fraction=self.init_budget_fraction,
    )
    shared_synth = shared_config.configure(
        budget=rho_shared,
        delta=delta_shared,
    )

    detail_synths = {}
    for type_name, type_schema in schema.per_type_schemas.items():
      if not type_schema.attributes:
        continue
      config = data_generation_v3.TabularConfig(
          domains=type_schema.attributes,
          discrete_mechanism=self.detail_mechanism,
          init_budget_fraction=self.init_budget_fraction,
      )
      detail_synths[type_name] = config.configure(
          budget=rho_detail,
          delta=delta_detail,
      )

    return NestedTabularMechanism(
        schema=schema,
        type_vocabulary=schema.type_vocabulary,
        shared_synth=shared_synth,
        detail_synths=detail_synths,
        detail_rho=rho_detail,
        detail_delta=delta_detail,
    )


# Set an alias for legacy users who try to invoke NestedTabularSynthesizer
NestedTabularSynthesizer = NestedTabularConfig
