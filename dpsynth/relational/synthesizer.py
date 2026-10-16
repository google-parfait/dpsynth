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

"""Multi-table relational differential privacy synthesizer."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping, Sequence
import dataclasses
import math
from typing import Any, Literal

from absl import logging
import dp_accounting
from dpsynth import _calibration
from dpsynth import api
from dpsynth import data_generation_v3
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth.discrete_mechanisms import common as dm_common
from dpsynth.local_mode import initialization
from dpsynth.local_mode import primitives
from dpsynth.local_mode import vectorized_transformations as vtx
from dpsynth.relational import domain as rel_domain
from dpsynth.relational import post_processing
from dpsynth.relational import transformations
import mbi
import numpy as np
import pandas as pd

# pylint: disable=unused-import
_LOGGING_UNUSED = logging
# pylint: enable=unused-import


def _validate_input_table_columns(
    domains: Mapping[str, domain.Schema],
    foreign_keys: Sequence[rel_domain.ForeignKeyRelation],
    table_columns: Mapping[str, Collection[str]],
) -> None:
  """Validates that all configured tables, schema columns, and keys exist.

  Accepts only schema metadata, ensuring zero access to sensitive records.

  Args:
    domains: Mapping from table name to per-column AttributeType schemas.
    foreign_keys: Sequence of foreign key relationships.
    table_columns: Mapping from table name to collection of column names.

  Raises:
    ValueError: If a table or required column is missing.
  """
  for table_name, schema in domains.items():
    if table_name not in table_columns:
      raise ValueError(
          f'Table {table_name!r} not found in input data. Available:'
          f' {list(table_columns.keys())}'
      )
    cols = table_columns[table_name]
    for col in schema:
      if col not in cols:
        raise ValueError(
            f'Column {col!r} not found in table {table_name!r}. Available:'
            f' {list(cols)}'
        )
  for fk in foreign_keys:
    if fk.parent_primary_key not in table_columns[fk.parent_table]:
      raise ValueError(
          f'Parent primary key column {fk.parent_primary_key!r} not found in'
          f' table {fk.parent_table!r}.'
      )
    if fk.child_foreign_key not in table_columns[fk.child_table]:
      raise ValueError(
          f'Child foreign key column {fk.child_foreign_key!r} not found in'
          f' table {fk.child_table!r}.'
      )


def _preprocess_weighted_tables(
    tables: Mapping[str, pd.DataFrame],
    hierarchy: Sequence[tuple[int, str, rel_domain.ForeignKeyRelation | None]],
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
  """Computes standalone hierarchical weights and filters tables to active rows.

  Args:
    tables: Mapping from table name to input DataFrame.
    hierarchy: Ordered topological synthesis levels from
      `topological_sort_hierarchy()`.
    rng: Random number generator for child record truncation.

  Returns:
    A tuple of (filtered_tables, filtered_weights) where filtered_tables
    contains only active rows with positive weight, and filtered_weights
    contains the 1D float64 weights for each filtered DataFrame.
  """
  raw_weights = transformations.compute_hierarchical_weights(
      tables, hierarchy, rng=rng
  )
  filtered_tables: dict[str, pd.DataFrame] = {}
  filtered_weights: dict[str, np.ndarray] = {}
  for table_name, df in tables.items():
    w = raw_weights[table_name]
    active_mask = w > 0.0
    if active_mask.all():
      filtered_tables[table_name] = df
      filtered_weights[table_name] = w
    else:
      filtered_tables[table_name] = df.loc[active_mask]
      filtered_weights[table_name] = w[active_mask]
  return filtered_tables, filtered_weights


def _measure_root_total_count(
    rng: np.random.Generator,
    root_record_count: int,
    total_count_sigma: float,
) -> tuple[float, mbi.LinearMeasurement]:
  """Measures the root parent table total count with Gaussian noise (Delta = 1).

  Formal Guarantees:
    - Root-Anchored Population Count: Only the root table total count is
      perturbed and measured.
    - Non-Negativity: Truncated to a minimum of 1.0 (estimated_total >= 1.0).
    - Measurement Variance: Standard deviation is total_count_sigma.

  Args:
    rng: NumPy random generator.
    root_record_count: Number of records in the root parent table.
    total_count_sigma: Gaussian noise sigma for root count.

  Returns:
    A tuple of (estimated_total, total_measurement) where total_measurement
    is an mbi.LinearMeasurement over clique ().
  """
  noisy_total = primitives.gaussian_mechanism(
      rng,
      root_record_count,
      sigma=total_count_sigma,
      l2_sensitivity=1.0,
  )
  total = max(1.0, float(noisy_total))
  total_measurement = mbi.LinearMeasurement(
      np.array([total]),
      (),
      stddev=total_count_sigma,
  )
  return total, total_measurement


def _run_single_col_initializer(
    init: api.CalibratedMechanism,
    rng: np.random.Generator,
    data: np.ndarray,
    weights: np.ndarray,
) -> initialization.ColumnMeasurement:
  """Runs a single column initializer on weighted standalone table data.

  Formal Guarantees:
    - Sensitivity Alignment: Weighted histogram evaluation with sum(w) = N_root
      ensures unit sensitivity (Delta = 1.0) without noise scaling.
    - Initializer Contract: Dispatches directly to from_summary() on
      NumericalInitializer, CategoricalInitializer, or OpenSetInitializer.

  Args:
    init: Calibrated column initializer.
    rng: NumPy random generator.
    data: 1D array of column values.
    weights: 1D float array of row sensitivity weights.

  Returns:
    A ColumnMeasurement containing the discovered categorical attribute,
    optional bin edges, and optional noisy 1-way marginal measurement.

  Raises:
    ValueError: If init is not a supported initializer type.
  """
  if isinstance(init, initialization.NumericalInitializer):
    attr = init.attribute
    values = np.asarray(data, dtype=float)
    if attr.clip_to_range:
      values = np.where(np.isnan(values), attr.min_value, values)
      ood_count = 0.0
    else:
      in_domain = (values >= attr.min_value) & (values <= attr.max_value)
      ood_count = float(weights[~in_domain].sum())
      values, weights = values[in_domain], weights[in_domain]
    if attr.dtype == 'int':
      values = np.round(values)
    lower, upper, gs = init.grid_spec
    delta = (upper - lower) / (gs - 1)
    indices = initialization.encode_to_grid(values, lower, upper, delta)
    counts = np.bincount(indices, weights=weights, minlength=gs)
    return init.from_summary(rng, counts, ood_count=ood_count)

  if isinstance(init, initialization.CategoricalInitializer):
    encoded = vtx.discrete_encode(data, init.attribute)
    counts = np.bincount(
        encoded, weights=weights, minlength=init.attribute.size
    )
    return init.from_summary(rng, counts)

  if isinstance(init, initialization.OpenSetInitializer):
    values = np.asarray(data, dtype=str)
    unique_values, inverse = np.unique(values, return_inverse=True)
    counts = np.bincount(inverse, weights=weights)
    return init.from_summary(rng, unique_values, counts)

  raise ValueError(f'Unsupported initializer type: {type(init)}')


def _run_table_initializers(
    calibrated_initializers: Mapping[
        str, Mapping[str, api.CalibratedMechanism]
    ],
    rng: np.random.Generator,
    tables: Mapping[str, pd.DataFrame],
    weights: Mapping[str, np.ndarray],
) -> dict[str, dict[str, initialization.ColumnMeasurement]]:
  """Runs column initializers across all tables on weighted data.

  Args:
    calibrated_initializers: Mapping from table and column name to calibrated
      initializers.
    rng: NumPy random generator.
    tables: Mapping from table name to filtered active DataFrames.
    weights: Mapping from table name to 1D sensitivity weights.

  Returns:
    A nested mapping from table and column name to its ColumnMeasurement.
  """
  results: dict[str, dict[str, initialization.ColumnMeasurement]] = {}
  for table_name, table_inits in calibrated_initializers.items():
    table_results: dict[str, initialization.ColumnMeasurement] = {}
    table_df = tables[table_name]
    table_w = weights[table_name]
    for col_name, init in table_inits.items():
      table_results[col_name] = _run_single_col_initializer(
          init=init,
          rng=rng,
          data=table_df[col_name].to_numpy(),
          weights=table_w,
      )
    results[table_name] = table_results
  return results


def _encode_and_compress_tables(
    domains: Mapping[str, domain.Schema],
    table_measurements: Mapping[
        str, Mapping[str, initialization.ColumnMeasurement]
    ],
    tables: Mapping[str, pd.DataFrame],
    weights: Mapping[str, np.ndarray],
    compress_columns: bool = True,
) -> tuple[
    dict[str, data_generation_v3.TabularCodec],
    dict[str, mbi.Dataset],
    dict[str, dict[str, np.ndarray]],
    dict[str, list[mbi.LinearMeasurement]],
]:
  """Encodes tables to mbi.Datasets and applies domain compression.

  Formal Guarantees:
    - Row Independence (No Cross-Example Mixing): Discretization encoding and
      domain compression operate strictly row-by-row within each table. Each
      input record maps to max 1 output row in its respective mbi.Dataset,
      so no two distinct input records can affect the same output row.
    - Sensitivity Preservation: Domain compression mappings are derived purely
      from DP one-way marginal measurements (post-processing), guaranteeing that
      compression does not consume additional privacy budget.

  Args:
    domains: Mapping from table name to per-column AttributeType schemas.
    table_measurements: Mapping from table and column name to ColumnMeasurement.
    tables: Mapping from table name to active filtered DataFrames.
    weights: Mapping from table name to 1D float sensitivity weights.
    compress_columns: Whether to compress rare domain categories (< 3*sigma).

  Returns:
    A tuple of (codecs, compressed_datasets, compression_mappings,
    one_ways_by_table).
  """
  codecs: dict[str, data_generation_v3.TabularCodec] = {}
  compressed_datasets: dict[str, mbi.Dataset] = {}
  compression_mappings: dict[str, dict[str, np.ndarray]] = {}
  one_ways_by_table: dict[str, list[mbi.LinearMeasurement]] = {}

  for table_name, schema in domains.items():
    codec = data_generation_v3.TabularCodec.from_measurements(
        table_measurements[table_name], schema
    )
    raw_dataset = codec.encode(tables[table_name])
    dataset = mbi.Dataset(
        raw_dataset.data, raw_dataset.domain, weights=weights[table_name]
    )
    one_ways = codec.one_way_measurements()
    raw_mappings = dm_common.compression_mappings(
        one_ways, compress_columns=compress_columns
    )
    mappings: dict[str, np.ndarray] = {
        str(col): arr for col, arr in raw_mappings.items()
    }
    if mappings:
      dataset = dataset.compress(mappings)
      one_ways = [m.compress(mappings, dataset.domain) for m in one_ways]

    codecs[table_name] = codec
    compressed_datasets[table_name] = dataset
    compression_mappings[table_name] = mappings
    one_ways_by_table[table_name] = one_ways

  return codecs, compressed_datasets, compression_mappings, one_ways_by_table


@dataclasses.dataclass(frozen=True)
class PreprocessedTables:
  """Engine-agnostic container for preprocessed relational tables.

  Attributes:
    compressed_datasets: Mapping from table name to discrete compressed
      mbi.Dataset.
    table_keys: Mapping from table name to dict of 1D key arrays (PKs and FKs).
    column_codecs: Mapping from table name to TabularCodec for
      encoding/decoding.
    compression_mappings: Mapping from table name to column compression dicts.
    noisy_root_total: Noisy root total count (N_root >= 1.0).
    root_total_measurement: 0-way mbi.LinearMeasurement on ().
    one_way_measurements: Mapping from table name to compressed 1-way marginals.
  """

  compressed_datasets: Mapping[str, mbi.Dataset]
  table_keys: Mapping[str, Mapping[str, np.ndarray]]
  column_codecs: Mapping[str, data_generation_v3.TabularCodec]
  compression_mappings: Mapping[str, Mapping[str, np.ndarray]]
  noisy_root_total: float
  root_total_measurement: mbi.LinearMeasurement
  one_way_measurements: Mapping[str, Sequence[mbi.LinearMeasurement]]


def _run_table_preprocessing(
    mechanism: MultiTableMechanism,
    rng: np.random.Generator,
    data: Mapping[str, pd.DataFrame],
) -> PreprocessedTables:
  """Executes standalone table preprocessing into compressed mbi.Datasets."""
  _validate_input_table_columns(
      mechanism.domains,
      mechanism.foreign_keys,
      {table: df.columns for table, df in data.items()},
  )

  hierarchy = rel_domain.topological_sort_hierarchy(
      list(mechanism.domains.keys()), mechanism.foreign_keys
  )
  root_table = hierarchy[0][1]

  filtered_tables, weights = _preprocess_weighted_tables(
      data, hierarchy, rng=rng
  )

  table_keys: dict[str, dict[str, np.ndarray]] = {}
  for table_name, df in filtered_tables.items():
    table_keys[table_name] = {
        col: df[col].to_numpy()
        for col in df.columns
        if col not in mechanism.domains[table_name]
    }

  noisy_root_total, root_measurement = _measure_root_total_count(
      rng,
      root_record_count=len(filtered_tables[root_table]),
      total_count_sigma=mechanism.total_count_sigma,
  )

  table_measurements = _run_table_initializers(
      mechanism.calibrated_initializers,
      rng=rng,
      tables=filtered_tables,
      weights=weights,
  )

  codecs, datasets, mappings, one_ways = _encode_and_compress_tables(
      mechanism.domains,
      table_measurements=table_measurements,
      tables=filtered_tables,
      weights=weights,
  )

  return PreprocessedTables(
      compressed_datasets=datasets,
      table_keys=table_keys,
      column_codecs=codecs,
      compression_mappings=mappings,
      noisy_root_total=noisy_root_total,
      root_total_measurement=root_measurement,
      one_way_measurements=one_ways,
  )


def _create_table_initializers(
    domains: Mapping[str, domain.Schema],
    numerical_bins: int,
) -> dict[str, dict[str, api.MechanismConfig]]:
  """Creates per-table and per-column initializers from relational schemas."""
  return {
      table: data_generation_v3.create_initializers(schema, numerical_bins)
      for table, schema in domains.items()
  }


def _compute_table_col_deltas(
    domains: Mapping[str, domain.Schema],
    delta: float,
) -> dict[str, dict[str, float]]:
  """Splits thresholding delta additively across open-set columns in all tables.

  DP Note: Only open-set categorical attributes consume delta (for Gaussian
  partition selection thresholding). Categorical and numerical attributes
  operate under pure zCDP (delta = 0.0).

  Args:
    domains: Mapping from table names to per-column AttributeType schemas.
    delta: Total DP delta for partition selection thresholding.

  Returns:
    A nested mapping from table name and column name to its allocated delta.

  Raises:
    ValueError: If open-set columns are present but delta <= 0.

  Formal Guarantees:
    - Only open-set categorical attributes consume delta
    - Categorical and numerical attributes operate under pure zCDP (delta = 0.0)
    - Sum of per-column deltas across all tables equals delta.
    - Invariance: If no open-set columns exist, all per-column deltas are 0.0.
  """
  num_open_set = 0
  for schema in domains.values():
    for attr in schema.values():
      if isinstance(attr, domain.OpenSetCategoricalAttribute):
        num_open_set += 1
  if num_open_set > 0 and delta <= 0:
    raise ValueError(
        'delta must be positive when open-set categorical attributes are'
        ' present. It is used for Gaussian partition selection.'
    )
  per_col_delta = delta / num_open_set if num_open_set > 0 else 0.0
  return {
      table: {
          col: (
              per_col_delta
              if isinstance(attr, domain.OpenSetCategoricalAttribute)
              else 0.0
          )
          for col, attr in schema.items()
      }
      for table, schema in domains.items()
  }


def _compute_link_sensitivities(
    hierarchy: Sequence[tuple[int, str, rel_domain.ForeignKeyRelation | None]],
) -> dict[str, int]:
  """Computes cascading sensitivity (Delta_k) per relational link.

  Overview:
    Relational multi-table synthesis generates data level-by-level down the
    foreign key hierarchy (e.g. Household -> Person -> Activity) using pairwise
    exploration datasets and wide graphical models stitched sequentially via
    quantile copula matching.

  Formal Guarantees:
    - Root Privacy Unit: Differential privacy is guaranteed with respect to the
      root parent entity (e.g. Household).
    - Descendant Bounded Impact: Modifying a single root parent record cascades
      to at most prod_{j=1}^{k-1} s_j immediate parent records in Link k, where
      s_j is the group capacity bound (max_children_per_parent) of ancestor j.
    - Sensitivity Scaling Soundness: Scaling discrete mechanism noise by
      Delta_k = prod s_ancestors strictly preserves root-level differential
      privacy across all child and subchild tables without materializing
      Cartesian joins.

  Args:
    hierarchy: Ordered topological synthesis levels from
      `topological_sort_hierarchy()`.

  Returns:
    A mapping from link name (f'{parent}->{child}') to its cascading integer
    sensitivity bound Delta_k.

  Example:
    Household -> Person (s_1 = 3) -> Activity (s_2 = 2):
      - 'Household->Person': Delta_1 = 1
      - 'Person->Activity': Delta_2 = s_1 = 3
    Result:
      {'Household->Person': 1, 'Person->Activity': 3}
  """
  cumulative_capacity: dict[str, int] = {}
  link_sensitivities: dict[str, int] = {}

  for _, table_name, fk in hierarchy:
    if fk is None:
      cumulative_capacity[table_name] = 1
    else:
      parent_capacity = cumulative_capacity[fk.parent_table]
      link_name = f'{fk.parent_table}->{fk.child_table}'
      link_sensitivities[link_name] = parent_capacity
      cumulative_capacity[table_name] = (
          parent_capacity * fk.max_children_per_parent
      )

  return link_sensitivities


def _fit_and_sample_wide_link_mrf(
    wide_domain: mbi.Domain,
    wide_measurements: Sequence[mbi.LinearMeasurement],
    wide_constraints: Sequence[mbi.Constraint],
    num_rows: int,
    iters: int = 5000,
) -> mbi.Dataset:
  """Fits an MRF on wide domain and samples num_rows wide records.

  Args:
    wide_domain: Generation mbi.Domain with s child slots.
    wide_measurements: Symmetrized noisy linear measurements.
    wide_constraints: Monolithic linear-chain constraints on child slots.
    num_rows: Number of wide family records to sample (>= 0).
    iters: Number of Mirror Descent iterations for PGM estimation.

  Returns:
    An mbi.Dataset containing num_rows sampled wide family records.
  """
  if num_rows <= 0:
    return mbi.Dataset.synthetic(wide_domain, 0)

  estimator = mbi.estimation.MirrorDescent()
  measurements_list = list(wide_measurements)
  callback_fn = mbi.callbacks.default(measurements_list, wide_domain)
  model = estimator.estimate(
      wide_domain,
      measurements_list,
      iters=iters,
      callback_fn=callback_fn,
      constraints=list(wide_constraints),
  )
  return model.synthetic_data(rows=num_rows)


@dataclasses.dataclass(frozen=True)
class SynthesizedLinkResult:
  """Results from synthesizing a single relational parent-child link.

  Attributes:
    unstacked_child_dataset: Unstacked active child records in child domain.
    parent_row_indices: 1D int array mapping each unstacked child record to its
      parent row index in the parent table.
    synth_parent_dataset: Synthesized parent dataset (extracted only for root
      table on Level 1; None for downstream links).
    discrete_mechanism_result: Optional result from the discrete mechanism.
  """

  unstacked_child_dataset: mbi.Dataset
  parent_row_indices: np.ndarray
  synth_parent_dataset: mbi.Dataset | None = None  # Needed for the root table.
  discrete_mechanism_result: Any | None = None


def _synthesize_relational_link(
    parent_dataset: mbi.Dataset,
    child_dataset: mbi.Dataset,
    parent_primary_keys: Sequence[Hashable],
    child_foreign_keys: Sequence[Hashable],
    fk_relation: rel_domain.ForeignKeyRelation,
    discrete_mechanism: api.CalibratedMechanism,
    num_permutation_slots: int,
    strategy: Literal['empty_token', 'size_sliced'],
    rng: np.random.Generator,
    synth_parents: mbi.Dataset | None = None,
    noisy_root_total: float = 1.0,
) -> SynthesizedLinkResult:
  """Synthesizes a single parent-to-child relational link down the hierarchy.

  Formal Guarantees:
    - Private Exploration on Real Data: Constructs permuted exploration table
      directly from the private parent and child tables with bounded capacity
      s = max_children_per_parent. The calibrated discrete mechanism measures
      queries under cascading link sensitivity Delta_k = prod_{i=1}^{k-1} s_i,
      using o-slot linear-chain constraints to avoid wasting privacy budget on
      invalid/mixed states.

  Post-Processing Guarantees:
    - Noisy exploration measurements are symmetrized across all s generation
      slots via data-independent linear combinations, and Private-PGM Mirror
      Descent is fitted on them with deterministic s-slot constraints

    - Downstream Linking on Synthetic Parents: Downstream child records are
      coupled to the upstream synthesized parent records (synth_parents) via
      Copula Quantile Matching (within-bin random tie-breaking and
      lexicographical sorting). This achieves exact 1-to-1 family coupling
      (N = |synth_parents|) and 0% orphan records without accessing private
      parent records during the generation/linking phase.

  Args:
    parent_dataset: Sensitive preprocessed parent table mbi.Dataset.
    child_dataset: Sensitive preprocessed child table mbi.Dataset.
    parent_primary_keys: Sequence of parent primary key identifiers.
    child_foreign_keys: Sequence of child foreign key references.
    fk_relation: ForeignKeyRelation specifying capacity bound s.
    discrete_mechanism: Calibrated discrete mechanism for this link.
    num_permutation_slots: Permutation exploration slot count (o).
    strategy: Exploration strategy ('empty_token' or 'size_sliced').
    rng: NumPy random generator.
    synth_parents: Optional upstream synthesized parent dataset to couple with.
      If None (e.g. root table at Level 1), row count is anchored by
      noisy_root_total.
    noisy_root_total: Noisy root parent total count (for Level 1 root anchors).

  Returns:
    A SynthesizedLinkResult containing unstacked child dataset, parent row index
    mapping, extracted root parent dataset (if Level 1), and diagnostics.
  """
  exploration_dataset = transformations.build_permuted_exploration_dataset(
      parent_dataset=parent_dataset,
      child_dataset=child_dataset,
      parent_primary_keys=parent_primary_keys,
      child_foreign_keys=child_foreign_keys,
      max_group_size=fk_relation.max_children_per_parent,
      num_permutation_slots=num_permutation_slots,
      strategy=strategy,
  )

  if strategy == 'empty_token':
    exploration_constraints = (
        post_processing.create_slot_linear_chain_constraints(
            child_domain=child_dataset.domain,
            num_permutation_slots=num_permutation_slots,
        )
    )
  else:
    exploration_constraints = ()

  mech_res: Any = discrete_mechanism(
      rng=rng,
      data=exploration_dataset,
      constraints=exploration_constraints,
  )
  assert hasattr(mech_res, 'measurements')

  # wide_measurements = post_processing.symmetrize_to_wide_domain(
  #    measurements=mech_res.measurements,
  #    max_children_per_parent=fk_relation.max_children_per_parent,
  #    num_permutation_slots=num_permutation_slots,
  # )

  wide_domain = transformations.build_exploration_domain(
      parent_domain=parent_dataset.domain,
      child_domain=child_dataset.domain,
      max_group_size=fk_relation.max_children_per_parent,
      num_permutation_slots=fk_relation.max_children_per_parent,
      strategy=strategy,
  )

  if strategy == 'empty_token':
    wide_constraints = post_processing.create_slot_linear_chain_constraints(
        child_domain=child_dataset.domain,
        num_permutation_slots=fk_relation.max_children_per_parent,
    )
  else:
    wide_constraints = ()

  num_rows = (
      synth_parents.records
      if synth_parents is not None
      else max(1, int(round(noisy_root_total)))
  )

  pgm_iters = getattr(
      getattr(discrete_mechanism, 'config', None), 'pgm_iters', 5000
  )

  synth_wide_records = _fit_and_sample_wide_link_mrf(
      wide_domain=wide_domain,
      wide_measurements=mech_res.measurements,  # wide_measurements
      wide_constraints=wide_constraints,
      num_rows=num_rows,
      iters=pgm_iters,
  )

  synth_parent_dataset: mbi.Dataset | None = None
  if synth_parents is None:
    # Depth=0 Root Table ONLY: extract root parent dataset from wide records
    parent_cols = list(parent_dataset.domain.attributes)
    root_data = {col: synth_wide_records.data[col] for col in parent_cols}
    synth_parent_dataset = mbi.Dataset(root_data, parent_dataset.domain)
  else:
    synth_wide_records = post_processing.quantile_copula_coupling(
        synth_parents=synth_parents,
        synth_wide_children=synth_wide_records,
        parent_columns=[str(col) for col in parent_dataset.domain.attributes],
        rng=rng,
    )

  unstacked_children, parent_row_indices = (
      post_processing.unstack_wide_family_records(
          synth_wide_dataset=synth_wide_records,
          child_domain=child_dataset.domain,
          max_children_per_parent=fk_relation.max_children_per_parent,
      )
  )

  return SynthesizedLinkResult(
      unstacked_child_dataset=unstacked_children,
      parent_row_indices=parent_row_indices,
      synth_parent_dataset=synth_parent_dataset,
      discrete_mechanism_result=mech_res,
  )


def _synthesize_relational_hierarchy(
    mechanism: MultiTableMechanism,
    preprocessed: PreprocessedTables,
    hierarchy: Sequence[tuple[int, str, rel_domain.ForeignKeyRelation | None]],
    rng: np.random.Generator,
) -> tuple[
    dict[str, mbi.Dataset],
    dict[str, np.ndarray],
    dict[str, Any],
]:
  """Synthesizes all tables in topological order across the relational DAG.

  Args:
    mechanism: Calibrated MultiTableMechanism.
    preprocessed: PreprocessedTables container holding encoded data.
    hierarchy: Ordered topological synthesis levels.
    rng: NumPy random generator.

  Returns:
    A tuple of (synth_datasets, parent_mappings, discrete_mechanism_results):
      - synth_datasets: Mapping from table name to synthesized compressed
        mbi.Dataset.
      - parent_mappings: Mapping from child table name to 1D int array of parent
        row indices.
      - discrete_mechanism_results: Mapping from link name to mechanism
        diagnostics.
  """
  synth_datasets: dict[str, mbi.Dataset] = {}
  parent_mappings: dict[str, np.ndarray] = {}
  discrete_mechanism_results: dict[str, Any] = {}

  for _, table_name, fk in hierarchy:
    if fk is None:
      continue

    link_name = f'{fk.parent_table}->{fk.child_table}'
    discrete_mech = mechanism.calibrated_discrete_mechanisms[link_name]
    parent_dataset = preprocessed.compressed_datasets[fk.parent_table]
    child_dataset = preprocessed.compressed_datasets[table_name]
    parent_pks = list(
        preprocessed.table_keys[fk.parent_table][fk.parent_primary_key]
    )
    child_fks = list(preprocessed.table_keys[table_name][fk.child_foreign_key])

    synth_parents = synth_datasets.get(fk.parent_table)
    link_res = _synthesize_relational_link(
        parent_dataset=parent_dataset,
        child_dataset=child_dataset,
        parent_primary_keys=parent_pks,
        child_foreign_keys=child_fks,
        fk_relation=fk,
        discrete_mechanism=discrete_mech,
        num_permutation_slots=mechanism.num_permutation_slots,
        strategy=mechanism.exploration_strategy,
        rng=rng,
        synth_parents=synth_parents,
        noisy_root_total=preprocessed.noisy_root_total,
    )

    if fk.parent_table not in synth_datasets:
      assert link_res.synth_parent_dataset is not None
      synth_datasets[fk.parent_table] = link_res.synth_parent_dataset

    synth_datasets[table_name] = link_res.unstacked_child_dataset
    parent_mappings[table_name] = link_res.parent_row_indices
    if link_res.discrete_mechanism_result is not None:
      discrete_mechanism_results[link_name] = link_res.discrete_mechanism_result

  return synth_datasets, parent_mappings, discrete_mechanism_results


def _decompress_synthetic_datasets(
    synth_datasets: Mapping[str, mbi.Dataset],
    compression_mappings: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, mbi.Dataset]:
  """Probabilistically decompresses synthetic discrete datasets for each table.

  Reverses domain category compression mappings by sampling original category
  preimages uniformly from composite <Other> bins.

  Args:
    synth_datasets: Mapping from table name to synthesized mbi.Dataset.
    compression_mappings: Mapping from table name to per-column compression
      mapping arrays.

  Returns:
    A mapping from table name to decompressed mbi.Dataset.
  """
  decompressed: dict[str, mbi.Dataset] = {}
  for table_name, dataset in synth_datasets.items():
    mappings = compression_mappings.get(table_name, {})
    if mappings:
      raw_mappings = {str(col): arr for col, arr in mappings.items()}
      decompressed[table_name] = dataset.decompress(raw_mappings)
    else:
      decompressed[table_name] = dataset
  return decompressed


def _decode_synthetic_tables(
    decompressed_datasets: Mapping[str, mbi.Dataset],
    column_codecs: Mapping[str, data_generation_v3.TabularCodec],
    domains: Mapping[str, domain.Schema],
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
  """Decodes discrete datasets into continuous/categorical DataFrames.

  Converts discrete integer bucket tokens back to native strings and continuous
  floats (uniformly dequantized within bucket intervals) per table schema.

  Args:
    decompressed_datasets: Mapping from table name to decompressed mbi.Dataset.
    column_codecs: Mapping from table name to TabularCodec.
    domains: Mapping from table name to per-column AttributeType schemas.
    rng: NumPy random generator for uniform interval dequantization.

  Returns:
    A mapping from table name to decoded pandas DataFrame.
  """
  decoded_tables: dict[str, pd.DataFrame] = {}
  for table_name, dataset in decompressed_datasets.items():
    codec = column_codecs[table_name]
    column_order = list(domains[table_name].keys())
    decoded_tables[table_name] = codec.decode(
        synthetic=dataset, rng=rng, column_order=column_order
    )
  return decoded_tables


def _assign_relational_keys(
    tables: Mapping[str, pd.DataFrame],
    foreign_keys: Sequence[rel_domain.ForeignKeyRelation],
    parent_mappings: Mapping[str, np.ndarray],
    hierarchy: Sequence[tuple[int, str, rel_domain.ForeignKeyRelation | None]],
) -> dict[str, pd.DataFrame]:
  """Assigns synthetic primary and foreign keys to link relational tables.

  For each table in topological order, generates integer surrogate primary keys
  (0..N-1) and assigns child foreign keys by indexing parent primary keys via
  the parent-row mappings from copula matching and unstacking.

  Args:
    tables: Mapping from table name to decoded DataFrame.
    foreign_keys: Sequence of foreign key relationships.
    parent_mappings: Mapping from child table name to 1D int array of parent row
      indices.
    hierarchy: Ordered topological synthesis levels.

  Returns:
    A mapping from table name to DataFrame with assigned PK and FK columns.
  """
  linked_tables = {name: df.copy() for name, df in tables.items()}
  primary_keys: dict[str, dict[str, np.ndarray]] = {}

  for fk in foreign_keys:
    if fk.parent_table not in primary_keys:
      primary_keys[fk.parent_table] = {}
    if fk.parent_primary_key not in primary_keys[fk.parent_table]:
      pk_arr = np.arange(len(linked_tables[fk.parent_table]), dtype=np.int64)
      primary_keys[fk.parent_table][fk.parent_primary_key] = pk_arr
      linked_tables[fk.parent_table][fk.parent_primary_key] = pk_arr

  for _, table_name, fk in hierarchy:
    if fk is not None:
      p_keys = primary_keys[fk.parent_table][fk.parent_primary_key]
      parent_indices = parent_mappings[table_name]
      linked_tables[table_name][fk.child_foreign_key] = p_keys[parent_indices]

  return linked_tables


@dataclasses.dataclass(frozen=True)
class MultiDataGenerationResult:
  """Results of multi-table relational DP synthetic data generation.

  Attributes:
    synthetic_tables: Mapping from table names to synthetic DataFrames.
    discrete_mechanism_results: Mapping from link/table names to mechanism
      diagnostics.
  """

  synthetic_tables: Mapping[str, pd.DataFrame]
  discrete_mechanism_results: Mapping[str, Any] = dataclasses.field(
      default_factory=dict
  )


@dataclasses.dataclass
class MultiTableMechanism(api.CalibratedMechanism):
  """Calibrated, runnable multi-table relational differential privacy mechanism.

  Attributes:
    domains: Mapping from table name to per-column attribute specifications.
    foreign_keys: Sequence of foreign key relationships defining the hierarchy.
    calibrated_discrete_mechanisms: Mapping from link names to calibrated
      discrete mechanisms.
    calibrated_initializers: Mapping from table and column to calibrated
      initializers.
    total_count_sigma: Sigma for the root table total-count mechanism.
    num_permutation_slots: Permutation exploration slot count (o), default 2.
    exploration_strategy: Exploration strategy ('empty_token' or 'size_sliced').
    link_sensitivities: Mapping from link name to cascading integer sensitivity.

  Note: For simplicity, user-defined contraints are not supported yet.
  """

  domains: Mapping[str, domain.Schema]
  foreign_keys: Sequence[rel_domain.ForeignKeyRelation]
  calibrated_discrete_mechanisms: Mapping[str, api.CalibratedMechanism]
  calibrated_initializers: Mapping[str, Mapping[str, api.CalibratedMechanism]]
  total_count_sigma: float = dataclasses.field(repr=False)
  link_sensitivities: Mapping[str, int]
  num_permutation_slots: int = 2
  exploration_strategy: Literal['empty_token', 'size_sliced'] = 'empty_token'

  @property
  def schema(self) -> Mapping[str, domain.Schema]:
    """Returns the mapping from table name to dataset Schema."""
    return self.domains

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed DpEvent combining all relational sub-mechanisms.

    Formally composes:
      - 1 GaussianDpEvent for the root parent table total count measurement.
      - Column initializer DpEvents across all tables.
      - Discrete mechanism DpEvents across all relational hierarchy links.
        E.g. Exploration of Household-Person and Person-Activity tables with AIM
    """
    events: list[dp_accounting.DpEvent] = []
    for table_inits in self.calibrated_initializers.values():
      for init in table_inits.values():
        events.append(init.dp_event)
    events.append(
        dp_accounting.GaussianDpEvent(noise_multiplier=self.total_count_sigma)
    )
    # Lift each link mechanism from link-parent level to root-entity level.
    for link_name, mech in self.calibrated_discrete_mechanisms.items():
      link_sens = self.link_sensitivities[link_name]
      events.append(_calibration.with_group_size(mech.dp_event, link_sens))
    return dp_accounting.ComposedDpEvent(events)

  def __call__(
      self,
      rng: np.random.Generator,
      data: Mapping[str, pd.DataFrame],
  ) -> MultiDataGenerationResult:
    """Generates differentially private synthetic multi-table relational data.

    Executes the multi-table synthesis process across three steps:
      1. Preprocessing: Weights children tables to guarantee sensitivity = 1 for
         each parent record. Then individually for each table, discretizes
         numerical columns into bins, encodes & compresses categories under DP.
      2. Candidate Exploration: Explores set of candidate marginals from
         permutations of children records and their parent, representing
         cross-table correlations.
      3. Relational Synthesis: Generates synthetic records top-down from root
         parent to leaf child tables, fitting the MRF per parent->child pair
         preserving the chosen cross-table marginal correlation marginals.
      4. Output Decoding & Key Assignment: Converts synthetic tokens back to
         original data types (continuous numbers & category strings) and assigns
         matching primary and foreign keys to ensure referential integrity.

    Args:
      rng: NumPy random number generator.
      data: Mapping from table name to input pandas DataFrame.

    Returns:
      A MultiDataGenerationResult containing synthetic DataFrames and
      diagnostics.
    """
    preprocessed = _run_table_preprocessing(self, rng=rng, data=data)

    hierarchy = rel_domain.topological_sort_hierarchy(
        list(self.domains.keys()), self.foreign_keys
    )
    synth_datasets, parent_mappings, discrete_results = (
        _synthesize_relational_hierarchy(
            mechanism=self,
            preprocessed=preprocessed,
            hierarchy=hierarchy,
            rng=rng,
        )
    )

    decompressed = _decompress_synthetic_datasets(
        synth_datasets=synth_datasets,
        compression_mappings=preprocessed.compression_mappings,
    )
    decoded_tables = _decode_synthetic_tables(
        decompressed_datasets=decompressed,
        column_codecs=preprocessed.column_codecs,
        domains=self.domains,
        rng=rng,
    )
    final_tables = _assign_relational_keys(
        tables=decoded_tables,
        foreign_keys=self.foreign_keys,
        parent_mappings=parent_mappings,
        hierarchy=hierarchy,
    )

    return MultiDataGenerationResult(
        synthetic_tables=final_tables,
        discrete_mechanism_results=discrete_results,
    )


@dataclasses.dataclass
class MultiTableConfig(api.MechanismConfig):
  """Recipe for multi-table relational differential privacy synthesis.

  Attributes:
    foreign_keys: Sequence of foreign key relationships defining the hierarchy.
    discrete_mechanism: Discrete mechanism config (e.g. AIM, MST) for relational
      links.
    numerical_bins: Number of bins for numerical attribute discretization.
    init_budget_fraction: Fraction of total privacy budget allocated to column
      initialization.
    initializers: Optional custom column initializers per table.
    num_permutation_slots: Permutation exploration slot count (o), default 2.
    exploration_strategy: Exploration strategy ('empty_token' or 'size_sliced').
  """

  foreign_keys: Sequence[rel_domain.ForeignKeyRelation] = ()
  discrete_mechanism: api.MechanismConfig = dataclasses.field(
      default_factory=discrete_mechanisms.AIMConfig
  )
  numerical_bins: int = 32
  init_budget_fraction: float = 0.1
  initializers: Mapping[str, Mapping[str, api.MechanismConfig]] | None = None
  num_permutation_slots: int = 2
  exploration_strategy: Literal['empty_token', 'size_sliced'] = 'empty_token'

  def __post_init__(self):
    if not (0.0 < self.init_budget_fraction < 1.0):
      raise ValueError(
          'init_budget_fraction must be strictly in (0.0, 1.0), got'
          f' {self.init_budget_fraction}.'
      )
    if self.numerical_bins < 1:
      raise ValueError(
          f'numerical_bins must be >= 1, got {self.numerical_bins}.'
      )
    if self.num_permutation_slots < 1:
      raise ValueError(
          'num_permutation_slots must be >= 1, got'
          f' {self.num_permutation_slots}.'
      )
    if self.exploration_strategy not in ('empty_token', 'size_sliced'):
      raise ValueError(
          f'Unsupported exploration_strategy {self.exploration_strategy!r}.'
      )
    if not isinstance(self.discrete_mechanism, api.MechanismConfig):
      raise ValueError(
          'discrete_mechanism must be an instance of MechanismConfig, got'
          f' {type(self.discrete_mechanism).__name__}.'
      )

  def configure(
      self,
      schema: (
          Mapping[str, domain.Schema | Mapping[str, domain.AttributeType]]
          | None
      ) = None,
      *,
      budget: float,
      delta: float = 0.0,
  ) -> MultiTableMechanism:
    """Configures privacy budgets across column initializers and links.

    Formal Guarantees:
      - Additive zCDP Partitioning: The total zCDP budget is
        additively split into init_rho (allocated to 1 root total-count
        measurement and N_total_columns per-column initializers) and
        total_discrete_rho (split evenly across relational hierarchy links).
      - Root-Anchored Population Count: Only the root parent table total count
        is measured with Gaussian noise (total_count_sigma).
      - Descendant table row counts are generated via post-processing from the
        wide discrete mechanism (unstacking non-empty slots under 'empty_token'
        strategy or group size K under 'size_sliced' strategy).
      - Pure zCDP for Gaussian Primitives: Numerical, closed categorical, and
        root count initializers operate under pure zCDP (delta = 0.0).
      - Thresholding Delta Partitioning: Open-set partition selection delta is
        split additively across all open-set columns in all tables.
      - Cascading Sensitivity Scaling: Downstream discrete mechanisms are
        configured with cascading sensitivities Delta_k = prod s_ancestors,
        guaranteeing root parent differential privacy without Cartesian joins.

    Args:
      schema: Mapping from table name to table domain schema (either a
        `domain.Schema` or a mapping of column name to `AttributeType`).
      budget: The total privacy budget (budget > 0).
      delta: Approximate DP delta for open-set Gaussian partition selection.

    Returns:
      A calibrated, runnable MultiTableMechanism.

    Raises:
      ValueError: If configuration hyperparameters, schemas, or budgets are
        invalid.
    """
    if schema is None:
      raise ValueError(
          'schema must be provided to MultiTableConfig.configure().'
      )
    if len(schema) < 2:
      raise ValueError(
          'MultiTableConfig requires at least two tables in schema, got'
          f' {len(schema)}. For single-table synthesis, use TabularConfig.'
      )
    if not self.foreign_keys:
      raise ValueError(
          'MultiTableConfig requires at least one foreign key relationship in'
          ' foreign_keys. For single-table synthesis, use TabularConfig.'
      )
    if budget <= 0:
      raise ValueError(f'budget must be positive, got {budget}.')

    # Normalize table schemas into domain.Schema instances.
    domains: dict[str, domain.Schema] = {
        table: (
            table_schema
            if isinstance(table_schema, domain.Schema)
            else domain.Schema(table_schema)
        )
        for table, table_schema in schema.items()
    }

    # 1. Validate table names, column names, and attribute types.
    for table_name, table_schema in domains.items():
      if '.' in table_name:
        raise ValueError(
            f"Table name {table_name!r} must not contain '.' characters."
        )
      if not table_schema:
        raise ValueError(f'Table {table_name!r} schema cannot be empty.')
      for col_name, attr in table_schema.items():
        if '.' in col_name:
          raise ValueError(
              f"Table {table_name!r} column {col_name!r} must not contain '.'"
              ' (reserved for wide relational slot prefixes).'
          )
        if col_name == 'group_size':
          raise ValueError(
              f"Table {table_name!r} column name 'group_size' is reserved for"
              ' relational exploration.'
          )
        if col_name.startswith('slot_'):
          raise ValueError(
              f'Table {table_name!r} column {col_name!r} cannot start with'
              " 'slot_' (reserved for permutation slots)."
          )
        if not isinstance(
            attr,
            (
                domain.NumericalAttribute,
                domain.CategoricalAttribute,
                domain.OpenSetCategoricalAttribute,
            ),
        ):
          raise ValueError(
              f'Table {table_name!r} column {col_name!r} has unsupported'
              f' attribute type {type(attr).__name__}.'
          )

    # 2. Validate DAG hierarchy (acyclicity, known tables, in-degree <= 1).
    hierarchy = rel_domain.topological_sort_hierarchy(
        list(domains.keys()), self.foreign_keys
    )
    roots = [t for _, t, fk in hierarchy if fk is None]
    if len(roots) > 1:
      raise ValueError(
          'MultiTableConfig expects a single root table, but found'
          f' {len(roots)}: {roots}. All tables must be connected in a single'
          ' tree hierarchy.'
      )

    # 3. Ensure PK and FK columns are not present in domain schemas.
    for fk in self.foreign_keys:
      if fk.parent_primary_key in domains[fk.parent_table]:
        raise ValueError(
            f'Primary key column {fk.parent_primary_key!r} of table'
            f' {fk.parent_table!r} must not be in schema[{fk.parent_table!r}].'
        )
      if fk.child_foreign_key in domains[fk.child_table]:
        raise ValueError(
            f'Foreign key column {fk.child_foreign_key!r} of table'
            f' {fk.child_table!r} must not be in schema[{fk.child_table!r}].'
        )

    # 4. Validate custom initializers structure if provided.
    if self.initializers is not None:
      if set(self.initializers.keys()) != set(domains.keys()):
        raise ValueError(
            f'Custom initializers tables {set(self.initializers.keys())} do not'
            f' match schema tables {set(domains.keys())}.'
        )
      for table_name, table_inits in self.initializers.items():
        if set(table_inits.keys()) != set(domains[table_name].keys()):
          raise ValueError(
              f'Custom initializers for table {table_name!r}'
              f' columns {set(table_inits.keys())} do not match'
              f' schema columns {set(domains[table_name].keys())}.'
          )
        for col_name, init_cfg in table_inits.items():
          if not isinstance(init_cfg, api.MechanismConfig):
            raise ValueError(
                f'Custom initializer for {table_name}.{col_name} must be an'
                f' api.MechanismConfig, got {type(init_cfg).__name__}.'
            )

    link_sensitivities = _compute_link_sensitivities(hierarchy)

    per_col_deltas = _compute_table_col_deltas(domains, delta=delta)
    inits = (
        self.initializers
        if self.initializers is not None
        else _create_table_initializers(domains, self.numerical_bins)
    )

    total_cols = sum(len(table_schema) for table_schema in domains.values())
    init_rho = self.init_budget_fraction * budget
    per_col_rho = init_rho / (total_cols + 1)  # +1 for root table total count.
    total_count_rho = per_col_rho
    total_discrete_rho = budget - init_rho
    per_link_rho = total_discrete_rho / len(link_sensitivities)

    calibrated_inits = {
        table: {
            col: init.configure(
                domains[table][col],
                budget=per_col_rho,
                delta=per_col_deltas[table][col],
            )
            for col, init in table_inits.items()
        }
        for table, table_inits in inits.items()
    }
    total_count_sigma = (
        math.sqrt(0.5 / total_count_rho) if total_count_rho > 0 else 0.0
    )

    calibrated_discrete = {
        link_name: self.discrete_mechanism.configure(
            budget=per_link_rho / (sensitivity**2),
        )
        for link_name, sensitivity in link_sensitivities.items()
    }

    return MultiTableMechanism(
        domains=domains,
        foreign_keys=self.foreign_keys,
        calibrated_discrete_mechanisms=calibrated_discrete,
        calibrated_initializers=calibrated_inits,
        total_count_sigma=total_count_sigma,
        num_permutation_slots=self.num_permutation_slots,
        exploration_strategy=self.exploration_strategy,
        link_sensitivities=link_sensitivities,
    )
