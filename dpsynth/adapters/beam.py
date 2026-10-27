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

"""Experimental Beam adapter for local-mode DP Synth.

.. warning:: This module is experimental.

This module provides a lightweight bridge between the local-mode
TabularConfig and Apache Beam, enabling local-mode features to
run on datasets too large to fit in memory. See the adapters README
for more information.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Sequence
import copy
import io
import math
import pickle
import shutil
import tempfile
from typing import Any, cast
import uuid

from absl import logging
import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from dpsynth import data_generation_v3
from dpsynth.local_mode import initialization
import mbi
import numpy as np

# A single row of tabular data: positional sequence in domain column order.
Row = Sequence[Any]

Initializer = (
    initialization.NumericalInitializerConfig
    | initialization.CategoricalInitializerConfig
    | initialization.OpenSetInitializerConfig
)

CalibratedInitializer = (
    initialization.NumericalInitializer
    | initialization.CategoricalInitializer
    | initialization.OpenSetInitializer
)


class _EncodeColumns(beam.DoFn):
  """Encodes a batch of rows into ((column, key), count) pairs."""

  def __init__(self, initializers: dict[str, CalibratedInitializer]):
    # Do all setup in __init__ so that process below is cheaper.
    # We handle all columns at once here to reduce the size of the DAG in Beam.
    super().__init__()
    self._specs: list[tuple[str, str, dict[str, Any]]] = []
    for column, init in initializers.items():
      if isinstance(init, initialization.NumericalInitializer):
        attr = init.attribute
        lower, upper, gs = init.grid_spec
        delta = (upper - lower) / (gs - 1)
        meta = dict(attribute=attr, lower=lower, upper=upper, delta=delta)
        self._specs.append((column, 'numerical', meta))

      elif isinstance(init, initialization.CategoricalInitializer):
        meta = {
            'lookup': init.attribute.lookup,
            'default': init.attribute.out_of_domain_index,
        }
        self._specs.append((column, 'categorical', meta))
      elif isinstance(init, initialization.OpenSetInitializer):
        self._specs.append((column, 'openset', {}))
      else:
        raise TypeError(f'Unsupported initializer type: {type(init)}')

  def process(self, batch: Sequence[Row]):
    cols = zip(*batch, strict=True)
    for (column, kind, params), values in zip(self._specs, cols, strict=True):
      if kind == 'numerical':
        # Expects pre-parsed numeric values (int | float | None), matching Pass
        # 2 and in-memory NumericalEncoding.encode_to_grid.
        keys = initialization.encode_to_grid(values, **params).tolist()
      elif kind == 'categorical':
        lookup, default = params['lookup'], params['default']
        keys = (lookup.get(str(v), default) for v in values)
      elif kind == 'openset':
        keys = (str(v) for v in values)
      else:
        raise ValueError(f'Unknown column kind: {kind}')
      for key, count in collections.Counter(keys).items():
        yield (column, key), count


def _unpack_count(element):
  """Restructures ((column, key), count) to (column, (key, count))."""
  (col, key), count = element
  return (col, (key, count))


def _filter_openset(element, min_counts):
  """Filters open-set values below min_count, passes others through."""
  col, pairs = element
  min_count = min_counts.get(col)
  if min_count is not None:
    pairs = [(k, c) for k, c in pairs if c >= min_count]
  return (col, pairs)


def _materialize_pairs(col, pairs):
  """Converts GroupByKey's lazy iterator to a concrete list."""
  return (col, list(pairs))


class ComputeSufficientStats(beam.PTransform):
  """Computes per-column sufficient statistics in a single pass.

  Each input row must be a positional sequence ordered to match ``initializers``
  keys. Encodes all columns in batches, then sums counts per ``(column, key)``
  and groups by column. The output is a ``PCollection`` of
  ``(column_name, sparse_counts_list)`` pairs.

  Attributes:
    initializers: Calibrated initializers keyed by column name.
  """

  def __init__(self, initializers: dict[str, CalibratedInitializer]):
    super().__init__()
    self._initializers = initializers
    self._openset_min_counts = {
        col: init.config.min_count
        for col, init in initializers.items()
        if isinstance(init, initialization.OpenSetInitializer)
    }

  def expand(
      self, rows: beam.PCollection[Row]
  ) -> beam.PCollection[tuple[str, list[tuple[Any, int]]]]:
    return (
        rows
        # Group rows into batches so _EncodeColumns pre-aggregates in memory.
        | 'Batch'
        >> beam.BatchElements(min_batch_size=10000, max_batch_size=50000)
        | 'Encode' >> beam.ParDo(_EncodeColumns(self._initializers))
        # Sum counts across batches for each (column, key) pair.
        | 'Count' >> beam.CombinePerKey(sum)
        | 'Unpack' >> beam.Map(_unpack_count)
        # Group (key, count) pairs by column name into a single list per column.
        | 'GroupByColumn' >> beam.GroupByKey()
        | 'ToLists' >> beam.MapTuple(_materialize_pairs)
        | 'FilterOpenSet'
        >> beam.Map(_filter_openset, min_counts=self._openset_min_counts)
    )


def _sparse_to_dense_numerical(sparse, grid_size):
  """Converts sparse (index, count) pairs to a dense histogram array."""
  counts = np.zeros(grid_size, dtype=np.float64)
  for idx, count in sparse:
    counts[idx] = count
  return counts


def _sparse_to_dense_categorical(sparse, size):
  """Converts sparse (index, count) pairs to a dense count vector."""
  counts = np.zeros(size, dtype=np.float64)
  for idx, count in sparse:
    counts[idx] = count
  return counts


def _sparse_to_openset(sparse):
  """Converts sparse (value, count) pairs to parallel arrays."""
  if not sparse:
    return np.array([], dtype=object), np.array([], dtype=np.float64)
  keys, vals = zip(*sorted(sparse))
  return np.array(keys), np.array(vals, dtype=np.float64)


# mbi) into the Beam pipeline, which can increase setup time for each worker.
def run_from_summary(
    sparse_stats: dict[str, list[tuple[Any, int]]],
    initializers: dict[str, CalibratedInitializer],
    rng: np.random.Generator,
    *,
    num_rows: int | None = None,
) -> dict[str, initialization.ColumnMeasurement]:
  """Converts materialized sparse stats to ColumnMeasurements on the driver.

  Meant to be called after ``ComputeSufficientStats`` results have been
  materialized (e.g. via ``beam.combiners.ToDict()``).

  Args:
    sparse_stats: Column-keyed dict of sparse (key, count) pair lists, as
      produced by ``ComputeSufficientStats``.
    initializers: Calibrated initializers keyed by column name.
    rng: NumPy random generator for DP noise.
    num_rows: Optional total row count, used to compute out-of-domain counts for
      numerical columns with ``clip_to_range=False``.

  Returns:
    Per-column ``ColumnMeasurement`` results.
  """
  results: dict[str, initialization.ColumnMeasurement] = {}
  for column, init in initializers.items():
    sparse = sparse_stats.get(column, [])
    if isinstance(init, initialization.NumericalInitializer):
      counts = _sparse_to_dense_numerical(sparse, init.grid_spec[2])
      ood_count = (
          float(num_rows - counts.sum())
          if (not init.attribute.clip_to_range and num_rows is not None)
          else 0.0
      )
      results[column] = init.from_summary(rng, counts, ood_count=ood_count)
    elif isinstance(init, initialization.CategoricalInitializer):
      counts = _sparse_to_dense_categorical(sparse, init.attribute.size)
      results[column] = init.from_summary(rng, counts)
    elif isinstance(init, initialization.OpenSetInitializer):
      unique_values, value_counts = _sparse_to_openset(sparse)
      results[column] = init.from_summary(rng, unique_values, value_counts)
  return results


class _EncodeAndProject(beam.DoFn):
  """Integer-encodes row batches and emits (clique_index, histogram) pairs."""

  def __init__(
      self,
      column_measurements: dict[str, initialization.ColumnMeasurement],
      domains: dict[str, Any],
      workload: list[mbi.Clique],
  ):
    super().__init__()
    # Reuse the shared per-column codec so Beam encoding matches the in-memory
    # path exactly, for both numerical binning and categorical lookups.
    self._codecs = {
        col: data_generation_v3.ColumnCodec(cm, domains[col])
        for col, cm in column_measurements.items()
    }
    self._clique_meta: list[tuple[int, mbi.Clique, tuple[int, ...]]] = []
    for idx, clique in enumerate(workload):
      shape = tuple(
          int(column_measurements[c].categorical_attribute.size) for c in clique  # pyrefly: ignore[bad-index]
      )
      self._clique_meta.append((idx, clique, shape))

  def process(self, batch: Sequence[Row]):
    cols = zip(*batch, strict=True)
    encoded = {
        col: codec.encode(np.asarray(vals, dtype=object))
        for (col, codec), vals in zip(self._codecs.items(), cols, strict=True)
    }
    # supporting_cliques() never returns the 0-way clique (), so shape is always
    # non-empty here and np.ravel_multi_index is safe. Supporting cliques are
    # small enough to fit as dense Factors on the driver, so dense O(N) bincount
    # + CombinePerKey(sum) beats sparse sorting.
    for clique_idx, clique_cols, shape in self._clique_meta:
      multi_index = tuple(encoded[c] for c in clique_cols)  # pyrefly: ignore[bad-index]
      linear = np.ravel_multi_index(multi_index, shape)
      counts = np.bincount(linear, minlength=math.prod(shape)).reshape(shape)
      yield clique_idx, counts


def _assemble_dense_marginal(element, workload, mbi_domain):
  """Converts a summed histogram array to an mbi.Factor for one clique."""
  clique_idx, counts = element
  clique_domain = mbi_domain.project(workload[clique_idx])
  return mbi.Factor(clique_domain, counts.astype(np.float64))  # pyrefly: ignore[bad-argument-type]


# Stage 2 of the two-pass pipeline: compute the joint marginals the DP mechanism
# needs. Using the domains from stage 1, Beam integer-encodes each row and, for
# every requested clique (a small set of columns), counts how many rows fall in
# each cell of that clique's joint histogram, summing across the whole dataset
# to build a single mbi.CliqueVector. These counts are exact/non-private: DP
# noise is added later on the driver by the discrete mechanism.
class ComputeMarginals(beam.PTransform):
  """Computes a workload of marginals over integer-encoded rows.

  Each input row must be a positional sequence ordered to match
  ``column_measurements`` keys. Takes raw rows plus the ``ColumnMeasurement``
  results from stage 1, integer-encodes each row in batches, and computes the
  contingency table for each clique in the workload. The output is a singleton
  ``PCollection`` containing one ``mbi.CliqueVector``.

  Attributes:
      column_measurements: Per-column results from stage 1 initialization.
      domains: Original attribute domain specs (needed for numerical encoding).
      workload: List of cliques (tuples of column names) to measure.
  """

  def __init__(
      self,
      column_measurements: dict[str, initialization.ColumnMeasurement],
      domains: dict[str, Any],
      workload: list[mbi.Clique],
  ):
    super().__init__()
    self._column_measurements = column_measurements
    self._domains = domains
    self._workload = workload
    self._mbi_domain = data_generation_v3.TabularCodec.from_measurements(
        column_measurements, domains
    ).mbi_domain

  def expand(self, rows: beam.PCollection[Row]):
    mbi_domain = self._mbi_domain

    def _to_clique_vector(factors):
      tables = {f.domain.attributes: f for f in factors}
      return mbi.CliqueVector(mbi_domain, tuple(self._workload), tables)

    return (
        rows
        | 'Batch'
        >> beam.BatchElements(min_batch_size=10000, max_batch_size=50000)
        | 'EncodeProject'
        >> beam.ParDo(
            _EncodeAndProject(
                self._column_measurements, self._domains, self._workload
            )
        )
        # Elementwise-sum histogram arrays across batches for each clique.
        | 'SumCounts' >> beam.CombinePerKey(sum)
        | 'ToFactor'
        >> beam.Map(
            _assemble_dense_marginal,
            workload=self._workload,
            mbi_domain=mbi_domain,
        )
        # Collapse all clique Factors into a single PCollection element.
        | 'ToList' >> beam.combiners.ToList()
        | 'BuildCliqueVector' >> beam.Map(_to_clique_vector)
    )


# End-to-end synthesis: the two Beam passes above learn each column's domain
# (stage 1) and the joint marginals the mechanism needs (stage 2). The driver
# then runs the discrete mechanism and decodes the synthetic output locally,
# since the graphical model and sampling are small enough to fit in memory.


def _write(value: Any, path: str) -> None:
  """Serializes a driver-bound pipeline result to ``path``."""
  # mbi.save expects a pytree of numeric arrays (like CliqueVector), whereas
  # Pass 1 sufficient stats and row count are plain Python containers/scalars.
  if isinstance(value, mbi.CliqueVector):
    buf = io.BytesIO()
    mbi.save(value, buf)
    data = buf.getvalue()
  else:
    data = pickle.dumps(value)
  # Stage to a unique sibling in temp_location and atomically rename so
  # speculative task retries never truncate a completed file.
  tmp_path = f'{path}.tmp.{uuid.uuid4().hex}'
  with FileSystems.create(tmp_path) as f:
    f.write(data)
  FileSystems.rename([tmp_path], [path])


def _read(path: str) -> Any:
  """Reads a value written by ``_write`` on the driver."""
  with FileSystems.open(path) as f:
    raw = f.read()
  if raw[:4] == b'PK\x03\x04':
    return mbi.load(io.BytesIO(raw))
  # Trusted input only: reads data this pipeline wrote to temp_location, which
  # must therefore not point at an untrusted or world-writable path.
  return pickle.loads(raw)  # pylint: disable=g-unsafe-pickle-load


def _copy_pipeline_options(options, pass_suffix: str):
  if options is None:
    return None
  copied = copy.deepcopy(options)
  gcloud_opts = copied.view_as(beam.options.pipeline_options.GoogleCloudOptions)
  if gcloud_opts.job_name:
    gcloud_opts.job_name = f'{gcloud_opts.job_name}-{pass_suffix}'
  return copied


def execute(
    mechanism: data_generation_v3.TabularMechanism,
    rng: np.random.Generator,
    create_rows_fn: Callable[[beam.Pipeline], beam.PCollection[Row]],
    *,
    temp_location: str | None = None,
    pipeline_options: (
        beam.options.pipeline_options.PipelineOptions | None
    ) = None,
    num_rows: int | None = None,
) -> data_generation_v3.DataGenerationResult:
  """Executes a calibrated TabularMechanism over a two-pass Beam pipeline.

  Usage::

      mech = dpsynth.TabularConfig().calibrate(domains, epsilon=1.0, delta=1e-6)
      result = beam.execute(mech, rng, create_rows_fn)

  Args:
    mechanism: A calibrated local-mode TabularMechanism.
    rng: NumPy random generator for DP noise and synthetic sampling.
    create_rows_fn: Callable ``(beam.Pipeline) -> PCollection[Row]`` where each
      row is a positional sequence ordered by ``mechanism.schema``.
    temp_location: Directory used to shuttle small singleton results between the
      pipeline and the driver. Must be readable and writable by all workers on
      distributed runners. Defaults to a local temporary directory.
    pipeline_options: Optional Beam pipeline options applied to both passes.
    num_rows: Optional number of synthetic rows to generate. Defaults to the
      fitted model's noisy total count.

  Returns:
    A DataGenerationResult containing the synthetic DataFrame.
  """
  if not hasattr(mechanism.config.discrete_mechanism, 'supporting_cliques'):
    raise ValueError(
        'mechanism.config.discrete_mechanism must have a supporting_cliques'
        ' method.'
    )
  if mechanism.config.compress_columns:
    raise ValueError('compress_columns is not supported by the Beam adapter.')

  inits = cast(dict[str, CalibratedInitializer], mechanism.initializers)

  created_temp_dir = temp_location is None
  temp_dir = temp_location or tempfile.mkdtemp(prefix='dpsynth_beam_')
  summary_path = FileSystems.join(temp_dir, 'sufficient_stats.bin')
  count_path = FileSystems.join(temp_dir, 'row_count.bin')
  marginals_path = FileSystems.join(temp_dir, 'clique_vector.bin')
  try:
    # Pass 1: privately learn distribution of each column independently.
    # Exiting the `with` block executes the Beam DAG; singleton PCollections are
    # written to temp_dir so the driver process can read them back.
    with beam.Pipeline(
        options=_copy_pipeline_options(pipeline_options, 'pass1')
    ) as p:
      rows = create_rows_fn(p)
      summary = (
          rows
          | ComputeSufficientStats(inits)
          | 'ToDict' >> beam.combiners.ToDict()
      )
      _ = summary | 'WriteSummary' >> beam.Map(_write, path=summary_path)
      count = rows | 'CountRows' >> beam.combiners.Count.Globally()
      _ = count | 'WriteRowCount' >> beam.Map(_write, path=count_path)
    # We run this on the driver so we don't have to track worker-side RNGs.
    sparse_stats = _read(summary_path)
    num_rows_in = int(_read(count_path))
    column_measurements = run_from_summary(
        sparse_stats, inits, rng, num_rows=num_rows_in
    )
    logging.info('[DPSynth/Beam]: Pass 1 complete.')

    # Ask the configured discrete mechanism which marginals it needs.
    mbi_domain = data_generation_v3.TabularCodec.from_measurements(
        column_measurements, mechanism.schema
    ).mbi_domain

    # pyrefly: ignore[missing-attribute]
    workload = mechanism.config.discrete_mechanism.supporting_cliques(
        mbi_domain
    )

    # Pass 2: compute the marginal workload.
    with beam.Pipeline(
        options=_copy_pipeline_options(pipeline_options, 'pass2')
    ) as p:
      rows = create_rows_fn(p)
      marginals = rows | ComputeMarginals(
          column_measurements,
          dict(mechanism.schema),
          workload,
      )
      _ = marginals | 'WriteCliqueVector' >> beam.Map(
          _write, path=marginals_path
      )
    clique_vector = _read(marginals_path)
    logging.info('[DPSynth/Beam]: Pass 2 complete.')

    # Run the discrete mechanism and decode on the driver.
    return mechanism.from_summary(
        rng,
        column_measurements,
        clique_vector,
        num_rows=num_rows,
    )
  finally:
    # Only remove a temp dir we created; never a user-supplied temp_location.
    if created_temp_dir:
      shutil.rmtree(temp_dir, ignore_errors=True)
