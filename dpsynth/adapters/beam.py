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

from collections.abc import Callable
import dataclasses
import io
import math
import pickle
import shutil
import tempfile
from typing import Any, cast

from absl import logging
import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
import dp_accounting
from dpsynth import api
from dpsynth import data_generation_v3
from dpsynth import domain
from dpsynth.discrete_mechanisms import common as dm_common
from dpsynth.local_mode import initialization
import mbi
import numpy as np

# A single row of tabular data: column name -> raw value.
# representation for large pipelines.  Consider supporting named tuples or
# a schema-aware format (e.g. Beam Rows, protos) to reduce per-element overhead.
Row = dict[str, Any]

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
  """Encodes each row into (column, key) pairs for all columns at once."""

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

  def process(self, row: Row):
    for column, kind, params in self._specs:
      value = row.get(column)
      if kind == 'numerical':
        attribute: domain.NumericalAttribute = params['attribute']
        value = attribute.standardize(value)
        if math.isnan(value):
          continue  # clip_to_range=False: standardize returns NaN --> drop.
        index = int(initialization.encode_to_grid(value, **params))
        yield (column, index)
      elif kind == 'categorical':
        index = params['lookup'].get(str(value), params['default'])
        yield (column, index)
      elif kind == 'openset':
        yield (column, str(value))


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

  Encodes all columns in one ``DoFn``, then counts via a single
  ``Count.PerElement`` and groups by column. The output is a ``PCollection``
  of ``(column_name, sparse_counts_list)`` pairs.

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
        | 'Encode' >> beam.ParDo(_EncodeColumns(self._initializers))
        | 'Count' >> beam.combiners.Count.PerElement()
        | 'Unpack' >> beam.Map(_unpack_count)
        # Aggregate data and materialize on the driver (see module header).
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
  keys, vals = zip(*sparse)
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
    sparse = sparse_stats[column]
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
  """Integer-encodes each row and emits (clique_index, linear_index) pairs."""

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

  def process(self, row: Row):
    # ColumnCodec.encode is vectorized, so wrap each scalar in a length-1 array.
    encoded = {
        col: int(codec.encode(np.asarray([row.get(col)]))[0])
        for col, codec in self._codecs.items()
    }
    # supporting_cliques() never returns the 0-way clique (), so shape is always
    # non-empty here and np.ravel_multi_index is safe.
    for clique_idx, clique_cols, shape in self._clique_meta:
      multi_index = tuple(encoded[c] for c in clique_cols)  # pyrefly: ignore[bad-index]
      linear = int(np.ravel_multi_index(multi_index, shape))
      yield clique_idx, linear


def _unpack_marginal_count(element):
  """Restructures ((clique_idx, linear_idx), count) for GroupByKey."""
  (clique_idx, linear_idx), count = element
  return clique_idx, (linear_idx, count)


def _assemble_dense_marginal(element, clique_meta, mbi_domain):
  """Converts sparse counts to an mbi.Factor for one clique."""
  clique_idx, sparse_pairs = element
  _, clique_cols, shape = clique_meta[clique_idx]
  total_size = math.prod(shape)
  dense = np.zeros(total_size, dtype=np.float64)
  for linear_idx, count in sparse_pairs:
    dense[linear_idx] = count
  return mbi.Factor(mbi_domain.project(clique_cols), dense.reshape(shape))  # pyrefly: ignore[bad-argument-type]


# Stage 2 of the two-pass pipeline: compute the joint marginals the DP mechanism
# needs. Using the domains from stage 1, Beam integer-encodes each row and, for
# every requested clique (a small set of columns), counts how many rows fall in
# each cell of that clique's joint histogram, summing across the whole dataset
# to build a single mbi.CliqueVector. These counts are exact/non-private: DP
# noise is added later on the driver by the discrete mechanism.
class ComputeMarginals(beam.PTransform):
  """Computes a workload of marginals over integer-encoded rows.

  Takes raw rows plus the ``ColumnMeasurement`` results from stage 1,
  integer-encodes each row, and computes the contingency table for each
  clique in the workload. The output is a singleton ``PCollection``
  containing one ``mbi.CliqueVector``.

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
    self._clique_meta = []
    for idx, clique in enumerate(workload):
      shape = self._mbi_domain.project(clique).shape
      self._clique_meta.append((idx, clique, shape))

  def expand(self, rows: beam.PCollection[Row]):
    mbi_domain = self._mbi_domain

    def _to_clique_vector(factors):
      cliques = tuple(f.domain.attributes for f in factors)
      tables = {cl: f for cl, f in zip(cliques, factors)}
      return mbi.CliqueVector(mbi_domain, cliques, tables)

    return (
        rows
        | 'EncodeProject'
        >> beam.ParDo(
            _EncodeAndProject(
                self._column_measurements, self._domains, self._workload
            )
        )
        | 'CountPerElement' >> beam.combiners.Count.PerElement()
        | 'Unpack' >> beam.Map(_unpack_marginal_count)
        | 'GroupByClique' >> beam.GroupByKey()
        | 'ToLists' >> beam.MapTuple(_materialize_pairs)
        | 'ToFactor'
        >> beam.Map(
            _assemble_dense_marginal,
            clique_meta=self._clique_meta,
            mbi_domain=mbi_domain,
        )
        | 'ToList' >> beam.combiners.ToList()
        | 'BuildCliqueVector' >> beam.Map(_to_clique_vector)
    )


# End-to-end synthesis: the two Beam passes above learn each column's domain
# (stage 1) and the joint marginals the mechanism needs (stage 2). The driver
# then runs the discrete mechanism and decodes the synthetic output locally,
# since the graphical model and sampling are small enough to fit in memory.


def _write(value: Any, path: str) -> None:
  """Serializes a driver-bound pipeline result to ``path``."""
  # Writing to a (possibly distributed) filesystem lets the driver read the
  # value back after the pipeline finishes, so it works on remote runners.
  buf = io.BytesIO()
  try:
    mbi.save(value, buf)
    data = buf.getvalue()
  except TypeError:
    data = pickle.dumps(value)
  with FileSystems.create(path) as f:
    f.write(data)


def _read(path: str) -> Any:
  """Reads a value written by ``_write`` on the driver."""
  with FileSystems.open(path) as f:
    raw = f.read()
  if raw[:4] == b'PK\x03\x04':
    return mbi.load(io.BytesIO(raw))
  # Trusted input only: reads data this pipeline wrote to temp_location, which
  # must therefore not point at an untrusted or world-writable path.
  return pickle.loads(raw)  # pylint: disable=g-unsafe-pickle-load


def generate_from_marginals(
    synth: data_generation_v3.TabularMechanism,
    rng: np.random.Generator,
    column_measurements: dict[str, initialization.ColumnMeasurement],
    marginals: mbi.CliqueVector,
) -> data_generation_v3.DataGenerationResult:
  """Runs the discrete mechanism and decoding from pre-computed marginals.

  Args:
    synth: A calibrated TabularMechanism.
    rng: NumPy random generator for the discrete mechanism's DP noise.
    column_measurements: Per-column results from pass 1 initialization.
    marginals: The exact joint marginals computed by pass 2.

  Returns:
    A DataGenerationResult containing the synthetic DataFrame.
  """
  # Emit columns in domain-declaration order for deterministic output.
  column_order = [c for c in synth.schema if c in column_measurements]
  codec = data_generation_v3.TabularCodec.from_measurements(
      column_measurements, synth.schema
  )

  initial_measurements = codec.one_way_measurements()
  logging.info('[DPSynth/Beam]: Running discrete mechanism.')
  # pyrefly: ignore[missing-attribute,not-callable]
  mechanism_result = synth.base_mechanism(
      rng,
      data=marginals,
      initial_measurements=initial_measurements,
  )
  if mechanism_result.synthetic_data is not None:
    synthetic_discrete = mechanism_result.synthetic_data
  else:
    synthetic_discrete = dm_common.generate_synthetic_data(
        mechanism_result.model,
        rng,
        use_jax=synth.config.use_jax_for_generation,
    )
  synthetic_data = codec.decode(synthetic_discrete, rng, column_order)
  return data_generation_v3.DataGenerationResult(
      synthetic_data=synthetic_data,
      discrete_mechanism_result=dataclasses.replace(
          mechanism_result,
          synthetic_data=synthetic_discrete,
      ),
      codec=codec,
  )


def _run_two_pass(
    synth: data_generation_v3.TabularMechanism,
    rng: np.random.Generator,
    create_rows_fn: Callable[[beam.Pipeline], beam.PCollection],
    *,
    temp_location: str | None = None,
    pipeline_kwargs: dict[str, Any] | None = None,
) -> data_generation_v3.DataGenerationResult:
  """Two-pass Beam pipeline that delegates to a local TabularConfig."""

  inits = cast(dict[str, CalibratedInitializer], synth.initializers)
  if pipeline_kwargs is None:
    pipeline_kwargs = {}

  created_temp_dir = temp_location is None
  temp_dir = temp_location or tempfile.mkdtemp(prefix='dpsynth_beam_')
  summary_path = FileSystems.join(temp_dir, 'sufficient_stats.bin')
  count_path = FileSystems.join(temp_dir, 'row_count.bin')
  marginals_path = FileSystems.join(temp_dir, 'clique_vector.bin')
  try:
    # Pass 1: privately learn distribution of each column independently.
    # Beam computes lightweight per-column sufficient statistics in a
    # distributed pass; these are small, so we materialize them on the driver.
    with beam.Pipeline(**pipeline_kwargs) as p:
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
    num_rows = int(_read(count_path))
    column_measurements = run_from_summary(
        sparse_stats, inits, rng, num_rows=num_rows
    )
    logging.info('[DPSynth/Beam]: Pass 1 complete.')

    # Ask the configured discrete mechanism which marginals it needs.
    mbi_domain = data_generation_v3.TabularCodec.from_measurements(
        column_measurements, synth.schema
    ).mbi_domain

    assert hasattr(synth.config.discrete_mechanism, 'supporting_cliques')
    workload = synth.config.discrete_mechanism.supporting_cliques(mbi_domain)

    # Pass 2: compute the marginal workload.
    with beam.Pipeline(**pipeline_kwargs) as p:
      rows = create_rows_fn(p)
      marginals = rows | ComputeMarginals(
          column_measurements,
          dict(synth.schema),
          workload,
      )
      _ = marginals | 'WriteCliqueVector' >> beam.Map(
          _write, path=marginals_path
      )
    clique_vector = _read(marginals_path)
    logging.info('[DPSynth/Beam]: Pass 2 complete.')

    # Run the discrete mechanism and decode on the driver.
    return generate_from_marginals(
        synth, rng, column_measurements, clique_vector
    )
  finally:
    # Only remove a temp dir we created; never a user-supplied temp_location.
    if created_temp_dir:
      shutil.rmtree(temp_dir, ignore_errors=True)


@dataclasses.dataclass(frozen=True)
class BeamTabularMechanism(api.CalibratedMechanism):
  """Beam-backed DPMechanism with the TabularMechanism calibrate->run API."""

  synthesizer: data_generation_v3.TabularMechanism
  temp_location: str | None = None
  pipeline_options: beam.options.pipeline_options.PipelineOptions | None = None

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    return self.synthesizer.dp_event

  def __call__(
      self,
      rng: np.random.Generator,
      create_rows_fn: Callable[[beam.Pipeline], beam.PCollection],
  ) -> data_generation_v3.DataGenerationResult:
    return _run_two_pass(
        self.synthesizer,
        rng,
        create_rows_fn,
        temp_location=self.temp_location,
        pipeline_kwargs={'options': self.pipeline_options},
    )


@dataclasses.dataclass(frozen=True)
class BeamTabularConfig(api.MechanismConfig):
  """Beam-backed DPMechanism with the TabularConfig calibrate->run API.

  Usage::

      config = data_generation_v3.TabularConfig(domains=domains)
      beam_synth = BeamTabularConfig(config).configure(budget=1.0)
      result = beam_synth(rng, create_rows_fn)

  Attributes:
    synthesizer: The wrapped local-mode TabularConfig. Supplies the domain,
      sub-mechanisms, constraints, and privacy calibration.
    temp_location: Directory used to shuttle small singleton results between the
      pipeline and the driver. Must be readable and writable by all workers --
      i.e. a shared distributed filesystem for distributed runners. Defaults to
      a local temp directory, which is only valid for in-process runners.
    pipeline_options: Optional Beam pipeline options applied to both passes.
  """

  synthesizer: data_generation_v3.TabularConfig
  temp_location: str | None = None
  pipeline_options: beam.options.pipeline_options.PipelineOptions | None = None

  def __post_init__(self):
    if not hasattr(self.synthesizer.discrete_mechanism, 'supporting_cliques'):
      raise ValueError(
          'self.synthesizer.discrete_mechanism must have a supporting_cliques'
          ' method.'
      )

  def configure(self, schema=None, *, budget, delta=0) -> BeamTabularMechanism:
    """Returns a copy whose synthesizer is configured with the given budget."""
    synthesizer = self.synthesizer.configure(
        schema,
        budget=budget,
        delta=delta,
    )
    return BeamTabularMechanism(
        synthesizer=synthesizer,
        temp_location=self.temp_location,
        pipeline_options=self.pipeline_options,
    )
