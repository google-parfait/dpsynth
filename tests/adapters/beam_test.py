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


from __future__ import annotations

import multiprocessing
import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.options import pipeline_options
from dpsynth import constraints
from dpsynth import data_generation_v3
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth.adapters import beam as beam_adapter
from dpsynth.local_mode import initialization
import mbi
import numpy as np
import pandas as pd

_manager = None
_test_results = None


def setUpModule():
  global _manager, _test_results
  _manager = multiprocessing.Manager()
  _test_results = _manager.list()


def tearDownModule():
  if _manager is not None:
    _manager.shutdown()


def _store(x):
  _test_results.append(x)


def _rows_fn(rows):
  """Returns a create_rows_fn that emits the given in-memory rows."""
  return lambda p: p | beam.Create(rows)


class NumericalHistogramTest(absltest.TestCase):

  def _run(self, rows, attr, max_grid_size=101, num_partitions=4):
    init = initialization.NumericalInitializerConfig(
        num_partitions=num_partitions,
        max_grid_size=max_grid_size,
    ).configure(attr, budget=np.inf)
    del _test_results[:]
    with beam.Pipeline() as p:
      stats = (
          p
          | beam.Create(rows)
          | beam_adapter.ComputeSufficientStats({'x': init})
      )
      _ = stats | beam.combiners.ToDict() | beam.Map(_store)
    return dict(_test_results[0]['x'])

  def _ref_counts(self, values, attr, max_grid_size=101, num_partitions=4):
    """In-memory grid histogram as an {index: count} dict."""
    init = initialization.NumericalInitializerConfig(
        num_partitions=num_partitions,
        max_grid_size=max_grid_size,
    ).configure(attr, budget=np.inf)
    dense = init._grid_histogram(np.asarray(values, dtype=float))
    return {i: int(c) for i, c in enumerate(dense) if c}

  def test_basic_histogram(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=100)
    rows = [(10,), (10,), (50,), (90,)]
    counts = self._run(rows, attr)
    self.assertEqual(counts.get(10, 0), 2)
    self.assertEqual(counts.get(50, 0), 1)
    self.assertEqual(counts.get(90, 0), 1)
    self.assertEqual(sum(counts.values()), 4)

  def test_nan_clip_to_range_true(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=True
    )
    rows = [(float('nan'),), (None,), (50,)]
    counts = self._run(rows, attr)
    self.assertEqual(counts.get(0, 0), 2)
    self.assertEqual(counts.get(50, 0), 1)
    self.assertEqual(sum(counts.values()), 3)

  def test_nan_clip_to_range_false(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=False
    )
    rows = [(float('nan'),), (50,), (75,)]
    counts = self._run(rows, attr)
    self.assertNotIn(0, counts)
    self.assertEqual(counts.get(50, 0), 1)
    self.assertEqual(counts.get(75, 0), 1)
    self.assertEqual(sum(counts.values()), 2)

  def test_beam_matches_in_memory_clip_true_nan(self):
    # clip_to_range=True: NaN/None fold into the minimum bin; nothing dropped.
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=True
    )
    values = [float('nan'), None, 50]
    beam_counts = self._run([(v,) for v in values], attr)
    ref_counts = self._ref_counts(values, attr)
    self.assertEqual(beam_counts, ref_counts)
    self.assertEqual(ref_counts, {0: 2, 50: 1})

  def test_beam_matches_in_memory_clip_false_drops_ood(self):
    # clip_to_range=False: NaN and values outside [min, max] are dropped.
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=False
    )
    values = [float('nan'), -5, 150, 50, 75]
    beam_counts = self._run([(v,) for v in values], attr)
    ref_counts = self._ref_counts(values, attr)
    self.assertEqual(beam_counts, ref_counts)
    self.assertEqual(ref_counts, {50: 1, 75: 1})

  def test_beam_matches_in_memory_int_at_max_value(self):
    # Regression: an integer value at max_value used to yield a grid index of
    # grid_size (IndexError) when the grid step > 1. Both paths must now fold
    # it into the top bin and agree.
    attr = domain.NumericalAttribute(min_value=0, max_value=100, dtype='int')
    values = [0, 50, 100, 100]
    beam_counts = self._run(
        [(v,) for v in values], attr, max_grid_size=2, num_partitions=1
    )
    ref_counts = self._ref_counts(
        values, attr, max_grid_size=2, num_partitions=1
    )
    self.assertEqual(beam_counts, ref_counts)
    self.assertEqual(sum(beam_counts.values()), 4)


class CategoricalCountsTest(absltest.TestCase):

  def test_basic_counts(self):
    attr = domain.CategoricalAttribute(
        possible_values=['unk', 'a', 'b', 'c'],
        out_of_domain_index=0,
    )
    init = initialization.CategoricalInitializerConfig().configure(
        attr, budget=np.inf
    )
    rows = [
        ('a',),
        ('a',),
        ('b',),
        ('c',),
        ('c',),
        ('c',),
        ('z',),  # unknown -> mapped to 'unk' (index 0)
    ]
    del _test_results[:]
    with beam.Pipeline() as p:
      stats = (
          p
          | beam.Create(rows)
          | beam_adapter.ComputeSufficientStats({'col': init})
      )
      _ = stats | beam.combiners.ToDict() | beam.Map(_store)
    counts = dict(_test_results[0]['col'])
    self.assertEqual(counts.get(0, 0), 1)
    self.assertEqual(counts.get(1, 0), 2)
    self.assertEqual(counts.get(2, 0), 1)
    self.assertEqual(counts.get(3, 0), 3)
    self.assertEqual(sum(counts.values()), 7)


class OpenSetCountsTest(absltest.TestCase):

  def test_basic_counts(self):
    attr = domain.OpenSetCategoricalAttribute(default_value='<OOD>')
    init = initialization.OpenSetInitializerConfig(min_count=1).configure(
        attr, budget=np.inf
    )
    rows = [
        ('apple',),
        ('apple',),
        ('banana',),
        ('cherry',),
        ('cherry',),
        ('cherry',),
    ]
    del _test_results[:]
    with beam.Pipeline() as p:
      stats = (
          p
          | beam.Create(rows)
          | beam_adapter.ComputeSufficientStats({'col': init})
      )
      _ = stats | beam.combiners.ToDict() | beam.Map(_store)
    counts = dict(_test_results[0]['col'])
    self.assertEqual(counts['apple'], 2)
    self.assertEqual(counts['banana'], 1)
    self.assertEqual(counts['cherry'], 3)
    self.assertEqual(sum(counts.values()), 6)


class RunFromSummaryTest(absltest.TestCase):

  def test_end_to_end_mixed(self):
    num_attr = domain.NumericalAttribute(min_value=0, max_value=100)
    cat_attr = domain.CategoricalAttribute(possible_values=['a', 'b'])
    open_attr = domain.OpenSetCategoricalAttribute(default_value='<OOD>')

    inits = {
        'score': (
            initialization.NumericalInitializerConfig(
                num_partitions=4
            ).configure(num_attr, budget=np.inf, delta=1)
        ),
        'grade': (
            initialization.CategoricalInitializerConfig().configure(
                cat_attr, budget=np.inf, delta=1
            )
        ),
        'tag': (
            initialization.OpenSetInitializerConfig(min_count=1).configure(
                open_attr, budget=np.inf, delta=1
            )
        ),
    }

    rows = [
        (25.0, 'a', 'p'),
        (50.0, 'b', 'q'),
        (75.0, 'a', 'p'),
    ]
    rng = np.random.default_rng(42)

    # Sufficient stats are computed in Beam, then DP init runs on the driver.
    del _test_results[:]
    with beam.Pipeline() as p:
      stats = p | beam.Create(rows) | beam_adapter.ComputeSufficientStats(inits)
      _ = stats | beam.combiners.ToDict() | beam.Map(_store)
    measurements = beam_adapter.run_from_summary(_test_results[0], inits, rng)

    self.assertLen(measurements, 3)
    for cm in measurements.values():
      self.assertIsInstance(cm, initialization.ColumnMeasurement)

  def test_all_ood_numerical_column_with_no_clip(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=False
    )
    init = initialization.NumericalInitializerConfig(
        num_partitions=2
    ).configure(attr, budget=np.inf)
    rows = [(float('nan'),), (-10.0,), (200.0,)]
    del _test_results[:]
    with beam.Pipeline() as p:
      stats = (
          p
          | beam.Create(rows)
          | beam_adapter.ComputeSufficientStats({'x': init})
      )
      _ = stats | beam.combiners.ToDict() | beam.Map(_store)
    self.assertNotIn('x', _test_results[0])
    cms = beam_adapter.run_from_summary(
        _test_results[0], {'x': init}, np.random.default_rng(0), num_rows=3
    )
    self.assertEqual(cms['x'].noisy_counts[0], 3.0)
    self.assertEqual(cms['x'].noisy_counts[1:].sum(), 0.0)


class ComputeMarginalsTest(absltest.TestCase):

  def test_marginals_match_manual_counts(self):
    cat_attr = domain.CategoricalAttribute(possible_values=['a', 'b', 'c'])
    num_attr = domain.NumericalAttribute(min_value=0, max_value=10)
    cat_init = initialization.CategoricalInitializerConfig().configure(
        cat_attr, budget=np.inf
    )
    num_init = initialization.NumericalInitializerConfig(
        num_partitions=4,
        max_grid_size=11,
    ).configure(num_attr, budget=np.inf)
    domains = {'color': cat_attr, 'size': num_attr}
    rows = [
        ('a', 0),
        ('a', 5),
        ('b', 5),
        ('b', 10),
        ('c', 0),
        ('c', 0),
    ]

    # Stage 1: get ColumnMeasurements.
    inits = {'color': cat_init, 'size': num_init}
    rng = np.random.default_rng(42)
    del _test_results[:]
    with beam.Pipeline() as p:
      stats = (
          p
          | 'Create1' >> beam.Create(rows)
          | beam_adapter.ComputeSufficientStats(inits)
      )
      _ = stats | 'ToDict1' >> beam.combiners.ToDict() | beam.Map(_store)
    cms = beam_adapter.run_from_summary(
        _test_results[0],
        inits,
        rng,
        num_rows=len(rows),
    )

    # Stage 2: compute marginals.
    workload = [('color', 'size'), ('size',), ('color',)]
    del _test_results[:]
    with beam.Pipeline() as p:
      result = (
          p
          | 'Create2' >> beam.Create(rows)
          | beam_adapter.ComputeMarginals(cms, domains, workload)
      )
      _ = result | beam.Map(_store)

    cv = _test_results[0]
    self.assertIsInstance(cv, mbi.CliqueVector)
    self.assertEqual(cv.cliques, tuple(workload))

    # 1-way: color [a=2, b=2, c=2].
    np.testing.assert_array_equal(
        cv.tables[('color',)].datavector(),
        [2, 2, 2],
    )
    # 1-way and 2-way: exact bin counts match ColumnCodec encoding.
    size_codec = data_generation_v3.ColumnCodec(cms['size'], num_attr)
    size_bins = size_codec.encode(np.array([r[1] for r in rows]))
    num_size_bins = cms['size'].categorical_attribute.size
    expected_size = np.bincount(size_bins, minlength=num_size_bins)
    np.testing.assert_array_equal(
        cv.tables[('size',)].datavector(), expected_size
    )
    expected_joint = np.zeros((3, num_size_bins), dtype=float)
    np.add.at(expected_joint, ([0, 0, 1, 1, 2, 2], size_bins), 1.0)
    np.testing.assert_array_equal(
        cv.tables[('color', 'size')].values, expected_joint
    )

  def test_mismatched_row_width_raises(self):
    attr = domain.CategoricalAttribute(possible_values=['a', 'b'])
    init = initialization.CategoricalInitializerConfig().configure(
        attr, budget=np.inf
    )
    encode_cols = beam_adapter._EncodeColumns({'x': init})
    with self.assertRaises(ValueError):
      list(encode_cols.process([('a',), ('a', 'b')]))
    with self.assertRaises(ValueError):
      list(encode_cols.process([('a', 'b')]))

    cm = init.from_summary(np.random.default_rng(0), np.array([1.0, 1.0]))
    encode_proj = beam_adapter._EncodeAndProject(
        {'x': cm}, {'x': attr}, [('x',)]
    )
    with self.assertRaises(ValueError):
      list(encode_proj.process([('a',), ('a', 'b')]))
    with self.assertRaises(ValueError):
      list(encode_proj.process([('a', 'b')]))


class ExecuteTest(parameterized.TestCase):
  """End-to-end tests for the public beam.execute API."""

  def _domains(self):
    return {
        'color': domain.CategoricalAttribute(possible_values=['r', 'g', 'b']),
        'size': domain.CategoricalAttribute(possible_values=['s', 'm', 'l']),
    }

  def test_end_to_end_generates_synthetic_data(self):
    mech = data_generation_v3.TabularConfig().configure(
        self._domains(), budget=100.0
    )
    rows = [
        ('r', 's'),
        ('g', 'm'),
        ('b', 'l'),
    ] * 200  # 600 rows for statistical stability.

    result = beam_adapter.execute(
        mech, np.random.default_rng(42), _rows_fn(rows)
    )

    self.assertIsInstance(result, data_generation_v3.DataGenerationResult)
    # MST uses a noisy total count, so the row count is approximate.
    self.assertBetween(len(result.synthetic_data), 550, 650)
    self.assertCountEqual(result.synthetic_data.columns, ['color', 'size'])

  def test_end_to_end_mixed_types(self):
    domains = {
        'age': domain.NumericalAttribute(min_value=0, max_value=100),
        'grade': domain.CategoricalAttribute(possible_values=['a', 'b', 'c']),
    }
    mech = data_generation_v3.TabularConfig().configure(domains, budget=100.0)
    rng_data = np.random.default_rng(0)
    rows = [
        (
            float(rng_data.integers(0, 100)),
            str(rng_data.choice(['a', 'b', 'c'])),
        )
        for _ in range(500)
    ]

    result = beam_adapter.execute(
        mech, np.random.default_rng(42), _rows_fn(rows)
    )

    self.assertIsInstance(result, data_generation_v3.DataGenerationResult)
    self.assertBetween(len(result.synthetic_data), 450, 550)
    self.assertCountEqual(result.synthetic_data.columns, ['age', 'grade'])

  @parameterized.named_parameters(
      ('mst', discrete_mechanisms.MSTConfig(pgm_iters=250)),
      (
          'independent',
          discrete_mechanisms.IndependentConfig(),
      ),
      (
          'direct',
          discrete_mechanisms.DirectConfig(
              prespecified_marginal_queries=[('a',), ('b',), ('a', 'b')],
              pgm_iters=250,
          ),
      ),
  )
  def test_runs_across_mechanisms(self, mechanism):
    """The pipeline generalizes to any mechanism via supporting_cliques."""
    domains = {
        'a': domain.CategoricalAttribute(possible_values=['x', 'y']),
        'b': domain.CategoricalAttribute(possible_values=['p', 'q', 'r']),
    }
    mech = data_generation_v3.TabularConfig(
        discrete_mechanism=mechanism
    ).configure(domains, budget=100.0)
    rows = [
        ('x', 'p'),
        ('y', 'q'),
        ('x', 'r'),
    ] * 100

    result = beam_adapter.execute(
        mech, np.random.default_rng(0), _rows_fn(rows)
    )

    self.assertIsInstance(result, data_generation_v3.DataGenerationResult)
    self.assertCountEqual(result.synthetic_data.columns, ['a', 'b'])
    self.assertNotEmpty(result.synthetic_data)

  def test_total_count_matches_input_under_high_budget(self):
    """With negligible noise, synthetic row count matches the input (F2)."""
    domains = {'a': domain.CategoricalAttribute(possible_values=['x', 'y'])}
    mech = data_generation_v3.TabularConfig().configure(domains, budget=1e8)
    rows = [('x',), ('y',)] * 150  # 300 rows.

    result = beam_adapter.execute(
        mech, np.random.default_rng(0), _rows_fn(rows)
    )

    self.assertBetween(len(result.synthetic_data), 298, 302)

  def test_num_rows_overrides_generated_count(self):
    domains = {'a': domain.CategoricalAttribute(possible_values=['x', 'y'])}
    mech = data_generation_v3.TabularConfig().configure(domains, budget=100.0)
    rows = [('x',), ('y',)] * 150  # 300 input rows.

    result = beam_adapter.execute(
        mech, np.random.default_rng(0), _rows_fn(rows), num_rows=25
    )

    self.assertLen(result.synthetic_data, 25)

  def test_respects_impossible_combinations(self):
    """Cross-attribute constraints reach the discrete mechanism (F4)."""
    a_attr = domain.CategoricalAttribute(possible_values=['a0', 'a1'])
    b_attr = domain.CategoricalAttribute(possible_values=['b0', 'b1'])
    domains = {'a': a_attr, 'b': b_attr}
    constraint = constraints.Constraint(
        attribute_names=('a', 'b'),
        attribute_domains=(a_attr, b_attr),
        impossible_combinations=[('a0', 'b1')],
    )
    schema = domain.Schema(domains, constraints=(constraint,))
    mech = data_generation_v3.TabularConfig().configure(schema, budget=100.0)
    # The data never contains (a0, b1). Without enforcement, independent
    # (a, b) marginals would put ~25% of mass on that cell; forwarding the
    # constraint suppresses it to a few percent (mbi's constrained sampling
    # may still leak a rare row).
    rows = [
        ('a0', 'b0'),
        ('a1', 'b1'),
    ] * 150

    result = beam_adapter.execute(
        mech, np.random.default_rng(0), _rows_fn(rows)
    )

    forbidden = (result.synthetic_data['a'] == 'a0') & (
        result.synthetic_data['b'] == 'b1'
    )
    self.assertLess(forbidden.mean(), 0.1)

  def test_preserves_domain_column_order(self):
    """Output columns follow domain declaration order (F5)."""
    domains = {
        'z': domain.CategoricalAttribute(possible_values=['a', 'b']),
        'm': domain.CategoricalAttribute(possible_values=['c', 'd']),
        'a': domain.CategoricalAttribute(possible_values=['e', 'f']),
    }
    mech = data_generation_v3.TabularConfig().configure(domains, budget=100.0)
    rows = [
        ('a', 'c', 'e'),
        ('b', 'd', 'f'),
    ] * 50

    result = beam_adapter.execute(
        mech, np.random.default_rng(0), _rows_fn(rows)
    )

    self.assertEqual(list(result.synthetic_data.columns), ['z', 'm', 'a'])

  def test_honors_temp_location(self):
    domains = {'a': domain.CategoricalAttribute(possible_values=['x', 'y'])}
    temp_dir = self.create_tempdir().full_path
    mech = data_generation_v3.TabularConfig().configure(domains, budget=100.0)
    rows = [('x',), ('y',)] * 50

    result = beam_adapter.execute(
        mech,
        np.random.default_rng(0),
        _rows_fn(rows),
        temp_location=temp_dir,
    )

    self.assertIsInstance(result, data_generation_v3.DataGenerationResult)
    self.assertTrue(os.path.exists(os.path.join(temp_dir, 'clique_vector.bin')))

  def _single_col_mech(self):
    domains = {'a': domain.CategoricalAttribute(possible_values=['x', 'y'])}
    return data_generation_v3.TabularConfig().configure(domains, budget=100.0)

  def _spy_mkdtemp(self):
    """Returns (created_paths_list, patched_mkdtemp) recording our temp dirs.

    Beam creates its own pipeline temp dirs via ``tempfile.mkdtemp`` too, so we
    only record the one our adapter creates (identified by its prefix).
    """
    created = []
    real_mkdtemp = tempfile.mkdtemp

    def fake_mkdtemp(*args, **kwargs):
      path = real_mkdtemp(*args, **kwargs)
      if kwargs.get('prefix') == 'dpsynth_beam_':
        created.append(path)
      return path

    return created, fake_mkdtemp

  def test_cleans_up_created_temp_dir_on_success(self):
    mech = self._single_col_mech()
    rows = [('x',), ('y',)] * 50
    created, fake_mkdtemp = self._spy_mkdtemp()

    with mock.patch.object(tempfile, 'mkdtemp', fake_mkdtemp):
      beam_adapter.execute(mech, np.random.default_rng(0), _rows_fn(rows))

    self.assertLen(created, 1)
    self.assertFalse(os.path.exists(created[0]))

  def test_cleans_up_created_temp_dir_on_failure(self):
    mech = self._single_col_mech()
    created, fake_mkdtemp = self._spy_mkdtemp()

    def failing_rows_fn(_):
      raise ValueError('boom')

    with mock.patch.object(tempfile, 'mkdtemp', fake_mkdtemp):
      with self.assertRaises(ValueError):
        beam_adapter.execute(mech, np.random.default_rng(0), failing_rows_fn)

    self.assertLen(created, 1)
    self.assertFalse(os.path.exists(created[0]))

  def test_forwards_pipeline_options_to_both_passes(self):
    domains = {'a': domain.CategoricalAttribute(possible_values=['x', 'y'])}
    options = pipeline_options.PipelineOptions(
        flags=['--runner=DirectRunner', '--job_name=my-job']
    )
    mech = data_generation_v3.TabularConfig().configure(domains, budget=100.0)
    rows = [('x',), ('y',)] * 50
    seen_options = []
    real_pipeline = beam.Pipeline

    def spy_pipeline(*args, **kwargs):
      seen_options.append(kwargs.get('options'))
      return real_pipeline(*args, **kwargs)

    with mock.patch.object(beam, 'Pipeline', spy_pipeline):
      beam_adapter.execute(
          mech,
          np.random.default_rng(0),
          _rows_fn(rows),
          pipeline_options=options,
      )

    # Each pass receives an independent copy with pass-suffixed job_name.
    self.assertLen(seen_options, 2)
    self.assertTrue(all(o is not options for o in seen_options))
    gcloud_cls = pipeline_options.GoogleCloudOptions
    self.assertEqual(options.view_as(gcloud_cls).job_name, 'my-job')
    self.assertEqual(
        seen_options[0].view_as(gcloud_cls).job_name, 'my-job-pass1'
    )
    self.assertEqual(
        seen_options[1].view_as(gcloud_cls).job_name, 'my-job-pass2'
    )

  def test_execute_validation_errors(self):
    domains = {
        'a': domain.CategoricalAttribute(possible_values=['x', 'y']),
        'b': domain.CategoricalAttribute(possible_values=['p', 'q']),
    }
    rng = np.random.default_rng(0)

    mech_compress = data_generation_v3.TabularConfig(
        compress_columns=True
    ).configure(domains, budget=10.0)
    with self.assertRaisesRegex(ValueError, 'compress_columns'):
      beam_adapter.execute(mech_compress, rng, _rows_fn([('x', 'p')]))

  def test_write_atomic_rename_preserves_completed_file(self):
    temp_dir = self.create_tempdir().full_path
    path = os.path.join(temp_dir, 'clique_vector.bin')
    cv = mbi.CliqueVector(
        mbi.Domain(['a'], [2]),
        [('a',)],
        {('a',): mbi.Factor(mbi.Domain(['a'], [2]), np.array([3.0, 7.0]))},
    )
    beam_adapter._write(cv, path)

    # Simulate a speculative backup task whose FileSystems.create opens a
    # sibling temp file and then gets canceled before rename.
    real_create = beam_adapter.FileSystems.create

    class _CanceledWriter:

      def __init__(self, f):
        self._f = f

      def __enter__(self):
        return self

      def __exit__(self, *args):
        self._f.close()

      def write(self, _):
        raise RuntimeError('speculative backup task canceled')

    with mock.patch.object(
        beam_adapter.FileSystems,
        'create',
        side_effect=lambda p: _CanceledWriter(real_create(p)),
    ):
      with self.assertRaises(RuntimeError):
        beam_adapter._write(cv, path)

    loaded = beam_adapter._read(path)
    np.testing.assert_allclose(loaded.project(('a',)).values, [3.0, 7.0])


class SynthesizeTest(parameterized.TestCase):

  @parameterized.parameters(
      (0, discrete_mechanisms.MSTConfig(pgm_iters=50), None),
      (42, discrete_mechanisms.MSTConfig(pgm_iters=50), 30),
      (7, discrete_mechanisms.IndependentConfig(pgm_iters=50), None),
  )
  def test_synthesize_matches_in_memory(self, seed, discrete_mech, num_rows):
    domains = {
        'a': domain.CategoricalAttribute(possible_values=['x', 'y', 'z']),
        'b': domain.CategoricalAttribute(possible_values=['p', 'q']),
    }
    rows = [('x', 'p'), ('y', 'q'), ('z', 'p'), ('x', 'q')] * 25
    df = pd.DataFrame(rows, columns=['a', 'b'])
    config = data_generation_v3.TabularConfig(
        discrete_mechanism=discrete_mech, use_jax_for_generation=True
    )
    mech = config.configure(domains, budget=10.0)

    inner_seed = int(np.random.default_rng(seed).integers(0, 2**31))
    expected = mech(np.random.default_rng(inner_seed), df, num_rows=num_rows)
    expected_records = expected.synthetic_data.to_dict(orient='records')

    del _test_results[:]
    with beam.Pipeline() as p:
      synthesize = beam_adapter.Synthesize(
          mech, np.random.default_rng(seed), num_rows=num_rows
      )
      _ = p | beam.Create(rows) | synthesize | beam.Map(_store)
    self.assertEqual(list(_test_results), expected_records)

  def test_synthesize_rejects_non_categorical_and_compress_columns(self):
    num_domains = {
        'a': domain.NumericalAttribute(min_value=0, max_value=10),
    }
    num_mech = data_generation_v3.TabularConfig().configure(
        num_domains, budget=10.0
    )
    with self.assertRaisesRegex(ValueError, 'CategoricalAttribute'):
      beam_adapter.Synthesize(num_mech, np.random.default_rng(0))

    cat_domains = {
        'a': domain.CategoricalAttribute(possible_values=['x', 'y']),
        'b': domain.CategoricalAttribute(possible_values=['p', 'q']),
    }
    compress_mech = data_generation_v3.TabularConfig(
        compress_columns=True
    ).configure(cat_domains, budget=10.0)
    with self.assertRaisesRegex(ValueError, 'compress_columns'):
      beam_adapter.Synthesize(compress_mech, np.random.default_rng(0))


if __name__ == '__main__':
  absltest.main()
