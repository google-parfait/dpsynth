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
    rows = [{'x': 10}, {'x': 10}, {'x': 50}, {'x': 90}]
    counts = self._run(rows, attr)
    self.assertEqual(counts.get(10, 0), 2)
    self.assertEqual(counts.get(50, 0), 1)
    self.assertEqual(counts.get(90, 0), 1)
    self.assertEqual(sum(counts.values()), 4)

  def test_nan_clip_to_range_true(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=True
    )
    rows = [{'x': float('nan')}, {'x': None}, {'x': 50}]
    counts = self._run(rows, attr)
    self.assertEqual(counts.get(0, 0), 2)
    self.assertEqual(counts.get(50, 0), 1)
    self.assertEqual(sum(counts.values()), 3)

  def test_nan_clip_to_range_false(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=False
    )
    rows = [{'x': float('nan')}, {'x': 50}, {'x': 75}]
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
    beam_counts = self._run([{'x': v} for v in values], attr)
    ref_counts = self._ref_counts(values, attr)
    self.assertEqual(beam_counts, ref_counts)
    self.assertEqual(ref_counts, {0: 2, 50: 1})

  def test_beam_matches_in_memory_clip_false_drops_ood(self):
    # clip_to_range=False: NaN and values outside [min, max] are dropped.
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=False
    )
    values = [float('nan'), -5, 150, 50, 75]
    beam_counts = self._run([{'x': v} for v in values], attr)
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
        [{'x': v} for v in values], attr, max_grid_size=2, num_partitions=1
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
        {'col': 'a'},
        {'col': 'a'},
        {'col': 'b'},
        {'col': 'c'},
        {'col': 'c'},
        {'col': 'c'},
        {'col': 'z'},  # unknown → mapped to 'unk' (index 0)
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
        {'col': 'apple'},
        {'col': 'apple'},
        {'col': 'banana'},
        {'col': 'cherry'},
        {'col': 'cherry'},
        {'col': 'cherry'},
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
        {'score': 25.0, 'grade': 'a', 'tag': 'p'},
        {'score': 50.0, 'grade': 'b', 'tag': 'q'},
        {'score': 75.0, 'grade': 'a', 'tag': 'p'},
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
        {'color': 'a', 'size': 0},
        {'color': 'a', 'size': 5},
        {'color': 'b', 'size': 5},
        {'color': 'b', 'size': 10},
        {'color': 'c', 'size': 0},
        {'color': 'c', 'size': 0},
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
    )

    # Stage 2: compute marginals.
    workload = [('color',), ('size',), ('color', 'size')]
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
    self.assertLen(cv.cliques, 3)

    # 1-way: color [a=2, b=2, c=2].
    np.testing.assert_array_equal(
        cv.tables[('color',)].datavector(),
        [2, 2, 2],
    )
    # 1-way: size total equals number of rows.
    self.assertEqual(cv.tables[('size',)].datavector().sum(), 6)
    # 2-way: shape matches product of column sizes, total equals rows.
    joint = cv.tables[('color', 'size')]
    expected_size = cms['color'].categorical_attribute.size
    expected_size *= cms['size'].categorical_attribute.size
    self.assertEqual(joint.domain.size(), expected_size)
    self.assertEqual(joint.datavector().sum(), 6)


class BeamTabularConfigTest(parameterized.TestCase):
  """End-to-end tests for the public BeamTabularConfig API."""

  def _domains(self):
    return {
        'color': domain.CategoricalAttribute(possible_values=['r', 'g', 'b']),
        'size': domain.CategoricalAttribute(possible_values=['s', 'm', 'l']),
    }

  def test_end_to_end_generates_synthetic_data(self):
    synth = data_generation_v3.TabularConfig()
    beam_synth = beam_adapter.BeamTabularConfig(synth).configure(
        self._domains(), budget=100.0
    )
    rows = [
        {'color': 'r', 'size': 's'},
        {'color': 'g', 'size': 'm'},
        {'color': 'b', 'size': 'l'},
    ] * 200  # 600 rows for statistical stability.

    result = beam_synth(np.random.default_rng(42), _rows_fn(rows))

    self.assertIsInstance(result, data_generation_v3.DataGenerationResult)
    # MST uses a noisy total count, so the row count is approximate.
    self.assertBetween(len(result.synthetic_data), 550, 650)
    self.assertCountEqual(result.synthetic_data.columns, ['color', 'size'])

  def test_end_to_end_mixed_types(self):
    domains = {
        'age': domain.NumericalAttribute(min_value=0, max_value=100),
        'grade': domain.CategoricalAttribute(possible_values=['a', 'b', 'c']),
    }
    synth = data_generation_v3.TabularConfig()
    beam_synth = beam_adapter.BeamTabularConfig(synth).configure(
        domains, budget=100.0
    )
    rng_data = np.random.default_rng(0)
    rows = [
        {
            'age': float(rng_data.integers(0, 100)),
            'grade': rng_data.choice(['a', 'b', 'c']),
        }
        for _ in range(500)
    ]

    result = beam_synth(np.random.default_rng(42), _rows_fn(rows))

    self.assertIsInstance(result, data_generation_v3.DataGenerationResult)
    self.assertBetween(len(result.synthetic_data), 450, 550)
    self.assertCountEqual(result.synthetic_data.columns, ['age', 'grade'])

  @parameterized.named_parameters(
      ('mst', discrete_mechanisms.MSTConfig(pgm_iters=250), None),
      (
          'independent',
          discrete_mechanisms.IndependentConfig(),
          None,
      ),
      (
          'direct',
          discrete_mechanisms.DirectConfig(pgm_iters=250),
          [('a',), ('b',), ('a', 'b')],
      ),
  )
  def test_runs_across_mechanisms(self, mechanism, workload):
    """The pipeline generalizes to any mechanism via supporting_cliques."""
    domains = {
        'a': domain.CategoricalAttribute(possible_values=['x', 'y']),
        'b': domain.CategoricalAttribute(possible_values=['p', 'q', 'r']),
    }
    synth = data_generation_v3.TabularConfig(discrete_mechanism=mechanism)
    beam_synth = beam_adapter.BeamTabularConfig(synth).configure(
        domains, budget=100.0, workload=workload
    )
    rows = [
        {'a': 'x', 'b': 'p'},
        {'a': 'y', 'b': 'q'},
        {'a': 'x', 'b': 'r'},
    ] * 100

    result = beam_synth(np.random.default_rng(0), _rows_fn(rows))

    self.assertIsInstance(result, data_generation_v3.DataGenerationResult)
    self.assertCountEqual(result.synthetic_data.columns, ['a', 'b'])
    self.assertNotEmpty(result.synthetic_data)

  def test_total_count_matches_input_under_high_budget(self):
    """With negligible noise, synthetic row count matches the input (F2)."""
    domains = {'a': domain.CategoricalAttribute(possible_values=['x', 'y'])}
    synth = data_generation_v3.TabularConfig()
    beam_synth = beam_adapter.BeamTabularConfig(synth).configure(
        domains, budget=1e8
    )
    rows = [{'a': 'x'}, {'a': 'y'}] * 150  # 300 rows.

    result = beam_synth(np.random.default_rng(0), _rows_fn(rows))

    self.assertBetween(len(result.synthetic_data), 298, 302)

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
    synth = data_generation_v3.TabularConfig()
    beam_synth = beam_adapter.BeamTabularConfig(synth).configure(
        schema, budget=100.0
    )
    # The data never contains (a0, b1). Without enforcement, independent
    # (a, b) marginals would put ~25% of mass on that cell; forwarding the
    # constraint suppresses it to a few percent (mbi's constrained sampling
    # may still leak a rare row).
    rows = [
        {'a': 'a0', 'b': 'b0'},
        {'a': 'a1', 'b': 'b1'},
    ] * 150

    result = beam_synth(np.random.default_rng(0), _rows_fn(rows))

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
    synth = data_generation_v3.TabularConfig()
    beam_synth = beam_adapter.BeamTabularConfig(synth).configure(
        domains, budget=100.0
    )
    rows = [
        {'z': 'a', 'm': 'c', 'a': 'e'},
        {'z': 'b', 'm': 'd', 'a': 'f'},
    ] * 50

    result = beam_synth(np.random.default_rng(0), _rows_fn(rows))

    self.assertEqual(list(result.synthetic_data.columns), ['z', 'm', 'a'])

  def test_configure_returns_calibrated_wrapper(self):
    beam_synth = beam_adapter.BeamTabularConfig(
        data_generation_v3.TabularConfig()
    )

    configured = beam_synth.configure(self._domains(), budget=1.0)

    self.assertIsInstance(configured, beam_adapter.BeamTabularMechanism)
    # dp_event is delegated to the wrapped, now-calibrated synthesizer.
    self.assertIsNotNone(configured.dp_event)
    # The original wrapper is left uncalibrated (configure returns a copy).
    with self.assertRaises(Exception):
      _ = beam_synth.dp_event

  def test_inherited_calibrate_produces_calibrated_wrapper(self):
    # calibrate is inherited from DPMechanism; it binary-searches a zCDP budget
    # by repeatedly calling our configure (which delegates to the synthesizer).
    beam_synth = beam_adapter.BeamTabularConfig(
        data_generation_v3.TabularConfig()
    )

    calibrated = beam_synth.calibrate(self._domains(), epsilon=1.0, delta=1e-6)

    self.assertIsInstance(calibrated, beam_adapter.BeamTabularMechanism)
    self.assertIsNotNone(calibrated.dp_event)

  def test_uncalibrated_call_raises(self):
    beam_synth = beam_adapter.BeamTabularConfig(
        data_generation_v3.TabularConfig()
    )
    with self.assertRaises(Exception):
      beam_synth(np.random.default_rng(0), lambda p: p)

  def test_honors_temp_location(self):
    domains = {'a': domain.CategoricalAttribute(possible_values=['x', 'y'])}
    synth = data_generation_v3.TabularConfig()
    temp_dir = self.create_tempdir().full_path
    beam_synth = beam_adapter.BeamTabularConfig(
        synth, temp_location=temp_dir
    ).configure(domains, budget=100.0)
    rows = [{'a': 'x'}, {'a': 'y'}] * 50

    result = beam_synth(np.random.default_rng(0), _rows_fn(rows))

    self.assertIsInstance(result, data_generation_v3.DataGenerationResult)
    self.assertTrue(os.path.exists(os.path.join(temp_dir, 'clique_vector.bin')))

  def _single_col_synth(self):
    domains = {'a': domain.CategoricalAttribute(possible_values=['x', 'y'])}
    synth = data_generation_v3.TabularConfig()
    return beam_adapter.BeamTabularConfig(synth).configure(
        domains, budget=100.0
    )

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
    beam_synth = self._single_col_synth()
    rows = [{'a': 'x'}, {'a': 'y'}] * 50
    created, fake_mkdtemp = self._spy_mkdtemp()

    with mock.patch.object(tempfile, 'mkdtemp', fake_mkdtemp):
      beam_synth(np.random.default_rng(0), _rows_fn(rows))

    self.assertLen(created, 1)
    self.assertFalse(os.path.exists(created[0]))

  def test_cleans_up_created_temp_dir_on_failure(self):
    beam_synth = self._single_col_synth()
    created, fake_mkdtemp = self._spy_mkdtemp()

    def failing_rows_fn(_):
      raise ValueError('boom')

    with mock.patch.object(tempfile, 'mkdtemp', fake_mkdtemp):
      with self.assertRaises(ValueError):
        beam_synth(np.random.default_rng(0), failing_rows_fn)

    self.assertLen(created, 1)
    self.assertFalse(os.path.exists(created[0]))

  def test_forwards_pipeline_options_to_both_passes(self):
    domains = {'a': domain.CategoricalAttribute(possible_values=['x', 'y'])}
    synth = data_generation_v3.TabularConfig()
    options = pipeline_options.PipelineOptions(flags=['--runner=DirectRunner'])
    beam_synth = beam_adapter.BeamTabularConfig(
        synth, pipeline_options=options
    ).configure(domains, budget=100.0)
    rows = [{'a': 'x'}, {'a': 'y'}] * 50
    seen_options = []
    real_pipeline = beam.Pipeline

    def spy_pipeline(*args, **kwargs):
      seen_options.append(kwargs.get('options'))
      return real_pipeline(*args, **kwargs)

    with mock.patch.object(beam, 'Pipeline', spy_pipeline):
      beam_synth(np.random.default_rng(0), _rows_fn(rows))

    # Both passes must receive the caller-provided options object.
    self.assertLen(seen_options, 2)
    self.assertTrue(all(o is options for o in seen_options))


if __name__ == '__main__':
  absltest.main()
