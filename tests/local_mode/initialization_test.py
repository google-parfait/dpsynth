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

from absl.testing import absltest
import dp_accounting
from dpsynth import domain
from dpsynth.local_mode import initialization
from dpsynth.local_mode import primitives
from dpsynth.local_mode import vectorized_transformations as vtx
import numpy as np


class InitializationTest(absltest.TestCase):

  def test_numerical_initializer_dp_event(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=10)
    initializer = initialization.NumericalInitializerConfig(num_partitions=4)
    event = initializer.configure(attr, budget=1.0).dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    self.assertLen(event.events, 3)
    for e in event.events[:2]:
      self.assertIsInstance(e, dp_accounting.ExponentialMechanismDpEvent)
    self.assertIsInstance(event.events[2], dp_accounting.GaussianDpEvent)
    self.assertAlmostEqual(event.events[2].noise_multiplier, 1.0)

  def test_numerical_initializer_call(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=10)
    rng = np.random.default_rng(0)
    initializer = initialization.NumericalInitializerConfig(num_partitions=4)

    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    measurement = initializer.configure(attr, budget=np.inf)(rng, data)

    self.assertIsInstance(measurement, initialization.NumericalMeasurement)
    self.assertEqual(measurement.categorical_attribute.size, 4)
    self.assertIsNotNone(measurement.noisy_counts)
    self.assertEqual(measurement.stddev, 0.0)
    self.assertIsNotNone(measurement.bin_edges)

    encoded_data = vtx.discretize(data, measurement.bin_edges, attr)
    counts = np.bincount(encoded_data)

    # All 9 data points assigned to exactly 4 bins, each non-empty.
    self.assertEqual(counts.sum(), 9)
    self.assertLen(counts, 4)
    self.assertTrue(np.all(counts > 0))
    np.testing.assert_array_equal(measurement.noisy_counts, counts)

  def test_numerical_initializer_out_of_domain_values(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=False
    )
    rng = np.random.default_rng(0)
    initializer = initialization.NumericalInitializerConfig(num_partitions=4)

    data = np.array(
        [-5.0, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 15.0]
    )
    measurement = initializer.configure(attr, budget=np.inf)(rng, data)

    self.assertEqual(measurement.categorical_attribute.size, 5)
    self.assertIsNotNone(measurement.noisy_counts)
    self.assertLen(measurement.noisy_counts, 5)
    # 3 OOD values (-5.0, nan, 15.0) map to bin 0.
    self.assertEqual(measurement.noisy_counts[0], 3.0)
    self.assertEqual(measurement.noisy_counts.sum(), 11.0)

  def test_numerical_initializer_deduplicates_bin_edges(self):
    """Concentrated data can make quantiles return duplicate edges."""
    attr = domain.NumericalAttribute(min_value=0, max_value=100)
    rng = np.random.default_rng(42)
    initializer = initialization.NumericalInitializerConfig(num_partitions=8)
    # Data is heavily concentrated at 50.
    data = np.array([50] * 100 + [1, 99])
    result = initializer.configure(attr, budget=1.0)(rng, data)

    # After dedup, bin_edges should be strictly increasing.
    edges = result.bin_edges
    self.assertTrue(
        np.all(np.diff(edges) > 0),
        f'bin_edges not strictly increasing: {edges}',
    )
    # And downstream discretize should not crash.
    encoded = vtx.discretize(data, edges, attr)
    self.assertEqual(encoded.shape, data.shape)

  def test_numerical_initializer_integer_data(self):
    """Integer data within a narrow range can collapse quantile edges."""
    attr = domain.NumericalAttribute(min_value=0, max_value=10, dtype='int')
    rng = np.random.default_rng(0)
    initializer = initialization.NumericalInitializerConfig(num_partitions=8)
    # Only 3 distinct values but 8 partitions requested.
    data = np.array([3, 3, 3, 3, 5, 5, 5, 7])
    result = initializer.configure(attr, budget=1.0)(rng, data)

    edges = result.bin_edges
    self.assertTrue(
        np.all(np.diff(edges) > 0),
        f'bin_edges not strictly increasing: {edges}',
    )
    # Domain size may be < 8 due to dedup, but must be >= 2.
    self.assertGreaterEqual(result.categorical_attribute.size, 2)

  def test_numerical_initializer_integer_edges_are_floored(self):
    """Integer attributes should produce integer-valued bin edges."""
    attr = domain.NumericalAttribute(min_value=0, max_value=100, dtype='int')
    rng = np.random.default_rng(42)
    initializer = initialization.NumericalInitializerConfig(num_partitions=4)
    data = np.arange(100)
    result = initializer.configure(attr, budget=100.0)(rng, data)
    # All edges should be integers (integer grid produces exact integer edges).
    np.testing.assert_array_equal(result.bin_edges, np.floor(result.bin_edges))
    # Edges must be within [min_value, max_value - 1].
    self.assertGreaterEqual(result.bin_edges[0], 0)
    self.assertLess(result.bin_edges[-1], 100)

  def test_max_grid_size_below_two_raises(self):
    for bad in (0, 1):
      with self.assertRaises(ValueError):
        initialization.NumericalInitializerConfig(
            num_partitions=10, max_grid_size=bad
        )

  def test_int_grid_reserves_budget_for_jitter_refinement(self):
    # A wide integer range with many partitions must not blow past
    # max_grid_size once the m-fold jitter refinement is applied.
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10_000_000, dtype='int'
    )
    max_grid_size = 100_000
    init = initialization.NumericalInitializerConfig(
        num_partitions=64, max_grid_size=max_grid_size
    )
    m = primitives.jitter_factor(init.num_partitions)
    self.assertLessEqual(
        init.configure(attr, budget=1.0).grid_size * m, max_grid_size
    )

  def test_numerical_initializer_epsilon_ratio_default_and_custom(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=10)
    # Default is 1.0 (uniform), and quantile_budget_fraction defaults to 0.5.
    init_default = initialization.NumericalInitializerConfig(num_partitions=4)
    self.assertEqual(init_default.epsilon_ratio, 1.0)
    self.assertEqual(init_default.quantile_budget_fraction, 0.5)
    calibrated_default = init_default.configure(attr, budget=4.0)
    # quantile_rho = 2.0, count_rho = 2.0.
    # For 4 partitions (2 levels), uniform budget splits quantile_rho equally:
    # 1.0 each. eps = sqrt(8 * rho_level) = sqrt(8 * 1.0) = sqrt(8).
    np.testing.assert_allclose(
        calibrated_default.epsilon_levels,
        (np.sqrt(8.0), np.sqrt(8.0)),
    )
    self.assertAlmostEqual(calibrated_default.sigma, 0.5)

    # Custom ratio (e.g. sqrt(2))
    init_custom = initialization.NumericalInitializerConfig(
        num_partitions=4, epsilon_ratio=np.sqrt(2)
    )
    self.assertEqual(init_custom.epsilon_ratio, np.sqrt(2))
    calibrated_custom = init_custom.configure(attr, budget=4.0)
    # quantile_rho = 2.0. rho_ratio = 2.0. budget_weights = [2.0, 1.0].
    # Leaves get 2/3, root gets 1/3.
    # rho_levels = [2.0 * 2/3, 2.0 * 1/3] = [4/3, 2/3]
    # eps_levels = [sqrt(8 * 4/3), sqrt(8 * 2/3)]
    np.testing.assert_allclose(
        calibrated_custom.epsilon_levels,
        (np.sqrt(8.0 * 4.0 / 3.0), np.sqrt(8.0 * 2.0 / 3.0)),
    )

  def test_integer_edges_at_max_value_absorbed_into_last_bin(self):
    """Edges at max_value are removed; their count goes to the last bin."""
    attr = domain.NumericalAttribute(min_value=0, max_value=10, dtype='int')
    rng = np.random.default_rng(0)
    initializer = initialization.NumericalInitializerConfig(num_partitions=8)
    # A spread of lower values carrying most of the mass, plus a moderate spike
    # at max_value.  The lower values form genuine interior bins, while the top
    # quantile edges land at max_value and must be absorbed into the last bin.
    data = np.concatenate([np.repeat(np.arange(0, 10), 20), np.full(100, 10)])
    result = initializer.configure(attr, budget=100.0)(rng, data)
    # No edge should equal max_value (they get absorbed).
    if len(result.bin_edges) > 0:
      self.assertLess(result.bin_edges[-1], 10)
    self.assertIsNotNone(result.noisy_counts)
    self.assertLen(result.noisy_counts, result.categorical_attribute.size)

  def test_integer_jitter_prevents_spurious_splits(self):
    """Positive jitter should prevent edges from splitting across integers."""
    attr = domain.NumericalAttribute(min_value=0, max_value=100, dtype='int')
    rng = np.random.default_rng(42)
    initializer = initialization.NumericalInitializerConfig(num_partitions=4)
    # Uniform data: with high budget, edges should land at 25, 50, 75.
    data = np.arange(101)
    result = initializer.configure(attr, budget=1000.0)(rng, data)
    # With 4 partitions and 3 edges, no dedup should be needed.
    self.assertLen(result.bin_edges, 3)
    # All edges should be integers.
    np.testing.assert_array_equal(result.bin_edges, np.floor(result.bin_edges))

  def test_integer_heterogeneous_data_buckets(self):
    """Heterogeneous integer data produces sensible bucket partitioning."""
    attr = domain.NumericalAttribute(min_value=0, max_value=10, dtype='int')
    rng = np.random.default_rng(42)
    initializer = initialization.NumericalInitializerConfig(num_partitions=4)
    # Deliberately lumpy distribution: 45 points across 4 distinct values.
    data = np.array([0] * 10 + [3] * 10 + [5] * 17 + [6] * 8)
    result = initializer.configure(attr, budget=np.inf)(rng, data)
    # Edges should be integers strictly inside [min_value, max_value).
    for e in result.bin_edges:
      self.assertEqual(e, int(e), f'edge {e} is not an integer')
      self.assertGreaterEqual(e, 0)
      self.assertLess(e, 10)

    # Discretize and verify every data point is assigned.
    encoded = vtx.discretize(data, result.bin_edges, attr)
    self.assertEqual(encoded.shape, data.shape)
    num_bins = result.categorical_attribute.size
    true_counts = np.bincount(encoded, minlength=num_bins)
    self.assertLen(data, true_counts.sum())
    # Bins covering the data range should be non-empty; trailing bins
    # beyond the data (e.g. [7, 10] when max data is 6) may be empty.
    self.assertGreater(
        np.count_nonzero(true_counts),
        1,
        f'too few occupied bins: counts={true_counts},'
        f' edges={result.bin_edges}',
    )

  def test_numerical_initializer_fixed_bin_edges_dp_event(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10, bin_edges=[2.0, 5.0, 8.0]
    )
    initializer = initialization.NumericalInitializerConfig(num_partitions=4)
    configured = initializer.configure(attr, budget=0.5)
    expected = initialization.NumericalInitializer(
        config=initializer,
        attribute=attr,
        epsilon_levels=(),
        sigma=1.0,
    )
    self.assertEqual(configured, expected)
    self.assertEqual(
        configured.dp_event,
        dp_accounting.ComposedDpEvent([dp_accounting.GaussianDpEvent(1.0)]),
    )

  def test_numerical_initializer_fixed_bin_edges_call(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10, bin_edges=[3.0, 7.0]
    )
    rng = np.random.default_rng(0)
    initializer = initialization.NumericalInitializerConfig(num_partitions=8)
    data = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 9.0, 10.0])
    measurement = initializer.configure(attr, budget=np.inf)(rng, data)

    np.testing.assert_array_equal(measurement.bin_edges, [3.0, 7.0])
    self.assertEqual(measurement.categorical_attribute.size, 3)
    np.testing.assert_array_equal(measurement.noisy_counts, [4.0, 2.0, 3.0])


class CategoricalInitializerTest(absltest.TestCase):

  def test_dp_event(self):
    attr = domain.CategoricalAttribute(possible_values=['A', 'B', 'C'])
    initializer = initialization.CategoricalInitializerConfig()
    event = initializer.configure(attr, budget=0.5).dp_event
    self.assertIsInstance(event, dp_accounting.GaussianDpEvent)
    # rho = 0.5 => sigma = 1/sqrt(2*0.5) = 1.0
    self.assertEqual(event.noise_multiplier, 1.0)

  def test_call_noiseless(self):
    attr = domain.CategoricalAttribute(possible_values=['A', 'B', 'C'])
    rng = np.random.default_rng(0)
    initializer = initialization.CategoricalInitializerConfig()
    data = np.array(['A', 'A', 'B', 'C', 'C', 'C'])
    result = initializer.configure(attr, budget=np.inf)(rng, data)

    self.assertIsInstance(result, initialization.CategoricalMeasurement)
    self.assertEqual(result.categorical_attribute, attr)
    self.assertIsNotNone(result.noisy_counts)
    np.testing.assert_array_equal(result.noisy_counts, [2, 1, 3])
    self.assertEqual(result.stddev, 0.0)

  def test_out_of_domain_values(self):
    attr = domain.CategoricalAttribute(
        possible_values=['<OOD>', 'X', 'Y'], out_of_domain_index=0
    )
    rng = np.random.default_rng(0)
    initializer = initialization.CategoricalInitializerConfig()
    data = np.array(['X', 'Y', 'Z', 'W'])
    result = initializer.configure(attr, budget=np.inf)(rng, data)

    # 'Z' and 'W' are OOD, mapped to index 0 (None).
    np.testing.assert_array_equal(result.noisy_counts, [2, 1, 1])


class OpenSetCategoricalInitializerTest(absltest.TestCase):

  def test_dp_event(self):
    attr = domain.OpenSetCategoricalAttribute(default_value='<OOD>')
    initializer = initialization.OpenSetInitializerConfig()
    event = initializer.configure(attr, budget=0.5, delta=1e-5).dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    self.assertLen(event.events, 2)
    self.assertIsInstance(event.events[0], dp_accounting.GaussianDpEvent)
    self.assertIsInstance(
        event.events[1], dp_accounting.dp_event.EpsilonDeltaDpEvent
    )
    self.assertEqual(event.events[1].delta, 1e-5)

  def test_calibrate(self):
    attr = domain.OpenSetCategoricalAttribute(default_value='<OOD>')
    config = initialization.OpenSetInitializerConfig()
    calibrated = config.calibrate(attr, epsilon=1.0, delta=1e-5)
    self.assertAlmostEqual(calibrated.delta, 5e-6)
    self.assertGreater(calibrated.sigma, 0.0)

  def test_call_noiseless(self):
    attr = domain.OpenSetCategoricalAttribute(default_value='<OOD>')
    rng = np.random.default_rng(42)
    initializer = initialization.OpenSetInitializerConfig()
    # 'A' appears 100 times, 'B' 50, 'C' 1 (rare).
    data = np.array(['A'] * 100 + ['B'] * 50 + ['C'] * 1)
    result = initializer.configure(attr, budget=np.inf, delta=1e-5)(rng, data)

    self.assertIsInstance(result, initialization.OpenSetMeasurement)
    self.assertIsNotNone(result.noisy_counts)
    # With infinite budget, all values with count > 0 should be selected.
    discovered = set(result.categorical_attribute.possible_values)
    self.assertIn('A', discovered)
    self.assertIn('B', discovered)
    self.assertIn('<OOD>', discovered)  # default value always present
    # Default value is always first.
    self.assertEqual(result.categorical_attribute.possible_values[0], '<OOD>')
    self.assertEqual(result.categorical_attribute.out_of_domain_index, 0)

  def test_undiscovered_values_map_to_default(self):
    attr = domain.OpenSetCategoricalAttribute(default_value='OTHER')
    rng = np.random.default_rng(0)
    initializer = initialization.OpenSetInitializerConfig()
    data = np.array(['A'] * 100 + ['B'] * 50)
    result = initializer.configure(attr, budget=np.inf, delta=1e-5)(rng, data)

    cat_attr = result.categorical_attribute
    # Discovered values map to valid indices.
    encoded = vtx.discrete_encode(np.array(['A']), cat_attr)
    self.assertGreater(encoded[0], 0)
    # Unknown value maps to the out-of-domain (default) index at 0.
    self.assertEqual(cat_attr.out_of_domain_index, 0)
    encoded_z = vtx.discrete_encode(np.array(['Z']), cat_attr)
    self.assertEqual(encoded_z[0], 0)

  def test_empty_data(self):
    attr = domain.OpenSetCategoricalAttribute(default_value='<OOD>')
    rng = np.random.default_rng(0)
    initializer = initialization.OpenSetInitializerConfig()
    data = np.array([], dtype=str)
    result = initializer.configure(attr, budget=np.inf, delta=1e-5)(rng, data)

    # Only the default value should be in the domain.
    self.assertEqual(result.categorical_attribute.possible_values, ['<OOD>'])
    self.assertEqual(result.categorical_attribute.size, 1)


class NumericalInitializerFromSummaryTest(absltest.TestCase):
  """Tests that NumericalInitializer.from_summary produces valid results."""

  def test_calibrate_sets_dp_event(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=100)
    init = initialization.NumericalInitializerConfig(
        num_partitions=4,
        max_grid_size=10001,
    ).configure(attr, budget=1.0)
    event = init.dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    # 4 partitions = 2 levels + 1 Gaussian count event = 3 events.
    self.assertLen(event.events, 3)

  def test_integer_attribute_snaps_edges(self):
    rng = np.random.default_rng(42)
    attr = domain.NumericalAttribute(min_value=0, max_value=10, dtype='int')
    init = initialization.NumericalInitializerConfig(
        num_partitions=4,
    ).configure(attr, budget=1.0)
    # Integer grid: grid_size = 11 (one bin per integer 0..10).
    counts = rng.integers(0, 30, size=init.grid_size)
    cm = init.from_summary(rng, counts)
    for edge in cm.bin_edges:
      self.assertEqual(edge, int(edge))

  def test_call_and_from_summary_produce_same_structure(self):
    """Both entry points should produce valid ColumnMeasurements."""
    rng = np.random.default_rng(42)
    attr = domain.NumericalAttribute(min_value=0.0, max_value=100.0)
    max_grid_size = 10001
    init = initialization.NumericalInitializerConfig(
        num_partitions=4,
        max_grid_size=max_grid_size,
    ).configure(attr, budget=1.0)

    data = np.linspace(0.0, 100.0, 500)
    cm_call = init(rng, data)
    self.assertIsNotNone(cm_call.bin_edges)
    self.assertIsNotNone(cm_call.noisy_counts)
    self.assertLen(cm_call.noisy_counts, cm_call.categorical_attribute.size)

    rng2 = np.random.default_rng(42)
    counts = np.random.randint(0, 100, size=max_grid_size)
    cm_summary = init.from_summary(rng2, counts)
    self.assertIsNotNone(cm_summary.bin_edges)
    self.assertIsNotNone(cm_summary.noisy_counts)
    self.assertLen(
        cm_summary.noisy_counts, cm_summary.categorical_attribute.size
    )

  def test_open_set_public_possible_values_retained(self):
    attr = domain.OpenSetCategoricalAttribute(
        default_value='<OOD>', public_possible_values=('PUB1', 'PUB2')
    )
    # 'a' has 100 occurrences (above threshold), 'b' has 1 (below min_count).
    # PUB1 and PUB2 have 0 occurrences but should still appear.
    data = np.array(['a'] * 100 + ['b'] * 1)
    init = initialization.OpenSetInitializerConfig(min_count=5).configure(
        attr, budget=1.0, delta=1e-6
    )
    col_meas = init(np.random.default_rng(0), data)
    cat_attr = col_meas.categorical_attribute
    self.assertIsInstance(cat_attr, domain.CategoricalAttribute)
    self.assertIn('<OOD>', cat_attr.possible_values)
    self.assertIn('PUB1', cat_attr.possible_values)
    self.assertIn('PUB2', cat_attr.possible_values)
    self.assertIn('a', cat_attr.possible_values)
    self.assertNotIn('b', cat_attr.possible_values)

  def test_open_set_heterogeneous_input_with_none(self):
    attr = domain.OpenSetCategoricalAttribute()
    # Heterogeneous array containing None, strings, integers, and floats.
    data = np.array(
        ['a'] * 100 + [None] * 50 + [123] * 100 + [np.nan] * 10, dtype=object
    )
    init = initialization.OpenSetInitializerConfig(min_count=5).configure(
        attr, budget=1.0, delta=1e-6
    )
    col_meas = init(np.random.default_rng(0), data)
    cat_attr = col_meas.categorical_attribute
    self.assertIsInstance(cat_attr, domain.CategoricalAttribute)
    self.assertIn('a', cat_attr.possible_values)
    self.assertIn('123', cat_attr.possible_values)

  def test_categorical_heterogeneous_input_with_none(self):
    attr = domain.CategoricalAttribute(possible_values=['a', 'b', 'c'])
    data = np.array(['a', None, 'b', np.nan, 999, 'c'], dtype=object)
    init = initialization.CategoricalInitializerConfig().configure(
        attr, budget=1.0
    )
    col_meas = init(np.random.default_rng(0), data)
    self.assertIsNotNone(col_meas.noisy_counts)
    self.assertLen(col_meas.noisy_counts, 3)

  def test_numerical_heterogeneous_input_with_none(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=100)
    data = np.array([10.5, None, 50.0, np.nan, '75.0'], dtype=object)
    init = initialization.NumericalInitializerConfig(
        num_partitions=4
    ).configure(attr, budget=1.0)
    col_meas = init(np.random.default_rng(0), data)
    self.assertIsNotNone(col_meas.bin_edges)


if __name__ == '__main__':
  absltest.main()
