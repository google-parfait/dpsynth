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

"""Tests for differentially private primitives."""

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth.local_mode import primitives
import numpy as np


class ExponentialMechanismTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_basic_selection(self):
    scores = np.array([5.0, 20.0, -10.0, 3.0])
    idx = primitives.exponential_mechanism(
        self.rng, scores, epsilon=1.0, sensitivity=1.0
    )
    self.assertIn(idx, [0, 1, 2, 3])

  def test_infinite_budget_selects_max_score(self):
    scores = np.array([5.0, 20.0, -10.0, 3.0])
    idx = primitives.exponential_mechanism(
        self.rng, scores, epsilon=np.inf, sensitivity=1.0
    )
    self.assertEqual(idx, 1)

  def test_high_budget_selects_max_score(self):
    scores = np.array([5.0, 20.0, -10.0, 3.0])
    idx = primitives.exponential_mechanism(
        self.rng, scores, epsilon=100.0, sensitivity=1.0
    )
    self.assertEqual(idx, 1)

  def test_zero_budget_uniform_distribution(self):
    scores = np.array([100.0, 0.0, -100.0])
    selections = [
        primitives.exponential_mechanism(
            self.rng, scores, epsilon=0.0, sensitivity=1.0
        )
        for _ in range(1500)
    ]
    counts = np.bincount(selections, minlength=3)
    for c in counts:
      self.assertBetween(c, 400, 600)

  def test_monotonic_scaling(self):
    scores = np.array([1.0, 2.0])
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    idx1 = primitives.exponential_mechanism(
        rng1, scores, epsilon=1.0, sensitivity=1.0, monotonic=True
    )
    idx2 = primitives.exponential_mechanism(
        rng2, scores, epsilon=2.0, sensitivity=1.0, monotonic=False
    )
    self.assertEqual(idx1, idx2)

  def test_invalid_inputs_raise(self):
    scores = np.array([1.0, 2.0])
    with self.assertRaises(ValueError):
      primitives.exponential_mechanism(
          self.rng, scores, epsilon=-1.0, sensitivity=1.0
      )
    with self.assertRaises(ValueError):
      primitives.exponential_mechanism(
          self.rng, scores, epsilon=1.0, sensitivity=0.0
      )
    with self.assertRaises(ValueError):
      primitives.exponential_mechanism(
          self.rng, scores, epsilon=1.0, sensitivity=-0.5
      )
    with self.assertRaises(ValueError):
      primitives.exponential_mechanism(
          self.rng, np.array([]), epsilon=1.0, sensitivity=1.0
      )
    with self.assertRaises(TypeError):
      primitives.exponential_mechanism(self.rng, scores, epsilon=1.0)  # pyrefly: ignore[missing-argument]


class JitterFactorTest(absltest.TestCase):

  def test_jitter_factor_calculation(self):
    self.assertEqual(primitives.jitter_factor(0), 1)
    self.assertEqual(primitives.jitter_factor(1), 4)
    self.assertEqual(primitives.jitter_factor(16), 64)


class QuantilesFromHistogramTest(parameterized.TestCase):

  def test_no_levels_returns_empty(self):
    rng = np.random.default_rng(0)
    counts = np.array([10])
    for jitter_strategy in ("symmetric", "refine"):
      edges = primitives.quantiles_from_histogram(
          rng, counts, np.array([]), jitter_strategy
      )
      self.assertEmpty(edges)

  @parameterized.product(
      levels=(1, 2, 3, 4),
      jitter_strategy=("symmetric", "refine"),
  )
  def test_edge_count_matches_levels(self, levels, jitter_strategy):
    rng = np.random.default_rng(0)
    grid_size = 10001
    counts = rng.integers(0, 20, size=grid_size)
    edges = primitives.quantiles_from_histogram(
        rng,
        counts,
        epsilon_levels=np.ones(levels),
        jitter_strategy=jitter_strategy,
    )
    self.assertLen(edges, 2**levels - 1)

  @parameterized.parameters(1, 2, 3, 4)
  def test_edge_count_matches_levels_with_spike(self, levels):
    counts = np.zeros(101, dtype=np.int64)
    counts[:40] = 1
    counts[40] = 1000
    counts[41:80] = 1
    edges = primitives.quantiles_from_histogram(
        np.random.default_rng(0),
        counts,
        epsilon_levels=np.array([np.inf] * levels),
        jitter_strategy="refine",
    )
    self.assertLen(edges, 2**levels - 1)

  def test_integer_edges_are_integer_indices(self):
    counts = np.zeros(101, dtype=np.int64)
    counts[40] = 5000
    counts[:40] = 50
    counts[41:] = 50
    edges = primitives.quantiles_from_histogram(
        np.random.default_rng(0),
        counts,
        epsilon_levels=np.array([np.inf] * 3),
        jitter_strategy="refine",
    )
    for edge in edges:
      self.assertEqual(edge, int(edge))
      self.assertBetween(edge, 0, counts.size - 1)

  def test_exact_budget_matches_numpy_smooth(self):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 100, size=50000)
    counts = np.bincount(data, minlength=101)
    edges = primitives.quantiles_from_histogram(
        rng,
        counts,
        epsilon_levels=np.array([np.inf] * 3),
        jitter_strategy="refine",
    )
    expected = np.quantile(data, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
    np.testing.assert_allclose(edges, expected, atol=1.0)

  def test_exact_budget_matches_numpy_with_spike(self):
    below = np.arange(1, 40).repeat(230)
    spike = np.full(13500, 40)
    above = np.arange(41, 80).repeat(190)
    data = np.concatenate([below, spike, above])
    counts = np.bincount(data, minlength=101)
    edges = primitives.quantiles_from_histogram(
        np.random.default_rng(0),
        counts,
        epsilon_levels=np.array([np.inf] * 3),
        jitter_strategy="refine",
    )
    expected = np.quantile(data, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
    np.testing.assert_allclose(edges, expected, atol=1.0)

  def test_spike_owns_consecutive_edges(self):
    counts = np.zeros(101, dtype=np.int64)
    counts[:40] = 20
    counts[40] = 20000  # ~96% of the mass.
    counts[41:80] = 20
    edges = primitives.quantiles_from_histogram(
        np.random.default_rng(0),
        counts,
        epsilon_levels=np.array([np.inf] * 3),
        jitter_strategy="refine",
    )
    self.assertEqual(edges[1:6], [40, 40, 40, 40, 40])


class GaussianThresholdingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_basic_operation(self):
    counts = np.array([50, 5])
    selected, estimated_counts = primitives.gaussian_thresholding(
        self.rng,
        counts,
        sigma=0.3,
        delta=1e-5,
        l2_sensitivity=1.0,
        linf_sensitivity=1.0,
    )
    self.assertIn(0, selected)
    self.assertEqual(selected.size, estimated_counts.size)

  def test_empty_counts(self):
    counts = np.array([], dtype=int)
    selected, estimated_counts = primitives.gaussian_thresholding(
        self.rng,
        counts,
        sigma=1.0,
        delta=1e-5,
        l2_sensitivity=1.0,
        linf_sensitivity=1.0,
    )
    self.assertEmpty(selected)
    self.assertEmpty(estimated_counts)

  def test_zero_sigma_selects_all_eligible(self):
    counts = np.array([1, 1, 1, 1, 1])
    selected, _ = primitives.gaussian_thresholding(
        self.rng,
        counts,
        sigma=0.0,
        delta=0.1,
        l2_sensitivity=1.0,
        linf_sensitivity=1.0,
    )
    self.assertCountEqual(selected, [0, 1, 2, 3, 4])

  def test_rare_items_not_selected(self):
    counts = np.array([100, 1])
    selected, _ = primitives.gaussian_thresholding(
        self.rng,
        counts,
        sigma=1.5,
        delta=1e-6,
        l2_sensitivity=1.0,
        linf_sensitivity=1.0,
    )
    self.assertIn(0, selected)
    self.assertNotIn(1, selected)

  def test_min_count_filters_low_count_partitions(self):
    counts = np.array([50, 3])
    selected, _ = primitives.gaussian_thresholding(
        self.rng,
        counts,
        sigma=0.3,
        delta=1e-5,
        l2_sensitivity=1.0,
        linf_sensitivity=1.0,
        min_count=5,
    )
    self.assertIn(0, selected)
    self.assertNotIn(1, selected)

  def test_min_count_one_matches_default(self):
    counts = np.array([50, 5])
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    result1 = primitives.gaussian_thresholding(
        rng1,
        counts,
        sigma=0.3,
        delta=1e-5,
        l2_sensitivity=1.0,
        linf_sensitivity=1.0,
    )
    result2 = primitives.gaussian_thresholding(
        rng2,
        counts,
        sigma=0.3,
        delta=1e-5,
        l2_sensitivity=1.0,
        linf_sensitivity=1.0,
        min_count=1,
    )
    np.testing.assert_array_equal(result1[0], result2[0])
    np.testing.assert_array_equal(result1[1], result2[1])

  def test_min_count_all_filtered_returns_empty(self):
    counts = np.array([1, 1, 1])
    selected, estimated_counts = primitives.gaussian_thresholding(
        self.rng,
        counts,
        sigma=0.3,
        delta=1e-5,
        l2_sensitivity=1.0,
        linf_sensitivity=1.0,
        min_count=5,
    )
    self.assertEmpty(selected)
    self.assertEmpty(estimated_counts)

  def test_linf_sensitivity_shifts_threshold(self):
    counts = np.array([3, 10])
    selected, _ = primitives.gaussian_thresholding(
        self.rng,
        counts,
        sigma=0.0,
        delta=0.1,
        l2_sensitivity=1.0,
        linf_sensitivity=5.0,
    )
    self.assertCountEqual(selected, [1])

  def test_min_count_increases_threshold(self):
    counts = np.array([10, 10])
    selected, _ = primitives.gaussian_thresholding(
        self.rng,
        counts,
        sigma=0.0,
        delta=0.1,
        l2_sensitivity=1.0,
        linf_sensitivity=1.0,
        min_count=10,
    )
    self.assertCountEqual(selected, [0, 1])

  def test_invalid_inputs_raise(self):
    counts = np.array([1, 2, 3])
    with self.assertRaises(ValueError):
      primitives.gaussian_thresholding(
          self.rng,
          counts,
          sigma=-1.0,
          delta=1e-5,
          l2_sensitivity=1.0,
          linf_sensitivity=1.0,
      )
    with self.assertRaises(ValueError):
      primitives.gaussian_thresholding(
          self.rng,
          counts,
          sigma=1.0,
          delta=0.0,
          l2_sensitivity=1.0,
          linf_sensitivity=1.0,
      )
    with self.assertRaises(ValueError):
      primitives.gaussian_thresholding(
          self.rng,
          counts,
          sigma=1.0,
          delta=1e-5,
          l2_sensitivity=0.0,
          linf_sensitivity=1.0,
      )
    with self.assertRaises(ValueError):
      primitives.gaussian_thresholding(
          self.rng,
          counts,
          sigma=1.0,
          delta=1e-5,
          l2_sensitivity=1.0,
          linf_sensitivity=0.0,
      )
    with self.assertRaises(ValueError):
      primitives.gaussian_thresholding(
          self.rng,
          counts,
          sigma=1.0,
          delta=1e-5,
          l2_sensitivity=1.0,
          linf_sensitivity=1.0,
          min_count=0,
      )
    with self.assertRaises(TypeError):
      primitives.gaussian_thresholding(
          self.rng, counts, sigma=1.0, delta=1e-5
      )  # pyrefly: ignore[missing-argument]


class EnsurePublicPartitionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_missing_partitions_appended_and_sorted(self):
    selected = np.array(["a", "c"])
    counts = np.array([10.0, 20.0])
    public = np.array(["b", "c"])
    sel, cts = primitives.ensure_public_partitions(
        self.rng, selected, counts, 0.0, public
    )
    np.testing.assert_array_equal(sel, ["a", "b", "c"])
    self.assertEqual(cts[0], 10.0)
    self.assertEqual(cts[1], 0.0)
    self.assertEqual(cts[2], 20.0)

  def test_all_present_is_noop(self):
    selected = np.array(["a", "b"])
    counts = np.array([10.0, 20.0])
    public = np.array(["a", "b"])
    sel, cts = primitives.ensure_public_partitions(
        self.rng, selected, counts, 1.0, public
    )
    np.testing.assert_array_equal(sel, selected)
    np.testing.assert_array_equal(cts, counts)

  def test_empty_selected(self):
    selected = np.array([], dtype=str)
    counts = np.array([], dtype=float)
    public = np.array(["y", "x"])
    sel, cts = primitives.ensure_public_partitions(
        self.rng, selected, counts, 1.0, public
    )
    np.testing.assert_array_equal(sel, ["x", "y"])
    self.assertLen(cts, 2)


class GaussianMechanismTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_scalar(self):
    noisy = primitives.gaussian_mechanism(
        self.rng, 100, sigma=1.0, l2_sensitivity=1.0
    )
    self.assertIsInstance(noisy, float)
    self.assertAlmostEqual(noisy, 100.0, delta=5.0)

  def test_1d_array(self):
    counts = np.array([10, 20, 30])
    noisy = primitives.gaussian_mechanism(
        self.rng, counts, sigma=1.0, l2_sensitivity=1.0
    )
    self.assertEqual(noisy.shape, (3,))
    np.testing.assert_allclose(noisy, counts, atol=5.0)

  def test_2d_array(self):
    counts = np.ones((2, 2)) * 10
    noisy = primitives.gaussian_mechanism(
        self.rng, counts, sigma=1.0, l2_sensitivity=1.0
    )
    self.assertEqual(noisy.shape, (2, 2))
    np.testing.assert_allclose(noisy, counts, atol=5.0)

  def test_l2_sensitivity_scales_noise(self):
    k = 4
    counts = np.array([10.0, 20.0, 30.0])
    base_rng = np.random.default_rng(0)
    base_noise = (
        primitives.gaussian_mechanism(
            base_rng, counts, sigma=1.0, l2_sensitivity=1.0
        )
        - counts
    )

    scaled_rng = np.random.default_rng(0)
    scaled_noise = (
        primitives.gaussian_mechanism(
            scaled_rng, counts, sigma=1.0, l2_sensitivity=k
        )
        - counts
    )

    np.testing.assert_allclose(scaled_noise, k * base_noise)

  def test_invalid_inputs_raise(self):
    with self.assertRaises(ValueError):
      primitives.gaussian_mechanism(
          self.rng, 10, sigma=-1.0, l2_sensitivity=1.0
      )
    with self.assertRaises(ValueError):
      primitives.gaussian_mechanism(self.rng, 10, sigma=1.0, l2_sensitivity=0.0)
    with self.assertRaises(TypeError):
      primitives.gaussian_mechanism(self.rng, 10, sigma=1.0)  # pyrefly: ignore[missing-argument]


if __name__ == "__main__":
  absltest.main()
