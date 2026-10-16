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
from absl.testing import parameterized
from dpsynth import domain
from dpsynth.local_mode import vectorized_transformations
import numpy as np


class DiscreteEncodeTest(absltest.TestCase):

  def test_basic_encoding(self):
    attr = domain.CategoricalAttribute(['None', 'a', 'b', 'c'])
    data = np.array(['None', 'a', 'b', 'c', 'a'])
    result = vectorized_transformations.discrete_encode(data, attr)
    np.testing.assert_array_equal(result, [0, 1, 2, 3, 1])

  def test_out_of_domain_maps_to_default(self):
    attr = domain.CategoricalAttribute(
        ['None', 'a', 'b'], out_of_domain_index=0
    )
    data = np.array(['a', 'x', 'b', 'z'])
    result = vectorized_transformations.discrete_encode(data, attr)
    np.testing.assert_array_equal(result, [1, 0, 2, 0])

  def test_integer_possible_values(self):
    attr = domain.CategoricalAttribute([10, 20, 30])
    data = np.array([20, 30, 10, 10])
    result = vectorized_transformations.discrete_encode(data, attr)
    np.testing.assert_array_equal(result, [1, 2, 0, 0])

  def test_empty_data(self):
    attr = domain.CategoricalAttribute(['a', 'b'])
    data = np.array([], dtype=object)
    result = vectorized_transformations.discrete_encode(data, attr)
    self.assertEqual(result.shape, (0,))


class DiscreteDecodeTest(absltest.TestCase):

  def test_basic_decoding(self):
    attr = domain.CategoricalAttribute(['<OOD>', 'a', 'b', 'c'])
    encoded = np.array([0, 1, 2, 3, 1])
    result = vectorized_transformations.discrete_decode(encoded, attr)
    expected = np.array(['<OOD>', 'a', 'b', 'c', 'a'], dtype=object)
    np.testing.assert_array_equal(result, expected)

  def test_roundtrip(self):
    attr = domain.CategoricalAttribute(['x', 'y', 'z'])
    data = np.array(['x', 'y', 'z', 'y', 'x'], dtype=object)
    encoded = vectorized_transformations.discrete_encode(data, attr)
    decoded = vectorized_transformations.discrete_decode(encoded, attr)
    np.testing.assert_array_equal(decoded, data)

  def test_empty(self):
    attr = domain.CategoricalAttribute(['a'])
    result = vectorized_transformations.discrete_decode(
        np.array([], dtype=int), attr
    )
    self.assertEqual(result.shape, (0,))


class DiscretizeTest(parameterized.TestCase):

  def test_clip_to_range_basic(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=True
    )
    data = np.array([1.0, 5.0, 5.00001, 8.0, -1.0, 11.0])
    result = vectorized_transformations.discretize(data, np.array([5.0]), attr)
    # (exclusive_min, 5] -> bin 0, (5, 10] -> bin 1
    np.testing.assert_array_equal(result, [0, 0, 1, 1, 0, 1])

  def test_no_clip_to_range_ood(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=False
    )
    data = np.array([5.0, 8.0, -1.0, 11.0, np.nan])
    result = vectorized_transformations.discretize(data, np.array([5.0]), attr)
    # OOD -> 0, (exclusive_min, 5] -> 1, (5, 10] -> 2
    np.testing.assert_array_equal(result, [1, 2, 0, 0, 0])

  def test_multiple_bins(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=True
    )
    data = np.array([10.0, 25.0, 50.0, 75.0, 90.0])
    result = vectorized_transformations.discretize(
        data, np.array([25.0, 50.0, 75.0]), attr
    )
    np.testing.assert_array_equal(result, [0, 0, 1, 2, 3])

  def test_integer_attribute(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10, dtype='int', clip_to_range=True
    )
    data = np.array([0.0, 5.0, 6.0, 10.0])
    result = vectorized_transformations.discretize(data, np.array([5.0]), attr)
    # For int dtype with bin_edge 5: (-1, 5] -> 0, (5, 10] -> 1
    np.testing.assert_array_equal(result, [0, 0, 1, 1])

  def test_empty_data(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=True
    )
    data = np.array([], dtype=float)
    result = vectorized_transformations.discretize(data, np.array([5.0]), attr)
    self.assertEqual(result.shape, (0,))

  def test_invalid_bin_edges_raises(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=10)
    # Empty edges = 1-bin case: all data maps to bin 0.
    result = vectorized_transformations.discretize(
        np.array([1.0, 5.0]), np.array([]), attr
    )
    np.testing.assert_array_equal(result, [0, 0])
    with self.assertRaises(ValueError):
      vectorized_transformations.discretize(
          np.array([1.0]), np.array([-1.0, 5.0]), attr
      )
    with self.assertRaises(ValueError):
      vectorized_transformations.discretize(
          np.array([1.0]), np.array([5.0, 3.0]), attr
      )


class UndiscretizeTest(parameterized.TestCase):

  def test_clip_to_range_midpoints(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=True
    )
    bin_indices = np.array([0, 1])
    result = vectorized_transformations.undiscretize(
        bin_indices, np.array([5.0]), attr
    )
    # Full edges: [exclusive_min, 5, 10]. Midpoints of intervals.
    self.assertEqual(result.shape, (2,))
    self.assertBetween(result[0], 0, 5)
    self.assertBetween(result[1], 5, 10)

  def test_no_clip_to_range_ood_nan(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=False
    )
    bin_indices = np.array([0, 1, 2])
    result = vectorized_transformations.undiscretize(
        bin_indices, np.array([5.0]), attr
    )
    self.assertTrue(np.isnan(result[0]))
    self.assertBetween(result[1], 0, 5)
    self.assertBetween(result[2], 5, 10)

  def test_integer_dtype_ceils(self):
    attr = domain.NumericalAttribute(
        min_value=1, max_value=5, dtype='int', clip_to_range=True
    )
    bin_indices = np.array([0, 1])
    result = vectorized_transformations.undiscretize(
        bin_indices, np.array([3.0]), attr
    )
    # All results should be integers (ceiled).
    for v in result:
      self.assertEqual(v, int(v))

  def test_roundtrip_clip(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=True
    )
    edges = np.array([25.0, 50.0, 75.0])
    data = np.array([10.0, 40.0, 60.0, 90.0])
    indices = vectorized_transformations.discretize(data, edges, attr)
    midpoints = vectorized_transformations.undiscretize(indices, edges, attr)
    # Each midpoint should be within the correct bin.
    full_edges = np.r_[attr.exclusive_min_value, edges, attr.max_value]
    for i, idx in enumerate(indices):
      self.assertBetween(midpoints[i], full_edges[idx], full_edges[idx + 1])

  def test_empty(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=True
    )
    result = vectorized_transformations.undiscretize(
        np.array([], dtype=int), np.array([5.0]), attr
    )
    self.assertEqual(result.shape, (0,))

  def test_sample_mode_values_in_bins(self):
    rng = np.random.default_rng(0)
    attr = domain.NumericalAttribute(
        min_value=0,
        max_value=10,
        clip_to_range=True,
        interval_handling='sample',
    )
    bin_indices = np.array([0, 1])
    edges = np.array([5.0])
    result = vectorized_transformations.undiscretize(
        bin_indices, edges, attr, rng=rng
    )
    full = np.r_[attr.exclusive_min_value, edges, attr.max_value]
    self.assertBetween(result[0], full[0], full[1])
    self.assertBetween(result[1], full[1], full[2])

  def test_sample_mode_ood_nan(self):
    rng = np.random.default_rng(0)
    attr = domain.NumericalAttribute(
        min_value=0,
        max_value=10,
        clip_to_range=False,
        interval_handling='sample',
    )
    result = vectorized_transformations.undiscretize(
        np.array([0, 1]), np.array([5.0]), attr, rng=rng
    )
    self.assertTrue(np.isnan(result[0]))

  def test_interval_mode(self):
    attr = domain.NumericalAttribute(
        min_value=0,
        max_value=10,
        clip_to_range=True,
        interval_handling='interval',
    )
    bin_indices = np.array([0, 1])
    result = vectorized_transformations.undiscretize(
        bin_indices, np.array([5.0]), attr
    )
    self.assertIn('[', result[0])  # First interval is closed: [min, edge]
    self.assertIn(']', result[0])

  def test_string_sentinel_interval(self):
    attr = domain.NumericalAttribute(
        min_value=0,
        max_value=10,
        clip_to_range=False,
        sentinel='MISSING',
        interval_handling='interval',
    )
    result = vectorized_transformations.undiscretize(
        np.array([0, 1]), np.array([5.0]), attr
    )
    self.assertEqual(result[0], 'MISSING')
    self.assertIn('[', result[1])  # First in-domain interval is closed

  def test_invalid_bin_edges_raises(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10, interval_handling='midpoint'
    )
    # Empty edges = 1-bin case: undiscretize returns domain midpoint.
    result = vectorized_transformations.undiscretize(
        np.array([0, 0]), np.array([]), attr
    )
    np.testing.assert_array_equal(result, [5.0, 5.0])
    with self.assertRaises(ValueError):
      vectorized_transformations.undiscretize(
          np.array([1]), np.array([-1.0, 5.0]), attr
      )
    with self.assertRaises(ValueError):
      vectorized_transformations.undiscretize(
          np.array([1]), np.array([5.0, 3.0]), attr
      )

  def test_custom_sentinel_midpoint(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=False, sentinel=-1
    )
    result = vectorized_transformations.undiscretize(
        np.array([0, 1, 2]), np.array([5.0]), attr
    )
    self.assertEqual(result[0], -1)
    self.assertBetween(result[1], 0, 5)
    self.assertBetween(result[2], 5, 10)

  def test_custom_sentinel_sample(self):
    rng = np.random.default_rng(0)
    attr = domain.NumericalAttribute(
        min_value=0,
        max_value=10,
        clip_to_range=False,
        sentinel=-1,
        interval_handling='sample',
    )
    result = vectorized_transformations.undiscretize(
        np.array([0, 1]), np.array([5.0]), attr, rng=rng
    )
    self.assertEqual(result[0], -1)

  @parameterized.parameters(('midpoint', None), ('sample', -1))
  def test_integer_dtype_no_clip(self, interval_handling, sentinel):
    rng = np.random.default_rng(0)
    attr = domain.NumericalAttribute(
        min_value=0,
        max_value=3,
        dtype='int',
        clip_to_range=False,
        sentinel=sentinel,
        interval_handling=interval_handling,
    )
    result = vectorized_transformations.undiscretize(
        np.array([0, 1, 2, 3, 4]), np.array([0.0, 1.0, 2.0]), attr, rng=rng
    )
    np.testing.assert_array_equal(result[1:], [0, 1, 2, 3])
    if sentinel is None:
      self.assertTrue(np.isnan(result[0]))
    else:
      self.assertTrue(np.issubdtype(result.dtype, np.integer))
      self.assertEqual(result[0], sentinel)


class MergeRareValuesTest(absltest.TestCase):

  def test_some_rare_values(self):
    rare_mask = np.array([True, False, True, False])
    data = np.array([0, 1, 2, 3])
    size, compressed = vectorized_transformations.merge_rare_values(
        data, rare_mask
    )
    self.assertEqual(size, 3)
    np.testing.assert_array_equal(compressed, [2, 0, 2, 1])

  def test_no_rare_values(self):
    rare_mask = np.array([False, False, False, False])
    data = np.array([0, 1, 2, 3])
    size, compressed = vectorized_transformations.merge_rare_values(
        data, rare_mask
    )
    self.assertEqual(size, 4)
    np.testing.assert_array_equal(compressed, [0, 1, 2, 3])

  def test_all_rare_values(self):
    rare_mask = np.array([True, True, True, True])
    data = np.array([0, 1, 2, 3])
    size, compressed = vectorized_transformations.merge_rare_values(
        data, rare_mask
    )
    self.assertEqual(size, 1)
    np.testing.assert_array_equal(compressed, [0, 0, 0, 0])

  def test_empty_data(self):
    rare_mask = np.array([True, False])
    data = np.array([], dtype=int)
    size, compressed = vectorized_transformations.merge_rare_values(
        data, rare_mask
    )
    self.assertEqual(size, 2)
    self.assertEqual(compressed.shape, (0,))


class UnmergeRareValuesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_roundtrip_some_rare(self):
    rare_mask = np.array([True, False, True, False])
    data = np.array([0, 1, 2, 3, 1, 0])
    _, compressed = vectorized_transformations.merge_rare_values(
        data, rare_mask
    )
    unmerged = vectorized_transformations.unmerge_rare_values(
        compressed, rare_mask, self.rng
    )
    # Common values should round-trip exactly.
    for i, val in enumerate(data):
      if not rare_mask[val]:
        self.assertEqual(unmerged[i], val)
      else:
        # Rare values should map to one of the original rare indices.
        self.assertIn(unmerged[i], [0, 2])

  def test_roundtrip_no_rare(self):
    rare_mask = np.array([False, False, False])
    data = np.array([0, 1, 2, 1])
    _, compressed = vectorized_transformations.merge_rare_values(
        data, rare_mask
    )
    unmerged = vectorized_transformations.unmerge_rare_values(
        compressed, rare_mask, self.rng
    )
    np.testing.assert_array_equal(unmerged, data)

  def test_roundtrip_all_rare(self):
    rare_mask = np.array([True, True, True])
    data = np.array([0, 1, 2, 0])
    _, compressed = vectorized_transformations.merge_rare_values(
        data, rare_mask
    )
    unmerged = vectorized_transformations.unmerge_rare_values(
        compressed, rare_mask, self.rng
    )
    for v in unmerged:
      self.assertIn(v, [0, 1, 2])

  def test_empty_data(self):
    rare_mask = np.array([True, False])
    result = vectorized_transformations.unmerge_rare_values(
        np.array([], dtype=int), rare_mask, self.rng
    )
    self.assertEqual(result.shape, (0,))


if __name__ == '__main__':
  absltest.main()
