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
import dpsynth
from dpsynth import transformations
import numpy as np
import pandas as pd


class TestDataTransformations(absltest.TestCase):

  def test_compose_and_inverse(self):
    transformer = transformations.DataTransformation(
        lambda x: x + 1, lambda x: x - 1
    )
    self.assertEqual(transformer(1), 2)
    self.assertEqual(transformer.inverse(2), 1)
    transformer3 = transformer @ transformer @ transformer
    self.assertEqual(transformer3(1), 4)
    self.assertEqual(transformer3.inverse(4), 1)

  def test_discrete_encoder_transform(self):
    attribute = dpsynth.CategoricalAttribute(
        possible_values=['<OOD>', 'a', 'b', '(0, 1]', 'False', '314159']
    )
    discrete_encoder = transformations.discrete_encoder(attribute)
    self.assertEqual(discrete_encoder('<OOD>'), 0)
    self.assertEqual(discrete_encoder('a'), 1)
    self.assertEqual(discrete_encoder('b'), 2)
    self.assertEqual(discrete_encoder('(0, 1]'), 3)
    self.assertEqual(discrete_encoder('False'), 4)
    self.assertEqual(discrete_encoder('314159'), 5)
    self.assertEqual(discrete_encoder('c'), 0)
    self.assertEqual(discrete_encoder('True'), 0)

  def test_discrete_encoder_inverse(self):
    attribute = dpsynth.CategoricalAttribute(
        possible_values=['a', 'b', 'c'], out_of_domain_index=0
    )
    discrete_encoder = transformations.discrete_encoder(attribute)
    self.assertEqual(discrete_encoder.inverse(0), 'a')
    self.assertEqual(discrete_encoder.inverse(1), 'b')
    self.assertEqual(discrete_encoder.inverse(2), 'c')
    with self.assertRaises((ValueError, KeyError)):
      discrete_encoder.inverse(4)

  def test_invalid_bin_edges_raises_error(self):
    attr = dpsynth.NumericalAttribute(min_value=0, max_value=10)
    with self.assertRaises(ValueError):
      transformations.create_discretize_transformation(attr, [-1, 2, 3])
    with self.assertRaises(ValueError):
      transformations.create_discretize_transformation(attr, [5, 11])
    with self.assertRaises(ValueError):
      transformations.create_discretize_transformation(attr, [1, 2, 3, 5, 4])

  def test_correct_mapping_clip_to_range(self):
    attr = dpsynth.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=True
    )

    categorical, transform_fn = (
        transformations.create_discretize_transformation(attr, [5])
    )
    interval1, interval2 = '[0.0, 5.0]', '(5.0, 10.0]'
    self.assertEqual(categorical.possible_values, [interval1, interval2])
    # Float values are mapped to the nearest correct interval.
    self.assertEqual(transform_fn(5), interval1)
    self.assertEqual(transform_fn(5.00001), interval2)
    self.assertEqual(transform_fn(11), interval2)
    self.assertEqual(transform_fn(8), interval2)
    self.assertEqual(transform_fn(-1), interval1)
    self.assertEqual(transform_fn(np.inf), interval2)
    self.assertEqual(transform_fn(-np.inf), interval1)
    # Non-float out-of-domain values are mapped to the first interval.
    self.assertEqual(transform_fn('A'), interval1)
    self.assertEqual(transform_fn(None), interval1)
    self.assertEqual(transform_fn(np.nan), interval1)
    self.assertEqual(transform_fn(interval2), interval1)

  def test_correct_mapping_no_clip_to_range(self):
    attr = dpsynth.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=False
    )
    categorical, transform_fn = (
        transformations.create_discretize_transformation(attr, [5])
    )
    interval1, interval2 = '[0.0, 5.0]', '(5.0, 10.0]'
    self.assertEqual(
        categorical.possible_values, ['<OOD>', interval1, interval2]
    )
    # Float values are mapped to the correct interval if they are in bounds.
    self.assertEqual(transform_fn(0), interval1)
    self.assertEqual(transform_fn(5), interval1)
    self.assertEqual(transform_fn(5.00001), interval2)
    self.assertEqual(transform_fn(8), interval2)
    self.assertEqual(transform_fn(10), interval2)
    # Float values are mapped to OOD if they are out-of-bounds.
    self.assertEqual(transform_fn(10.001), '<OOD>')
    self.assertEqual(transform_fn(-0.001), '<OOD>')
    self.assertEqual(transform_fn(np.inf), '<OOD>')
    self.assertEqual(transform_fn(-np.inf), '<OOD>')
    # Non-float out-of-domain values are mapped to OOD as well.
    self.assertEqual(transform_fn('A'), '<OOD>')
    self.assertEqual(transform_fn(None), '<OOD>')
    self.assertEqual(transform_fn(np.nan), '<OOD>')
    self.assertEqual(transform_fn(interval2), '<OOD>')

  def test_valid_discretization_clip_to_range_inverse(self):
    attr = dpsynth.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=True
    )

    categorical, transform_fn = (
        transformations.create_discretize_transformation(attr, [5])
    )
    interval1, interval2 = '[0.0, 5.0]', '(5.0, 10.0]'
    self.assertEqual(categorical.possible_values, [interval1, interval2])
    self.assertBetween(transform_fn.inverse(interval1), 0, 5)
    self.assertBetween(transform_fn.inverse(interval2), 5, 10)
    # We make no guarantees about the behavior of inverse when the input is not
    # a valid interval in the domain.  The function could fail with an
    # exception, or silently return a value without failing.

  def test_integer_discretization_invertible(self):
    attr = dpsynth.NumericalAttribute(0, 10, dtype='int')
    _, fn = transformations.create_uniform_discretize_transformation(attr, 5)
    # Non-boundary values round-trip exactly via midpoint inversion.
    for i in [1, 3, 5, 7, 9]:
      self.assertEqual(fn.inverse(fn(i)), i)

  def test_valid_discretization_no_clip_to_range_inverse(self):
    attr = dpsynth.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=False
    )
    categorical, transform_fn = (
        transformations.create_discretize_transformation(attr, [5])
    )
    interval1, interval2 = '[0.0, 5.0]', '(5.0, 10.0]'
    self.assertEqual(
        categorical.possible_values, ['<OOD>', interval1, interval2]
    )

    self.assertBetween(transform_fn.inverse(interval1), 0, 5)
    self.assertBetween(transform_fn.inverse(interval2), 5, 10)
    self.assertTrue(np.isnan(transform_fn.inverse('<OOD>')))

  def test_discretize_inverse_sentinel_default_nan(self):
    attr = dpsynth.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=False
    )
    _, transform_fn = transformations.create_discretize_transformation(
        attr, [5]
    )
    self.assertTrue(np.isnan(transform_fn.inverse('<OOD>')))

  def test_discretize_inverse_custom_sentinel(self):
    attr = dpsynth.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=False, sentinel=-1
    )
    _, transform_fn = transformations.create_discretize_transformation(
        attr, [5]
    )
    self.assertEqual(transform_fn.inverse('<OOD>'), -1)

  def test_valid_discretization_for_int_attribute(self):
    attr = dpsynth.NumericalAttribute(min_value=0, max_value=10, dtype='int')
    categorical, transform_fn = (
        transformations.create_discretize_transformation(attr, [5])
    )
    interval1, interval2 = '[0.0, 5.0]', '(5.0, 10.0]'
    self.assertEqual(categorical.possible_values, [interval1, interval2])
    self.assertEqual(transform_fn(5), interval1)
    self.assertEqual(transform_fn(6), interval2)
    self.assertIsInstance(transform_fn.inverse(interval1), int)
    self.assertIsInstance(transform_fn.inverse(interval2), int)
    self.assertBetween(transform_fn.inverse(interval1), 0, 5)
    self.assertBetween(transform_fn.inverse(interval2), 5, 10)

  def test_discretize_interval_handling_sample(self):
    attr = dpsynth.NumericalAttribute(
        min_value=0, max_value=100, interval_handling='sample'
    )
    _, transform_fn = transformations.create_discretize_transformation(
        attr, [50]
    )
    interval = '(50.0, 100.0]'
    values = set()
    for _ in range(50):
      value = transform_fn.inverse(interval)
      self.assertBetween(value, 50, 100)
      values.add(value)
    # Sample mode should produce non-constant output (unlike midpoint).
    self.assertGreater(len(values), 1)
    self.assertTrue(np.isnan(transform_fn.inverse('<OOD>')))

  def test_discretize_interval_handling_interval(self):
    attr = dpsynth.NumericalAttribute(
        min_value=0, max_value=10, interval_handling='interval'
    )
    _, transform_fn = transformations.create_discretize_transformation(
        attr, [5]
    )
    interval = '(5.0, 10.0]'
    self.assertEqual(transform_fn.inverse(interval), interval)
    self.assertEqual(transform_fn.inverse('<OOD>'), '')

  def test_discretize_reverse_semi_infinite_intervals(self):
    # Midpoint mode: semi-infinite intervals should return the finite endpoint.
    attr = dpsynth.NumericalAttribute(min_value=0, max_value=10)
    _, transform_fn = transformations.create_discretize_transformation(
        attr, [5]
    )
    self.assertBetween(transform_fn.inverse('(5.0, 10.0]'), 5, 10)
    self.assertBetween(transform_fn.inverse('[0.0, 5.0]'), 0, 5)
    # Sample mode: semi-infinite intervals should also return the finite end.
    attr_sample = dpsynth.NumericalAttribute(
        min_value=0, max_value=10, interval_handling='sample'
    )
    _, transform_fn_sample = transformations.create_discretize_transformation(
        attr_sample, [5]
    )
    self.assertBetween(transform_fn_sample.inverse('(5.0, 10.0]'), 5, 10)
    self.assertBetween(transform_fn_sample.inverse('[0.0, 5.0]'), 0, 5)

  def test_rare_value_merging_some_rare_values(self):
    rare_mask = np.array([True, False, True, False])
    size, transform_fn = (
        transformations.create_rare_value_merging_transformation(rare_mask)
    )
    self.assertEqual(size, 3)
    self.assertEqual(transform_fn(0), 2)
    self.assertEqual(transform_fn(1), 0)
    self.assertEqual(transform_fn(2), 2)
    self.assertEqual(transform_fn(3), 1)

    self.assertIn(transform_fn.inverse(2), [0, 2])
    self.assertEqual(transform_fn.inverse(0), 1)
    self.assertEqual(transform_fn.inverse(1), 3)

  def test_no_rare_values(self):
    rare_mask = np.array([False, False, False, False])
    size, transform_fn = (
        transformations.create_rare_value_merging_transformation(rare_mask)
    )
    self.assertEqual(size, 4)
    self.assertEqual(transform_fn(0), 0)
    self.assertEqual(transform_fn(1), 1)
    self.assertEqual(transform_fn(2), 2)
    self.assertEqual(transform_fn(3), 3)
    self.assertEqual(transform_fn.inverse(0), 0)
    self.assertEqual(transform_fn.inverse(1), 1)
    self.assertEqual(transform_fn.inverse(2), 2)
    self.assertEqual(transform_fn.inverse(3), 3)

  def test_all_rare_values_merged(self):
    rare_mask = np.array([True, True, True, True])
    size, transform_fn = (
        transformations.create_rare_value_merging_transformation(rare_mask)
    )
    self.assertEqual(size, 1)
    self.assertEqual(transform_fn(0), 0)
    self.assertEqual(transform_fn(1), 0)
    self.assertEqual(transform_fn(2), 0)
    self.assertEqual(transform_fn(3), 0)
    self.assertIn(transform_fn.inverse(0), [0, 1, 2, 3])

  def test_discretize_raises_on_oob_bins(self):
    attr = dpsynth.NumericalAttribute(min_value=0, max_value=10)
    with self.assertRaises(ValueError):
      transformations.create_discretize_transformation(attr, [-1, 2, 3])
    with self.assertRaises(ValueError):
      transformations.create_discretize_transformation(attr, [5, 11])

  def test_uniform_discretize(self):
    attr = dpsynth.NumericalAttribute(min_value=0, max_value=10, dtype='int')
    categorical, transform = (
        transformations.create_uniform_discretize_transformation(attr, 5)
    )
    self.assertLen(categorical.possible_values, 5)
    # All values are now string intervals.
    for v in categorical.possible_values:
      self.assertIsInstance(v, str)

    # Check that the transform function returns correct intervals in-domain.
    for i in range(0, 11):
      interval = transform(i)
      self.assertIsInstance(interval, str)

  def test_binary_discretize(self):
    attr = dpsynth.NumericalAttribute(min_value=0, max_value=1, dtype='int')
    categorical, transformation = (
        transformations.create_uniform_discretize_transformation(attr, 2)
    )
    self.assertEqual(categorical.possible_values, ['[0.0, 0.5]', '(0.5, 1.0]'])
    self.assertEqual(transformation(0), '[0.0, 0.5]')
    self.assertEqual(transformation(1), '(0.5, 1.0]')

  def test_applies_transformations_to_dataframe_drop_extra_columns(self):
    values = ['A', 'B', 'C']
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': values,
        'c': ['X', 'Y', 'Z'],
    })
    transforms = {
        'a': transformations.DataTransformation(
            lambda x: x + 1, lambda x: x - 1
        ),
        'b': transformations.DataTransformation(
            values.index, lambda x: values[x]
        ),
    }

    expected = pd.DataFrame({
        'a': [2, 3, 4],
        'b': [0, 1, 2],
        'c': ['X', 'Y', 'Z'],
    })

    df2 = transformations.apply(df, transforms)
    df3 = transformations.apply(df2, transforms, reverse=True)

    pd.testing.assert_frame_equal(df2, expected[['a', 'b']])
    pd.testing.assert_frame_equal(df3, df[['a', 'b']])

  def test_applies_transformations_to_dataframe_keep_extra_columns(self):
    values = ['A', 'B', 'C']
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': values,
        'c': ['X', 'Y', 'Z'],
    })
    transforms = {
        'a': transformations.DataTransformation(
            lambda x: x + 1, lambda x: x - 1
        ),
        'b': transformations.DataTransformation(
            values.index, lambda x: values[x]
        ),
    }

    expected = pd.DataFrame({
        'a': [2, 3, 4],
        'b': [0, 1, 2],
        'c': ['X', 'Y', 'Z'],
    })

    df2 = transformations.apply(df, transforms, drop_extra_columns=False)
    df3 = transformations.apply(
        df2, transforms, reverse=True, drop_extra_columns=False
    )

    pd.testing.assert_frame_equal(df2, expected)
    pd.testing.assert_frame_equal(df3, df)


if __name__ == '__main__':
  absltest.main()
