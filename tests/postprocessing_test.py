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
import mbi
import numpy as np
import pandas as pd


class PostprocessingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    attr1 = dpsynth.CategoricalAttribute(possible_values=['a', 'b', 'c'])
    attr2 = dpsynth.CategoricalAttribute(possible_values=['x', 'y'])
    attr3 = dpsynth.CategoricalAttribute(possible_values=['1', '2', '3'])

    self.marginals = [
        pd.DataFrame({
            'A': ['a', 'a', 'b', 'b', 'c'],
            'B': ['x', 'y', 'x', 'y', 'x'],
            'count': [5, 10, 15, 20, 25],
        }),
        pd.DataFrame({
            'C': ['1', '2', '3'],
            'count': [20, 25, 30],
        }),
    ]
    self.attribute_domains = {'A': attr1, 'B': attr2, 'C': attr3}

  def test_infer_categorical_domain(self):

    inferred_domain = dpsynth.infer_categorical_domain(self.marginals)
    expected_domain = self.attribute_domains

    for key in ['A', 'B', 'C']:
      inferred = set(inferred_domain[key].possible_values)
      expected = set(expected_domain[key].possible_values)
      self.assertEqual(inferred, expected)

  def test_raises_error_on_wrong_format(self):
    marginals = [self.marginals[0].drop(columns='count')]

    with self.assertRaisesRegex(ValueError, 'found B instead'):
      dpsynth.infer_categorical_domain(marginals)

    with self.assertRaisesRegex(ValueError, 'found B instead'):
      dpsynth.encode_marginals(marginals, {})

    with self.assertRaisesRegex(ValueError, 'found B instead'):
      dpsynth.generate_synthetic_data_from_marginals(marginals)

  def test_encode_marginals(self):

    encoded = dpsynth.encode_marginals(self.marginals, self.attribute_domains)
    self.assertLen(encoded, 2)
    self.assertEqual(encoded[0].domain, mbi.Domain.fromdict({'A': 3, 'B': 2}))
    self.assertEqual(encoded[1].domain, mbi.Domain.fromdict({'C': 3}))

    np.testing.assert_allclose(
        encoded[0].values,
        np.array([[5, 10], [15, 20], [25, np.nan]]),
    )
    np.testing.assert_allclose(
        encoded[1].values,
        np.array([20, 25, 30]),
    )

  def test_estimate_model_from_marginals(self):

    synthetic_data = dpsynth.generate_synthetic_data_from_marginals(
        self.marginals, iters=100
    )

    self.assertEqual(synthetic_data.shape, (75, 3))
    self.assertEqual(set(synthetic_data.columns), {'A', 'B', 'C'})
    self.assertEqual(set(synthetic_data['A'].unique()), {'a', 'b', 'c'})
    self.assertEqual(set(synthetic_data['B'].unique()), {'x', 'y'})
    self.assertEqual(set(synthetic_data['C'].unique()), {'1', '2', '3'})

  def test_estimate_model_from_marginals_with_exact_marginals(self):
    exact_marginals = [
        pd.DataFrame({
            'A': ['a', 'a', 'b', 'b', 'c', 'd'],
            'B': ['x', 'y', 'x', 'y', 'x', 'x'],
            'count': [5, 10, 15, 20, 25, 0],
        }),
        pd.DataFrame({
            'C': ['1', '2', '3'],
            'count': [20, 25, 30],
        }),
    ]

    synthetic_data = dpsynth.generate_synthetic_data_from_marginals(
        self.marginals,
        iters=100,
        exact_marginals=exact_marginals,
    )

    self.assertEqual(synthetic_data.shape, (75, 3))
    self.assertEqual(set(synthetic_data.columns), {'A', 'B', 'C'})
    self.assertEqual(set(synthetic_data['B'].unique()), {'x', 'y'})
    self.assertEqual(set(synthetic_data['C'].unique()), {'1', '2', '3'})

  def test_extra_domain_elements(self):

    # Total mass on B is 80, while on A is 75.  The extra 5 elements should
    # be allocated to A='d', even though it wasn't measured.
    marginals = [
        pd.DataFrame({'A': ['a', 'b', 'c'], 'count': [25, 25, 25]}),
        pd.DataFrame({'B': ['x', 'y'], 'count': [40, 40]}),
    ]

    synthetic_data = dpsynth.generate_synthetic_data_from_marginals(
        marginals,
        iters=100,
        extra_domain_elements={'A': ['d']},
    )
    self.assertEqual(set(synthetic_data['A'].unique()), {'a', 'b', 'c', 'd'})
    self.assertEqual(synthetic_data.shape, (80, 2))
    self.assertEqual(sum(synthetic_data['A'] == 'a'), 25)
    self.assertEqual(sum(synthetic_data['A'] == 'b'), 25)
    self.assertEqual(sum(synthetic_data['A'] == 'c'), 25)
    self.assertEqual(sum(synthetic_data['A'] == 'd'), 5)
    self.assertEqual(sum(synthetic_data['B'] == 'x'), 40)
    self.assertEqual(sum(synthetic_data['B'] == 'y'), 40)


if __name__ == '__main__':
  absltest.main()
