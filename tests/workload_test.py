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

"""Tests for the Workload class and schema consistency validation."""

import math

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth import workload


class WorkloadTest(parameterized.TestCase):

  def test_unweighted_workload(self):
    w = workload.Workload([('a', 'b'), ('b', 'c')])
    self.assertEqual(w.cliques, (('a', 'b'), ('b', 'c')))
    self.assertIsNone(w.weights)
    self.assertEqual(w.as_dict(), {('a', 'b'): 1.0, ('b', 'c'): 1.0})

  def test_weighted_workload(self):
    w = workload.Workload([('a', 'b'), ('c',)], weights=[2.5, 0.5])
    self.assertEqual(w.cliques, (('a', 'b'), ('c',)))
    self.assertEqual(w.weights, (2.5, 0.5))
    self.assertEqual(w.as_dict(), {('a', 'b'): 2.5, ('c',): 0.5})

  def test_all_kway(self):
    w2 = workload.Workload.all_kway(['a', 'b', 'c'], degree=2)
    self.assertEqual(w2.cliques, (('a', 'b'), ('a', 'c'), ('b', 'c')))
    self.assertIsNone(w2.weights)

    w3_clamped = workload.Workload.all_kway(['a', 'b'], degree=3)
    self.assertEqual(w3_clamped.cliques, (('a', 'b'),))

    with self.assertRaises(ValueError):
      workload.Workload.all_kway(['a', 'b'], degree=0)

  def test_from_any(self):
    self.assertIsNone(workload.Workload.from_any(None))
    w = workload.Workload([('a', 'b')])
    self.assertIs(workload.Workload.from_any(w), w)
    self.assertEqual(workload.Workload.from_any([('a', 'b')]), w)
    weighted = workload.Workload.from_any({('a', 'b'): 2.5, ('c',): 0.5})
    expected = workload.Workload([('a', 'b'), ('c',)], weights=[2.5, 0.5])
    self.assertEqual(weighted, expected)

  @parameterized.named_parameters(
      dict(testcase_name='bare_string', cliques=['ab'], weights=None),
      dict(testcase_name='empty_clique', cliques=[()], weights=None),
      dict(testcase_name='non_string_element', cliques=[(1, 2)], weights=None),
      dict(
          testcase_name='duplicate_in_clique',
          cliques=[('a', 'a')],
          weights=None,
      ),
      dict(
          testcase_name='mismatched_weights_len',
          cliques=[('a',), ('b',)],
          weights=[1.0],
      ),
      dict(testcase_name='zero_weight', cliques=[('a',)], weights=[0.0]),
      dict(testcase_name='negative_weight', cliques=[('a',)], weights=[-1.0]),
      dict(testcase_name='nan_weight', cliques=[('a',)], weights=[math.nan]),
      dict(testcase_name='inf_weight', cliques=[('a',)], weights=[math.inf]),
  )
  def test_invalid_workload_raises(self, cliques, weights):
    with self.assertRaises(ValueError):
      workload.Workload(cliques, weights=weights)

  def test_validate_unknown_attribute_raises(self):
    w = workload.Workload([('a', 'b'), ('b', 'unknown')])
    with self.assertRaisesRegex(ValueError, 'Unknown workload attributes'):
      w.validate(['a', 'b', 'c'])

  def test_validate_uncovered_attribute_logs_warning(self):
    w = workload.Workload([('a', 'b')])
    with self.assertLogs(level='WARNING') as logs:
      w.validate(['a', 'b', 'c'])
    matched = ['Attributes not in workload' in m for m in logs.output]
    self.assertTrue(any(matched))


if __name__ == '__main__':
  absltest.main()
