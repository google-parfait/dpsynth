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
import dpsynth
import mbi
import numpy as np


class ConstraintValidationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.software = dpsynth.CategoricalAttribute(
        ['GameSuite', 'OfficePro', 'DevTool']
    )
    self.os = dpsynth.CategoricalAttribute(['Windows', 'Linux', 'MacOS'])

  def test_raises_on_unequal_attribute_lengths(self):
    with self.assertRaisesRegex(ValueError, 'must have the same length'):
      dpsynth.Constraint(
          attribute_names=('Software',),
          attribute_domains=(self.software, self.os),
          possible_combinations=[],
      )

  def test_raises_on_bad_combination_length(self):
    with self.assertRaisesRegex(ValueError, 'must have length equal'):
      dpsynth.Constraint(
          attribute_names=('Software', 'OS'),
          attribute_domains=(self.software, self.os),
          possible_combinations=[('GameSuite',)],
      )

  def test_raises_on_no_mode(self):
    with self.assertRaisesRegex(ValueError, 'exactly one'):
      dpsynth.Constraint(
          attribute_names=('Software', 'OS'),
          attribute_domains=(self.software, self.os),
      )

  def test_raises_on_multiple_modes(self):
    with self.assertRaisesRegex(ValueError, 'exactly one'):
      dpsynth.Constraint(
          attribute_names=('Software', 'OS'),
          attribute_domains=(self.software, self.os),
          possible_combinations=[('GameSuite', 'Windows')],
          impossible_combinations=[('DevTool', 'Windows')],
      )

  def test_functional_dependency_requires_two_attributes(self):
    a = dpsynth.CategoricalAttribute(['x', 'y'])
    b = dpsynth.CategoricalAttribute(['p', 'q'])
    c = dpsynth.CategoricalAttribute(['r', 's'])
    with self.assertRaisesRegex(ValueError, 'exactly 2 attributes'):
      dpsynth.Constraint(
          attribute_names=('a', 'b', 'c'),
          attribute_domains=(a, b, c),
          functional_dependency={'x': 'p'},
      )


class ConstraintToMbiTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.software = dpsynth.CategoricalAttribute(
        ['GameSuite', 'OfficePro', 'DevTool']
    )
    self.os = dpsynth.CategoricalAttribute(['Windows', 'Linux', 'MacOS'])

  def test_possible_combinations(self):
    c = dpsynth.Constraint(
        attribute_names=('Software', 'OS'),
        attribute_domains=(self.software, self.os),
        possible_combinations=[
            ('GameSuite', 'Windows'),
            ('OfficePro', 'Windows'),
            ('OfficePro', 'MacOS'),
            ('DevTool', 'Linux'),
            ('DevTool', 'MacOS'),
        ],
    )
    mbi_c = c.to_mbi()
    self.assertIsInstance(mbi_c, mbi.Constraint)
    expected = np.full((3, 3), -np.inf)
    expected[0, 0] = 0  # GameSuite, Windows
    expected[1, 0] = 0  # OfficePro, Windows
    expected[1, 2] = 0  # OfficePro, MacOS
    expected[2, 1] = 0  # DevTool, Linux
    expected[2, 2] = 0  # DevTool, MacOS
    np.testing.assert_array_equal(mbi_c.potential.values, expected)

  def test_impossible_combinations(self):
    c = dpsynth.Constraint(
        attribute_names=('Software', 'OS'),
        attribute_domains=(self.software, self.os),
        impossible_combinations=[
            ('GameSuite', 'Linux'),
            ('GameSuite', 'MacOS'),
        ],
    )
    vals = np.asarray(c.to_mbi().potential.values)
    self.assertEqual(vals[0, 1], -np.inf)  # GameSuite, Linux
    self.assertEqual(vals[0, 2], -np.inf)  # GameSuite, MacOS
    self.assertEqual(vals[0, 0], 0.0)
    self.assertEqual(vals[1, 0], 0.0)
    self.assertEqual(vals[2, 2], 0.0)

  def test_functional_dependency(self):
    fine = dpsynth.CategoricalAttribute(['a', 'b', 'c', 'd'])
    coarse = dpsynth.CategoricalAttribute(['X', 'Y'])
    c = dpsynth.Constraint(
        attribute_names=('fine', 'coarse'),
        attribute_domains=(fine, coarse),
        functional_dependency={'a': 'X', 'b': 'X', 'c': 'Y', 'd': 'Y'},
    )
    mbi_c = c.to_mbi()
    self.assertTrue(mbi_c.is_deterministic)
    vals = np.asarray(mbi_c.potential.values)
    self.assertEqual(vals[0, 0], 0.0)  # a -> X
    self.assertEqual(vals[1, 0], 0.0)  # b -> X
    self.assertEqual(vals[2, 1], 0.0)  # c -> Y
    self.assertEqual(vals[3, 1], 0.0)  # d -> Y
    self.assertEqual(vals[0, 1], -np.inf)
    self.assertEqual(vals[2, 0], -np.inf)


if __name__ == '__main__':
  absltest.main()
