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

import math

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth import domain
import numpy as np


class TestDomain(parameterized.TestCase):

  def test_valid_attribute(self):
    attribute = domain.CategoricalAttribute(
        possible_values=['a', 'b', 'c'], out_of_domain_index=0
    )
    self.assertEqual(attribute.size, 3)

    attribute = domain.NumericalAttribute(
        min_value=0, max_value=10, clip_to_range=True
    )
    self.assertEqual(attribute.min_value, 0)
    self.assertEqual(attribute.max_value, 10)
    self.assertEqual(attribute.clip_to_range, True)

  def test_empty_possible_values(self):
    with self.assertRaises(ValueError):
      domain.CategoricalAttribute(possible_values=[])

  def test_invalid_out_of_domain_index(self):
    with self.assertRaises(ValueError):
      domain.CategoricalAttribute(
          possible_values=['a', 'b'], out_of_domain_index=2
      )

  def test_mixed_types_rejected(self):
    with self.assertRaises(ValueError):
      domain.CategoricalAttribute(possible_values=['a', 1])
    with self.assertRaises(ValueError):
      domain.CategoricalAttribute(possible_values=[None, 'a'])

  def test_invalid_range(self):
    with self.assertRaises(ValueError):
      domain.NumericalAttribute(min_value=10, max_value=0)

  def test_to_from_yaml_roundtrip(self):
    original_domain = {
        'cat': domain.CategoricalAttribute(possible_values=['A', 'B', 'C']),
        'num': domain.NumericalAttribute(min_value=0, max_value=10),
    }

    temp_file = self.create_tempfile('temp.yaml', mode='w+')
    domain.to_yaml_file(original_domain, temp_file.full_path)
    loaded_domain = domain.from_yaml_file(temp_file.full_path)
    self.assertEqual(loaded_domain, original_domain)

  def test_interval_handling_yaml_roundtrip(self):
    original_domain = {
        'num': domain.NumericalAttribute(
            min_value=0, max_value=10, interval_handling='sample'
        ),
    }
    temp_file = self.create_tempfile('temp.yaml', mode='w+')
    domain.to_yaml_file(original_domain, temp_file.full_path)
    loaded_domain = domain.from_yaml_file(temp_file.full_path)
    self.assertEqual(loaded_domain, original_domain)

  def test_invalid_interval_handling(self):
    with self.assertRaises(ValueError):
      domain.NumericalAttribute(0, 10, interval_handling='bad')

  def test_numerical_attribute_bin_edges_float(self):
    attr = domain.NumericalAttribute(0, 10, bin_edges=(2, 5.5, 8))
    self.assertEqual(attr.bin_edges, [2.0, 5.5, 8.0])

  def test_numerical_attribute_bin_edges_int(self):
    int_attr = domain.NumericalAttribute(
        0, 10, dtype='int', bin_edges=[2, 5, 8]
    )
    self.assertEqual(int_attr.bin_edges, [2.0, 5.0, 8.0])

  def test_numerical_attribute_bin_edges_yaml_roundtrip(self):
    original = {
        'float_num': domain.NumericalAttribute(0, 10, bin_edges=(2, 5.5, 8)),
        'int_num': domain.NumericalAttribute(
            0, 10, dtype='int', bin_edges=[2, 5, 8]
        ),
    }
    temp_file = self.create_tempfile('bin_edges.yaml', mode='w+')
    domain.to_yaml_file(original, temp_file.full_path)
    loaded = domain.from_yaml_file(temp_file.full_path)
    self.assertEqual(loaded, original)

  @parameterized.named_parameters(
      dict(testcase_name='below_min', bin_edges=[-1.0, 5.0]),
      dict(testcase_name='above_max', bin_edges=[2.0, 10.0]),
      dict(testcase_name='duplicate', bin_edges=[5.0, 5.0]),
      dict(testcase_name='unsorted', bin_edges=[6.0, 4.0]),
      dict(testcase_name='non_int_for_int', bin_edges=[2.5, 5.0], dtype='int'),
      dict(testcase_name='nan', bin_edges=[math.nan, 5.0]),
      dict(testcase_name='inf', bin_edges=[2.0, math.inf]),
  )
  def test_numerical_attribute_bin_edges_invalid(
      self, bin_edges, dtype='float'
  ):
    with self.assertRaises(ValueError):
      domain.NumericalAttribute(0, 10, dtype=dtype, bin_edges=bin_edges)

  def test_standardize_categorical(self):
    attribute = domain.CategoricalAttribute(
        possible_values=['a', 'b', 'c'], out_of_domain_index=1
    )
    self.assertEqual(attribute.standardize('a'), 'a')
    self.assertEqual(attribute.standardize('b'), 'b')
    self.assertEqual(attribute.standardize('c'), 'c')
    self.assertEqual(attribute.standardize(0), 'b')
    self.assertEqual(attribute.standardize(None), 'b')
    self.assertEqual(attribute.standardize([1, 2, 3]), 'b')

  def test_standardize_numerical(self):

    in_domain_values = [0, 5, 10]
    ood_values = [-5, 15, math.nan, -math.inf, math.inf, 'a', None, (3, 4)]

    attribute = domain.NumericalAttribute(0, 10, clip_to_range=True)
    for value in in_domain_values + ood_values:
      self.assertBetween(attribute.standardize(value), 0, 10)

    attribute = domain.NumericalAttribute(0, 10, clip_to_range=False)
    for value in in_domain_values:
      self.assertEqual(attribute.standardize(value), value)

    for value in ood_values:
      self.assertTrue(math.isnan(attribute.standardize(value)))

  def test_numerical_attribute_default_sentinel(self):
    attribute = domain.NumericalAttribute(0, 10)
    self.assertIsNone(attribute.sentinel)
    self.assertTrue(np.isnan(attribute.resolved_sentinel))

  def test_numerical_attribute_custom_sentinel(self):
    attribute = domain.NumericalAttribute(0, 10, sentinel=-1)
    self.assertEqual(attribute.sentinel, -1)
    self.assertEqual(attribute.resolved_sentinel, -1)

  def test_sentinel_yaml_roundtrip(self):
    original = {
        'num': domain.NumericalAttribute(
            min_value=0, max_value=10, clip_to_range=False, sentinel=-1
        ),
    }
    temp_file = self.create_tempfile('temp.yaml', mode='w+')
    domain.to_yaml_file(original, temp_file.full_path)
    loaded = domain.from_yaml_file(temp_file.full_path)
    self.assertEqual(loaded['num'].sentinel, -1)

  def test_string_sentinel_allowed_with_interval_handling(self):
    attr = domain.NumericalAttribute(
        0, 10, sentinel='MISSING', interval_handling='interval'
    )
    self.assertEqual(attr.sentinel, 'MISSING')
    self.assertEqual(attr.resolved_sentinel, 'MISSING')

  def test_interval_handling_default_sentinel(self):
    attr = domain.NumericalAttribute(0, 10, interval_handling='interval')
    self.assertIsNone(attr.sentinel)
    self.assertEqual(attr.resolved_sentinel, '')

  def test_string_sentinel_rejected_with_midpoint_handling(self):
    with self.assertRaises(ValueError):
      domain.NumericalAttribute(0, 10, sentinel='MISSING')

  def test_numeric_sentinel_rejected_with_interval_handling(self):
    with self.assertRaises(ValueError):
      domain.NumericalAttribute(
          0, 10, sentinel=42, interval_handling='interval'
      )

  def test_numpy_numeric_sentinel_accepted(self):
    attr = domain.NumericalAttribute(0, 10, sentinel=np.int32(-1))
    self.assertEqual(attr.sentinel, -1)
    attr = domain.NumericalAttribute(0, 10, sentinel=np.float32(0.5))
    self.assertAlmostEqual(attr.sentinel, 0.5, places=5)

  def test_freeform_text_defaults(self):
    attribute = domain.FreeFormTextAttribute()
    self.assertEqual(attribute.max_tokens, 256)
    self.assertIsNone(attribute.description)
    self.assertIsNone(attribute.formatting)
    self.assertIsNone(attribute.exemplar)

  def test_freeform_text_custom_values(self):
    attribute = domain.FreeFormTextAttribute(
        max_tokens=512,
        description='A patient note.',
        formatting='Bullet points only.',
        exemplar='- Presents with headache.',
    )
    self.assertEqual(attribute.max_tokens, 512)
    self.assertEqual(attribute.description, 'A patient note.')
    self.assertEqual(attribute.formatting, 'Bullet points only.')
    self.assertEqual(attribute.exemplar, '- Presents with headache.')

  def test_freeform_text_yaml_roundtrip(self):
    original_domain = {
        'cat': domain.CategoricalAttribute(possible_values=['A', 'B']),
        'num': domain.NumericalAttribute(min_value=0, max_value=10),
        'text': domain.FreeFormTextAttribute(
            max_tokens=128,
            description='Free text.',
            formatting='Paragraphs.',
            exemplar='Hello world.',
        ),
    }
    temp_file = self.create_tempfile('temp.yaml', mode='w+')
    domain.to_yaml_file(original_domain, temp_file.full_path)
    loaded_domain = domain.from_yaml_file(temp_file.full_path)
    self.assertEqual(loaded_domain, original_domain)

  def test_freeform_text_yaml_backward_compatibility(self):
    """A YAML file written with fewer fields can still be loaded."""
    yaml_content = 'text:\n  max_tokens: 64\n'
    temp_file = self.create_tempfile('compat.yaml', content=yaml_content)
    loaded = domain.from_yaml_file(temp_file.full_path)
    self.assertEqual(
        loaded['text'], domain.FreeFormTextAttribute(max_tokens=64)
    )

  def test_open_set_yaml_roundtrip(self):
    original_domain = {
        'open_cat': domain.OpenSetCategoricalAttribute(
            default_value='<UNKNOWN>', public_possible_values=('X', 'Y')
        ),
    }
    temp_file = self.create_tempfile('open_cat.yaml', mode='w+')
    domain.to_yaml_file(original_domain, temp_file.full_path)
    loaded_domain = domain.from_yaml_file(temp_file.full_path)
    self.assertEqual(loaded_domain, original_domain)

  def test_yaml_deprecation_warning(self):
    attrs = {'a': domain.CategoricalAttribute(possible_values=['x'])}
    temp_file = self.create_tempfile('dep.yaml', mode='w+')
    with self.assertWarns(DeprecationWarning):
      domain.to_yaml_file(attrs, temp_file.full_path)
    with self.assertWarns(DeprecationWarning):
      _ = domain.from_yaml_file(temp_file.full_path)


class SchemaTest(absltest.TestCase):

  def test_from_dict(self):
    attrs = {'a': domain.CategoricalAttribute(possible_values=['x', 'y'])}
    s = domain.Schema(attrs)
    self.assertLen(s, 1)
    self.assertIsInstance(s['a'], domain.CategoricalAttribute)
    self.assertListEqual(list(s), ['a'])
    self.assertIn('a', s)
    self.assertNotIn('b', s)

  def test_empty_schema(self):
    s = domain.Schema({})
    self.assertEmpty(s)
    self.assertEmpty(list(s))

  def test_getitem_missing_raises(self):
    s = domain.Schema({'a': domain.CategoricalAttribute(['x'])})
    with self.assertRaises(KeyError):
      _ = s['missing']

  def test_default_constraints(self):
    s = domain.Schema({'a': domain.CategoricalAttribute(['x'])})
    self.assertEmpty(s.constraints)

  def test_with_constraints(self):
    mock_constraint = object()
    s = domain.Schema(
        attributes={'a': domain.CategoricalAttribute(['x'])},
        constraints=[mock_constraint],
    )
    self.assertSequenceEqual(s.constraints, (mock_constraint,))


if __name__ == '__main__':
  absltest.main()
