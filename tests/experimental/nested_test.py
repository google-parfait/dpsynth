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

import doctest

from absl.testing import absltest
import dp_accounting
from dpsynth import domain
from dpsynth.experimental import nested
import numpy as np
import pandas as pd


class NestedTabularConfigTest(absltest.TestCase):

  def _make_schema(self):
    """Creates a simple two-type schema for testing."""
    shared_schema = domain.Schema({
        'platform': domain.CategoricalAttribute(
            possible_values=['web', 'mobile']
        ),
    })
    per_type_schemas = {
        'click': domain.Schema({
            'element': domain.CategoricalAttribute(
                possible_values=['button', 'link', 'image']
            ),
        }),
        'purchase': domain.Schema({
            'amount': domain.CategoricalAttribute(
                possible_values=['low', 'medium', 'high']
            ),
        }),
    }
    return nested.NestedSchema(
        shared_schema=shared_schema,
        per_type_schemas=per_type_schemas,
    )

  def test_type_vocabulary(self):
    schema = self._make_schema()
    self.assertEqual(schema.type_vocabulary, ['click', 'purchase'])

  def test_calibrate_returns_new_instance(self):
    schema = self._make_schema()
    synth = nested.NestedTabularConfig()
    calibrated = synth.configure(schema, budget=10.0)
    self.assertIsInstance(calibrated, nested.NestedTabularMechanism)

  def test_dp_event_is_composed(self):
    schema = self._make_schema()
    synth = nested.NestedTabularConfig()
    calibrated = synth.configure(schema, budget=10.0)
    event = calibrated.dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    # Detail level should be a ZCDpEvent (conservative parallel composition).
    self.assertLen(event.events, 2)
    self.assertIsInstance(event.events[1], dp_accounting.ZCDpEvent)

  def test_end_to_end(self):
    schema = self._make_schema()
    synth = nested.NestedTabularConfig()
    calibrated = synth.configure(schema, budget=100.0)
    rng = np.random.default_rng(42)
    data = {
        'click': pd.DataFrame({
            'platform': ['web', 'mobile', 'web'] * 10,
            'element': ['button', 'link', 'image'] * 10,
        }),
        'purchase': pd.DataFrame({
            'platform': ['web', 'mobile', 'mobile'] * 10,
            'amount': ['low', 'medium', 'high'] * 10,
        }),
    }
    result = calibrated(rng, data)
    self.assertIsInstance(result, nested.NestedSynthesisResult)
    self.assertIsNotNone(result.shared_result)
    self.assertIn('click', result.detail_results)
    self.assertIn('purchase', result.detail_results)
    self.assertTrue(result.synthetic_data)
    for type_name, df in result.synthetic_data.items():
      self.assertIn(type_name, ['click', 'purchase'])
      self.assertIsInstance(df, pd.DataFrame)
      self.assertIn('platform', df.columns)
    self.assertIn('element', result.synthetic_data['click'].columns)
    self.assertIn('amount', result.synthetic_data['purchase'].columns)

  def test_empty_detail_schema(self):
    shared_schema = domain.Schema({
        'platform': domain.CategoricalAttribute(
            possible_values=['web', 'mobile']
        ),
    })
    per_type_schemas = {
        'click': domain.Schema({
            'element': domain.CategoricalAttribute(
                possible_values=['button', 'link']
            ),
        }),
        'marker': domain.Schema({}),
    }
    schema = nested.NestedSchema(
        shared_schema=shared_schema,
        per_type_schemas=per_type_schemas,
    )
    synth = nested.NestedTabularConfig()
    calibrated = synth.configure(schema, budget=100.0)
    self.assertNotIn('marker', calibrated.detail_synths)
    self.assertIn('click', calibrated.detail_synths)

    rng = np.random.default_rng(42)
    data = {
        'click': pd.DataFrame({
            'platform': ['web', 'mobile'] * 10,
            'element': ['button', 'link'] * 10,
        }),
        'marker': pd.DataFrame({
            'platform': ['web', 'mobile'] * 10,
        }),
    }
    result = calibrated(rng, data)
    self.assertIn('marker', result.synthetic_data)
    self.assertIn('platform', result.synthetic_data['marker'].columns)
    self.assertLen(result.synthetic_data['marker'].columns, 1)


def load_tests(loader, tests, ignore):
  del loader, ignore  # Unused.
  tests.addTests(doctest.DocTestSuite(nested))
  return tests


if __name__ == '__main__':
  absltest.main()
