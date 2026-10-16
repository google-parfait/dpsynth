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

"""Unit tests for privacy report generation and serialization."""

from __future__ import annotations

import doctest

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
import dpsynth
from dpsynth import reporting
from dpsynth import serialize


class ReportingTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.report = reporting.PrivacyReport.from_dp_event(
        dp_accounting.GaussianDpEvent(noise_multiplier=2.0)
    )

  def test_from_dp_event_gaussian_defaults(self):
    self.assertEqual(
        self.report.dp_event,
        dp_accounting.GaussianDpEvent(noise_multiplier=2.0),
    )
    self.assertEqual(self.report.neighboring_relation, 'ADD_OR_REMOVE_ONE')
    self.assertLen(
        self.report.epsilon_deltas, len(reporting.DEFAULT_TARGET_DELTAS)
    )
    self.assertIsNotNone(self.report.gdp_estimate)
    assert self.report.gdp_estimate is not None
    self.assertAlmostEqual(self.report.gdp_estimate, 0.5, delta=1e-2)
    self.assertIsNotNone(self.report.trade_off_curve)
    assert self.report.trade_off_curve is not None
    self.assertLen(
        self.report.trade_off_curve,
        len(reporting.DEFAULT_TARGET_FPRS),
    )
    self.assertIsNone(self.report.notes)

  @parameterized.parameters(
      (1e-10, 3.0994303302431994),
      (1e-9, 2.909732380763828),
      (1e-8, 2.7076063342227563),
      (1e-7, 2.4903385293404336),
      (1e-6, 2.254084650219736),
      (1e-5, 1.9930914044151198),
      (1e-4, 1.6980725317367753),
      (1e-3, 1.3522762448025536),
  )
  def test_from_dp_event_gaussian_epsilons(self, delta, expected_eps):
    eps_map = dict((d, eps) for eps, d in self.report.epsilon_deltas)
    self.assertAlmostEqual(eps_map[delta], expected_eps, delta=1e-5)

  def test_from_dp_event_trade_off_curve_properties(self):
    trade_off = self.report.trade_off_curve
    assert trade_off is not None
    prev_tpr, prev_fpr = 0.0, 0.0
    for tpr, fpr in trade_off:
      self.assertGreaterEqual(fpr, 0.0)
      self.assertLessEqual(fpr, 1.0)
      self.assertGreaterEqual(tpr, 0.0)
      self.assertLessEqual(tpr, 1.0)
      self.assertGreaterEqual(tpr, fpr)
      self.assertGreaterEqual(fpr, prev_fpr)
      self.assertGreaterEqual(tpr, prev_tpr)
      prev_tpr, prev_fpr = tpr, fpr

  def test_from_dp_event_trade_off_curve_custom_fprs(self):
    event = dp_accounting.GaussianDpEvent(noise_multiplier=2.0)
    report_single = reporting.PrivacyReport.from_dp_event(
        event,
        value_discretization_interval=0.1,
        target_false_positive_rates=[0.05],
    )
    assert report_single.trade_off_curve is not None
    self.assertLen(report_single.trade_off_curve, 1)
    self.assertAlmostEqual(report_single.trade_off_curve[0][1], 0.05)

    report_none = reporting.PrivacyReport.from_dp_event(
        event,
        value_discretization_interval=0.1,
        target_false_positive_rates=None,
    )
    self.assertIsNone(report_none.trade_off_curve)

  def test_from_dp_event_notes(self):
    event = dp_accounting.GaussianDpEvent(noise_multiplier=2.0)
    report = reporting.PrivacyReport.from_dp_event(
        event,
        value_discretization_interval=0.1,
        notes='Column bounds were computed from raw data without DP.',
    )
    self.assertEqual(
        report.notes,
        'Column bounds were computed from raw data without DP.',
    )

    yaml_str = serialize.to_yaml(report)
    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, report)

  def test_from_dp_event_custom_accountant_args(self):
    event = dp_accounting.GaussianDpEvent(noise_multiplier=2.0)
    report = reporting.PrivacyReport.from_dp_event(
        event,
        orders=[2.0, 3.0, 4.0],
        value_discretization_interval=1e-3,
    )
    self.assertLen(report.epsilon_deltas, len(reporting.DEFAULT_TARGET_DELTAS))

  def test_from_dp_event_rdp_fallback(self):
    event = dp_accounting.ZCDpEvent(rho=0.5)
    report = reporting.PrivacyReport.from_dp_event(event, target_deltas=1e-6)
    self.assertIsNone(report.gdp_estimate)
    self.assertLen(report.epsilon_deltas, 1)
    self.assertGreater(report.epsilon_deltas[0][0], 0.0)

  def test_from_dp_event_unsupported_event(self):
    class UnknownEvent(dp_accounting.DpEvent):
      pass

    report = reporting.PrivacyReport.from_dp_event(
        UnknownEvent(), target_deltas=1e-6
    )
    self.assertEqual(report.epsilon_deltas[0][0], float('inf'))
    self.assertIsNone(report.trade_off_curve)

  def test_serialize_yaml_roundtrip(self):
    yaml_str = serialize.to_yaml(self.report)
    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, self.report)

  def test_public_reexport(self):
    self.assertIs(dpsynth.PrivacyReport, reporting.PrivacyReport)


def load_tests(loader, tests, ignore):
  del loader, ignore  # Unused.
  tests.addTests(
      doctest.DocTestSuite(reporting, optionflags=doctest.NORMALIZE_WHITESPACE)
  )
  return tests


if __name__ == '__main__':
  absltest.main()
