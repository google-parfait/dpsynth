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
import jax_privacy
import numpy as np


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
    self.assertIsInstance(self.report.trade_off_curve, reporting.TradeOffCurve)
    self.assertIsNone(self.report.trade_off_curve.auroc)
    self.assertLen(
        self.report.trade_off_curve.points,
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
    for tpr, fpr in trade_off.points:
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
    self.assertLen(report_single.trade_off_curve.points, 1)
    self.assertAlmostEqual(report_single.trade_off_curve.points[0][1], 0.05)

    report_none = reporting.PrivacyReport.from_dp_event(
        event,
        value_discretization_interval=0.1,
        target_false_positive_rates=None,
    )
    self.assertIsNone(report_none.trade_off_curve)
    self.assertEqual(
        serialize.from_yaml(serialize.to_yaml(report_none)), report_none
    )

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
    self.assertIs(dpsynth.AuditReport, reporting.AuditReport)
    self.assertIs(dpsynth.TradeOffCurve, reporting.TradeOffCurve)

  def test_from_auditor_defaults_and_yaml_roundtrip(self):
    in_scores = np.arange(80, dtype=float) + 20.0
    out_scores = np.arange(120, dtype=float)
    auditor = jax_privacy.auditing.CanaryScoreAuditor(in_scores, out_scores)
    report = reporting.AuditReport.from_auditor(
        auditor,
        num_bootstrap_samples=200,
        seed=42,
        notes='Synthetic canary audit.',
    )
    self.assertEqual(report.num_canaries, (80, 120))
    self.assertEqual(report.confidence, 0.95)
    self.assertEqual(report.notes, 'Synthetic canary audit.')
    assert report.trade_off_curve.auroc is not None
    assert report.lower_bound.auroc is not None
    assert report.upper_bound.auroc is not None
    self.assertAlmostEqual(
        report.trade_off_curve.auroc, float(auditor.attack_auroc())
    )
    self.assertLessEqual(report.lower_bound.auroc, report.upper_bound.auroc)
    expected_fprs = list(reporting.DEFAULT_TARGET_FPRS)
    self.assertEqual(
        [fpr for _, fpr in report.trade_off_curve.points], expected_fprs
    )
    self.assertEqual(
        [fpr for _, fpr in report.lower_bound.points], expected_fprs
    )
    self.assertEqual(
        [fpr for _, fpr in report.upper_bound.points], expected_fprs
    )
    for (tpr, fpr), (lo, _), (hi, _) in zip(
        report.trade_off_curve.points,
        report.lower_bound.points,
        report.upper_bound.points,
    ):
      self.assertAlmostEqual(tpr, float(auditor.tpr_at_given_fpr(fpr)))
      self.assertLessEqual(lo, hi)
      self.assertGreaterEqual(lo, 0.0)
      self.assertLessEqual(hi, 1.0)

    report_same_seed = reporting.AuditReport.from_auditor(
        auditor,
        num_bootstrap_samples=200,
        seed=42,
        notes='Synthetic canary audit.',
    )
    self.assertEqual(report, report_same_seed)

    yaml_str = serialize.to_yaml(report)
    self.assertNotIn('!!python/tuple', yaml_str)
    self.assertIn('confidence: 0.95', yaml_str)
    self.assertIn('num_canaries: [80, 120]', yaml_str)
    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, report)
    self.assertIsInstance(loaded.num_canaries, tuple)
    self.assertIsInstance(loaded.trade_off_curve, reporting.TradeOffCurve)
    self.assertIsInstance(loaded.trade_off_curve.points[0], tuple)
    self.assertIsInstance(loaded.lower_bound, reporting.TradeOffCurve)
    self.assertIsInstance(loaded.lower_bound.points[0], tuple)

  @parameterized.parameters(0.0, 0.5, 2.0)
  def test_from_auditor_custom_args_and_exact_bootstrap_match(self, shift):
    rng = np.random.default_rng(123)
    auditor = jax_privacy.auditing.CanaryScoreAuditor(
        rng.normal(loc=shift, size=150),
        rng.normal(loc=0.0, size=200),
    )
    report = reporting.AuditReport.from_auditor(
        auditor,
        confidence=0.90,
        num_bootstrap_samples=100,
        target_false_positive_rates=[0.2, 0.05],
        seed=7,
    )
    expected_bootstrap = (
        jax_privacy.auditing.BootstrapParams.confidence_interval(
            num_samples=100,
            confidence=0.90,
            seed=7,
        )
    )
    expected_auroc_ci = np.asarray(
        auditor.attack_auroc(bootstrap_params=expected_bootstrap)
    )
    self.assertEqual(report.num_canaries, (150, 200))
    self.assertEqual(report.confidence, 0.90)
    self.assertEqual(
        [fpr for _, fpr in report.trade_off_curve.points], [0.2, 0.05]
    )
    self.assertIsNone(report.notes)
    assert report.trade_off_curve.auroc is not None
    assert report.lower_bound.auroc is not None
    assert report.upper_bound.auroc is not None
    self.assertAlmostEqual(
        report.trade_off_curve.auroc, float(auditor.attack_auroc())
    )
    self.assertAlmostEqual(
        report.lower_bound.auroc, float(expected_auroc_ci[0])
    )
    self.assertAlmostEqual(
        report.upper_bound.auroc, float(expected_auroc_ci[1])
    )
    self.assertLessEqual(report.lower_bound.auroc, report.trade_off_curve.auroc)
    self.assertLessEqual(report.trade_off_curve.auroc, report.upper_bound.auroc)
    for (est, fpr), (lo, lo_fpr), (hi, hi_fpr) in zip(
        report.trade_off_curve.points,
        report.lower_bound.points,
        report.upper_bound.points,
    ):
      expected_ci = np.asarray(
          auditor.tpr_at_given_fpr(fpr, bootstrap_params=expected_bootstrap)
      )
      self.assertEqual(lo_fpr, fpr)
      self.assertEqual(hi_fpr, fpr)
      self.assertAlmostEqual(est, float(auditor.tpr_at_given_fpr(fpr)))
      self.assertAlmostEqual(lo, float(expected_ci[0]))
      self.assertAlmostEqual(hi, float(expected_ci[1]))
      self.assertLessEqual(lo, est)
      self.assertLessEqual(est, hi)

  @parameterized.parameters(
      dict(confidence=0.0),
      dict(confidence=1.0),
      dict(confidence=-0.1),
      dict(num_bootstrap_samples=0),
      dict(num_bootstrap_samples=-5),
      dict(target_false_positive_rates=[-0.1]),
      dict(target_false_positive_rates=[1.5]),
  )
  def test_from_auditor_invalid_args_raises(self, **kwargs):
    auditor = jax_privacy.auditing.CanaryScoreAuditor(
        np.arange(20, dtype=float), np.arange(20, dtype=float)
    )
    with self.assertRaises(ValueError):
      reporting.AuditReport.from_auditor(auditor, **kwargs)


def load_tests(loader, tests, ignore):
  del loader, ignore  # Unused.
  tests.addTests(
      doctest.DocTestSuite(reporting, optionflags=doctest.NORMALIZE_WHITESPACE)
  )
  return tests


if __name__ == '__main__':
  absltest.main()
