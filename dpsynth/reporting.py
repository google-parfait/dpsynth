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

"""Privacy reporting utilities for DP Synth."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
from typing import TYPE_CHECKING, TypeAlias

from absl import logging
import dp_accounting
import numpy as np

# Guarded by TYPE_CHECKING so jax_privacy remains an optional dependency.
if TYPE_CHECKING:
  import jax_privacy  # pylint: disable=g-import-not-at-top

FloatPair: TypeAlias = tuple[float, float]

DEFAULT_TARGET_DELTAS = (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
DEFAULT_TARGET_FPRS = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
DEFAULT_AUDIT_FPRS = (0.01, 0.1)


@dataclasses.dataclass(frozen=True)
class TradeOffCurve:
  """Hypothesis testing trade-off curve (TPR vs. FPR) and optional AUROC.

  Attributes:
    points: List of ``(tpr, fpr)`` pairs evaluated at target FPRs.
    auroc: Optional area under the receiver operating characteristic curve.
  """

  points: list[FloatPair]
  auroc: float | None = None


@dataclasses.dataclass(frozen=True)
class PrivacyReport:
  """Privacy guarantees computed from a differential privacy event.

  This dataclass is intended to provide a comprehensive summary of the privacy
  guarantees of a given mechanism, rather than distilling the guarantees to a
  single (epsilon, delta) pair. This dataclass is likely to expand over time to
  include additional metrics, trade-off curves, and other information that is
  helpful to characterize the privacy properties of a mechanism.
  """

  dp_event: dp_accounting.DpEvent
  epsilon_deltas: list[FloatPair]
  gdp_estimate: float | None = None
  trade_off_curve: TradeOffCurve | None = None
  notes: str | None = None
  # All mechanisms in dpsynth assume the ADD_OR_REMOVE_ONE neighboring relation.
  neighboring_relation: str = 'ADD_OR_REMOVE_ONE'

  @classmethod
  def from_dp_event(
      cls,
      event: dp_accounting.DpEvent,
      *,
      target_deltas: float | Sequence[float] = DEFAULT_TARGET_DELTAS,
      target_false_positive_rates: Sequence[float] | None = DEFAULT_TARGET_FPRS,
      orders: Sequence[float] | None = None,
      value_discretization_interval: float | None = None,
      notes: str | None = None,
  ) -> 'PrivacyReport':
    """Computes a PrivacyReport from a DpEvent.

    Evaluates privacy guarantees across multiple evaluation points taking the
    tightest (minimum) epsilon across supported accountants (PLD and RDP).
    Computes the Gaussian Differential Privacy (GDP) parameter estimate and
    hypothesis testing trade-off curve (TPR vs. FPR) via PLD when supported.

    Example:
      >>> event = dp_accounting.GaussianDpEvent(noise_multiplier=10.0)
      >>> report = PrivacyReport.from_dp_event(
      ...     event,
      ...     notes='Column bounds were computed from raw data without DP.',
      ... )
      >>> report.notes
      'Column bounds were computed from raw data without DP.'

    Args:
      event: The DpEvent to analyze.
      target_deltas: Evaluation delta(s) for (epsilon, delta)-DP.
      target_false_positive_rates: Optional evaluation FPRs for hypothesis
        testing trade-off curve (TPR vs. FPR).
      orders: Optional Renyi differential privacy orders for RdpAccountant.
      value_discretization_interval: Optional discretization interval for
        PLDAccountant.
      notes: Optional string capturing custom notes or non-DP disclosures about
        the privacy guarantee.

    Returns:
      A PrivacyReport containing computed privacy guarantees.
    """

    if isinstance(target_deltas, (int, float)):
      target_deltas = [float(target_deltas)]

    assert all(0 < d <= 1 for d in target_deltas), 'target_deltas not in (0, 1]'

    if orders is not None:
      rdp_acc = functools.partial(
          dp_accounting.rdp.RdpAccountant, orders=orders
      )
    else:
      rdp_acc = dp_accounting.rdp.RdpAccountant

    if value_discretization_interval is not None:
      pld_acc = functools.partial(
          dp_accounting.pld.PLDAccountant,
          value_discretization_interval=value_discretization_interval,
      )
    else:
      pld_acc = dp_accounting.pld.PLDAccountant

    rdp_epsilons = _get_epsilons(event, rdp_acc, target_deltas)
    pld_epsilons = _get_epsilons(event, pld_acc, target_deltas)
    best_epsilons = [min(e) for e in zip(rdp_epsilons, pld_epsilons)]

    try:
      pld_accountant = pld_acc().compose(event)
      gdp_estimate = pld_accountant.get_gdp_parameter_estimate()
    except (dp_accounting.UnsupportedEventError, NotImplementedError):
      pld_accountant = None
      gdp_estimate = None

    trade_off_curve = None
    if pld_accountant is not None and target_false_positive_rates is not None:
      target_fprs = [float(x) for x in target_false_positive_rates]
      assert all(0 <= f <= 1 for f in target_fprs), 'FPRs must be in [0, 1]'
      try:
        tprs = pld_accountant.get_true_positive_rates(np.asarray(target_fprs))
        tpr_list = [float(t) for t in tprs]
        trade_off_curve = TradeOffCurve(points=list(zip(tpr_list, target_fprs)))
      except (dp_accounting.UnsupportedEventError, NotImplementedError):
        trade_off_curve = None

    return cls(
        dp_event=event,
        epsilon_deltas=list(zip(best_epsilons, target_deltas)),
        gdp_estimate=float(gdp_estimate) if gdp_estimate is not None else None,
        trade_off_curve=trade_off_curve,
        notes=notes,
    )


@dataclasses.dataclass(frozen=True)
class AuditReport:
  """Empirical privacy audit summary computed from a CanaryScoreAuditor.

  Complements ``PrivacyReport`` with an empirical measure of privacy loss,
  recording point estimates alongside two-sided bootstrap confidence bounds.

  Attributes:
    num_canaries: Tuple of ``(held_in, held_out)`` canary counts.
    confidence: Confidence level in ``(0, 1)`` for the bootstrap bounds.
    trade_off_curve: Point-estimate trade-off curve and AUROC.
    lower_bound: Lower bootstrap confidence bounds for the curve and AUROC.
    upper_bound: Upper bootstrap confidence bounds for the curve and AUROC.
    notes: Optional string capturing custom notes about the audit.
  """

  num_canaries: tuple[int, int]
  confidence: float
  trade_off_curve: TradeOffCurve
  lower_bound: TradeOffCurve
  upper_bound: TradeOffCurve
  notes: str | None = None

  @classmethod
  def from_auditor(
      cls,
      auditor: jax_privacy.auditing.CanaryScoreAuditor,
      *,
      confidence: float = 0.95,
      num_bootstrap_samples: int = 1000,
      target_false_positive_rates: Sequence[float] = DEFAULT_AUDIT_FPRS,
      seed: int | None = None,
      notes: str | None = None,
  ) -> 'AuditReport':
    """Computes an AuditReport from a CanaryScoreAuditor.

    Evaluates empirical membership inference AUROC and true positive rates at
    target false positive rates, alongside two-sided bias-corrected bootstrap
    confidence intervals.

    Example:
      >>> import jax_privacy
      >>> auditor = jax_privacy.auditing.CanaryScoreAuditor(
      ...     in_canary_scores=np.arange(100) + 9.5,
      ...     out_canary_scores=np.arange(100),
      ... )
      >>> report = AuditReport.from_auditor(auditor, seed=0)
      >>> round(report.trade_off_curve.auroc, 3)
      0.595

    Args:
      auditor: The CanaryScoreAuditor to summarize.
      confidence: Desired confidence level in ``(0, 1)`` for bootstrap bounds.
      num_bootstrap_samples: Number of bootstrap resamples per metric.
      target_false_positive_rates: Target FPRs at which to evaluate TPR.
      seed: Optional integer seed for reproducible bootstrap intervals.
      notes: Optional string capturing custom notes about the audit.

    Returns:
      An AuditReport containing empirical privacy audit metrics and intervals.
    """
    import jax_privacy  # pylint: disable=g-import-not-at-top

    if num_bootstrap_samples < 1:
      raise ValueError('num_bootstrap_samples must be positive.')

    bootstrap = jax_privacy.auditing.BootstrapParams.confidence_interval(
        num_samples=num_bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )

    tprs = auditor.tpr_at_given_fpr(target_false_positive_rates)
    min_fpr = min(target_false_positive_rates, default=1.0)
    if min(auditor.num_canaries) * min_fpr < 20:
      logging.warning('Too few canaries for reliable TPR at FPR=%g.', min_fpr)

    points = list(zip(tprs.tolist(), target_false_positive_rates))
    lo_points, hi_points = [], []
    for fpr in target_false_positive_rates:
      lo, hi = auditor.tpr_at_given_fpr(fpr, bootstrap_params=bootstrap)
      lo_points.append((float(lo), fpr))
      hi_points.append((float(hi), fpr))

    auroc = auditor.attack_auroc()
    auroc_lo, auroc_hi = auditor.attack_auroc(bootstrap_params=bootstrap)

    return cls(
        num_canaries=auditor.num_canaries,
        confidence=confidence,
        trade_off_curve=TradeOffCurve(points=points, auroc=auroc),
        lower_bound=TradeOffCurve(points=lo_points, auroc=float(auroc_lo)),
        upper_bound=TradeOffCurve(points=hi_points, auroc=float(auroc_hi)),
        notes=notes,
    )


def _get_epsilons(
    event: dp_accounting.DpEvent,
    make_accountant: Callable[[], dp_accounting.PrivacyAccountant],
    target_deltas: Sequence[float],
) -> list[float]:
  """Returns the epsilon values for the given deltas and accountant."""
  try:
    epsilons: list[float] = []
    accountant = make_accountant().compose(event)
    for delta in target_deltas:
      epsilons.append(float(accountant.get_epsilon(delta)))
    return epsilons
  except (dp_accounting.UnsupportedEventError, NotImplementedError):
    return [float('inf') for _ in target_deltas]
