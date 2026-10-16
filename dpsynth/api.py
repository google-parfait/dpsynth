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

"""Core abstractions for differentially private mechanisms.

This module defines the ``DPMechanism`` base class, which is the primary
building block for all differentially private algorithms in DP Synth.

Example usage::

  mechanism = dpsynth.discrete_mechanisms.AIMMechanism(pgm_iters=500)

  # Calibrate to (epsilon, delta)-DP.
  calibrated = dpsynth.calibrate(mechanism, epsilon=1.0, delta=1e-5)

  # Run the mechanism.
  result = calibrated(rng, data)
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any
import warnings

import dp_accounting
from dpsynth import _calibration


class CalibratedMechanism(abc.ABC):
  """A privacy-calibrated, runnable differentially private mechanism.

  Produced by ``MechanismConfig.configure()`` / ``.calibrate()``: its natural
  privacy parameter (e.g. Gaussian sigma) is populated and it is ready to run.
  It exposes the exact ``DpEvent`` characterizing its privacy cost and is
  directly callable on data.

  Subclasses must implement:

  - ``dp_event``: return the exact record-level ``DpEvent`` characterizing the
    mechanism.
  - ``__call__``: run the mechanism on data.
  """

  @property
  @abc.abstractmethod
  def dp_event(self) -> dp_accounting.DpEvent:
    """The record-level DpEvent characterizing the privacy cost of this mechanism.

    This property always reports the base record-level (``group_size=1``),
    un-subsampled (``poisson_sampling_prob=1.0``) privacy guarantee of the
    mechanism itself, even if the mechanism was produced via
    ``dpsynth.calibrate(..., group_size=k, poisson_sampling_prob=q)``.

    To compute group-level or subsampled privacy guarantees from this event
    (e.g. when composing mechanisms manually in an external accountant), apply
    ``dpsynth.with_group_size(mech.dp_event, group_size)`` and/or wrap the
    result in ``dp_accounting.PoissonSampledDpEvent(poisson_sampling_prob,
    ...)``.
    """

  @abc.abstractmethod
  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Runs the mechanism on the given data.

    Subclass signatures vary, but typically accept at least the data to operate
    on and a source of randomness.

    Args:
      *args: Positional arguments (subclass-specific).
      **kwargs: Keyword arguments (subclass-specific).
    """


class MechanismConfig(abc.ABC):
  """A recipe that produces a calibrated, runnable mechanism.

  A config holds the mechanism's structural and hyperparameter fields and knows
  how to turn a privacy budget into a runnable ``CalibratedMechanism``.
  Usage generally follows a three-phase pattern:

  1. **Construct**: Create the config with algorithm-specific parameters
     (e.g., ``AIMConfig(pgm_iters=500)``).
  2. **Calibrate**: Call ``dpsynth.calibrate(config, ..., epsilon=...,
     delta=...)`` to bind a privacy budget, returning a
     ``CalibratedMechanism`` with the mechanism's natural privacy parameter set.
  3. **Run**: Call the calibrated mechanism on data via ``__call__``.

  **Design: configure vs calibrate.**  The API separates two concerns:

  - ``configure(domain=None, *, budget, delta=0)`` is the low-level primitive
    that each config must implement. It maps a dummy scalar ``budget``
    (roughly interpretable as a zCDP ``rho`` guarantee, though not a formal
    privacy guarantee) to the mechanism's natural privacy parameter (e.g.,
    Gaussian sigma) and returns a runnable mechanism. Callers should always go
    through the top-level ``dpsynth.calibrate`` API instead.

  - ``calibrate(epsilon, delta, **kwargs)`` is the high-level entry point
    defined once on the base class. When called with ``(epsilon, delta)``,
    it performs a binary search over ``budget`` values using
    ``dp_accounting.calibrate_dp_mechanism``, calling ``configure`` at each
    candidate and inspecting the resulting ``dp_event`` for tight PLD-based
    accounting. This gives each mechanism the maximum possible budget that
    still satisfies the target (epsilon, delta) guarantee. The (epsilon, delta)
    path is more precise but more expensive than the direct ``budget`` path.

  **Why zCDP as the intermediate.**  Calibrating to zCDP rho makes it easy to
  split a privacy budget across a heterogeneous composition of mechanisms:
  simply divide rho additively in any ratio and each share is a valid zCDP
  guarantee.

  **Tight accounting via dp_events.**  Mechanisms may be tighter than their
  zCDP guarantee implies (e.g., GDP mechanisms). The ``calibrate`` binary
  search exploits this: it evaluates each candidate's raw ``dp_event`` rather
  than relying on the zCDP conversion, so the final calibration is as tight
  as the mechanism's own privacy characterization allows.
  """

  _registry: dict[str, type[MechanismConfig]] = {}

  def __init_subclass__(cls, **kwargs: Any):
    super().__init_subclass__(**kwargs)
    MechanismConfig._registry[cls.__name__] = cls

  @classmethod
  def get_subclass(cls, name: str) -> type[MechanismConfig] | None:
    """Returns the registered MechanismConfig subclass by name."""
    return cls._registry.get(name)

  @abc.abstractmethod
  def configure(self, domain=None, *, budget, delta=0) -> CalibratedMechanism:
    """Returns a calibrated mechanism for the given dummy budget.

    Converts the budget into the mechanism's natural privacy parameter
    (e.g., Gaussian sigma) and returns a runnable ``CalibratedMechanism`` with
    that parameter set. Note that ``budget`` is a dummy budget that can roughly
    be interpreted as the zCDP ``rho`` guarantee, but is not a formal privacy
    guarantee. Callers should always go through the top-level
    ``dpsynth.calibrate`` API instead.

    Most mechanisms are pure zCDP and ignore ``delta``. Mechanisms that
    consume approximate DP budget (e.g., partition selection with Gaussian
    thresholding) should raise ``ValueError`` if ``delta`` is required but
    not provided (i.e., is 0).

    Args:
      domain: Optional domain specification over which the mechanism operates
        (e.g., a dataset ``dpsynth.Schema`` or mapping from column names to
        attribute domain specifications for tabular mechanisms, an individual
        ``AttributeType`` for initializers, or a relational domain mapping).
        Mechanisms that do not need a domain ignore this argument.
      budget: Dummy privacy budget (roughly interpretable as zCDP rho, but not a
        formal guarantee).
      delta: Approximate DP delta consumed by the mechanism itself (e.g., for
        thresholding). Defaults to 0 (pure zCDP). Mechanisms that need delta
        should raise if it is 0.

    Returns:
      A calibrated, runnable mechanism.
    """

  def calibrate(
      self,
      domain=None,
      /,
      *,
      epsilon: float,
      delta: float,
      delta_split: float = 0.5,
      poisson_sampling_prob: float = 1.0,
      group_size: int = 1,
      accountant_fn: (
          Callable[[], dp_accounting.PrivacyAccountant] | None
      ) = None,
  ) -> CalibratedMechanism:
    """Deprecated. Use ``dpsynth.calibrate`` instead."""
    warnings.warn(
        'MechanismConfig.calibrate is deprecated. Use dpsynth.calibrate'
        ' instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    return _calibration.calibrate(
        self,
        domain,
        epsilon=epsilon,
        delta=delta,
        delta_split=delta_split,
        poisson_sampling_prob=poisson_sampling_prob,
        group_size=group_size,
        accountant_fn=accountant_fn,
    )


class DPMechanism(MechanismConfig, CalibratedMechanism, abc.ABC):
  """Transitional monolithic base: both a config and a runnable mechanism.

  Historically every mechanism was a single class that was constructed, then
  ``configure``d in place, then run. That design is being split into a
  ``MechanismConfig`` recipe and a ``CalibratedMechanism`` runnable (so a
  calibrated instance can never be un-calibrated and needs no nullable
  privacy-parameter fields or guards). Mechanisms are migrated one layer at a
  time; those not yet split still subclass this monolith, which exposes exactly
  the historical abstract surface (``configure`` + ``dp_event`` + ``__call__``,
  with ``calibrate`` inherited). Remove once every mechanism is split.
  """
