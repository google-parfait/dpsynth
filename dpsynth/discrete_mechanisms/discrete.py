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

"""Wrapper that adds one-way marginals and domain compression to any mechanism.

``DiscreteConfig`` composes an inner discrete mechanism (AIM, MST,
SWIFT, etc.) with shared pre- and post-processing: one-way marginal
measurement, domain compression, and decompression.  It is the recommended
entry point for purely discrete tables; mixed-type tables should use
``dpsynth.TabularSynthesizer`` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
from dpsynth.discrete_mechanisms import mst
import mbi
import numpy as np


@dataclasses.dataclass(frozen=True)
class DiscreteConfig(api.MechanismConfig):
  """Wraps an inner mechanism with one-way measurement and compression.

  Attributes:
    mechanism: The base mechanism config (e.g. ``AIMConfig()``).
    compress_columns: Domain compression config. True = all, list = specific.
    one_way_budget_fraction: Fraction of zCDP budget for one-way marginals.
    constraints: Default MBI constraints to enforce. Can be overridden at call
      time via the ``constraints`` kwarg on ``DiscreteMechanism.__call__``.
    use_jax_for_bincount: Whether to use JAX-accelerated bincount (via
      mbi.extensions.precompute_marginals) to compute marginals from Dataset.
    use_jax_for_generation: Whether to use JAX-accelerated generation (via
      mbi.extensions.synthetic_data) to generate synthetic data from the model.
  """

  mechanism: api.MechanismConfig = mst.MSTConfig()
  compress_columns: bool | Sequence[str] = False
  one_way_budget_fraction: float = 0.1
  constraints: Sequence[mbi.Constraint] = ()
  use_jax_for_bincount: bool = False
  use_jax_for_generation: bool = False

  def configure(self, _=None, *, budget, delta=0):
    """Configures the synthesizer with a zCDP budget."""
    one_way_rho = budget * self.one_way_budget_fraction
    remaining_rho = budget * (1 - self.one_way_budget_fraction)
    inner = self.mechanism.configure(
        budget=remaining_rho,
        delta=delta,
    )
    return DiscreteMechanism(
        config=self,
        base_mechanism=inner,
        one_way_gdp_budget=accounting.zcdp_to_gdp(one_way_rho),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class DiscreteMechanism(api.CalibratedMechanism):
  """Calibrated synthesizer: one-way + compress + inner + decompress."""

  config: DiscreteConfig
  base_mechanism: api.CalibratedMechanism
  one_way_gdp_budget: float

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Composes one-way measurement event with the inner mechanism's event."""
    events = []
    if self.one_way_gdp_budget > 0:
      noise_multiplier = accounting.gdp_gaussian_sigma(self.one_way_gdp_budget)
      events.append(dp_accounting.GaussianDpEvent(noise_multiplier))

    inner_event = self.base_mechanism.dp_event
    if isinstance(inner_event, dp_accounting.ComposedDpEvent):
      events.extend(inner_event.events)
    elif not isinstance(inner_event, dp_accounting.NoOpDpEvent):
      events.append(inner_event)

    # Filter out any lingering NoOpDpEvents from the inner event
    events = [e for e in events if not isinstance(e, dp_accounting.NoOpDpEvent)]

    if not events:
      return dp_accounting.NoOpDpEvent()
    if len(events) == 1:
      return events[0]
    return dp_accounting.ComposedDpEvent(events)

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: Sequence[mbi.LinearMeasurement] | None = None,
      constraints: Sequence[mbi.Constraint] | None = None,
  ) -> common.DiscreteMechanismResult:
    """Runs the one-way + compress + inner mechanism + decompress pipeline.

    Args:
      rng: A numpy random number generator.
      data: The dataset to generate synthetic data for.
      initial_measurements: Pre-measured one-way marginals to use instead of
        measuring them here (e.g. fed from TabularSynthesizer).
      constraints: MBI constraints to enforce. If None, falls back to
        ``self.config.constraints``.

    Returns:
      A DiscreteMechanismResult containing the synthetic data and diagnostics.
    """
    if constraints is None:
      constraints = self.config.constraints

    if initial_measurements is not None:
      measurements = list(initial_measurements)
    elif self.one_way_gdp_budget > 0:
      one_way_cliques = [(a,) for a in data.domain]
      if hasattr(data, 'cliques'):
        supported = common.downward_closure(data.cliques)
        one_way_cliques = [cl for cl in one_way_cliques if cl in supported]
      measurements = common.measure_marginals_with_noise(
          rng=rng,
          data=data,  # pyrefly: ignore[bad-argument-type]
          marginal_queries=one_way_cliques,  # pyrefly: ignore[bad-argument-type]
          gdp_sigma=accounting.gdp_gaussian_sigma(self.one_way_gdp_budget),
      )
    else:
      measurements = []

    mappings = common.compression_mappings(
        measurements,
        self.config.compress_columns,
        constraints,
    )
    if mappings and isinstance(data, mbi.Dataset):
      data = data.compress(mappings)  # pyrefly: ignore[bad-argument-type]
      measurements = [m.compress(mappings, data.domain) for m in measurements]  # pyrefly: ignore[bad-argument-type]
    logging.info('[DPSynth]: Compressed discrete domain:\n%s', data.domain)

    cfg = self.config.mechanism
    if isinstance(data, mbi.Dataset) and hasattr(cfg, 'supporting_cliques'):
      cliques = cfg.supporting_cliques(data.domain)
      data = common.precompute_marginals(
          data,
          cliques,  # pyrefly: ignore[bad-argument-type]
          use_jax=self.config.use_jax_for_bincount,
      )

    result = self.base_mechanism(
        rng,
        data,
        initial_measurements=measurements,
        constraints=constraints,
    )
    if result.synthetic_data is not None:
      synthetic_data = result.synthetic_data
    else:
      synthetic_data = common.generate_synthetic_data(
          result.model,
          rng,
          use_jax=self.config.use_jax_for_generation,
      )
    if mappings:
      synthetic_data = synthetic_data.decompress(mappings)  # pyrefly: ignore[bad-argument-type]

    return dataclasses.replace(
        result,
        synthetic_data=synthetic_data,
        mappings=mappings,
    )
