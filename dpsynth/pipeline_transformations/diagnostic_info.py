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

"""Module for updating diagnostic information."""

import copy
from dpsynth.pipeline_transformations import types
import pipeline_dp

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Attributes:
  attributes: list[int] = field(default_factory=list)

@dataclass
class MarginalL1Distance:
  attributes: Optional[Attributes] = None
  value: float = 0.0

@dataclass
class RoundInfo:
  l1_distances: list[MarginalL1Distance] = field(default_factory=list)
  selected_attributes: list[Attributes] = field(default_factory=list)

@dataclass
class DPOperation:
  name: str = ""
  mechanism_type: str = ""
  epsilon: float = 0.0
  delta: float = 0.0
  sigma: float = 0.0
  count: int = 0

@dataclass
class DiagnosticInformation:
  epsilon: float = 0.0
  delta: float = 0.0
  mechanism: str = ""
  dp_operations: list[DPOperation] = field(default_factory=list)
  attribute_names: list[str] = field(default_factory=list)
  compressed_attribute_sizes: list[int] = field(default_factory=list)
  round_info: list[RoundInfo] = field(default_factory=list)


# Type alias for clique
Clique = tuple[int, ...]


def update_diagnostic_info(
    backend: pipeline_dp.PipelineBackend,
    diagnostic_info_collection: types.Collection[DiagnosticInformation],
    errors_singleton: types.Collection[list[tuple[Clique, float]]],
    selected_marginals: types.Collection[list[Clique]],
    stage_name: str,
) -> types.Collection[DiagnosticInformation]:
  """Updates diagnostic info with round information."""

  def _update_fn(
      diagnostic_info: DiagnosticInformation,
      errors: list[tuple[Clique, float]],
      selected_marginals: list[Clique],
  ) -> DiagnosticInformation:
    result = copy.copy(diagnostic_info)
    round_info = RoundInfo()

    for clique, value in errors:
      marginal_l1 = MarginalL1Distance(
          attributes=Attributes(attributes=clique),
          value=value,
      )
      round_info.l1_distances.append(marginal_l1)

    for marginal in selected_marginals:
      selected = Attributes(attributes=marginal)
      round_info.selected_attributes.append(selected)

    result.round_info.append(round_info)
    return result

  return backend.map_with_side_inputs(
      diagnostic_info_collection,
      _update_fn,
      [errors_singleton, selected_marginals],
      stage_name,
  )
