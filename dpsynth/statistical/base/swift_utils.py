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

"""Module implementing the SWIFT budget allocation heuristic."""

# NOTE: This module is tested in swift_test.py.

from collections.abc import Sequence
import dataclasses
import math
from typing import Any


@dataclasses.dataclass(frozen=True)
class Candidate:
  """Represents a potential item for budget allocation.

  Attributes:
      id: A unique identifier for the candidate (e.g., a string or tuple).
      error: The baseline error value (e).
      size: The scaling factor for the penalty (s).
      weight: The importance weight (w).
  """

  id: Any
  error: float
  size: float
  weight: float


@dataclasses.dataclass(frozen=True)
class _Point:
  """Internal representation of a candidate's cost-benefit attributes."""

  id: Any
  reward: float  # a_i = w_i * e_i
  cost_factor: float  # y_i = (w_i * s_i * sqrt(2/pi))^(2/3)
  ratio: float  # reward / cost_factor (used for greedy sorting)


def best_subset_and_allocation(
    candidates: Sequence[Candidate], budget: float
) -> dict[Any, float]:
  """Finds the subset of candidates and budget allocation that maximizes score.

  Formal Problem Statement:
  Given a set of candidates C and a total budget B > 0, find a subset S ⊆ C
  and an allocation B_i for each i ∈ S such that:
      1. sum_{i ∈ S} B_i = B
      2. B_i > 0 for all i ∈ S

  The objective is to maximize the total score:
      Score = sum_{i ∈ S} [ w_i * (e_i - s_i * sqrt(2/pi) * B_i^(-0.5)) ]

  Optimization Logic:
  For any fixed subset S, the optimal budget allocation is:
      B_i = B * (y_i / sum_{j ∈ S} y_j)
  where y_i = (w_i * s_i * sqrt(2/pi))^(2/3).

  This reduces the problem to selecting a subset S that maximizes:
      Objective = sum_{i ∈ S} (w_i * e_i) - (sum_{i ∈ S} y_i)^1.5 / sqrt(B)

  Args:
      candidates: A list of Candidate objects.
      budget: The total fixed budget B to be distributed. Can be interpreted as
        either a GDP or RDP budget.

  Returns:
      A dictionary mapping candidate IDs to their optimally allocated budget.
      Candidates not included in the subset are omitted.
  """
  if not candidates or budget <= 0:
    return {}

  # 1. Transform candidates into internal points and calculate ratios.
  points = []
  constant = math.sqrt(2 / math.pi)

  for candidate in candidates:
    reward = candidate.weight * candidate.error
    # y_i is the factor derived from the derivative of the budget allocation
    cost_factor = (candidate.weight * candidate.size * constant) ** (2 / 3)

    # Calculate ratio for sorting. If cost_factor is 0, the item provides
    # a fixed reward regardless of budget; we treat it as infinitely efficient.
    ratio = reward / cost_factor if cost_factor > 0 else float('inf')

    points.append(_Point(candidate.id, reward, cost_factor, ratio))

  # 2. Sort candidates by efficiency ratio (Reward per Cost Factor).
  # Since the penalty (sum y)^1.5 is convex, the marginal cost of adding
  # candidates increases. We must pick the most efficient ones first.
  points.sort(key=lambda p: p.ratio, reverse=True)

  # 3. Greedy selection of the optimal subset.
  best_subset_points = []
  current_reward_sum = 0.0
  current_cost_factor_sum = 0.0
  max_score = -float('inf')

  for point in points:
    # If the cost factor is 0, this candidate provides a constant reward
    # and takes no budget. Include it if the reward is positive.
    if point.cost_factor == 0:
      if point.reward > 0:
        best_subset_points.append(point)
        current_reward_sum += point.reward
        # Score of a set with only free items is just the sum of rewards
        if current_cost_factor_sum == 0:
          max_score = current_reward_sum
      continue

    new_reward_sum = current_reward_sum + point.reward
    new_cost_factor_sum = current_cost_factor_sum + point.cost_factor

    score_with_candidate = _calculate_score(
        new_reward_sum, new_cost_factor_sum, budget
    )

    # If adding this candidate improves our reduced objective, include it.
    if score_with_candidate > max_score:
      max_score = score_with_candidate
      current_reward_sum = new_reward_sum
      current_cost_factor_sum = new_cost_factor_sum
      best_subset_points.append(point)
    else:
      # Due to convexity of the penalty, once the marginal reward
      # no longer outweighs the marginal penalty, we have reached the peak.
      break

  # 4. Map the selected subset to the final optimal budget allocation.
  return _allocate_final_budgets(best_subset_points, budget)


def _calculate_score(
    sum_reward: float, sum_cost_factor: float, budget: float
) -> float:
  """Calculates the objective: sum(reward) - (sum(y)^1.5 / sqrt(budget))."""
  if sum_cost_factor <= 0:
    return sum_reward
  return sum_reward - (sum_cost_factor**1.5) / math.sqrt(budget)


def _allocate_final_budgets(
    subset_points: Sequence[_Point], budget: float
) -> dict[Any, float]:
  """Applies the proportional distribution: B_i = budget * (y_i / total_y)."""
  total_cost_factor = sum(p.cost_factor for p in subset_points)

  # If all selected items have a cost factor of 0, they require no budget.
  if total_cost_factor <= 0:
    return {p.id: 0.0 for p in subset_points}

  # Only items with cost_factor > 0 receive budget.
  return {
      p.id: budget * (p.cost_factor / total_cost_factor)
      for p in subset_points
      if p.cost_factor > 0
  }
