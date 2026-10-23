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

"""Attribute-Conditioned Text Generation (ACTG) mechanism.

Synthesizes mixed tabular and free-form text records in three stages:
1. Synthesize structured tabular attributes with ``TabularMechanism``.
2. Fine-tune a conditional language model ``P(text | tabular)`` via DP-SFT
   (or use the base model directly when ``tabular_budget_fraction == 1.0``).
3. Sample synthetic text conditioned on the synthesized tabular rows.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
from typing import Any

import dp_accounting
from dpsynth import api
from dpsynth import data_generation_v3
from dpsynth import domain
from dpsynth.text import callbacks
from dpsynth.text import dp_sft
from dpsynth.text import model
from flax import nnx
import numpy as np
import pandas as pd


def _json_scalar(val: Any) -> Any:
  """Converts a numpy/pandas scalar into a JSON-serializable Python value."""
  if pd.isna(val):
    return None
  if isinstance(val, np.generic):
    return val.item()
  return val


def _str_val(val: Any) -> str:
  """Converts a scalar value to string, mapping None/NaN to empty string."""
  if val is None or (np.ndim(val) == 0 and pd.isna(val)):
    return ''
  return str(val)


def _partition_schema(
    schema: domain.Schema,
) -> tuple[domain.Schema, dict[str, domain.FreeFormTextAttribute]]:
  """Splits a mixed Schema into its tabular sub-schema and text attributes."""
  tab_attrs: dict[str, domain.AttributeType] = {}
  text_attrs: dict[str, domain.FreeFormTextAttribute] = {}
  for col, attr in schema.items():
    if isinstance(attr, domain.FreeFormTextAttribute):
      text_attrs[col] = attr
    else:
      tab_attrs[col] = attr
  if not tab_attrs:
    raise ValueError('ACTG requires at least one tabular attribute.')
  if not text_attrs:
    raise ValueError('ACTG requires at least one FreeFormTextAttribute.')
  tab_schema = domain.Schema(tab_attrs, constraints=schema.constraints)
  return tab_schema, text_attrs


def _strip_code_fence(text: str) -> str:
  """Strips optional surrounding markdown code fences from a model response."""
  stripped = text.strip()
  if not stripped.startswith('```'):
    return stripped
  lines = stripped.splitlines()
  if len(lines) >= 2 and lines[-1].strip() == '```':
    return '\n'.join(lines[1:-1]).strip()
  return stripped


@dataclasses.dataclass(frozen=True)
class ACTGResult:
  """Result of running ``ACTGMechanism`` end-to-end.

  Attributes:
    synthetic_data: DataFrame with synthesized tabular and text columns.
    tabular_result: Result from the underlying ``TabularMechanism``.
    fine_tune_result: Result from ``DPFineTuner``, or None in zero-shot mode.
  """

  synthetic_data: pd.DataFrame
  tabular_result: data_generation_v3.DataGenerationResult
  fine_tune_result: dp_sft.FineTuneResult | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ACTGConfig(api.MechanismConfig):
  """Configuration for Attribute-Conditioned Text Generation (ACTG).

  Attributes:
    tabular_config: Configuration for synthesizing structured attributes.
    fine_tuner_config: Configuration for DP-SFT of the conditional text model.
    tabular_budget_fraction: Budget share for tabular stage (1.0 skips SFT).
    sampling_temperature: Sampling temperature for conditional text generation.
    sampling_batch_size: Per-step batch size for ``GemmaSampler``.
  """

  tabular_config: data_generation_v3.TabularConfig = (
      data_generation_v3.TabularConfig()
  )
  fine_tuner_config: dp_sft.DPFineTunerConfig = dp_sft.DPFineTunerConfig()
  tabular_budget_fraction: float = 0.125
  sampling_temperature: float = 1.0
  sampling_batch_size: int = 32

  def __post_init__(self):
    frac = self.tabular_budget_fraction
    if not (0.0 < frac <= 1.0):
      raise ValueError(f'tabular_budget_fraction must be in (0, 1]: {frac}.')
    if self.sampling_temperature < 0.0:
      raise ValueError(f'Negative temperature: {self.sampling_temperature}.')
    if self.sampling_batch_size <= 0:
      raise ValueError(f'Non-positive batch_size: {self.sampling_batch_size}.')

  def configure(
      self,
      schema: domain.Schema | Mapping[str, domain.AttributeType] | None = None,
      *,
      budget: float,
      delta: float = 0.0,
  ) -> ACTGMechanism:
    """Calibrates sub-mechanisms by splitting ``budget`` across stages."""
    if schema is None:
      raise ValueError('schema is required for ACTGConfig.configure().')
    if not isinstance(schema, domain.Schema):
      schema = domain.Schema(schema)
    tab_schema, text_attrs = _partition_schema(schema)

    tab_budget = self.tabular_budget_fraction * budget
    tab_cfg = self.tabular_config
    tab_mech = tab_cfg.configure(tab_schema, budget=tab_budget, delta=delta)
    fine_tuner = None
    if self.tabular_budget_fraction < 1.0:
      text_budget = (1.0 - self.tabular_budget_fraction) * budget
      fine_tuner = self.fine_tuner_config.configure(budget=text_budget)

    return ACTGMechanism(self, schema, tab_mech, fine_tuner, text_attrs)


@dataclasses.dataclass(frozen=True)
class ACTGMechanism(api.CalibratedMechanism):
  """Calibrated Attribute-Conditioned Text Generation (ACTG) mechanism.

  Attributes:
    config: The uncalibrated ``ACTGConfig``.
    schema: Full dataset schema containing both tabular and text attributes.
    tabular_mechanism: Calibrated ``TabularMechanism`` for structured columns.
    fine_tuner: Calibrated ``DPFineTuner``, or None in zero-shot mode.
    text_attributes: Mapping of text column names to ``FreeFormTextAttribute``.
  """

  config: ACTGConfig
  schema: domain.Schema
  tabular_mechanism: data_generation_v3.TabularMechanism
  fine_tuner: dp_sft.DPFineTuner | None
  text_attributes: Mapping[str, domain.FreeFormTextAttribute]

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed DpEvent across tabular synthesis and DP-SFT."""
    tab_event = self.tabular_mechanism.dp_event
    if self.fine_tuner is None:
      return tab_event
    return dp_accounting.ComposedDpEvent([tab_event, self.fine_tuner.dp_event])

  def format_prompt(self, row: Mapping[str, Any]) -> str:
    """Formats a record's tabular attributes into a conditioning prompt."""
    tab_cols = self.tabular_mechanism.schema
    attrs = {col: _json_scalar(row[col]) for col in tab_cols}
    attrs_json = json.dumps(attrs)
    if len(self.text_attributes) == 1:
      col, spec = next(iter(self.text_attributes.items()))
      parts = [f'Generate {col} matching the following attributes:']
      parts.append(attrs_json)
      if spec.description:
        parts.append(f'Description: {spec.description}')
      if spec.formatting:
        parts.append(f'Formatting: {spec.formatting}')
      return '\n'.join(parts)

    header = 'Generate synthetic text fields as JSON matching these attributes:'
    lines = [header, attrs_json, 'Fields:']
    for col, spec in self.text_attributes.items():
      details = [p for p in (spec.description, spec.formatting) if p]
      suffix = f": {' '.join(details)}" if details else ''
      lines.append(f'- {col}{suffix}')
    return '\n'.join(lines)

  def format_response(self, row: Mapping[str, Any]) -> str:
    """Formats a record's text attributes into an SFT target string."""
    if len(self.text_attributes) == 1:
      col = next(iter(self.text_attributes))
      return _str_val(row[col])
    payload = {col: _str_val(row[col]) for col in self.text_attributes}
    return json.dumps(payload)

  def format_example(self, row: Mapping[str, Any]) -> tuple[str, str]:
    """Formats a record into a ``(prompt, response)`` pair for DP-SFT."""
    return self.format_prompt(row), self.format_response(row)

  def parse_response(self, response: str) -> dict[str, str]:
    """Parses a sampled response string into a dict of text column values."""
    if len(self.text_attributes) == 1:
      col = next(iter(self.text_attributes))
      return {col: response.strip()}
    cleaned = _strip_code_fence(response)
    try:
      parsed = json.loads(cleaned)
    except json.JSONDecodeError:
      parsed = {}
    if not isinstance(parsed, Mapping):
      parsed = {}
    return {col: _str_val(parsed.get(col)) for col in self.text_attributes}

  def sample_text(
      self,
      trained_model: nnx.Module,
      tabular_data: pd.DataFrame,
      *,
      rng: np.random.Generator | int = 0,
  ) -> pd.DataFrame:
    """Samples text columns conditioned on ``tabular_data`` rows."""
    if isinstance(rng, np.random.Generator):
      rng = int(rng.integers(2**31))
    records = tabular_data.to_dict(orient='records')
    prompts = [self.format_prompt(row) for row in records]
    max_len = self.config.fine_tuner_config.max_seq_length
    temp = self.config.sampling_temperature
    sampler = model.GemmaSampler(
        trained_model, max_seq_length=max_len, temperature=temp
    )
    batch_size = self.config.sampling_batch_size
    responses = sampler(prompts, rng=int(rng), batch_size=batch_size)
    parsed_rows = [self.parse_response(resp) for resp in responses]
    out = tabular_data.copy()
    for col in self.text_attributes:
      out[col] = [r[col] for r in parsed_rows]
    ordered_cols = [col for col in self.schema if col in out.columns]
    return out[ordered_cols]

  def __call__(
      self,
      rng: np.random.Generator,
      data: pd.DataFrame,
      *,
      num_rows: int | None = None,
      metrics_fn: callbacks.MetricsFn | None = None,
  ) -> ACTGResult:
    """Runs tabular synthesis, conditional DP-SFT, and conditional sampling."""
    for col in self.schema:
      if col not in data.columns:
        raise ValueError(f'Missing column {col!r} in {list(data.columns)}.')

    tab_result = self.tabular_mechanism(rng, data, num_rows=num_rows)
    ft_cfg = self.config.fine_tuner_config
    if self.fine_tuner is not None:
      records = data.to_dict(orient='records')
      examples = [self.format_example(row) for row in records]
      ft_result = self.fine_tuner(rng, examples, metrics_fn=metrics_fn)
      gen_model = ft_result.model
    else:
      ft_result = None
      lora_cfg = model.LoraConfig(rank=ft_cfg.lora_rank)
      gen_model = model.load_gemma(ft_cfg.model, lora_cfg)

    tab_df = tab_result.synthetic_data
    synth_df = self.sample_text(gen_model, tab_df, rng=rng)
    return ACTGResult(synth_df, tab_result, ft_result)
