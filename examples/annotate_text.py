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

"""Structured feature annotation of unstructured text via the Pydantic adapter.

Demonstrates how to turn unstructured user-LLM conversations from the WildChat
dataset into a structured tabular dataset and companion DPSynth domain schema.
Given a Pydantic model defining the 20 conversation features of interest, the
Pydantic adapter automatically infers the corresponding DPSynth domain YAML, and
each conversation is annotated into a Pydantic instance via the Gemini API and
saved as a Parquet table ready for ``tabular_synthesis.py``.
"""

from __future__ import annotations

from concurrent import futures
import dataclasses
import functools
import time
from typing import Literal

from absl import logging
import dpsynth
from dpsynth.adapters import pydantic as pydantic_adapter
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
import pandas as pd
import pydantic
import simple_parsing

_SYSTEM_INSTRUCTION = (
    'Analyze the provided user-LLM conversation and extract each requested '
    'feature from the predefined categories.'
)

MainTopic = Literal[
    'Technology',
    'Science',
    'Mathematics',
    'History',
    'Literature',
    'Arts & Culture',
    'Business & Finance',
    'Health & Medicine',
    'Programming & Software Development',
    'Education & Learning',
    'Travel & Geography',
    'Food & Cooking',
    'Sports & Recreation',
    'Personal Advice',
    'General Knowledge Query',
    'Creative Input Request',
    'Symbolism & Interpretation',
    'Ambiguous/Unclear',
    'Other',
]
ProgrammingSubTopic = Literal[
    'Python',
    'JavaScript',
    'Java',
    'C++',
    'SQL',
    'Web Development',
    'Mobile Development',
    'Data Science/ML',
    'Algorithms',
    'DevOps',
    'Game Development',
    'Cybersecurity',
    'Code Completion',
    'Code Debugging',
    'Other Programming',
    'Not Applicable',
]
UserIntent = Literal[
    'Factual Q&A',
    'Summarization',
    'Creative Writing/Generation',
    'Brainstorming/Ideation',
    'Code Generation',
    'Translation',
    'Rewriting/Editing',
    'Mathematical Calculation',
    'Role-playing',
    'Instruction Following',
    'Advice Seeking',
    'Image/Media Request',
    'General Conversation',
    'Other',
]
TaskComplexity = Literal['Low', 'Medium', 'High', 'Very High']
SpecificityLevel = Literal[
    'Vague/Ambiguous', 'General', 'Specific', 'Highly Specific & Constrained'
]
PromptFormat = Literal[
    'Prose',
    'Direct Question',
    'List / Bullet Points',
    'Code Block',
    'Instructional Steps',
    'Mixed Format',
    'Other',
]
Language = Literal[
    'English',
    'Spanish',
    'Russian',
    'French',
    'German',
    'Chinese',
    'Japanese',
    'Code-Only',
    'Mixed Language',
    'Other',
]
JargonUsage = Literal['None', 'Low', 'Medium', 'High']
ConstraintPresence = Literal[
    'No Constraints',
    'Implicit Constraints',
    'Explicit Positive Constraints',
    'Explicit Negative Constraints',
    'Complex/Layered Constraints',
]
CreativityLevel = Literal['None', 'Low', 'Medium', 'High']
Tone = Literal[
    'Neutral/Formal',
    'Conversational/Informal',
    'Humorous/Playful',
    'Demanding/Authoritative',
    'Frustrated/Annoyed',
    'Curious/Inquisitive',
    'Urgent',
]
PersonaAdoption = Literal[
    'No Persona Requested',
    'Implicit Persona',
    'Expert Persona',
    'Fictional Character',
    'Other Defined Role',
]
ConversationDepth = Literal[
    'Single Turn', 'Short (2-3 turns)', 'Medium (4-6 turns)', 'Long (7+ turns)'
]
TopicConsistency = Literal[
    'Consistent Topic',
    'Topic Refinement/Clarification',
    'Topic Shift (Natural Progression)',
    'Topic Abandonment (New Topic)',
    'Not Applicable (Single Turn)',
]
AssistantStance = Literal[
    'Fully Cooperative & Direct',
    'Enthusiastic & Creative',
    'Helpful Refusal (Capability)',
    'Helpful Refusal (Policy/Safety)',
    'Seeking Clarification',
    'Neutral & Factual',
]
RefusalType = Literal[
    'No Refusal',
    'Capability Limitation',
    'Safety/Policy Violation',
    'Ethical/Legal Constraint',
    'Request Ambiguity',
    'Other',
]
SafetyRisk = Literal['None', 'Low', 'Medium', 'High']
Groundedness = Literal[
    'Not Required',
    'Partially Required',
    'Strictly Required',
    'Logically Grounded',
]
UserExpertise = Literal['Novice', 'Intermediate', 'Expert', 'Undeterminable']
ResponseFormatRequested = Literal[
    'No Format Specified',
    'List',
    'Table',
    'Code Block',
    'JSON',
    'Specific Length',
    'Specific Style',
    'Other Specified Format',
]


class ConversationFeatures(pydantic.BaseModel):
  """Structured features extracted from a user-LLM conversation."""

  model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

  main_topic: MainTopic
  """The main topic of the conversation."""

  programming_sub_topic: ProgrammingSubTopic
  """Specific programming sub-topic if applicable, otherwise 'Not Applicable'."""

  user_intent: UserIntent
  """Primary goal the user wants to accomplish."""

  task_complexity: TaskComplexity
  """Difficulty and effort required for the assistant to fulfill the request."""

  specificity_level: SpecificityLevel
  """How specific and detailed the instructions are in the user's prompt."""

  prompt_format: PromptFormat
  """Basic structural format of the user's prompt."""

  language: Language
  """Primary natural language used in the conversation."""

  jargon_usage: JargonUsage
  """Extent to which the prompt uses specialized domain-specific terminology."""

  constraint_presence: ConstraintPresence
  """Whether the user provides explicit rules or negative constraints."""

  creativity_level: CreativityLevel
  """How much novel, creative generation is required for the response."""

  tone: Tone
  """Overall tone of the user's prompt."""

  persona_adoption: PersonaAdoption
  """Whether the prompt asks the assistant to adopt a specific persona."""

  conversation_depth: ConversationDepth
  """Number of turns in the conversation."""

  topic_consistency: TopicConsistency
  """How the topic evolves across turns ('Not Applicable (Single Turn)' if 1)."""

  assistant_stance: AssistantStance
  """Primary stance the assistant takes in its responses."""

  refusal_type: RefusalType
  """Primary reason if the assistant refuses or pushes back."""

  safety_risk: SafetyRisk
  """Potential for the conversation to involve sensitive or harmful topics."""

  groundedness: Groundedness
  """Whether the response requires grounding in verifiable factual information."""

  user_expertise: UserExpertise
  """User's apparent level of expertise based on their phrasing and questions."""

  response_format_requested: ResponseFormatRequested
  """Whether the user explicitly asks for a specific output format."""


_GENERATE_CONFIG = genai_types.GenerateContentConfig(
    system_instruction=_SYSTEM_INSTRUCTION,
    response_mime_type='application/json',
    response_schema=ConversationFeatures,
    temperature=0.0,
)


@dataclasses.dataclass(frozen=True)
class Config:
  """Command-line configuration for structured text annotation.

  Attributes:
    input_path: Path to input Parquet file containing a text column.
    output_path: Path where the annotated feature Parquet table is written.
    domain_path: Path where the inferred DPSynth domain YAML is written.
    gemini_model: Model name used for per-record structured annotation.
    max_workers: Number of concurrent worker threads for Gemini API requests.
    limit: Optional maximum number of conversations to annotate.
  """

  input_path: str = 'datasets/wildchat.parquet'
  output_path: str = 'datasets/wildchat_features.parquet'
  domain_path: str = 'datasets/wildchat_domain.yaml'
  gemini_model: str = 'gemini-3.5-flash-lite'
  max_workers: int = 50
  limit: int | None = None


def _annotate(
    client: genai.Client, model: str, text: str
) -> ConversationFeatures | None:
  """Extracts structured conversation features from a single conversation."""
  for attempt in range(8):
    try:
      resp = client.models.generate_content(
          model=model, contents=text[:32_000], config=_GENERATE_CONFIG
      )
      if not resp.text:
        return None
      return ConversationFeatures.model_validate_json(resp.text)
    except (genai_errors.APIError, pydantic.ValidationError):
      time.sleep(2.0**attempt)
  return None


def main(cfg: Config) -> None:
  """Exports the inferred domain YAML and annotates conversations to Parquet."""
  logging.set_verbosity(logging.INFO)

  # Infer the DPSynth domain from the Pydantic model and save the domain YAML.
  domains = pydantic_adapter.infer_domain(ConversationFeatures)
  dpsynth.to_yaml(domains, cfg.domain_path)

  # Load conversations and extract structured features concurrently.
  texts = pd.read_parquet(cfg.input_path)['text'].tolist()[: cfg.limit]
  client = genai.Client()
  annotate_fn = functools.partial(_annotate, client, cfg.gemini_model)
  with futures.ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
    annotated = list(pool.map(annotate_fn, texts))

  # Drop conversations blocked by API safety filters while preserving the
  # original row index so the feature table stays aligned with the raw text.
  rows = {
      i: pydantic_adapter.to_tuple(r, schema=domains)
      for i, r in enumerate(annotated)
      if r is not None
  }
  out_df = pd.DataFrame.from_dict(rows, orient='index', columns=list(domains))
  out_df.to_parquet(cfg.output_path)
  logging.info('Wrote %d annotated rows to %s', len(out_df), cfg.output_path)


if __name__ == '__main__':
  main(simple_parsing.parse(Config))
