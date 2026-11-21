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

"""Differentially private fine-tuning of Gemma on unstructured text.

Demonstrates generating synthetic free-form text by fine-tuning a Gemma model
with BandMF on private documents paired with a shared instruction prompt,
optionally inserting canary strings to empirically audit memorization alongside
the formal privacy guarantee. Because device sharding is handled automatically,
the same script runs without modification on a single GPU or across a large
cluster of GPUs or TPUs. Training is the privacy-consuming step while sampling
from the tuned weights is free post-processing, producing both a reusable model
checkpoint and synthetic text samples (using PubMed abstracts by default).
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json

from absl import app
from absl import flags
from absl import logging
import dpsynth
import dpsynth.training
from etils import epath
import jax
import numpy as np
from orbax import checkpoint as ocp
import simple_parsing


@dataclasses.dataclass(frozen=True)
class Config:
  """Command-line configuration for DP fine-tuning and sampling of Gemma.

  Attributes:
    data_path: Path to a JSON file containing a list of text strings.
    output_dir: Directory to save the DP-safe fine-tuned model and samples.
    instruction: Fixed instruction prompt paired with each training document.
    checkpoint_dir: Optional directory for mid-training checkpoints.
    run_id: Run identifier required when checkpoint_dir is set.
    canary_path: Optional path to a JSON file containing canary text strings.
    model: Gemma model variant to fine-tune.
    microbatch_size: Microbatch size for per-example gradient clipping.
    num_samples: Number of synthetic documents to generate.
    seed: RNG seed for training and sampling.
  """

  data_path: str = 'datasets/pubmed.json'
  output_dir: str = 'dpsynth_output'
  instruction: str = 'Write the abstract of a research paper.'
  checkpoint_dir: str | None = None
  run_id: str | None = None
  canary_path: str | None = None
  model: dpsynth.training.ModelName = 'gemma4_e2b_it'
  microbatch_size: int = 8
  num_samples: int = 64
  seed: int = 0


def _load_examples(path: str, instruction: str) -> list[tuple[str, str]]:
  """Loads a JSON list of texts and pairs each with the instruction prompt."""
  # epath.Path.open raises FileNotFoundError with the path if missing.
  with epath.Path(path).open() as f:
    texts = [t.strip() for t in json.load(f) if t and t.strip()]
  # Frame each document as a (fixed instruction prompt -> text) SFT pair.
  return [(instruction, text) for text in texts]


def main(argv: Sequence[str]) -> None:
  cfg = simple_parsing.parse(Config, args=argv[1:])

  train_data = _load_examples(cfg.data_path, cfg.instruction)
  canaries = None
  if cfg.canary_path is not None:
    canaries = _load_examples(cfg.canary_path, cfg.instruction)

  fine_tuner = dpsynth.training.DPFineTunerConfig(
      model=cfg.model,
      iterations=1000,
      epochs=8.0,
      num_bands=8,
      microbatch_size=cfg.microbatch_size,
  ).calibrate(epsilon=8.0, delta=1e-5)

  target_prompts = [cfg.instruction] * cfg.num_samples
  out_dir = epath.Path(cfg.output_dir)
  logging.info('DP fine-tuning on %d documents...', len(train_data))
  with (
      jax.set_mesh(dpsynth.training.make_mesh()),
      dpsynth.checkpoint(cfg.checkpoint_dir, cfg.run_id),
  ):
    result = fine_tuner(
        rng=np.random.default_rng(cfg.seed),
        data=train_data,
        canaries=canaries,
        target_prompts=target_prompts,
    )
    with ocp.StandardCheckpointer() as checkpointer:
      checkpointer.save(
          (out_dir / 'params').resolve(), result.model.params, force=True
      )

  if result.auditor is not None:
    audit_report = dpsynth.AuditReport.from_auditor(result.auditor)
    dpsynth.to_yaml(audit_report, out_dir / 'audit_report.yaml')

  assert result.samples is not None
  out_path = out_dir / 'synthetic_samples.jsonl'
  with out_path.open('w') as f:
    for _, text in result.samples:
      f.write(json.dumps({'text': text}) + '\n')
  logging.info('Wrote %d samples to %s.', cfg.num_samples, out_path)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda a: flags.FLAGS(a, known_only=True))
