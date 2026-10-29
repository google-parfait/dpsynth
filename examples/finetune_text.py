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

r"""Differentially private fine-tuning of Gemma on text datasets.

Fine-tunes a small Gemma model on real text documents with DP-BandMF, then
samples *synthetic* documents from it. The input ``--data_path`` must be a JSON
file containing a list of text strings (``["doc 1", "doc 2", ...]``). Every
training example pairs a document with the same fixed instruction prompt so the
model learns the unconditional distribution of documents; we generate by feeding
that prompt back with a non-zero temperature.

Training is the expensive, privacy-consuming step; sampling is free
post-processing. This example is for demonstration purposes only: it's not super
highly tuned, and is missing some of the bells and whistles of a production job.

Example usage::

  python examples/finetune_text.py \
      --data_path=/path/to/pubmed.json \
      --output_dir=/path/to/output \
      --instruction="Write the abstract of a research paper."
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json

from absl import app
from absl import flags
from absl import logging
import dpsynth
import dpsynth.text
from etils import epath
import jax
from jax_privacy import execution_plan
import numpy as np
import simple_parsing


@dataclasses.dataclass(frozen=True)
class Config:
  """Command-line configuration for DP fine-tuning and sampling of Gemma.

  Attributes:
    data_path: Path to a JSON file containing a list of text strings.
    output_dir: Directory to save the DP-safe fine-tuned model and samples.
    instruction: Fixed instruction prompt paired with each training document.
    checkpoint_dir: Optional directory for mid-training checkpoints.
    dp_plan: Optional YAML path for a BandMFConfig execution plan.
    model: Gemma model variant to fine-tune.
    microbatch_size: Microbatch size for per-example gradient clipping.
    num_samples: Number of synthetic documents to generate.
    seed: RNG seed for training and sampling.
  """

  data_path: str
  output_dir: str
  instruction: str
  checkpoint_dir: str | None = None
  dp_plan: str | None = None
  model: dpsynth.text.ModelName = 'gemma4_e2b_it'
  microbatch_size: int = 8
  num_samples: int = 64
  seed: int = 0


def main(argv: Sequence[str]) -> None:
  cfg = simple_parsing.parse(Config, args=argv[1:])

  with epath.Path(cfg.data_path).open() as f:
    texts = [t.strip() for t in json.load(f) if t and t.strip()]
  # Frame each document as a (fixed instruction prompt -> text) SFT pair.
  train_data = [(cfg.instruction, text) for text in texts]

  if cfg.dp_plan is not None:
    plan = dpsynth.from_yaml(cfg.dp_plan, execution_plan.BandMFConfig)
  else:
    plan = execution_plan.BandMFConfig.default(
        num_bands=8,
        iterations=1000,
        expected_participations=8.0,
    )

  fine_tuner = dpsynth.text.DPFineTunerConfig(
      model=cfg.model,
      mechanism_config=plan,
      microbatch_size=cfg.microbatch_size,
  ).calibrate(epsilon=8.0, delta=1e-5)

  target_prompts = [cfg.instruction] * cfg.num_samples
  out_dir = epath.Path(cfg.output_dir)
  logging.info('DP fine-tuning on %d documents...', len(train_data))
  with jax.set_mesh(dpsynth.text.make_mesh()):
    with dpsynth.checkpoint(cfg.checkpoint_dir):
      result = fine_tuner(
          rng=np.random.default_rng(cfg.seed),
          data=train_data,
          target_prompts=target_prompts,
      )
    result.save(out_dir / 'params')

  assert result.samples is not None
  out_path = out_dir / 'synthetic_samples.jsonl'
  with out_path.open('w') as f:
    for _, text in result.samples:
      f.write(json.dumps({'text': text}) + '\n')
  logging.info('Wrote %d samples to %s.', cfg.num_samples, out_path)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda a: flags.FLAGS(a, known_only=True))
