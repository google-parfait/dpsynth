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

r"""Differentially private fine-tuning of Gemma on PubMed abstracts.

Fine-tunes a small Gemma model on real biomedical abstracts with DP-BandMF, then
samples *synthetic* abstracts from it. Every training example uses the same
fixed instruction prompt, so the model learns the unconditional distribution of
abstracts; we generate by feeding that prompt back with a non-zero temperature.

Training is the expensive, privacy-consuming step; sampling is free
post-processing. The --train / --sample flags let you train once and then draw
more synthetic data later without retraining. This example is for demonstration
purposes only: it's not super highly tuned, and is missing some of the bells
and whistles of a production job.
"""

from __future__ import annotations

import dataclasses
import json

from absl import logging
import datasets
import dpsynth
from dpsynth.text import dp_sft
from dpsynth.text import model
from etils import epath
from jax_privacy import execution_plan
import numpy as np
import simple_parsing

_INSTRUCTION = 'Write the abstract of a research paper.'
_DATASET = 'ccdv/pubmed-summarization'


@dataclasses.dataclass(frozen=True)
class Config:
  """Command-line configuration for DP fine-tuning and sampling of Gemma.

  Attributes:
    workdir: Directory to save the fine-tuned model to and load it from.
    dp_plan: Optional YAML path for a BandMFConfig execution plan.
    data_path: Optional path to a JSON list of abstract strings.
    model: Gemma model variant to fine-tune.
    train: Whether to DP-fine-tune the model.
    sample: Whether to generate synthetic abstracts.
    microbatch_size: Microbatch size for per-example gradient clipping.
    num_samples: Number of synthetic abstracts to generate.
    seed: RNG seed for sampling; change it to draw a different batch.
  """

  workdir: str
  dp_plan: str | None = None
  data_path: str = ''
  model: model.ModelName = 'gemma4_e2b_it'
  train: bool = True
  sample: bool = True
  microbatch_size: int = 8
  num_samples: int = 64
  seed: int = 0


def _load_abstracts(cfg: Config) -> list[str]:
  """Loads abstracts from --data_path (JSON) or the HF Hub."""
  if cfg.data_path:
    with epath.Path(cfg.data_path).open() as f:
      abstracts = json.load(f)
  else:
    data = datasets.load_dataset(
        _DATASET, split='train', trust_remote_code=True
    )
    abstracts = data['abstract']
  return [t.strip() for t in abstracts if t and t.strip()]


def train(cfg: Config) -> dp_sft.FineTuneResult:
  """DP-fine-tunes Gemma on PubMed abstracts."""
  abstracts = _load_abstracts(cfg)
  # Frame each abstract as a (fixed instruction prompt -> abstract) SFT pair.
  train_data = [(_INSTRUCTION, abstract) for abstract in abstracts]

  if cfg.dp_plan is not None:
    plan = dpsynth.from_yaml(cfg.dp_plan, execution_plan.BandMFConfig)
  else:
    plan = execution_plan.BandMFConfig.default(
        num_bands=8,
        iterations=1000,
        expected_participations=8.0,
    )

  fine_tuner = dp_sft.DPFineTunerConfig(
      model=cfg.model,
      mechanism_config=plan,
      microbatch_size=cfg.microbatch_size,
  ).calibrate(epsilon=8.0, delta=1e-5)
  logging.info('DP fine-tuning on %d abstracts...', len(train_data))
  return fine_tuner(rng=np.random.default_rng(0), data=train_data)


def sample(cfg: Config, result: dp_sft.FineTuneResult) -> None:
  """Generates synthetic abstracts and writes them to <workdir> as JSONL."""
  sampler = model.GemmaSampler(result.model)
  responses = sampler([_INSTRUCTION] * cfg.num_samples, rng=cfg.seed)

  out_path = epath.Path(cfg.workdir) / 'synthetic_abstracts.jsonl'
  with out_path.open('w') as f:
    for abstract in responses:
      f.write(json.dumps({'abstract': abstract}) + '\n')
  logging.info('Wrote %d abstracts to %s.', cfg.num_samples, out_path)


def main(cfg: Config) -> None:
  if not cfg.train and not cfg.sample:
    raise ValueError('Set at least one of --train / --sample.')

  ckpt_dir = epath.Path(cfg.workdir) / 'params'
  if cfg.train:
    result = train(cfg)
    result.save(ckpt_dir)
    logging.info('Saved fine-tuned model to %s.', ckpt_dir)
  else:
    loaded = model.load_gemma(cfg.model, checkpoint_path=ckpt_dir)
    result = dp_sft.FineTuneResult(model=loaded)

  if cfg.sample:
    sample(cfg, result)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  main(simple_parsing.parse(Config))
