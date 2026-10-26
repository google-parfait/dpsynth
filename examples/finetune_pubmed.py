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

import json
import typing

from absl import app
from absl import flags
from absl import logging
import datasets
from dpsynth import training
from etils import epath
import jax
from jax_privacy import execution_plan
import numpy as np
import optax

_MODEL = flags.DEFINE_enum(
    'model',
    'gemma4_e2b_it',
    [
        'gemma4_e2b_it',
        'gemma4_e4b_it',
    ],
    'Gemma model variant to fine-tune.',
)
_WORKDIR = flags.DEFINE_string(
    'workdir', '', 'Directory to save the fine-tuned model to and load it from.'
)
_TRAIN = flags.DEFINE_bool('train', True, 'Whether to DP-fine-tune the model.')
_SAMPLE = flags.DEFINE_bool(
    'sample', True, 'Whether to generate synthetic abstracts.'
)
_ITERATIONS = flags.DEFINE_integer('iterations', 1000, 'Training steps.')
_MICROBATCH_SIZE = flags.DEFINE_integer(
    'microbatch_size',
    8,
    'Microbatch size for gradient clipping.',
    lower_bound=1,
)
_EPSILON = flags.DEFINE_float('epsilon', 8.0, 'Target DP epsilon.')
_DELTA = flags.DEFINE_float('delta', 1e-5, 'Target DP delta.')
_LORA_RANK = flags.DEFINE_integer('lora_rank', 16, 'LoRA rank.')
_MAX_SEQ_LENGTH = flags.DEFINE_integer('max_seq_length', 512, 'Max tokens.')
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 1e-4, 'AdamW learning rate.'
)
_NUM_SAMPLES = flags.DEFINE_integer('num_samples', 64, 'Abstracts to generate.')
_SAMPLE_BATCH_SIZE = flags.DEFINE_integer(
    'sample_batch_size',
    32,
    'Batch size for sampling across devices; must be divisible by the number of'
    ' available devices.',
    lower_bound=1,
)
_TEMPERATURE = flags.DEFINE_float('temperature', 1.0, 'Sampling temperature.')
_SEED = flags.DEFINE_integer(
    'seed', 0, 'RNG seed for sampling; change it to draw a different batch.'
)
_DATA_PATH = flags.DEFINE_string(
    'data_path',
    '',
    'Optional path to a JSON list of abstract strings. When empty, the '
    'abstracts are downloaded from the HuggingFace Hub instead.',
)

_INSTRUCTION = 'Write the abstract of a research paper.'
_DATASET = 'ccdv/pubmed-summarization'


def _model_name() -> training.ModelName:
  # ``_MODEL`` is a DEFINE_enum whose choices are a subset of ``ModelName``.
  return typing.cast(training.ModelName, _MODEL.value)


def _load_abstracts() -> list[str]:
  """Loads abstracts from --data_path (JSON) or the HF Hub."""
  if _DATA_PATH.value:
    with epath.Path(_DATA_PATH.value).open() as f:
      abstracts = json.load(f)
  else:
    data = datasets.load_dataset(
        _DATASET, split='train', trust_remote_code=True
    )
    abstracts = data['abstract']
  return [t.strip() for t in abstracts if t and t.strip()]


def train() -> training.FineTuneResult:
  """DP-fine-tunes Gemma on PubMed abstracts."""
  abstracts = _load_abstracts()
  # Frame each abstract as a (fixed instruction prompt -> abstract) SFT pair.
  train_data = [(_INSTRUCTION, abstract) for abstract in abstracts]

  fine_tuner = training.DPFineTunerConfig(
      model=_model_name(),
      mechanism_config=execution_plan.BandMFConfig.default(
          num_bands=8,
          iterations=_ITERATIONS.value,
          expected_participations=8.0,
      ),
      lora_rank=_LORA_RANK.value,
      max_seq_length=_MAX_SEQ_LENGTH.value,
      optimizer=optax.adamw(_LEARNING_RATE.value),
      microbatch_size=_MICROBATCH_SIZE.value,
  ).calibrate(epsilon=_EPSILON.value, delta=_DELTA.value)

  logging.info('DP fine-tuning on %d abstracts...', len(train_data))
  return fine_tuner(rng=np.random.default_rng(0), data=train_data)


def load_finetuned() -> training.FineTuneResult:
  """Rebuilds the model and loads previously saved fine-tuned params.

  Uses the same --model / --lora_rank / --max_seq_length as training, so the
  reconstructed architecture matches the saved parameters.

  Returns:
    A FineTuneResult holding the reconstructed model and loaded params.
  """
  loaded = training.load_gemma(
      _model_name(),
      training.LoraConfig(rank=_LORA_RANK.value),
      checkpoint_path=_ckpt_dir(),
  )
  return training.FineTuneResult(model=loaded)


def sample(result: training.FineTuneResult) -> None:
  """Generates synthetic abstracts and writes them to <workdir> as JSONL."""
  sampler = training.GemmaSampler(
      result.model,
      max_seq_length=_MAX_SEQ_LENGTH.value,
      temperature=_TEMPERATURE.value,
  )
  prompts = [_INSTRUCTION] * _NUM_SAMPLES.value
  responses = sampler(
      prompts,
      rng=_SEED.value,
      batch_size=_SAMPLE_BATCH_SIZE.value,
  )

  out_path = epath.Path(_WORKDIR.value) / 'synthetic_abstracts.jsonl'
  with out_path.open('w') as f:
    for i, abstract in enumerate(responses):
      if i % _SAMPLE_BATCH_SIZE.value == 0:
        logging.info('Synthetic abstract %d:\n%s', i + 1, abstract)
      f.write(json.dumps({'abstract': abstract}) + '\n')
  logging.info('Wrote %d abstracts to %s.', _NUM_SAMPLES.value, out_path)


def _ckpt_dir() -> epath.Path:
  """Path to store the final fine-tuned params."""
  return epath.Path(_WORKDIR.value) / 'params'


def main(_) -> None:
  if not _WORKDIR.value:
    raise app.UsageError('--workdir is required.')
  if not _TRAIN.value and not _SAMPLE.value:
    raise app.UsageError('Set at least one of --train / --sample.')

  if _TRAIN.value:
    result = train()
    result.save(_ckpt_dir())
    logging.info('Saved fine-tuned model to %s.', _ckpt_dir())
  else:
    result = load_finetuned()

  if _SAMPLE.value:
    if _SAMPLE_BATCH_SIZE.value % len(jax.devices()) != 0:
      raise app.UsageError(
          f'--sample_batch_size ({_SAMPLE_BATCH_SIZE.value}) must be divisible'
          f' by the number of available devices ({len(jax.devices())}).'
      )
    sample(result)


if __name__ == '__main__':
  app.run(main)
