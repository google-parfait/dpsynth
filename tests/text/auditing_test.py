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

from unittest import mock

from absl.testing import absltest
import dpsynth
from dpsynth.text import auditing
from dpsynth.text import model
from flax import nnx
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np


class _TinyLoraModel(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.embed = nnx.Embed(16, 8, rngs=rngs)
    self.lora = nnx.LoRA(
        8, 4, 16, base_module=nnx.Linear(8, 16, rngs=rngs), rngs=rngs
    )

  def __call__(self, tokens, *args):
    del args
    return self.lora(self.embed(tokens)), None


def _fake_tokenize(
    example: tuple[str, str], max_seq_length: int = 8
) -> dict[str, np.ndarray]:
  prompt, response = example
  tokens = np.zeros((max_seq_length,), dtype=np.int32)
  mask = np.zeros((max_seq_length,), dtype=np.int32)
  p_toks = [1] + [(ord(c) % 14) + 2 for c in prompt[:2]]
  r_toks = [(ord(c) % 14) + 2 for c in response[:3]]
  all_toks = (p_toks + r_toks)[:max_seq_length]
  tokens[: len(all_toks)] = all_toks
  mask[len(p_toks) : len(all_toks)] = 1
  return {'input_tokens': tokens, 'loss_mask': mask}


class AuditingTest(absltest.TestCase):

  def test_mix_canaries_sequence(self):
    train = [('p1', 'r1'), ('p2', 'r2')]
    canaries = [('c1', 'cr1')]
    mixed = auditing.mix_canaries(train, canaries, repetitions=3)
    self.assertEqual(
        mixed,
        [
            ('p1', 'r1'),
            ('p2', 'r2'),
            ('c1', 'cr1'),
            ('c1', 'cr1'),
            ('c1', 'cr1'),
        ],
    )

  def test_mix_canaries_map_dataset(self):
    train = grain.MapDataset.source([('p1', 'r1'), ('p2', 'r2')])
    canaries = [('c1', 'cr1'), ('c2', 'cr2')]
    mixed = auditing.mix_canaries(train, canaries, repetitions=2)
    self.assertIsInstance(mixed, grain.MapDataset)
    self.assertLen(mixed, 6)
    self.assertEqual(
        list(mixed),
        [
            ('p1', 'r1'),
            ('p2', 'r2'),
            ('c1', 'cr1'),
            ('c2', 'cr2'),
            ('c1', 'cr1'),
            ('c2', 'cr2'),
        ],
    )

  def test_mix_canaries_invalid_repetitions_raises(self):
    with self.assertRaises(ValueError):
      auditing.mix_canaries([('p', 'r')], [('c', 'r')], repetitions=0)

  def test_score_canaries_zero_at_init_and_positive_after_training(self):
    tiny_model = _TinyLoraModel(nnx.Rngs(0))
    held_in = [('in1', 'abc'), ('in2', 'def'), ('in3', 'ghi')]
    held_out = [('out1', 'xyz'), ('out2', 'uvw'), ('out3', 'rst')]

    with mock.patch.object(
        model, 'tokenize_example', side_effect=_fake_tokenize
    ):
      init_scores = auditing.score_canaries(
          tiny_model, held_in, max_seq_length=8, batch_size=2
      )
      np.testing.assert_allclose(init_scores, np.zeros(3), atol=1e-6)

      # Train LoRA weights to memorize held_in examples.
      records = [_fake_tokenize(ex, max_seq_length=8) for ex in held_in]
      batch = jax.tree.map(lambda *xs: jnp.stack(xs), *records)
      graphdef, lora_p, base_p = nnx.split(tiny_model, nnx.LoRAParam, ...)

      def loss_fn(lp):
        return model.sft_loss_fn(nnx.merge(graphdef, lp, base_p), batch)[0]

      grad_fn = jax.grad(loss_fn)
      for _ in range(80):
        grads = grad_fn(lora_p)
        lora_p = jax.tree.map(lambda p, g: p - 0.2 * g, lora_p, grads)
      trained_model = nnx.merge(graphdef, lora_p, base_p)

      in_scores = auditing.score_canaries(
          trained_model, held_in, max_seq_length=8, batch_size=2
      )
      out_scores = auditing.score_canaries(
          trained_model,
          grain.MapDataset.source(held_out),
          max_seq_length=8,
          batch_size=2,
      )
      unbatched = auditing.score_canaries(
          trained_model, held_in, max_seq_length=8, batch_size=10
      )

    np.testing.assert_allclose(in_scores, unbatched, rtol=1e-5, atol=1e-5)
    self.assertTrue(np.all(in_scores > 0.0))
    self.assertGreater(float(np.mean(in_scores)), float(np.mean(out_scores)))

  def test_score_canaries_empty_and_invalid_batch_size(self):
    tiny_model = _TinyLoraModel(nnx.Rngs(0))
    with mock.patch.object(
        model, 'tokenize_example', side_effect=_fake_tokenize
    ):
      empty = auditing.score_canaries(tiny_model, [], max_seq_length=8)
      self.assertEqual(empty.shape, (0,))
      with self.assertRaises(ValueError):
        auditing.score_canaries(
            tiny_model, [('a', 'b')], max_seq_length=8, batch_size=0
        )

  def test_audit_scores_and_yaml_roundtrip(self):
    rng = np.random.default_rng(0)
    in_scores = rng.normal(loc=3.0, scale=0.5, size=200)
    out_scores = rng.normal(loc=0.0, scale=0.5, size=200)
    res = auditing.audit_scores(
        in_scores, out_scores, delta=1e-5, significance=0.05, min_count=10
    )
    self.assertGreater(res.auroc, 0.99)
    self.assertGreater(res.max_accuracy, 0.95)
    self.assertGreater(res.epsilon_one_run_fdp, 0.5)
    self.assertGreater(res.epsilon_one_run, 0.5)
    self.assertGreater(res.epsilon_clopper_pearson, 0.5)
    self.assertGreater(res.epsilon_raw_counts, 0.5)
    self.assertEqual(res.num_held_in, 200)
    self.assertEqual(res.num_held_out, 200)

    yaml_str = dpsynth.to_yaml(res)
    restored = dpsynth.from_yaml(yaml_str, auditing.EmpiricalAuditResult)
    self.assertEqual(restored, res)

  def test_audit_canaries_end_to_end(self):
    tiny_model = _TinyLoraModel(nnx.Rngs(0))
    held_in = [('in1', 'abc'), ('in2', 'def'), ('in3', 'ghi'), ('in4', 'jkl')]
    held_out = [
        ('out1', 'xyz'),
        ('out2', 'uvw'),
        ('out3', 'rst'),
        ('out4', 'opq'),
    ]
    with mock.patch.object(
        model, 'tokenize_example', side_effect=_fake_tokenize
    ):
      res = auditing.audit_canaries(
          tiny_model,
          held_in,
          held_out,
          delta=1e-5,
          min_count=1,
          max_seq_length=8,
          batch_size=2,
      )
    self.assertEqual(res.num_held_in, 4)
    self.assertEqual(res.num_held_out, 4)
    self.assertAlmostEqual(res.auroc, 0.5, places=5)


if __name__ == '__main__':
  absltest.main()
