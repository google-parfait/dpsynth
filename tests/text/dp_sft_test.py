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

"""Tests for dpsynth.text.model and dpsynth.text.dp_sft.

These tests verify the loss function, configuration, and calibration logic
without loading any real model checkpoints.
"""

import dataclasses
import math

from absl.testing import absltest
from dpsynth.text import dp_sft
from dpsynth.text import model
from flax import nnx
import jax
import jax.numpy as jnp
from jax_privacy import execution_plan


def _default_config():
  return execution_plan.BandMFConfig.default(
      num_bands=1,
      iterations=100,
      expected_participations=1.0,
  )


def _make_tiny_model(vocab_size=8, embed_dim=4):
  """Creates a minimal Flax NNX model for testing."""

  class TinyModel(nnx.Module):

    def __init__(self, rngs: nnx.Rngs):
      self.embed = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
      self.dense = nnx.Linear(embed_dim, vocab_size, rngs=rngs)

    def __call__(self, tokens, *args):
      del args
      return self.dense(self.embed(tokens)), None

  return TinyModel(nnx.Rngs(0))


class SftLossFnTest(absltest.TestCase):

  def test_loss_is_finite_and_positive(self):
    module = _make_tiny_model()
    data = {
        'input_tokens': jnp.array([[1, 2, 3, 4, 0]], dtype=jnp.int32),
        'loss_mask': jnp.array([[1, 1, 1, 1, 0]], dtype=jnp.int32),
    }
    loss, aux = model.sft_loss_fn(module, data)
    self.assertTrue(jnp.isfinite(loss))
    self.assertGreater(float(loss), 0.0)
    self.assertIn('loss', aux)

  def test_loss_decreases_with_gradient_descent(self):
    """Gradient descent on a tiny model should reduce loss."""
    module = _make_tiny_model()
    tokens = jnp.array([[1, 2, 3, 1]], dtype=jnp.int32)
    mask = jnp.ones_like(tokens)
    data = {'input_tokens': tokens, 'loss_mask': mask}

    loss_before = float(model.sft_loss_fn(module, data)[0])

    # Overfit on this single example.
    graphdef, params = nnx.split(module)
    for _ in range(200):

      def step_fn(p):
        return model.sft_loss_fn(nnx.merge(graphdef, p), data)

      (_, _), grads = jax.value_and_grad(step_fn, has_aux=True)(params)
      params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)

    loss_after = float(model.sft_loss_fn(nnx.merge(graphdef, params), data)[0])
    self.assertLess(loss_after, loss_before)

  def test_masked_tokens_do_not_contribute(self):
    module = _make_tiny_model()
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    mask_short = jnp.array([[1, 1, 0, 0]], dtype=jnp.int32)
    mask_full = jnp.array([[1, 1, 1, 1]], dtype=jnp.int32)

    loss_short = float(
        model.sft_loss_fn(
            module, {'input_tokens': tokens, 'loss_mask': mask_short}
        )[0]
    )
    loss_full = float(
        model.sft_loss_fn(
            module, {'input_tokens': tokens, 'loss_mask': mask_full}
        )[0]
    )
    # Different masks should give different losses.
    self.assertNotAlmostEqual(loss_short, loss_full, places=3)


class GemmaModelTest(absltest.TestCase):

  def test_default_preset(self):
    m = model.GemmaModel.default('gemma4_e2b_it')
    self.assertEqual(m.name, 'gemma4_e2b_it')
    self.assertIsNotNone(m.checkpoint_path)

  def test_gemma4_31b_preset(self):
    m = model.GemmaModel.default('gemma4_31b_it')
    self.assertEqual(m.name, 'gemma4_31b_it')
    self.assertIsNotNone(m.checkpoint_path)

  def test_make_mesh(self):
    mesh = model.make_mesh(tp=1)
    self.assertEqual(mesh.axis_names, ('replica', 'fsdp', 'tp'))
    self.assertEqual(mesh.shape['fsdp'], 1)
    self.assertEqual(mesh.shape['tp'], 1)

  def test_custom_checkpoint_path(self):
    m = dataclasses.replace(
        model.GemmaModel.default('gemma4_e2b_it'),
        checkpoint_path='/custom/path',
    )
    self.assertEqual(m.checkpoint_path, '/custom/path')

  def test_unknown_name_raises(self):
    with self.assertRaises(ValueError):
      model.GemmaModel.default('nonexistent')


class LoraConfigTest(absltest.TestCase):

  def test_defaults(self):
    cfg = model.LoraConfig()
    self.assertEqual(cfg.rank, 16)
    self.assertIsNone(cfg.alpha)
    self.assertIsNone(cfg.weight_qtype)

  def test_custom_values(self):
    cfg = model.LoraConfig(rank=8, alpha=16.0, targets='.*q_einsum')
    self.assertEqual(cfg.rank, 8)
    self.assertEqual(cfg.alpha, 16.0)
    self.assertEqual(cfg.targets, '.*q_einsum')


class DPFineTunerTest(absltest.TestCase):

  def test_creates_valid_mechanism_config(self):
    mechanism = dp_sft.DPFineTuner(
        model_variant=model.GemmaModel.default('gemma4_e2b_it'),
        mechanism_config=_default_config(),
    )
    self.assertEqual(mechanism.mechanism_config.iterations, 100)
    self.assertIsNone(mechanism.mechanism_config.noise_multiplier)

  def test_calibrate_sets_noise_multiplier(self):
    mechanism = dp_sft.DPFineTuner(
        model_variant=model.GemmaModel.default('gemma4_e2b_it'),
        mechanism_config=_default_config(),
    ).configure(budget=0.5)
    self.assertIsNotNone(mechanism.mechanism_config.noise_multiplier)
    self.assertGreater(mechanism.mechanism_config.noise_multiplier, 0.0)

  def test_calibrate_noise_formula(self):
    mechanism = dp_sft.DPFineTuner(
        model_variant=model.GemmaModel.default('gemma4_e2b_it'),
        mechanism_config=_default_config(),
    ).configure(budget=0.5)
    # Single band: rounds = iterations, sigma = sqrt(T / (2*rho)).
    expected = math.sqrt(100 / (2.0 * 0.5))
    self.assertAlmostEqual(
        mechanism.mechanism_config.noise_multiplier, expected
    )

  def test_dp_event_before_calibration_raises(self):
    mechanism = dp_sft.DPFineTuner(
        model_variant=model.GemmaModel.default('gemma4_e2b_it'),
        mechanism_config=_default_config(),
    )
    with self.assertRaises(ValueError):
      _ = mechanism.dp_event

  def test_dp_event_after_calibration(self):
    mechanism = dp_sft.DPFineTuner(
        model_variant=model.GemmaModel.default('gemma4_e2b_it'),
        mechanism_config=_default_config(),
    ).configure(budget=0.5)
    event = mechanism.dp_event
    self.assertIsNotNone(event)


if __name__ == '__main__':
  absltest.main()
