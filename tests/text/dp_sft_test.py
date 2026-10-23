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
import typing
from unittest import mock

from absl.testing import absltest
from dpsynth.text import dp_sft
from dpsynth.text import model
from flax import nnx
import grain.python as grain
import jax
import jax.numpy as jnp
import jax_privacy
from jax_privacy import execution_plan
import numpy as np


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

    def __call__(self, tokens, *args, **kwargs):
      del args, kwargs
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

  def test_validate_model_name_presets(self):
    for name in typing.get_args(model.ModelName):
      model.validate_model_name(name)

  def test_make_mesh(self):
    mesh = model.make_mesh(tp=1)
    self.assertEqual(mesh.axis_names, ('replica', 'fsdp', 'tp'))
    self.assertEqual(mesh.shape['fsdp'], 1)
    self.assertEqual(mesh.shape['tp'], 1)

  def test_make_mesh_multislice(self):
    cpu_dev = jax.devices()[0]
    fake_devices = [
        mock.Mock(wraps=cpu_dev, slice_index=i // 4) for i in range(8)
    ]
    hybrid_grid = model.np.full((4, 1, 2), cpu_dev, dtype=object)
    with (
        mock.patch.object(jax, 'device_count', return_value=8),
        mock.patch.object(jax, 'devices', return_value=fake_devices),
        mock.patch.object(
            model.mesh_utils,
            'create_hybrid_device_mesh',
            return_value=hybrid_grid,
        ) as mock_hybrid,
    ):
      mesh = model.make_mesh(tp=2)
      with self.assertRaises(ValueError):
        model.make_mesh(tp=8)
    mock_hybrid.assert_called_once_with(
        mesh_shape=(2, 1, 2),
        dcn_mesh_shape=(2, 1, 1),
        devices=fake_devices,
        allow_split_physical_axes=True,
    )
    self.assertEqual(mesh.axis_names, ('replica', 'fsdp', 'tp'))
    self.assertEqual(mesh.shape, {'replica': 4, 'fsdp': 1, 'tp': 2})

  def test_unknown_name_raises(self):
    with self.assertRaises(ValueError):
      model.validate_model_name(typing.cast(model.ModelName, 'nonexistent'))
    with self.assertRaises(ValueError):
      model.load_gemma(typing.cast(model.ModelName, 'nonexistent'))


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

  def test_save_and_load_lora_params(self):
    class _TinyLora(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.lora = nnx.LoRA(
            4, 2, 8, base_module=nnx.Linear(4, 8, rngs=rngs), rngs=rngs
        )

    m1 = _TinyLora(nnx.Rngs(0))
    m1.lora.lora_b.value = jnp.ones_like(m1.lora.lora_b.value)
    ckpt_dir = self.create_tempdir().full_path + '/lora_ckpt'
    dp_sft.FineTuneResult(model=m1).save(ckpt_dir)

    m2 = _TinyLora(nnx.Rngs(1))
    model.load_lora_params(m2, ckpt_dir)
    self.assertTrue(jnp.allclose(m1.lora.lora_b.value, m2.lora.lora_b.value))


class DPFineTunerTest(absltest.TestCase):

  def test_creates_valid_mechanism_config(self):
    config = dp_sft.DPFineTunerConfig(
        model='gemma4_e2b_it',
        mechanism_config=_default_config(),
    )
    self.assertEqual(config.model, 'gemma4_e2b_it')
    self.assertEqual(config.learning_rate, 1e-4)
    self.assertEqual(config.mechanism_config.iterations, 100)
    self.assertIsNone(config.mechanism_config.noise_multiplier)
    with self.assertRaises(ValueError):
      dataclasses.replace(config, learning_rate=0.0)
    with self.assertRaises(ValueError):
      dataclasses.replace(config, log_every=3, checkpoint_every=10)
    with self.assertRaises(ValueError):
      dp_sft.DPFineTunerConfig(
          model=typing.cast(model.ModelName, 'nonexistent'),
          mechanism_config=_default_config(),
      )

  def test_calibrate_sets_noise_multiplier(self):
    mechanism = dp_sft.DPFineTunerConfig(
        model='gemma4_e2b_it',
        mechanism_config=_default_config(),
    ).configure(budget=0.5)
    self.assertIsNotNone(mechanism.config.mechanism_config.noise_multiplier)
    self.assertGreater(mechanism.config.mechanism_config.noise_multiplier, 0.0)

  def test_calibrate_noise_formula(self):
    mechanism = dp_sft.DPFineTunerConfig(
        model='gemma4_e2b_it',
        mechanism_config=_default_config(),
    ).configure(budget=0.5)
    # Single band: rounds = iterations, sigma = sqrt(T / (2*rho)).
    expected = math.sqrt(100 / (2.0 * 0.5))
    sigma = mechanism.config.mechanism_config.noise_multiplier
    self.assertAlmostEqual(sigma, expected)

  def test_dp_event_after_calibration(self):
    mechanism = dp_sft.DPFineTunerConfig(
        model='gemma4_e2b_it',
        mechanism_config=_default_config(),
    ).configure(budget=0.5)
    event = mechanism.dp_event
    self.assertIsNotNone(event)

  def test_map_dataset_lazy_tokenization(self):
    class _TinyLoraModel(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.embed = nnx.Embed(8, 4, rngs=rngs)
        self.lora = nnx.LoRA(
            4, 2, 8, base_module=nnx.Linear(4, 8, rngs=rngs), rngs=rngs
        )

      def __call__(self, tokens, *args, **kwargs):
        del args, kwargs
        return self.lora(self.embed(tokens)), None

    tiny_model = _TinyLoraModel(nnx.Rngs(0))
    fake_example = {
        'input_tokens': jnp.array([1, 2, 3, 4], dtype=jnp.int32),
        'loss_mask': jnp.array([0, 1, 1, 1], dtype=jnp.int32),
    }
    config = execution_plan.BandMFConfig.default(
        num_bands=1,
        iterations=2,
        expected_participations=1.0,
    )
    mechanism = dp_sft.DPFineTunerConfig(
        model='gemma4_e2b_it',
        mechanism_config=config,
    ).configure(budget=1.0)
    orig_fit = jax_privacy.training.DPTrainer.fit

    def fit_spy(self_trainer, batch_data, *args, **kwargs):
      self.assertIsInstance(batch_data, grain.MapDataset)
      mock_tok.assert_not_called()
      return orig_fit(self_trainer, batch_data, *args, **kwargs)

    ds = grain.MapDataset.source([('p1', 'r1'), ('p2', 'r2')])
    with (
        mock.patch.object(
            model, 'load_gemma', return_value=tiny_model
        ) as mock_load,
        mock.patch.object(
            model, 'tokenize_example', return_value=fake_example
        ) as mock_tok,
        mock.patch.object(
            jax_privacy.training.DPTrainer,
            'fit',
            autospec=True,
            side_effect=fit_spy,
        ),
    ):
      result = mechanism(rng=np.random.default_rng(0), data=ds)

    mock_load.assert_called_once()
    self.assertGreater(mock_tok.call_count, 0)
    self.assertIsInstance(result, dp_sft.FineTuneResult)
    self.assertIsInstance(result.model, _TinyLoraModel)

  def test_sequence_tokenization(self):
    class _TinyLoraModel(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.embed = nnx.Embed(8, 4, rngs=rngs)
        self.lora = nnx.LoRA(
            4, 2, 8, base_module=nnx.Linear(4, 8, rngs=rngs), rngs=rngs
        )

      def __call__(self, tokens, *args, **kwargs):
        del args, kwargs
        return self.lora(self.embed(tokens)), None

    tiny_model = _TinyLoraModel(nnx.Rngs(0))
    fake_example = {
        'input_tokens': jnp.array([1, 2, 3, 4], dtype=jnp.int32),
        'loss_mask': jnp.array([0, 1, 1, 1], dtype=jnp.int32),
    }
    config = execution_plan.BandMFConfig.default(
        num_bands=1,
        iterations=2,
        expected_participations=1.0,
    )
    mechanism = dp_sft.DPFineTunerConfig(
        model='gemma4_e2b_it',
        mechanism_config=config,
        log_every=1,
    ).configure(budget=1.0)
    examples = [('p1', 'r1'), ('p2', 'r2')]
    logged_steps = []
    with (
        mock.patch.object(
            model, 'load_gemma', return_value=tiny_model
        ) as mock_load,
        mock.patch.object(
            model, 'tokenize_example', return_value=fake_example
        ) as mock_tok,
    ):
      log_fn = lambda step, m: logged_steps.append((step, m))
      result = mechanism(np.random.default_rng(0), examples, metrics_fn=log_fn)

    mock_load.assert_called_once()
    self.assertEqual(mock_tok.call_count, 2)
    self.assertEqual([step for step, _ in logged_steps], [1, 2])
    self.assertIn('loss', logged_steps[0][1])
    self.assertIsInstance(result, dp_sft.FineTuneResult)
    self.assertIsInstance(result.model, _TinyLoraModel)


if __name__ == '__main__':
  absltest.main()
