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

"""Tests for dpsynth.training.

These tests verify the loss function, configuration, and calibration logic
without loading any real model checkpoints.
"""

import dataclasses
import functools
import math
import os
import sys
import typing
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth import serialize
from dpsynth import training as dp_sft
from dpsynth.training import _gemma as model
from dpsynth.training import _sequence_model
from flax import nnx
import grain.python as grain
import jax
import jax.numpy as jnp
import jax_privacy
from jax_privacy import execution_plan
import numpy as np
import optax


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


def _make_tiny_lora_model(vocab_size=8, embed_dim=4, lora_rank=2):
  """Creates a minimal LoRA-wrapped Flax NNX model for testing."""

  class TinyLoraModel(nnx.Module):

    def __init__(self, rngs: nnx.Rngs):
      base = nnx.Linear(embed_dim, vocab_size, rngs=rngs)
      self.embed = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
      self.lora = nnx.LoRA(
          embed_dim, lora_rank, vocab_size, base_module=base, rngs=rngs
      )

    def __call__(self, tokens, *args, **kwargs):
      del args, kwargs
      return self.lora(self.embed(tokens)), None

  return TinyLoraModel(nnx.Rngs(0))


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

  def test_format_prompt_and_response(self):
    self.assertEqual(
        model.format_prompt('Hello'),
        '<|turn>user\nHello<turn|>\n<|turn>model\n',
    )
    self.assertEqual(model.format_response('World'), 'World<turn|>')

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

  def test_epath_and_resolve_path(self):
    self.assertEqual(os.environ.get('EPATH_USE_TF'), 'false')
    self.assertTrue(model._resolve_path('rel/lora_ckpt').is_absolute())
    self.assertEqual(
        model._resolve_path('gs://bucket/lora_ckpt'),
        model.epath.Path('gs://bucket/lora_ckpt'),
    )

  def test_deprecated_text_subpackage_reexports(self):
    sys.modules.pop('dpsynth.text', None)
    with self.assertWarnsRegex(DeprecationWarning, 'dpsynth.training'):
      import dpsynth.text  # pylint: disable=g-import-not-at-top
    self.assertIs(dpsynth.text.DPFineTuner, dp_sft.DPFineTuner)
    self.assertIs(dpsynth.text.DPTrainer, dp_sft.DPTrainer)

  def test_load_gemma_sequence_model_and_loss_fn(self):
    tiny_model = _make_tiny_lora_model()
    fake_example = {
        'input_tokens': np.array([1, 2, 3, 4], dtype=np.int32),
        'loss_mask': np.array([0, 1, 1, 1], dtype=np.int32),
    }
    fake_sample = lambda prompts: [f'gen_{p}' for p in prompts]
    with (
        mock.patch.object(model, 'load_gemma', return_value=tiny_model),
        mock.patch.object(model, 'tokenize_example', return_value=fake_example),
        mock.patch.object(model, 'GemmaSampler', return_value=fake_sample),
    ):
      seq_model = dp_sft.load_gemma_sequence_model('gemma4_e2b_it')
      assert seq_model.sample_fn is not None
      samples = seq_model.sample_fn(seq_model.params, ['q1'], 512)

    self.assertIsInstance(seq_model, dp_sft.SequenceModel)
    self.assertEqual(seq_model.tokenize_fn(('p', 'r')), fake_example)
    self.assertEqual(samples, ['gen_q1'])
    batch = {
        'input_tokens': jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),
        'loss_mask': jnp.array([[0, 1, 1, 1]], dtype=jnp.int32),
    }
    loss, aux = seq_model.loss_fn(seq_model.params, batch, jax.random.key(0))
    self.assertTrue(jnp.isfinite(loss))
    self.assertIn('loss', aux)

  def test_tokenize_example_unpadded(self):
    fake_sp = mock.Mock()
    fake_sp.bos_id.return_value = 2
    fake_sp.eos_id.return_value = 1
    fake_sp.EncodeAsIds.side_effect = lambda s: [10 + len(s), 20 + len(s)]
    with mock.patch.object(model, '_get_tokenizer', return_value=fake_sp):
      out = model.tokenize_example(('Hi', 'World'))
    p_len = 1 + 2  # BOS + 2 prompt tokens
    r_len = 2 + 1  # 2 response tokens + EOS
    self.assertEqual(out['input_tokens'].dtype, np.int32)
    self.assertEqual(out['loss_mask'].dtype, np.int32)
    self.assertEqual(out['input_tokens'].shape, (p_len + r_len,))
    np.testing.assert_array_equal(out['loss_mask'], [0] * p_len + [1] * r_len)
    self.assertEqual(int(out['input_tokens'][0]), 2)
    self.assertEqual(int(out['input_tokens'][-1]), 1)


class LoraCheckpointTest(absltest.TestCase):

  def test_save_and_load_lora_params(self):
    class _TinyLora(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.lora = nnx.LoRA(
            4, 2, 8, base_module=nnx.Linear(4, 8, rngs=rngs), rngs=rngs
        )

    m1 = _TinyLora(nnx.Rngs(0))
    m1.lora.lora_b.value = jnp.ones_like(m1.lora.lora_b.value)
    ckpt_dir = self.create_tempdir().full_path + '/lora_ckpt'
    model.save_lora_params(m1, ckpt_dir)

    m2 = _TinyLora(nnx.Rngs(1))
    model.load_lora_params(m2, ckpt_dir)
    self.assertTrue(jnp.allclose(m1.lora.lora_b.value, m2.lora.lora_b.value))


class DPFineTunerTest(parameterized.TestCase):

  def test_creates_valid_mechanism_config(self):
    config = dp_sft.DPFineTunerConfig(
        model='gemma4_e2b_it',
        iterations=100,
        epochs=1.0,
    )
    self.assertEqual(config.model, 'gemma4_e2b_it')
    self.assertEqual(config.iterations, 100)
    self.assertEqual(config.epochs, 1.0)
    with self.assertRaises(ValueError):
      dataclasses.replace(config, log_every=3, checkpoint_every=10)
    with self.assertRaises(ValueError):
      dataclasses.replace(config, canary_repeats=0)
    with self.assertRaises(ValueError):
      dp_sft.DPFineTunerConfig(
          model=typing.cast(model.ModelName, 'nonexistent'),
          iterations=100,
          epochs=1.0,
      )

  def test_direct_dp_fine_tuner_validates_calibration(self):
    model_fn = functools.partial(
        model.load_gemma_sequence_model, 'gemma4_e2b_it'
    )
    uncalibrated_plan = _default_config()
    with self.assertRaisesRegex(ValueError, 'noise_multiplier is not set'):
      dp_sft.DPFineTuner(
          model=model_fn,
          mechanism_config=uncalibrated_plan,
          optimizer=optax.adamw(1e-3),
      )
    calibrated_plan = uncalibrated_plan.calibrate(epsilon=1.0, delta=1e-5)
    mechanism = dp_sft.DPFineTuner(
        model=model_fn,
        mechanism_config=calibrated_plan,
        optimizer=optax.adamw(1e-3),
    )
    self.assertIsNotNone(mechanism.dp_event)
    with self.assertRaises(ValueError):
      dataclasses.replace(mechanism, log_every=3, checkpoint_every=10)
    with self.assertRaises(ValueError):
      dataclasses.replace(mechanism, canary_repeats=0)

  def test_calibrate_sets_noise_multiplier(self):
    mechanism = dp_sft.DPFineTunerConfig(
        model='gemma4_e2b_it',
        iterations=100,
        epochs=1.0,
    ).configure(budget=0.5)
    self.assertIsNotNone(mechanism.mechanism_config.noise_multiplier)
    self.assertGreater(mechanism.mechanism_config.noise_multiplier, 0.0)

  def test_calibrate_noise_formula(self):
    mechanism = dp_sft.DPFineTunerConfig(
        model='gemma4_e2b_it',
        iterations=100,
        epochs=1.0,
    ).configure(budget=0.5)
    # Single band: rounds = iterations, sigma = sqrt(T / (2*rho)).
    expected = math.sqrt(100 / (2.0 * 0.5))
    sigma = mechanism.mechanism_config.noise_multiplier
    self.assertAlmostEqual(sigma, expected)

  def test_dp_event_after_calibration(self):
    mechanism = dp_sft.DPFineTunerConfig(
        model='gemma4_e2b_it',
        iterations=100,
        epochs=1.0,
    ).configure(budget=0.5)
    event = mechanism.dp_event
    self.assertIsNotNone(event)

  def test_map_dataset_lazy_tokenization(self):
    tiny_model = _make_tiny_lora_model()
    fake_example = {
        'input_tokens': jnp.array([1, 2, 3, 4], dtype=jnp.int32),
        'loss_mask': jnp.array([0, 1, 1, 1], dtype=jnp.int32),
    }
    mechanism = dp_sft.DPFineTunerConfig(
        model='gemma4_e2b_it',
        iterations=2,
        epochs=1.0,
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
    self.assertIsInstance(result.model, dp_sft.SequenceModel)

  @parameterized.parameters((None,), (['q1', 'q2'],))
  def test_sequence_tokenization(self, target_prompts):
    tiny_model = _make_tiny_lora_model()
    fake_example = {
        'input_tokens': jnp.array([1, 2, 3, 4], dtype=jnp.int32),
        'loss_mask': jnp.array([0, 1, 1, 1], dtype=jnp.int32),
    }
    mechanism = dp_sft.DPFineTunerConfig(
        model='gemma4_e2b_it',
        iterations=2,
        epochs=1.0,
        log_every=1,
    ).configure(budget=1.0)
    examples = [('p1', 'r1'), ('p2', 'r2')]
    logged_steps = []
    fake_sample = lambda prompts: [f'gen_{p}' for p in prompts]
    with (
        mock.patch.object(
            model, 'load_gemma', return_value=tiny_model
        ) as mock_load,
        mock.patch.object(
            model, 'tokenize_example', return_value=fake_example
        ) as mock_tok,
        mock.patch.object(
            model, 'GemmaSampler', return_value=fake_sample
        ) as mock_sampler,
    ):
      log_fn = lambda step, m: logged_steps.append((step, m))
      result = mechanism(
          np.random.default_rng(0),
          examples,
          metrics_fn=log_fn,
          target_prompts=target_prompts,
      )

    mock_load.assert_called_once()
    self.assertEqual(mock_tok.call_count, 2)
    self.assertEqual([step for step, _ in logged_steps], [1, 2])
    self.assertIn('loss', logged_steps[0][1])
    self.assertIsInstance(result, dp_sft.FineTuneResult)
    self.assertIsInstance(result.model, dp_sft.SequenceModel)
    self.assertIsNone(result.auditor)

    if target_prompts is None:
      mock_sampler.assert_not_called()
      self.assertIsNone(result.samples)
    else:
      mock_sampler.assert_called_once_with(mock.ANY, max_seq_length=512)
      self.assertEqual(result.samples, [('q1', 'gen_q1'), ('q2', 'gen_q2')])

  @parameterized.parameters(False, True)
  def test_canaries_auditing(self, use_map_dataset):
    def fake_tokenize(example):
      _, resp = example
      tok = int(resp)
      return {
          'input_tokens': np.array([0, 1, tok, tok], dtype=np.int32),
          'loss_mask': np.array([0, 0, 1, 1], dtype=np.int32),
      }

    config = execution_plan.BandMFConfig.default(
        num_bands=1,
        iterations=25,
        expected_participations=12.0,
        noise_multiplier=0.0,  # No noise to ensure memorization.
    )
    mechanism = dp_sft.DPFineTuner(
        model=functools.partial(
            model.load_gemma_sequence_model, 'gemma4_e2b_it'
        ),
        mechanism_config=config,
        optimizer=optax.adamw(0.1),
        max_seq_length=4,
        canary_repeats=2,
    )
    num_examples = 8
    examples = [('p', '1')] * num_examples
    data = grain.MapDataset.source(examples) if use_map_dataset else examples
    canaries = [('c', str(2 + i)) for i in range(9)]
    tiny_model = _make_tiny_lora_model(vocab_size=16, embed_dim=8, lora_rank=4)
    with (
        mock.patch.object(model, 'load_gemma', return_value=tiny_model),
        mock.patch.object(model, 'tokenize_example', side_effect=fake_tokenize),
    ):
      result = mechanism(np.random.default_rng(0), data, canaries=canaries)

    self.assertIsNotNone(result.auditor)
    assert result.auditor is not None
    self.assertEqual(result.auditor.num_canaries, (5, 4))
    self.assertAlmostEqual(result.auditor.attack_auroc(), 1.0, places=5)

  def test_yaml_round_trip(self):
    config = dp_sft.DPFineTunerConfig(
        model='gemma4_e2b_it',
        iterations=100,
        epochs=1.0,
        learning_rate=2e-3,
        lora_rank=8,
    )
    restored = serialize.from_yaml(
        serialize.to_yaml(config), dp_sft.DPFineTunerConfig
    )
    self.assertEqual(restored.model, config.model)
    self.assertEqual(restored.learning_rate, 2e-3)
    self.assertEqual(restored.lora_rank, 8)
    self.assertEqual(restored.iterations, 100)
    self.assertEqual(restored.epochs, 1.0)

  def test_pad_and_truncate(self):
    rng = np.random.default_rng(42)
    for _ in range(32):
      seq_len = int(rng.integers(0, 16))
      target_len = int(rng.integers(1, 12))
      x = rng.integers(1, 100, size=(seq_len,), dtype=np.int32)
      out = _sequence_model.pad_and_truncate(x, target_len)
      self.assertEqual(out.shape, (target_len,))
      self.assertEqual(out.dtype, np.int32)
      n = min(seq_len, target_len)
      np.testing.assert_array_equal(out[:n], x[:n])
      np.testing.assert_array_equal(out[n:], 0)


if __name__ == '__main__':
  absltest.main()
