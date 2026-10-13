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


import dataclasses

from absl.testing import absltest
import dp_accounting
import dpsynth
from dpsynth.text import dp_trainer
from flax import nnx
import grain.python as grain
import jax.numpy as jnp
import jax_privacy
import numpy as np
import optax


def _dummy_params_and_loss():
  """Creates a trivial pytree and loss function for testing."""
  params = {'w': jnp.ones((4, 4))}

  def loss_fn(params, batch, prng):
    del prng
    return jnp.sum(params['w'] * batch['x']), ()

  return params, loss_fn


class DPTrainerTest(absltest.TestCase):

  @property
  def dpsgd_config(self):
    return jax_privacy.execution_plan.BandMFConfig.default(
        iterations=100,
        expected_participations=4,
        num_bands=1,
        l2_clip_norm=1,
    )

  def test_default_creates_valid_config(self):
    params, loss_fn = _dummy_params_and_loss()
    config = dp_trainer.DPTrainerConfig(
        init_params=params,
        loss_fn=loss_fn,
        mechanism_config=self.dpsgd_config,
        optimizer=optax.adamw(1e-4),
    )
    self.assertEqual(config.mechanism_config.iterations, 100)
    self.assertIsNone(config.mechanism_config.noise_multiplier)

  def test_calibrate_sets_noise_multiplier(self):
    params, loss_fn = _dummy_params_and_loss()
    trainer = dp_trainer.DPTrainerConfig(
        init_params=params,
        loss_fn=loss_fn,
        mechanism_config=self.dpsgd_config,
        optimizer=optax.adamw(1e-4),
    ).configure(budget=0.5)
    # Single band: sigma = sqrt(T / (2 * rho)) = 10.
    sigma = trainer.config.mechanism_config.noise_multiplier
    self.assertAlmostEqual(sigma, 10.0)
    self.assertIsNotNone(trainer.dp_event)

  def test_nnx_split_merge_round_trip(self):
    """Demonstrates the NNX split/merge pattern for LoRA fine-tuning."""
    base = nnx.Linear(4, 4, rngs=nnx.Rngs(0))
    lora_model = nnx.LoRA(4, 2, 4, base_module=base, rngs=nnx.Rngs(0))

    graphdef, trainable, frozen = nnx.split(lora_model, nnx.LoRAParam, ...)

    def loss_fn(params, batch, prng):
      del prng
      model = nnx.merge(graphdef, params, frozen)
      x = batch['x']
      return jnp.mean(model(x) ** 2), ()

    full_batch_config = jax_privacy.execution_plan.BandMFConfig.default(
        num_bands=1,
        iterations=5,
        expected_participations=5,
        l2_clip_norm=1.0,
    )
    trainer = dp_trainer.DPTrainerConfig(
        init_params=trainable,
        loss_fn=loss_fn,
        mechanism_config=full_batch_config,
        optimizer=optax.adamw(1e-4),
    ).configure(budget=1.0)

    train_state = trainer(rng=42, data={'x': jnp.ones((10, 4, 4))})
    self.assertIsNotNone(train_state)

  def test_non_private_config(self):
    params, loss_fn = _dummy_params_and_loss()
    non_private_config = jax_privacy.execution_plan.NonPrivateConfig(
        iterations=5,
        batch_size=2,
    )
    trainer = dp_trainer.DPTrainerConfig(
        init_params=params,
        loss_fn=loss_fn,
        mechanism_config=non_private_config,
        optimizer=optax.adamw(1e-4),
    ).configure(budget=float('inf'))

    self.assertIsInstance(trainer.dp_event, dp_accounting.NonPrivateDpEvent)
    train_state = trainer(rng=42, data={'x': jnp.ones((10, 4, 4))})
    self.assertIsNotNone(train_state)

  def test_map_dataset(self):
    params, loss_fn = _dummy_params_and_loss()
    non_private_config = jax_privacy.execution_plan.NonPrivateConfig(
        iterations=3,
        batch_size=2,
    )
    trainer = dp_trainer.DPTrainerConfig(
        init_params=params,
        loss_fn=loss_fn,
        mechanism_config=non_private_config,
        optimizer=optax.adamw(1e-4),
    ).configure(budget=float('inf'))

    ds = grain.MapDataset.source([{'x': np.ones((4, 4))} for _ in range(6)])
    train_state = trainer(rng=42, data=ds)
    self.assertIsNotNone(train_state)

  def test_checkpoint_resumes_mid_training_and_skips_completed(self):
    ckpt_dir = self.create_tempdir().full_path
    params, loss_fn = _dummy_params_and_loss()
    data = {'x': jnp.arange(128, dtype=jnp.float32).reshape((8, 4, 4))}
    mech_config = jax_privacy.execution_plan.BandMFConfig.default(
        num_bands=2,
        iterations=6,
        expected_participations=3,
        l2_clip_norm=1.0,
    )
    base_config = dp_trainer.DPTrainerConfig(
        init_params=params,
        loss_fn=loss_fn,
        mechanism_config=mech_config,
        optimizer=optax.adamw(1e-2),
        checkpoint_every=2,
    )
    base_trainer = base_config.configure(budget=1.0)

    # Uninterrupted reference run without checkpointing.
    expected = base_trainer(rng=42, data=data)

    # First run inside dpsynth.checkpoint: simulate preemption after step 4.
    steps_run_1 = []

    def interrupt_at_4(step, state, aux):
      del state, aux
      steps_run_1.append(step)
      if step == 4:
        raise RuntimeError('Simulated preemption at step 4')

    interrupted_trainer = dataclasses.replace(
        base_config, callback=interrupt_at_4
    ).configure(budget=1.0)
    with self.assertRaises(RuntimeError):
      with dpsynth.checkpoint(ckpt_dir):
        interrupted_trainer(rng=42, data=data)
    self.assertEqual(steps_run_1, [1, 2, 3, 4])

    # Second run: resumes from step 4 checkpoint and executes only steps 5..6.
    steps_run_2 = []
    resumed_trainer = dataclasses.replace(
        base_config, callback=lambda step, s, a: steps_run_2.append(step)
    ).configure(budget=1.0)
    with dpsynth.checkpoint(ckpt_dir):
      resumed = resumed_trainer(rng=42, data=data)
    self.assertEqual(steps_run_2, [5, 6])
    np.testing.assert_allclose(
        resumed.params['w'], expected.params['w'], rtol=1e-5
    )

    # Third run: restores final step 6 checkpoint and executes 0 steps.
    steps_run_3 = []
    cached_trainer = dataclasses.replace(
        base_config, callback=lambda step, s, a: steps_run_3.append(step)
    ).configure(budget=1.0)
    with dpsynth.checkpoint(ckpt_dir):
      cached = cached_trainer(rng=42, data=data)
    self.assertEqual(steps_run_3, [])
    np.testing.assert_allclose(
        cached.params['w'], expected.params['w'], rtol=1e-5
    )

  def test_checkpoint_single_band_zero_size_noise_state(self):
    ckpt_dir = self.create_tempdir().full_path
    params, loss_fn = _dummy_params_and_loss()
    data = {'x': jnp.ones((4, 4, 4))}
    mech_config = jax_privacy.execution_plan.BandMFConfig.default(
        num_bands=1,
        iterations=3,
        expected_participations=3,
        l2_clip_norm=1.0,
    )
    trainer = dp_trainer.DPTrainerConfig(
        init_params=params,
        loss_fn=loss_fn,
        mechanism_config=mech_config,
        optimizer=optax.adamw(1e-2),
        checkpoint_every=2,
    ).configure(budget=1.0)
    with dpsynth.checkpoint(ckpt_dir):
      first = trainer(rng=7, data=data)
      second = trainer(rng=7, data=data)
    np.testing.assert_allclose(first.params['w'], second.params['w'], rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
