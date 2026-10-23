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
from unittest import mock

from absl.testing import absltest
import dp_accounting
import dpsynth
from dpsynth.text import dp_trainer
from flax import nnx
import grain.python as grain
import jax
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
    with self.assertRaises(ValueError):
      dataclasses.replace(config, log_every=3, checkpoint_every=10)
    with self.assertRaises(ValueError):
      dataclasses.replace(config, snapshot_steps=[0])
    with self.assertRaises(ValueError):
      dataclasses.replace(config, snapshot_steps=[101])

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

    train_state = trainer(
        rng=np.random.default_rng(42), data={'x': jnp.ones((10, 4, 4))}
    )
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
    train_state = trainer(
        rng=np.random.default_rng(42), data={'x': jnp.ones((10, 4, 4))}
    )
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

    data = grain.MapDataset.source([{'x': np.ones((4, 4))} for _ in range(6)])
    train_state = trainer(rng=np.random.default_rng(42), data=data)
    self.assertIsNotNone(train_state)

  def test_checkpoint_resumes_mid_training_and_skips_completed(self):
    # Runs a 6-step training job (saving checkpoints every 2 steps) across
    # three invocations: (1) raises at step 4 via interrupt_at_4 to simulate
    # preemption after the step-4 checkpoint, (2) resumes from step 4 to finish
    # steps 5..6, and (3) verifies a third call restores step 6 with 0 steps.
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
        log_every=1,
        checkpoint_every=2,
    )
    base_trainer = base_config.configure(budget=1.0)

    # Uninterrupted reference run without checkpointing.
    expected = base_trainer(rng=np.random.default_rng(42), data=data)

    # First run inside dpsynth.checkpoint: simulate preemption after step 4.
    steps_run_1 = []

    def interrupt_at_4(step, metrics):
      del metrics
      steps_run_1.append(step)
      if step == 4:
        raise RuntimeError('Simulated preemption at step 4')

    with self.assertRaises(RuntimeError):
      with dpsynth.checkpoint(ckpt_dir):
        base_trainer(
            rng=np.random.default_rng(42),
            data=data,
            metrics_fn=interrupt_at_4,
        )
    self.assertEqual(steps_run_1, [1, 2, 3, 4])

    # Second run: resumes from step 4 checkpoint and executes only steps 5..6,
    # restoring the initial RNG seed even when a fresh RNG is passed.
    steps_run_2 = []
    log_2 = lambda s, m: steps_run_2.append(s)
    with dpsynth.checkpoint(ckpt_dir):
      resumed = base_trainer(np.random.default_rng(999), data, metrics_fn=log_2)
    self.assertEqual(steps_run_2, [5, 6])
    np.testing.assert_allclose(
        resumed.params['w'], expected.params['w'], rtol=1e-5
    )

    # Third run: restores final step 6 checkpoint and executes 0 steps.
    steps_run_3 = []
    log_3 = lambda s, m: steps_run_3.append(s)
    with dpsynth.checkpoint(ckpt_dir):
      cached = base_trainer(np.random.default_rng(123), data, metrics_fn=log_3)
    self.assertEqual(steps_run_3, [])
    np.testing.assert_allclose(
        cached.params['w'], expected.params['w'], rtol=1e-5
    )

  def test_multihost_seed_broadcast(self):
    # Simulates multi-host broadcast in a single process, assuming
    # broadcast_one_to_all broadcasts from process_index == 0 as documented.
    params, loss_fn = _dummy_params_and_loss()
    data = {'x': jnp.arange(128, dtype=jnp.float32).reshape((8, 4, 4))}
    mech_config = dataclasses.replace(self.dpsgd_config, iterations=4)
    trainer = dp_trainer.DPTrainerConfig(
        init_params=params,
        loss_fn=loss_fn,
        mechanism_config=mech_config,
        optimizer=optax.adamw(1e-2),
    ).configure(budget=1.0)
    broadcast_seed = None

    def fake_broadcast(x):
      nonlocal broadcast_seed
      if jax.process_index() == 0:
        broadcast_seed = x
      return broadcast_seed

    with mock.patch.object(
        dp_trainer.multihost_utils, 'broadcast_one_to_all', fake_broadcast
    ):
      host0 = trainer(rng=np.random.default_rng(42), data=data)
      with mock.patch.object(jax, 'process_index', return_value=1):
        host1 = trainer(rng=np.random.default_rng(999), data=data)

    np.testing.assert_allclose(host1.params['w'], host0.params['w'], rtol=1e-5)

  def test_metrics_fn_reports_step_metrics(self):
    params = {'w': jnp.ones((4, 4))}

    def loss_fn(params, batch, prng):
      del prng
      loss = jnp.sum(params['w'] * batch['x'])
      return loss, {'custom_aux': loss * 2.0}

    mech_config = jax_privacy.execution_plan.BandMFConfig.default(
        num_bands=1,
        iterations=4,
        expected_participations=2,
        l2_clip_norm=1.0,
    )
    trainer = dp_trainer.DPTrainerConfig(
        init_params=params,
        loss_fn=loss_fn,
        mechanism_config=mech_config,
        optimizer=optax.adamw(1e-2),
        log_every=2,
    ).configure(budget=1.0)

    recorded = []
    data = {'x': jnp.ones((8, 4, 4))}
    log_rec = lambda s, m: recorded.append((s, m))
    trainer(np.random.default_rng(0), data, metrics_fn=log_rec)

    per_step = []
    step_trainer = dp_trainer.DPTrainerConfig(
        init_params=params,
        loss_fn=loss_fn,
        mechanism_config=mech_config,
        optimizer=optax.adamw(1e-2),
        log_every=1,
    ).configure(budget=1.0)
    log_step = lambda s, m: per_step.append(m)
    step_trainer(np.random.default_rng(0), data, metrics_fn=log_step)

    m0, m1 = per_step[0], per_step[1]
    expected_bs = 0.5 * (m0['batch_size'] + m1['batch_size'])
    expected_max = max(m0['grad_norm_max'], m1['grad_norm_max'])
    self.assertEqual([step for step, _ in recorded], [2, 4])
    self.assertAlmostEqual(recorded[0][1]['batch_size'], expected_bs)
    self.assertAlmostEqual(recorded[0][1]['grad_norm_max'], expected_max)
    for _, metrics in recorded:
      self.assertGreater(metrics['batch_size'], 0.0)
      self.assertLessEqual(metrics['batch_size'], 8.0)
      self.assertAlmostEqual(metrics['custom_aux'], 2.0 * metrics['loss'])
      self.assertIn('grad_norm_mean', metrics)
      self.assertIn('grad_norm_min', metrics)
      self.assertIn('grad_norm_median', metrics)
      self.assertIn('grad_norm_max', metrics)
      self.assertGreaterEqual(metrics['step_time_sec'], 0.0)

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
      first = trainer(rng=np.random.default_rng(7), data=data)
      second = trainer(rng=np.random.default_rng(7), data=data)
    np.testing.assert_allclose(first.params['w'], second.params['w'], rtol=1e-5)

  def test_canaries_auditing_array_and_map_dataset(self):
    def loss_fn(params, batch, prng):
      del prng
      return jnp.mean((params['w'] - batch['x']) ** 2), ()

    config = jax_privacy.execution_plan.BandMFConfig.default(
        num_bands=1,
        iterations=20,
        expected_participations=10,
        l2_clip_norm=1.0,
        noise_multiplier=0.0,
    )
    trainer = dp_trainer.DPTrainer(
        dp_trainer.DPTrainerConfig(
            init_params={'w': jnp.zeros(8)},
            loss_fn=loss_fn,
            mechanism_config=config,
            optimizer=optax.sgd(0.5),
        )
    )
    canaries = {'x': jnp.eye(8)}
    result = trainer(
        rng=np.random.default_rng(0),
        data={'x': jnp.zeros((8, 8))},
        canaries=canaries,
    )
    self.assertIsInstance(result, dp_trainer.TrainingResult)
    assert result.auditor is not None
    self.assertAlmostEqual(float(result.auditor.attack_auroc()), 1.0, places=5)

    data = grain.MapDataset.source([{'x': np.zeros(8, dtype=np.float32)}] * 8)
    result = trainer(rng=np.random.default_rng(0), data=data, canaries=canaries)
    assert result.auditor is not None
    self.assertAlmostEqual(float(result.auditor.attack_auroc()), 1.0, places=5)

    with self.assertRaises(ValueError):
      trainer(
          rng=np.random.default_rng(0),
          data={'x': jnp.zeros((8, 8))},
          canaries={'x': jnp.ones((1, 8))},
      )

  def test_snapshots_and_checkpoint_resume(self):
    def loss_fn(params, batch, prng):
      del prng
      return jnp.mean((params['w'] - batch['x']) ** 2), ()

    config = jax_privacy.execution_plan.BandMFConfig.default(
        num_bands=1,
        iterations=6,
        expected_participations=6,
        l2_clip_norm=1.0,
        noise_multiplier=0.0,
    )
    trainer = dp_trainer.DPTrainer(
        dp_trainer.DPTrainerConfig(
            init_params={'w': jnp.zeros(8)},
            loss_fn=loss_fn,
            mechanism_config=config,
            optimizer=optax.sgd(0.5),
            log_every=1,
            checkpoint_every=2,
            snapshot_steps=[2, 5],
        )
    )
    data = {'x': jnp.zeros((8, 8))}
    canaries = {'x': jnp.eye(8)}
    expected = trainer(np.random.default_rng(0), data, canaries=canaries)
    self.assertEqual(set(expected.snapshots.keys()), {2, 5})
    self.assertFalse(
        jnp.allclose(expected.snapshots[2].params['w'], expected.params['w'])
    )
    assert expected.snapshots[2].auditor is not None
    assert expected.snapshots[5].auditor is not None

    ckpt_dir = self.create_tempdir().full_path

    def interrupt_at_4(step, metrics):
      del metrics
      if step == 4:
        raise RuntimeError('Simulated preemption at step 4')

    with self.assertRaises(RuntimeError):
      with dpsynth.checkpoint(ckpt_dir):
        trainer(
            np.random.default_rng(0),
            data,
            metrics_fn=interrupt_at_4,
            canaries=canaries,
        )

    with dpsynth.checkpoint(ckpt_dir):
      resumed = trainer(np.random.default_rng(999), data, canaries=canaries)
    self.assertEqual(set(resumed.snapshots.keys()), {2, 5})
    for step in (2, 5):
      np.testing.assert_allclose(
          resumed.snapshots[step].params['w'],
          expected.snapshots[step].params['w'],
          rtol=1e-5,
      )
      assert resumed.snapshots[step].auditor is not None
      assert expected.snapshots[step].auditor is not None
      self.assertAlmostEqual(
          float(resumed.snapshots[step].auditor.attack_auroc()),
          float(expected.snapshots[step].auditor.attack_auroc()),
          places=5,
      )


if __name__ == '__main__':
  absltest.main()
