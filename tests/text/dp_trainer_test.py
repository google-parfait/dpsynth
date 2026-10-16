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


from absl.testing import absltest
import dp_accounting
from dpsynth.text import dp_trainer
from flax import nnx
import jax.numpy as jnp
import jax_privacy
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
    trainer = dp_trainer.DPTrainer(
        init_params=params,
        loss_fn=loss_fn,
        mechanism_config=self.dpsgd_config,
        optimizer=optax.adamw(1e-4),
    )
    self.assertEqual(trainer.mechanism_config.iterations, 100)
    self.assertIsNone(trainer.mechanism_config.noise_multiplier)

  def test_calibrate_sets_noise_multiplier(self):
    params, loss_fn = _dummy_params_and_loss()
    trainer = dp_trainer.DPTrainer(
        init_params=params,
        loss_fn=loss_fn,
        mechanism_config=self.dpsgd_config,
        optimizer=optax.adamw(1e-4),
    ).configure(budget=0.5)
    # Single band: sigma = sqrt(T / (2 * rho)) = 10.
    self.assertAlmostEqual(trainer.mechanism_config.noise_multiplier, 10.0)
    self.assertIsNotNone(trainer.dp_event)

  def test_raises_before_calibration(self):
    params, loss_fn = _dummy_params_and_loss()
    trainer = dp_trainer.DPTrainer(
        init_params=params,
        loss_fn=loss_fn,
        mechanism_config=self.dpsgd_config,
        optimizer=optax.adamw(1e-4),
    )
    with self.assertRaises(ValueError):
      _ = trainer.dp_event

    with self.assertRaises(ValueError):
      trainer(rng=42, data={'x': jnp.ones((10, 4, 4))})

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
    trainer = dp_trainer.DPTrainer(
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
    trainer = dp_trainer.DPTrainer(
        init_params=params,
        loss_fn=loss_fn,
        mechanism_config=non_private_config,
        optimizer=optax.adamw(1e-4),
    ).configure(budget=float('inf'))

    self.assertIsInstance(trainer.dp_event, dp_accounting.NonPrivateDpEvent)
    train_state = trainer(rng=42, data={'x': jnp.ones((10, 4, 4))})
    self.assertIsNotNone(train_state)


if __name__ == '__main__':
  absltest.main()
