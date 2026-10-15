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
import functools

from absl.testing import absltest
import dp_accounting
from dpsynth.text import dp_trainer
from flax import nnx
import jax.numpy as jnp
from jax_privacy import execution_plan
import optax


class _DummyTask(dp_trainer.TrainingTask):
  """Trivial TrainingTask for testing DPTrainerConfig and DPTrainer."""

  def init_params(self):
    return {'w': jnp.ones((4, 4))}

  def loss_fn(self, params, data, prng):
    del prng
    return jnp.sum(params['w'] * data['x']), ()


@dataclasses.dataclass(frozen=True)
class _LoraTask(dp_trainer.TrainingTask):
  """TrainingTask for testing NNX LoRA split/merge round-trip."""

  model: nnx.Module

  @functools.cached_property
  def _split_model(self):
    return nnx.split(self.model, nnx.LoRAParam, ...)

  def init_params(self):
    _, trainable, _ = self._split_model
    return trainable

  def loss_fn(self, params, data, prng):
    del prng
    graphdef, _, frozen = self._split_model
    model = nnx.merge(graphdef, params, frozen)
    x = data['x']
    return jnp.mean(model(x) ** 2), ()


@dataclasses.dataclass
class _RecordingTask(_DummyTask):
  """TrainingTask that records init_params and preprocess_fn call counts."""

  init_calls: int = 0
  preprocess_calls: int = 0

  def init_params(self):
    self.init_calls += 1
    return super().init_params()

  def preprocess_fn(self, record):
    self.preprocess_calls += 1
    return {'x': record * 2.0}

  def postprocess_fn(self, params):
    return params['w'].shape


class DPTrainerTest(absltest.TestCase):

  @property
  def dpsgd_config(self):
    return execution_plan.BandMFConfig.default(
        iterations=100,
        expected_participations=4,
        num_bands=1,
        l2_clip_norm=1,
    )

  def test_default_creates_valid_config(self):
    config = dp_trainer.DPTrainerConfig(
        task=_DummyTask(),
        mechanism_config=self.dpsgd_config,
        optimizer=optax.adamw(1e-4),
    )
    self.assertEqual(config.mechanism_config.iterations, 100)
    self.assertIsNone(config.mechanism_config.noise_multiplier)

  def test_calibrate_sets_noise_multiplier(self):
    trainer = dp_trainer.DPTrainerConfig(
        task=_DummyTask(),
        mechanism_config=self.dpsgd_config,
        optimizer=optax.adamw(1e-4),
    ).configure(budget=0.5)
    # Single band: sigma = sqrt(T / (2 * rho)) = 10.
    self.assertAlmostEqual(
        trainer.config.mechanism_config.noise_multiplier, 10.0
    )
    self.assertIsNotNone(trainer.dp_event)

  def test_raises_before_calibration(self):
    config = dp_trainer.DPTrainerConfig(
        task=_DummyTask(),
        mechanism_config=self.dpsgd_config,
        optimizer=optax.adamw(1e-4),
    )
    trainer = dp_trainer.DPTrainer(config=config)
    with self.assertRaises(ValueError):
      _ = trainer.dp_event

    with self.assertRaises(ValueError):
      trainer(rng=42, data=[{'x': jnp.ones((4, 4))} for _ in range(10)])

  def test_nnx_split_merge_round_trip(self):
    """Demonstrates the NNX split/merge pattern for LoRA fine-tuning."""
    base = nnx.Linear(4, 4, rngs=nnx.Rngs(0))
    lora_model = nnx.LoRA(4, 2, 4, base_module=base, rngs=nnx.Rngs(0))

    full_batch_config = execution_plan.BandMFConfig.default(
        num_bands=1,
        iterations=5,
        expected_participations=5,
        l2_clip_norm=1.0,
    )
    trainer = dp_trainer.DPTrainerConfig(
        task=_LoraTask(lora_model),
        mechanism_config=full_batch_config,
        optimizer=optax.adamw(1e-4),
    ).configure(budget=1.0)

    records = [{'x': jnp.ones((4, 4))} for _ in range(10)]
    params = trainer(rng=42, data=records)
    self.assertIsNotNone(params)

  def test_non_private_config(self):
    non_private_config = execution_plan.NonPrivateConfig(
        iterations=5,
        batch_size=2,
    )
    trainer = dp_trainer.DPTrainerConfig(
        task=_DummyTask(),
        mechanism_config=non_private_config,
        optimizer=optax.adamw(1e-4),
    ).configure(budget=float('inf'))

    self.assertIsInstance(trainer.dp_event, dp_accounting.NonPrivateDpEvent)
    records = [{'x': jnp.ones((4, 4))} for _ in range(10)]
    params = trainer(rng=42, data=records)
    self.assertIsNotNone(params)

  def test_lazy_init_params_and_per_record_pre_post_process(self):
    task = _RecordingTask()
    non_private_config = execution_plan.NonPrivateConfig(
        iterations=3,
        batch_size=2,
    )
    trainer = dp_trainer.DPTrainerConfig(
        task=task,
        mechanism_config=non_private_config,
        optimizer=optax.adamw(1e-4),
    ).configure(budget=float('inf'))

    # Configuring should not invoke init_params.
    self.assertEqual(task.init_calls, 0)

    records = [jnp.ones((4, 4)) for _ in range(6)]
    out_shape = trainer(rng=42, data=records)
    self.assertEqual(task.init_calls, 1)
    self.assertEqual(task.preprocess_calls, 6)
    self.assertEqual(out_shape, (4, 4))


if __name__ == '__main__':
  absltest.main()
