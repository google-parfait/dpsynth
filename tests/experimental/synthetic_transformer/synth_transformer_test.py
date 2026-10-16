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

"""Unit tests for stateful SyntheticTransformer class in NNX."""

import io
import typing
from absl.testing import absltest
import dp_accounting
from dpsynth.experimental.synthetic_transformer import synth_transformer
from flax import nnx
import jax
import jax.numpy as jnp
import mbi
import numpy as np

# Disable compilation cache to prevent SIGBUS write errors in sandbox /tmp
jax.config.update("jax_enable_compilation_cache", False)


class TabularTransformerTest(absltest.TestCase):

  def test_initialization_and_loss(self):
    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[3, 2],
        num_layers=1,
        n_head=2,
        emb_dim=16,
    )
    model = synth_transformer.TabularTransformerModel(config, seed=42)

    self.assertFalse(model._is_initialized)
    self.assertIsNotNone(model.model)

    # Test loss computation on a dummy row
    x_row = jnp.array([1, 0], dtype=jnp.int32)
    loss = model.model.compute_loss_single(x_row, train=False)
    self.assertIsNotNone(loss)
    self.assertEqual(loss.shape, ())  # Should be a scalar

  def test_save_and_load_file_object(self):
    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[3, 2],
        num_layers=1,
        n_head=2,
        emb_dim=16,
    )
    model = synth_transformer.TabularTransformerModel(config)
    model._is_initialized = True

    # Save to BytesIO
    f = io.BytesIO()
    model.save(f)
    f.seek(0)

    # Load from BytesIO
    loaded_model = synth_transformer.TabularTransformerModel.load(f)  # pylint: disable=g-unsafe-pickle-load
    self.assertTrue(loaded_model._is_initialized)
    self.assertEqual(
        loaded_model._config.num_elements_per_feature,
        config.num_elements_per_feature,
    )
    self.assertIsNotNone(loaded_model.model)

  def test_sample(self):
    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[3, 2],
        num_layers=1,
        n_head=2,
        emb_dim=16,
    )
    model = synth_transformer.TabularTransformerModel(config, seed=42)
    model._is_initialized = True  # Mock training completion

    # 1. Test sampling exactly the microbatch size (256)
    samples_exact = model.sample(num_rows=256)
    self.assertEqual(samples_exact.shape, (256, 2))

    # 2. Test sampling with a custom microbatch size (runs 1 block of size 10)
    samples_custom = model.sample(num_rows=10, microbatch_size=10)
    self.assertEqual(samples_custom.shape, (10, 2))
    self.assertTrue(jnp.all(samples_custom[:, 0] < 3))
    self.assertTrue(jnp.all(samples_custom[:, 1] < 2))

    # 3. Test sampling large batch with chunking (runs 3 blocks of size 10)
    samples_chunked = model.sample(num_rows=30, microbatch_size=10)
    self.assertEqual(samples_chunked.shape, (30, 2))

    # 4. Test that non-multiple of microbatch_size raises ValueError
    with self.assertRaisesRegex(ValueError, "must be a multiple of"):
      model.sample(num_rows=15, microbatch_size=10)

  def test_compute_loss_with_key(self):
    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[3, 2],
        num_layers=1,
        n_head=2,
        emb_dim=16,
    )
    model = synth_transformer.TabularTransformerModel(config, seed=42)
    x_row = jnp.array([1, 0], dtype=jnp.int32)
    key = jax.random.key(123)
    loss = model.model.compute_loss_single(x_row, train=True, key=key)
    self.assertIsNotNone(loss)

  def test_sample_untrained_raises_error(self):
    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[3, 2],
    )
    model = synth_transformer.TabularTransformerModel(config)
    with self.assertRaisesRegex(ValueError, "trained using fit"):
      model.sample(num_rows=5)

  def test_save_untrained_raises_error(self):
    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[3, 2],
    )
    model = synth_transformer.TabularTransformerModel(config)
    f = io.BytesIO()
    with self.assertRaisesRegex(ValueError, "untrained model"):
      model.save(f)

  def test_synth_model_e2e_correlation(self):
    """Verifies that the model can learn conditional correlations."""
    num_samples = 64
    key = jax.random.key(42)
    k0, k1 = jax.random.split(key, 2)

    col0 = jax.random.randint(k0, shape=(num_samples,), minval=0, maxval=2)
    cond_probs = jnp.array([0.1, 0.9])
    probs_col1 = cond_probs[col0]
    col1 = (jax.random.uniform(k1, shape=(num_samples,)) < probs_col1).astype(
        jnp.int32
    )
    dataset = jnp.stack([col0, col1], axis=1)

    orig_p_col1_given_col0_0 = float(jnp.mean(col1[col0 == 0]))
    orig_p_col1_given_col0_1 = float(jnp.mean(col1[col0 == 1]))

    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[2, 2],
        num_layers=1,
        n_head=2,
        emb_dim=16,
    )
    model = synth_transformer.TabularTransformerModel(config)

    model.fit(dataset, num_epochs=100, num_iterations=100, learning_rate=5e-3)
    self.assertTrue(model._is_initialized)

    sample_key = jax.random.key(123)
    synthetic_data = model.sample(num_rows=3072, key=sample_key)
    self.assertEqual(synthetic_data.shape, (3072, 2))

    synth_col0 = synthetic_data[:, 0]
    synth_col1 = synthetic_data[:, 1]

    synth_p_col1_given_col0_0 = float(jnp.mean(synth_col1[synth_col0 == 0]))
    synth_p_col1_given_col0_1 = float(jnp.mean(synth_col1[synth_col0 == 1]))

    self.assertTrue(jnp.all(synthetic_data >= 0))
    self.assertTrue(jnp.all(synthetic_data < 2))

    # Verify learning accuracy (tolerance 0.10)
    self.assertAlmostEqual(
        synth_p_col1_given_col0_0, orig_p_col1_given_col0_0, delta=0.10
    )
    self.assertAlmostEqual(
        synth_p_col1_given_col0_1, orig_p_col1_given_col0_1, delta=0.10
    )

  def test_synth_model_e2e_independent_distribution(self):
    """Verifies that the model can learn independent categorical marginals."""
    num_samples = 64
    key = jax.random.key(42)
    k0, k1, k2 = jax.random.split(key, 3)

    probs_col0 = jnp.array([0.2, 0.5, 0.3])
    probs_col1 = jnp.array([0.1, 0.2, 0.4, 0.3])
    probs_col2 = jnp.array([0.7, 0.3])

    col0 = jax.random.categorical(
        k0, jnp.log(probs_col0 + 1e-10), shape=(num_samples,)
    )
    col1 = jax.random.categorical(
        k1, jnp.log(probs_col1 + 1e-10), shape=(num_samples,)
    )
    col2 = jax.random.categorical(
        k2, jnp.log(probs_col2 + 1e-10), shape=(num_samples,)
    )
    dataset = jnp.stack([col0, col1, col2], axis=1)

    orig_freq_col0 = [float(jnp.mean(col0 == k)) for k in range(3)]
    orig_freq_col1 = [float(jnp.mean(col1 == k)) for k in range(4)]
    orig_freq_col2 = [float(jnp.mean(col2 == k)) for k in range(2)]

    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[3, 4, 2],
        num_layers=1,
        n_head=2,
        emb_dim=16,
    )
    model = synth_transformer.TabularTransformerModel(config)

    model.fit(dataset, num_epochs=100, num_iterations=100, learning_rate=5e-3)
    self.assertTrue(model._is_initialized)

    sample_key = jax.random.key(123)
    synthetic_data = model.sample(num_rows=3072, key=sample_key)
    self.assertEqual(synthetic_data.shape, (3072, 3))

    synth_col0 = synthetic_data[:, 0]
    synth_col1 = synthetic_data[:, 1]
    synth_col2 = synthetic_data[:, 2]

    synth_freq_col0 = [float(jnp.mean(synth_col0 == k)) for k in range(3)]
    synth_freq_col1 = [float(jnp.mean(synth_col1 == k)) for k in range(4)]
    synth_freq_col2 = [float(jnp.mean(synth_col2 == k)) for k in range(2)]

    num_elements = config.num_elements_per_feature
    assert num_elements is not None
    for i in range(3):
      self.assertTrue(jnp.all(synthetic_data[:, i] >= 0))
      self.assertTrue(jnp.all(synthetic_data[:, i] < num_elements[i]))

    for k in range(3):
      self.assertAlmostEqual(synth_freq_col0[k], orig_freq_col0[k], delta=0.08)
    for k in range(4):
      self.assertAlmostEqual(synth_freq_col1[k], orig_freq_col1[k], delta=0.08)
    for k in range(2):
      self.assertAlmostEqual(synth_freq_col2[k], orig_freq_col2[k], delta=0.08)

  def test_synth_model_dp_training(self):
    """Verifies that DP-SGD training works without throwing errors."""
    num_samples = 64
    key = jax.random.key(0)
    col0 = jax.random.categorical(
        key, jnp.log(jnp.array([0.5, 0.5]) + 1e-10), shape=(num_samples,)
    )
    dataset = jnp.stack([col0], axis=1)

    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[2],
        num_layers=1,
        n_head=1,
        emb_dim=8,
    )
    model = synth_transformer.TabularTransformerModel(config)

    # Train model with DP-SGD (10 updates)
    model.fit(
        dataset,
        num_epochs=10,
        num_iterations=10,
        learning_rate=1e-2,
        zcdp_rho=1.0,
    )
    self.assertTrue(model._is_initialized)

    synthetic_data = model.sample(num_rows=10, key=key, microbatch_size=10)
    self.assertEqual(synthetic_data.shape, (10, 1))

  def test_synth_model_dropout(self):
    """Verifies that training with dropout > 0 works without throwing errors."""
    num_samples = 64
    key = jax.random.key(0)
    col0 = jax.random.categorical(
        key, jnp.log(jnp.array([0.5, 0.5]) + 1e-10), shape=(num_samples,)
    )
    dataset = jnp.stack([col0], axis=1)

    # Configure with dropout > 0
    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[2],
        num_layers=1,
        n_head=1,
        emb_dim=8,
        dropout=0.1,
    )
    model = synth_transformer.TabularTransformerModel(config)

    # Non-DP training
    model.fit(dataset, num_epochs=5, num_iterations=5, learning_rate=1e-2)
    self.assertTrue(model._is_initialized)

    # DP training
    model_dp = synth_transformer.TabularTransformerModel(config)
    model_dp.fit(
        dataset,
        num_epochs=5,
        num_iterations=5,
        learning_rate=1e-2,
        zcdp_rho=1.0,
    )
    self.assertTrue(model_dp._is_initialized)

  def test_tabular_transformer_mechanism(self):
    """Verifies that the DPMechanism subclass works with mbi.Dataset."""
    domain = mbi.Domain(["A", "B"], [2, 2])
    # Create simple dataset with 64 records
    data_dict = {
        "A": np.array([0, 1, 0, 1] * 16, dtype=np.int32),
        "B": np.array([1, 0, 1, 0] * 16, dtype=np.int32),
    }
    dataset = mbi.Dataset(typing.cast(typing.Any, data_dict), domain)

    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[],  # Configured dynamically by mechanism!
        num_layers=1,
        n_head=1,
        emb_dim=8,
    )

    mechanism = synth_transformer.TabularTransformer(
        config=config,
        num_epochs=5,
        num_iterations=5,
        learning_rate=1e-2,
    )

    # Calibrate it
    mechanism = mechanism.configure(budget=1.0)
    self.assertEqual(mechanism.zcdp_rho, 1.0)

    # Run the mechanism
    rng = np.random.default_rng(0)
    result = mechanism(rng, dataset)

    # Check output
    self.assertIsNone(result.model)
    self.assertEqual(result.synthetic_data.domain, domain)
    # The output dataset size should match the input dataset size (64)
    self.assertEqual(result.synthetic_data.records, 64)

  def test_model_init_empty_features_raises_error(self):
    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[],
        num_layers=1,
        n_head=1,
        emb_dim=8,
    )
    with self.assertRaisesRegex(
        ValueError, "num_elements_per_feature must be non-empty"
    ):
      synth_transformer.TabularTransformerModel(config)

  def test_mechanism_call_unconfigured_raises_error(self):
    domain = mbi.Domain(["A"], [2])
    dataset = mbi.Dataset({"A": np.array([0, 1] * 8, dtype=np.int32)}, domain)
    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[],
        num_layers=1,
        n_head=1,
        emb_dim=8,
    )
    mechanism = synth_transformer.TabularTransformer(
        config=config,
        num_epochs=5,
        num_iterations=5,
        learning_rate=1e-2,
    )
    rng = np.random.default_rng(0)
    with self.assertRaisesRegex(
        ValueError, "Mechanism has not been configured"
    ):
      mechanism(rng, dataset)

    with self.assertRaisesRegex(
        ValueError, "Mechanism has not been configured"
    ):
      _ = mechanism.dp_event

  def test_training_is_stochastic_with_dropout(self):
    """Verifies that dropout introduces stochasticity during training."""
    dataset = jnp.array([[0], [1], [0], [1]] * 16, dtype=jnp.int32)
    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[2],
        num_layers=1,
        n_head=1,
        emb_dim=8,
        dropout=0.5,
    )
    # Instantiate two models with same initial weights
    model1 = synth_transformer.TabularTransformerModel(config, seed=42)
    model2 = synth_transformer.TabularTransformerModel(config, seed=42)

    # Change seed of model2 so it uses different dropout masks
    model2._key = jax.random.key(100)

    # Fit both models
    model1.fit(dataset, num_epochs=5, num_iterations=5, learning_rate=1e-2)
    model2.fit(dataset, num_epochs=5, num_iterations=5, learning_rate=1e-2)

    # Extract final states
    state1_final = nnx.state(model1.model)
    state2_final = nnx.state(model2.model)

    # If dropout is active (train=True), states must have diverged.
    leaves1, _ = jax.tree_util.tree_flatten(state1_final)
    leaves2, _ = jax.tree_util.tree_flatten(state2_final)

    any_different = False
    for l1, l2 in zip(leaves1, leaves2):
      if not jnp.allclose(l1, l2, atol=1e-5, rtol=1e-5):
        any_different = True
        break
    self.assertTrue(any_different, "Models did not diverge despite dropout!")

  def test_dp_event_calibrated(self):
    config = synth_transformer.TabularTransformerConfig(
        num_elements_per_feature=[2],
        num_layers=1,
        n_head=1,
        emb_dim=8,
    )
    mechanism = synth_transformer.TabularTransformer(
        config=config,
        num_epochs=1,
        num_iterations=1,
    )

    # 1. Non-private configuration
    non_private_mechanism = mechanism.configure(budget=float("inf"))
    self.assertIsInstance(
        non_private_mechanism.dp_event, dp_accounting.NonPrivateDpEvent
    )

    # 2. Private configuration
    private_mechanism = mechanism.configure(budget=1.0)
    # The returned event should be the precise composed event from JAX Privacy,
    # which is not a simple ZCDpEvent.
    self.assertIsInstance(private_mechanism.dp_event, dp_accounting.DpEvent)
    self.assertNotIsInstance(
        private_mechanism.dp_event, dp_accounting.ZCDpEvent
    )


if __name__ == "__main__":
  absltest.main()
