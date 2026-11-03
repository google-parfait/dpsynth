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

"""Unit tests for TabularTransformerConfig and TabularTransformer."""

import typing
from absl.testing import absltest
import dp_accounting
from dpsynth import training
from dpsynth.training import _tabular_transformer as synth_transformer
import jax
import jax.numpy as jnp
from jax_privacy import execution_plan
import mbi
import numpy as np

# Disable compilation cache to prevent SIGBUS write errors in sandbox /tmp
jax.config.update("jax_enable_compilation_cache", False)


class TabularTransformerTest(absltest.TestCase):

  def test_init_model_and_loss(self):
    domain = mbi.Domain(["A", "B"], [3, 2])
    config = synth_transformer.TabularTransformerConfig(
        num_layers=1,
        n_head=2,
        emb_dim=16,
        dropout=0.1,
    )
    rng = np.random.default_rng(42)
    _, params, _, loss_fn = synth_transformer._init_model_and_loss(
        rng, config, domain
    )
    example = {"x": jnp.array([1, 0], dtype=jnp.int32)}
    loss, aux = loss_fn(params, example, jax.random.key(123))
    self.assertEqual(loss.shape, ())
    self.assertEqual(aux, {})

  def test_sample_flat_nnx(self):
    domain = mbi.Domain(["A", "B"], [3, 2])
    config = synth_transformer.TabularTransformerConfig(
        num_layers=1,
        n_head=2,
        emb_dim=16,
    )
    rng = np.random.default_rng(42)
    graphdef, params, frozen, _ = synth_transformer._init_model_and_loss(
        rng, config, domain
    )
    key = jax.random.key(0)

    samples_exact = synth_transformer.sample_flat_nnx(
        params, frozen, graphdef, 256, 2, 256, key
    )
    self.assertEqual(samples_exact.shape, (256, 2))

    samples_custom = synth_transformer.sample_flat_nnx(
        params, frozen, graphdef, 10, 2, 10, key
    )
    self.assertEqual(samples_custom.shape, (10, 2))
    self.assertTrue(jnp.all(samples_custom[:, 0] < 3))
    self.assertTrue(jnp.all(samples_custom[:, 1] < 2))

    samples_chunked = synth_transformer.sample_flat_nnx(
        params, frozen, graphdef, 30, 2, 10, key
    )
    self.assertEqual(samples_chunked.shape, (30, 2))

    with self.assertRaisesRegex(ValueError, "must be a multiple of"):
      synth_transformer.sample_flat_nnx(
          params, frozen, graphdef, 15, 2, 10, key
      )

  def test_e2e_correlation(self):
    """Verifies that TabularTransformer learns conditional correlations."""
    num_samples = 256
    key = jax.random.key(42)
    k0, k1 = jax.random.split(key, 2)

    col0 = np.asarray(
        jax.random.randint(k0, shape=(num_samples,), minval=0, maxval=2),
        dtype=np.int32,
    )
    probs_col1 = np.array([0.1, 0.9])[col0]
    col1 = (
        np.asarray(jax.random.uniform(k1, shape=(num_samples,))) < probs_col1
    ).astype(np.int32)

    orig_p_col1_given_col0_0 = float(np.mean(col1[col0 == 0]))
    orig_p_col1_given_col0_1 = float(np.mean(col1[col0 == 1]))

    domain = mbi.Domain(["col0", "col1"], [2, 2])
    dataset = mbi.Dataset(
        typing.cast(typing.Any, {"col0": col0, "col1": col1}), domain
    )
    config = synth_transformer.TabularTransformerConfig(
        num_layers=1,
        n_head=2,
        emb_dim=16,
        num_epochs=100,
        num_iterations=100,
        learning_rate=5e-3,
    )
    mechanism = synth_transformer.TabularTransformer(
        config=config,
        mechanism_config=execution_plan.NonPrivateConfig(
            iterations=100, batch_size=num_samples
        ),
    )
    result = mechanism(np.random.default_rng(0), dataset)
    assert result.synthetic_data is not None
    synth_dict = result.synthetic_data.to_dict()
    synth_col0 = synth_dict["col0"]
    synth_col1 = synth_dict["col1"]

    synth_p_col1_given_col0_0 = float(np.mean(synth_col1[synth_col0 == 0]))
    synth_p_col1_given_col0_1 = float(np.mean(synth_col1[synth_col0 == 1]))

    self.assertAlmostEqual(
        synth_p_col1_given_col0_0, orig_p_col1_given_col0_0, delta=0.15
    )
    self.assertAlmostEqual(
        synth_p_col1_given_col0_1, orig_p_col1_given_col0_1, delta=0.15
    )

  def test_e2e_independent_distribution(self):
    """Verifies that TabularTransformer learns categorical marginals."""
    num_samples = 256
    rng = np.random.default_rng(42)
    col0 = rng.choice(3, size=num_samples, p=[0.2, 0.5, 0.3]).astype(np.int32)
    col1 = rng.choice(4, size=num_samples, p=[0.1, 0.2, 0.4, 0.3]).astype(
        np.int32
    )
    col2 = rng.choice(2, size=num_samples, p=[0.7, 0.3]).astype(np.int32)

    domain = mbi.Domain(["col0", "col1", "col2"], [3, 4, 2])
    dataset = mbi.Dataset(
        typing.cast(typing.Any, {"col0": col0, "col1": col1, "col2": col2}),
        domain,
    )
    config = synth_transformer.TabularTransformerConfig(
        num_layers=1,
        n_head=2,
        emb_dim=16,
        num_epochs=100,
        num_iterations=100,
        learning_rate=5e-3,
    )
    mechanism = synth_transformer.TabularTransformer(
        config=config,
        mechanism_config=execution_plan.NonPrivateConfig(
            iterations=100, batch_size=num_samples
        ),
    )
    result = mechanism(np.random.default_rng(0), dataset)
    assert result.synthetic_data is not None
    synth_dict = result.synthetic_data.to_dict()

    for col, k_max in [("col0", 3), ("col1", 4), ("col2", 2)]:
      orig_col = dataset.to_dict()[col]
      synth_col = synth_dict[col]
      self.assertTrue(np.all((synth_col >= 0) & (synth_col < k_max)))
      for k in range(k_max):
        self.assertAlmostEqual(
            float(np.mean(synth_col == k)),
            float(np.mean(orig_col == k)),
            delta=0.12,
        )

  def test_tabular_transformer_mechanism_dp_and_dropout(self):
    """Verifies DP-SGD training and dropout with mbi.Dataset."""
    domain = mbi.Domain(["A", "B"], [2, 2])
    data_dict = {
        "A": np.array([0, 1, 0, 1] * 16, dtype=np.int32),
        "B": np.array([1, 0, 1, 0] * 16, dtype=np.int32),
    }
    dataset = mbi.Dataset(typing.cast(typing.Any, data_dict), domain)

    config = synth_transformer.TabularTransformerConfig(
        num_layers=1,
        n_head=1,
        emb_dim=8,
        dropout=0.1,
        num_epochs=5,
        num_iterations=5,
        learning_rate=1e-2,
    )
    mechanism = config.configure(budget=1.0)
    result = mechanism(np.random.default_rng(0), dataset)
    assert result.synthetic_data is not None
    self.assertIsNone(result.model)
    self.assertEqual(result.synthetic_data.domain, domain)
    self.assertEqual(result.synthetic_data.records, 64)

  def test_dp_event_calibrated(self):
    config = training.TabularTransformerConfig(
        num_layers=1,
        n_head=1,
        emb_dim=8,
        num_epochs=1,
        num_iterations=1,
    )
    private_mechanism = config.configure(budget=1.0)
    self.assertIsInstance(private_mechanism.dp_event, dp_accounting.DpEvent)
    self.assertNotIsInstance(
        private_mechanism.dp_event, dp_accounting.ZCDpEvent
    )

    non_private_mechanism = training.TabularTransformer(
        config=config,
        mechanism_config=execution_plan.NonPrivateConfig(
            iterations=1, batch_size=1
        ),
    )
    self.assertIsInstance(
        non_private_mechanism.dp_event, dp_accounting.NonPrivateDpEvent
    )


if __name__ == "__main__":
  absltest.main()
