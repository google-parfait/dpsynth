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

"""Stateful Tabular Generative Transformer Model in Flax NNX.

This module defines the public user-facing API for training JAX Transformer
models on tabular data using Flax NNX.
"""

import dataclasses
import pickle
import typing

import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import common as dm_common
from dpsynth.experimental.synthetic_transformer import transformer_model
from dpsynth.text import dp_trainer
from flax import nnx
import jax
import jax.numpy as jnp
from jax_privacy import execution_plan
import mbi
import numpy as np
import optax
import optax.microbatching


@dataclasses.dataclass(frozen=True)
class TabularTransformerConfig(api.MechanismConfig):
  """Configuration for the Tabular Transformer."""

  num_elements_per_feature: typing.Sequence[int] | None = None
  num_layers: int = 4
  n_head: int = 8
  emb_dim: int = 64
  dropout: float = 0.0

  def configure(self, _=None, *, budget, delta=0) -> "TabularTransformer":
    return TabularTransformer(config=self, zcdp_rho=budget)

  def __post_init__(self):
    if self.num_elements_per_feature is not None:
      # Convert list to tuple to ensure hashability if needed
      object.__setattr__(
          self, "num_elements_per_feature", tuple(self.num_elements_per_feature)
      )


# --- Stateful Model Interface ---


class TabularTransformerModel:
  """Class that encapsulates JAX Transformer model on tabular data.

  This class encapsulates all logic to initialize the Transformer model and to
  sample synthetic data from it.
  """

  def __init__(
      self,
      config: TabularTransformerConfig,
      seed: int = 0,
  ):
    """Instantiates the JAX model class.

    Args:
        config: A fully-defined TabularTransformerConfig layout.
        seed: An integer seed for parameter initialization.
    """
    if not config.num_elements_per_feature:
      raise ValueError("num_elements_per_feature must be non-empty.")
    self._config = config
    self._is_initialized = False
    self._key = jax.random.key(seed)

    # Always initialize the model (Flax NNX supports empty features init)
    self._init_model()

  def _init_model(self):
    """Initializes the inner NNX model."""
    self._key, init_key, dropout_key = jax.random.split(self._key, 3)
    rngs = nnx.Rngs(params=init_key, dropout=dropout_key)
    num_elements = typing.cast(
        typing.Sequence[int], self._config.num_elements_per_feature
    )
    self.model = transformer_model.TransformerModel(
        num_elements_per_feature=num_elements,
        num_layers=self._config.num_layers,
        n_head=self._config.n_head,
        emb_dim=self._config.emb_dim,
        dropout=self._config.dropout,
        rngs=rngs,
    )

  def fit(
      self,
      dataset: jax.typing.ArrayLike,
      num_epochs: int = 200,
      num_iterations: int = 1000,
      learning_rate: float = 1e-3,
      zcdp_rho: float | None = None,
      clipping_norm: float = 1.0,
  ) -> None:
    """Trains the generative model.

    Args:
        dataset: A 2D integer array-like of shape (num_samples, num_features).
        num_epochs: Number of training epochs (passes over the data).
        num_iterations: Total number of gradient updates (steps).
        learning_rate: SGD optimizer rate.
        zcdp_rho: The zCDP privacy budget (rho).
        clipping_norm: Gradient clipping threshold for DP-SGD.
    """
    x_batch = jnp.asarray(dataset, dtype=jnp.int32)

    use_dp_sgd = zcdp_rho is not None and zcdp_rho != float("inf")

    graphdef, trainable, frozen = nnx.split(self.model, nnx.Param, ...)

    # Per-example loss function (no manual vmap, keep_batch_dim=False)
    def dp_loss_fn(params, data, prng):
      merged_model = nnx.merge(graphdef, params, frozen)
      loss = merged_model.compute_loss_single(data["x"], train=True, key=prng)
      return loss, {}

    self._key, training_key = jax.random.split(self._key)
    seed = int(jax.random.randint(training_key, (), 0, 2**31 - 1))

    # We use keep_batch_dim=False to allow compute_loss_single to be
    # vectorized by jax_privacy.
    performance_flags = execution_plan.PerformanceFlags(keep_batch_dim=False)

    if use_dp_sgd:
      if zcdp_rho is None:
        raise ValueError("zcdp_rho must be provided if use_dp_sgd=True")

      expected_participations = num_epochs
      mechanism_config = execution_plan.BandMFConfig.default(
          num_bands=1,
          iterations=num_iterations,
          expected_participations=expected_participations,
          l2_clip_norm=clipping_norm,
      )
    else:
      # Standard training using NonPrivateConfig.
      train_size, _ = x_batch.shape
      batch_size = int(train_size * (num_epochs / num_iterations))
      batch_size = max(1, min(batch_size, train_size))

      mechanism_config = execution_plan.NonPrivateConfig(
          iterations=num_iterations,
          batch_size=batch_size,
      )

    trainer = dp_trainer.DPTrainer(
        init_params=typing.cast(typing.Any, trainable),
        loss_fn=dp_loss_fn,
        mechanism_config=mechanism_config,
        optimizer=optax.adam(learning_rate),
        performance_flags=performance_flags,
    ).configure(budget=zcdp_rho if use_dp_sgd else float("inf"))

    final_state = trainer(rng=seed, data={"x": x_batch})

    self.model = nnx.merge(graphdef, final_state.params, frozen)
    self._is_initialized = True

  def sample(
      self,
      num_rows: int,
      key: jax.Array | None = None,
      microbatch_size: int = 256,
  ) -> jax.Array:
    """Generates synthetic data.

    Args:
        num_rows: Quantity of synthetic records to generate.
        key: Optional JAX PRNG random key. If not provided, a random key will be
          generated automatically.
        microbatch_size: Configures the batch size of parallel JIT executions of
          sampling to prevent memory scaling issues (OOM).

    Returns:
        A 2D integer JAX array of shape (num_rows, num_features).
    """
    if not self._is_initialized:
      raise ValueError("Model must be trained using fit() before sampling.")

    if key is None:
      self._key, key = jax.random.split(self._key)

    # Split model to get parameters as PyTree (read-only inside JIT)
    graphdef, trainable, frozen = nnx.split(self.model, nnx.Param, ...)

    # Step-by-step autoregressive generation
    return sample_flat_nnx(
        trainable,
        frozen,
        graphdef,
        num_rows,
        self._config,
        microbatch_size,
        key,
    )

  def save(self, file_object: typing.BinaryIO):
    """Serializes the model to the file."""
    if not self._is_initialized:
      raise ValueError("Cannot save an untrained model.")

    # Get state as PyTree and convert to CPU
    state = nnx.state(self.model)
    state_cpu = jax.device_get(state)
    model_data = {"state": state_cpu, "config": self._config}
    pickle.dump(model_data, file_object)

  @classmethod
  def load(cls, file_object: typing.BinaryIO) -> "TabularTransformerModel":
    """Loads the model from the file."""
    model_data = pickle.load(file_object)  # pylint: disable=g-unsafe-pickle-load

    # Reconstruct configuration and instantiate model
    config = model_data["config"]
    model = cls(config)

    # Update state
    nnx.update(model.model, model_data["state"])
    model._is_initialized = True
    return model


# --- Batched Autoregressive Sampling Helper (NNX) ---


@jax.jit(static_argnums=(3, 4, 5))
def sample_flat_nnx(
    trainable,
    frozen,
    graphdef,
    num_rows,
    config,
    microbatch_size,
    key,
):
  """Generates synthetic data batch column-by-column (NNX JIT helper)."""
  if num_rows % microbatch_size != 0:
    raise ValueError(
        f"num_rows ({num_rows}) must be a multiple of microbatch_size"
        f" ({microbatch_size})."
    )

  model = nnx.merge(graphdef, trainable, frozen)
  num_features = len(config.num_elements_per_feature)

  def sample_single(single_key):
    keys = jax.random.split(single_key, num_features)
    c = model.header(jnp.ones((1, 1)))[0]  # (emb_dim,)

    sampled_cols = []
    for i in range(num_features):
      padding_len = num_features - i
      padding = [jnp.zeros((), dtype=jnp.int32)] * padding_len
      x = jnp.stack(sampled_cols + padding)  # (num_features,)

      h_e = model.encode(x, train=False, key=None)
      h_d = model.decode(c, h_e, train=False, key=None)

      w = model.embeddings[i].embedding.value
      logits = jnp.dot(h_d[i], w.T)  # (num_classes,)
      sampled_val = jax.random.categorical(keys[i], logits).astype(jnp.int32)
      sampled_cols.append(sampled_val)

    return jnp.stack(sampled_cols)

  keys = jax.random.split(key, num_rows)
  return optax.microbatching.micro_vmap(
      sample_single, microbatch_size=microbatch_size
  )(keys)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TabularTransformer(api.CalibratedMechanism):
  """Differentially private tabular generator via generative Transformer.

  This class implements the dpsynth DPMechanism interface, making it directly
  compatible with TabularSynthesizer.
  """

  config: TabularTransformerConfig  # pyrefly: ignore[bad-override]
  learning_rate: float = 1e-3
  num_epochs: int = 10
  num_iterations: int = 1000
  clipping_norm: float = 1.0

  # DPMechanism required attributes
  zcdp_rho: float | None = None  # pyrefly: ignore[bad-override]
  _configured_trainer: dp_trainer.DPTrainer | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )

  def configure(
      self, _=None, *, budget: float, delta: float = 0.0, **kwargs: typing.Any
  ) -> "TabularTransformer":
    del delta, kwargs
    new_obj = dataclasses.replace(self, zcdp_rho=budget)
    if budget == float("inf"):
      object.__setattr__(new_obj, "_configured_trainer", None)  # pylint: disable=protected-access
      return new_obj

    mechanism_config = execution_plan.BandMFConfig.default(
        num_bands=1,
        iterations=self.num_iterations,
        expected_participations=self.num_epochs,
        l2_clip_norm=self.clipping_norm,
    )

    def dummy_loss_fn(params, data, prng):
      del params, data, prng
      return 0.0, {}

    trainer = dp_trainer.DPTrainer(
        init_params={},
        loss_fn=dummy_loss_fn,
        mechanism_config=mechanism_config,
        optimizer=optax.identity(),
    ).configure(budget=budget)

    object.__setattr__(new_obj, "_configured_trainer", trainer)  # pylint: disable=protected-access
    return new_obj

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    if self.zcdp_rho is None:
      raise ValueError("Mechanism has not been configured (calibrated) yet.")
    if self.zcdp_rho == float("inf"):
      return dp_accounting.NonPrivateDpEvent()
    if self._configured_trainer is None:
      raise ValueError("Mechanism configured privately but no trainer found.")
    return self._configured_trainer.dp_event

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset,
      *,
      initial_measurements=None,
      initial_potentials=None,
      constraints=(),
      **kwargs: typing.Any,
  ) -> dm_common.DiscreteMechanismResult:
    if self.zcdp_rho is None:
      raise ValueError("Mechanism has not been configured (calibrated) yet.")

    # 1. Convert mbi.Dataset to numpy array (N, F)
    discrete_dict = data.to_dict()
    columns = list(data.domain.attributes)
    arrays = [discrete_dict[col] for col in columns]
    dataset_array = np.stack(arrays, axis=1)

    # 2. Configure model bounds based on dataset domain sizes
    feature_bounds = [data.domain.size((col,)) for col in columns]
    config = dataclasses.replace(
        self.config, num_elements_per_feature=feature_bounds
    )

    # 3. Create and train the model
    seed = int(rng.integers(0, 2**31 - 1))
    model = TabularTransformerModel(config, seed=seed)

    model.fit(
        dataset_array,
        num_epochs=self.num_epochs,
        num_iterations=self.num_iterations,
        learning_rate=self.learning_rate,
        zcdp_rho=self.zcdp_rho,
        clipping_norm=self.clipping_norm,
    )

    # 4. Autoregressively sample from the model
    num_rows = dataset_array.shape[0]
    microbatch_size = 256
    padded_num_rows = (
        (num_rows + microbatch_size - 1) // microbatch_size
    ) * microbatch_size

    sample_seed = int(rng.integers(0, 2**31 - 1))
    sample_key = jax.random.key(sample_seed)

    synthetic_tokens = model.sample(
        padded_num_rows, key=sample_key, microbatch_size=microbatch_size
    )[:num_rows]

    # 5. Convert back to mbi.Dataset and return result
    synthetic_dict = {}
    for i, col in enumerate(columns):
      synthetic_dict[col] = np.array(synthetic_tokens[:, i], dtype=np.int32)

    synthetic_dataset = mbi.Dataset(synthetic_dict, data.domain)
    return dm_common.DiscreteMechanismResult(
        model=typing.cast(mbi.Model, None),
        synthetic_data=synthetic_dataset,
    )
