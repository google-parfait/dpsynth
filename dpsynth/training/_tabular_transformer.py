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

"""Differentially private tabular generation via a Flax NNX Transformer."""

from __future__ import annotations

import dataclasses
import math
import typing

import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import common as dm_common
from dpsynth.training import _trainer as dp_trainer
from dpsynth.training import _transformer_model as transformer_model
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

  num_layers: int = 4
  n_head: int = 8
  emb_dim: int = 64
  dropout: float = 0.0
  learning_rate: float = 1e-3
  num_epochs: int = 10
  num_iterations: int = 1000
  clipping_norm: float = 1.0

  def configure(
      self, _=None, *, budget: float, delta: float = 0.0, **kwargs: typing.Any
  ) -> TabularTransformer:
    del delta, kwargs
    noise_multiplier = math.sqrt(self.num_iterations / (2.0 * budget))
    mechanism_config = execution_plan.BandMFConfig.default(
        num_bands=1,
        iterations=self.num_iterations,
        expected_participations=self.num_epochs,
        l2_clip_norm=self.clipping_norm,
        noise_multiplier=noise_multiplier,
    )
    return TabularTransformer(config=self, mechanism_config=mechanism_config)


def _init_model_and_loss(
    rng: np.random.Generator,
    config: TabularTransformerConfig,
    domain: mbi.Domain,
):
  """Initializes the NNX Transformer and returns its state and per-example loss."""
  feature_bounds = [domain.size((col,)) for col in domain.attributes]
  init_key, dropout_key = jax.random.split(
      jax.random.key(int(rng.integers(2**31)))
  )
  model = transformer_model.TransformerModel(
      num_elements_per_feature=feature_bounds,
      num_layers=config.num_layers,
      n_head=config.n_head,
      emb_dim=config.emb_dim,
      dropout=config.dropout,
      rngs=nnx.Rngs(params=init_key, dropout=dropout_key),
  )
  graphdef, trainable_params, frozen_params = nnx.split(model, nnx.Param, ...)

  def loss_fn(params, example, prng):
    merged = nnx.merge(graphdef, params, frozen_params)
    return merged.compute_loss_single(example["x"], train=True, key=prng), {}

  return graphdef, trainable_params, frozen_params, loss_fn


@jax.jit(static_argnums=(3, 4, 5))
def sample_flat_nnx(
    trainable,
    frozen,
    graphdef,
    num_rows: int,
    num_features: int,
    microbatch_size: int,
    key: jax.Array,
) -> jax.Array:
  """Generates a synthetic data batch column-by-column."""
  if num_rows % microbatch_size != 0:
    raise ValueError(
        f"num_rows ({num_rows}) must be a multiple of microbatch_size"
        f" ({microbatch_size})."
    )

  model = nnx.merge(graphdef, trainable, frozen)

  def sample_single(single_key):
    keys = jax.random.split(single_key, num_features)
    c = model.header(jnp.ones((1, 1)))[0]

    sampled_cols = []
    for i in range(num_features):
      padding = [jnp.zeros((), dtype=jnp.int32)] * (num_features - i)
      x = jnp.stack(sampled_cols + padding)

      h_e = model.encode(x, train=False, key=None)
      h_d = model.decode(c, h_e, train=False, key=None)

      w = model.embeddings[i].embedding.value
      logits = jnp.dot(h_d[i], w.T)
      sampled_val = jax.random.categorical(keys[i], logits).astype(jnp.int32)
      sampled_cols.append(sampled_val)

    return jnp.stack(sampled_cols)

  keys = jax.random.split(key, num_rows)
  return optax.microbatching.micro_vmap(
      sample_single, microbatch_size=microbatch_size
  )(keys)


def _sample_dataset(
    rng: np.random.Generator,
    trainable,
    frozen,
    graphdef,
    domain: mbi.Domain,
    num_rows: int,
    microbatch_size: int = 256,
) -> mbi.Dataset:
  """Post-processes trained model parameters by sampling an mbi.Dataset."""
  columns = list(domain.attributes)
  padded_rows = (
      (num_rows + microbatch_size - 1) // microbatch_size
  ) * microbatch_size
  sample_key = jax.random.key(int(rng.integers(2**31)))
  tokens = sample_flat_nnx(
      trainable,
      frozen,
      graphdef,
      padded_rows,
      len(columns),
      microbatch_size,
      sample_key,
  )[:num_rows]
  synthetic_dict = {
      col: np.array(tokens[:, i], dtype=np.int32)
      for i, col in enumerate(columns)
  }
  return mbi.Dataset(synthetic_dict, domain)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TabularTransformer(api.CalibratedMechanism):
  """Differentially private tabular generator via generative Transformer.

  This class implements the dpsynth CalibratedMechanism interface, making it
  directly compatible with TabularConfig.
  """

  config: TabularTransformerConfig
  mechanism_config: execution_plan.ExecutionPlanConfig

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    return self.mechanism_config.make().dp_event

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
    del initial_measurements, initial_potentials, constraints, kwargs
    # 1-to-1 per-record mapping preserves record-level differential privacy.
    columns = list(data.domain.attributes)
    discrete_dict = data.to_dict()
    rows = jnp.stack([discrete_dict[col] for col in columns], axis=1)

    graphdef, trainable_params, frozen_params, loss_fn = _init_model_and_loss(
        rng, self.config, data.domain
    )
    trainer_config = dp_trainer.DPTrainerConfig(
        init_params=typing.cast(typing.Any, trainable_params),
        loss_fn=loss_fn,
        mechanism_config=self.mechanism_config,
        optimizer=optax.adam(self.config.learning_rate),
        performance_flags=execution_plan.PerformanceFlags(keep_batch_dim=False),
    )
    state = dp_trainer.DPTrainer(trainer_config)(rng, {"x": rows})

    synthetic_dataset = _sample_dataset(
        rng,
        state.params,
        frozen_params,
        graphdef,
        data.domain,
        num_rows=len(rows),
    )
    return dm_common.DiscreteMechanismResult(
        model=typing.cast(mbi.Model, None),
        synthetic_data=synthetic_dataset,
    )
