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

"""Flax NNX model components for Tabular Transformer.

This module defines the underlying Transformer architecture layers using Flax
NNX.
"""

import typing
from flax import nnx
import jax
import jax.numpy as jnp


class MLPBlock(nnx.Module):
  """Multilayer Perceptron block."""

  def __init__(self, in_features: int, dropout: float = 0.0, *, rngs: nnx.Rngs):
    super().__init__()
    self.linear1 = nnx.Linear(
        in_features, in_features * 4, use_bias=True, rngs=rngs
    )
    self.linear2 = nnx.Linear(
        in_features * 4, in_features, use_bias=True, rngs=rngs
    )
    self.dropout = nnx.Dropout(rate=dropout)

  def __call__(
      self,
      x: jax.Array,
      *,
      train: bool = True,
      rngs: nnx.Rngs | None = None,
  ) -> jax.Array:
    x = self.linear1(x)
    x = jax.nn.gelu(x)
    x = self.linear2(x)
    x = self.dropout(x, deterministic=not train, rngs=rngs)
    return x


class TransformerBlock(nnx.Module):
  """Transformer block combining self-attention and MLP."""

  def __init__(
      self,
      emb_dim: int,
      n_head: int,
      dropout: float = 0.0,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.attn = nnx.MultiHeadAttention(
        num_heads=n_head,
        in_features=emb_dim,
        qkv_features=emb_dim,
        out_features=emb_dim,
        use_bias=True,
        dropout_rate=dropout,
        decode=False,
        rngs=rngs,
    )
    self.mlp = MLPBlock(in_features=emb_dim, dropout=dropout, rngs=rngs)
    self.norm1 = nnx.LayerNorm(num_features=emb_dim, epsilon=1e-5, rngs=rngs)
    self.norm2 = nnx.LayerNorm(num_features=emb_dim, epsilon=1e-5, rngs=rngs)

  def __call__(
      self,
      x: jax.Array,
      mask: jax.Array | None = None,
      *,
      train: bool = True,
      rngs: nnx.Rngs | None = None,
  ) -> jax.Array:
    x = x + self.attn(
        self.norm1(x), mask=mask, deterministic=not train, rngs=rngs
    )
    x = x + self.mlp(self.norm2(x), train=train, rngs=rngs)
    return x


class FlatAttentionBlock(nnx.Module):
  """Sequence of Transformer blocks with causal masking."""

  def __init__(
      self,
      num_layers: int,
      emb_dim: int,
      n_head: int,
      dropout: float = 0.0,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.layers = nnx.List([
        TransformerBlock(
            emb_dim=emb_dim,
            n_head=n_head,
            dropout=dropout,
            rngs=rngs,
        )
        for _ in range(num_layers)
    ])

  def __call__(
      self, x: jax.Array, *, train: bool = True, rngs: nnx.Rngs | None = None
  ) -> jax.Array:
    seq_len = x.shape[-2]
    mask = nnx.make_causal_mask(jnp.ones(seq_len))
    for layer in self.layers:
      x = layer(x, mask=mask, train=train, rngs=rngs)
    return x


class TransformerModel(nnx.Module):
  """Core Transformer Model containing layers and parameters."""

  def __init__(
      self,
      num_elements_per_feature: typing.Sequence[int],
      num_layers: int,
      n_head: int,
      emb_dim: int,
      dropout: float = 0.0,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.embeddings = nnx.List([
        nnx.Embed(num_embeddings=num_classes, features=emb_dim, rngs=rngs)
        for num_classes in num_elements_per_feature
    ])
    self.h_e = FlatAttentionBlock(
        num_layers=num_layers,
        emb_dim=emb_dim,
        n_head=n_head,
        dropout=dropout,
        rngs=rngs,
    )
    self.h_d = FlatAttentionBlock(
        num_layers=num_layers,
        emb_dim=emb_dim,
        n_head=n_head,
        dropout=dropout,
        rngs=rngs,
    )
    self.header = nnx.Linear(1, emb_dim, use_bias=False, rngs=rngs)
    self.emb_dim = emb_dim
    self.num_elements_per_feature = num_elements_per_feature

  def encode(
      self,
      x_row: jax.Array,
      *,
      train: bool = True,
      key: jax.Array | None = None,
  ) -> jax.Array:
    e_list = [
        self.embeddings[i](x_row[i])
        for i in range(len(self.num_elements_per_feature))
    ]
    e_seq = jnp.stack(e_list, axis=0)
    rngs = nnx.Rngs(dropout=key) if key is not None else None
    h_e = self.h_e(e_seq, train=train, rngs=rngs)
    return h_e

  def decode(
      self,
      c_row: jax.Array,
      h_e: jax.Array,
      *,
      train: bool = True,
      key: jax.Array | None = None,
  ) -> jax.Array:
    c_expanded = jnp.expand_dims(c_row, axis=0)
    inp_to_attention = jnp.concatenate([c_expanded, h_e[:-1]], axis=0)
    rngs = nnx.Rngs(dropout=key) if key is not None else None
    h_d = self.h_d(inp_to_attention, train=train, rngs=rngs)
    return h_d

  def compute_loss_single(
      self,
      x_row: jax.Array,
      *,
      train: bool = True,
      key: jax.Array | None = None,
  ) -> jax.Array:
    c_row = self.header(jnp.ones((1, 1)))[0]

    if key is not None:
      k_enc, k_dec = jax.random.split(key, 2)
    else:
      k_enc, k_dec = None, None

    h_e = self.encode(x_row, train=train, key=k_enc)
    h_d = self.decode(c_row, h_e, train=train, key=k_dec)

    loss = jnp.array(0.0)
    for i in range(len(self.num_elements_per_feature)):
      w = self.embeddings[i].embedding.value
      logits = jnp.dot(w, h_d[i])
      target = x_row[i]
      loss += -logits[target] + jax.nn.logsumexp(logits)

    return loss
