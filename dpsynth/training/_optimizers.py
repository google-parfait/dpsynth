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

"""Denoised adaptive optimizers for differentially private training."""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


class ScaleByDenoisedAdamState(NamedTuple):
  """State for ``scale_by_denoised_adam``."""

  count: jax.Array
  mu: optax.Updates
  nu: optax.Updates
  grad_avg: optax.Updates


def _lerp(a: jax.Array, b: jax.Array, w: float) -> jax.Array:
  """Computes the convex combination w * a + (1 - w) * b in a.dtype."""
  return jnp.astype(w * a + (1.0 - w) * b, a.dtype)


def _bias_correct(tree: optax.Updates, decay: float, step: jax.Array):
  """Applies exponential moving average bias correction."""
  denom = 1.0 - jnp.power(decay, jnp.maximum(step, 1))
  return jax.tree.map(lambda x: jnp.astype(x / denom, x.dtype), tree)


def scale_by_denoised_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    denoise_beta: float = 0.85,
) -> optax.GradientTransformation:
  """Scales updates using Adam with an EMA-denoised second-moment estimate.

  In differentially private training, each step's privatized gradient is the
  sum of the clipped signal gradient and zero-mean DP noise (i.i.d. Gaussian
  noise under DP-SGD, or negatively correlated noise across consecutive steps
  under multi-band BandMF). While linear momentum naturally averages out
  zero-mean noise over time, squaring raw single-step gradients for Adam's
  second moment rectifies the noise signs so that noise variance adds to every
  step instead of canceling.

  Smoothing incoming gradients with a short linear EMA before squaring cancels
  most of this zero-mean noise while it is still signed, allowing the second
  moment to track the underlying signal curvature rather than the DP noise
  floor.

  Note: Because ``nu`` is estimated from the smoothed gradient while ``mu``
  averages the raw gradient, per-step updates are larger than Adam's in
  noise-dominated coordinates, so tuned learning rates are typically lower than
  for ``optax.adamw``. Maintaining ``grad_avg`` alongside ``mu`` and ``nu``
  also raises optimizer state from 2x to 3x model parameters.

  Args:
    b1: Exponential decay rate for the first moment.
    b2: Exponential decay rate for the denoised second moment.
    eps: Small additive constant in the preconditioner denominator.
    denoise_beta: Exponential decay rate for smoothing gradients before squaring
      for the second moment.

  Returns:
    An ``optax.GradientTransformation`` implementing EMA-denoised Adam.
  """
  lerp_g = functools.partial(_lerp, w=denoise_beta)
  lerp_nu = functools.partial(_lerp, w=b2)
  lerp_mu = functools.partial(_lerp, w=b1)
  scale = lambda m, v: jnp.astype(m / (jnp.sqrt(v) + eps), m.dtype)

  def init_fn(params: optax.Params) -> ScaleByDenoisedAdamState:
    z = jax.tree.map(jnp.zeros_like, params)
    return ScaleByDenoisedAdamState(jnp.zeros((), jnp.int32), z, z, z)

  def update_fn(updates, state: optax.OptState, params=None):
    del params
    assert isinstance(state, ScaleByDenoisedAdamState)
    step = state.count + 1

    # 1. Denoise incoming gradients with a short EMA while noise is signed.
    grad_avg = jax.tree.map(lerp_g, state.grad_avg, updates)
    g_clean = _bias_correct(grad_avg, denoise_beta, step)

    # 2. Estimate the second moment from the squared denoised gradient.
    nu = jax.tree.map(lerp_nu, state.nu, jax.tree.map(jnp.square, g_clean))
    nu_hat = _bias_correct(nu, b2, step)

    # 3. Accumulate first-moment momentum and precondition with denoised nu_hat.
    mu = jax.tree.map(lerp_mu, state.mu, updates)
    mu_hat = _bias_correct(mu, b1, step)
    out = jax.tree.map(scale, mu_hat, nu_hat)
    return out, ScaleByDenoisedAdamState(step, mu, nu, grad_avg)

  return optax.GradientTransformation(init_fn, update_fn)


def denoised_adamw(
    learning_rate: optax.ScalarOrSchedule = 1e-3,
    *,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    denoise_beta: float = 0.85,
) -> optax.GradientTransformation:
  """Constructs a denoised AdamW optimizer for noisy DP gradients.

  Note: Because ``nu`` is estimated from the EMA-smoothed gradient while ``mu``
  averages the raw gradient, per-step updates are larger than AdamW's in
  noise-dominated coordinates, so tuned learning rates are typically lower than
  for ``optax.adamw``. Maintaining ``grad_avg`` alongside ``mu`` and ``nu``
  also raises optimizer state from 2x to 3x model parameters.

  Args:
    learning_rate: Scalar learning rate or schedule.
    b1: Exponential decay rate for the first moment.
    b2: Exponential decay rate for the denoised second moment.
    eps: Small additive constant in the preconditioner denominator.
    weight_decay: Decoupled weight decay coefficient.
    denoise_beta: Exponential decay rate for smoothing gradients before squaring
      for the second moment.

  Returns:
    An ``optax.GradientTransformation`` chaining ``scale_by_denoised_adam``,
    decoupled weight decay, and learning-rate scaling.
  """
  return optax.chain(
      scale_by_denoised_adam(b1=b1, b2=b2, eps=eps, denoise_beta=denoise_beta),
      optax.add_decayed_weights(weight_decay),
      optax.scale_by_learning_rate(learning_rate),
  )
