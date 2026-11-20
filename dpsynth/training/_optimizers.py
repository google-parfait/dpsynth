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

import optax


def denoised_adamw(
    learning_rate: optax.ScalarOrSchedule = 1e-3,
    *,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
) -> optax.GradientTransformation:
  """Constructs an EMA-denoised AdamW optimizer for noisy DP gradients.

  In differentially private training, squaring raw single-step gradients for
  Adam's second moment rectifies zero-mean DP noise (both i.i.d. DP-SGD noise
  and negatively correlated BandMF noise) so that noise variance adds at every
  step instead of canceling. Applying optax.ema before optax.scale_by_rms
  cancels most of this zero-mean noise while it is still signed, allowing the
  second moment to track signal curvature while keeping optimizer state at 2x
  model parameters.

  Note: Because the second moment is estimated from the EMA-smoothed gradient,
  per-step updates are larger than optax.adamw in noise-dominated coordinates,
  so tuned learning rates are typically lower than for optax.adamw.

  Args:
    learning_rate: Scalar learning rate or schedule.
    b1: Exponential decay rate for gradient EMA smoothing and momentum.
    b2: Exponential decay rate for the denoised second moment.
    eps: Small additive constant outside the square root in the denominator.
    weight_decay: Decoupled weight decay coefficient.

  Returns:
    An optax.GradientTransformation chaining EMA denoising, bias-corrected RMS
    scaling, decoupled weight decay, and learning-rate scaling.
  """
  # Equivalent to optax.chain(optax.ema(b1), optax.adamw(..., b1=0.0)), using
  # scale_by_rms to spare scale_by_adam's unused b1=0 first-moment buffer.
  return optax.chain(
      optax.ema(decay=b1),
      optax.scale_by_rms(
          decay=b2, eps=eps, eps_in_sqrt=False, bias_correction=True
      ),
      optax.add_decayed_weights(weight_decay),
      optax.scale_by_learning_rate(learning_rate),
  )
