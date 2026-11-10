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
from dpsynth.training import _optimizers
import jax
import jax.numpy as jnp
import optax


class DenoisedAdamWTest(absltest.TestCase):

  def test_step_zero_matches_normalized_sign_and_jit(self):
    opt = _optimizers.scale_by_denoised_adam(b1=0.0, b2=0.999, eps=0.0)
    params = {'w': jnp.array([[2.0, -4.0], [0.5, -1.0]], dtype=jnp.float32)}
    grads = {'w': jnp.array([[3.0, -6.0], [0.25, -0.5]], dtype=jnp.float32)}
    state = jax.jit(opt.init)(params)
    updates, _ = jax.jit(opt.update)(grads, state, params)
    self.assertTrue(
        bool(jnp.allclose(updates['w'], jnp.sign(grads['w']), atol=1e-5))
    )

  def test_zero_denoise_beta_matches_standard_adam(self):
    opt_denoised = _optimizers.scale_by_denoised_adam(denoise_beta=0.0)
    opt_adam = optax.scale_by_adam()
    params = {'w': jnp.array([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32)}
    s_denoised = opt_denoised.init(params)
    s_adam = opt_adam.init(params)
    for step in range(6):
      grads = {'w': jax.random.normal(jax.random.key(step), (4,))}
      u_denoised, s_denoised = opt_denoised.update(grads, s_denoised, params)
      u_adam, s_adam = opt_adam.update(grads, s_adam, params)
      self.assertTrue(
          bool(jnp.allclose(u_denoised['w'], u_adam['w'], atol=1e-6))
      )

  def test_ema_denoising_suppresses_alternating_noise_in_nu(self):
    params = {'w': jnp.zeros((4,), dtype=jnp.float32)}
    g_pos = {'w': jnp.ones((4,), dtype=jnp.float32)}
    g_neg = {'w': -jnp.ones((4,), dtype=jnp.float32)}

    opt_denoised = _optimizers.scale_by_denoised_adam(denoise_beta=0.85)
    opt_raw = _optimizers.scale_by_denoised_adam(denoise_beta=0.0)
    s_denoised = opt_denoised.init(params)
    s_raw = opt_raw.init(params)
    for step in range(8):
      g = g_pos if step % 2 == 0 else g_neg
      _, s_denoised = opt_denoised.update(g, s_denoised, params)
      _, s_raw = opt_raw.update(g, s_raw, params)

    assert isinstance(s_denoised, _optimizers.ScaleByDenoisedAdamState)
    assert isinstance(s_raw, _optimizers.ScaleByDenoisedAdamState)
    nu_denoised = float(jnp.mean(s_denoised.nu['w']))
    nu_raw = float(jnp.mean(s_raw.nu['w']))
    self.assertLess(nu_denoised, 0.25 * nu_raw)

  def test_denoised_adamw_finite_updates_and_dtypes(self):
    opt = _optimizers.denoised_adamw(learning_rate=3e-3, weight_decay=1e-4)
    params = {
        'w32': jnp.ones((8, 4), dtype=jnp.float32),
        'w16': jnp.ones((8, 4), dtype=jnp.bfloat16),
    }
    grads = {
        'w32': jax.random.normal(jax.random.key(0), (8, 4), dtype=jnp.float32),
        'w16': jax.random.normal(jax.random.key(1), (8, 4), dtype=jnp.bfloat16),
    }
    state = opt.init(params)
    for _ in range(4):
      updates, state = opt.update(grads, state, params)
      self.assertEqual(updates['w32'].dtype, jnp.float32)
      self.assertEqual(updates['w16'].dtype, jnp.bfloat16)
      for leaf in jax.tree.leaves(updates):
        self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))

  def test_converges_on_noisy_quadratic(self):
    target = jnp.array([1.5, -2.0, 0.5, 3.0], dtype=jnp.float32)
    scales = jnp.array([1.0, 4.0, 0.25, 2.0], dtype=jnp.float32)
    loss_fn = lambda w: 0.5 * jnp.sum(scales * jnp.square(w - target))

    def run_opt(opt):
      params = {'w': jnp.zeros_like(target)}
      state = opt.init(params)
      for step in range(150):
        noise = 0.2 * (1.0 if step % 2 == 0 else -1.0)
        grads = {'w': scales * (params['w'] - target) + noise}
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
      return float(loss_fn(params['w']))

    loss_denoised = run_opt(_optimizers.denoised_adamw(learning_rate=0.05))
    loss_adamw = run_opt(optax.adamw(learning_rate=0.05))
    self.assertLess(loss_denoised, 1e-2)
    self.assertLessEqual(loss_denoised, loss_adamw + 1e-3)


if __name__ == '__main__':
  absltest.main()
