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
from absl.testing import parameterized
from dpsynth.training import _optimizers
import jax
import jax.numpy as jnp
import optax


class DenoisedAdamWTest(parameterized.TestCase):

  def test_step_zero_matches_normalized_sign_and_jit(self):
    opt = _optimizers.denoised_adamw(
        learning_rate=1.0, b1=0.0, b2=0.999, eps=0.0, weight_decay=0.0
    )
    params = {'w': jnp.array([[2.0, -4.0], [0.5, -1.0]], dtype=jnp.float32)}
    grads = {'w': jnp.array([[3.0, -6.0], [0.25, -0.5]], dtype=jnp.float32)}
    state = jax.jit(opt.init)(params)
    updates, _ = jax.jit(opt.update)(grads, state, params)
    self.assertTrue(
        bool(jnp.allclose(updates['w'], -jnp.sign(grads['w']), atol=1e-5))
    )

  @parameterized.parameters(0.0, 0.85, 0.9)
  def test_matches_ema_chained_with_adamw(self, b1):
    opt_denoised = _optimizers.denoised_adamw(learning_rate=2e-3, b1=b1)
    opt_adamw = optax.chain(
        optax.ema(decay=b1), optax.adamw(learning_rate=2e-3, b1=0.0)
    )
    params = {'w': jnp.array([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32)}
    s_denoised = opt_denoised.init(params)
    s_adamw = opt_adamw.init(params)
    for step in range(6):
      grads = {'w': jax.random.normal(jax.random.key(step), (4,))}
      u_denoised, s_denoised = opt_denoised.update(grads, s_denoised, params)
      u_adamw, s_adamw = opt_adamw.update(grads, s_adamw, params)
      self.assertTrue(
          bool(jnp.allclose(u_denoised['w'], u_adamw['w'], atol=1e-6))
      )

  @parameterized.parameters(0.0, 0.85, 0.9)
  def test_matches_explicit_recurrence_and_uses_2x_state(self, b1):
    b2, eps, lr, wd = 0.999, 1e-8, 2e-3, 1e-4
    opt = _optimizers.denoised_adamw(
        learning_rate=lr, b1=b1, b2=b2, eps=eps, weight_decay=wd
    )
    params = {
        'w32': jnp.linspace(-1.0, 2.0, 8, dtype=jnp.float32),
        'w16': jnp.linspace(-1.0, 2.0, 8, dtype=jnp.bfloat16),
    }
    state = opt.init(params)
    param_arrays = [x for x in jax.tree.leaves(state) if x.shape == (8,)]
    self.assertLen(param_arrays, 4)

    mu = jax.tree.map(jnp.zeros_like, params)
    nu = jax.tree.map(jnp.zeros_like, params)
    lerp = lambda a, b, w: jnp.astype(w * a + (1.0 - w) * b, a.dtype)
    for step in range(1, 8):
      grads = {
          'w32': jax.random.normal(jax.random.key(step), (8,), jnp.float32),
          'w16': jax.random.normal(jax.random.key(step), (8,), jnp.bfloat16),
      }
      updates, state = opt.update(grads, state, params)

      mu = jax.tree.map(lambda m, g: lerp(m, g, b1), mu, grads)
      c1 = 1.0 - b1**step
      mu_hat = jax.tree.map(lambda x, c=c1: jnp.astype(x / c, x.dtype), mu)
      nu = jax.tree.map(lambda v, g: lerp(v, g**2, b2), nu, mu_hat)
      c2 = 1.0 - b2**step
      nu_hat = jax.tree.map(lambda x, c=c2: jnp.astype(x / c, x.dtype), nu)
      ref = jax.tree.map(
          lambda m, v, p: jnp.astype(
              -lr * (jnp.astype(m / (jnp.sqrt(v) + eps), m.dtype) + wd * p),
              p.dtype,
          ),
          mu_hat,
          nu_hat,
          params,
      )
      self.assertTrue(
          bool(jnp.allclose(updates['w32'], ref['w32'], rtol=1e-5, atol=1e-6))
      )
      self.assertTrue(
          bool(jnp.allclose(updates['w16'], ref['w16'], rtol=2e-2, atol=1e-4))
      )

  def test_ema_denoising_suppresses_alternating_noise_in_nu(self):
    params = {'w': jnp.zeros((4,), dtype=jnp.float32)}
    g_pos = {'w': jnp.ones((4,), dtype=jnp.float32)}
    g_neg = {'w': -jnp.ones((4,), dtype=jnp.float32)}

    opt_denoised = _optimizers.denoised_adamw(b1=0.9)
    opt_raw = _optimizers.denoised_adamw(b1=0.0)
    s_denoised = opt_denoised.init(params)
    s_raw = opt_raw.init(params)
    for step in range(8):
      g = g_pos if step % 2 == 0 else g_neg
      _, s_denoised = opt_denoised.update(g, s_denoised, params)
      _, s_raw = opt_raw.update(g, s_raw, params)

    nu_denoised = float(jnp.mean(optax.tree.get(s_denoised, 'nu')['w']))
    nu_raw = float(jnp.mean(optax.tree.get(s_raw, 'nu')['w']))
    self.assertLess(nu_denoised, 0.15 * nu_raw)

  def test_denoised_adamw_finite_updates_and_dtypes_under_scan(self):
    opt = _optimizers.denoised_adamw(learning_rate=3e-3, weight_decay=1e-4)
    params = {
        'w32': jnp.ones((8, 4), dtype=jnp.float32),
        'w16': jnp.ones((8, 4), dtype=jnp.bfloat16),
    }
    grads = {
        'w32': jax.random.normal(jax.random.key(0), (8, 4), dtype=jnp.float32),
        'w16': jax.random.normal(jax.random.key(1), (8, 4), dtype=jnp.bfloat16),
    }
    init_state = opt.init(params)
    step_fn = lambda s, _: tuple(reversed(opt.update(grads, s, params)))
    _, updates = jax.lax.scan(step_fn, init_state, None, length=4)
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
