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

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
from dpsynth import _calibration


class CalibrationTest(parameterized.TestCase):

  def test_with_group_size(self):
    with_group_size = _calibration.with_group_size

    g = dp_accounting.GaussianDpEvent(noise_multiplier=6.0)
    self.assertEqual(
        with_group_size(g, 3),
        dp_accounting.GaussianDpEvent(noise_multiplier=2.0),
    )
    self.assertIs(with_group_size(g, 1), g)

    l = dp_accounting.LaplaceDpEvent(noise_multiplier=6.0)
    self.assertEqual(
        with_group_size(l, 3),
        dp_accounting.LaplaceDpEvent(noise_multiplier=2.0),
    )

    e = dp_accounting.ExponentialMechanismDpEvent(epsilon=0.5)
    self.assertEqual(
        with_group_size(e, 4),
        dp_accounting.ExponentialMechanismDpEvent(epsilon=2.0),
    )

    z = dp_accounting.ZCDpEvent(rho=0.2)
    self.assertEqual(
        with_group_size(z, 3),
        dp_accounting.ZCDpEvent(rho=1.8),
    )
    with self.assertRaises(dp_accounting.UnsupportedEventError):
      with_group_size(dp_accounting.ZCDpEvent(rho=0.2, xi=0.5), 3)

    noop = dp_accounting.NoOpDpEvent()
    self.assertEqual(with_group_size(noop, 5), noop)

    non_priv = dp_accounting.NonPrivateDpEvent()
    self.assertEqual(with_group_size(non_priv, 5), non_priv)

    composed = dp_accounting.ComposedDpEvent([
        dp_accounting.SelfComposedDpEvent(g, 2),
        e,
    ])
    expected_composed = dp_accounting.ComposedDpEvent([
        dp_accounting.SelfComposedDpEvent(
            dp_accounting.GaussianDpEvent(noise_multiplier=2.0), 2
        ),
        dp_accounting.ExponentialMechanismDpEvent(epsilon=1.5),
    ])
    self.assertEqual(with_group_size(composed, 3), expected_composed)

    ed = dp_accounting.dp_event.EpsilonDeltaDpEvent(0.0, 1e-5)
    self.assertEqual(with_group_size(ed, 1), ed)
    with self.assertRaises(dp_accounting.UnsupportedEventError):
      with_group_size(ed, 2)

    with self.assertRaises(ValueError):
      with_group_size(g, 0)

  def test_as_zcdp(self):
    as_zcdp = _calibration._as_zcdp

    self.assertEqual(as_zcdp(dp_accounting.NoOpDpEvent()), (0.0, 0.0))
    self.assertEqual(
        as_zcdp(dp_accounting.GaussianDpEvent(noise_multiplier=0.5)),
        (2.0, 0.0),
    )
    self.assertEqual(
        as_zcdp(dp_accounting.ExponentialMechanismDpEvent(epsilon=4.0)),
        (2.0, 0.0),
    )
    self.assertEqual(as_zcdp(dp_accounting.ZCDpEvent(rho=1.5)), (1.5, 0.0))
    self.assertEqual(
        as_zcdp(dp_accounting.dp_event.EpsilonDeltaDpEvent(2.0, 1e-5)),
        (2.0, 1e-5),
    )

    composed = dp_accounting.ComposedDpEvent([
        dp_accounting.SelfComposedDpEvent(
            dp_accounting.ExponentialMechanismDpEvent(epsilon=2.0), 3
        ),
        dp_accounting.GaussianDpEvent(noise_multiplier=1.0),
        dp_accounting.dp_event.EpsilonDeltaDpEvent(0.0, 1e-5),
    ])
    self.assertEqual(as_zcdp(composed), (2.0, 1e-5))

    with self.assertRaises(dp_accounting.UnsupportedEventError):
      as_zcdp(dp_accounting.LaplaceDpEvent(noise_multiplier=1.0))
    with self.assertRaises(dp_accounting.UnsupportedEventError):
      as_zcdp(dp_accounting.ZCDpEvent(rho=1.0, xi=0.1))

  def test_parallel_compose_event(self):
    parallel = _calibration._parallel_compose_event

    self.assertEqual(parallel([]), dp_accounting.NoOpDpEvent())

    e1 = dp_accounting.GaussianDpEvent(noise_multiplier=1.0)  # rho = 0.5
    e2 = dp_accounting.ExponentialMechanismDpEvent(epsilon=4.0)  # rho = 2.0
    self.assertEqual(parallel([e1, e2]), dp_accounting.ZCDpEvent(rho=2.0))

    e3 = dp_accounting.ComposedDpEvent([
        dp_accounting.GaussianDpEvent(noise_multiplier=1.0),
        dp_accounting.dp_event.EpsilonDeltaDpEvent(0.0, 1e-4),
    ])
    self.assertEqual(
        parallel([e2, e3]),
        dp_accounting.ComposedDpEvent([
            dp_accounting.ZCDpEvent(rho=2.0),
            dp_accounting.dp_event.EpsilonDeltaDpEvent(0.0, 1e-4),
        ]),
    )


if __name__ == '__main__':
  absltest.main()
