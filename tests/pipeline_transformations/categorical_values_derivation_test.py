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
from dpsynth.core import domain
from dpsynth.pipeline_transformations import categorical_values_derivation
import pipeline_dp


class DeriveCategoricalValuesTest(absltest.TestCase):

  def test_derive_categorical_values(self):
    backend = pipeline_dp.LocalBackend()
    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=5.0, total_delta=1e-10
    )
    dp_engine = pipeline_dp.DPEngine(accountant, backend)

    # The first 100 rows are from 100 privacy units, so the will be chosen.
    # The last row is from different privacy unit, so it will be dropped.
    input_data = [("A", 1, 10), (3, 0, 5), (True, 1, 2)] * 100 + [("C", 0, 3)]
    got = categorical_values_derivation.derive_categorical_values(
        input_data=input_data,
        backend=backend,
        dp_engine=dp_engine,
        attribute_keys_to_derive=[0, 2],
    )
    accountant.compute_budgets()
    got = list(got)
    self.assertEqual(
        got[0],
        {
            0: domain.CategoricalAttribute(("<OOD>", "3", "A", "True")),
            2: domain.CategoricalAttribute(("<OOD>", "10", "2", "5")),
        },
    )


if __name__ == "__main__":
  absltest.main()
