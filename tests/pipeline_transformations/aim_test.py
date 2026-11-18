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

from unittest import mock

from absl.testing import absltest
from dpsynth import data_generation
from dpsynth.core import domain
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import aim
from dpsynth.pipeline_transformations import diagnostic_info
import mbi
import numpy as np
import pipeline_dp


class AimTest(absltest.TestCase):

  def _get_dataset_descriptor(
      self, size: int, shape: tuple[int, ...]
  ) -> dataset_descriptor.DatasetDescriptor:
    """Returns a DatasetDescriptor for the given data."""
    attributes = []
    for i, dim in enumerate(shape):
      attributes.append(
          dataset_descriptor.AttributeDescriptor(
              name=f"attr{i}",
              data_type=dataset_descriptor.DataType.INT,
              categorical_attribute=domain.CategoricalAttribute(
                  possible_values=list(range(dim))
              ),
              measurement=mbi.LinearMeasurement(
                  noisy_measurement=np.array([size / dim] * dim),
                  clique=(i,),
                  stddev=1.0,
              ),
          )
      )

    return dataset_descriptor.DatasetDescriptor(
        attributes=attributes,
        data_record_converter=mock.MagicMock(),
    )

  def _get_aim_parameters(self) -> aim.AIMParameters:
    return aim.AIMParameters(
        rounds=1,
        pgm_iters=1,
        max_model_size=1000,
    )

  def _get_uniform_random_data(
      self, size: int, shape: tuple[int, ...]
  ) -> list[tuple[int, ...]]:
    """Returns random data for the given shape."""
    data = []
    for _ in range(size):
      data.append(tuple(np.random.randint(0, dim) for dim in shape))
    return data

  def test_fit_model(self):
    backend = pipeline_dp.LocalBackend()
    accountant = pipeline_dp.PLDBudgetAccountant(
        total_epsilon=1.0,
        total_delta=1e-5,
    )
    domain_shape = (4, 2, 3)
    # Generate random data.

    compressed_data = self._get_uniform_random_data(1000, domain_shape)

    descriptor = self._get_dataset_descriptor(shape=(4, 2, 3), size=1000)
    model = aim.fit_model(
        backend,
        accountant,
        compressed_data,
        [descriptor],
        self._get_aim_parameters(),
    )
    accountant.compute_budgets()
    model = list(model)
    self.assertLen(model, 1)
    model = model[0]
    self.assertIsInstance(model, mbi.MarkovRandomField)
    self.assertEqual(
        model.domain, mbi.Domain(attributes=(0, 1, 2), shape=(4, 2, 3))
    )
    # check 1-way marginals
    count_estimation = model.marginals.project([0]).values.sum()
    self.assertAlmostEqual(count_estimation, 1000, delta=100)
    for i in range(3):
      marginal = model.marginals.project([i])
      self.assertAlmostEqual(
          marginal.sum().values, count_estimation, delta=1e-1
      )
      self.assertAlmostEqual(
          marginal.values[0], count_estimation / marginal.values.size, delta=3
      )

  def test_fit_model_with_diagnostic_info(self):
    backend = pipeline_dp.LocalBackend()
    accountant = pipeline_dp.PLDBudgetAccountant(
        total_epsilon=1.0,
        total_delta=1e-5,
    )
    domain_shape = (4, 2, 3)
    compressed_data = self._get_uniform_random_data(100, domain_shape)
    descriptor = self._get_dataset_descriptor(shape=domain_shape, size=100)

    additional_output = data_generation.AdditionalOutput()
    diag_info = diagnostic_info.DiagnosticInformation()
    additional_output.diagnostic_info = backend.to_collection(
        [diag_info], compressed_data, "create diag info"
    )

    aim.fit_model(
        backend,
        accountant,
        compressed_data,
        [descriptor],
        self._get_aim_parameters(),
        additional_output=additional_output,
    )
    accountant.compute_budgets()

    self.assertIsNotNone(additional_output.diagnostic_info)
    diag_infos = list(additional_output.diagnostic_info)
    self.assertLen(diag_infos, 1)
    diag_info = diag_infos[0]
    # self._get_aim_parameters returns rounds=1
    # So we expect 1 RoundInfo.
    self.assertLen(diag_info.round_info, 1)
    round_info = diag_info.round_info[0]
    self.assertNotEmpty(round_info.l1_distances)
    # Check that l1_distances are populated
    for m in round_info.l1_distances:
      self.assertNotEmpty(m.attributes.attributes)
      self.assertIsInstance(m.value, float)
    self.assertNotEmpty(round_info.selected_attributes)

  def test_generate_workload(self):
    dom = mbi.Domain(
        attributes=(0, 1, 2, 3, 4, 5),
        shape=(10, 100, 1000, 1000, 10000, 100000),
    )
    workload = aim._generate_workload(dom)
    self.assertEqual(workload, [(0, 1, 2), (0, 1, 3)])

  def test_add_dp_noise(self):
    clique_marginal = ((0, 1), np.ones(100000))
    mechanism_spec = pipeline_dp.budget_accounting.MechanismSpec(
        pipeline_dp.budget_accounting.MechanismType.GAUSSIAN
    )
    mechanism_spec.set_noise_standard_deviation(2.0)

    noised_marginal = aim._add_dp_noise(clique_marginal, mechanism_spec)
    self.assertIsInstance(noised_marginal, mbi.LinearMeasurement)
    self.assertEqual(noised_marginal.clique, (0, 1))
    self.assertEqual(noised_marginal.stddev, 2.0)
    actual_mean = np.mean(noised_marginal.noisy_measurement)
    actual_std = np.std(noised_marginal.noisy_measurement)
    self.assertAlmostEqual(actual_mean, 1.0, delta=0.1)
    self.assertAlmostEqual(actual_std, 2.0, delta=0.1)


if __name__ == "__main__":
  absltest.main()
