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

"""Tests for SWIFT pipeline transformations."""

from absl.testing import absltest
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline import _data_generation
from dpsynth.pipeline_transformations import diagnostic_info
from dpsynth.pipeline_transformations import swift
import mbi
import numpy as np
import pipeline_dp


class DummyDataRecordConverter(dataset_descriptor.DataRecordConverter):

  def to_tuple(self, record):
    return record

  def from_tuple(self, record, proto_object=None):
    return record


class SwiftTest(absltest.TestCase):

  def test_fit_model(self):
    backend = pipeline_dp.LocalBackend()
    data = [(0, 1), (0, 1), (1, 0), (1, 1)]

    one_way_m0 = mbi.LinearMeasurement(np.array([2.0, 2.0]), (0,), 0.1)
    one_way_m1 = mbi.LinearMeasurement(np.array([1.0, 3.0]), (1,), 0.1)

    attr0 = dataset_descriptor.AttributeDescriptor(
        name="attr0",
        data_type=dataset_descriptor.DataType.INT,
        measurement=one_way_m0,
    )
    attr1 = dataset_descriptor.AttributeDescriptor(
        name="attr1",
        data_type=dataset_descriptor.DataType.INT,
        measurement=one_way_m1,
    )

    descriptor = dataset_descriptor.DatasetDescriptor(
        attributes=[attr0, attr1],
        data_record_converter=DummyDataRecordConverter(),
    )

    descriptor_col = [descriptor]

    parameters = swift.SwiftParameters(pgm_iters=10)  # Small iters for test

    budget_accountant = pipeline_dp.PLDBudgetAccountant(1.0, 1e-5)

    # Setup additional output
    additional_output = _data_generation.AdditionalOutput()
    diag_info_proto = diagnostic_info.DiagnosticInformation()
    additional_output.diagnostic_info = backend.to_collection(
        [diag_info_proto], data, "create diag info"
    )

    result = swift.fit_model(
        backend,
        budget_accountant,
        data,
        descriptor_col,
        parameters,
        workload=[(0, 1)],
        additional_output=additional_output,
    )

    budget_accountant.compute_budgets()
    result_list = list(result)
    self.assertLen(result_list, 1)
    fitted_model = result_list[0]

    self.assertIsInstance(fitted_model, mbi.MarkovRandomField)
    self.assertEqual(fitted_model.domain.shape, (2, 2))

    # Check diagnostic info
    self.assertIsNotNone(additional_output.diagnostic_info)
    diag_infos = list(additional_output.diagnostic_info)
    self.assertLen(diag_infos, 1)
    diag_info = diag_infos[0]
    self.assertLen(diag_info.round_info, 1)
    round_info = diag_info.round_info[0]
    self.assertNotEmpty(round_info.l1_distances)
    self.assertNotEmpty(round_info.selected_attributes)

    # Verify selection is non-deterministic across runs due to DP noise
    attr0_selected_count = 0
    num_runs = 10
    for _ in range(num_runs):
      acc = pipeline_dp.PLDBudgetAccountant(1.0, 1e-5)
      out = _data_generation.AdditionalOutput()
      out.diagnostic_info = backend.to_collection(
          [diagnostic_info.DiagnosticInformation()], data, "create diag info"
      )
      res = swift.fit_model(
          backend,
          acc,
          data,
          descriptor_col,
          parameters,
          workload=[(0, 1)],
          additional_output=out,
      )
      acc.compute_budgets()
      _ = list(res)
      sel = list(list(out.diagnostic_info)[0].round_info[0].selected_attributes)
      if (0,) in [tuple(a.attributes) for a in sel]:
        attr0_selected_count += 1

    self.assertGreater(attr0_selected_count, 0)
    self.assertLess(attr0_selected_count, num_runs)


if __name__ == "__main__":
  absltest.main()
