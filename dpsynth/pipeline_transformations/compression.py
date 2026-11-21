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

"""Pipeline DP functions providing domain compression functionality."""

from typing import TypeAlias

from dpsynth import transformations
from dpsynth.pipeline_transformations import types
from dpsynth.statistical.base import common
import mbi
import pipeline_dp


# The Input & Output types are one-element collections because prior to this,
# the rows are grouped by attributes to calculate noisy marginals. The order of
# both LinearMeasurement & DataTransformation objects is defined by
# DatasetDescriptor.
_MeasurementList: TypeAlias = types.Collection[list[mbi.LinearMeasurement]]
_TransformsList: TypeAlias = types.Collection[
    list[tuple[int, transformations.DataTransformation[int, int]]]
]


def get_domain_compression_transforms(
    one_way_marginals: _MeasurementList,
    backend: pipeline_dp.PipelineBackend,
    stage_name: str,
) -> _TransformsList:
  """Returns compression transforms for attribute in the given LinearMeasurements.

  Args:
    one_way_marginals: A singleton collection of a list of mbi.LinearMeasurement
      objects for each attribute.
    backend: The pipeline backend to use.
    stage_name: The name of the stage to use.

  Returns:
    A singleton collection of a list[(compressed_size, transform_fn)] with
    order of attributes matching LinearMeasurements in the input collection.
  """

  return backend.map(
      one_way_marginals,
      lambda x: [common.compression_transformation(m) for m in x],
      stage_name,
  )


def apply_compression_transforms(
    one_way_marginals: _MeasurementList,
    compressed_column_transforms_collection: _TransformsList,
    backend: pipeline_dp.PipelineBackend,
    stage_name: str,
) -> _MeasurementList:
  """Applies the compression transforms to the given collection of linear measurements.

  Args:
    one_way_marginals: A singleton collection of a list of mbi.LinearMeasurement
      objects for each attribute.
    compressed_column_transforms_collection: Singleton collection of a
      list[(compressed_size, transform_fn)]
    backend: The pipeline backend to use.
    stage_name: The name of the stage to use.

  Returns:
    A singleton collection of a list of mbi.LinearMeasurement objects with the
    compression transforms applied.
  """

  def apply_transforms(linear_measurements, compression_transforms):
    compressed_measurements = []
    for linear_measurement, (compressed_size, transform) in zip(
        linear_measurements, compression_transforms
    ):
      compressed_measurements.append(
          common.compressed_measurement(
              linear_measurement, compressed_size, transform
          )
      )
    return compressed_measurements

  return backend.map_with_side_inputs(
      one_way_marginals,
      apply_transforms,
      [compressed_column_transforms_collection],
      stage_name,
  )
