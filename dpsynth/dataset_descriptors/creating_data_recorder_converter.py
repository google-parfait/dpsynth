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

"""Creates DataRecordConverters for specific dataset types."""

import typing
from typing import Any

from dpsynth.dataset_descriptors import csv_descriptor
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import types


def create_data_record_converter(
    dataset_desc: dataset_descriptor.DatasetDescriptor,
    data_format: types.DataFormat,
    proto_type: Any = None,
) -> dataset_descriptor.DataRecordConverter:
  """Creates a DataRecordConverter for the specified format."""
  if data_format == types.DataFormat.CSV:
    attributes_dict = {
        attr.name: attr.data_type for attr in dataset_desc.attributes
    }
    return csv_descriptor.CSVConverter(attributes_dict)
  elif data_format == types.DataFormat.TFRECORD:
    from dpsynth.dataset_descriptors import tfrecord_descriptor
    attributes_dict = {
        attr.name: attr.data_type for attr in dataset_desc.attributes
    }
    return tfrecord_descriptor.TFRecordConverter(attributes_dict)
  else:
    raise ValueError(f"Unsupported data format: {data_format}")
