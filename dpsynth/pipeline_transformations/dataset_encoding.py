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

"""Transformations to encode each column of the dataset to 0, ... n-1."""

from typing import Any

from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import categorical_values_derivation
from dpsynth.pipeline_transformations import numerical_values_derivation
from dpsynth.pipeline_transformations import types
import pipeline_dp


def get_indices_to_discretisize(
    descriptor: dataset_descriptor.DatasetDescriptor,
) -> tuple[list[int], list[int]]:
  """Finds categorical and numerical indices to derive."""
  categorical_attr_indices_to_derive = []
  numerical_attr_indices_to_derive = []
  for i, attribute in enumerate(descriptor.attributes):
    if attribute.numerical_attribute is not None:
      # Numerical attribute with is pre-defined from yaml, we need to derive
      # the quantiles.
      numerical_attr_indices_to_derive.append(i)
    elif not attribute.is_initialized:
      if attribute.data_type is dataset_descriptor.DataType.FLOAT:
        numerical_attr_indices_to_derive.append(i)
      else:
        categorical_attr_indices_to_derive.append(i)
  return categorical_attr_indices_to_derive, numerical_attr_indices_to_derive


def encode_dataset(
    data: types.Collection[tuple[Any, ...]],
    backend: pipeline_dp.PipelineBackend,
    dp_engine: pipeline_dp.DPEngine,
    descriptor: dataset_descriptor.DatasetDescriptor,
    num_quantiles: int = 100,
) -> tuple[
    types.Collection[tuple[int, ...]],
    types.Collection[dataset_descriptor.DatasetDescriptor],
]:
  """Encodes the dataset to a discretized dataset."""

  (
      categorical_attr_indices_to_derive,
      numerical_attr_indices_to_derive,
  ) = get_indices_to_discretisize(descriptor)

  #  Derive categorical values.
  derived_categorical_values = (
      categorical_values_derivation.derive_categorical_values(
          data, backend, dp_engine, categorical_attr_indices_to_derive
      )
  )  # (dict[int, domain.CategoricalAttribute])

  derived_numerical_values = (
      numerical_values_derivation.derive_numerical_attributes(
          data,
          backend,
          dp_engine,
          numerical_attr_indices_to_derive,
          num_quantiles,
      )
  )  # (Collection[key, attribute, quantiles]) i.e.
  # (Collection[str | int, domain.NumericalAttribute, tuple[float,..]])

  descriptor = backend.to_collection([descriptor], data, 'ToCollection')

  if derived_categorical_values is not None:
    descriptor = categorical_values_derivation.add_derived_values(  # pyrefly: ignore[bad-assignment]
        backend, descriptor, derived_categorical_values
    )  # (dataset_descriptor)

  if derived_numerical_values is not None:
    descriptor = numerical_values_derivation.add_numerical_values(  # pyrefly: ignore[bad-assignment]
        backend, descriptor, derived_numerical_values
    )  # (dataset_descriptor)

  def encode_fn(row, d: dataset_descriptor.DatasetDescriptor):
    return d.encode(row)

  encoded_data = backend.map_with_side_inputs(
      data, encode_fn, [descriptor], 'EncodeData'
  )

  return encoded_data, descriptor


def decode_dataset(
    data: types.Collection[tuple[int, ...]],
    descriptor: types.Collection[dataset_descriptor.DatasetDescriptor],
    backend: pipeline_dp.PipelineBackend,
) -> tuple[
    types.Collection[tuple[Any, ...]],
    types.Collection[dataset_descriptor.DatasetDescriptor],
]:
  """Decodes the dataset to a discretized dataset."""

  def decode_fn(row, d: dataset_descriptor.DatasetDescriptor):
    return d.decode(row)

  return backend.map_with_side_inputs(
      data, decode_fn, [descriptor], 'DecodeData'
  )
