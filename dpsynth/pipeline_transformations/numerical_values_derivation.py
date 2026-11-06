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

"""Compute quantiles with DP for numerical attributes, for differentially private synthetic data generation.

Differential Privacy (DP) is a framework for bounding information that a dataset
release contains about individual contributions.
"""

import copy
import dataclasses
from typing import Any, Generic, TypeAlias, TypeVar

from dpsynth.core import domain
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import types
import numpy as np
import pipeline_dp
from pipeline_dp import pipeline_functions

Key: TypeAlias = TypeVar('Key', bound=str | int)
Record: TypeAlias = tuple[Any, ...] | dict[Key, Any]


@dataclasses.dataclass(frozen=True)
class NumericalAttributeOutput(Generic[Key]):
  key: Key
  attribute: domain.NumericalAttribute
  quantiles: tuple[float, ...]


def derive_numerical_attributes(
    input_data: types.Collection[Record],
    backend: pipeline_dp.PipelineBackend,
    dp_engine: pipeline_dp.DPEngine,
    attribute_keys_to_derive: list[Key],
    num_quantile_buckets: int,
) -> types.Collection[NumericalAttributeOutput] | None:
  """Derives new NumericalAttribute objects by computing DP quantiles.

  This function computes differentially private quantiles for the specified
  numerical attributes and uses these quantiles to create new
  `domain.NumericalAttribute` objects, which can be used for discretization.

  Args:
    input_data: A collection of Records.
    backend: A PipelineBackend instance.
    dp_engine: A DPEngine instance configured for differential privacy.
    attribute_keys_to_derive: A list of field Keys that are to be derived.
    num_quantile_buckets: The number of quantile buckets to use for
      discretization. This means `num_quantile_buckets - 1` boundaries will be
      computed.

  Returns:
    A collection of `NumericalAttributeOutput` objects, where
    each object contains the field key, the min/max-based
    `domain.NumericalAttribute`, and the DP-computed quantiles. Returns None if
    there are no attributes to derive values for.
  """
  if not attribute_keys_to_derive:
    # No attributes to derive values for.
    return None

  def extract_field_values(row):
    for key in attribute_keys_to_derive:
      yield key, row[key]

  key_val_pairs = backend.flat_map(
      input_data,
      extract_field_values,
      'Extract key-value pairs for all keys in attribute_keys_to_derive.',
  )  # (key, value)

  min_max = pipeline_functions.min_max_per_key(
      backend, key_val_pairs, 'Compute min and max per key.'
  )  # (key, (min, max))

  def _make_numerical_attr(min_max):
    lo, hi = min_max
    if lo >= hi:
      # Constant column: pad range so the attribute is valid.
      hi = lo + 1.0
    return domain.NumericalAttribute(min_value=lo, max_value=hi)  # pyrefly: ignore[unexpected-keyword]

  key_to_attr = backend.map_values(
      min_max,
      _make_numerical_attr,
      'Create numerical attribute from min/max',
  )  # (key, domain.NumericalAttribute)

  # Conversion of key_to_attr to a singleton collection
  key_to_attr = backend.to_list(
      key_to_attr, 'key_to_attr to list'
  )  # singleton [(key, domain.NumericalAttribute)]
  key_to_attr = backend.map(
      key_to_attr, dict, 'key_to_attr list to dict'
  )  # singleton {key: domain.NumericalAttribute}

  quantiles = _compute_dp_quantiles(
      input_data,
      backend,
      dp_engine,
      key_to_attr,
      len(attribute_keys_to_derive),
      num_quantile_buckets,
  )

  def create_attribute_from_quantiles(row, key_to_attr):
    key, quantiles_list = row
    original_attribute = key_to_attr[key]
    return NumericalAttributeOutput(
        key=key, attribute=original_attribute, quantiles=quantiles_list
    )

  return backend.map_with_side_inputs(
      quantiles,
      create_attribute_from_quantiles,
      [key_to_attr],
      'Create new NumericalAttribute objects from quantiles.',
  )


def _privacy_id_extractor(_: tuple[Key, float]) -> None:
  return None


def _partition_extractor(row: tuple[Key, float]) -> Key:
  return row[0]


def _value_extractor(row: tuple[Key, float]) -> float:
  return row[1]


def _compute_dp_quantiles(
    input_data: types.Collection[Record],
    backend: pipeline_dp.PipelineBackend,
    dp_engine: pipeline_dp.DPEngine,
    key_to_attr: types.Collection[dict[Key, domain.NumericalAttribute]],
    num_attributes: int,
    num_quantile_buckets: int,
) -> types.Collection[tuple[Key, tuple[float, ...]]]:
  """Computes quantiles of numerical fields using DP.

  The values are first normalized to [0, 1] to allow for a single DP aggregation
  step across fields with different ranges. The resulting scaled quantiles are
  then denormalized back to the original scale before they are returned.

  Args:
    input_data: A collection of Records (rows).
    backend: A PipelineBackend instance.
    dp_engine: A DPEngine instance.
    key_to_attr: A Collection  containing a single dictionary mapping field Keys
      to their `domain.NumericalAttribute`. This is used as a side input.
    num_attributes: The number of numerical attributes being processed. This is
      used to set `max_partitions_contributed` for the DP aggregation.
    num_quantile_buckets: The number of quantile boundaries to compute (N-1
      boundaries for N buckets).

  Returns:
    A collection of (key, quantiles) pairs, where quantiles is a tuple of
    sorted, differentially private quantile values in the original scale of the
    data (which serve as the boundaries for discretization).
  """

  # Step 1: Normalize fields to the [0, 1] range for single step dp aggregation.
  def extract_and_normalize_fields(row, key_to_attr):
    for key, attribute in key_to_attr.items():
      val = attribute.standardize(row[key])
      if val is not None:
        if attribute.max_value == attribute.min_value:
          yield key, 0.0
        else:
          yield key, (val - attribute.min_value) / (
              attribute.max_value - attribute.min_value
          )

  extracted_fields = backend.flat_map_with_side_inputs(
      input_data,
      extract_and_normalize_fields,
      [key_to_attr],
      'Extract and normalize fields to range [0, 1].',
  )  # (key, normalized_values_list) pairs

  # Define the DP mechanism parameters.
  params = pipeline_dp.AggregateParams(
      metrics=[
          pipeline_dp.Metrics.PERCENTILE(p)
          for p in np.linspace(0, 100, num_quantile_buckets + 1)[1:-1]
      ],
      noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
      max_partitions_contributed=num_attributes,
      max_contributions_per_partition=1,
      min_value=0,
      max_value=1,
      contribution_bounds_already_enforced=True,
      public_partitions_already_filtered=True,
  )

  extractors = pipeline_dp.DataExtractors(
      privacy_id_extractor=_privacy_id_extractor,
      partition_extractor=_partition_extractor,
      value_extractor=_value_extractor,
  )
  # Step 2: Compute the requested percentiles on the normalized values,
  # adding DP noise.
  scaled_quantiles = dp_engine.aggregate(
      extracted_fields, params, extractors
  )  # (key, scaled_quantiles_tuple)

  # Step 3: Denormalize data - reverse scaling on the DP-computed quantiles
  # from [0,1] to original range using the original max/min values.
  def reverse_scaling(row, key_to_attr):
    key, scaled_quantiles_tuple = row
    attribute = key_to_attr[key]
    result_quantiles = [
        q * (attribute.max_value - attribute.min_value) + attribute.min_value
        for q in scaled_quantiles_tuple
    ]
    return key, tuple(sorted(result_quantiles))

  return backend.map_with_side_inputs(
      scaled_quantiles,
      reverse_scaling,
      [key_to_attr],
      'Reverse scaling to get original-scale quantile boundaries.',
  )


def add_numerical_values(
    backend: pipeline_dp.PipelineBackend,
    descriptors: types.Collection[dataset_descriptor.DatasetDescriptor],
    numerical_output: types.Collection[NumericalAttributeOutput],
) -> types.Collection[dataset_descriptor.DatasetDescriptor]:
  """Adds the derived numerical values and quantiles to the DatasetDescriptor.

  This function takes the output of `derive_numerical_attributes` and updates
  the provided `DatasetDescriptor` by populating the `numerical_attribute`
  and `quantiles` fields for the relevant attributes.

  Args:
    backend: A PipelineBackend instance.
    descriptors: A Collection containing the `DatasetDescriptor` to be updated.
    numerical_output: A Collection of `NumericalAttributeOutput` objects
      containing the derived numerical attributes and quantiles.

  Returns:
    A Collection containing the updated `DatasetDescriptor` with the
    numerical attributes and quantiles added.
  """
  numerical_output_list = backend.to_list(
      numerical_output, 'NumericalAttributeOutput to list'
  )

  def to_dict(values):
    return {v.key: v for v in values}

  numerical_output_dict = backend.map(
      numerical_output_list, to_dict, 'NumericalAttributeOutput to dict'
  )

  def add_values(
      descriptors, numerical_values: dict[int, NumericalAttributeOutput]
  ):
    descriptors = copy.deepcopy(descriptors)
    for ind, op in numerical_values.items():
      descriptors.attributes[ind].numerical_attribute = op.attribute
      descriptors.attributes[ind].quantiles = op.quantiles
    return descriptors

  return backend.map_with_side_inputs(
      col=descriptors,
      fn=add_values,
      side_input_cols=[numerical_output_dict],
      stage_name='AddNumericalValues',
  )
