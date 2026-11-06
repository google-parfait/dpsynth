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

"""Derive categorical values from the data."""

import copy
from typing import Any, TypeAlias, TypeVar

from dpsynth.core import domain
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import types
import pipeline_dp


Key: TypeAlias = TypeVar('Key', bound=str | int)
Record: TypeAlias = tuple[Any, ...] | dict[Key, Any]
PartitionSelectionOutput: TypeAlias = tuple[Key, domain.CategoricalAttribute]


def derive_categorical_values(
    input_data: types.Collection[Record],
    backend: pipeline_dp.PipelineBackend,
    dp_engine: pipeline_dp.DPEngine,
    attribute_keys_to_derive: list[Key],
) -> types.Collection[list[PartitionSelectionOutput]] | None:
  """Derives categorical values from the data in DP way.

  It derives categorical values for all attributes which doesn't have possible
  values yet. All DP operations are performed with 'dp_engine', which provides
  differential privacy guarantees per pipeline. The budget management is
  performed with BudgetAccountant class which is passed to the DPEngine.
  Otherwise from budget requests for BudgetAccountant, DPEngine is stateless.

  Args:
    input_data: The input data to derive categorical values from. It is a
      collection of records, where each record is a tuple or dictionary
      representing a row in the dataset.
    backend: The pipeline backend to use.
    dp_engine: The DP engine to use.
    attribute_keys_to_derive: A list of attribute indices/names to derive
      values for. If the input recfords are tuples, this should be a list of
      attribute indices. If the input records are dictionaries, this should be
      a list of attribute names.

  Returns:
    The derived categorical values: a collection with 1 element, which is a list
    of tuples: (index of the attribute, list of possible values). Returns None
    if there are no attributes to derive values for.
  """
  # within the same privacy budget).

  if not attribute_keys_to_derive:
    # No attributes to derive values for.
    return None

  def extract_categorical_values(row):
    for key in attribute_keys_to_derive:
      yield key, row[key]

  candidate_categorical_values = backend.flat_map(
      input_data, extract_categorical_values, 'ExtractCategoricalValues'
  )  # (attribute_index, value)

  # Here we assume each user contributes 1 record.
  strategy = pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING
  params = pipeline_dp.SelectPartitionsParams(
      max_partitions_contributed=len(attribute_keys_to_derive),
      partition_selection_strategy=strategy,
      contribution_bounds_already_enforced=True,
  )
  extractors = pipeline_dp.DataExtractors(
      # We assume every row is owned by a unique user, so no need to extract
      # privacy id.
      privacy_id_extractor=lambda row: None,
      partition_extractor=lambda row: row,
  )
  selected_categorical_values = dp_engine.select_partitions(
      candidate_categorical_values, params, extractors
  )  # (attribute_index, value)

  selected_categorical_values = backend.group_by_key(
      selected_categorical_values, 'GroupByAttributeIndex'
  )  # (attribute_index, [value])

  def to_categorical_attribute(values) -> domain.CategoricalAttribute:
    values = sorted(values, key=str)  # to make sure the order is deterministic
    # Cast to str for homogeneity with the '<OOD>' sentinel.
    return domain.CategoricalAttribute(['<OOD>'] + [str(v) for v in values])  # pyrefly: ignore[bad-argument-count]

  categorical_attributes = backend.map_values(
      selected_categorical_values,
      to_categorical_attribute,
      'SortPossibleValues',
  )  # (attribute_index, domain.CategoricalAttribute)

  categorical_attributes = backend.to_list(categorical_attributes, 'ToList')

  def to_full_dict(attributes) -> dict[Key, domain.CategoricalAttribute]:
    res = dict(attributes)
    for key in attribute_keys_to_derive:
      if key not in res:
        res[key] = domain.CategoricalAttribute(['<OOD>'])  # pyrefly: ignore[bad-argument-count]
    return res

  return backend.map(
      categorical_attributes,
      to_full_dict,
      'ToDict',
  )


def add_derived_values(
    backend: pipeline_dp.PipelineBackend,
    descriptors: types.Collection[dataset_descriptor.DatasetDescriptor],
    categorical_attributes: types.Collection[
        dict[int, domain.CategoricalAttribute]
    ],
) -> types.Collection[dataset_descriptor.DatasetDescriptor]:
  """Adds possible values to attributes."""

  def add_possible_values(
      descriptors,
      categorical_attributes: dict[int, domain.CategoricalAttribute],
  ):
    copy_descriptors = copy.deepcopy(descriptors)
    for i, categorical_attribute in categorical_attributes.items():
      copy_descriptors.attributes[i].categorical_attribute = (
          categorical_attribute
      )

    return copy_descriptors

  return backend.map_with_side_inputs(
      col=descriptors,
      fn=add_possible_values,
      side_input_cols=[categorical_attributes],
      stage_name='AddDerivedValues',
  )
