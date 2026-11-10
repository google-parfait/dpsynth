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

"""Use DP mechanism to automatically dsicretize numerical data."""

import math
from typing import Any

from dpsynth import transformations
from dpsynth.core import domain
from dpsynth.pipeline_transformations import types
import numpy as np
import pipeline_dp


def create_transformations_via_dp_quantiles(
    pcol: types.Collection[Any],
    engine: pipeline_dp.DPEngine,
    backend: pipeline_dp.PipelineBackend,
    field_name_to_attribute: dict[str, domain.NumericalAttribute],
    num_quanitle_buckets: int,
) -> types.Collection[
    tuple[
        str,
        domain.CategoricalAttribute,
        transformations.DiscretizeTransformation,
    ]
]:
  """Automatically discretize numerical data using DP quantiles.


  Args:
    pcol: Collection of rows.
    engine: A DPEngine instance.
    backend: A PipelineBackend instance.
    field_name_to_attribute: A dictionary mapping field names to their
      attributes.
    num_quanitle_buckets: The number of quantile buckets to use.

  Returns:
    Collection of (field_name, CategoricalAttribute, DiscretizeTransformation)
    tuples, where the field_name is a string, CategoricalAttribute is the
    attribute of the field, and the DiscretizeTransformation is a transformation
    that discretizes the field using differentially private quantiles.
  """
  quantiles = _quantiles(
      pcol,
      engine,
      backend,
      field_name_to_attribute,
      num_quanitle_buckets,
  )

  def create_transformation(row):
    field_name, quantiles_list = row
    attribute, transformation = (
        transformations.create_discretize_transformation(
            field_name_to_attribute[field_name], quantiles_list
        )
    )
    return (field_name, attribute, transformation)

  return backend.map(quantiles, create_transformation, "Create transformations")


def _quantiles(
    pcol: types.Collection[Any],
    engine: pipeline_dp.DPEngine,
    backend: pipeline_dp.PipelineBackend,
    field_name_to_attribute: dict[str, domain.NumericalAttribute],
    num_quanitle_buckets: int,
) -> types.Collection[tuple[str, tuple[float, ...]]]:
  """Computes quantiles of numerical fields using DP.

  Args:
    pcol: A collection of rows.
    engine: A DPEngine instance.
    backend: A PipelineBackend instance.
    field_name_to_attribute: A dictionary mapping field names to their
      attributes.
    num_quanitle_buckets: The number of quantile buckets to use.

  Returns:
    A collection of (field_name, quantiles) pairs, where quantiles is a tuple of
    sorted quantile values, which can be used to discretize the values.
  """

  # We normalize the fields to the range [0, 1], so that we can use the quantile
  # mechanism in a single aggregation step, even if fields have different
  # ranges.  We undo the scaling after the quantiles have been computed.
  # We drop None values here; these are safe to ignore for DP.
  def extract_and_normalize_fields(row):
    for field_name, attribute in field_name_to_attribute.items():
      value = attribute.standardize(row[field_name])
      if value is not None and not (
          isinstance(value, float) and math.isnan(value)
      ):
        normalizer = attribute.max_value - attribute.min_value
        yield field_name, (value - attribute.min_value) / normalizer

  extracted_fields = backend.flat_map(
      pcol, extract_and_normalize_fields, "Extract and scale fields"
  )  # (field_name, value)

  params = pipeline_dp.AggregateParams(
      metrics=[
          pipeline_dp.Metrics.PERCENTILE(p)
          for p in np.linspace(0, 100, num_quanitle_buckets + 1)[1:-1]
      ],
      max_partitions_contributed=len(field_name_to_attribute),
      # assumption: every input is from a different user
      max_contributions_per_partition=1,
      # all inputs are normalized to [0, 1]
      min_value=0,
      max_value=1,
      contribution_bounds_already_enforced=True,
      public_partitions_already_filtered=True,
  )
  extractors = pipeline_dp.DataExtractors(
      privacy_id_extractor=lambda row: None,
      partition_extractor=lambda row: row[0],
      value_extractor=lambda row: row[1],
  )
  scaled_quantiles = engine.aggregate(
      extracted_fields,
      params,
      extractors,
  )  # (field_name, scaled_quantiles)

  # Undo normalization, so that returned quantiles are for the original scale.
  def reverse_scaling(row):
    field_name, scaled_quantiles = row
    attribute = field_name_to_attribute[field_name]
    result_quantiles = []
    for q in scaled_quantiles:
      result_quantiles.append(
          q * (attribute.max_value - attribute.min_value) + attribute.min_value
      )
    return field_name, tuple(sorted(result_quantiles))

  return backend.map(scaled_quantiles, reverse_scaling, "Reverse scaling")
