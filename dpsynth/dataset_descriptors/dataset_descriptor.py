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

"""Descriptors for a dataset.

Descriptors (DatasetDescriptor, AttributeDescriptor) represent all needed
information about a dataset, which is needed for mapping from raw data to
encoded dataset and to compressed dataset and back.
Descriptors are created from data scheme (e.g. proto, CSV, TFRecord etc).
Then on next stages information for encoding and for compressing is added.
"""

import abc
from collections.abc import Mapping
import dataclasses
import enum
import functools
from typing import Any

from dpsynth import transformations
from dpsynth.core import domain
from dpsynth.discrete_mechanisms import common
import mbi


class DataType(enum.Enum):
  UNSUPPORTED = 0
  INT = 1
  ENUM = 2
  BOOL = 3
  STR = 4
  FLOAT = 5


class DataRecordConverter(abc.ABC):
  """Converts a record of a dataset to a tuple and back."""

  @abc.abstractmethod
  def to_tuple(self, record: Any) -> tuple[Any, ...]:
    pass

  @abc.abstractmethod
  def from_tuple(
      self,
      record: tuple[Any, ...],
      proto_object: Any | None = None,
  ) -> Any:
    """Converts a tuple to a record of the dataset.

    Args:
      record: A tuple of values of the dataset.
      proto_object: If the dataset is represented as a proto, this argument
      contains the proto object of the corresponding proto type. This argument
      is needed only for the formats which are represented as protos.

    Returns:
      A record of the dataset.
    """


CategoricalValue = bool | int | str


@dataclasses.dataclass
class AttributeDescriptor:
  """Descriptor of an attribute of a dataset.

  It can represent SQL column, proto field, or CSV column. The object might
  not know yet details about the attribute, e.g. possible values. For now only
  attributes with possible values (i.e. categorical attributes) are supported.

  Attributes:
    name: The name of the attribute.
    data_type: The data type of the attribute.
    categorical_attribute: Contains information about possible values of the
      attribute. If set, the attribute is categorical.
    quantiles: The computed quantiles of the numerical attribute.
    numerical_attribute: Contains information about the bounds of the attribute.
      If set, the attribute is numerical.
    measurement: The one way DP marginal of the attribute. It used for
      compression (properties compressed_size and compress_transform).
  """

  # non-categorical attributes.
  name: str
  data_type: DataType
  categorical_attribute: domain.CategoricalAttribute | None = None
  quantiles: list[float] | None = None
  numerical_attribute: domain.NumericalAttribute | None = None
  measurement: mbi.LinearMeasurement | None = None

  @property
  def is_initialized(self) -> bool:
    """Returns True if the attribute domain is known, i.e. initialized."""
    return (
        self.categorical_attribute is not None
        or self.numerical_attribute is not None
    )

  def __getstate__(self) -> dict:  # pylint: disable=g-bare-generic
    state = self.__dict__.copy()
    # Do not serialize the transformation objects, they are created on demand.
    # Not serializing them speeds up the pipeline and avoid serialization
    # issues.
    if 'encoding_transform' in state:
      del state['encoding_transform']

    if 'compress_transform' in state:
      del state['compress_transform']

    return state

  @functools.cached_property
  def encoded_size(self) -> int:
    """Returns the number of possible values of the encoded attribute."""
    if self.categorical_attribute is not None:
      return len(self.categorical_attribute.possible_values)  # pylint: disable=attribute-error
    if self.numerical_attribute is not None:
      if self.quantiles is None:
        raise ValueError(
            '`encoded_size` is called for numerical attribute before'
            ' quantiles are derived.'
        )
      bins = len(self.quantiles) + 1
      if not self.numerical_attribute.clip_to_range:
        bins += 1
      return bins
    else:
      raise ValueError('`encoded_size` is called before values are derived.')

  @functools.cached_property
  def compressed_size(self) -> int:
    """Returns the number of possible values of the compressed attribute."""
    if self.measurement is None:
      raise ValueError(
          '`compressed_size` is called before one way marginals are computed.'
      )
    return common.compression_transformation(self.measurement)[0]

  @functools.cached_property
  def encoding_transform(
      self,
  ) -> transformations.DataTransformation[Any, int]:
    """Returns the transformation that encodes the attribute values."""
    if self.categorical_attribute is not None:
      return transformations.discrete_encoder(self.categorical_attribute)
    if self.numerical_attribute is not None and self.quantiles is not None:
      categorical_attr, discretize_transform = (
          transformations.create_discretize_transformation(
              self.numerical_attribute,
              self.quantiles,
          )
      )
      encoder = transformations.discrete_encoder(categorical_attr)
      # Discretize then encode: float -> str interval -> int.
      return encoder @ discretize_transform

    raise ValueError(
        '`encoding_transform` is called before values are derived.'
    )

  def compressed_measurement(self) -> mbi.LinearMeasurement:
    """Returns the compressed measurement of the attribute."""
    if self.measurement is None:
      raise ValueError(
          '`compressed_measurement` is called before one way marginals are'
          ' computed.'
      )
    return common.compressed_measurement(
        self.measurement,
        self.compressed_size,
        self.compress_transform,
    )

  @functools.cached_property
  def compress_transform(
      self,
  ) -> transformations.DataTransformation[int, int]:
    """Returns the transformation that compresses the attribute values."""
    if self.measurement is None:
      raise ValueError(
          '`compress_transform` is called before rare values are found.'
      )
    return common.compression_transformation(self.measurement)[1]


@dataclasses.dataclass
class DatasetDescriptor:
  """Represents a dataset.

  It contains information about the attributes of the dataset which is needed
  for synthetic data generation and how to convert the records of the dataset to
  tuples and back.

  On the creation the descriptor contains all attributes of the dataset, but
  some of them might not have all the details, e.g. possible values, bounds etc.

  Attributes:
    attributes: The attributes of the dataset.
    data_record_converter: The converter of the dataset records. It depends on
      the format of the dataset, e.g. proto, CSV, TFRecord etc.
  """

  attributes: list[AttributeDescriptor]
  data_record_converter: DataRecordConverter

  @property
  def all_attributes_initialized(self) -> bool:
    """Returns True if all attributes are initialized."""
    return all(attr.is_initialized for attr in self.attributes)

  @property
  def encoded_shape(self) -> tuple[int, ...]:
    """Returns the shape of the encoded dataset."""
    return tuple(attr.encoded_size for attr in self.attributes)

  @property
  def compressed_shape(self) -> tuple[int, ...]:
    """Returns the shape of the compressed dataset."""
    return tuple(attr.compressed_size for attr in self.attributes)

  def encode(self, values: tuple[Any, ...]) -> tuple[int, ...]:
    """Encodes the records of the original dataset."""
    return tuple(
        attribute.encoding_transform(value)
        for value, attribute in zip(values, self.attributes)
    )

  def decode(self, values: tuple[int, ...]) -> tuple[Any, ...]:
    """Decode the records of the encoded dataset to the original domain."""
    return tuple(
        attribute.encoding_transform.inverse(value)
        for value, attribute in zip(values, self.attributes)
    )

  def compress(self, values: tuple[int, ...]) -> tuple[int, ...]:
    """Compresses the records of the encoded dataset."""
    return tuple(
        int(attribute.compress_transform(value))  # int() to avoid np.int
        for value, attribute in zip(values, self.attributes)
    )

  def compressed_measurements(self) -> tuple[mbi.LinearMeasurement, ...]:
    """Returns the measurements of the compressed dataset."""
    return tuple(
        attribute.compressed_measurement() for attribute in self.attributes
    )

  def uncompress(self, values: tuple[int, ...]) -> tuple[Any, ...]:
    """Uncompresses the records of the compressed dataset."""
    return tuple(
        attribute.compress_transform.inverse(value)
        for value, attribute in zip(values, self.attributes)
    )

  @property
  def encoded_domain(self) -> mbi.Domain:
    """Returns the domain of the encoded dataset."""
    # Attributes names are not important for the algorithm and they can be
    # large (e.g. nested proto fields). So for performance reasons we will use
    # only indices. DatasetDescriptor can convert back to the original names if
    # needed.
    attributes_indices = tuple(range(len(self.attributes)))
    return mbi.Domain(
        attributes=attributes_indices,  # pyrefly: ignore[bad-argument-type]
        shape=self.encoded_shape,
    )

  @property
  def compressed_domain(self) -> mbi.Domain:
    """Returns the domain of the compressed dataset."""
    # Attributes names are not important for the algorithm and they can be
    # large (e.g. nested proto fields). So for performance reasons we will use
    # only indices. DatasetDescriptor can convert back to the original names if
    # needed.
    attributes_indices = tuple(range(len(self.attributes)))
    return mbi.Domain(
        attributes=attributes_indices,  # pyrefly: ignore[bad-argument-type]
        shape=self.compressed_shape,
    )

  @property
  def attribute_names(self) -> tuple[str, ...]:
    """Returns the names of the attributes."""
    return tuple(attr.name for attr in self.attributes)

  def update_from_domain_specification(
      self, domain_spec: Mapping[str, Any]
  ) -> None:
    """Updates the attributes of the dataset descriptor from a domain specification."""
    for attr_desc in self.attributes:
      if attr_desc.name in domain_spec:
        spec = domain_spec[attr_desc.name]
        if isinstance(spec, domain.CategoricalAttribute):
          attr_desc.categorical_attribute = spec
        elif isinstance(spec, domain.NumericalAttribute):
          attr_desc.numerical_attribute = spec
