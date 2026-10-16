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
"""Protobuf descriptor to dpsynth domain adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
import operator
from typing import Any, Literal

from absl import logging
from dpsynth import domain
from google.protobuf import descriptor
from google.protobuf import message
from google.protobuf import message_factory

OpenSetAttribute = domain.OpenSetCategoricalAttribute

_INTEGER_TYPES = {
    descriptor.FieldDescriptor.TYPE_INT32,
    descriptor.FieldDescriptor.TYPE_INT64,
    descriptor.FieldDescriptor.TYPE_UINT32,
    descriptor.FieldDescriptor.TYPE_UINT64,
    descriptor.FieldDescriptor.TYPE_SINT32,
    descriptor.FieldDescriptor.TYPE_SINT64,
    descriptor.FieldDescriptor.TYPE_FIXED32,
    descriptor.FieldDescriptor.TYPE_FIXED64,
    descriptor.FieldDescriptor.TYPE_SFIXED32,
    descriptor.FieldDescriptor.TYPE_SFIXED64,
}

_FLOAT_TYPES = {
    descriptor.FieldDescriptor.TYPE_FLOAT,
    descriptor.FieldDescriptor.TYPE_DOUBLE,
}


def _is_repeated(field) -> bool:
  return (
      getattr(field, "is_repeated", False)
      or getattr(field, "label", None)
      == descriptor.FieldDescriptor.LABEL_REPEATED
  )


def _resolve_descriptor(proto):
  """Resolves input to a protobuf Message Descriptor."""
  if isinstance(proto, descriptor.Descriptor):
    return proto
  desc = getattr(proto, "DESCRIPTOR", None)
  if isinstance(desc, descriptor.Descriptor):
    return desc
  raise TypeError(
      "Expected a Protobuf Message class, instance, or Descriptor, got"
      f" {type(proto)}."
  )


def infer_domain(
    proto: descriptor.Descriptor | type[message.Message] | message.Message,
    *,
    numerical_bounds: Mapping[str, tuple[float, float]] | None = None,
    enum_format: Literal["name", "number"] = "name",
    ignore_unsupported_fields: bool = False,
) -> dict[str, domain.AttributeType]:
  """Infers attribute domains from a Protobuf definition for a flat schema.

  Non-repeated nested message (`TYPE_MESSAGE`) fields are recursively flattened
  into dot-separated attribute names (e.g. `"account.balance"`). Repeated
  fields are not supported.

  Example:
    ```python
    domains = protobuf.infer_domain(
        UserProfile,
        numerical_bounds={"age": (0.0, 120.0), "account.balance": (0.0, 1e4)},
        ignore_unsupported_fields=True,
    )
    ```

  Args:
    proto: Protobuf `Message` class, instance, or `Descriptor`.
    numerical_bounds: Mapping from (dot-separated) field name to `(min, max)`
      bounds for numeric fields.
    enum_format: Enum representation, either `"name"` or `"number"`.
    ignore_unsupported_fields: Whether to skip unsupported or unbounded fields.

  Returns:
    Mapping from field name to `domain.AttributeType`.
  """
  if enum_format not in ("name", "number"):
    raise ValueError(f"Unknown enum_format '{enum_format}'.")

  msg_desc = _resolve_descriptor(proto)
  attributes: dict[str, domain.AttributeType] = {}
  numerical_bounds = numerical_bounds or {}

  for field in msg_desc.fields:
    if _is_repeated(field):
      if ignore_unsupported_fields:
        logging.info("Skipping repeated field %s", field.name)
        continue
      raise ValueError(f"Repeated field '{field.name}' is not supported.")

    if field.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
      # Strip the submessage prefix from numerical_bounds before recursing.
      prefix = f"{field.name}."
      sub_bounds = {
          k.removeprefix(prefix): v
          for k, v in numerical_bounds.items()
          if k.startswith(prefix)
      }
      sub_domain = infer_domain(
          field.message_type,
          numerical_bounds=sub_bounds,
          enum_format=enum_format,
          ignore_unsupported_fields=ignore_unsupported_fields,
      )
      attributes.update({f"{prefix}{k}": v for k, v in sub_domain.items()})
      continue

    if field.type == descriptor.FieldDescriptor.TYPE_ENUM:
      values = [
          v.name if enum_format == "name" else v.number
          for v in field.enum_type.values
      ]
      attributes[field.name] = domain.CategoricalAttribute(
          possible_values=values, out_of_domain_index=0
      )
    elif field.type in _INTEGER_TYPES or field.type in _FLOAT_TYPES:
      dtype = "int" if field.type in _INTEGER_TYPES else "float"
      if field.name not in numerical_bounds:
        if ignore_unsupported_fields:
          logging.info("Skipping numeric field %s", field.name)
          continue
        raise ValueError(
            f"Numerical bounds must be specified for field '{field.name}'."
        )
      bounds = numerical_bounds[field.name]
      attributes[field.name] = domain.NumericalAttribute(
          min_value=bounds[0], max_value=bounds[1], dtype=dtype
      )
    elif field.type == descriptor.FieldDescriptor.TYPE_STRING:
      attributes[field.name] = domain.OpenSetCategoricalAttribute()
    elif field.type == descriptor.FieldDescriptor.TYPE_BOOL:
      attributes[field.name] = domain.CategoricalAttribute(
          possible_values=[False, True], out_of_domain_index=0
      )
    else:
      if ignore_unsupported_fields:
        logging.info("Skipping unsupported field %s", field.name)
        continue
      raise ValueError(
          f"Field '{field.name}' of type {field.type} is not supported."
      )

  return attributes


def infer_schema(
    proto: descriptor.Descriptor | type[message.Message] | message.Message,
    *,
    numerical_bounds: Mapping[str, tuple[float, float]] | None = None,
    enum_format: Literal["name", "number"] = "name",
    ignore_unsupported_fields: bool = False,
) -> domain.Schema:
  """Derives a dpsynth.Schema from a Protobuf definition for a flat schema."""
  attributes = infer_domain(
      proto,
      numerical_bounds=numerical_bounds,
      enum_format=enum_format,
      ignore_unsupported_fields=ignore_unsupported_fields,
  )
  return domain.Schema(attributes)


@functools.lru_cache(maxsize=64)
def _leaf_field_paths(desc):
  """Returns dot-separated field paths for all non-repeated leaf fields."""
  paths = []
  for field in desc.fields:
    if _is_repeated(field):
      continue
    if field.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
      sub_paths = _leaf_field_paths(field.message_type)
      paths.extend(f"{field.name}.{p}" for p in sub_paths)
    else:
      paths.append(field.name)
  return tuple(paths)


def _resolve_leaf(msg, path):
  """Resolves a dot-separated path to its parent message and leaf field name."""
  *parents, leaf = path.split(".")
  for part in parents:
    msg = getattr(msg, part)
  return msg, leaf


@functools.lru_cache(maxsize=64)
def _compile_getters(desc, paths, enum_format):
  """Precompiles per-field getters for fast tuple extraction."""
  getters = []
  for path in paths:
    # Walk nested message descriptors to inspect the leaf FieldDescriptor.
    sub_desc = desc
    *parents, leaf = path.split(".")
    for part in parents:
      sub_desc = sub_desc.fields_by_name[part].message_type
    field = sub_desc.fields_by_name[leaf]
    # operator.attrgetter resolves dotted paths ("a.b.c") natively in C.
    getter = operator.attrgetter(path)
    if (
        field.type == descriptor.FieldDescriptor.TYPE_ENUM
        and enum_format == "name"
    ):
      # Proto enum attributes return integer wire numbers; map to enum names.
      names = {v.number: v.name for v in field.enum_type.values}
      getter = lambda m, g=getter, n=names: n[g(m)]
    getters.append(getter)
  return tuple(getters)


def to_tuple(
    msg: message.Message,
    *,
    schema: domain.Schema | Mapping[str, domain.AttributeType] | None = None,
    enum_format: Literal["name", "number"] = "name",
) -> tuple[Any, ...]:
  """Converts a Protobuf message instance to a tuple of field values."""
  # Convert schema to a tuple of field names so it is hashable for lru_cache.
  paths = _leaf_field_paths(msg.DESCRIPTOR) if schema is None else tuple(schema)
  getters = _compile_getters(msg.DESCRIPTOR, paths, enum_format)
  return tuple(g(msg) for g in getters)


def from_tuple(
    values: Sequence[Any],
    proto: descriptor.Descriptor | type[message.Message] | message.Message,
    *,
    schema: domain.Schema | Mapping[str, domain.AttributeType] | None = None,
) -> message.Message:
  """Populates a Protobuf message from a sequence of field values."""
  desc = _resolve_descriptor(proto)
  fields = _leaf_field_paths(desc) if schema is None else schema
  # Instantiate an empty message class from the Descriptor and populate leaves.
  msg = message_factory.GetMessageClass(desc)()
  for path, val in zip(fields, values):
    sub_msg, leaf = _resolve_leaf(msg, path)
    setattr(sub_msg, leaf, val)
  return msg


infer_domain_from_proto = infer_domain
schema_from_proto = infer_schema
