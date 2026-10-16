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

"""Pydantic model to dpsynth domain adapter.

Provides three functions (matching `dpsynth.adapters.protobuf`) to bridge
`pydantic.BaseModel` schemas with `dpsynth` mechanisms:

- `infer_domain`: Infers attribute domains from a `BaseModel` class.
- `to_tuple`: Converts a `BaseModel` instance to a tuple of leaf field values.
- `from_tuple`: Reconstructs a `BaseModel` instance from a sequence of values.

Nested `BaseModel` fields are flattened into dot-separated keys (e.g.,
`"metrics.balance"`).
"""

from collections.abc import Mapping, Sequence
import enum
import math
import types
import typing  # for typing.get_origin and typing.get_args
from typing import Any, Literal, TypeVar

from dpsynth import domain
import pydantic
from pydantic.fields import annotated_types
from pydantic.fields import FieldInfo  # pylint: disable=g-importing-member


def _get_base_type(annotation: type[Any]) -> tuple[bool, type[Any]]:
  """Determines the base type of a type annotation and if it is optional."""

  if annotation is types.NoneType:
    raise ValueError("Unexpected None type annotation.")

  is_optional = annotation == (annotation | None)

  if is_optional:
    # It's a Union like `int | None`. We need the non-None part.
    args = typing.get_args(annotation)
    non_none_args = [arg for arg in args if arg is not types.NoneType]
    assert len(non_none_args) == 1, "Union must have exactly one non-None type."
    annotation = non_none_args[0]

  is_class = isinstance(annotation, type)
  is_enum = is_class and issubclass(annotation, enum.Enum)
  is_model = is_class and issubclass(annotation, pydantic.BaseModel)
  is_primitive = annotation in (int, float, str, bool)
  is_literal = typing.get_origin(annotation) is Literal
  if is_model and is_optional:
    raise ValueError(f"Optional nested models are not supported: {annotation}.")
  if not (is_primitive or is_model or is_enum or is_literal):
    raise ValueError(f"Unexpected type annotation: {annotation}.")

  return is_optional, annotation


def _numerical_attribute_from_field_info(
    field_info: FieldInfo,
    bounds: tuple[float, float] | None = None,
) -> domain.NumericalAttribute:
  """Infers a NumericalAttribute from a pydantic FieldInfo."""
  # NumericalAttribute uses a convention where both the min_value and max_value
  # are assumed to be inclusive.  More general bounds are supported by the
  # pydantic metadata, so we convert to the expected representation here.
  optional, base_type = _get_base_type(field_info.annotation)  # pyrefly: ignore[bad-argument-type]

  lower_bound = upper_bound = None
  for meta in field_info.metadata:
    match type(meta):
      case annotated_types.Ge:
        lower_bound = meta.ge
      case annotated_types.Le:
        upper_bound = meta.le
      case annotated_types.Gt:
        if base_type is int:
          lower_bound = meta.gt + 1
        else:
          lower_bound = math.nextafter(meta.gt, math.inf)
      case annotated_types.Lt:
        if base_type is int:
          upper_bound = meta.lt - 1
        else:
          upper_bound = math.nextafter(meta.lt, -math.inf)
      case _:
        continue
  if bounds is not None:
    lower_bound, upper_bound = bounds
  if lower_bound is None or upper_bound is None:
    raise ValueError("Must specify lower and upper bounds for numeric fields.")

  return domain.NumericalAttribute(
      min_value=lower_bound,  # pyrefly: ignore[unexpected-keyword]
      max_value=upper_bound,  # pyrefly: ignore[unexpected-keyword]
      clip_to_range=not optional,  # pyrefly: ignore[unexpected-keyword]
      dtype=base_type.__name__,  # pyrefly: ignore[unexpected-keyword]
      description=field_info.description,  # pyrefly: ignore[unexpected-keyword]
  )


def _categorical_attribute_from_field_info(
    field_info: FieldInfo,
) -> domain.CategoricalAttribute:
  """Infers a CategoricalAttribute from a pydantic FieldInfo."""
  optional, base_type = _get_base_type(field_info.annotation)  # pyrefly: ignore[bad-argument-type]
  if isinstance(base_type, type) and issubclass(base_type, enum.Enum):
    possible_values = [str(e.value) for e in base_type]
  elif typing.get_origin(base_type) is Literal:
    possible_values = [str(v) for v in typing.get_args(base_type)]
  elif base_type is bool:
    possible_values = [str(False), str(True)]
  else:
    raise ValueError(f"Unexpected type annotation: {base_type}.")

  if optional:
    possible_values = [str(None)] + possible_values

  return domain.CategoricalAttribute(
      possible_values=possible_values,  # pyrefly: ignore[unexpected-keyword]
      out_of_domain_index=0,  # pyrefly: ignore[unexpected-keyword]
      description=field_info.description,  # pyrefly: ignore[unexpected-keyword]
  )


def infer_domain(
    model_cls: type[pydantic.BaseModel],
    *,
    numerical_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, domain.AttributeType]:
  """Infers the domain of a pydantic model."""
  numerical_bounds = numerical_bounds or {}
  attributes: dict[str, domain.AttributeType] = {}
  for name, meta in model_cls.model_fields.items():
    _, base_type = _get_base_type(meta.annotation)  # pyrefly: ignore[bad-argument-type]
    is_class = isinstance(base_type, type)
    is_enum = is_class and issubclass(base_type, enum.Enum)
    is_model = is_class and issubclass(base_type, pydantic.BaseModel)
    is_literal = typing.get_origin(base_type) is Literal
    if is_model:
      prefix = f"{name}."
      sub_bounds = {
          k.removeprefix(prefix): v
          for k, v in numerical_bounds.items()
          if k.startswith(prefix)
      }
      sub = infer_domain(base_type, numerical_bounds=sub_bounds)
      attributes.update({f"{prefix}{k}": v for k, v in sub.items()})
    elif base_type in (int, float):
      attributes[name] = _numerical_attribute_from_field_info(
          meta, bounds=numerical_bounds.get(name)
      )
    elif base_type is str:
      attributes[name] = domain.OpenSetCategoricalAttribute(
          description=meta.description
      )
    elif base_type is bool or is_enum or is_literal:
      attributes[name] = _categorical_attribute_from_field_info(meta)
    else:
      raise ValueError(f"Unexpected type annotation: {base_type}.")
  return attributes


RecordT = TypeVar("RecordT", bound=pydantic.BaseModel)


def to_tuple(
    record: RecordT,
    *,
    schema: Mapping[str, domain.AttributeType] | None = None,
) -> tuple[Any, ...]:
  """Converts a pydantic model instance to a tuple of field values."""
  schema = schema or infer_domain(type(record))
  row = record.model_dump(mode="json")
  cat_types = (domain.CategoricalAttribute, domain.OpenSetCategoricalAttribute)
  result = []
  for col, attr in schema.items():
    val = row
    for part in col.split("."):
      val = val[part]
    if isinstance(attr, cat_types):
      val = str(val)
    elif isinstance(attr, domain.NumericalAttribute) and val is None:
      # the NumericalAttribute contract, but currently doesn't.
      if attr.clip_to_range:
        val = attr.min_value
    result.append(val)
  return tuple(result)


def from_tuple(
    values: Sequence[Any],
    model_cls: type[RecordT],
    *,
    schema: Mapping[str, domain.AttributeType] | None = None,
) -> RecordT:
  """Converts a sequence of field values back to a pydantic model instance."""
  schema = schema or infer_domain(model_cls)

  def _coerce(attr, v):
    if v in (None, "None", "<OOD>"):
      return None
    if isinstance(attr, domain.NumericalAttribute):
      if math.isnan(v):
        return None
      return round(float(v)) if attr.dtype == "int" else v
    return v

  kwargs: dict[str, Any] = {}
  for (col, attr), val in zip(schema.items(), values):
    *parents, leaf = col.split(".")
    curr = kwargs
    for part in parents:
      curr = curr.setdefault(part, {})
    curr[leaf] = _coerce(attr, val)
  return model_cls(**kwargs)  # pyrefly: ignore[bad-unpacking]
