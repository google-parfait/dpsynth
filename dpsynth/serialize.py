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

"""Lightweight, generic YAML serialization for DPSynth dataclasses."""

from collections.abc import Mapping, Sequence
import dataclasses
import os
import types
import typing
from typing import Any

import cattrs
import dp_accounting
from dpsynth import api
from dpsynth import domain
from dpsynth import reporting
from dpsynth.relational import domain as relational_domain
from etils import epath
import mbi
import yaml

PathType = epath.PathLike


def _resolve_type(type_name: str) -> type[Any] | None:
  """Resolves a class name to its Python class."""
  cls = api.MechanismConfig.get_subclass(type_name)
  if cls is not None:
    return cls
  if hasattr(domain, type_name):
    return getattr(domain, type_name)
  if hasattr(relational_domain, type_name):
    return getattr(relational_domain, type_name)
  if hasattr(reporting, type_name):
    return getattr(reporting, type_name)
  if hasattr(dp_accounting.dp_event, type_name):
    candidate = getattr(dp_accounting.dp_event, type_name)
    if isinstance(candidate, type) and issubclass(
        candidate, dp_accounting.DpEvent
    ):
      return candidate
  if hasattr(mbi.estimation, type_name):
    candidate = getattr(mbi.estimation, type_name)
    if isinstance(candidate, type) and issubclass(candidate, mbi.Estimator):
      return candidate
  return None


def _unstructure_dataclass(cl: type[Any], conv: cattrs.Converter) -> Any:
  base_fn = cattrs.gen.make_dict_unstructure_fn(
      cl, conv, _cattrs_omit_if_default=True
  )
  return lambda obj: {'type': obj.__class__.__name__, **base_fn(obj)}


def _structure_polymorphic(data: Any, _: Any, conv: cattrs.Converter) -> Any:
  if isinstance(data, Mapping) and 'type' in data:
    cls = _resolve_type(data['type'])
    if cls is not None:
      return cattrs.gen.make_dict_structure_fn(cls, conv)(data, cls)
    raise ValueError(f"Unknown type: '{data['type']}'")
  return data


def _make_converter() -> cattrs.Converter:
  """Creates and configures a generic cattrs Converter for dataclasses."""
  conv = cattrs.Converter(
      structure_fallback_factory=lambda _: lambda val, _: val
  )

  # 1. Unstructure: omit default values and prepend 'type' tag
  conv.register_unstructure_hook_factory(
      dataclasses.is_dataclass, lambda cl: _unstructure_dataclass(cl, conv)
  )
  conv.register_unstructure_hook(
      api.MechanismConfig,
      lambda obj: _unstructure_dataclass(obj.__class__, conv)(obj),
  )
  conv.register_unstructure_hook(
      dp_accounting.DpEvent,
      lambda obj: _unstructure_dataclass(obj.__class__, conv)(obj),
  )
  conv.register_unstructure_hook_factory(
      lambda cl: isinstance(cl, type) and issubclass(cl, dp_accounting.DpEvent),
      lambda cl: _unstructure_dataclass(cl, conv),
  )
  conv.register_unstructure_hook(
      mbi.Estimator,
      lambda obj: _unstructure_dataclass(obj.__class__, conv)(obj),
  )
  conv.register_unstructure_hook(
      tuple,
      lambda val: [conv.unstructure(x) for x in val],
  )

  # 2. Polymorphic structuring for abstract base classes and unions
  conv.register_structure_hook(
      api.MechanismConfig, lambda data, _: _structure_polymorphic(data, _, conv)
  )
  conv.register_structure_hook(
      domain.AttributeType,
      lambda data, _: _structure_polymorphic(data, _, conv),
  )
  conv.register_structure_hook(
      dp_accounting.DpEvent,
      lambda data, _: _structure_polymorphic(data, _, conv),
  )
  conv.register_structure_hook(
      mbi.Estimator,
      lambda data, _: _structure_polymorphic(data, _, conv),
  )

  # 3. Structure Sequence[T] consistently as list
  conv.register_structure_hook_func(
      lambda typ: typing.get_origin(typ) is Sequence or typ is Sequence,
      lambda data, typ: [
          conv.structure(
              x, typing.get_args(typ)[0] if typing.get_args(typ) else Any
          )
          for x in data
      ],
  )

  # 4. Structure tuple[tuple[float, float], ...] consistently
  def _is_tuple_of_pairs(typ: Any) -> bool:
    if typ == tuple[tuple[float, float], ...]:
      return True
    origin = typing.get_origin(typ)
    if origin in (typing.Union, types.UnionType):
      return tuple[tuple[float, float], ...] in typing.get_args(typ)
    return False

  conv.register_structure_hook_func(
      _is_tuple_of_pairs,
      lambda data, _: (
          tuple((float(x[0]), float(x[1])) for x in data)
          if data is not None
          else None
      ),
  )

  return conv


converter = _make_converter()

# Ensure tuples are dumped as standard YAML lists (no !!python/tuple tag).
yaml.SafeDumper.add_representer(
    tuple, yaml.representer.SafeRepresenter.represent_list
)


def to_yaml(obj: Any, filepath: str | PathType | None = None) -> str:
  """Serializes an object into YAML, optionally writing to a file.

  Args:
    obj: The dataclass or configuration object to serialize.
    filepath: Optional file path to write the YAML output to.

  Returns:
    The YAML string representation.
  """
  unstructured = converter.unstructure(obj)
  yaml_str = yaml.dump(
      unstructured,
      Dumper=yaml.SafeDumper,
      default_flow_style=False,
      sort_keys=False,
  )
  if filepath is not None:
    epath.Path(filepath).write_text(yaml_str)
  return yaml_str


def from_yaml(
    source: str | PathType, expected_type: type[Any] | None = None
) -> Any:
  """Loads an object from a YAML string or file path.

  Args:
    source: YAML string content or path to a YAML file.
    expected_type: Optional expected target class or type annotation.

  Returns:
    The deserialized object instance.
  """
  if isinstance(source, (os.PathLike, epath.Path)):
    yaml_str = epath.Path(source).read_text()
  elif (
      isinstance(source, str)
      and '\n' not in source
      and (source.endswith(('.yaml', '.yml')) or epath.Path(source).exists())
  ):
    yaml_str = epath.Path(source).read_text()
  else:
    yaml_str = str(source)

  data = yaml.safe_load(yaml_str)
  if expected_type is not None:
    return converter.structure(data, expected_type)

  if isinstance(data, Mapping) and 'type' in data:
    type_name = data['type']
    target_cls = _resolve_type(type_name)
    if target_cls is not None:
      return cattrs.gen.make_dict_structure_fn(target_cls, converter)(
          data, target_cls
      )
    raise ValueError(f"Unknown type: '{type_name}'")

  return data
