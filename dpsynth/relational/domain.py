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

"""Domain representations, schema definitions, and DAG validators for relational data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from absl import logging
from dpsynth.core import domain
from etils import epath
import networkx as nx
import yaml

PathType = epath.PathLike


@dataclasses.dataclass(frozen=True)
class ForeignKeyRelation:
  """Defines a directed foreign key relationship between parent and child tables.

  Attributes:
    parent_table: Name of the parent table (e.g. 'households').
    parent_primary_key: Name of the parent primary key column (e.g.
      'household_id').
    child_table: Name of the child table (e.g. 'persons').
    child_foreign_key: Name of the child foreign key column referencing parent.
    max_children_per_parent: Maximum number of children associated with a single
      parent record (group size capacity bound s). Must be >= 1. Determines the
      wide MRF generation slot count (s) and directly scales cascading DP
      sensitivity (Delta_k = prod s_ancestors) for downstream child tables.
  """

  parent_table: str
  parent_primary_key: str
  child_table: str
  child_foreign_key: str
  max_children_per_parent: int

  def __post_init__(self):
    if self.max_children_per_parent < 1:
      raise ValueError(
          'max_children_per_parent must be >= 1, got'
          f' {self.max_children_per_parent}.'
      )


def topological_sort_hierarchy(
    tables: Sequence[str],
    foreign_keys: Sequence[ForeignKeyRelation],
) -> list[tuple[int, str, ForeignKeyRelation | None]]:
  """Validates DAG tree structure and computes topological synthesis levels.

  Args:
    tables: Sequence of all table names in the database.
    foreign_keys: Sequence of foreign key relationships between tables.

  Returns:
    An ordered list of (depth, table_name, foreign_key_relation) tuples, where
    depth is 0 for root tables (foreign_key_relation is None) and depth >= 1 for
    child tables (foreign_key_relation links the table to its immediate parent).

  Raises:
    ValueError: If foreign keys contain cycles, missing tables, or if a child
      table references more than one parent table (in-degree > 1).
  """
  logging.debug(
      'Computing topological sort for %d tables with %d foreign keys.',
      len(tables),
      len(foreign_keys),
  )
  table_set = set(tables)
  graph = nx.DiGraph()
  graph.add_nodes_from(tables)

  # Maps child_table -> ForeignKeyRelation (incoming edge, e.g.
  # 'persons' -> fk_household_person).
  incoming_fk_map: dict[str, ForeignKeyRelation] = {}
  for fk in foreign_keys:
    if fk.parent_table not in table_set or fk.child_table not in table_set:
      raise ValueError(f'Foreign key references unknown table in {fk}.')
    if fk.child_table in incoming_fk_map:
      raise ValueError(
          f'Child table {fk.child_table!r} has multiple parents;'
          ' in-degree must be <= 1.'
      )
    if fk.parent_table == fk.child_table:
      raise ValueError(f'Self-referential cycle in table {fk.parent_table!r}.')
    incoming_fk_map[fk.child_table] = fk
    graph.add_edge(fk.parent_table, fk.child_table)

  if not nx.is_directed_acyclic_graph(graph):
    raise ValueError('Cycle detected in foreign keys.')

  roots = [t for t in tables if t not in incoming_fk_map]
  logging.debug('Identified %d root privacy unit tables: %s', len(roots), roots)

  result: list[tuple[int, str, ForeignKeyRelation | None]] = []
  for depth, generation in enumerate(nx.topological_generations(graph)):
    for table in generation:
      result.append((depth, table, incoming_fk_map.get(table)))

  logging.info(
      'Computed topological synthesis levels: %s',
      [(d, t) for d, t, _ in result],
  )
  return result


_ATTRIBUTE_TYPE_MAP: Mapping[str, type[domain.AttributeType]] = {
    'CategoricalAttribute': domain.CategoricalAttribute,
    'NumericalAttribute': domain.NumericalAttribute,
    'OpenSetCategoricalAttribute': domain.OpenSetCategoricalAttribute,
    'FreeFormTextAttribute': domain.FreeFormTextAttribute,
}


def _parse_attribute(
    table_name: str, col_name: str, spec: Any
) -> domain.AttributeType:
  """Parses a single attribute specification."""
  if isinstance(
      spec,
      (
          domain.CategoricalAttribute,
          domain.NumericalAttribute,
          domain.OpenSetCategoricalAttribute,
          domain.FreeFormTextAttribute,
      ),
  ):
    return spec
  if not isinstance(spec, Mapping):
    raise ValueError(
        f'Invalid attribute specification for {table_name}.{col_name}: {spec}'
    )
  attr_data = dict(spec)
  if 'type' not in attr_data:
    raise ValueError(
        f'Attribute specification for {table_name}.{col_name} is missing'
        " required 'type' field."
    )
  attr_type_name = attr_data.pop('type')
  attr_cls = _ATTRIBUTE_TYPE_MAP.get(attr_type_name)
  if attr_cls is None:
    raise ValueError(
        f'Unknown attribute type {attr_type_name!r} for'
        f' {table_name}.{col_name}. Expected one of'
        f' {list(_ATTRIBUTE_TYPE_MAP.keys())}.'
    )
  return attr_cls(**attr_data)


def from_dict(
    config: Mapping[str, Any],
) -> tuple[dict[str, domain.Schema], list[ForeignKeyRelation]]:
  """Parses multi-table schema and foreign keys from a dictionary.

  Args:
    config: Dictionary with 'tables' and optional 'foreign_keys' blocks.

  Returns:
    A tuple of (table_domains, foreign_keys).

  Raises:
    ValueError: If configuration format or attribute specifications are invalid.
  """
  if 'tables' not in config or not isinstance(config['tables'], Mapping):
    raise ValueError("'tables' block missing or invalid in config.")

  logging.debug(
      'Parsing multi-table schema dictionary for %d tables.',
      len(config['tables']),
  )
  table_domains: dict[str, domain.Schema] = {}
  for table_name, table_schema in config['tables'].items():
    if not isinstance(table_schema, Mapping):
      raise ValueError(f'Table schema for {table_name!r} must be a mapping.')
    table_domains[table_name] = domain.Schema({
        col_name: _parse_attribute(table_name, col_name, spec)
        for col_name, spec in table_schema.items()
    })

  foreign_keys: list[ForeignKeyRelation] = []
  for fk in config.get('foreign_keys', []):
    if isinstance(fk, ForeignKeyRelation):
      foreign_keys.append(fk)
    elif isinstance(fk, Mapping):
      foreign_keys.append(ForeignKeyRelation(**fk))
    else:
      raise ValueError(f'Invalid foreign key specification: {fk}')

  logging.info(
      'Successfully parsed multi-table schema: %d tables, %d foreign keys.',
      len(table_domains),
      len(foreign_keys),
  )
  return table_domains, foreign_keys


def from_yaml_file(
    filepath: str | PathType,
) -> tuple[dict[str, domain.Schema], list[ForeignKeyRelation]]:
  """Reads multi-table schema and foreign keys from a YAML file.

  Args:
    filepath: Path to the YAML schema file.

  Returns:
    A tuple of (table_domains, foreign_keys).
  """
  logging.info('Loading relational domain schema from YAML file: %s', filepath)
  path = epath.Path(filepath)
  with path.open('r') as f:
    config = yaml.safe_load(f)
  if not isinstance(config, Mapping):
    raise ValueError(f'YAML root in {filepath} must be a mapping.')
  return from_dict(config)


def to_dict(
    table_domains: Mapping[str, domain.Schema],
    foreign_keys: Sequence[ForeignKeyRelation] = (),
) -> dict[str, Any]:
  """Converts multi-table schemas and foreign keys to a dictionary.

  Args:
    table_domains: Mapping from table name to per-column AttributeType schemas.
    foreign_keys: Optional sequence of ForeignKeyRelation objects.

  Returns:
    A dictionary with 'tables' and optional 'foreign_keys' blocks.
  """
  tables_dict: dict[str, dict[str, Any]] = {}
  for table_name, schema in table_domains.items():
    table_dict: dict[str, Any] = {}
    attrs = schema.attributes if isinstance(schema, domain.Schema) else schema
    for col_name, attr in attrs.items():
      attr_dict = dataclasses.asdict(attr)
      attr_dict['type'] = attr.__class__.__name__
      table_dict[col_name] = attr_dict
    tables_dict[table_name] = table_dict

  result: dict[str, Any] = {'tables': tables_dict}
  if foreign_keys:
    result['foreign_keys'] = [dataclasses.asdict(fk) for fk in foreign_keys]
  return result


def to_yaml_file(
    table_domains: Mapping[str, domain.Schema],
    foreign_keys: Sequence[ForeignKeyRelation],
    filepath: str | PathType,
) -> None:
  """Writes multi-table schema and foreign keys to a YAML file.

  Args:
    table_domains: Mapping from table name to per-column AttributeType schemas.
    foreign_keys: Sequence of ForeignKeyRelation objects.
    filepath: Destination path for the YAML schema file.
  """
  logging.info(
      'Saving relational domain schema (%d tables, %d foreign keys) to: %s',
      len(table_domains),
      len(foreign_keys),
      filepath,
  )
  data = to_dict(table_domains, foreign_keys)
  path = epath.Path(filepath)
  with path.open('w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
