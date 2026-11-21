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

"""Utilities for iteratively building a clique tree.

A clique tree is a data structure that is useful for doing efficient marginal
inference in a graphical model. It is an undirected graph where nodes are
cliques (subsets of attributes) and edges are pairs of cliques, characterized by
their separator (intersection of the two connected cliques). In general,
marginal inference is tractable as long as this clique tree is not too large,
where the size is determined by the product of domain sizes of the clique
attributes for each node in the tree. If marginal inference is tractable,
marginal-based synthetic data mechanisms are also tractable.

The clique tree initially starts off empty, where there is one node for every
attribute, and no edges. We iteratively build this tree up by incorporating new
cliques, either by adding a clique supported by (a) two disconnected nodes or
(b) an edge in the clique tree. Incorporating cliques according to these rules
ensures the tree structure remains tractable for downstream marginal inference
algorithms.
"""

# NOTE: This module is tested in swift_test.py.

from collections.abc import Sequence
import itertools

import mbi
import networkx as nx


def _cost_connected(
    node1: mbi.Clique, node2: mbi.Clique, clique: mbi.Clique, domain: mbi.Domain
) -> int:
  """Size of the clique created by merging a new clique along (node1, node2)."""
  # We can fold clique into either node - the cost is the smaller of the two.
  cl3 = tuple(set(node1) | set(clique))
  cl4 = tuple(set(node2) | set(clique))
  return min(domain.size(cl3), domain.size(cl4))


def best_supporting_edge(
    clique: mbi.Clique,
    edges: Sequence[tuple[mbi.Clique, mbi.Clique]],
    domain: mbi.Domain,
) -> tuple[tuple[mbi.Clique, mbi.Clique] | None, float]:
  """Finds the best supporting edge for a clique in a clique tree.

  An edge is a pair of cliques, which supports the given clique if the union of
  the edge's cliques contains the given clique. The best supporting edge is the
  one that minimizes the domain size of the union of the edge's cliques and the
  given clique.

  Args:
    clique: The clique to find a supporting edge for.
    edges: The edges of the clique tree.
    domain: The domain of the data.

  Returns:
    The best supporting edge and its cost.
  """
  best_edge = None
  best_cost = float('inf')
  for cl1, cl2 in edges:
    # issue: cost_connected != cost_disconnected in general.
    if set(clique) <= set(cl1) | set(cl2):
      cost = _cost_connected(cl1, cl2, clique, domain)
      if cost < best_cost:
        best_cost = cost
        best_edge = (cl1, cl2)

  return best_edge, best_cost


# pylint: disable=invalid-name
def _local_update_connected(
    clique_tree: nx.Graph,
    node1: mbi.Clique,
    node2: mbi.Clique,
    clique: mbi.Clique,
    domain: mbi.Domain,
) -> nx.Graph:
  """Updates a clique tree in-place by adding a clique to an edge."""
  # We assume that clique is supported by the union of node1 & node2.
  # This invariant should be satisfied by construction when selecting a clique
  # with e.g., derive_supporting_edges.
  assert set(clique) <= set(node1) | set(node2)
  G = clique_tree

  cl3 = tuple(set(node1) | set(clique))
  cl4 = tuple(set(node2) | set(clique))
  if domain.size(cl3) < domain.size(cl4):
    new_node = cl3
    mapping = {node1: new_node}
    if set(new_node) >= set(node2):
      mapping[node2] = new_node
  else:
    new_node = cl4
    mapping = {node2: new_node}
    if set(new_node) >= set(node1):
      mapping[node1] = new_node

  G = nx.relabel_nodes(G, mapping, copy=False)
  if G.has_edge(new_node, new_node):
    G.remove_edge(new_node, new_node)

  return G


def _local_update_disconnected(
    clique_tree: nx.Graph,
    node1: mbi.Clique,
    node2: mbi.Clique,
    clique: mbi.Clique,
) -> nx.Graph:
  """Updates a clique tree by adding a clique to two disconnected components."""

  assert set(clique) <= set(node1) | set(node2)
  G = clique_tree
  if set(node1) | set(node2) == set(clique):
    G = nx.relabel_nodes(G, {node1: clique, node2: clique}, copy=False)

  elif set(node1) <= set(clique):
    G = nx.relabel_nodes(G, {node1: clique}, copy=False)
    G.add_edge(clique, node2)

  elif set(node2) <= set(clique):
    G = nx.relabel_nodes(G, {node2: clique}, copy=False)
    G.add_edge(clique, node1)

  else:
    G.add_node(clique)
    G.add_edge(node1, clique)
    G.add_edge(node2, clique)

  return G


def derive_supporting_edges(
    clique_tree: nx.Graph,
) -> list[tuple[mbi.Clique, mbi.Clique]]:
  """Derives the supporting edges for a clique in a clique tree.

  An edge is a pair of cliques, which supports a given clique if the union of
  the edge's cliques contains the given clique. This function returns pairs of
  nodes that are either (a) part of two disconnected components or (b) an edge
  in the clique tree. Cliques supported by these edges can be added to the tree
  using local_update.

  Args:
    clique_tree: The tree where nodes are cliques.

  Returns:
    The supporting edges of the clique tree.
  """
  components = list(nx.connected_components(clique_tree))
  result = []
  for tree1, tree2 in itertools.combinations(components, r=2):
    for cl1 in tree1:
      for cl2 in tree2:
        result.append((cl1, cl2))

  for cl1, cl2 in clique_tree.edges:
    result.append((cl1, cl2))

  return result


def local_update(
    clique_tree: nx.Graph,
    clique: mbi.Clique,
    domain: mbi.Domain,
) -> nx.Graph:
  """Updates a clique tree to support a new clique by merging edges/components.

  Args:
    clique_tree: The tree where nodes are cliques.
    clique: The new clique to support.
    domain: The domain of the data.

  Returns:
    The updated clique tree.
  """
  edges = derive_supporting_edges(clique_tree)
  edge, _ = best_supporting_edge(clique, edges, domain)
  if edge is None:
    raise ValueError('No supporting edge found.')
  node1, node2 = edge
  if edge in clique_tree.edges:
    return _local_update_connected(clique_tree, node1, node2, clique, domain)
  else:
    return _local_update_disconnected(clique_tree, node1, node2, clique)
