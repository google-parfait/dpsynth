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

import itertools
import os

from absl.testing import absltest
from dpsynth import _checkpoint
from dpsynth.discrete_mechanisms import clique_tree
from dpsynth.discrete_mechanisms import common
from dpsynth.discrete_mechanisms import swift
from dpsynth.discrete_mechanisms import swift_utils
import mbi
import networkx as nx
import numpy as np


class CliqueTreeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.domain = mbi.Domain(['a', 'b', 'c', 'd'], [2, 3, 4, 5])

  def test_best_supporting_edge(self):
    edges = [(('a',), ('b',)), (('b',), ('c',))]
    clique = ('a', 'b')
    edge, cost = clique_tree.best_supporting_edge(clique, edges, self.domain)
    self.assertEqual(edge, (('a',), ('b',)))
    self.assertEqual(cost, 6)

    clique = ('a', 'c')
    edge, cost = clique_tree.best_supporting_edge(clique, edges, self.domain)
    self.assertIsNone(edge)
    self.assertEqual(cost, float('inf'))

  def test_derive_supporting_edges_connected(self):
    tree = nx.Graph()
    tree.add_edges_from([(('a',), ('b',)), (('b',), ('c',))])
    edges = clique_tree.derive_supporting_edges(tree)
    self.assertCountEqual(edges, [(('a',), ('b',)), (('b',), ('c',))])

  def test_derive_supporting_edges_disconnected(self):
    tree = nx.Graph()
    tree.add_nodes_from([('a',), ('b',), ('c',)])
    edges = clique_tree.derive_supporting_edges(tree)
    self.assertCountEqual(
        edges, [(('a',), ('b',)), (('a',), ('c',)), (('b',), ('c',))]
    )

  def test_local_update(self):
    tree = nx.Graph()
    tree.add_nodes_from([('a',), ('b',), ('c',)])
    clique = ('a', 'b')
    updated_tree = clique_tree.local_update(tree, clique, self.domain)
    self.assertIn(('a', 'b'), updated_tree.nodes)
    self.assertNotIn(('a',), updated_tree.nodes)
    self.assertNotIn(('b',), updated_tree.nodes)
    self.assertIn(('c',), updated_tree.nodes)
    self.assertEmpty(list(updated_tree.edges))

    clique = ('b', 'c')
    updated_tree = clique_tree.local_update(updated_tree, clique, self.domain)
    self.assertCountEqual(updated_tree.nodes, [('a', 'b'), ('b', 'c')])
    self.assertEqual(updated_tree.number_of_edges(), 1)
    self.assertTrue(updated_tree.has_edge(('a', 'b'), ('b', 'c')))


class SWIFTTest(absltest.TestCase):

  def test_select_queries(self):
    errors = {('a', 'b'): 100, ('b', 'c'): 100, ('a', 'c'): 100}
    domain = mbi.Domain(['a', 'b', 'c'], [2, 3, 4])
    candidates = {key: 1.0 for key in errors}
    max_clique_size = 100
    gdp_budget = 100.0
    selected, jtree = swift.select_queries(
        errors, candidates, domain, max_clique_size, gdp_budget
    )
    self.assertLen(selected, 3)
    self.assertLen(jtree.nodes, 1)

    ab, ac, bc = 6 ** (2 / 3), 8 ** (2 / 3), 12 ** (2 / 3)
    expected_budget_ab = gdp_budget * ab / (ab + ac + bc)
    expected_budget_ac = gdp_budget * ac / (ab + ac + bc)
    expected_budget_bc = gdp_budget * bc / (ab + ac + bc)
    self.assertAlmostEqual(selected[('a', 'b')], expected_budget_ab)
    self.assertAlmostEqual(selected[('a', 'c')], expected_budget_ac)
    self.assertAlmostEqual(selected[('b', 'c')], expected_budget_bc)

  def test_select_queries_nonpositive_errors(self):
    errors = {('a', 'b'): 0.0, ('b', 'c'): -1.0, ('a', 'c'): -2.0}
    domain = mbi.Domain(['a', 'b', 'c'], [2, 3, 4])
    candidates = {key: 1.0 for key in errors}
    gdp_budget = 100.0

    selected, jtree = swift.select_queries(
        errors, candidates, domain, max_clique_size=100, gdp_budget=gdp_budget
    )

    self.assertNotEmpty(jtree.nodes)
    self.assertAlmostEqual(sum(selected.values()), gdp_budget)

  def test_best_subset_and_allocation(self):
    candidates = [
        swift_utils.Candidate(id='a', error=1.0, size=1.0, weight=1.0),
        swift_utils.Candidate(id='b', error=2.0, size=2.0, weight=2.0),
        swift_utils.Candidate(id='c', error=3.0, size=3.0, weight=3.0),
    ]
    budget = 100.0
    allocation = swift_utils.best_subset_and_allocation(candidates, budget)
    self.assertLen(allocation, 3)

  def test_build_clique_tree(self):
    domain = mbi.Domain(['a', 'b', 'c', 'd', 'e', 'f'], [3, 4, 5, 6, 7, 8])
    max_clique_size = 100
    errors = {key: 1.0 for key in itertools.combinations(domain.attributes, 2)}

    tree = swift.build_clique_tree(domain, errors, max_clique_size)
    actual_max_clique_size = max(domain.size(cl) for cl in tree.nodes)
    self.assertLessEqual(actual_max_clique_size, max_clique_size)

    tree = swift.build_best_clique_tree(domain, errors, max_clique_size)
    actual_max_clique_size = max(domain.size(cl) for cl in tree.nodes)
    self.assertLessEqual(actual_max_clique_size, max_clique_size)

  def test_fits_one_way_marginals(self):
    data = mbi.Dataset.synthetic(mbi.Domain(['a', 'b', 'c'], [3, 4, 5]), N=1000)

    config = swift.SWIFTConfig(pgm_iters=500).configure(budget=10000)

    result = config(np.random.default_rng(0), data)

    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertNotEmpty(result.measurements)
    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = result.model.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=1)

  def test_checkpointing(self):
    temp_dir = self.create_tempdir().full_path
    data = mbi.Dataset.synthetic(mbi.Domain(['a', 'b', 'c'], [3, 4, 5]), N=1000)
    config = swift.SWIFTConfig(pgm_iters=10).configure(budget=10.0)

    with _checkpoint.checkpoint(temp_dir):
      result1 = config(np.random.default_rng(0), data)

    measurements_path = os.path.join(temp_dir, 'measurements.npz')
    model_path = os.path.join(temp_dir, 'model.npz')
    self.assertTrue(os.path.exists(measurements_path))
    self.assertTrue(os.path.exists(model_path))

    with _checkpoint.checkpoint(temp_dir):
      result2 = config(np.random.default_rng(1), data)

    self.assertIsInstance(result2, common.DiscreteMechanismResult)
    self.assertLen(result2.measurements, len(result1.measurements))
    self.assertEqual(result2.model.domain, result1.model.domain)


if __name__ == '__main__':
  absltest.main()
