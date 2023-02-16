from __future__ import absolute_import

from funlib.persistence.graphs.mongodb_graph_provider import MongoDbGraphProvider
from funlib.persistence.graphs.file_graph_provider import FileGraphProvider

from funlib.geometry import Roi, Coordinate

import logging
import unittest

logger = logging.getLogger(__name__)
# daisy.scheduler._NO_SPAWN_STATUS_THREAD = True


class TestGraph(unittest.TestCase):
    def mongo_provider_factory(self, mode):
        return MongoDbGraphProvider("test_daisy_graph", mode=mode)

    def file_provider_factory(self, mode):
        return FileGraphProvider("test_daisy_graph", chunk_size=(10, 10, 10), mode=mode)

    # test basic graph io
    def test_graph_io_mongo(self):
        self.run_test_graph_io(self.mongo_provider_factory)

    def test_graph_io_file(self):
        self.run_test_graph_io(self.file_provider_factory)

    # test fail_if_exists flag when writing subgraph
    def test_graph_fail_if_exists_mongo(self):
        self.run_test_graph_fail_if_exists(self.mongo_provider_factory)

    # test fail_if_not_exists flag when writing subgraph
    def test_graph_fail_if_not_exists_mongo(self):
        self.run_test_graph_fail_if_not_exists(self.mongo_provider_factory)

    # test that only specified attributes are written to backend
    def test_graph_write_attributes_mongo(self):
        self.run_test_graph_write_attributes(self.mongo_provider_factory)

    # test that only write nodes inside the write_roi
    def test_graph_write_roi_mongo(self):
        self.run_test_graph_write_roi(self.mongo_provider_factory)

    def test_graph_write_roi_file(self):
        self.run_test_graph_write_roi(self.file_provider_factory)

    # test connected components
    def test_graph_connected_components_mongo(self):
        self.run_test_graph_connected_components(self.mongo_provider_factory)

    # test has_edge
    def test_graph_has_edge_mongo(self):
        self.run_test_graph_has_edge(self.mongo_provider_factory)

    def run_test_graph_io(self, provider_factory):
        graph_provider = provider_factory("w")

        graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

        graph.add_node(2, comment="without position")
        graph.add_node(42, position=(1, 1, 1))
        graph.add_node(23, position=(5, 5, 5), swip="swap")
        graph.add_node(57, position=Coordinate((7, 7, 7)), zap="zip")
        graph.add_edge(42, 23)
        graph.add_edge(57, 23)
        graph.add_edge(2, 42)

        graph.write_nodes()
        graph.write_edges()

        graph_provider = provider_factory("r")
        compare_graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

        nodes = sorted(list(graph.nodes()))
        nodes.remove(2)  # node 2 has no position and will not be queried
        compare_nodes = sorted(list(compare_graph.nodes()))

        edges = sorted(list(graph.edges()))
        edges.remove((2, 42))  # node 2 has no position and will not be queried
        compare_edges = sorted(list(compare_graph.edges()))

        self.assertEqual(nodes, compare_nodes)
        self.assertEqual(edges, compare_edges)

    def run_test_graph_fail_if_exists(self, provider_factory):
        graph_provider = provider_factory("w")
        graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

        graph.add_node(2, comment="without position")
        graph.add_node(42, position=(1, 1, 1))
        graph.add_node(23, position=(5, 5, 5), swip="swap")
        graph.add_node(57, position=Coordinate((7, 7, 7)), zap="zip")
        graph.add_edge(42, 23)
        graph.add_edge(57, 23)
        graph.add_edge(2, 42)

        graph.write_nodes()
        graph.write_edges()
        with self.assertRaises(Exception):
            graph.write_nodes(fail_if_exists=True)
        with self.assertRaises(Exception):
            graph.write_edges(fail_if_exists=True)

    def run_test_graph_fail_if_not_exists(self, provider_factory):
        graph_provider = provider_factory("w")
        graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

        graph.add_node(2, comment="without position")
        graph.add_node(42, position=(1, 1, 1))
        graph.add_node(23, position=(5, 5, 5), swip="swap")
        graph.add_node(57, position=Coordinate((7, 7, 7)), zap="zip")
        graph.add_edge(42, 23)
        graph.add_edge(57, 23)
        graph.add_edge(2, 42)

        with self.assertRaises(Exception):
            graph.write_nodes(fail_if_not_exists=True)
        with self.assertRaises(Exception):
            graph.write_edges(fail_if_not_exists=True)

    def run_test_graph_write_attributes(self, provider_factory):
        graph_provider = provider_factory("w")
        graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

        graph.add_node(2, comment="without position")
        graph.add_node(42, position=(1, 1, 1))
        graph.add_node(23, position=(5, 5, 5), swip="swap")
        graph.add_node(57, position=Coordinate((7, 7, 7)), zap="zip")
        graph.add_edge(42, 23)
        graph.add_edge(57, 23)
        graph.add_edge(2, 42)

        graph.write_nodes(attributes=["position", "swip"])
        graph.write_edges()

        graph_provider = provider_factory("r")
        compare_graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

        nodes = []
        for node, data in graph.nodes(data=True):
            if node == 2:
                continue
            if "zap" in data:
                del data["zap"]
            data["position"] = list(data["position"])
            nodes.append((node, data))

        compare_nodes = compare_graph.nodes(data=True)
        compare_nodes = [
            (node_id, data) for node_id, data in compare_nodes if len(data) > 0
        ]
        self.assertCountEqual(nodes, compare_nodes)

    def run_test_graph_write_roi(self, provider_factory):
        graph_provider = provider_factory("w")
        graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

        graph.add_node(2, comment="without position")
        graph.add_node(42, position=(1, 1, 1))
        graph.add_node(23, position=(5, 5, 5), swip="swap")
        graph.add_node(57, position=Coordinate((7, 7, 7)), zap="zip")
        graph.add_edge(42, 23)
        graph.add_edge(57, 23)
        graph.add_edge(2, 42)

        write_roi = Roi((0, 0, 0), (6, 6, 6))
        graph.write_nodes(roi=write_roi)
        graph.write_edges(roi=write_roi)

        graph_provider = provider_factory("r")
        compare_graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

        nodes = sorted(list(graph.nodes()))
        nodes.remove(2)  # node 2 has no position and will not be queried
        nodes.remove(57)  # node 57 is outside of the write_roi
        compare_nodes = compare_graph.nodes(data=True)
        compare_nodes = [node_id for node_id, data in compare_nodes if len(data) > 0]
        compare_nodes = sorted(list(compare_nodes))
        edges = sorted(list(graph.edges()))
        edges.remove((2, 42))  # node 2 has no position and will not be queried
        compare_edges = sorted(list(compare_graph.edges()))

        self.assertEqual(nodes, compare_nodes)
        self.assertEqual(edges, compare_edges)

    def run_test_graph_connected_components(self, provider_factory):
        graph_provider = provider_factory("w")
        graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

        graph.add_node(2, comment="without position")
        graph.add_node(42, position=(1, 1, 1))
        graph.add_node(23, position=(5, 5, 5), swip="swap")
        graph.add_node(57, position=Coordinate((7, 7, 7)), zap="zip")
        graph.add_edge(57, 23)
        graph.add_edge(2, 42)

        components = graph.get_connected_components()
        self.assertEqual(len(components), 2)
        c1, c2 = components
        n1 = sorted(list(c1.nodes()))
        n2 = sorted(list(c2.nodes()))

        compare_n1 = [2, 42]
        compare_n2 = [23, 57]

        if 2 in n2:
            temp = n2
            n2 = n1
            n1 = temp

        self.assertCountEqual(n1, compare_n1)
        self.assertCountEqual(n2, compare_n2)

    def run_test_graph_has_edge(self, provider_factory):
        graph_provider = provider_factory("w")

        roi = Roi((0, 0, 0), (10, 10, 10))
        graph = graph_provider[roi]

        graph.add_node(2, comment="without position")
        graph.add_node(42, position=(1, 1, 1))
        graph.add_node(23, position=(5, 5, 5), swip="swap")
        graph.add_node(57, position=Coordinate((7, 7, 7)), zap="zip")
        graph.add_edge(42, 23)
        graph.add_edge(57, 23)

        write_roi = Roi((0, 0, 0), (6, 6, 6))
        graph.write_nodes(roi=write_roi)
        graph.write_edges(roi=write_roi)

        self.assertTrue(graph_provider.has_edges(roi))
