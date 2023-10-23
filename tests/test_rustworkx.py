from funlib.geometry import Roi

import rustworkx as rx
import pytest


def test_graph_read_meta_values(provider_factory):
    roi = Roi((0, 0, 0), (10, 10, 10))
    provider_factory("w", True, roi)
    graph_provider = provider_factory("r", None, None)
    assert True == graph_provider.directed
    assert roi == graph_provider.total_roi


def test_graph_default_meta_values(provider_factory):
    provider = provider_factory("w", None, None)
    assert False == provider.directed
    assert provider.total_roi == Roi((None, None, None), (None, None, None))
    graph_provider = provider_factory("r", None, None)
    assert False == graph_provider.directed
    assert graph_provider.total_roi == Roi((None, None, None), (None, None, None))


def test_graph_nonmatching_meta_values(provider_factory):
    roi = Roi((0, 0, 0), (10, 10, 10))
    roi2 = Roi((1, 0, 0), (10, 10, 10))
    provider_factory("w", True, None)
    with pytest.raises(ValueError):
        provider_factory("r", False, None)
    provider_factory("w", None, roi)
    with pytest.raises(ValueError):
        provider_factory("r", None, roi2)


def test_graph_write_roi(provider_factory):
    graph_provider = provider_factory(
        "w", node_attrs=["swip", "zap"], edge_attrs=["swop", "zop"]
    )
    graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

    n_2 = graph.add_node({"id": 2, "comment": "without position"})
    n_42 = graph.add_node({"id": 42, "z": 1, "y": 1, "x": 1})
    n_23 = graph.add_node({"id": 23, "z": 5, "y": 5, "x": 5, "swip": "swap"})
    n_57 = graph.add_node({"id": 57, "z": 7, "y": 7, "x": 7, "zap": "zip"})
    graph.add_edge(n_42, n_23, {"swop": "swup"})
    graph.add_edge(n_57, n_23, {"zop": "zup"})
    graph.add_edge(n_2, n_42, {})

    write_roi = Roi((0, 0, 0), (6, 6, 6))
    graph_provider.write_graph(graph, roi=write_roi)

    graph_provider = provider_factory("r")
    compare_graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

    nodes = sorted([node_attrs["id"] for node_attrs in graph.nodes()])
    nodes.remove(2)  # node 2 has no position and will not be queried
    nodes.remove(57)  # node 57 is outside of the write_roi
    compare_nodes = [node_attrs["id"] for node_attrs in compare_graph.nodes()]
    compare_nodes = sorted(list(compare_nodes))
    edges = sorted(
        [
            (
                graph.nodes()[u]["id"],
                graph.nodes()[v]["id"],
                edge_data.get("swop", None),
            )
            for (u, v), edge_data in zip(graph.edge_list(), graph.edges())
        ]
    )
    edges.remove((2, 42, None))  # node 2 has no position and will not be queried
    edges.remove((57, 23, None))  # node 57 is outside the roi and will not be queried
    compare_edges = sorted(
        [
            (
                compare_graph.nodes()[u]["id"],
                compare_graph.nodes()[v]["id"],
                edge_data.get("swop", None),
            )
            for (u, v), edge_data in zip(
                compare_graph.edge_list(), compare_graph.edges()
            )
        ]
    )

    assert set([edge[2] for edge in compare_edges]) == {"swup"}, compare_edges

    assert nodes == compare_nodes
    assert edges == compare_edges


def test_graph_connected_components(provider_factory):
    graph_provider = provider_factory("w")
    graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

    n_2 = graph.add_node({"id": 2, "comment": "without position"})
    n_42 = graph.add_node({"id": 42, "z": 1, "y": 1, "x": 1})
    n_23 = graph.add_node({"id": 23, "z": 5, "y": 5, "x": 5})
    n_57 = graph.add_node({"id": 57, "z": 7, "y": 7, "x": 7})
    graph.add_edge(n_57, n_23, {})
    graph.add_edge(n_2, n_42, {})

    nodes = graph.nodes()
    components = rx.connected_components(graph)
    assert len(components) == 2
    c1, c2 = components
    n1 = sorted([nodes[n]["id"] for n in c1])
    n2 = sorted([nodes[n]["id"] for n in c2])

    compare_n1 = [2, 42]
    compare_n2 = [23, 57]

    if 2 in n2:
        temp = n2
        n2 = n1
        n1 = temp

    assert n1 == compare_n1
    assert n2 == compare_n2


def test_graph_has_edge(provider_factory):
    graph_provider = provider_factory("w")

    roi = Roi((0, 0, 0), (10, 10, 10))
    graph = graph_provider[roi]

    n_2 = graph.add_node({"id": 2, "comment": "without position"})
    n_42 = graph.add_node({"id": 42, "z": 1, "y": 1, "x": 1})
    n_23 = graph.add_node({"id": 23, "z": 5, "y": 5, "x": 5})
    n_57 = graph.add_node({"id": 57, "z": 7, "y": 7, "x": 7})
    graph.add_edge(n_42, n_23, {})
    graph.add_edge(n_57, n_23, {})

    write_roi = Roi((0, 0, 0), (6, 6, 6))
    graph_provider.write_graph(graph, roi=write_roi)

    assert graph_provider.has_edges(roi)
