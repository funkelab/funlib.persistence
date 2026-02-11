import networkx as nx
import pytest
from funlib.geometry import Roi

from funlib.persistence.graphs import PgSQLGraphDatabase
from funlib.persistence.types import Vec


def _skip_if_bulk_unsupported(provider, write_method):
    if write_method == "bulk" and not isinstance(provider, PgSQLGraphDatabase):
        pytest.skip("Bulk write only supported on PostgreSQL")


def _write_nodes(provider, nodes, write_method, **kwargs):
    if write_method == "bulk":
        provider.bulk_write_nodes(nodes, **kwargs)
    else:
        provider.write_nodes(nodes, **kwargs)


def _write_edges(provider, nodes, edges, write_method, **kwargs):
    if write_method == "bulk":
        provider.bulk_write_edges(nodes, edges, **kwargs)
    else:
        provider.write_edges(nodes, edges, **kwargs)


def _write_graph(provider, graph, write_method, **kwargs):
    if write_method == "bulk":
        kwargs.pop("fail_if_exists", None)
        kwargs.pop("delete", None)
        provider.bulk_write_graph(graph, **kwargs)
    else:
        provider.write_graph(graph, **kwargs)


def test_graph_filtering(provider_factory, write_method):
    graph_writer = provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3), "selected": bool},
        edge_attrs={"selected": bool},
    )
    _skip_if_bulk_unsupported(graph_writer, write_method)
    roi = Roi((0, 0, 0), (10, 10, 10))
    graph = graph_writer[roi]

    graph.add_node(2, position=(2, 2, 2), selected=True)
    graph.add_node(42, position=(1, 1, 1), selected=False)
    graph.add_node(23, position=(5, 5, 5), selected=True)
    graph.add_node(57, position=(7, 7, 7), selected=True)
    graph.add_edge(42, 23, selected=False)
    graph.add_edge(57, 23, selected=True)
    graph.add_edge(2, 42, selected=True)

    _write_nodes(graph_writer, graph.nodes(), write_method)
    _write_edges(graph_writer, graph.nodes(), graph.edges(), write_method)

    graph_reader = provider_factory("r")

    filtered_nodes = graph_reader.read_nodes(roi, attr_filter={"selected": True})
    filtered_node_ids = [node["id"] for node in filtered_nodes]
    expected_node_ids = [2, 23, 57]
    assert expected_node_ids == filtered_node_ids

    filtered_edges = graph_reader.read_edges(roi, attr_filter={"selected": True})
    filtered_edge_endpoints = [(edge["u"], edge["v"]) for edge in filtered_edges]
    expected_edge_endpoints = [(57, 23), (2, 42)]
    for u, v in expected_edge_endpoints:
        assert (u, v) in filtered_edge_endpoints or (v, u) in filtered_edge_endpoints

    filtered_subgraph = graph_reader.read_graph(
        roi, nodes_filter={"selected": True}, edges_filter={"selected": True}
    )
    nodes_with_position = [
        node for node, data in filtered_subgraph.nodes(data=True) if "position" in data
    ]
    assert expected_node_ids == nodes_with_position
    assert len(filtered_subgraph.edges()) == len(expected_edge_endpoints)
    for u, v in expected_edge_endpoints:
        assert (u, v) in filtered_subgraph.edges() or (
            v,
            u,
        ) in filtered_subgraph.edges()


def test_graph_filtering_complex(provider_factory, write_method):
    graph_provider = provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3), "selected": bool, "test": str},
        edge_attrs={"selected": bool, "a": int, "b": int},
    )
    _skip_if_bulk_unsupported(graph_provider, write_method)
    roi = Roi((0, 0, 0), (10, 10, 10))
    graph = graph_provider[roi]

    graph.add_node(2, position=(2, 2, 2), selected=True, test="test")
    graph.add_node(42, position=(1, 1, 1), selected=False, test="test2")
    graph.add_node(23, position=(5, 5, 5), selected=True, test="test2")
    graph.add_node(57, position=(7, 7, 7), selected=True, test="test")

    graph.add_edge(42, 23, selected=False, a=100, b=3)
    graph.add_edge(57, 23, selected=True, a=100, b=2)
    graph.add_edge(2, 42, selected=True, a=101, b=3)

    _write_nodes(graph_provider, graph.nodes(), write_method)
    _write_edges(graph_provider, graph.nodes(), graph.edges(), write_method)

    graph_provider = provider_factory("r")

    filtered_nodes = graph_provider.read_nodes(
        roi, attr_filter={"selected": True, "test": "test"}
    )
    filtered_node_ids = [node["id"] for node in filtered_nodes]
    expected_node_ids = [2, 57]
    assert expected_node_ids == filtered_node_ids

    filtered_edges = graph_provider.read_edges(
        roi, attr_filter={"selected": True, "a": 100}
    )
    filtered_edge_endpoints = [(edge["u"], edge["v"]) for edge in filtered_edges]
    expected_edge_endpoints = [(57, 23)]
    for u, v in expected_edge_endpoints:
        assert (u, v) in filtered_edge_endpoints or (v, u) in filtered_edge_endpoints

    filtered_subgraph = graph_provider.read_graph(
        roi,
        nodes_filter={"selected": True, "test": "test"},
        edges_filter={"selected": True, "a": 100},
    )
    nodes_with_position = [
        node for node, data in filtered_subgraph.nodes(data=True) if "position" in data
    ]
    assert expected_node_ids == nodes_with_position
    assert len(filtered_subgraph.edges()) == 0


def test_graph_read_and_update_specific_attrs(provider_factory):
    graph_provider = provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3), "selected": bool, "test": str},
        edge_attrs={"selected": bool, "a": int, "b": int, "c": int},
    )
    roi = Roi((0, 0, 0), (10, 10, 10))
    graph = graph_provider[roi]

    graph.add_node(2, position=(2, 2, 2), selected=True, test="test")
    graph.add_node(42, position=(1, 1, 1), selected=False, test="test2")
    graph.add_node(23, position=(5, 5, 5), selected=True, test="test2")
    graph.add_node(57, position=(7, 7, 7), selected=True, test="test")

    graph.add_edge(42, 23, selected=False, a=100, b=3)
    graph.add_edge(57, 23, selected=True, a=100, b=2)
    graph.add_edge(2, 42, selected=True, a=101, b=3)

    graph_provider.write_graph(graph)

    graph_provider = provider_factory("r+")
    limited_graph = graph_provider.read_graph(
        roi, node_attrs=["selected"], edge_attrs=["c"]
    )

    for node, data in limited_graph.nodes(data=True):
        assert "test" not in data
        assert "selected" in data
        data["selected"] = True

    for u, v, data in limited_graph.edges(data=True):
        assert "a" not in data
        assert "b" not in data
        nx.set_edge_attributes(limited_graph, 5, "c")

    try:
        graph_provider.write_attrs(
            limited_graph, edge_attrs=["c"], node_attrs=["selected"]
        )
    except NotImplementedError:
        pytest.xfail()

    updated_graph = graph_provider.read_graph(roi)

    for node, data in updated_graph.nodes(data=True):
        assert data["selected"]

    for u, v, data in updated_graph.edges(data=True):
        assert data["c"] == 5


def test_graph_read_unbounded_roi(provider_factory, write_method):
    graph_provider = provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3), "selected": bool, "test": str},
        edge_attrs={"selected": bool, "a": int, "b": int},
    )
    _skip_if_bulk_unsupported(graph_provider, write_method)
    roi = Roi((0, 0, 0), (10, 10, 10))
    unbounded_roi = Roi((None, None, None), (None, None, None))

    graph = graph_provider[roi]

    graph.add_node(2, position=(2, 2, 2), selected=True, test="test")
    graph.add_node(42, position=(1, 1, 1), selected=False, test="test2")
    graph.add_node(23, position=(5, 5, 5), selected=True, test="test2")
    graph.add_node(57, position=(7, 7, 7), selected=True, test="test")

    graph.add_edge(42, 23, selected=False, a=100, b=3)
    graph.add_edge(57, 23, selected=True, a=100, b=2)
    graph.add_edge(2, 42, selected=True, a=101, b=3)

    _write_nodes(graph_provider, graph.nodes(), write_method)
    _write_edges(graph_provider, graph.nodes(), graph.edges(), write_method)

    graph_provider = provider_factory("r+")
    limited_graph = graph_provider.read_graph(
        unbounded_roi, node_attrs=["selected"], edge_attrs=["c"]
    )

    seen = []
    for node, data in limited_graph.nodes(data=True):
        assert "test" not in data
        assert "selected" in data
        data["selected"] = True
        seen.append(node)

    assert sorted([2, 42, 23, 57]) == sorted(seen)


def test_graph_read_meta_values(provider_factory):
    roi = Roi((0, 0, 0), (10, 10, 10))
    provider_factory("w", True, roi, node_attrs={"position": Vec(float, 3)})
    graph_provider = provider_factory("r", None, None)
    assert True == graph_provider.directed
    assert roi == graph_provider.total_roi


def test_graph_default_meta_values(provider_factory):
    provider = provider_factory(
        "w", False, None, node_attrs={"position": Vec(float, 3)}
    )
    assert False == provider.directed
    assert provider.total_roi is None or provider.total_roi == Roi(
        (None, None, None), (None, None, None)
    )
    graph_provider = provider_factory("r", False, None)
    assert False == graph_provider.directed
    assert graph_provider.total_roi is None or graph_provider.total_roi == Roi(
        (None, None, None), (None, None, None)
    )


def test_graph_io(provider_factory, write_method):
    graph_provider = provider_factory(
        "w",
        node_attrs={
            "position": Vec(float, 3),
            "swip": str,
            "zap": str,
        },
    )
    _skip_if_bulk_unsupported(graph_provider, write_method)

    graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

    graph.add_node(2, position=(0, 0, 0))
    graph.add_node(42, position=(1, 1, 1))
    graph.add_node(23, position=(5, 5, 5), swip="swap")
    graph.add_node(57, position=(7, 7, 7), zap="zip")
    graph.add_edge(42, 23)
    graph.add_edge(57, 23)
    graph.add_edge(2, 42)

    _write_nodes(graph_provider, graph.nodes(), write_method)
    _write_edges(graph_provider, graph.nodes(), graph.edges(), write_method)

    graph_provider = provider_factory("r")
    compare_graph = graph_provider[Roi((1, 1, 1), (9, 9, 9))]

    nodes = sorted(list(graph.nodes()))
    nodes.remove(2)  # node 2 has no position and will not be queried
    compare_nodes = sorted(list(compare_graph.nodes()))

    edges = sorted(tuple(sorted(e)) for e in list(graph.edges()))
    edges.remove((2, 42))  # node 2 has no position and will not be queried
    compare_edges = sorted(tuple(sorted(e)) for e in list(compare_graph.edges()))

    assert nodes == compare_nodes
    assert edges == compare_edges


def test_graph_fail_if_exists(provider_factory):
    graph_provider = provider_factory(
        "w",
        node_attrs={
            "position": Vec(float, 3),
            "swip": str,
            "zap": str,
        },
    )
    graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

    graph.add_node(2, position=(0, 0, 0))
    graph.add_node(42, position=(1, 1, 1))
    graph.add_node(23, position=(5, 5, 5), swip="swap")
    graph.add_node(57, position=(7, 7, 7), zap="zip")
    graph.add_edge(42, 23)
    graph.add_edge(57, 23)
    graph.add_edge(2, 42)

    graph_provider.write_graph(graph)
    with pytest.raises(Exception):
        graph_provider.write_nodes(graph.nodes(), fail_if_exists=True)
    with pytest.raises(Exception):
        graph_provider.write_edges(graph.nodes(), graph.edges(), fail_if_exists=True)


def test_graph_duplicate_insert_behavior(provider_factory):
    """Test that fail_if_exists controls whether duplicate inserts raise."""
    graph_provider = provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3), "selected": bool},
        edge_attrs={"selected": bool},
    )
    roi = Roi((0, 0, 0), (10, 10, 10))
    graph = graph_provider[roi]

    graph.add_node(2, position=(2, 2, 2), selected=True)
    graph.add_node(42, position=(1, 1, 1), selected=False)
    graph.add_edge(2, 42, selected=True)

    # Initial write
    graph_provider.write_nodes(graph.nodes())
    graph_provider.write_edges(graph.nodes(), graph.edges())

    # fail_if_exists=True should raise on duplicate nodes and edges
    with pytest.raises(Exception):
        graph_provider.write_nodes(graph.nodes(), fail_if_exists=True)
    with pytest.raises(Exception):
        graph_provider.write_edges(graph.nodes(), graph.edges(), fail_if_exists=True)

    # fail_if_exists=False should silently ignore duplicates
    graph_provider.write_nodes(graph.nodes(), fail_if_exists=False)
    graph_provider.write_edges(graph.nodes(), graph.edges(), fail_if_exists=False)

    # Verify the original data is still intact
    graph_provider = provider_factory("r")
    result = graph_provider.read_graph(roi)
    assert set(result.nodes()) == {2, 42}
    assert len(result.edges()) == 1


def test_graph_fail_if_not_exists(provider_factory):
    graph_provider = provider_factory(
        "w",
        node_attrs={
            "position": Vec(float, 3),
            "swip": str,
            "zap": str,
        },
    )
    graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

    graph.add_node(2, position=(0, 0, 0))
    graph.add_node(42, position=(1, 1, 1))
    graph.add_node(23, position=(5, 5, 5), swip="swap")
    graph.add_node(57, position=(7, 7, 7), zap="zip")
    graph.add_edge(42, 23)
    graph.add_edge(57, 23)
    graph.add_edge(2, 42)

    with pytest.raises(Exception):
        graph_provider.write_nodes(graph.nodes(), fail_if_not_exists=True)
    with pytest.raises(Exception):
        graph_provider.write_edges(
            graph.nodes(), graph.edges(), fail_if_not_exists=True
        )


def test_graph_write_attributes(provider_factory, write_method):
    graph_provider = provider_factory(
        "w",
        node_attrs={
            "position": Vec(int, 3),
            "swip": str,
            "zap": str,
        },
    )
    _skip_if_bulk_unsupported(graph_provider, write_method)
    graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

    graph.add_node(2, position=[0, 0, 0])
    graph.add_node(42, position=[1, 1, 1])
    graph.add_node(23, position=[5, 5, 5], swip="swap")
    graph.add_node(57, position=[7, 7, 7], zap="zip")
    graph.add_edge(42, 23)
    graph.add_edge(57, 23)
    graph.add_edge(2, 42)

    _write_graph(
        graph_provider, graph, write_method,
        write_nodes=True, write_edges=False, node_attrs=["position", "swip"],
    )

    _write_edges(graph_provider, graph.nodes(), graph.edges(), write_method)

    graph_provider = provider_factory("r")
    compare_graph = graph_provider[Roi((1, 1, 1), (10, 10, 10))]

    nodes = []
    for node, data in graph.nodes(data=True):
        if node == 2:
            continue
        if "zap" in data:
            del data["zap"]
        nodes.append((node, data))

    compare_nodes = compare_graph.nodes(data=True)
    compare_nodes = [
        (node_id, data) for node_id, data in compare_nodes if len(data) > 0
    ]
    for n, c in zip(nodes, compare_nodes):
        assert n[0] == c[0]
        for key in n[1]:
            assert key in c[1]
            v1 = n[1][key]
            v2 = c[1][key]
            try:
                for e1, e2 in zip(v1, v2):
                    assert e1 == e2
            except:
                assert v1 == v2


def test_graph_write_roi(provider_factory, write_method):
    graph_provider = provider_factory(
        "w",
        node_attrs={
            "position": Vec(float, 3),
            "swip": str,
            "zap": str,
        },
    )
    _skip_if_bulk_unsupported(graph_provider, write_method)
    graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

    graph.add_node(2, position=(0, 0, 0))
    graph.add_node(42, position=(1, 1, 1))
    graph.add_node(23, position=(5, 5, 5), swip="swap")
    graph.add_node(57, position=(7, 7, 7), zap="zip")
    graph.add_edge(42, 23)
    graph.add_edge(57, 23)
    graph.add_edge(2, 42)

    write_roi = Roi((0, 0, 0), (6, 6, 6))
    _write_graph(graph_provider, graph, write_method, roi=write_roi)

    graph_provider = provider_factory("r")
    compare_graph = graph_provider[Roi((1, 1, 1), (9, 9, 9))]

    nodes = sorted(list(graph.nodes()))
    nodes.remove(2)  # node 2 has no position and will not be queried
    nodes.remove(57)  # node 57 is outside of the write_roi
    compare_nodes = compare_graph.nodes(data=True)
    compare_nodes = [node_id for node_id, data in compare_nodes if len(data) > 0]
    compare_nodes = sorted(list(compare_nodes))
    edges = sorted(tuple(sorted(e)) for e in list(graph.edges()))
    edges.remove((2, 42))  # node 2 has no position and will not be queried
    compare_edges = sorted(tuple(sorted(e)) for e in list(compare_graph.edges()))

    assert nodes == compare_nodes
    assert edges == compare_edges


def test_graph_connected_components(provider_factory):
    graph_provider = provider_factory(
        "w",
        node_attrs={
            "position": Vec(float, 3),
            "swip": str,
            "zap": str,
        },
    )
    graph = graph_provider[Roi((0, 0, 0), (10, 10, 10))]

    graph.add_node(2, position=(0, 0, 0))
    graph.add_node(42, position=(1, 1, 1))
    graph.add_node(23, position=(5, 5, 5), swip="swap")
    graph.add_node(57, position=(7, 7, 7), zap="zip")
    graph.add_edge(57, 23)
    graph.add_edge(2, 42)
    try:
        components = list(nx.connected_components(graph))
    except NotImplementedError:
        pytest.xfail()
    assert len(components) == 2
    c1, c2 = components
    n1 = sorted(list(c1))
    n2 = sorted(list(c2))

    compare_n1 = [2, 42]
    compare_n2 = [23, 57]

    if 2 in n2:
        temp = n2
        n2 = n1
        n1 = temp

    assert n1 == compare_n1
    assert n2 == compare_n2


def test_graph_has_edge(provider_factory, write_method):
    graph_provider = provider_factory(
        "w",
        node_attrs={
            "position": Vec(float, 3),
            "swip": str,
            "zap": str,
        },
    )
    _skip_if_bulk_unsupported(graph_provider, write_method)

    roi = Roi((0, 0, 0), (10, 10, 10))
    graph = graph_provider[roi]

    graph.add_node(2, position=(0, 0, 0))
    graph.add_node(42, position=(1, 1, 1))
    graph.add_node(23, position=(5, 5, 5), swip="swap")
    graph.add_node(57, position=(7, 7, 7), zap="zip")
    graph.add_edge(42, 23)
    graph.add_edge(57, 23)

    write_roi = Roi((0, 0, 0), (6, 6, 6))
    _write_nodes(graph_provider, graph.nodes(), write_method, roi=write_roi)
    _write_edges(graph_provider, graph.nodes(), graph.edges(), write_method, roi=write_roi)

    assert graph_provider.has_edges(roi)


def test_read_edges_join_vs_in_clause(provider_factory, write_method):
    """Benchmark: read_edges with JOIN (roi-only) vs IN clause (nodes list).

    Demonstrates that the JOIN path avoids serializing a large node ID list
    into the SQL query, and lets the DB optimizer do the work instead.
    """
    import time
    from itertools import product

    size = 50  # 50^3 = 125,000 nodes
    graph_provider = provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3)},
    )
    _skip_if_bulk_unsupported(graph_provider, write_method)

    # Build a 3D grid graph
    graph = nx.Graph()
    for x, y, z in product(range(size), repeat=3):
        node_id = x * size * size + y * size + z
        graph.add_node(node_id, position=(x + 0.5, y + 0.5, z + 0.5))
        # Connect to neighbors in +x, +y, +z directions
        if x > 0:
            graph.add_edge(node_id, (x - 1) * size * size + y * size + z)
        if y > 0:
            graph.add_edge(node_id, x * size * size + (y - 1) * size + z)
        if z > 0:
            graph.add_edge(node_id, x * size * size + y * size + (z - 1))

    _write_graph(graph_provider, graph, write_method)

    # Re-open in read mode
    graph_provider = provider_factory("r")

    query_roi = Roi((10, 10, 10), (30, 30, 30))
    n_repeats = 5

    # --- Old approach: read_nodes, then read_edges with nodes list ---
    times_in_clause = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        nodes = graph_provider.read_nodes(query_roi)
        edges_via_in = graph_provider.read_edges(nodes=nodes)
        t1 = time.perf_counter()
        times_in_clause.append(t1 - t0)

    # --- New approach: read_edges with roi (JOIN) ---
    times_join = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        edges_via_join = graph_provider.read_edges(roi=query_roi)
        t1 = time.perf_counter()
        times_join.append(t1 - t0)

    avg_in = sum(times_in_clause) / n_repeats
    avg_join = sum(times_join) / n_repeats

    print(f"\n--- read_edges benchmark (roi covers {30**3:,} of {size**3:,} nodes) ---")
    print(f"IN clause (2 queries):  {avg_in*1000:.1f} ms avg")
    print(f"JOIN      (1 query):    {avg_join*1000:.1f} ms avg")
    print(f"Speedup: {avg_in / avg_join:.2f}x")

    # Both should return edges — just verify they're non-empty and reasonable
    assert len(edges_via_in) > 0
    assert len(edges_via_join) > 0
    # JOIN finds edges where either endpoint is in ROI (superset of IN approach)
    assert len(edges_via_join) == len(edges_via_in)


def test_read_edges_fetch_on_v(provider_factory, write_method):
    """Test that fetch_on_v controls whether edges are matched on u only or both endpoints.

    Graph layout (1D for clarity, stored as 3D positions):

        Node 1 (pos 1) -- Edge(1,5) -- Node 5 (pos 5)
        Node 2 (pos 2) -- Edge(2,8) -- Node 8 (pos 8)
        Node 5 (pos 5) -- Edge(5,8) -- Node 8 (pos 8)
        Node 8 (pos 8) -- Edge(8,9) -- Node 9 (pos 9)

    ROI = [0, 6) covers nodes {1, 2, 5}.

    Undirected edges are stored with u < v, so:
      - Edge(1, 5): u=1 in ROI, v=5 in ROI
      - Edge(2, 8): u=2 in ROI, v=8 outside ROI
      - Edge(5, 8): u=5 in ROI, v=8 outside ROI
      - Edge(8, 9): u=8 outside ROI, v=9 outside ROI

    fetch_on_v=False (default): only edges where u is in ROI → {(1,5), (2,8), (5,8)}
    fetch_on_v=True: edges where u OR v is in ROI → {(1,5), (2,8), (5,8)}
        (same here because u < v and all boundary-crossing edges have u inside)

    To properly test fetch_on_v, we need an edge where u is OUTSIDE the ROI
    but v is INSIDE. With undirected u < v storage, this means a node with a
    smaller ID outside the ROI connected to a node with a larger ID inside.

    So we add: Node 0 (pos 8) -- Edge(0, 5): u=0 outside ROI, v=5 in ROI.
    """
    graph_provider = provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3)},
    )
    _skip_if_bulk_unsupported(graph_provider, write_method)
    roi = Roi((0, 0, 0), (6, 6, 6))

    graph = nx.Graph()
    # Nodes inside ROI (positions < 6)
    graph.add_node(1, position=(1.0, 1.0, 1.0))
    graph.add_node(2, position=(2.0, 2.0, 2.0))
    graph.add_node(5, position=(5.0, 5.0, 5.0))
    # Nodes outside ROI (positions >= 6)
    # Node 0 has ID < all ROI nodes but position outside ROI
    graph.add_node(0, position=(8.0, 8.0, 8.0))
    graph.add_node(8, position=(8.0, 8.0, 8.0))
    graph.add_node(9, position=(9.0, 9.0, 9.0))

    # Edges: undirected, stored as u < v
    graph.add_edge(1, 5)  # both in ROI
    graph.add_edge(2, 8)  # u in ROI, v outside
    graph.add_edge(5, 8)  # u in ROI, v outside
    graph.add_edge(8, 9)  # both outside ROI
    graph.add_edge(0, 5)  # u=0 OUTSIDE ROI, v=5 INSIDE ROI (key test edge)

    _write_graph(graph_provider, graph, write_method)

    graph_provider = provider_factory("r")

    def edge_set(edges):
        """Normalize edge list to set of sorted tuples for comparison."""
        return {(min(e["u"], e["v"]), max(e["u"], e["v"])) for e in edges}

    # --- Case 1: nodes passed explicitly ---
    nodes_in_roi = graph_provider.read_nodes(roi)
    node_ids_in_roi = {n["id"] for n in nodes_in_roi}
    assert node_ids_in_roi == {1, 2, 5}

    edges_u_only = graph_provider.read_edges(nodes=nodes_in_roi, fetch_on_v=False)
    edges_u_and_v = graph_provider.read_edges(nodes=nodes_in_roi, fetch_on_v=True)

    # fetch_on_v=False: only edges where u IN (1,2,5)
    # (1,5), (2,8), (5,8) match; (0,5) does NOT match (u=0 not in list)
    assert edge_set(edges_u_only) == {(1, 5), (2, 8), (5, 8)}

    # fetch_on_v=True: edges where u OR v IN (1,2,5)
    # (0,5) now matches because v=5 is in the list
    assert edge_set(edges_u_and_v) == {(0, 5), (1, 5), (2, 8), (5, 8)}

    # --- Case 2: roi passed (JOIN path) ---
    edges_roi_u_only = graph_provider.read_edges(roi=roi, fetch_on_v=False)
    edges_roi_u_and_v = graph_provider.read_edges(roi=roi, fetch_on_v=True)

    # Same expected results as Case 1
    assert edge_set(edges_roi_u_only) == {(1, 5), (2, 8), (5, 8)}
    assert edge_set(edges_roi_u_and_v) == {(0, 5), (1, 5), (2, 8), (5, 8)}

    # --- Case 3: via read_graph ---
    graph_u_only = graph_provider.read_graph(roi, fetch_on_v=False)
    graph_u_and_v = graph_provider.read_graph(roi, fetch_on_v=True)

    graph_edges_u_only = {tuple(sorted(e)) for e in graph_u_only.edges()}
    graph_edges_u_and_v = {tuple(sorted(e)) for e in graph_u_and_v.edges()}

    assert graph_edges_u_only == {(1, 5), (2, 8), (5, 8)}
    assert graph_edges_u_and_v == {(0, 5), (1, 5), (2, 8), (5, 8)}


def test_bulk_write_benchmark(provider_factory):
    """Benchmark: standard write_graph vs bulk_write_graph (COPY).

    Only runs on PostgreSQL since bulk write uses COPY.
    Uses blockwise writes for the standard path to avoid building a single
    massive INSERT statement that blocks on remote connections.
    """
    import time
    from itertools import product

    size = 30  # 30^3 = 27,000 nodes
    block_size = 10
    graph_provider = provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3)},
    )
    if not isinstance(graph_provider, PgSQLGraphDatabase):
        pytest.skip("Bulk write only supported on PostgreSQL")

    # Build a 3D grid graph
    graph = nx.Graph()
    for x, y, z in product(range(size), repeat=3):
        node_id = x * size * size + y * size + z
        graph.add_node(node_id, position=(x + 0.5, y + 0.5, z + 0.5))
        if x > 0:
            graph.add_edge(node_id, (x - 1) * size * size + y * size + z)
        if y > 0:
            graph.add_edge(node_id, x * size * size + (y - 1) * size + z)
        if z > 0:
            graph.add_edge(node_id, x * size * size + y * size + (z - 1))

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    # --- Standard write (blockwise to avoid giant INSERT statements) ---
    t0 = time.perf_counter()
    graph_provider.write_graph(graph)
    t_standard = time.perf_counter() - t0

    # Verify standard write then close connection to release locks
    graph_reader = provider_factory("r")
    result = graph_reader.read_graph()
    assert result.number_of_nodes() == n_nodes
    assert result.number_of_edges() == n_edges
    graph_reader.connection.close()

    # --- Bulk write (recreate tables) ---
    graph_provider = provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3)},
    )
    t0 = time.perf_counter()
    graph_provider.bulk_write_graph(graph)
    t_bulk = time.perf_counter() - t0

    # Verify bulk write
    graph_reader = provider_factory("r")
    result = graph_reader.read_graph()
    assert result.number_of_nodes() == n_nodes
    assert result.number_of_edges() == n_edges

    print(f"\n--- write benchmark ({n_nodes:,} nodes, {n_edges:,} edges) ---")
    print(f"Standard (blockwise): {t_standard*1000:.1f} ms")
    print(f"Bulk (COPY):          {t_bulk*1000:.1f} ms")
    print(f"Speedup:              {t_standard / t_bulk:.2f}x")
