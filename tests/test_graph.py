import networkx as nx
import numpy as np
import pytest
from funlib.geometry import Roi

from funlib.persistence.graphs import SQLiteGraphDataBase
from funlib.persistence.types import Vec


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


def _selected_graph():
    graph = nx.Graph()
    graph.add_node(2, position=(2, 2, 2), selected=True, test="test")
    graph.add_node(42, position=(1, 1, 1), selected=False, test="test2")
    graph.add_node(23, position=(5, 5, 5), selected=True, test="test2")
    graph.add_node(57, position=(7, 7, 7), selected=True, test="test")
    graph.add_edge(42, 23, selected=False, a=100, b=3)
    graph.add_edge(57, 23, selected=True, a=100, b=2)
    graph.add_edge(2, 42, selected=True, a=101, b=3)
    return graph


SELECTED = dict(
    node_attrs={"position": Vec(float, 3), "selected": bool, "test": str},
    edge_attrs={"selected": bool, "a": int, "b": int},
)


def _swip_zap_graph():
    graph = nx.Graph()
    graph.add_node(2, position=(0, 0, 0))
    graph.add_node(42, position=(1, 1, 1))
    graph.add_node(23, position=(5, 5, 5), swip="swap")
    graph.add_node(57, position=(7, 7, 7), zap="zip")
    graph.add_edge(42, 23)
    graph.add_edge(57, 23)
    graph.add_edge(2, 42)
    return graph


SWIP_ZAP = dict(node_attrs={"position": Vec(float, 3), "swip": str, "zap": str})


def test_graph_filtering(provider_factory, write_method):
    roi = Roi((0, 0, 0), (10, 10, 10))
    graph = _selected_graph()

    with provider_factory("w", **SELECTED) as graph_writer:
        _write_nodes(graph_writer, graph.nodes(), write_method)
        _write_edges(graph_writer, graph.nodes(), graph.edges(), write_method)

    with provider_factory("r") as graph_reader:
        filtered_nodes = graph_reader.read_nodes(roi, attr_filter={"selected": True})
        filtered_node_ids = [node["id"] for node in filtered_nodes]
        expected_node_ids = [2, 23, 57]
        assert expected_node_ids == filtered_node_ids

        filtered_edges = graph_reader.read_edges(roi, attr_filter={"selected": True})
        filtered_edge_endpoints = [(edge["u"], edge["v"]) for edge in filtered_edges]
        expected_edge_endpoints = [(57, 23), (2, 42)]
        for u, v in expected_edge_endpoints:
            assert (u, v) in filtered_edge_endpoints or (
                v,
                u,
            ) in filtered_edge_endpoints

        filtered_subgraph = graph_reader.read_graph(
            roi, nodes_filter={"selected": True}, edges_filter={"selected": True}
        )
        nodes_with_position = [
            node
            for node, data in filtered_subgraph.nodes(data=True)
            if "position" in data
        ]
        assert expected_node_ids == nodes_with_position
        assert len(filtered_subgraph.edges()) == len(expected_edge_endpoints)
        for u, v in expected_edge_endpoints:
            assert (u, v) in filtered_subgraph.edges() or (
                v,
                u,
            ) in filtered_subgraph.edges()


def test_graph_filtering_complex(provider_factory, write_method):
    roi = Roi((0, 0, 0), (10, 10, 10))
    graph = _selected_graph()

    with provider_factory("w", **SELECTED) as graph_provider:
        _write_nodes(graph_provider, graph.nodes(), write_method)
        _write_edges(graph_provider, graph.nodes(), graph.edges(), write_method)

    with provider_factory("r") as graph_provider:
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
            assert (u, v) in filtered_edge_endpoints or (
                v,
                u,
            ) in filtered_edge_endpoints

        filtered_subgraph = graph_provider.read_graph(
            roi,
            nodes_filter={"selected": True, "test": "test"},
            edges_filter={"selected": True, "a": 100},
        )
        nodes_with_position = [
            node
            for node, data in filtered_subgraph.nodes(data=True)
            if "position" in data
        ]
        assert expected_node_ids == nodes_with_position
        assert len(filtered_subgraph.edges()) == 0


def test_graph_read_and_update_specific_attrs(provider_factory):
    roi = Roi((0, 0, 0), (10, 10, 10))
    graph = _selected_graph()

    with provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3), "selected": bool, "test": str},
        edge_attrs={"selected": bool, "a": int, "b": int, "c": int},
    ) as graph_provider:
        graph_provider.write_graph(graph)

    with provider_factory("r+") as graph_provider:
        limited_graph = graph_provider.read_graph(
            roi, node_attrs=["selected"], edge_attrs=["c"]
        )

        for _node, data in limited_graph.nodes(data=True):
            assert set(data) == {"selected"}
            data["selected"] = True

        for _u, _v, data in limited_graph.edges(data=True):
            assert set(data) == {"c"}
        nx.set_edge_attributes(limited_graph, 5, "c")

        try:
            graph_provider.write_attrs(
                limited_graph, edge_attrs=["c"], node_attrs=["selected"]
            )
        except NotImplementedError:
            pytest.xfail()

        updated_graph = graph_provider.read_graph(roi)

        for _node, data in updated_graph.nodes(data=True):
            assert data["selected"]

        for _u, _v, data in updated_graph.edges(data=True):
            assert data["c"] == 5


def test_graph_update_vector_node_attr(provider_factory):
    # Regression: update_nodes must handle a Vec (multi-column) node attribute,
    # not just scalars. Previously the SQLite backend emitted
    # `UPDATE nodes SET embedding = [1.0, 2.0, 3.0]` (invalid SQL) because it
    # wrote the whole sequence into one column instead of expanding to the
    # per-dimension columns (embedding_0, embedding_1, ...).
    roi = Roi((0, 0, 0), (10, 10, 10))
    graph = nx.Graph()
    graph.add_node(7, position=(5, 5, 5), embedding=[0.0, 0.0, 0.0], score=0.0)

    with provider_factory(
        "w",
        node_attrs={
            "position": Vec(float, 3),
            "embedding": Vec(float, 3),
            "score": float,
        },
    ) as gp:
        gp.write_graph(graph)

    with provider_factory("r+") as gp:
        g = gp.read_graph(roi)
        # numpy values (the typical case, e.g. embeddings computed with numpy)
        # must be normalized to native python on their way into the SQL
        # statement, for arrays and scalars alike
        g.nodes[7]["embedding"] = np.array([1.0, 2.0, 3.0])
        g.nodes[7]["score"] = np.float64(0.5)
        try:
            gp.update_nodes(g.nodes, roi=roi, attributes=["embedding", "score"])
        except NotImplementedError:
            pytest.xfail()

    with provider_factory("r") as gp:
        updated = gp.read_graph(roi)
        assert list(updated.nodes[7]["embedding"]) == [1.0, 2.0, 3.0]
        assert updated.nodes[7]["score"] == 0.5


def test_graph_read_unbounded_roi(provider_factory, write_method):
    unbounded_roi = Roi((None, None, None), (None, None, None))
    graph = _selected_graph()

    with provider_factory("w", **SELECTED) as graph_provider:
        _write_nodes(graph_provider, graph.nodes(), write_method)
        _write_edges(graph_provider, graph.nodes(), graph.edges(), write_method)

    with provider_factory("r+") as graph_provider:
        limited_graph = graph_provider.read_graph(
            unbounded_roi, node_attrs=["selected"], edge_attrs=["c"]
        )

        seen = []
        for node, data in limited_graph.nodes(data=True):
            assert set(data) == {"selected"}
            seen.append(node)

        assert sorted([2, 42, 23, 57]) == sorted(seen)


def test_graph_read_meta_values(provider_factory):
    roi = Roi((0, 0, 0), (10, 10, 10))
    with provider_factory("w", True, roi, node_attrs={"position": Vec(float, 3)}):
        pass
    with provider_factory("r", None, None) as graph_provider:
        assert graph_provider.directed is True
        assert roi == graph_provider.total_roi


def test_graph_default_meta_values(provider_factory):
    with provider_factory(
        "w", False, None, node_attrs={"position": Vec(float, 3)}
    ) as provider:
        assert provider.directed is False
        assert provider.total_roi is None or provider.total_roi == Roi(
            (None, None, None), (None, None, None)
        )
    with provider_factory("r", False, None) as graph_provider:
        assert graph_provider.directed is False
        assert graph_provider.total_roi is None or graph_provider.total_roi == Roi(
            (None, None, None), (None, None, None)
        )


def test_graph_io(provider_factory, write_method):
    graph = _swip_zap_graph()

    with provider_factory("w", **SWIP_ZAP) as graph_provider:
        _write_nodes(graph_provider, graph.nodes(), write_method)
        _write_edges(graph_provider, graph.nodes(), graph.edges(), write_method)

    with provider_factory("r") as graph_provider:
        compare_graph = graph_provider[Roi((1, 1, 1), (9, 9, 9))]

    nodes = sorted(list(graph.nodes()))
    nodes.remove(2)  # node 2 has no position and will not be queried
    compare_nodes = sorted(list(compare_graph.nodes()))

    edges = sorted(tuple(sorted(e)) for e in list(graph.edges()))
    edges.remove((2, 42))  # node 2 has no position and will not be queried
    compare_edges = sorted(tuple(sorted(e)) for e in list(compare_graph.edges()))

    assert nodes == compare_nodes
    assert edges == compare_edges


def test_graph_duplicate_insert_is_ignored_without_fail_if_exists(provider_factory):
    roi = Roi((0, 0, 0), (10, 10, 10))
    graph = nx.Graph()
    graph.add_node(2, position=(2, 2, 2), selected=True)
    graph.add_node(42, position=(1, 1, 1), selected=False)
    graph.add_edge(2, 42, selected=True)

    with provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3), "selected": bool},
        edge_attrs={"selected": bool},
    ) as graph_provider:
        # Initial write
        graph_provider.write_nodes(graph.nodes())
        graph_provider.write_edges(graph.nodes(), graph.edges())

        # fail_if_exists=False should silently ignore duplicates
        graph_provider.write_nodes(graph.nodes(), fail_if_exists=False)
        graph_provider.write_edges(graph.nodes(), graph.edges(), fail_if_exists=False)

    # Verify the original data is still intact
    with provider_factory("r") as graph_provider:
        result = graph_provider.read_graph(roi)
        assert set(result.nodes()) == {2, 42}
        assert [tuple(sorted(edge)) for edge in result.edges()] == [(2, 42)]


def test_graph_write_attributes(provider_factory, write_method):
    graph = nx.Graph()
    graph.add_node(2, position=[0, 0, 0])
    graph.add_node(42, position=[1, 1, 1])
    graph.add_node(23, position=[5, 5, 5], swip="swap")
    graph.add_node(57, position=[7, 7, 7], zap="zip")
    graph.add_edge(42, 23)
    graph.add_edge(57, 23)
    graph.add_edge(2, 42)

    with provider_factory(
        "w",
        node_attrs={"position": Vec(int, 3), "swip": str, "zap": str},
    ) as graph_provider:
        _write_graph(
            graph_provider,
            graph,
            write_method,
            write_nodes=True,
            write_edges=False,
            node_attrs=["position", "swip"],
        )
        _write_edges(graph_provider, graph.nodes(), graph.edges(), write_method)

    with provider_factory("r") as graph_provider:
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
    for n, c in zip(nodes, compare_nodes, strict=False):
        assert n[0] == c[0]
        for key in n[1]:
            assert key in c[1]
            v1 = n[1][key]
            v2 = c[1][key]
            try:
                for e1, e2 in zip(v1, v2, strict=False):
                    assert e1 == e2
            except (TypeError, AssertionError):
                assert v1 == v2


def test_graph_write_roi(provider_factory, write_method):
    graph = _swip_zap_graph()

    write_roi = Roi((0, 0, 0), (6, 6, 6))
    with provider_factory("w", **SWIP_ZAP) as graph_provider:
        _write_graph(graph_provider, graph, write_method, roi=write_roi)

    with provider_factory("r") as graph_provider:
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
    with provider_factory("w", **SWIP_ZAP) as graph_provider:
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
    roi = Roi((0, 0, 0), (10, 10, 10))
    graph = _swip_zap_graph()

    write_roi = Roi((0, 0, 0), (6, 6, 6))
    with provider_factory("w", **SWIP_ZAP) as graph_provider:
        _write_nodes(graph_provider, graph.nodes(), write_method, roi=write_roi)
        _write_edges(
            graph_provider, graph.nodes(), graph.edges(), write_method, roi=write_roi
        )
        assert graph_provider.has_edges(roi)


def test_read_edges_by_roi_matches_read_edges_by_the_nodes_in_it(
    provider_factory, write_method
):
    """`read_edges(roi=...)` (one JOIN) returns the edges that `read_nodes(roi)`
    followed by `read_edges(nodes=...)` (an IN clause) returns."""
    # a path 0 - 1 - 2 - 3 - 4 along x, one node per unit cell
    graph = nx.Graph()
    for i in range(5):
        graph.add_node(i, position=(i + 0.5, 0.5, 0.5))
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    with provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3)},
    ) as graph_provider:
        _write_graph(graph_provider, graph, write_method)

    with provider_factory("r") as graph_provider:
        query_roi = Roi((1, 0, 0), (2, 1, 1))  # holds nodes 1 and 2
        nodes = graph_provider.read_nodes(query_roi)
        edges_via_in = graph_provider.read_edges(nodes=nodes)
        edges_via_join = graph_provider.read_edges(roi=query_roi)

    # an edge is stored once, as (min, max), and found by its first endpoint
    assert sorted((e["u"], e["v"]) for e in edges_via_join) == [(1, 2), (2, 3)]
    assert sorted((e["u"], e["v"]) for e in edges_via_in) == [(1, 2), (2, 3)]


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

    fetch_on_v=False (default): only edges where u is in ROI -> {(1,5), (2,8), (5,8)}
    fetch_on_v=True: edges where u OR v is in ROI -> {(1,5), (2,8), (5,8)}
        (same here because u < v and all boundary-crossing edges have u inside)

    To properly test fetch_on_v, we need an edge where u is OUTSIDE the ROI
    but v is INSIDE. With undirected u < v storage, this means a node with a
    smaller ID outside the ROI connected to a node with a larger ID inside.

    So we add: Node 0 (pos 8) -- Edge(0, 5): u=0 outside ROI, v=5 in ROI.
    """
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

    with provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3)},
    ) as graph_provider:
        _write_graph(graph_provider, graph, write_method)

    with provider_factory("r") as graph_provider:

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


def _single_column_indices(provider, table):
    """Return the set of columns that have a single-column index on ``table``."""
    from funlib.persistence.graphs import SQLiteGraphDataBase

    cur = provider.cur
    columns = set()
    if isinstance(provider, SQLiteGraphDataBase):
        cur.execute(f"PRAGMA index_list({table})")
        for index in [row[1] for row in cur.fetchall()]:
            cur.execute(f"PRAGMA index_info({index})")
            info = cur.fetchall()
            if len(info) == 1:
                columns.add(info[0][2])
    else:
        cur.execute(
            "SELECT a.attname FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid "
            "AND a.attnum = i.indkey[0] "
            "WHERE i.indrelid = %s::regclass AND i.indnatts = 1",
            (table,),
        )
        columns = {row[0] for row in cur.fetchall()}
    return columns


def test_edges_indexed_on_both_endpoints(provider_factory):
    """read_edges(fetch_on_v=True) matches on v as well as u, so v needs an
    index of its own: the (u, v) primary key only serves lookups on u."""
    graph = nx.Graph()
    graph.add_node(1, position=(1.0, 1.0, 1.0))
    graph.add_node(2, position=(2.0, 2.0, 2.0))
    graph.add_edge(1, 2)

    with provider_factory(
        "w", node_attrs={"position": Vec(float, 3)}
    ) as graph_provider:
        assert "v" in _single_column_indices(graph_provider, "edges")

        # bulk_write_mode may drop indices while writing; they must come back
        with graph_provider.bulk_write_mode():
            graph_provider.bulk_write_graph(graph)
        assert "v" in _single_column_indices(graph_provider, "edges")


def test_graph_roi_upper_bound_exclusive(provider_factory):
    """A ROI is half-open, [begin, end): a node on the lower bound is read, a node
    whose position equals the end in any dimension is not."""
    roi = Roi((0, 0, 0), (10, 10, 10))

    graph = nx.Graph()
    graph.add_node(1, position=(5.0, 5.0, 5.0))  # interior
    graph.add_node(2, position=(0.0, 0.0, 0.0))  # on the lower bound
    graph.add_node(3, position=(10.0, 5.0, 5.0))  # x == end
    graph.add_node(4, position=(5.0, 10.0, 5.0))  # y == end
    graph.add_node(5, position=(5.0, 5.0, 10.0))  # z == end
    graph.add_node(6, position=(10.0, 10.0, 10.0))  # all dims == end
    graph.add_edge(1, 3)  # u inside, v on the upper bound

    with provider_factory(
        "w",
        node_attrs={"position": Vec(float, 3)},
    ) as gp:
        gp.write_graph(graph)

    with provider_factory("r") as gp:
        assert {n["id"] for n in gp.read_nodes(roi)} == {1, 2}

        # read_graph: the same nodes, plus node 3 as a bare node (no position)
        # because the edge (1, 3) starts inside the ROI
        result = gp.read_graph(roi)
        assert {node: set(data) for node, data in result.nodes(data=True)} == {
            1: {"position"},
            2: {"position"},
            3: set(),
        }
        assert sorted(tuple(sorted(e)) for e in result.edges()) == [(1, 3)]


def test_graph_mode_w_starts_from_an_empty_database(provider_factory, write_method):
    graph = nx.Graph()
    graph.add_node(1, position=(0.5, 0.5, 0.5))
    graph.add_node(2, position=(1.5, 0.5, 0.5))
    graph.add_edge(1, 2)
    replacement = nx.Graph()
    replacement.add_node(3, position=(2.5, 0.5, 0.5))

    with provider_factory("w", node_attrs={"position": Vec(float, 3)}) as provider:
        _write_graph(provider, graph, write_method)
    with provider_factory("w", node_attrs={"position": Vec(float, 3)}) as provider:
        _write_graph(provider, replacement, write_method)

    with provider_factory("r") as provider:
        result = provider.read_graph()
        assert sorted(result.nodes()) == [3]
        assert list(result.edges()) == []


def _path_graph(graph_type=nx.Graph):
    """Nodes 1 - 2 - 3 along x, one per unit cell, with a score of 0 everywhere."""
    graph = graph_type()
    for node in (1, 2, 3):
        graph.add_node(node, position=(node - 0.5, 0.5, 0.5), score=0.0)
    graph.add_edge(1, 2, score=0.0)
    graph.add_edge(2, 3, score=0.0)
    return graph


SCORED = dict(
    node_attrs={"position": Vec(float, 3), "score": float},
    edge_attrs={"score": float},
)


def test_directed_graph_keeps_edge_direction(provider_factory, write_method):
    graph = nx.DiGraph()
    graph.add_node(1, position=(0.5, 0.5, 0.5))
    graph.add_node(2, position=(1.5, 0.5, 0.5))
    graph.add_edge(2, 1)

    with provider_factory(
        "w", directed=True, node_attrs={"position": Vec(float, 3)}
    ) as provider:
        _write_graph(provider, graph, write_method)

    with provider_factory("r") as provider:
        result = provider.read_graph()
        assert result.is_directed() is True
        assert list(result.edges()) == [(2, 1)]


def test_num_nodes_counts_the_nodes_in_the_roi(provider_factory):
    with provider_factory("w", **SCORED) as provider:
        provider.write_graph(_path_graph())
        assert provider.num_nodes(Roi((0, 0, 0), (2, 1, 1))) == 2


def test_read_edges_of_no_nodes_is_empty(provider_factory):
    with provider_factory("w", **SCORED) as provider:
        provider.write_graph(_path_graph())
        assert provider.read_edges(nodes=[]) == []


def test_read_edges_refuses_a_roi_together_with_nodes(provider_factory):
    with provider_factory("w", **SCORED) as provider:
        provider.write_graph(_path_graph())
        with pytest.raises(ValueError, match="both roi and nodes"):
            provider.read_edges(roi=Roi((0, 0, 0), (1, 1, 1)), nodes=[])


def test_write_roi_limits_bulk_writes(provider_factory):
    """Nodes are written where they lie in the ROI, edges where their first endpoint
    does."""
    with provider_factory("w", **SCORED) as provider:
        provider.bulk_write_graph(_path_graph(), roi=Roi((0, 0, 0), (1, 1, 1)))
        assert [node["id"] for node in provider.read_nodes()] == [1]
        assert [(edge["u"], edge["v"]) for edge in provider.read_edges()] == [(1, 2)]


def test_write_attrs_roi_limits_the_update(provider_factory):
    graph = _path_graph()
    with provider_factory("w", **SCORED) as provider:
        provider.write_graph(graph)
        nx.set_node_attributes(graph, 1.0, "score")
        nx.set_edge_attributes(graph, 1.0, "score")
        provider.write_attrs(
            graph,
            roi=Roi((0, 0, 0), (1, 1, 1)),
            node_attrs=["score"],
            edge_attrs=["score"],
        )

        result = provider.read_graph()
        assert sorted(result.nodes(data="score")) == [(1, 1.0), (2, 0.0), (3, 0.0)]
        assert sorted(
            (min(u, v), max(u, v), score) for u, v, score in result.edges(data="score")
        ) == [(1, 2, 1.0), (2, 3, 0.0)]


@pytest.mark.parametrize(
    "write",
    [
        lambda provider, graph: provider.write_nodes(graph.nodes),
        lambda provider, graph: provider.write_edges(graph.nodes, graph.edges),
        lambda provider, graph: provider.bulk_write_nodes(graph.nodes),
        lambda provider, graph: provider.bulk_write_edges(graph.nodes, graph.edges),
        lambda provider, graph: provider.update_nodes(graph.nodes),
        lambda provider, graph: provider.update_edges(graph.nodes, graph.edges),
    ],
    ids=[
        "write_nodes",
        "write_edges",
        "bulk_write_nodes",
        "bulk_write_edges",
        "update_nodes",
        "update_edges",
    ],
)
def test_mode_r_refuses_writes(provider_factory, write):
    graph = _path_graph()
    with provider_factory("w", **SCORED) as provider:
        provider.write_graph(graph)
    with provider_factory("r") as provider:
        with pytest.raises(RuntimeError, match="Trying to write to read-only DB"):
            write(provider, graph)


@pytest.mark.parametrize(
    "write",
    [
        lambda provider, graph: provider.write_nodes(graph.nodes, delete=True),
        lambda provider, graph: provider.write_edges(
            graph.nodes, graph.edges, delete=True
        ),
    ],
    ids=["write_nodes", "write_edges"],
)
def test_delete_before_writing_is_not_implemented(provider_factory, write):
    with provider_factory("w", **SCORED) as provider:
        with pytest.raises(NotImplementedError, match="Delete not implemented"):
            write(provider, _path_graph())


def test_sqlite_mode_r_refuses_a_file_that_holds_no_graph(tmp_path):
    with pytest.raises(RuntimeError, match="metadata does not exist"):
        SQLiteGraphDataBase(
            tmp_path / "empty.db", position_attribute="position", mode="r"
        )


@pytest.mark.parametrize(
    "parts, nodes, edges",
    [
        (dict(write_edges=False), [1, 2, 3], []),
        (dict(write_nodes=False), [], [(1, 2), (2, 3)]),
    ],
    ids=["nodes-only", "edges-only"],
)
def test_write_graph_writes_the_parts_asked_for(
    provider_factory, write_method, parts, nodes, edges
):
    with provider_factory("w", **SCORED) as provider:
        _write_graph(provider, _path_graph(), write_method, **parts)
        assert [node["id"] for node in provider.read_nodes()] == nodes
        assert [(edge["u"], edge["v"]) for edge in provider.read_edges()] == edges


def test_read_graph_without_edges(provider_factory):
    with provider_factory("w", **SCORED) as provider:
        provider.write_graph(_path_graph())
        result = provider.read_graph(read_edges=False)
        assert sorted(result.nodes()) == [1, 2, 3]
        assert list(result.edges()) == []
