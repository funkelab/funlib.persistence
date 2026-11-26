"""The CSR graph database against a brute-force reference."""

import multiprocessing as mp

import networkx as nx
import numpy as np
import pytest
from funlib.geometry import Roi

from funlib.persistence.graphs import CSRGraphDataBase, SQLiteGraphDataBase
from funlib.persistence.graphs.csr_graph_database import UnfinalizedWrites
from funlib.persistence.graphs.graph_database import AttributeType
from funlib.persistence.types import Vec

NODE_ATTRS: dict[str, AttributeType] = {"position": Vec(float, 3), "size": int}
EDGE_ATTRS: dict[str, AttributeType] = {"adj": "float32", "lr": "float32"}


def random_graph(n=3000, m=12000, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.choice(10 * n, size=n, replace=False).astype(np.uint64) + 1
    pos = rng.uniform(-50, 350, size=(n, 3))
    size = rng.integers(1, 100, size=n)
    a, b = rng.integers(n, size=(2, m))
    keep = a != b
    u, v = (
        np.minimum(ids[a[keep]], ids[b[keep]]),
        np.maximum(ids[a[keep]], ids[b[keep]]),
    )
    pairs = np.unique(np.stack([u, v], axis=1), axis=0)
    u, v = pairs[:, 0], pairs[:, 1]
    adj = rng.random(len(u)).astype(np.float32)
    lr = rng.random(len(u)).astype(np.float32)
    lr[::3] = np.nan
    return ids, pos, size, u, v, adj, lr


def create(path, cell_size=40.0):
    return CSRGraphDataBase(
        path,
        mode="w",
        node_attrs=NODE_ATTRS,
        edge_attrs=EDGE_ATTRS,
        cell_size=cell_size,
    )


@pytest.fixture
def filled(tmp_path):
    graph = random_graph()
    ids, pos, size, u, v, adj, lr = graph
    db = create(tmp_path / "g.csr")
    with db.bulk_write_mode():
        half = len(ids) // 2
        db.write_node_arrays(ids[:half], position=pos[:half], size=size[:half])
        db.write_node_arrays(ids[half:], position=pos[half:], size=size[half:])
        db.write_edge_arrays(v, u, adj=adj, lr=lr)  # endpoint order is free
    return db, graph


def reference(graph, lo, hi, steps):
    ids, pos, _, u, v, _, _ = graph
    inside = ((pos >= lo) & (pos < hi)).all(axis=1)
    step = {int(i): 0 for i in ids[inside]}
    g = nx.Graph()
    g.add_nodes_from(ids.tolist())
    g.add_edges_from(zip(u.tolist(), v.tolist(), strict=True))
    frontier = list(step)
    for k in range(1, steps + 1):
        reached = {n for f in frontier for n in g[f] if n not in step}
        for n in reached:
            step[n] = k
        frontier = list(reached)
    edges = {
        (a, b)
        for a, b in zip(u.tolist(), v.tolist(), strict=True)
        if a in step and b in step
    }
    return step, edges


@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize(
    "box",
    [
        ((0, 0, 0), (100, 120, 90)),
        ((-60, -60, -60), (400, 400, 400)),
        ((10, 10, 10), (11, 11, 11)),
    ],
)
def test_read_arrays_matches_reference(filled, box, steps):
    db, graph = filled
    lo, hi = np.array(box[0]), np.array(box[1])
    nodes, edges = db.read_arrays(Roi(lo, hi - lo), num_steps=steps)
    step, want_edges = reference(graph, lo, hi, steps)
    assert dict(zip(nodes["id"].tolist(), nodes["step"].tolist(), strict=True)) == step
    assert set(zip(edges["u"].tolist(), edges["v"].tolist(), strict=True)) == want_edges
    ids, pos, size, u, v, adj, lr = graph
    at = {int(i): k for k, i in enumerate(ids)}
    rows = [at[i] for i in nodes["id"].tolist()]
    np.testing.assert_array_equal(nodes["position"], pos[rows])
    np.testing.assert_array_equal(nodes["size"], size[rows])
    pair = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(u, v, strict=True))}
    which = [
        pair[e] for e in zip(edges["u"].tolist(), edges["v"].tolist(), strict=True)
    ]
    np.testing.assert_array_equal(edges["adj"], adj[which])
    np.testing.assert_array_equal(edges["lr"], lr[which])


def test_whole_graph_and_counts(filled):
    db, graph = filled
    nodes, edges = db.read_arrays()
    assert len(nodes["id"]) == db.num_nodes == len(graph[0])
    assert len(edges["u"]) == db.num_edges == len(graph[3])
    reopened = CSRGraphDataBase(db.path, mode="r")
    assert reopened.num_edges == db.num_edges
    assert set(reopened.node_attrs) == {"position", "size"}


def test_matches_sqlite_read_graph(filled, tmp_path):
    db, graph = filled
    ids, pos, size, u, v, adj, lr = graph
    sql = SQLiteGraphDataBase(
        tmp_path / "g.db",
        position_attribute="position",
        mode="w",
        node_attrs=NODE_ATTRS,
        edge_attrs={"adj": float, "lr": float},
    )
    g = nx.Graph()
    for i, p, s in zip(ids.tolist(), pos.tolist(), size.tolist(), strict=True):
        g.add_node(i, position=tuple(p), size=s)
    for a, b, w in zip(u.tolist(), v.tolist(), adj.tolist(), strict=True):
        g.add_edge(a, b, adj=w)
    sql.write_graph(g)
    roi = Roi((0, 20, 40), (150, 100, 200))
    for fetch_on_v in (False, True):
        want = sql.read_graph(roi, fetch_on_v=fetch_on_v)
        got = db.read_graph(roi, fetch_on_v=fetch_on_v)
        assert set(got.nodes) == set(want.nodes)
        assert {frozenset(e) for e in got.edges} == {frozenset(e) for e in want.edges}
        inside = [n for n, d in want.nodes(data=True) if "position" in d]
        assert all(got.nodes[n]["size"] == want.nodes[n]["size"] for n in inside)
    filtered = db.read_graph(roi, nodes_filter={"size": int(size[0])})
    assert all(
        d["size"] == size[0] for _, d in filtered.nodes(data=True) if "size" in d
    )


def test_networkx_writes_roi_and_rewrites(tmp_path):
    db = create(tmp_path / "g.csr", cell_size=5.0)
    g = nx.Graph()
    g.add_node(1, position=(1.0, 1.0, 1.0), size=3)
    g.add_node(2, position=(2.0, 2.0, 2.0), size=4)
    g.add_node(9, position=(50.0, 50.0, 50.0), size=5)
    g.add_edge(2, 1, adj=0.5)
    g.add_edge(2, 9, adj=0.25)
    g.add_edge(9, 77, adj=0.1)  # 77 is not a node
    db.write_graph(g, Roi((0, 0, 0), (10, 10, 10)))
    assert db.num_nodes == 2 and db.num_edges == 1  # 9 is outside; (2, 9) lost its end
    db.write_graph(g)
    assert db.num_nodes == 3 and db.num_edges == 2
    assert db.meta["dropped_edges"] == 1
    # a second pass adds an attribute to an existing edge and changes a node
    h = nx.Graph()
    h.add_node(1, position=(1.0, 1.0, 1.0), size=30)
    h.add_node(2, position=(2.0, 2.0, 2.0))
    h.add_edge(1, 2, lr=0.75)
    db.write_graph(h)
    out = db.read_graph()
    assert out.nodes[1]["size"] == 30 and out.nodes[2]["size"] == 4
    assert out.edges[1, 2] == {"adj": 0.5, "lr": 0.75}
    assert out.edges[2, 9]["lr"] is None
    db.drop_edges()
    assert db.num_edges == 0 and db.num_nodes == 3
    with pytest.raises(NotImplementedError):
        db.write_graph(g, fail_if_exists=True)


def _worker(path, part):
    ids, pos, size, u, v, adj, lr = random_graph()
    db = CSRGraphDataBase(path, mode="r+")
    with db.bulk_write_mode(worker=True):
        rows = np.arange(len(ids)) % 4 == part
        db.write_node_arrays(ids[rows], position=pos[rows], size=size[rows])
        rows = np.arange(len(u)) % 4 == part
        db.write_edge_arrays(u[rows], v[rows], adj=adj[rows], lr=lr[rows])


def test_workers_write_then_main_finalizes(tmp_path):
    path = tmp_path / "g.csr"
    db = create(path)
    with db.bulk_write_mode():
        procs = [
            mp.get_context("spawn").Process(target=_worker, args=(path, k))
            for k in range(4)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        assert all(p.exitcode == 0 for p in procs)
        with pytest.raises(UnfinalizedWrites):
            CSRGraphDataBase(path, mode="r").read_arrays()
    graph = random_graph()
    assert db.num_nodes == len(graph[0]) and db.num_edges == len(graph[3])
    assert not list((path / "segments").glob("*.npys"))


def test_open_errors(tmp_path):
    with pytest.raises(RuntimeError, match="metadata does not exist"):
        CSRGraphDataBase(tmp_path / "missing", mode="r")
    with pytest.raises(ValueError, match="cell_size"):
        CSRGraphDataBase(tmp_path / "x", mode="w", node_attrs=NODE_ATTRS)
    db = create(tmp_path / "g.csr")
    assert db.read_graph().number_of_nodes() == 0
    with pytest.raises(RuntimeError, match="read-only"):
        CSRGraphDataBase(db.path, mode="r").write_node_arrays(
            np.array([1]), position=np.zeros((1, 3))
        )


def _finalizer(path, part):
    ids, pos, size, u, v, adj, lr = random_graph()
    db = CSRGraphDataBase(path, mode="r+")
    rows = np.arange(len(ids)) % 4 == part
    # outside bulk_write_mode: every write finalizes, here from four processes
    db.write_node_arrays(ids[rows], position=pos[rows], size=size[rows])


def test_processes_that_each_finalize_do_not_collide(tmp_path):
    path = tmp_path / "g.csr"
    create(path)
    procs = [
        mp.get_context("spawn").Process(target=_finalizer, args=(path, k))
        for k in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert all(p.exitcode == 0 for p in procs)
    assert CSRGraphDataBase(path, mode="r").num_nodes == len(random_graph()[0])


@pytest.mark.parametrize("steps", [0, 2])
def test_a_region_of_empty_cells_inside_the_graph_holds_nothing(tmp_path, steps):
    """Two clusters of nodes with empty space between them: a read of the
    space finds the cells of neither, and returns no node and no edge."""
    rng = np.random.default_rng(0)
    pos = np.concatenate([rng.uniform(0, 30, (20, 3)), rng.uniform(970, 1000, (20, 3))])
    ids = np.arange(1, 41, dtype=np.uint64)
    db = create(tmp_path / "g.csr")
    with db.bulk_write_mode():
        db.write_node_arrays(ids, position=pos, size=np.ones(40, dtype=np.int64))
        db.write_edge_arrays(
            ids[:-1], ids[1:], adj=np.ones(39, np.float32), lr=np.ones(39, np.float32)
        )
    nodes, edges = db.read_arrays(Roi((400, 400, 400), (100, 100, 100)), steps)
    assert len(nodes["id"]) == 0
    assert nodes["position"].shape == (0, 3)
    assert len(edges["u"]) == 0


def test_from_arrays_writes_a_whole_graph_in_the_dtypes_of_its_arrays(tmp_path):
    ids, pos, size, u, v, adj, lr = random_graph()
    vectors = np.arange(len(ids) * 2, dtype=np.float32).reshape(-1, 2)
    db = CSRGraphDataBase.from_arrays(
        tmp_path / "g.csr",
        ids,
        {"position": pos, "size": size, "vec": vectors},
        u,
        v,
        {"adj": adj, "lr": lr},
        cell_size=40.0,
    )
    assert (db.num_nodes, db.num_edges) == (len(ids), len(u))
    assert db.node_attrs == {
        "position": Vec("float64", 3),
        "size": "int64",
        "vec": Vec("float32", 2),
    }
    assert db.edge_attrs == {"adj": "float32", "lr": "float32"}
    nodes, edges = CSRGraphDataBase(tmp_path / "g.csr").read_arrays()
    order = np.argsort(ids)
    back = np.argsort(nodes["id"])
    assert nodes["id"][back].tolist() == ids[order].tolist()
    assert nodes["vec"][back].tolist() == vectors[order].tolist()
    assert sorted(zip(edges["u"].tolist(), edges["v"].tolist(), strict=True)) == sorted(
        zip(u.tolist(), v.tolist(), strict=True)
    )


def test_from_arrays_stores_an_attribute_in_the_type_named_for_it(tmp_path):
    ids, pos, size, u, v, adj, _ = random_graph(n=50, m=100)
    db = CSRGraphDataBase.from_arrays(
        tmp_path / "g.csr",
        ids,
        {"position": pos, "size": size},
        u,
        v,
        {"adj": adj.astype(np.float64)},
        cell_size=40.0,
        node_types={"size": "int32"},
        edge_types={"adj": "float32"},
    )
    assert db.node_attrs["size"] == "int32" and db.edge_attrs == {"adj": "float32"}
    # written again, it replaces what was there
    again = CSRGraphDataBase.from_arrays(
        tmp_path / "g.csr", ids[:5], {"position": pos[:5]}, cell_size=40.0
    )
    assert (again.num_nodes, again.num_edges) == (5, 0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"u": np.array([1], dtype=np.uint64)}, "both u and v"),
        ({"edge_attrs": {"adj": np.zeros(1)}}, "without edges"),
        ({"node_types": {"nope": "int32"}}, r"types were given for \['nope'\]"),
    ],
)
def test_from_arrays_refuses_arguments_that_do_not_fit(tmp_path, kwargs, match):
    ids, pos, *_ = random_graph(n=10, m=10)
    with pytest.raises(ValueError, match=match):
        CSRGraphDataBase.from_arrays(
            tmp_path / "g.csr", ids, {"position": pos}, cell_size=40.0, **kwargs
        )


def one_per(n, dtype=np.float32):
    return np.ones(n, dtype=dtype)


@pytest.mark.parametrize(
    "change, error, match",
    [
        # a text attribute cannot be stored (was a TypeError from numpy)
        (
            {"node_attrs": {"name": np.array(["a"] * 10)}},
            NotImplementedError,
            "node attribute 'name'.*not supported by the CSR backend",
        ),
        ({"node_types": {"size": str}}, NotImplementedError, "not supported"),
        # an array that is not one entry per node or edge is not spread
        (
            {"node_attrs": {"size": np.array([9])}},
            ValueError,
            r"node attribute 'size' has shape \(1,\).*each of the 10 nodes",
        ),
        (
            {"node_attrs": {"size": np.arange(3)}},
            ValueError,
            r"node attribute 'size' has shape \(3,\)",
        ),
        (
            {"edge_attrs": {"adj": one_per(1)}},
            ValueError,
            r"edge attribute 'adj' has shape \(1,\).*each of the 9 edges",
        ),
        ({"node_attrs": {"size": np.int64(9)}}, ValueError, r"has shape \(\)"),
        ({"v": np.arange(2, 5)}, ValueError, "u and v have shapes"),
        ({"ids": np.ones((10, 1))}, ValueError, "ids has shape"),
        (
            {"node_types": {"size": Vec(int, 2)}},
            ValueError,
            "does not fit its type",
        ),
        # one that only fails while the arrays are built
        ({"cell_size": 1e-6}, ValueError, "more grid cells than a Morton code"),
    ],
)
def test_from_arrays_that_fails_keeps_the_database_that_was_there(
    tmp_path, change, error, match
):
    path = tmp_path / "g.csr"
    ids = np.arange(1, 11, dtype=np.uint64)
    pos = np.stack([np.zeros(10), np.zeros(10), np.arange(10) * 10.0], axis=1)
    given = {
        "ids": ids,
        "node_attrs": {"position": pos, "size": np.arange(10)},
        "u": ids[:-1],
        "v": ids[1:],
        "edge_attrs": {"adj": one_per(9)},
        "cell_size": 10.0,
    }
    CSRGraphDataBase.from_arrays(path, **given)
    for name, value in change.items():
        if name in ("node_attrs", "edge_attrs"):
            given[name] = {**given[name], **value}
        else:
            given[name] = value
    with pytest.raises(error, match=match):
        CSRGraphDataBase.from_arrays(path, **given)
    # the earlier graph is whole, and nothing is left beside it
    before = CSRGraphDataBase(path)
    assert (before.num_nodes, before.num_edges) == (10, 9)
    nodes, edges = before.read_arrays()
    assert sorted(nodes["size"].tolist()) == list(range(10))
    assert edges["adj"].tolist() == [1.0] * 9
    assert [p.name for p in tmp_path.iterdir()] == ["g.csr"]


def test_from_arrays_replaces_a_database_only_when_the_new_one_is_complete(tmp_path):
    path = tmp_path / "deep" / "g.csr"
    ids, pos, size, u, v, adj, _ = random_graph(n=50, m=100)
    first = CSRGraphDataBase.from_arrays(
        path, ids, {"position": pos, "size": size}, u, v, {"adj": adj}, cell_size=40.0
    )
    held = first.read_arrays()[0]["id"]
    second = CSRGraphDataBase.from_arrays(
        path, ids[:5], {"position": pos[:5]}, cell_size=40.0
    )
    assert second.path == path and second.mode == "r+"
    assert (second.num_nodes, second.num_edges) == (5, 0)
    assert second.node_attrs == {"position": Vec("float64", 3)}
    assert len(held) == 50  # a read taken before the replacement stays valid
    # the handle returned accepts further writes
    second.write_node_arrays(ids[5:6], position=pos[5:6])
    assert CSRGraphDataBase(path).num_nodes == 6
    assert [p.name for p in path.parent.iterdir()] == ["g.csr"]


def test_a_database_is_not_removed_by_a_constructor_that_refuses_its_types(tmp_path):
    db = chain(tmp_path)
    with pytest.raises(NotImplementedError, match="not supported by the CSR backend"):
        CSRGraphDataBase(
            db.path,
            mode="w",
            node_attrs={"position": Vec(float, 3), "name": str},
            cell_size=10.0,
        )
    assert CSRGraphDataBase(db.path).num_nodes == 6


def chain(tmp_path):
    """Nodes 1..6 on a line, 10 apart; 1-2-3-4-5-6 joined by ``adj``, and a
    long edge 1-6 that carries only ``lr``."""
    ids = np.arange(1, 7, dtype=np.uint64)
    pos = np.stack([np.zeros(6), np.zeros(6), np.arange(6) * 10.0 + 5], axis=1)
    u = np.array([1, 2, 3, 4, 5, 1], dtype=np.uint64)
    v = np.array([2, 3, 4, 5, 6, 6], dtype=np.uint64)
    adj = np.array([0.1, 0.2, 0.3, 0.4, 0.5, np.nan], dtype=np.float32)
    lr = np.array([np.nan] * 5 + [0.9], dtype=np.float32)
    return CSRGraphDataBase.from_arrays(
        tmp_path / "chain.csr",
        ids,
        {"position": pos},
        u,
        v,
        {"adj": adj, "lr": lr},
        cell_size=10.0,
    )


def test_steps_follow_only_the_edges_named(tmp_path):
    db = chain(tmp_path)
    first = Roi((-5, -5, 0), (10, 10, 10))  # node 1 alone
    every, _ = db.read_arrays(first, num_steps=1)
    assert sorted(every["id"].tolist()) == [1, 2, 6]
    adj, edges = db.read_arrays(first, num_steps=1, step_edges=["adj"])
    assert sorted(adj["id"].tolist()) == [1, 2]
    lr, _ = db.read_arrays(first, num_steps=1, step_edges=["lr"])
    assert sorted(lr["id"].tolist()) == [1, 6]
    either, edges = db.read_arrays(first, num_steps=1, step_edges=["adj", "lr"])
    assert sorted(either["id"].tolist()) == [1, 2, 6]
    # the edges among the nodes read are returned, followed or not
    two, edges = db.read_arrays(first, num_steps=5, step_edges=["adj"])
    assert len(two["id"]) == 6 and len(edges["u"]) == 6
    with pytest.raises(ValueError, match="unknown edge attribute 'nope'"):
        db.read_arrays(first, num_steps=1, step_edges=["nope"])
    # no attribute named: no edge is a step (None is what follows every edge)
    none, edges = db.read_arrays(first, num_steps=3, step_edges=[])
    assert none["id"].tolist() == [1] and len(edges["u"]) == 0
    none, _ = db.read_block(first, num_steps=3, step_edges=())
    assert none["id"].tolist() == [1]


def test_steps_cannot_be_named_by_an_attribute_that_is_never_unset(tmp_path):
    # an integer reads 0 where it was never written, which is a value like
    # any other: following "the edges that have it" would follow every edge
    ids = np.arange(1, 4, dtype=np.uint64)
    pos = np.stack([np.zeros(3), np.zeros(3), np.arange(3) * 10.0 + 5], axis=1)
    db = CSRGraphDataBase.from_arrays(
        tmp_path / "g.csr",
        ids,
        {"position": pos},
        ids[:-1],
        ids[1:],
        {"kind": np.array([1, 0], dtype=np.int8), "flag": np.array([True, False])},
        cell_size=10.0,
    )
    first = Roi((-5, -5, 0), (10, 10, 10))
    for name, dtype in (("kind", "int8"), ("flag", "bool")):
        with pytest.raises(ValueError, match=f"'{name}' is {dtype}; only a float"):
            db.read_arrays(first, num_steps=1, step_edges=[name])
        with pytest.raises(ValueError, match="only a float"):
            db.read_block(first, num_steps=1, step_edges=[name])


def test_read_block_marks_the_nodes_of_the_block_among_its_surroundings(tmp_path):
    db = chain(tmp_path)
    block = Roi((-5, -5, 20), (10, 10, 20))  # nodes 3 and 4

    def own(nodes):
        return sorted(nodes["id"][nodes["own"]].tolist())

    alone, edges = db.read_block(block)
    assert sorted(alone["id"].tolist()) == own(alone) == [3, 4]
    assert len(edges["u"]) == 1

    near, edges = db.read_block(block, context=[0, 0, 10], node_attrs=[])
    assert sorted(near["id"].tolist()) == [2, 3, 4, 5] and own(near) == [3, 4]
    assert "position" not in near and len(edges["u"]) == 3

    stepped, _ = db.read_block(block, num_steps=1, step_edges=["adj"])
    assert sorted(stepped["id"].tolist()) == [2, 3, 4, 5] and own(stepped) == [3, 4]

    # steps start from the nodes the context reached
    both, _ = db.read_block(block, context=10.0, num_steps=1)
    assert sorted(both["id"].tolist()) == [1, 2, 3, 4, 5, 6] and own(both) == [3, 4]
    assert both["step"][both["id"] == 2].tolist() == [0]

    with pytest.raises(ValueError, match="num_steps must be >= 0"):
        db.read_block(block, num_steps=-1)
    # a block whose surroundings hold no node, with a context or without
    far = Roi((500, 500, 500), (10, 10, 10))
    for context in (None, 0, 5, [1, 2, 3]):
        nodes, edges = db.read_block(far, context=context, num_steps=1)
        assert nodes["position"].shape == (0, 3) and nodes["own"].shape == (0,)
        assert nodes["own"].dtype == bool and len(edges["u"]) == 0
    for negative in (-1, -0.5, [0, 0, -0.25], float("nan")):
        with pytest.raises(ValueError, match="context must be >= 0"):
            db.read_block(block, context=negative)


def test_a_database_is_told_from_other_directories(tmp_path):
    db = chain(tmp_path)
    assert CSRGraphDataBase.is_database(db.path)
    assert not CSRGraphDataBase.is_database(tmp_path)
    assert not CSRGraphDataBase.is_database(tmp_path / "missing")
    other = tmp_path / "other"
    other.mkdir()
    (other / "meta.json").write_text('{"version": 1}')
    assert not CSRGraphDataBase.is_database(other)
    (other / "meta.json").write_text("not json")
    assert not CSRGraphDataBase.is_database(other)


def pairs(edges):
    return sorted(zip(edges["u"].tolist(), edges["v"].tolist(), strict=True))


@pytest.mark.parametrize(
    "edge_ends, fetch_on_v", [("u", False), ("either", True)], ids=["u", "either"]
)
def test_an_array_read_returns_the_edges_read_graph_returns_from_sqlite(
    tmp_path, edge_ends, fetch_on_v
):
    """The scenario of volara#111: nodes 1 and 2 inside the roi, node 3
    outside, edges (1, 3) and (2, 3)."""
    ids = np.array([1, 2, 3], dtype=np.uint64)
    pos = np.array([[10.0, 10, 10], [30, 30, 30], [90, 90, 90]])
    u, v = np.array([1, 2], dtype=np.uint64), np.array([3, 3], dtype=np.uint64)
    score = np.array([0.5, 0.75], dtype=np.float32)
    roi = Roi((0, 0, 0), (50, 50, 50))

    sqlite = SQLiteGraphDataBase(
        tmp_path / "g.db",
        position_attribute="position",
        mode="w",
        node_attrs={"position": Vec(float, 3)},
        edge_attrs={"score": float},
    )
    graph = nx.Graph()
    graph.add_nodes_from(
        (int(i), {"position": tuple(p)}) for i, p in zip(ids, pos.tolist(), strict=True)
    )
    graph.add_edges_from(
        (int(a), int(b), {"score": float(s)})
        for a, b, s in zip(u, v, score, strict=True)
    )
    sqlite.write_graph(graph)
    csr = CSRGraphDataBase.from_arrays(
        tmp_path / "g.csr", ids, {"position": pos}, u, v, {"score": score}, cell_size=20
    )

    def listed(db, region):
        graph = db.read_graph(region, fetch_on_v=fetch_on_v)
        return sorted((min(e), max(e)) for e in graph.edges)

    expected = listed(sqlite, roi)
    assert expected == [(1, 3), (2, 3)]
    assert listed(csr, roi) == expected

    nodes, edges = csr.read_arrays(roi, edge_ends=edge_ends)
    assert sorted(nodes["id"].tolist()) == [1, 2]  # node 3 is named, not read
    assert pairs(edges) == expected
    assert sorted(edges["score"].tolist()) == [0.5, 0.75]
    nodes, edges = csr.read_block(roi, edge_ends=edge_ends)
    assert pairs(edges) == expected and nodes["own"].all()

    # the default is unchanged: only the edges among the nodes read
    assert pairs(csr.read_arrays(roi)[1]) == pairs(csr.read_block(roi)[1]) == []

    # from the other side, node 3 is the larger end of both edges
    outside = Roi((50, 50, 50), (50, 50, 50))
    expected = [(1, 3), (2, 3)] if fetch_on_v else []
    assert listed(sqlite, outside) == listed(csr, outside) == expected
    assert pairs(csr.read_arrays(outside, edge_ends=edge_ends)[1]) == expected


@pytest.mark.parametrize("fetch_on_v", [False, True])
def test_edge_ends_is_the_rule_of_read_edges_on_any_region(filled, fetch_on_v):
    db, _ = filled
    edge_ends = "either" if fetch_on_v else "u"

    def scored(edges):
        order = np.lexsort((edges["v"], edges["u"]))
        return dict(zip(pairs(edges), edges["adj"][order].tolist(), strict=True))

    for roi in (
        Roi((0, 0, 0), (120, 120, 120)),
        Roi((100, -50, 37), (33, 400, 80)),
        Roi((1000, 1000, 1000), (10, 10, 10)),
        None,
    ):
        listed = db.read_edges(roi, fetch_on_v=fetch_on_v)
        _, edges = db.read_arrays(roi, edge_ends=edge_ends)
        assert scored(edges) == {(e["u"], e["v"]): e["adj"] for e in listed}
    # with steps, the rule applies to every node read
    roi = Roi((0, 0, 0), (120, 120, 120))
    nodes, edges = db.read_arrays(roi, num_steps=1, edge_ends=edge_ends)
    listed = db.read_edges(
        nodes=[{"id": i} for i in nodes["id"].tolist()], fetch_on_v=fetch_on_v
    )
    assert pairs(edges) == sorted((e["u"], e["v"]) for e in listed)
    assert set(pairs(db.read_arrays(roi, num_steps=1)[1])) < set(pairs(edges))


def test_an_unknown_edge_ends_is_refused(tmp_path):
    with pytest.raises(ValueError, match="edge_ends is 'v'; it is one of"):
        chain(tmp_path).read_arrays(edge_ends="v")
