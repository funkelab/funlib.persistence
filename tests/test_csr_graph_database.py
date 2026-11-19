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
