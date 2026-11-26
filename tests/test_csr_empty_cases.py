"""Every read and write of the CSR graph database on the four ways a result
can be empty: a region with no node, a region with nodes and no edge, a
graph with no edge at all, and a graph with no node."""

import itertools

import networkx as nx
import numpy as np
import pytest
from funlib.geometry import Roi

from funlib.persistence.graphs import CSRGraphDataBase
from funlib.persistence.types import Vec

NONE = np.zeros(0, dtype=np.uint64)
WHOLE = Roi((0, 0, 0), (1000, 1000, 1000))


def build(path, case):
    """The graph of a case, the region to read, and the ids of the nodes in
    that region and of those within reach of it."""
    if case == "no node in the graph":
        ids, pos = NONE, np.zeros((0, 3))
        u = v = NONE
    else:
        # 1-2-3 joined near the origin; 7 and 8 on their own, far away
        ids = np.array([1, 2, 3, 7, 8], dtype=np.uint64)
        pos = np.array(
            [[5, 5, 5], [5, 5, 15], [5, 5, 25], [505, 505, 505], [505, 505, 515]],
            dtype=np.float64,
        )
        u, v = ids[:2], ids[1:3]
    if case == "no edge in the graph":
        u = v = NONE
    weight = np.full(len(u), 0.5, dtype=np.float32)
    kind = np.ones(len(u), dtype=np.int8)
    if case == "no edge in the graph, none given":
        db = CSRGraphDataBase.from_arrays(
            path, ids, {"position": pos, "size": np.arange(len(ids))}, cell_size=20.0
        )
    else:
        db = CSRGraphDataBase.from_arrays(
            path,
            ids,
            {"position": pos, "size": np.arange(len(ids))},
            u,
            v,
            {"weight": weight, "kind": kind},
            cell_size=20.0,
        )
    db.close()
    roi = {
        "no node in the region": Roi((800, 800, 800), (10, 10, 10)),
        "nodes and no edge in the region": Roi((500, 500, 500), (20, 20, 20)),
    }.get(case, WHOLE)
    lo, hi = np.array(roi.begin), np.array(roi.end)
    inside = ids[((pos >= lo) & (pos < hi)).all(axis=1)] if len(ids) else NONE
    return roi, sorted(inside.tolist())


CASES = [
    "no node in the region",
    "nodes and no edge in the region",
    "no edge in the graph",
    "no edge in the graph, none given",
    "no node in the graph",
]


def check(db, nodes, edges, expected_ids, extras, own=False, node_attrs=None):
    """A result of the documented shape, whatever it holds."""
    n = len(nodes["id"])
    assert nodes["id"].dtype == np.uint64 and nodes["id"].ndim == 1
    assert nodes["step"].dtype == np.int16 and nodes["step"].shape == (n,)
    if expected_ids is not None:
        assert sorted(nodes["id"].tolist()) == expected_ids
    names = list(db.node_attrs) if node_attrs is None else node_attrs
    assert set(nodes) == {"id", "step", *names, *(["own"] if own else [])}
    if "position" in names:
        assert nodes["position"].shape == (n, 3)
        assert nodes["position"].dtype == np.float64
    if "size" in names:
        assert nodes["size"].shape == (n,) and nodes["size"].dtype == np.int64
    if extras and node_attrs is None:
        assert nodes["vec"].shape == (n, 3) and nodes["vec"].dtype == np.float32
        assert nodes["label"].shape == (n,) and nodes["label"].dtype == np.int32
    if own:
        assert nodes["own"].dtype == bool and nodes["own"].shape == (n,)
    m = len(edges["u"])
    assert edges["u"].dtype == edges["v"].dtype == np.uint64
    assert edges["v"].shape == (m,)
    assert set(edges) == {"u", "v", *db.edge_attrs}
    for name, dtype in (("weight", np.float32), ("kind", np.int8)):
        if name in edges:
            assert edges[name].shape == (m,) and edges[name].dtype == dtype
    return n, m


@pytest.mark.parametrize("extras", [False, True], ids=["plain", "extra fields"])
@pytest.mark.parametrize("case", CASES)
def test_every_read_and_write_on_an_empty_result(tmp_path, case, extras):
    path = tmp_path / "g.csr"
    roi, inside = build(path, case)
    ids_all = sorted(CSRGraphDataBase(path).read_arrays()[0]["id"].tolist())
    has_weight = "weight" in CSRGraphDataBase(path).edge_attrs

    if extras:
        # -- the extra-fields writes and their finalize ---------------------
        db = CSRGraphDataBase(path, mode="r+", extra_fields=tmp_path / "pred")
        db.finalize()  # nothing written yet
        with db.bulk_write_mode():
            pass
        nodes_in, _ = db.read_arrays(roi)
        with db.bulk_write_mode():
            # the nodes of the region, which may be none
            db.write_node_arrays(
                nodes_in["id"],
                vec=np.ones((len(nodes_in["id"]), 3), np.float32),
                label=np.full(len(nodes_in["id"]), 4, np.int32),
            )
        assert db.node_attrs["vec"] == Vec("float32", 3)
        graph = db.read_graph(roi, node_attrs=["position", "label"])
        for _, data in graph.nodes(data=True):
            if "position" in data:
                data["label"] = 9
        db.write_attrs(graph, roi=roi, node_attrs=["label"])
        db.write_nodes({}, attributes=["label", "another"])
        db.write_nodes({int(i): {"label": 9} for i in inside})
        db.finalize()
        db = CSRGraphDataBase(path, extra_fields=tmp_path / "pred")
        labels = db.read_arrays()[0]
        written = sorted(labels["id"][labels["label"] == 9].tolist())
        assert written == inside
    else:
        db = CSRGraphDataBase(path)

    # -- read_arrays -------------------------------------------------------
    step_choices = [None, [], *([["weight"]] if has_weight else [])]
    for steps, step_edges, edge_ends in itertools.product(
        (0, 1, 3), step_choices, ("both", "u", "either")
    ):
        nodes, edges = db.read_arrays(
            roi, num_steps=steps, step_edges=step_edges, edge_ends=edge_ends
        )
        n, m = check(db, nodes, edges, inside if steps == 0 else None, extras)
        assert set(inside) <= set(nodes["id"].tolist()) <= set(ids_all)
        if edge_ends == "both":
            assert set(edges["u"].tolist()) | set(edges["v"].tolist()) <= set(
                nodes["id"].tolist()
            )
    nodes, edges = db.read_arrays(roi, node_attrs=[], edge_attrs=[])
    assert set(nodes) == {"id", "step"} and set(edges) == {"u", "v"}
    nodes, edges = db.read_arrays()
    assert check(db, nodes, edges, ids_all, extras)[0] == len(ids_all)

    # -- read_block --------------------------------------------------------
    contexts = (None, 0, 5, 2.5, [0, 0, 30], 2000)
    for context, steps, edge_ends in itertools.product(
        contexts, (0, 2), ("both", "u", "either")
    ):
        for step_edges in step_choices:
            for node_attrs in (None, [], ["size"]):
                nodes, edges = db.read_block(
                    roi,
                    context=context,
                    num_steps=steps,
                    node_attrs=node_attrs,
                    step_edges=step_edges,
                    edge_ends=edge_ends,
                )
                check(db, nodes, edges, None, extras, own=True, node_attrs=node_attrs)
                assert sorted(nodes["id"][nodes["own"]].tolist()) == inside
                if context is None and steps == 0:
                    assert nodes["own"].all()

    # -- the reads that give dictionaries and a networkx graph -------------
    listed = db.read_nodes(roi)
    assert sorted(n["id"] for n in listed) == inside
    assert db.read_nodes(roi, attr_filter={"size": 10**6}) == []
    for fetch_on_v in (False, True):
        from_roi = db.read_edges(roi, fetch_on_v=fetch_on_v)
        from_nodes = db.read_edges(nodes=listed, fetch_on_v=fetch_on_v)
        assert sorted((e["u"], e["v"]) for e in from_roi) == sorted(
            (e["u"], e["v"]) for e in from_nodes
        )
        assert db.read_edges(nodes=[], fetch_on_v=fetch_on_v) == []
        graph = db.read_graph(roi, fetch_on_v=fetch_on_v)
        assert set(inside) <= set(graph.nodes)
        assert graph.number_of_edges() == len(from_roi)
        _, edges = db.read_arrays(roi, edge_ends="either" if fetch_on_v else "u")
        assert len(edges["u"]) == len(from_roi)
    assert db.read_graph(roi, read_edges=False).number_of_edges() == 0
    assert db.read_graph(roi, nodes_filter={"size": 10**6}).number_of_nodes() == 0
    assert isinstance(db.read_graph(), nx.Graph)
    assert db.has_edges(roi) is bool(db.read_edges(roi, fetch_on_v=True))
    assert db.num_nodes == len(ids_all)


def test_a_database_that_was_created_and_never_written_reads_as_empty(tmp_path):
    db = CSRGraphDataBase(
        tmp_path / "g.csr",
        mode="w",
        node_attrs={"position": Vec(float, 3), "size": int},
        edge_attrs={"weight": "float32", "kind": "int8"},
        cell_size=20.0,
    )
    for handle in (db, CSRGraphDataBase(tmp_path / "g.csr")):
        for context, steps, edge_ends in itertools.product(
            (None, 0, 5), (0, 2), ("both", "u", "either")
        ):
            nodes, edges = handle.read_block(
                WHOLE, context=context, num_steps=steps, edge_ends=edge_ends
            )
            assert check(handle, nodes, edges, [], False, own=True) == (0, 0)
        nodes, edges = handle.read_arrays(num_steps=1, step_edges=["weight"])
        assert check(handle, nodes, edges, [], False) == (0, 0)
        assert handle.read_nodes(WHOLE) == [] and handle.read_edges(WHOLE) == []
        assert handle.read_graph(WHOLE).number_of_nodes() == 0
        assert handle.has_edges(WHOLE) is False
    db.finalize()  # nothing to merge
    assert db.num_nodes == 0
