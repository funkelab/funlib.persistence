"""Extra node attributes of a CSR graph, kept in directories of their own."""

import hashlib
import multiprocessing as mp
from typing import Any

import networkx as nx
import numpy as np
import pytest
from funlib.geometry import Roi

from funlib.persistence.graphs import CSRGraphDataBase
from funlib.persistence.graphs.csr_graph_database import (
    GraphIsReadOnly,
    UnfinalizedWrites,
)
from funlib.persistence.graphs.graph_database import AttributeType
from funlib.persistence.types import Vec

N = 600
TYPES: dict[str, AttributeType] = {
    "vec": Vec("float32", 3),
    "label": "int32",
    "seen": bool,
    "score": float,
}


def graph_arrays(n=N, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.choice(10 * n, size=n, replace=False).astype(np.uint64) + 1
    pos = rng.uniform(0, 100, size=(n, 3))
    u, v = ids[:-1], ids[1:]  # a path through the nodes in the order given
    return ids, pos, u, v


def make_graph(path, n=N, seed=0):
    ids, pos, u, v = graph_arrays(n, seed)
    weight = np.ones(len(u), dtype=np.float32)
    CSRGraphDataBase.from_arrays(
        path, ids, {"position": pos}, u, v, {"weight": weight}, cell_size=20.0
    ).close()
    return ids, pos


def vec_of(ids):
    ids = np.asarray(ids, dtype=np.float32)
    return np.stack([ids, ids * 2, ids * 3], axis=1)


def expanded(tmp_path, mode="r+", names=("vec", "label", "seen", "score"), **kwargs):
    return CSRGraphDataBase(
        tmp_path / "g.csr",
        mode=mode,
        node_attrs={k: TYPES[k] for k in names} if mode == "r+" else None,
        extra_fields={k: tmp_path / "pred" for k in names},
        **kwargs,
    )


def snapshot(folder):
    """Every file under a directory with the sha256 of its bytes."""
    return {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def by_id(nodes, name):
    return dict(zip(nodes["id"].tolist(), nodes[name].tolist(), strict=True))


def test_extra_attributes_are_written_and_read_through_the_graph(tmp_path):
    ids, pos = make_graph(tmp_path / "g.csr")
    before = snapshot(tmp_path / "g.csr")
    db = expanded(tmp_path)
    assert db.node_attrs == {
        "position": Vec("float64", 3),
        "vec": Vec("float32", 3),
        "label": "int32",
        "seen": "bool",
        "score": "float64",
    }
    half = ids[: N // 2]
    with db.bulk_write_mode():
        db.write_node_arrays(
            half,
            vec=vec_of(half),
            label=(half % 5 + 1).astype(np.int32),
            seen=np.ones(len(half), dtype=bool),
            score=half / 7,
        )

    # the graph's own directory: no file added, removed or changed
    assert snapshot(tmp_path / "g.csr") == before
    # the extra directory: meta.json, the lock, one array per attribute
    names = sorted(p.name.split("-")[0] for p in (tmp_path / "pred").iterdir())
    assert names == [
        "finalize.lock",
        "meta.json",
        "node_label",
        "node_score",
        "node_seen",
        "node_vec",
        "segments",
    ]
    assert list((tmp_path / "pred" / "segments").iterdir()) == []

    # read back, by a new handle that names what it wants
    db = expanded(tmp_path, mode="r")
    nodes, edges = db.read_arrays()
    assert len(nodes["id"]) == N and len(edges["u"]) == N - 1
    written = np.isin(nodes["id"], half)
    assert nodes["vec"][written].tolist() == vec_of(nodes["id"][written]).tolist()
    assert nodes["label"][written].tolist() == (nodes["id"][written] % 5 + 1).tolist()
    assert nodes["seen"][written].all()
    assert nodes["score"][written].tolist() == (nodes["id"][written] / 7).tolist()
    # a node that was never written: NaN for a float, 0 for an integer or bool
    assert np.isnan(nodes["vec"][~written]).all()
    assert np.isnan(nodes["score"][~written]).all()
    assert not nodes["label"][~written].any() and not nodes["seen"][~written].any()
    # values are in the graph's row order: the arrays line up with its ids
    rows = np.load(tmp_path / "g.csr" / db.meta["generation"] / "ids.npy")
    (file,) = (tmp_path / "pred").glob("node_label-*.npy")
    stored = np.load(file)
    assert stored.shape == (N,)
    assert stored.tolist() == np.where(np.isin(rows, half), rows % 5 + 1, 0).tolist()


def test_region_and_step_reads_return_extra_attributes(tmp_path):
    ids, pos = make_graph(tmp_path / "g.csr")
    db = expanded(tmp_path, names=("vec", "label"))
    with db.bulk_write_mode():
        db.write_node_arrays(ids, vec=vec_of(ids), label=(ids % 9).astype(np.int32))
    plain = CSRGraphDataBase(tmp_path / "g.csr")
    roi = Roi((0, 0, 0), (50, 60, 70))

    for steps in (0, 2):
        wanted, wanted_edges = plain.read_arrays(roi, num_steps=steps)
        nodes, edges = db.read_arrays(roi, num_steps=steps)
        assert (
            nodes["id"].tolist() == wanted["id"].tolist() and 0 < len(nodes["id"]) < N
        )
        assert nodes["step"].tolist() == wanted["step"].tolist()
        assert edges["u"].tolist() == wanted_edges["u"].tolist()
        assert nodes["vec"].tolist() == vec_of(nodes["id"]).tolist()
        assert nodes["label"].tolist() == (nodes["id"] % 9).tolist()

    nodes, _ = db.read_block(roi, context=10, num_steps=1, node_attrs=["label"])
    assert set(nodes) == {"id", "label", "step", "own"}
    assert nodes["label"].tolist() == (nodes["id"] % 9).tolist()
    assert 0 < nodes["own"].sum() < len(nodes["id"])

    listed = db.read_nodes(roi, read_attrs=["label", "vec"])
    assert sorted(n["id"] for n in listed) == sorted(plain.read_arrays(roi)[0]["id"])
    assert all(n["label"] == n["id"] % 9 for n in listed)
    assert all(n["vec"] == (n["id"], n["id"] * 2, n["id"] * 3) for n in listed)
    only = db.read_nodes(roi, attr_filter={"label": 4}, read_attrs=["label"])
    assert only and all(n["id"] % 9 == 4 for n in only)

    graph = db.read_graph(roi)
    inside = [n for n, d in graph.nodes(data=True) if "position" in d]
    assert all(graph.nodes[n]["label"] == n % 9 for n in inside)
    assert graph.number_of_edges() == plain.read_graph(roi).number_of_edges()

    with pytest.raises(ValueError, match="unknown node attribute 'nope'"):
        db.read_arrays(roi, node_attrs=["nope"])
    # the plain handle does not know the extra attributes
    with pytest.raises(ValueError, match="unknown node attribute 'label'"):
        plain.read_arrays(roi, node_attrs=["label"])


def _worker(root, part, parts, start):
    ids, _, _, _ = graph_arrays()
    db = CSRGraphDataBase(
        root / "g.csr",
        mode="r+",
        node_attrs={"vec": TYPES["vec"], "label": TYPES["label"]},
        extra_fields={"vec": root / "pred", "label": root / "pred"},
    )
    start.wait()
    with db.bulk_write_mode(worker=True):
        for block in np.array_split(ids[part::parts], 10):
            db.write_node_arrays(
                block, vec=vec_of(block), label=(block % 9).astype(np.int32)
            )


def test_several_processes_write_one_extra_directory(tmp_path):
    ids, _ = make_graph(tmp_path / "g.csr")
    before = snapshot(tmp_path / "g.csr")
    parts = 4
    ctx = mp.get_context("spawn")
    start = ctx.Barrier(parts)
    procs = [
        ctx.Process(target=_worker, args=(tmp_path, part, parts, start))
        for part in range(parts)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert [p.exitcode for p in procs] == [0] * parts
    assert len(list((tmp_path / "pred" / "segments").glob("*.npys"))) == parts

    db = expanded(tmp_path, names=("vec", "label"))
    # until the finalize a read of the extra attributes is refused, a read
    # of the graph's own is not
    with pytest.raises(UnfinalizedWrites, match="pred has writes"):
        db.read_arrays()
    assert len(db.read_arrays(node_attrs=["position"])[0]["id"]) == N
    db.finalize()
    nodes, _ = db.read_arrays()
    assert sorted(nodes["id"].tolist()) == sorted(ids.tolist())
    assert nodes["vec"].tolist() == vec_of(nodes["id"]).tolist()
    assert nodes["label"].tolist() == (nodes["id"] % 9).tolist()
    assert list((tmp_path / "pred" / "segments").iterdir()) == []
    assert snapshot(tmp_path / "g.csr") == before


def test_a_later_finalize_adds_to_the_values_that_are_there(tmp_path):
    ids, _ = make_graph(tmp_path / "g.csr")
    db = expanded(tmp_path, names=("score", "label"))
    a, b, c = ids[:200], ids[150:400], ids[400:]
    with db.bulk_write_mode():
        db.write_node_arrays(a, score=np.full(len(a), 1.0))
    files = sorted(p.name for p in (tmp_path / "pred").glob("node_*"))
    with db.bulk_write_mode():
        # written twice: the last value stays; a NaN is not a value
        db.write_node_arrays(b, score=np.full(len(b), 2.0))
        db.write_node_arrays(a[:10], score=np.full(10, np.nan))
        db.write_node_arrays(b[:5], score=np.full(5, 3.0))
    # one array per attribute: the one before is gone, label was never built
    again = sorted(p.name for p in (tmp_path / "pred").glob("node_*"))
    assert len(files) == len(again) == 1 and files != again
    score = by_id(db.read_arrays(node_attrs=["score", "label"])[0], "score")
    assert all(score[i] == 1.0 for i in a[:150].tolist())
    assert all(score[i] == 3.0 for i in b[:5].tolist())
    assert all(score[i] == 2.0 for i in b[5:].tolist())
    assert all(np.isnan(score[i]) for i in c.tolist())
    labels = db.read_arrays(node_attrs=["label"])[0]["label"]
    assert labels.shape == (N,) and not labels.any()

    # a run taken up again by another handle
    other = expanded(tmp_path, names=("score",))
    other.write_node_arrays(c, score=np.full(len(c), 4.0))  # finalizes on its own
    score = by_id(expanded(tmp_path, "r", ("score",)).read_arrays()[0], "score")
    assert not np.isnan(list(score.values())).any()
    assert all(score[i] == 4.0 for i in c.tolist())


def test_the_graph_is_read_only_through_a_handle_with_extra_fields(tmp_path):
    ids, pos = make_graph(tmp_path / "g.csr")
    before = snapshot(tmp_path / "g.csr")
    db = expanded(tmp_path, names=("label",))
    graph = nx.Graph()
    graph.add_node(int(ids[0]), position=tuple(pos[0]), label=3)
    graph.add_edge(int(ids[0]), int(ids[1]), weight=2.0)
    one = ids[:1]
    refused = [
        lambda: db.write_node_arrays(one, position=pos[:1]),
        lambda: db.write_node_arrays(one, position=pos[:1], label=[1]),
        lambda: db.write_edge_arrays(ids[:1], ids[2:3], weight=[1.0]),
        lambda: db.write_nodes(graph.nodes, attributes=["position"]),
        lambda: db.write_edges(graph.nodes, graph.edges),
        lambda: db.bulk_write_edges(graph.nodes, graph.edges),
        lambda: db.write_graph(graph),
        lambda: db.bulk_write_graph(graph),
        lambda: db.write_attrs(graph, edge_attrs=["weight"]),
        lambda: db.write_attrs(graph, node_attrs=["position"]),
        lambda: db.drop_edges(),
    ]
    for write in refused:
        with pytest.raises(GraphIsReadOnly, match="read-only through a handle"):
            write()
    with pytest.raises(ValueError, match="unknown node attributes"):
        db.write_node_arrays(one, nope=[1])
    db.finalize()
    assert snapshot(tmp_path / "g.csr") == before
    assert list((tmp_path / "pred" / "segments").glob("*.npys")) == []
    assert not db.read_arrays()[0]["label"].any()
    # a handle that only reads refuses every write
    with pytest.raises(RuntimeError, match="read-only DB"):
        expanded(tmp_path, mode="r", names=("label",)).write_node_arrays(one, label=[1])


def test_nodes_given_as_dictionaries_write_their_extra_attributes(tmp_path):
    ids, pos = make_graph(tmp_path / "g.csr")
    db = expanded(tmp_path, names=("label", "vec"))
    roi = Roi((0, 0, 0), (50, 100, 100))
    graph = db.read_graph(node_attrs=["position"])
    for node, data in graph.nodes(data=True):
        data["label"] = node % 9 + 1
        if node % 2:
            data["vec"] = (node, 0.0, 0.0)
    db.write_attrs(graph, roi=roi, node_attrs=["label", "vec"])
    nodes, _ = db.read_arrays()
    inside = (nodes["position"][:, 0] < 50).tolist()
    for node, label, vec, here in zip(
        nodes["id"].tolist(), nodes["label"], nodes["vec"], inside, strict=True
    ):
        assert label == (node % 9 + 1 if here else 0)
        assert (vec[0] == node) if here and node % 2 else np.isnan(vec).all()

    # without a region no position is needed; every extra attribute is taken
    values = {int(i): {"label": 77} for i in ids[:3]}
    db.write_nodes(values)
    assert by_id(db.read_arrays()[0], "label")[int(ids[0])] == 77
    db.write_graph(nx.Graph([(int(ids[3]), int(ids[4]))]), write_edges=False)
    nx_graph = nx.Graph()
    nx_graph.add_node(int(ids[5]), label=78)
    db.write_graph(nx_graph, write_edges=False, node_attrs=["label"])
    assert by_id(db.read_arrays()[0], "label")[int(ids[5])] == 78


def test_extra_fields_of_another_graph_are_refused(tmp_path):
    ids, pos = make_graph(tmp_path / "g.csr")
    db = expanded(tmp_path, names=("label",))
    db.write_node_arrays(ids, label=(ids % 9).astype(np.int32))
    fields = {"label": tmp_path / "pred"}

    make_graph(tmp_path / "fewer.csr", n=N - 1)
    with pytest.raises(ValueError, match=f"a graph of {N} nodes; .* has {N - 1}"):
        CSRGraphDataBase(tmp_path / "fewer.csr", extra_fields=fields)
    make_graph(tmp_path / "other.csr", seed=1)  # as many nodes, other ids
    with pytest.raises(ValueError, match="a graph with other nodes"):
        CSRGraphDataBase(tmp_path / "other.csr", extra_fields=fields)
    with pytest.raises(ValueError, match="a graph with other nodes"):
        CSRGraphDataBase(tmp_path / "other.csr", mode="r+", extra_fields=fields)

    # the same nodes built again, or edges added, are the same rows
    make_graph(tmp_path / "again.csr")
    again = CSRGraphDataBase(tmp_path / "again.csr", extra_fields=fields)
    nodes, _ = again.read_arrays()
    assert nodes["label"].tolist() == (nodes["id"] % 9).tolist()
    plain = CSRGraphDataBase(tmp_path / "g.csr", mode="r+")
    plain.write_edge_arrays(ids[:1], ids[5:6], weight=np.ones(1, np.float32))
    assert plain.num_edges == N
    nodes, edges = expanded(tmp_path, mode="r", names=("label",)).read_arrays()
    assert nodes["label"].tolist() == (nodes["id"] % 9).tolist()
    assert len(edges["u"]) == N
    # a graph that changed under an open handle is noticed at the next write
    plain.write_node_arrays(np.array([10**9], dtype=np.uint64), position=pos[:1])
    with pytest.raises(ValueError, match=f"a graph of {N} nodes"):
        db.write_node_arrays(ids[:1], label=[1])


def test_a_value_for_an_id_that_is_not_a_node_stops_the_finalize(tmp_path):
    ids, _ = make_graph(tmp_path / "g.csr")
    db = expanded(tmp_path, names=("label",))
    db.write_node_arrays(ids[:5], label=np.full(5, 1, np.int32))
    stranger = np.array([ids[5], 10**12], dtype=np.uint64)
    with pytest.raises(ValueError, match=r"1 ids that are not nodes.*1000000000000"):
        db.write_node_arrays(stranger, label=[2, 2])
    # nothing was finalized: the arrays are as they were, the segment stays
    assert len(list((tmp_path / "pred" / "segments").glob("*.npys"))) == 1
    reader = expanded(tmp_path, mode="r", names=("label",))
    assert reader.read_arrays(node_attrs=["position"])[0]["id"].shape == (N,)
    with pytest.raises(UnfinalizedWrites):
        reader.read_arrays()
    for segment in (tmp_path / "pred" / "segments").glob("*.npys"):
        segment.unlink()
    assert int(reader.read_arrays()[0]["label"].sum()) == 5


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"mode": "w", "cell_size": 1.0}, ValueError, "mode 'w' creates the graph"),
        ({"mode": "r"}, ValueError, r"\['label'\] are not in .*Open with mode 'r\+'"),
        ({"mode": "r+"}, ValueError, r"\['label'\] are new in .*types in node_attrs"),
        (
            {"mode": "r+", "node_attrs": {"label": str}},
            NotImplementedError,
            "not supported by the CSR backend",
        ),
        (
            {"mode": "r+", "extra_fields": {"position": "x"}},
            ValueError,
            r"\['position'\] are node attributes of .* itself",
        ),
        (
            {"mode": "r+", "extra_fields": {"label": "g.csr"}},
            ValueError,
            "outside the graph's directory",
        ),
    ],
)
def test_extra_fields_that_cannot_be_opened_are_refused(
    tmp_path, monkeypatch, kwargs, error, match
):
    make_graph(tmp_path / "g.csr")
    before = snapshot(tmp_path / "g.csr")
    monkeypatch.chdir(tmp_path)
    given: dict[str, Any] = {"extra_fields": {"label": "pred"}, **kwargs}
    with pytest.raises(error, match=match):
        CSRGraphDataBase("g.csr", **given)
    assert snapshot(tmp_path / "g.csr") == before
    assert not (tmp_path / "pred" / "meta.json").exists()


def test_a_declared_type_has_to_be_the_stored_one(tmp_path):
    ids, _ = make_graph(tmp_path / "g.csr")
    expanded(tmp_path, names=("vec", "label")).close()
    fields = {"vec": tmp_path / "pred", "label": tmp_path / "pred"}
    other: list[dict[str, AttributeType]] = [
        {"vec": Vec("float32", 4)},
        {"label": "int64"},
    ]
    for types in other:
        with pytest.raises(ValueError, match="in .*pred, not"):
            CSRGraphDataBase(tmp_path / "g.csr", node_attrs=types, extra_fields=fields)
    # stored attributes are still checked against the graph
    with pytest.raises(ValueError, match="node attribute 'nope' is not in"):
        CSRGraphDataBase(
            tmp_path / "g.csr", node_attrs={"nope": int}, extra_fields=fields
        )
    db = CSRGraphDataBase(
        tmp_path / "g.csr",
        node_attrs={"position": Vec(float, 3), **{k: TYPES[k] for k in fields}},
        extra_fields=fields,
    )
    assert list(db.node_attrs) == ["position", "vec", "label"]
    with pytest.raises(ValueError, match=r"'label' has shape \(1,\).*each of the 3"):
        expanded(tmp_path, names=("label",)).write_node_arrays(ids[:3], label=[1])


def test_a_graph_that_was_never_finalized_takes_no_extra_fields(tmp_path):
    CSRGraphDataBase(
        tmp_path / "g.csr",
        mode="w",
        node_attrs={"position": Vec(float, 3)},
        cell_size=10.0,
    )
    with pytest.raises(ValueError, match="was never finalized"):
        expanded(tmp_path, names=("label",))


def test_a_directory_that_holds_something_else_is_not_taken_for_extra_fields(tmp_path):
    make_graph(tmp_path / "g.csr")
    make_graph(tmp_path / "h.csr")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "meta.json").write_text('{"version": 1}')
    for folder in (tmp_path / "h.csr", tmp_path / "other"):
        before = snapshot(folder)
        with pytest.raises(ValueError, match="not a directory of extra node attr"):
            CSRGraphDataBase(
                tmp_path / "g.csr",
                mode="r+",
                node_attrs={"label": int},
                extra_fields={"label": folder},
            )
        assert snapshot(folder) == before


def test_attributes_in_several_directories_and_several_in_one(tmp_path):
    ids, _ = make_graph(tmp_path / "g.csr")
    a, b = tmp_path / "model_a", tmp_path / "deep" / "model_b"
    db = CSRGraphDataBase(
        tmp_path / "g.csr",
        mode="r+",
        node_attrs={"vec": TYPES["vec"], "label": "int32", "score": float},
        extra_fields={"vec": a, "label": a, "score": b},
    )
    with db.bulk_write_mode():
        db.write_node_arrays(ids, vec=vec_of(ids), score=ids / 2)
        db.write_node_arrays(ids, label=(ids % 9).astype(np.int32))
    assert CSRGraphDataBase.stored_extra_fields(a) == {
        "vec": Vec("float32", 3),
        "label": "int32",
    }
    assert CSRGraphDataBase.stored_extra_fields(b) == {"score": "float64"}
    assert CSRGraphDataBase.stored_extra_fields(tmp_path / "g.csr") == {}
    assert CSRGraphDataBase.stored_extra_fields(tmp_path / "missing") == {}
    assert not CSRGraphDataBase.is_database(a)

    # a handle sees the attributes it names, under the directory it names
    one = CSRGraphDataBase(tmp_path / "g.csr", extra_fields={"label": a, "score": b})
    assert list(one.node_attrs) == ["position", "label", "score"]
    nodes, _ = one.read_arrays()
    assert "vec" not in nodes
    assert nodes["label"].tolist() == (nodes["id"] % 9).tolist()
    assert nodes["score"].tolist() == (nodes["id"] / 2).tolist()
    # pending writes in one directory do not stop a read of the other
    with db.bulk_write_mode(worker=True):
        db.write_node_arrays(ids[:1], score=np.array([0.5]))
    assert len(db.read_arrays(node_attrs=["vec", "label"])[0]["id"]) == N
    with pytest.raises(UnfinalizedWrites, match="model_b"):
        db.read_block(Roi((0, 0, 0), (100, 100, 100)))
    db.finalize()
    assert by_id(db.read_arrays()[0], "score")[int(ids[0])] == 0.5
