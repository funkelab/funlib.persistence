"""Extra node attributes of a CSR graph, kept in directories of their own."""

import hashlib
import json
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


def expanded(tmp_path, mode="r+", names=("vec", "label", "seen", "score")):
    """The graph with the directory ``pred``; in mode r+ the attributes
    ``names`` are declared up front, with the types of ``TYPES``."""
    return CSRGraphDataBase(
        tmp_path / "g.csr",
        mode=mode,
        node_attrs={k: TYPES[k] for k in names} if mode == "r+" else None,
        extra_fields=tmp_path / "pred",
    )


def opened(tmp_path, mode="r", folder="pred"):
    """The graph opened with a directory, by its path alone."""
    return CSRGraphDataBase(
        tmp_path / "g.csr", mode=mode, extra_fields=tmp_path / folder
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
    # the path alone: every worker creates the attributes with its first write
    db = CSRGraphDataBase(root / "g.csr", mode="r+", extra_fields=root / "pred")
    start.wait()
    with db.bulk_write_mode(worker=True):
        for block in np.array_split(ids[part::parts], 10):
            db.write_node_arrays(
                block, vec=vec_of(block), label=(block % 9).astype(np.int32)
            )


def test_several_processes_create_and_write_the_same_attributes(tmp_path):
    ids, _ = make_graph(tmp_path / "g.csr")
    before = snapshot(tmp_path / "g.csr")
    assert not (tmp_path / "pred").exists()
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
    # they agreed on one declaration, the type of the values written
    assert CSRGraphDataBase.stored_extra_fields(tmp_path / "pred") == {
        "vec": Vec("float32", 3),
        "label": "int32",
    }

    db = opened(tmp_path, "r+")
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


def test_the_path_alone_opens_what_a_directory_holds(tmp_path):
    ids, _ = make_graph(tmp_path / "g.csr")
    with pytest.raises(ValueError, match="holds no extra node attributes.*'r\\+'"):
        opened(tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="holds no extra node attributes"):
        opened(tmp_path, folder="empty")
    db = opened(tmp_path, "r+")  # a directory that does not exist yet
    assert not (tmp_path / "pred").exists()
    assert db.node_attrs == {"position": Vec("float64", 3)}
    assert len(db.read_arrays()[0]["id"]) == N

    # the first write of a name creates it, with the type of its values
    with db.bulk_write_mode():
        db.write_node_arrays(
            ids,
            vec=vec_of(ids),
            label=(ids % 9).astype(np.int16),
            seen=ids % 2 == 0,
        )
    types: dict[str, AttributeType] = {
        "vec": Vec("float32", 3),
        "label": "int16",
        "seen": "bool",
    }
    assert CSRGraphDataBase.stored_extra_fields(tmp_path / "pred") == types
    assert db.node_attrs == {"position": Vec("float64", 3), **types}
    stored = json.loads((tmp_path / "pred" / "meta.json").read_text())
    assert stored["node_attrs"]["vec"] == {"dtype": "<f4", "size": 3}

    # a reader names nothing and gives no type
    pred = tmp_path / "pred"
    for extra_fields in (pred, str(pred), [pred], (str(pred),)):
        reader = CSRGraphDataBase(tmp_path / "g.csr", extra_fields=extra_fields)
        assert reader.node_attrs == {"position": Vec("float64", 3), **types}
        nodes, _ = reader.read_arrays(Roi((0, 0, 0), (50, 50, 50)))
        assert nodes["label"].dtype == np.int16 and nodes["vec"].shape[1] == 3
        assert nodes["label"].tolist() == (nodes["id"] % 9).tolist()
    mapping: Any = {"vec": pred}
    with pytest.raises(TypeError, match="a directory or a sequence of directories"):
        CSRGraphDataBase(tmp_path / "g.csr", extra_fields=mapping)


def test_a_later_write_has_the_type_that_was_recorded(tmp_path):
    ids, _ = make_graph(tmp_path / "g.csr")
    db = opened(tmp_path, "r+")
    few = ids[:4]
    db.write_node_arrays(few, vec=vec_of(few), label=np.ones(4, np.int32))
    refused: list[tuple[dict[str, Any], str]] = [
        ({"vec": vec_of(few).astype(np.float64)}, "'vec' is Vec.*float32.*, not Vec"),
        ({"vec": vec_of(few)[:, :2]}, "'vec' is Vec.*size=3.*size=2"),
        ({"vec": vec_of(few)[:, 0]}, "'vec' is Vec.*not float32"),
        ({"label": np.ones(4, np.int64)}, "'label' is int32 in .*pred, not int64"),
        ({"label": [1, 1, 1, 1]}, "'label' is int32 in .*pred, not int64"),
    ]
    for values, match in refused:
        with pytest.raises(ValueError, match=match):
            db.write_node_arrays(few, **values)
    # values that cannot be an attribute at all
    with pytest.raises(NotImplementedError, match="'name'.*not supported"):
        db.write_node_arrays(few, name=np.array(["a", "b", "c", "d"]))
    with pytest.raises(ValueError, match=r"'cube' has shape \(4, 2, 2\)"):
        db.write_node_arrays(few, cube=np.zeros((4, 2, 2)))
    with pytest.raises(ValueError, match=r"'label' has shape \(1,\).*each of the 4"):
        db.write_node_arrays(few, label=np.ones(1, np.int32))
    # the names the reads use for arrays of their own
    for name in ("id", "step", "own"):
        with pytest.raises(ValueError, match=f"'{name}'.* are not names of node attr"):
            db.write_node_arrays(few, **{name: np.ones(4, np.float32)})
        with pytest.raises(ValueError, match="are not names of node attr"):
            db.write_nodes({int(few[0]): {name: 1}}, attributes=[name])
        with pytest.raises(ValueError, match="are not names of node attr"):
            CSRGraphDataBase(
                tmp_path / "g.csr",
                mode="r+",
                node_attrs={name: "float32"},
                extra_fields=tmp_path / "pred",
            )
    # a call that is refused for one attribute declares none of the others
    with pytest.raises(ValueError, match="'label' is int32 in .*pred, not int64"):
        db.write_node_arrays(few, fresh=np.ones(4), label=np.ones(4, np.int64))
    with pytest.raises(ValueError, match="'cube' has shape"):
        db.write_node_arrays(few, fresh=np.ones(4), cube=np.zeros((4, 2, 2)))
    with pytest.raises(ValueError, match="each of the 4"):
        db.write_node_arrays(few, fresh=np.ones(4), short=np.ones(1))
    with pytest.raises(ValueError, match="'label' is int32"):
        db.write_nodes(
            {int(few[0]): {"fresh": 1, "label": 0.5}}, attributes=["fresh", "label"]
        )
    assert list((tmp_path / "pred" / "segments").glob("*.npys")) == []
    assert sorted(CSRGraphDataBase.stored_extra_fields(tmp_path / "pred")) == [
        "label",
        "vec",
    ]
    nodes, _ = opened(tmp_path).read_arrays()
    assert sorted(nodes) == ["id", "label", "position", "step", "vec"]
    assert nodes["id"].dtype == np.uint64 and sorted(nodes["id"]) == sorted(ids)
    # a directory that holds such a name (written by an earlier version) is
    # refused when it is opened, not read with the node ids overwritten
    meta = json.loads((tmp_path / "pred" / "meta.json").read_text())
    poisoned = tmp_path / "poisoned"
    poisoned.mkdir()
    meta["node_attrs"]["id"] = {"dtype": "<f4", "size": 0}
    (poisoned / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="holds a node attribute named 'id'"):
        opened(tmp_path, folder="poisoned")
    # another handle that writes the same name has to agree with the record
    other = opened(tmp_path, "r+")
    with pytest.raises(ValueError, match="'label' is int32 in .*pred, not float64"):
        other.write_node_arrays(few, label=np.ones(4))


def test_node_attrs_may_declare_an_extra_attribute_and_is_never_needed(tmp_path):
    ids, _ = make_graph(tmp_path / "g.csr")
    fields = tmp_path / "pred"
    db = CSRGraphDataBase(
        tmp_path / "g.csr",
        mode="r+",
        node_attrs={"score": "float32", "position": Vec(float, 3)},
        extra_fields=fields,
    )
    assert db.node_attrs["score"] == "float32"
    assert CSRGraphDataBase.stored_extra_fields(fields) == {"score": "float32"}
    # declared and never written: every node reads as missing
    assert np.isnan(opened(tmp_path).read_arrays()[0]["score"]).all()
    # nodes given as dictionaries are cast to the declared type
    db.write_nodes({int(ids[0]): {"score": 0.1}})
    assert by_id(db.read_arrays()[0], "score")[int(ids[0])] == np.float32(0.1)

    # naming it again checks the recorded type; a reader cannot create one
    other: list[dict[str, AttributeType]] = [
        {"score": "float64"},
        {"score": Vec("float32", 2)},
    ]
    for types in other:
        with pytest.raises(ValueError, match="'score' is float32 in .*pred, not"):
            CSRGraphDataBase(tmp_path / "g.csr", node_attrs=types, extra_fields=fields)
    with pytest.raises(ValueError, match=r"\['nope'\] are neither in .* nor in its"):
        CSRGraphDataBase(
            tmp_path / "g.csr", node_attrs={"nope": int}, extra_fields=fields
        )
    with pytest.raises(NotImplementedError, match="not supported by the CSR backend"):
        CSRGraphDataBase(
            tmp_path / "g.csr", mode="r+", node_attrs={"t": str}, extra_fields=fields
        )
    # without extra_fields, a name the graph does not store is refused as before
    with pytest.raises(ValueError, match="node attribute 'score' is not in"):
        CSRGraphDataBase(tmp_path / "g.csr", node_attrs={"score": "float32"})


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
    other = opened(tmp_path, "r+")
    other.write_node_arrays(c, score=np.full(len(c), 4.0))  # finalizes on its own
    score = by_id(opened(tmp_path).read_arrays()[0], "score")
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
        lambda: db.write_node_arrays(one, position=pos[:1], label=np.ones(1, np.int32)),
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
    db.finalize()
    assert snapshot(tmp_path / "g.csr") == before
    assert list((tmp_path / "pred" / "segments").glob("*.npys")) == []
    assert not db.read_arrays()[0]["label"].any()
    # a handle that only reads refuses every write
    with pytest.raises(RuntimeError, match="read-only DB"):
        opened(tmp_path).write_node_arrays(one, label=np.ones(1, np.int32))
    with pytest.raises(RuntimeError, match="read-only DB"):
        opened(tmp_path).write_node_arrays(one, new=np.ones(1))


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

    # with a region the nodes carry their position: one that has none is not
    # silently left out
    bare = db.read_graph(roi, node_attrs=["label"])
    with pytest.raises(ValueError, match="carries no 'position'.*without a roi"):
        db.write_attrs(bare, roi=roi, node_attrs=["label"])
    assert list((tmp_path / "pred" / "segments").glob("*.npys")) == []

    # a value is stored in the recorded type or refused, never changed:
    # label is int32, vec a float32 vector of 3
    one = int(ids[0])
    before = {k: by_id(db.read_arrays()[0], k)[one] for k in ("label", "vec")}
    changed: list[dict[str, Any]] = [
        {"label": 1.7},
        {"label": np.float32(2.0)},
        {"label": (1, 2)},
        {"label": "3"},
        {"vec": (9.0,)},
        {"vec": (1.0, 2.0)},
        {"vec": 9.0},
        {"vec": ("a", "b", "c")},
    ]
    for values in changed:
        with pytest.raises(ValueError, match="is not stored as that without chang"):
            db.write_nodes({one: values}, attributes=list(values))
    with pytest.raises(OverflowError):
        db.write_nodes({one: {"label": 2**40}}, attributes=["label"])
    after = {k: by_id(db.read_arrays()[0], k)[one] for k in before}
    np.testing.assert_equal(after, before)
    assert list((tmp_path / "pred" / "segments").glob("*.npys")) == []
    # numbers of the recorded kind are stored in its width; an integer is a
    # float without loss
    db.write_nodes({one: {"label": np.int64(5), "vec": (1, 2.5, np.float64(3))}})
    read = db.read_arrays()[0]
    assert by_id(read, "label")[one] == 5 and by_id(read, "vec")[one] == [1, 2.5, 3]

    # without a region no position is needed; left out, the attributes are
    # the extra ones that exist
    values = {int(i): {"label": 77, "unrelated": "x"} for i in ids[:3]}
    db.write_nodes(values)
    assert by_id(db.read_arrays()[0], "label")[int(ids[0])] == 77
    db.write_graph(nx.Graph([(int(ids[3]), int(ids[4]))]), write_edges=False)
    nx_graph = nx.Graph()
    nx_graph.add_node(int(ids[5]), label=78)
    db.write_graph(nx_graph, write_edges=False, node_attrs=["label"])
    assert by_id(db.read_arrays()[0], "label")[int(ids[5])] == 78


def test_nodes_given_as_dictionaries_create_an_attribute_that_is_named(tmp_path):
    ids, _ = make_graph(tmp_path / "g.csr")
    db = opened(tmp_path, "r+")
    a, b, c = (int(i) for i in ids[:3])
    nodes: dict[int, dict[str, Any]] = {
        a: {"count": 3, "weight": 0.5, "flag": True, "dir": (1.0, 0.0), "at": (1, 2)},
        b: {"count": 4, "weight": 1.5, "flag": False, "dir": (0.0, 1.0)},
        c: {"count": None},
    }
    db.write_nodes(nodes, attributes=["count", "weight", "flag", "dir", "at"])
    assert CSRGraphDataBase.stored_extra_fields(tmp_path / "pred") == {
        "count": "int64",
        "weight": "float64",
        "flag": "bool",
        "dir": Vec("float64", 2),
        "at": Vec("int64", 2),
    }
    read = db.read_arrays()[0]
    assert by_id(read, "count")[a] == 3 and by_id(read, "weight")[b] == 1.5
    assert by_id(read, "dir")[b] == [0.0, 1.0] and by_id(read, "at")[a] == [1, 2]
    assert by_id(read, "count")[c] == 0

    # a type that the values do not settle is not guessed
    unclear: list[tuple[dict[int, dict[str, Any]], str]] = [
        ({a: {"x": 1}, b: {"x": 0.5}}, r"kinds \['f', 'i'\]"),
        ({a: {"x": (1, 2)}, b: {"x": (1, 2, 3)}}, "shapes"),
        ({a: {"x": 1}, b: {"x": (1, 2)}}, "shapes"),
        ({a: {"x": "word"}}, r"kinds \['U'\]"),
        ({a: {"x": [[1, 2], [3, 4]]}}, "shapes"),
    ]
    for given, match in unclear:
        with pytest.raises(ValueError, match=f"'x' is new.*{match}.*node_attrs"):
            db.write_nodes(given, attributes=["x"])
    assert "x" not in CSRGraphDataBase.stored_extra_fields(tmp_path / "pred")
    # no node carries a value (an empty block): nothing is written, nothing
    # is created, and nothing is raised, whoever writes first
    for nothing in ({}, {a: {"x": None}, b: {}}):
        db.write_nodes(nothing, attributes=["x"])
    db.write_attrs(nx.Graph(), node_attrs=["x"])
    assert "x" not in CSRGraphDataBase.stored_extra_fields(tmp_path / "pred")
    assert list((tmp_path / "pred" / "segments").glob("*.npys")) == []
    # left out, attributes are those that exist: a new key is not created
    db.write_nodes({a: {"count": 9, "y": 1}})
    assert "y" not in db.node_attrs and by_id(db.read_arrays()[0], "count")[a] == 9
    # flag is a bool: a number is not one
    for value in (1, 0.5):
        with pytest.raises(ValueError, match="'flag' is bool, one value"):
            db.write_nodes({a: {"flag": value}}, attributes=["flag"])
    db.write_nodes({a: {"flag": False, "weight": 2, "count": True}})
    read = db.read_arrays()[0]
    assert by_id(read, "flag")[a] is False and by_id(read, "weight")[a] == 2.0
    assert by_id(read, "count")[a] == 1


def test_extra_fields_of_another_graph_are_refused(tmp_path):
    ids, pos = make_graph(tmp_path / "g.csr")
    db = expanded(tmp_path, names=("label",))
    db.write_node_arrays(ids, label=(ids % 9).astype(np.int32))
    fields = tmp_path / "pred"

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
    nodes, edges = opened(tmp_path).read_arrays()
    assert nodes["label"].tolist() == (nodes["id"] % 9).tolist()
    assert len(edges["u"]) == N
    # a graph that changed under an open handle is noticed at the next write
    plain.write_node_arrays(np.array([10**9], dtype=np.uint64), position=pos[:1])
    with pytest.raises(ValueError, match=f"a graph of {N} nodes"):
        db.write_node_arrays(ids[:1], label=np.ones(1, np.int32))


def test_a_value_for_an_id_that_is_not_a_node_stops_the_finalize(tmp_path):
    ids, _ = make_graph(tmp_path / "g.csr")
    db = opened(tmp_path, "r+")
    db.write_node_arrays(ids[:5], label=np.full(5, 1, np.int32))
    stranger = np.array([ids[5], 10**12], dtype=np.uint64)
    with pytest.raises(ValueError, match=r"1 ids that are not nodes.*1000000000000"):
        db.write_node_arrays(stranger, label=np.full(2, 2, np.int32))
    # nothing was finalized: the arrays are as they were, the segment stays
    assert len(list((tmp_path / "pred" / "segments").glob("*.npys"))) == 1
    reader = opened(tmp_path)
    assert reader.read_arrays(node_attrs=["position"])[0]["id"].shape == (N,)
    with pytest.raises(UnfinalizedWrites):
        reader.read_arrays()
    for segment in (tmp_path / "pred" / "segments").glob("*.npys"):
        segment.unlink()
    assert int(reader.read_arrays()[0]["label"].sum()) == 5


def test_extra_fields_that_cannot_be_opened_are_refused(tmp_path):
    make_graph(tmp_path / "g.csr")
    make_graph(tmp_path / "h.csr")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "meta.json").write_text('{"version": 1}')
    graph, pred = tmp_path / "g.csr", tmp_path / "pred"
    before = {f: snapshot(tmp_path / f) for f in ("g.csr", "h.csr", "other")}
    refused: list[tuple[dict[str, Any], str]] = [
        ({"mode": "w", "cell_size": 1.0}, "mode 'w' creates the graph"),
        ({"extra_fields": graph}, "outside the graph's directory"),
        ({"extra_fields": [pred, tmp_path / "x" / ".." / "pred"]}, "is given twice"),
        # a directory that holds something else
        ({"extra_fields": tmp_path / "h.csr"}, "not a directory of extra node attr"),
        ({"extra_fields": tmp_path / "other"}, "not a directory of extra node attr"),
    ]
    for kwargs, match in refused:
        given: dict[str, Any] = {"mode": "r+", "extra_fields": pred, **kwargs}
        with pytest.raises(ValueError, match=match):
            CSRGraphDataBase(graph, **given)
    assert {f: snapshot(tmp_path / f) for f in before} == before
    assert not pred.exists()

    CSRGraphDataBase(
        tmp_path / "never.csr",
        mode="w",
        node_attrs={"position": Vec(float, 3)},
        cell_size=10.0,
    )
    with pytest.raises(ValueError, match="was never finalized"):
        CSRGraphDataBase(tmp_path / "never.csr", mode="r+", extra_fields=pred)


def test_an_attribute_held_twice_is_refused_when_the_graph_is_opened(tmp_path):
    ids, _ = make_graph(tmp_path / "g.csr")
    graph, a, b = tmp_path / "g.csr", tmp_path / "a", tmp_path / "b"
    for folder, value in ((a, 1), (b, 2)):
        db = CSRGraphDataBase(graph, mode="r+", extra_fields=folder)
        db.write_node_arrays(ids, label=np.full(N, value, np.int32))
    for mode in ("r", "r+"):
        with pytest.raises(ValueError, match="'label' is held by .*a and by .*b"):
            CSRGraphDataBase(graph, mode=mode, extra_fields=[a, b])
    # an attribute that the graph itself stores
    with pytest.raises(GraphIsReadOnly, match="cannot write the node attributes"):
        db.write_node_arrays(ids, position=np.zeros((N, 3)))
    meta = json.loads((a / "meta.json").read_text())
    meta["node_attrs"]["position"] = meta["node_attrs"]["label"]
    (a / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="'position' of .*a is also stored in .*g.csr"):
        CSRGraphDataBase(graph, extra_fields=a)


def test_several_directories_are_read_together_and_written_where_a_name_is(tmp_path):
    ids, _ = make_graph(tmp_path / "g.csr")
    graph, a, b = tmp_path / "g.csr", tmp_path / "model_a", tmp_path / "deep" / "b"
    one = CSRGraphDataBase(graph, mode="r+", extra_fields=a)
    with one.bulk_write_mode():
        one.write_node_arrays(ids, vec=vec_of(ids))
        one.write_node_arrays(ids, label=(ids % 9).astype(np.int32))
    CSRGraphDataBase(graph, mode="r+", extra_fields=b).write_node_arrays(
        ids, score=ids / 2
    )
    assert CSRGraphDataBase.stored_extra_fields(a) == {
        "vec": Vec("float32", 3),
        "label": "int32",
    }
    assert CSRGraphDataBase.stored_extra_fields(b) == {"score": "float64"}
    assert CSRGraphDataBase.stored_extra_fields(graph) == {}
    assert CSRGraphDataBase.stored_extra_fields(tmp_path / "missing") == {}
    assert not CSRGraphDataBase.is_database(a)

    both = CSRGraphDataBase(graph, mode="r+", extra_fields=[a, str(b)])
    assert list(both.node_attrs) == ["position", "vec", "label", "score"]
    nodes, _ = both.read_arrays()
    assert nodes["label"].tolist() == (nodes["id"] % 9).tolist()
    assert nodes["score"].tolist() == (nodes["id"] / 2).tolist()

    # a new attribute has no directory to go into when several are open
    with pytest.raises(ValueError, match=r"\['fresh'\] are new .* 2 extra directories"):
        both.write_node_arrays(ids, fresh=np.ones(N))
    with pytest.raises(ValueError, match=r"\['fresh'\] are new .* 2 extra directories"):
        both.write_nodes({int(ids[0]): {"fresh": 1}}, attributes=["fresh"])
    with pytest.raises(ValueError, match=r"\['fresh'\] are new .* 2 extra directories"):
        CSRGraphDataBase(
            graph, mode="r+", node_attrs={"fresh": int}, extra_fields=[a, b]
        )
    assert "fresh" not in CSRGraphDataBase.stored_extra_fields(a)
    assert "fresh" not in CSRGraphDataBase.stored_extra_fields(b)

    # attributes that exist are written into the directory that holds them;
    # pending writes in one directory do not stop a read of the other
    with both.bulk_write_mode(worker=True):
        both.write_node_arrays(
            ids[:1], score=np.array([0.5]), label=np.array([8], np.int32)
        )
        both.finalize()
        both.write_node_arrays(ids[:1], score=np.array([0.25]))
    assert (
        by_id(both.read_arrays(node_attrs=["vec", "label"])[0], "label")[int(ids[0])]
        == 8
    )
    with pytest.raises(UnfinalizedWrites, match="deep/b"):
        both.read_block(Roi((0, 0, 0), (100, 100, 100)))
    both.finalize()
    assert by_id(both.read_arrays()[0], "score")[int(ids[0])] == 0.25
    # an attribute another handle created since is found by the next write
    one.write_node_arrays(ids[:2], late=np.ones(2, np.float32))
    both.write_node_arrays(ids[2:4], late=np.full(2, 2, np.float32))
    late = by_id(CSRGraphDataBase(graph, extra_fields=[a, b]).read_arrays()[0], "late")
    assert [late[int(i)] for i in ids[:4]] == [1.0, 1.0, 2.0, 2.0]


def test_a_finalize_with_nothing_to_merge_does_nothing(tmp_path):
    """A main process that opens the graph and finalizes after workers that
    had nothing to write: the directory may not exist yet."""
    make_graph(tmp_path / "g.csr")
    db = opened(tmp_path, "r+")
    db.finalize()
    with db.bulk_write_mode():
        pass
    db.write_nodes({}, attributes=["label"])
    db.write_attrs(nx.Graph(), node_attrs=["label"])
    assert not (tmp_path / "pred").exists()
    # an error inside the block is the one that is raised
    with pytest.raises(KeyError, match="mine"):
        with db.bulk_write_mode():
            raise KeyError("mine")
    with pytest.raises(ValueError, match="'step'.* are not names of node attr"):
        with db.bulk_write_mode():
            db.write_node_arrays(np.zeros(0, np.uint64), step=np.zeros(0))
    assert not (tmp_path / "pred").exists()


def test_every_directory_is_finalized_and_the_errors_come_together(tmp_path, caplog):
    ids, _ = make_graph(tmp_path / "g.csr")
    graph = tmp_path / "g.csr"
    folders = [tmp_path / name for name in ("a", "b", "c")]
    for k, folder in enumerate(folders):
        CSRGraphDataBase(graph, mode="r+", extra_fields=folder).write_node_arrays(
            ids[:1], **{f"x{k}": np.ones(1)}
        )
    stranger = np.array([10**12], dtype=np.uint64)
    db = CSRGraphDataBase(graph, mode="r+", extra_fields=folders)
    with db.bulk_write_mode(worker=True):
        db.write_node_arrays(stranger, x0=np.ones(1))
        db.write_node_arrays(ids[1:2], x1=np.full(1, 2.0))
    # one directory fails: the others are merged all the same
    with pytest.raises(ValueError, match="ids that are not nodes.*nothing was final"):
        db.finalize()
    assert list((folders[1] / "segments").glob("*.npys")) == []
    assert len(list((folders[0] / "segments").glob("*.npys"))) == 1
    assert by_id(db.read_arrays(node_attrs=["x1"])[0], "x1")[int(ids[1])] == 2.0
    # two fail: both errors are raised, the third directory is merged
    with db.bulk_write_mode(worker=True):
        db.write_node_arrays(stranger, x1=np.ones(1))
        db.write_node_arrays(ids[2:3], x2=np.full(1, 3.0))
    with pytest.raises(ExceptionGroup, match="2 extra directories") as raised:
        db.finalize()
    assert [type(e) for e in raised.value.exceptions] == [ValueError, ValueError]
    assert by_id(db.read_arrays(node_attrs=["x2"])[0], "x2")[int(ids[2])] == 3.0
    # the error of the block is not replaced by that of the finalize
    with pytest.raises(KeyError, match="mine"):
        with db.bulk_write_mode():
            raise KeyError("mine")
    assert "finalize of" in caplog.text


def test_an_array_that_no_record_names_is_removed_by_the_next_finalize(tmp_path):
    """What a finalize leaves that was killed after it saved an array and
    before it recorded it."""
    ids, _ = make_graph(tmp_path / "g.csr")
    db = opened(tmp_path, "r+")
    db.write_node_arrays(ids[:2], label=np.ones(2, np.int32))
    pred = tmp_path / "pred"
    (kept,) = pred.glob("node_label-*.npy")
    np.save(pred / "node_label-0123456789ab.npy", np.zeros(N, np.int32))
    np.save(pred / "node_gone-0123456789ab.npy", np.zeros(N))
    (pred / "notes.npy").write_bytes(b"not ours")
    assert by_id(opened(tmp_path).read_arrays()[0], "label")[int(ids[0])] == 1
    db.finalize()  # nothing to merge: nothing is touched
    assert len(list(pred.glob("node_*.npy"))) == 3
    db.write_node_arrays(ids[2:3], label=np.ones(1, np.int32))
    (now,) = pred.glob("node_*.npy")
    assert now.name.startswith("node_label-") and now != kept
    assert (pred / "notes.npy").exists()
    assert int(opened(tmp_path).read_arrays()[0]["label"].sum()) == 3
