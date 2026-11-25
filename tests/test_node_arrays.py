"""Node attributes stored in arrays beside the graph they were computed on."""

import json
import multiprocessing as mp

import numpy as np
import pytest

from funlib.persistence.graphs import (
    NodeArrays,
    NodeArraysWriter,
    finalize_node_arrays,
    write_node_arrays,
)

IDS = np.array([40, 7, 19, 3, 88], dtype=np.uint64)
VECTORS = np.arange(15, dtype=np.float32).reshape(5, 3)
SCORES = np.array([0.5, 0.25, 0.75, 1.0, 0.0], dtype=np.float32)


def stored(tmp_path, **kwargs):
    return NodeArrays(
        write_node_arrays(
            tmp_path / "out.nodes", IDS, {"vec": VECTORS, "score": SCORES}, **kwargs
        )
    )


def test_values_read_back_in_the_order_the_ids_are_asked_for(tmp_path):
    found, values = stored(tmp_path).read([3, 88, 40])
    assert found.tolist() == [True, True, True]
    assert values["vec"].tolist() == VECTORS[[3, 4, 0]].tolist()
    assert values["score"].tolist() == [1.0, 0.0, 0.5]


def test_a_node_that_was_never_written_is_not_found_and_reads_as_nan(tmp_path):
    found, values = stored(tmp_path).read([7, 5, 1000], attrs=["score"])
    assert found.tolist() == [True, False, False]
    assert list(values) == ["score"]  # a read names the attributes it wants
    assert values["score"][0] == 0.25
    assert np.isnan(values["score"][1:]).all()


def test_an_unknown_attribute_is_refused_with_the_ones_stored(tmp_path):
    with pytest.raises(KeyError, match=r"no attribute 'nope'.*\['score', 'vec'\]"):
        stored(tmp_path).read([7], attrs=["nope"])


def test_the_store_says_what_it_holds_and_where_it_comes_from(tmp_path):
    arrays = stored(tmp_path, source="fragments")
    assert arrays.num_nodes == 5
    assert arrays.attrs == {"vec": 3, "score": None}
    assert arrays.source == "fragments"
    assert "graph_digest" not in arrays.meta
    assert arrays.ids.tolist() == IDS.tolist()


def test_the_layout_on_disk_is_plain_npy_files_and_a_meta_json(tmp_path):
    path = stored(tmp_path).path
    assert sorted(p.name for p in path.iterdir()) == [
        "ids.npy",
        "meta.json",
        "node_score.npy",
        "node_vec.npy",
        "order.npy",
    ]
    assert np.load(path / "order.npy").tolist() == np.argsort(IDS).tolist()
    assert np.load(path / "node_vec.npy").shape == (5, 3)
    assert json.loads((path / "meta.json").read_text())["attrs"] == {
        "vec": {"dtype": "<f4", "size": 3},
        "score": {"dtype": "<f4", "size": 0},
    }


def test_blocks_appended_one_by_one_read_as_one_store(tmp_path):
    path = tmp_path / "out.nodes"
    with NodeArraysWriter(path, {"vec": 3}) as out:
        out.append(IDS[:2], vec=VECTORS[:2])
        out.append(IDS[2:], vec=VECTORS[2:])
    found, values = NodeArrays(path).read(IDS)
    assert found.all()
    assert values["vec"].tolist() == VECTORS.tolist()


def test_a_store_with_no_nodes_finds_nothing(tmp_path):
    path = tmp_path / "out.nodes"
    with NodeArraysWriter(path, {"vec": 3}):
        pass
    found, values = NodeArrays(path).read([1, 2])
    assert found.tolist() == [False, False]
    assert values["vec"].shape == (2, 3)


def test_a_writer_that_fails_leaves_no_store(tmp_path):
    path = tmp_path / "out.nodes"
    with pytest.raises(RuntimeError, match="stop"):
        with NodeArraysWriter(path, {"vec": 3}) as out:
            out.append(IDS, vec=VECTORS)
            raise RuntimeError("stop")
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []  # and nothing beside it


def test_a_writer_that_fails_leaves_the_store_that_was_there(tmp_path):
    path = write_node_arrays(tmp_path / "out.nodes", IDS, {"vec": VECTORS})
    with pytest.raises(ValueError, match="written twice"):
        with NodeArraysWriter(path, {"vec": 3}) as out:
            out.append(IDS[:2], vec=VECTORS[:2])
            out.append(IDS[:1], vec=VECTORS[:1])
    assert NodeArrays(path).num_nodes == 5
    assert [p.name for p in tmp_path.iterdir()] == ["out.nodes"]


def test_a_second_write_replaces_the_store(tmp_path):
    path = tmp_path / "out.nodes"
    write_node_arrays(path, IDS, {"vec": VECTORS})
    write_node_arrays(path, IDS[:1], {"vec": VECTORS[:1]})
    assert NodeArrays(path).num_nodes == 1


def test_a_node_written_twice_is_refused(tmp_path):
    with pytest.raises(ValueError, match="written twice"):
        with NodeArraysWriter(tmp_path / "out.nodes", {"vec": 3}) as out:
            out.append(IDS, vec=VECTORS)
            out.append(IDS[:1], vec=VECTORS[:1])


@pytest.mark.parametrize(
    "values, match",
    [
        ({"vec": VECTORS[:, :2]}, r"'vec' has shape \(5, 2\), expected \(5, 3\)"),
        ({"vec": VECTORS, "extra": SCORES}, r"expected the attributes \['vec'\]"),
        ({}, r"expected the attributes \['vec'\]"),
    ],
)
def test_an_append_that_does_not_fit_the_declared_attributes_is_refused(
    tmp_path, values, match
):
    with pytest.raises(ValueError, match=match):
        with NodeArraysWriter(tmp_path / "out.nodes", {"vec": 3}) as out:
            out.append(IDS, **values)


def test_a_directory_that_is_not_a_store_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a node attribute store"):
        NodeArrays(tmp_path)


@pytest.mark.parametrize(
    "attrs, match",
    [
        ({"ids": None}, "'ids' is not an attribute name"),
        ({"vec": 0}, "'vec': a vector has a length of 1 or more"),
        ({"vec": -2}, "'vec': a vector has a length of 1 or more"),
        ({}, "at least one attribute"),
    ],
)
def test_a_declaration_the_store_cannot_hold_is_refused(tmp_path, attrs, match):
    with pytest.raises(ValueError, match=match):
        NodeArraysWriter(tmp_path / "out.nodes", attrs)
    assert list(tmp_path.iterdir()) == []


def test_values_in_memory_that_the_store_cannot_hold_are_refused(tmp_path):
    with pytest.raises(ValueError, match="'ids' is not an attribute name"):
        write_node_arrays(tmp_path / "out.nodes", [1, 2], {"ids": np.zeros(2)})
    with pytest.raises(ValueError, match="a vector has a length of 1 or more"):
        write_node_arrays(tmp_path / "out.nodes", [1, 2], {"s": np.zeros((2, 0))})


# -- several writers ---------------------------------------------------------


def test_two_writers_to_one_path_do_not_touch_each_others_files(tmp_path):
    """Both orders of leaving: the store is that of the writer that left
    last, whole, and no scratch directory stays."""
    path = tmp_path / "out.nodes"
    for first_out in ("a", "b"):
        a = NodeArraysWriter(path, {"vec": 3})
        b = NodeArraysWriter(path, {"vec": 3})
        a.__enter__()
        a.append(IDS[:3], vec=VECTORS[:3])
        b.__enter__()
        b.append(IDS[3:], vec=VECTORS[3:])
        first, second = (a, b) if first_out == "a" else (b, a)
        first.__exit__(None, None, None)
        assert NodeArrays(path).num_nodes == first.num_nodes
        second.__exit__(None, None, None)
        last = NodeArrays(path)
        wanted = IDS[:3] if second is a else IDS[3:]
        assert last.ids.tolist() == wanted.tolist()
        found, values = last.read(IDS)
        assert found.tolist() == np.isin(IDS, wanted).tolist()
        assert values["vec"][found].tolist() == VECTORS[found].tolist()
        assert [p.name for p in tmp_path.iterdir()] == ["out.nodes"]


BLOCK = 500


def _block(k):
    ids = np.arange(k * BLOCK, (k + 1) * BLOCK, dtype=np.uint64)[::-1]
    vec = np.stack([ids, ids * 2, ids * 3], axis=1).astype(np.float32)
    return ids, vec, (ids % 7).astype(np.int16)


def _worker(path, part, parts, blocks, start):
    start.wait()
    with NodeArraysWriter(
        path, {"vec": 3, "label": None}, dtypes={"label": "int16"}, worker=True
    ) as out:
        for k in range(part, blocks, parts):
            ids, vec, label = _block(k)
            out.append(ids, vec=vec, label=label)


def _write_in_processes(path, parts, blocks):
    ctx = mp.get_context("spawn")
    start = ctx.Barrier(parts)
    procs = [
        ctx.Process(target=_worker, args=(path, part, parts, blocks, start))
        for part in range(parts)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert [p.exitcode for p in procs] == [0] * parts


def test_several_processes_write_one_store(tmp_path):
    path = tmp_path / "out.nodes"
    parts, blocks = 4, 40
    _write_in_processes(path, parts, blocks)
    # nothing is at the path before the finalize; the segments are beside it
    assert not path.exists()
    assert len(list((tmp_path / "out.nodes.segments").glob("*.npys"))) == parts
    with pytest.raises(FileNotFoundError, match="not a node attribute store"):
        NodeArrays(path)

    assert finalize_node_arrays(path, source="four workers") == path
    store = NodeArrays(path)
    assert store.num_nodes == blocks * BLOCK and store.source == "four workers"
    assert store.attrs == {"vec": 3, "label": None}
    assert store.array("label").dtype == np.int16
    ids = np.arange(blocks * BLOCK + 3, dtype=np.uint64)
    found, values = store.read(ids)
    assert found.tolist() == [True] * (blocks * BLOCK) + [False] * 3
    known = ids[found]
    assert values["vec"][found].tolist() == (
        np.stack([known, known * 2, known * 3], axis=1).astype(np.float32).tolist()
    )
    assert values["label"][found].tolist() == (known % 7).tolist()
    # a block stays together on disk, in the order it was appended
    stored = np.asarray(store.ids).reshape(blocks, BLOCK)
    assert (np.diff(stored.astype(np.int64), axis=1) == -1).all()
    # nothing is left beside the store
    assert [p.name for p in tmp_path.iterdir()] == ["out.nodes"]
    assert sorted(p.name for p in path.iterdir()) == [
        "ids.npy",
        "meta.json",
        "node_label.npy",
        "node_vec.npy",
        "order.npy",
    ]


def _finalizer(path):
    finalize_node_arrays(path)


def test_finalizes_at_once_wait_for_each_other(tmp_path):
    path = tmp_path / "out.nodes"
    _write_in_processes(path, 2, 20)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_finalizer, args=(path,)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert [p.exitcode for p in procs] == [0] * 4
    store = NodeArrays(path)
    assert store.num_nodes == 20 * BLOCK
    assert len(np.unique(store.ids)) == 20 * BLOCK
    assert [p.name for p in tmp_path.iterdir()] == ["out.nodes"]


def worker(path, **kwargs):
    return NodeArraysWriter(path, {"vec": 3}, worker=True, **kwargs)


def test_a_finalize_adds_to_the_store_that_is_there(tmp_path):
    """A run that is taken up again: the nodes of the first finalize stay."""
    path = tmp_path / "out.nodes"
    with worker(path) as out:
        out.append(IDS[:2], vec=VECTORS[:2])
    finalize_node_arrays(path, source="run")
    assert NodeArrays(path).ids.tolist() == IDS[:2].tolist()

    before = NodeArrays(path)
    with worker(path) as out:
        out.append(IDS[2:4], vec=VECTORS[2:4])
        # until the finalize, a reader sees the store of the last one
        assert NodeArrays(path).num_nodes == 2
    with worker(path) as out:
        out.append(IDS[4:], vec=VECTORS[4:])
    finalize_node_arrays(path)
    store = NodeArrays(path)
    assert store.ids.tolist() == IDS.tolist() and store.source == "run"
    found, values = store.read(IDS)
    assert found.all() and values["vec"].tolist() == VECTORS.tolist()
    # a reader opened before reads what it opened
    assert before.read(IDS)[0].tolist() == [True, True, False, False, False]

    # with nothing new, the store is left as it is
    stamp = (path / "meta.json").stat().st_mtime_ns
    finalize_node_arrays(path, source="ignored")
    assert (path / "meta.json").stat().st_mtime_ns == stamp
    assert [p.name for p in tmp_path.iterdir()] == ["out.nodes"]


def test_a_worker_that_fails_keeps_the_blocks_it_had_written(tmp_path):
    path = tmp_path / "out.nodes"
    with pytest.raises(RuntimeError, match="stop"):
        with worker(path) as out:
            out.append(IDS[:2], vec=VECTORS[:2])
            raise RuntimeError("stop")
    finalize_node_arrays(path)
    assert NodeArrays(path).ids.tolist() == IDS[:2].tolist()


@pytest.mark.parametrize("cut", [10, 3 * 20 + 30], ids=["rows", "header"])
def test_a_record_cut_short_is_left_out(tmp_path, caplog, cut):
    """What a writer killed inside an append leaves: the last record lacks
    some of its rows (3 rows of 20 bytes), or part of its header."""
    path = tmp_path / "out.nodes"
    with worker(path) as out:
        out.append(IDS[:2], vec=VECTORS[:2])
        out.append(IDS[2:], vec=VECTORS[2:])
    (segment,) = (tmp_path / "out.nodes.segments").glob("*.npys")
    segment.write_bytes(segment.read_bytes()[:-cut])
    finalize_node_arrays(path)
    assert "truncated record" in caplog.text
    assert NodeArrays(path).ids.tolist() == IDS[:2].tolist()


def test_a_node_written_by_two_workers_is_refused_or_keeps_its_last_value(tmp_path):
    path = write_node_arrays(tmp_path / "out.nodes", IDS[:1], {"vec": VECTORS[:1]})
    with worker(path) as out:
        out.append(IDS[:3], vec=VECTORS[:3] + 100)
    with worker(path) as out:
        out.append(IDS[1:], vec=VECTORS[1:] + 200)
        out.append(IDS[4:], vec=VECTORS[4:] + 300)
    with pytest.raises(ValueError, match="written twice"):
        finalize_node_arrays(path)
    # refused: the store and the segments are as they were
    assert NodeArrays(path).ids.tolist() == IDS[:1].tolist()
    assert len(list((tmp_path / "out.nodes.segments").glob("*.npys"))) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "out.nodes",
        "out.nodes.segments",
    ]

    finalize_node_arrays(path, duplicates="last")
    store = NodeArrays(path)
    assert sorted(store.ids.tolist()) == sorted(IDS.tolist())
    found, values = store.read(IDS)
    assert found.all()
    assert (values["vec"] - VECTORS)[:, 0].tolist() == [100, 200, 200, 200, 300]
    assert np.load(path / "order.npy").tolist() == np.argsort(store.ids).tolist()
    assert sorted(p.name for p in path.iterdir()) == [
        "ids.npy",
        "meta.json",
        "node_vec.npy",
        "order.npy",
    ]
    with pytest.raises(ValueError, match="duplicates is 'first'"):
        finalize_node_arrays(path, duplicates="first")


def test_writers_that_declare_other_attributes_are_refused(tmp_path):
    path = tmp_path / "out.nodes"
    with worker(path) as out:
        out.append(IDS[:2], vec=VECTORS[:2])
    with NodeArraysWriter(path, {"vec": 2}, worker=True) as out:
        out.append(IDS[2:], vec=VECTORS[2:, :2])
    with pytest.raises(ValueError, match="was written with the attributes"):
        finalize_node_arrays(path)
    assert not path.exists()

    # against a store that is there, the writer is refused when it opens
    path = write_node_arrays(tmp_path / "other.nodes", IDS, {"vec": VECTORS})
    with pytest.raises(ValueError, match="holds the attributes"):
        with NodeArraysWriter(path, {"vec": 3}, dtypes={"vec": "f8"}, worker=True):
            pass
    with pytest.raises(ValueError, match="give it to finalize_node_arrays"):
        worker(path, source="mine")


def test_workers_with_no_node_to_write_leave_an_empty_store(tmp_path):
    path = tmp_path / "out.nodes"
    with pytest.raises(FileNotFoundError, match="nothing was written"):
        finalize_node_arrays(path)
    with worker(path) as out:
        out.append(IDS[:0], vec=VECTORS[:0])
    finalize_node_arrays(path)
    store = NodeArrays(path)
    assert store.num_nodes == 0 and store.attrs == {"vec": 3}
    assert store.read([1])[0].tolist() == [False]
