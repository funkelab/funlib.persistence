"""Node attributes stored in arrays beside the graph they were computed on."""

import json

import numpy as np
import pytest

from funlib.persistence.graphs import NodeArrays, NodeArraysWriter, write_node_arrays

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
    assert values["score"][0] == 0.25
    assert np.isnan(values["score"][1:]).all()


def test_a_read_names_the_attributes_it_wants(tmp_path):
    _, values = stored(tmp_path).read([7], attrs=["vec"])
    assert list(values) == ["vec"]


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
