from typing import Any

import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi
from zarr.codecs import ZstdCodec

from funlib.persistence.arrays import Array
from funlib.persistence.arrays.datasets import open_ds, prepare_ds
from funlib.persistence.arrays.metadata import MetaDataFormat

stores = {
    "zarr": "test_array.zarr",
    "n5": "test_array.n5",
    "zarr_ds": "test_array.zarr/test_group/test_data",
    "n5_ds": "test_array.n5/test_group/test_data",
    "zipped_zarr": "test_array.zarr.zip",
    "zipped_zarr_ds": "test_array.zarr.zip/test_group/test_data",
}


@pytest.mark.parametrize("store", stores.keys())
def test_metadata(tmp_path, store):
    store = tmp_path / store

    # test prepare_ds creates array if it does not exist and mode is write
    array = prepare_ds(
        store,
        (10, 10),
        mode="w",
        custom_metadata={"custom": "metadata"},
    )
    assert array.attrs["custom"] == "metadata"
    array.attrs["custom2"] = "new metadata"

    assert open_ds(store).attrs["custom2"] == "new metadata"


# every store with one dtype, every dtype with one store
STORES_AND_DTYPES = [(store, np.float32) for store in stores] + [
    ("zarr", np.uint8),
    ("zarr", np.uint64),
]


def _assert_placed(array, metadata, shape, voxel_size=None):
    """The array carries `metadata`, for an array of `shape` (and `voxel_size`)."""
    voxel_size = metadata.voxel_size if voxel_size is None else voxel_size
    assert array.roi == Roi(metadata.offset, voxel_size * Coordinate(*shape[-3:]))
    assert array.voxel_size == voxel_size
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units


@pytest.mark.parametrize("store, dtype", STORES_AND_DTYPES)
def test_helpers(tmp_path, store, dtype):
    shape = Coordinate(1, 1, 10, 20, 30)
    chunk_shape = Coordinate(2, 3, 10, 10, 10)
    store = tmp_path / store
    metadata = MetaDataFormat().parse(
        shape,
        {
            "offset": [100, 200, 400],
            "voxel_size": [1, 2, 3],
            "axis_names": ["sample^", "channel^", "t", "y", "x"],
            "units": ["nm", "nm", "nm"],
            "types": ["sample", "channel", "time", "space", "space"],
        },
    )

    def prepare(shape, mode, voxel_size=metadata.voxel_size):
        return prepare_ds(
            store,
            shape,
            metadata.offset,
            voxel_size,
            metadata.axis_names,
            metadata.units,
            metadata.types,
            chunk_shape=chunk_shape,
            dtype=dtype,
            mode=mode,
        )

    def assert_placed(array, shape, voxel_size=metadata.voxel_size):
        _assert_placed(array, metadata, shape, voxel_size)
        assert array.types == metadata.types

    # test prepare_ds fails if array does not exist and mode is read
    with pytest.raises(FileNotFoundError):
        prepare(shape, "r")

    # test prepare_ds creates array if it does not exist and mode is write
    assert_placed(prepare(shape, "w"), shape)

    # test prepare_ds opens array if it exists and mode is read
    assert_placed(prepare(shape, "r"), shape)

    # test prepare_ds fails if array exists and is opened in read mode
    # with incompatible arguments
    with pytest.raises(PermissionError):
        prepare(chunk_shape, "r")

    # test prepare_ds overwrite existing array in write mode
    assert_placed(prepare(chunk_shape, "w"), chunk_shape)

    # test prepare_ds updates metadata existing array in "r+" or "a" mode if it exists
    assert_placed(
        prepare(chunk_shape, "r+", metadata.voxel_size * 2),
        chunk_shape,
        metadata.voxel_size * 2,
    )

    # test prepare_ds updates existing array in "r+" or "a" mode if it exists
    array = prepare(chunk_shape, "a")
    assert_placed(array, chunk_shape)

    # test prepare_ds with mode "w" overwrites existing array even if compatible
    array[:] = 2
    assert np.all(np.isclose(array[:], 2))
    array = prepare(chunk_shape, "w")
    assert np.all(np.isclose(array[:], 0))
    assert_placed(array, chunk_shape)


@pytest.mark.parametrize("store, dtype", STORES_AND_DTYPES)
def test_open_ds(tmp_path, store, dtype):
    shape = Coordinate(1, 1, 10, 20, 30)
    store = tmp_path / store
    metadata = MetaDataFormat().parse(
        shape,
        {
            "offset": [100, 200, 400],
            "voxel_size": [1, 2, 3],
            "axis_names": ("sample^", "channel^", "z", "y", "x"),
            "units": ("nm", "nm", "nm"),
            "types": ["sample", "channel", "space", "space", "space"],
        },
    )
    placement: dict[str, Any] = dict(
        offset=metadata.offset,
        voxel_size=metadata.voxel_size,
        axis_names=metadata.axis_names,
        units=metadata.units,
    )

    # test open_ds fails if array does not exist and mode is read
    with pytest.raises(FileNotFoundError):
        open_ds(store, types=metadata.types, mode="r", **placement)

    # test open_ds opens what prepare_ds created
    created = prepare_ds(
        store, shape, types=metadata.types, dtype=dtype, mode="w", **placement
    )
    _assert_placed(created, metadata, shape)
    _assert_placed(open_ds(store, mode="r", **placement), metadata, shape)

    # test open_ds arguments take the place of the stored metadata
    array = open_ds(
        store,
        offset=(1, 2, 3),
        voxel_size=(1, 2, 3),
        axis_names=None,
        units=None,
    )
    assert array.offset == Coordinate(1, 2, 3)
    assert array.voxel_size == Coordinate(1, 2, 3)
    assert array.roi == Roi((1, 2, 3), (10, 40, 90))


# --- sharding: chunk_shape is the WRITE unit (the shard), inner_chunk_shape the read unit -------

SHARD = Coordinate(2, 64, 32)  # what a writer covers whole; the stored object
INNER = Coordinate(2, 16, 16)  # what a reader can fetch on its own


def _sharded(store, mode="w", **over):
    kw: dict[str, Any] = dict(
        shape=(2, 64, 64),
        voxel_size=Coordinate(1, 1),
        axis_names=["c^", "y", "x"],
        chunk_shape=SHARD,
        sharding=True,
        inner_chunk_shape=INNER,
        dtype=np.uint8,
        mode=mode,
    )
    kw.update(over)
    return prepare_ds(store, **kw)


SHARDED_STORES = ["zarr", "zarr_ds", "zipped_zarr"]


def _sharded_with_data(store):
    """A sharded array at `store` holding a ramp; returns the ramp."""
    array = _sharded(store)
    data = np.arange(np.prod(array.shape), dtype=np.uint8).reshape(array.shape)
    array[array.roi] = data
    return data


@pytest.mark.parametrize("store", SHARDED_STORES)
@pytest.mark.parametrize("mode", ["r", "r+"])
def test_sharded_read_modes_raise_where_nothing_is(tmp_path, store, mode):
    with pytest.raises(FileNotFoundError):
        _sharded(tmp_path / stores[store], mode=mode)


@pytest.mark.parametrize("store", SHARDED_STORES)
def test_sharded(tmp_path, store):
    store = tmp_path / stores[store]

    array = _sharded(store)
    # chunk_shape IS the shard: the parallel-safe write unit, as on an unsharded array
    assert array.chunk_shape == SHARD
    assert array.shard_shape == SHARD
    assert array.inner_chunk_shape == INNER
    meta = array._source_data.metadata.to_dict()
    assert meta["codecs"][0]["name"] == "sharding_indexed"
    assert tuple(meta["codecs"][0]["configuration"]["chunk_shape"]) == tuple(INNER)
    assert tuple(meta["chunk_grid"]["configuration"]["chunk_shape"]) == tuple(SHARD)
    assert array.voxel_size == Coordinate(1, 1)

    data = np.arange(np.prod(array.shape), dtype=np.uint8).reshape(array.shape)
    array[array.roi] = data
    reopened = open_ds(store)
    assert (reopened.chunk_shape, reopened.inner_chunk_shape) == (SHARD, INNER)
    np.testing.assert_array_equal(reopened[reopened.roi], data)


@pytest.mark.parametrize(
    "over",
    [
        {},
        {"sharding": False, "inner_chunk_shape": None},
        {"inner_chunk_shape": None},
    ],
    ids=["same-layout", "sharding-unsaid", "inner-chunk-unsaid"],
)
def test_sharded_is_reused_by_a_caller_with_the_same_write_unit(tmp_path, over):
    """`sharding` is a creation option only, and no inner chunk means "don't care"."""
    store = tmp_path / stores["zarr"]
    data = _sharded_with_data(store)
    same = _sharded(store, mode="a", **over)
    np.testing.assert_array_equal(same[same.roi], data)
    assert same.chunk_shape == SHARD


@pytest.mark.parametrize(
    "over",
    [{"chunk_shape": (2, 32, 32)}, {"inner_chunk_shape": (2, 8, 8)}],
    ids=["other-write-unit", "other-inner-chunk"],
)
@pytest.mark.parametrize("mode", ["r", "r+", "a"])
def test_sharded_with_another_layout_is_refused_and_keeps_its_data(
    tmp_path, over, mode
):
    store = tmp_path / stores["zarr"]
    data = _sharded_with_data(store)
    with pytest.raises(PermissionError):
        _sharded(store, mode=mode, **over)
    reopened = open_ds(store)
    np.testing.assert_array_equal(reopened[reopened.roi], data)


@pytest.mark.parametrize("store", SHARDED_STORES)
def test_sharded_w_recreates_with_another_write_unit(tmp_path, store):
    store = tmp_path / stores[store]
    _sharded_with_data(store)
    recreated = _sharded(store, mode="w", chunk_shape=Coordinate(2, 32, 32))
    assert (recreated.chunk_shape, recreated.inner_chunk_shape) == (
        Coordinate(2, 32, 32),
        INNER,
    )
    np.testing.assert_array_equal(recreated[recreated.roi], 0)


def test_unsharded_is_reused_by_a_sharding_caller_with_the_same_write_unit(tmp_path):
    """`sharding` is a creation option; an existing array is judged by its write unit."""
    store = tmp_path / stores["zarr"]
    plain = prepare_ds(store, (8, 8), chunk_shape=Coordinate(4, 4), mode="w")
    assert (
        plain.shard_shape is None
        and plain.chunk_shape == plain.inner_chunk_shape == (4, 4)
    )
    again = prepare_ds(
        store, (8, 8), chunk_shape=Coordinate(4, 4), sharding=True, mode="a"
    )
    assert again.shard_shape is None, "reused as it is; nothing recreated"
    with pytest.raises(PermissionError):
        prepare_ds(store, (8, 8), chunk_shape=Coordinate(8, 8), sharding=True, mode="a")


@pytest.mark.parametrize(
    "shard, inner",
    [
        ((2, 256, 128), (2, 64, 64)),
        ((2, 32, 32), (2, 32, 32)),
        ((3, 100, 96, 200), (3, 50, 48, 50)),
        ((1, 128, 1000, 4096), (1, 64, 50, 64)),
        ((2, 127, 134, 70), (2, 127, 134, 35)),
    ],
)
def test_inner_chunk_defaults(tmp_path, shard, inner):
    """The largest divisor of each shard axis that is at most 64; the whole axis where the
    shard is smaller, or where no divisor comes near 64."""
    array = prepare_ds(
        tmp_path / "a.zarr",
        shard,
        chunk_shape=Coordinate(shard),
        sharding=True,
        dtype=np.uint8,
        mode="w",
    )
    assert array.inner_chunk_shape == Coordinate(inner)
    assert array.chunk_shape == Coordinate(shard)


@pytest.mark.parametrize(
    "options, error, match",
    [
        (
            dict(inner_chunk_shape=Coordinate(4, 4)),
            ValueError,
            "only applies with sharding=True",
        ),
        (dict(chunk_shape=None, sharding=True), ValueError, "needs a chunk_shape"),
        (
            dict(sharding=True, inner_chunk_shape=Coordinate(3, 3)),
            ValueError,
            "whole number of inner_chunk_shape",
        ),
        (
            dict(sharding=True, inner_chunk_shape=Coordinate(4)),
            ValueError,
            "differ in dimensions",
        ),
        (
            dict(sharding=True, inner_chunk_shape=Coordinate(0, 8)),
            ValueError,
            "positive per axis",
        ),
        (
            dict(sharding=True, zarr_format=2),  # sharding is zarr v3 only
            ValueError,
            None,  # zarr's own refusal
        ),
        (dict(shard_shape=(8, 8)), TypeError, "has no shard_shape"),
    ],
    ids=[
        "inner-chunk-without-sharding",
        "sharding-without-chunk-shape",
        "inner-chunk-does-not-divide",
        "inner-chunk-dims",
        "inner-chunk-zero",
        "zarr-v2",
        "shard-shape-kwarg",
    ],
)
def test_sharding_option_refusals(tmp_path, options, error, match):
    kw: dict[str, Any] = dict(chunk_shape=Coordinate(8, 8), mode="w")
    kw.update(options)
    with pytest.raises(error, match=match):
        prepare_ds(tmp_path / stores["zarr"], (8, 8), **kw)
    assert list(tmp_path.iterdir()) == [], "refused before the location was touched"


@pytest.mark.parametrize("mode", ["a", "r+"])
def test_sharded_relabel_reopens_instead_of_recreating(tmp_path, mode):
    """Data-compatible but metadata-incompatible is a RELABEL of the existing array, on the
    sharded path as on the unsharded one."""
    store = tmp_path / stores["zarr"]
    array = _sharded(store)
    array[:] = 9
    relabeled = _sharded(store, mode=mode, voxel_size=Coordinate(4, 8))
    assert relabeled.voxel_size == Coordinate(4, 8)
    assert (relabeled.chunk_shape, relabeled.inner_chunk_shape) == (SHARD, INNER)
    np.testing.assert_array_equal(relabeled[relabeled.roi], 9)
    assert open_ds(store).voxel_size == Coordinate(4, 8)


def test_sharded_w_minus_refuses_an_existing_array_and_keeps_its_data(tmp_path):
    store = tmp_path / stores["zarr"]
    _sharded(store)[:] = 9
    with pytest.raises(FileExistsError):
        _sharded(store, mode="w-", voxel_size=Coordinate(4, 8))
    reopened = open_ds(store)
    np.testing.assert_array_equal(reopened[reopened.roi], 9)
    assert reopened.voxel_size == Coordinate(1, 1)


def test_sharded_default_mode_creates_where_nothing_is(tmp_path):
    store = tmp_path / stores["zarr"]
    array = prepare_ds(
        store, (2, 64, 64), chunk_shape=SHARD, sharding=True, dtype=np.uint8
    )
    assert array.shard_shape == SHARD
    assert open_ds(store).shard_shape == SHARD


def test_sharded_a_creates_beside_other_files_and_w_minus_refuses_them(tmp_path):
    """ "a" leaves whatever else the location holds (as the legacy path); "w-" refuses a
    non-empty location outright, stray files and orphaned shards included."""
    store = tmp_path / stores["zarr"]
    store.mkdir()
    (store / "README").write_text("not an array")
    with pytest.raises(FileExistsError):
        _sharded(store, mode="w-")
    assert [p.name for p in store.iterdir()] == ["README"]

    _sharded(store, mode="a")
    assert (store / "README").exists(), "mode 'a' must not wipe the location"
    assert open_ds(store).shard_shape == SHARD

    open_ds(store, mode="r+")[:] = 7
    (
        store / "zarr.json"
    ).unlink()  # orphaned shards must not come back as data under "w-"
    with pytest.raises(FileExistsError):
        _sharded(store, mode="w-")
    recreated = _sharded(store, mode="w")
    np.testing.assert_array_equal(
        recreated[recreated.roi], 0, "mode 'w' starts from nothing"
    )


@pytest.mark.parametrize(
    "key, shard, inner",
    [
        (np.s_[0], (64, 32), (16, 16)),
        (np.s_[:, 3], (2, 32), (2, 16)),
        (np.s_[:, :, 5], (2, 64), (2, 16)),
        (np.s_[:, :48], (2, 64, 32), (2, 16, 16)),
        (np.s_[1, 0:32], (64, 32), (16, 16)),
        # two or more ints: the dimensions THEY index collapse
        (np.s_[0, 3], (32,), (16,)),
        (np.s_[0, :, 3], (64,), (16,)),
        (np.s_[:, 3, 5], (2,), (2,)),
        (Roi((0, 32), (32, 32)), (2, 64, 32), (2, 16, 16)),
    ],
    ids=[
        "[0]",
        "[:, 3]",
        "[:, :, 5]",
        "[:, :48]",
        "[1, 0:32]",
        "[0, 3]",
        "[0, :, 3]",
        "[:, 3, 5]",
        "roi-on-a-shard-boundary",
    ],
)
def test_chunk_shapes_follow_collapsed_and_truncated_dims(tmp_path, key, shard, inner):
    """Dropping or truncating an axis keeps the view on the stored chunk grid, so the
    shapes follow with that axis gone."""
    store = tmp_path / stores["zarr"]
    _sharded(store)
    array = open_ds(store)
    array.lazy_op(key)
    assert array.chunk_shape == array.shard_shape == Coordinate(*shard)
    assert array.inner_chunk_shape == Coordinate(*inner)
    assert array.is_writeable is True


@pytest.mark.parametrize(
    "key, writeable",
    [
        (np.s_[:, 8:], True),
        (np.s_[:, ::2], True),
        (np.s_[[0, 1]], True),
        (Roi((0, 8), (32, 32)), True),
        (np.array([True, False]), False),
        (lambda d: d.sum(axis=0), False),
    ],
    ids=[
        "a-start",
        "a-step",
        "a-selection-list",
        "roi-inside-a-shard",
        "a-mask",
        "a-callable",
    ],
)
def test_chunk_shapes_are_none_off_the_stored_grid(tmp_path, key, writeable):
    """A view that a lazy op moved off the grid, or that cannot be written, reports no
    chunk shape at all rather than a grid its own blocks do not tile."""
    store = tmp_path / stores["zarr"]
    _sharded(store)
    array = open_ds(store)
    array.lazy_op(key)
    assert array.is_writeable is writeable
    assert array.chunk_shape is None
    assert array.inner_chunk_shape is None
    assert array.shard_shape is None


def test_chunk_shapes_of_a_plain_array_follow_the_same_rule(tmp_path):
    plain = prepare_ds(
        tmp_path / "plain.zarr", (8, 8), chunk_shape=Coordinate(4, 4), mode="w"
    )
    assert plain.chunk_shape == plain.inner_chunk_shape == Coordinate(4, 4)
    assert plain.shard_shape is None
    plain.lazy_op(np.s_[2:])
    assert plain.chunk_shape is None and plain.inner_chunk_shape is None


def _copy_layout(a, store):
    """The documented recipe, for a sharded and an unsharded source alike."""
    return prepare_ds(
        store,
        a.shape,
        chunk_shape=a.chunk_shape,
        sharding=a.shard_shape is not None,
        inner_chunk_shape=a.inner_chunk_shape,
        dtype=a.dtype,
        mode="w",
    )


def test_copying_a_layout(tmp_path):
    b = _copy_layout(_sharded(tmp_path / stores["zarr"]), tmp_path / "copy.zarr")
    assert (b.chunk_shape, b.shard_shape, b.inner_chunk_shape) == (SHARD, SHARD, INNER)

    plain = prepare_ds(
        tmp_path / "plain.zarr", (128, 128), chunk_shape=Coordinate(64, 64), mode="w"
    )
    c = _copy_layout(plain, tmp_path / "plain_copy.zarr")
    assert (c.chunk_shape, c.shard_shape, c.inner_chunk_shape) == (
        (64, 64),
        None,
        (64, 64),
    )
    # the recipe against the plain source ITSELF reuses it (the restated inner chunk matches)
    plain[:] = 2
    again = prepare_ds(
        tmp_path / "plain.zarr",
        (128, 128),
        chunk_shape=plain.chunk_shape,
        inner_chunk_shape=plain.inner_chunk_shape,
        mode="a",
    )
    np.testing.assert_array_equal(again[again.roi], 2)


@pytest.mark.parametrize("mode", ["a", "r+"])
def test_a_sharding_caller_relabels_a_plain_array_in_place(tmp_path, mode):
    """`sharding` is a creation option: against an existing PLAIN array with the same write unit
    and other metadata, the array is reopened and relabelled, not recreated and not refused."""
    store = tmp_path / stores["zarr"]
    plain = prepare_ds(
        store,
        (2, 64, 64),
        chunk_shape=SHARD,
        voxel_size=Coordinate(1, 1),
        axis_names=["c^", "y", "x"],
        dtype=np.uint8,
        mode="w",
    )
    plain[:] = 6
    # no explicit inner chunk: one that the plain array does not have is a data incompatibility
    relabeled = _sharded(
        store, mode=mode, voxel_size=Coordinate(4, 8), inner_chunk_shape=None
    )
    assert relabeled.voxel_size == Coordinate(4, 8)
    assert relabeled.shard_shape is None, "reused as it is"
    with pytest.raises(PermissionError):
        _sharded(
            store, mode=mode, voxel_size=Coordinate(4, 8)
        )  # asks for inner chunks it has not got
    assert relabeled.chunk_shape == SHARD
    np.testing.assert_array_equal(relabeled[relabeled.roi], 6)


def test_sharded_w_minus_creates_on_an_empty_location_with_creation_kwargs(tmp_path):
    store = tmp_path / stores["zarr"]
    array = _sharded(store, mode="w-", compressors=[ZstdCodec(level=3)])
    array[:] = 4
    reopened = open_ds(store)
    assert reopened.shard_shape == SHARD
    np.testing.assert_array_equal(reopened[reopened.roi], 4)
    inner_codecs = reopened._source_data.metadata.to_dict()["codecs"][0][
        "configuration"
    ]["codecs"]
    assert any(c["name"] == "zstd" for c in inner_codecs), inner_codecs


def test_sharded_name_and_path_both_place_the_array(tmp_path):
    """`name=` (create_array's spelling) and `path=` (the legacy spelling) both place the
    array, on the create, reuse and relabel paths; "w-" checks emptiness at THAT location."""
    nested = tmp_path / "nested.zarr"
    sub = _sharded(nested, mode="w-", name="sub")
    assert sub.shard_shape == SHARD
    assert open_ds(nested / "sub").shard_shape == SHARD
    with pytest.raises(FileExistsError):
        _sharded(nested, mode="w-", path="sub")
    assert _sharded(nested, mode="w-", path="other").shard_shape == SHARD
    with pytest.raises(ValueError, match="different sub-locations"):
        _sharded(nested, mode="w", path="a", name="b")
    another = _sharded(nested, mode="w-", name="another")
    assert _sharded(nested, mode="a", name="another").roi == another.roi
    assert _sharded(
        nested, mode="a", name="another", voxel_size=Coordinate(4, 8)
    ).voxel_size == Coordinate(4, 8)


@pytest.mark.parametrize(
    "bad, error, match",
    [
        (
            dict(inner_chunk_shape=Coordinate(3, 16, 16)),
            ValueError,
            "whole number of inner_chunk_shape",
        ),
        (dict(compressors=ZstdCodec), TypeError, None),  # zarr's own refusal
        (dict(chunks=(2, 16, 16)), TypeError, "prepare_ds sets chunks itself"),
        (dict(overwrite=True), TypeError, "prepare_ds sets overwrite itself"),
        (dict(bogus=1), TypeError, "unexpected keyword argument 'bogus'"),
    ],
    ids=[
        "inner-chunk-does-not-divide",
        "codec-class-not-instance",
        "chunks-kwarg",
        "overwrite-kwarg",
        "unknown-kwarg",
    ],
)
def test_sharded_w_minus_leaves_nothing_behind_a_refused_create(
    tmp_path, bad, error, match
):
    """A create zarr refuses on its VALUES, or a kwarg prepare_ds owns, is found before the
    location is touched; the corrected retry then goes through."""
    store = tmp_path / stores["zarr"]
    with pytest.raises(error, match=match):
        _sharded(store, mode="w-", **bad)
    assert list(tmp_path.iterdir()) == [], "refused before the location was touched"
    assert _sharded(store, mode="w-").shard_shape == SHARD


@pytest.mark.parametrize("mode", ["w-", "w", "a"])
def test_creation_kwargs_do_not_break_the_existence_probe(tmp_path, mode):
    """An existing directory with no array in it, plus creation kwargs: the probe must say
    "no array here" and the create must proceed, in every creating mode."""
    store = tmp_path / stores["zarr"]
    store.mkdir()
    assert _sharded(store, mode=mode, compressors=[ZstdCodec()]).shard_shape == SHARD


@pytest.mark.parametrize("mode", ["w-", "w", "a"])
def test_a_stray_placeholder_group_does_not_block_a_sharded_create(tmp_path, mode):
    """An EMPTY group with no attrs at the location (a "w-" placeholder whose process died
    before the create) is never data and counts as nothing there."""
    import zarr

    store = tmp_path / stores["zarr"]
    zarr.open_group(str(store), mode="w-", zarr_format=3)
    assert _sharded(store, mode=mode).shard_shape == SHARD
    assert open_ds(store).shard_shape == SHARD


@pytest.mark.parametrize("mode", ["w-", "w", "a"])
@pytest.mark.parametrize("content", ["attrs", "member"])
def test_a_group_with_content_is_refused_and_kept(tmp_path, mode, content):
    """A group holding attrs or members is a real group, whatever the mode."""
    import zarr

    store = tmp_path / stores["zarr"]
    group = zarr.open_group(str(store), mode="w-", zarr_format=3)
    if content == "attrs":
        group.attrs["kept"] = True
    else:
        group.create_array("kept", shape=(2,), dtype="uint8")
    with pytest.raises(TypeError, match="got Group"):
        _sharded(store, mode=mode)
    kept = zarr.open_group(str(store), mode="r")
    assert (dict(kept.attrs), list(kept.keys())) == (
        ({"kept": True}, []) if content == "attrs" else ({}, ["kept"])
    )


@pytest.mark.parametrize("mode", ["w-", "w", "a"])
def test_unsharded_name_is_path(tmp_path, mode):
    store = tmp_path / stores["zarr"]
    array = prepare_ds(
        store, (8, 8), chunk_shape=Coordinate(4, 4), mode=mode, name="sub"
    )
    assert array.shape == (8, 8)
    assert open_ds(store / "sub").chunk_shape == Coordinate(4, 4)
    assert prepare_ds(store, (8, 8), mode="a", path="sub").shape == (8, 8)


def test_the_chunk_grid_is_history_and_is_set_only_once_an_op_applied(tmp_path):
    """Off the grid stays off, whatever follows; an op the array refuses leaves the grid
    as it was; a constructor-time op counts like any other."""
    store = tmp_path / stores["zarr"]
    _sharded(store)
    array = open_ds(store)
    array.lazy_op(np.s_[:, 8:])
    array.lazy_op(np.s_[0])  # on-grid by itself
    assert array.chunk_shape is None and array.inner_chunk_shape is None

    array = open_ds(store)
    with pytest.raises(TypeError, match="not a supported lazy_op"):
        array.lazy_op(None)  # ty: ignore[invalid-argument-type]
    assert array.chunk_shape == SHARD and array.lazy_ops == []

    source = open_ds(store)
    array = Array(
        source._source_data,
        source.offset,
        source.voxel_size,
        source.axis_names,
        source.units,
        source.types,
        chunks=source._source_data.chunks,
        lazy_op=np.s_[:, 8:],
    )
    assert array.chunk_shape is None


def test_a_roi_is_on_the_grid_by_the_arrays_own_offset_and_voxel_size(tmp_path):
    """The chunk boundary is in the array's voxels from ITS origin, not in world units
    from zero."""
    store = tmp_path / stores["zarr"]
    _sharded(store, offset=Coordinate(100, 200), voxel_size=Coordinate(2, 4))
    array = open_ds(store)
    array.lazy_op(
        Roi((100, 200 + 32 * 4), (64 * 2, 32 * 4))
    )  # voxel (0, 32): a shard boundary
    assert array.chunk_shape == array.shard_shape == SHARD
    array = open_ds(store)
    array.lazy_op(
        Roi((100, 200 + 8 * 4), (64 * 2, 32 * 4))
    )  # voxel (0, 8): inside a shard
    assert array.chunk_shape is None and array.shard_shape is None
    # the Roi checks apply_lazy_ops makes keep their own messages
    array = open_ds(store)
    with pytest.raises(AssertionError, match="same number of dimensions"):
        array.lazy_op(Roi((0,), (8,)))
    array = open_ds(store)
    array.lazy_op(np.s_[:, 0])
    with pytest.raises(AssertionError, match="collapsed dimensions"):
        array.lazy_op(Roi((100, 200), (8, 8)))


@pytest.mark.parametrize(
    "field, value",
    [
        ("types", ["channel", "space", "time"]),
        ("axis_names", ["c^", "y", "z"]),
        ("units", ["nm", "um"]),
    ],
)
def test_every_metadata_field_is_compared_and_relabelled(tmp_path, field, value):
    """A sharded array that differs from the request in one metadata field only is refused
    under 'r' and relabelled under 'a', as a plain one is."""
    store = tmp_path / stores["zarr"]
    _sharded(store)
    with pytest.raises(PermissionError, match="metadata is not compatible"):
        _sharded(store, mode="r", **{field: value})
    assert getattr(_sharded(store, mode="a", **{field: value}), field) == value
    assert getattr(open_ds(store), field) == value


def test_a_different_dtype_or_a_list_shape_against_a_sharded_array(tmp_path):
    store = tmp_path / stores["zarr"]
    _sharded(store)
    with pytest.raises(PermissionError, match="not compatible, but mode is not 'w'"):
        _sharded(store, mode="a", dtype=np.float32)
    # a list or array shape describes the same array as the tuple does
    assert _sharded(store, mode="a", shape=[2, 64, 64]).shard_shape == SHARD
    assert _sharded(store, mode="a", shape=np.array([2, 64, 64])).shard_shape == SHARD


def test_incompatible_dataset_error_names_the_store_and_each_difference(tmp_path):
    """The PermissionError for an incompatible existing dataset keeps its historical
    prefix and lists every differing field as `what: (given) ... vs (parsed) ...`."""
    store = tmp_path / stores["zarr"]
    prepare_ds(
        store,
        (10, 10),
        voxel_size=Coordinate(1, 1),
        chunk_shape=Coordinate(5, 5),
        dtype="uint8",
    )

    with pytest.raises(PermissionError) as excinfo:
        prepare_ds(
            store,
            (10, 12),
            voxel_size=Coordinate(1, 1),
            chunk_shape=Coordinate(5, 5),
            dtype="uint8",
        )
    assert str(excinfo.value) == (
        f"Existing dataset is not compatible, but mode is not 'w' (store: {store}).\n"
        "Shapes: (given) (10, 12) vs (parsed) (10, 10)"
    )

    # several differences give one line each; metadata differences are listed too
    with pytest.raises(PermissionError) as excinfo:
        prepare_ds(
            store,
            (10, 12),
            voxel_size=Coordinate(2, 2),
            chunk_shape=Coordinate(5, 5),
            dtype="float32",
        )
    assert str(excinfo.value) == (
        f"Existing dataset is not compatible, but mode is not 'w' (store: {store}).\n"
        "Shapes: (given) (10, 12) vs (parsed) (10, 10)\n"
        "dtypes: (given) float32 vs (parsed) uint8\n"
        "voxel_size: (given) (2, 2) vs (parsed) (1, 1)"
    )

    # metadata only, under "r": the metadata prefix with the differing field
    with pytest.raises(PermissionError) as excinfo:
        prepare_ds(
            store,
            (10, 10),
            voxel_size=Coordinate(2, 2),
            chunk_shape=Coordinate(5, 5),
            dtype="uint8",
            mode="r",
        )
    assert str(excinfo.value) == (
        "Existing metadata is not compatible, but mode is 'r' and the metadata can't "
        f"be updated (store: {store}).\n"
        "voxel_size: (given) (2, 2) vs (parsed) (1, 1)"
    )


@pytest.mark.parametrize("scheduler", [None, "single-threaded", "threads"])
def test_scheduler_is_passed_through_open_ds_and_prepare_ds(tmp_path, scheduler):
    store = tmp_path / stores["zarr"]
    created = prepare_ds(
        store,
        (10, 10),
        voxel_size=Coordinate(1, 1),
        chunk_shape=Coordinate(5, 5),
        dtype="uint8",
        scheduler=scheduler,
    )
    assert created.scheduler == scheduler
    created[created.roi] = 3

    opened = open_ds(store, scheduler=scheduler)
    assert opened.scheduler == scheduler
    assert opened[opened.roi].max() == 3
    # an existing compatible dataset is reopened with the requested scheduler
    reopened = prepare_ds(
        store,
        (10, 10),
        voxel_size=Coordinate(1, 1),
        chunk_shape=Coordinate(5, 5),
        dtype="uint8",
        scheduler=scheduler,
    )
    assert reopened.scheduler == scheduler


# --- the dask view follows the write unit: one block per shard, never per inner chunk -------


@pytest.mark.parametrize("store", SHARDED_STORES)
def test_sharded_dask_view_is_chunked_by_shard(tmp_path, store):
    """`open_ds` and `prepare_ds` chunk the dask view by shard; `inner_chunk_shape`
    still names the stored inner chunk, and a `chunks=` override is honoured."""
    store = tmp_path / stores[store]
    data = _sharded_with_data(store)

    created = _sharded(tmp_path / f"created_{stores['zarr']}")
    assert Coordinate(created.data.chunksize) == SHARD

    reopened = open_ds(store)
    assert Coordinate(reopened.data.chunksize) == SHARD
    assert (reopened.chunk_shape, reopened.inner_chunk_shape) == (SHARD, INNER)
    shards = [-(-n // s) for n, s in zip(reopened.shape, SHARD, strict=False)]
    assert list(reopened.data.numblocks) == shards
    np.testing.assert_array_equal(reopened[reopened.roi], data)

    appended = _sharded(store, mode="a")
    assert Coordinate(appended.data.chunksize) == SHARD

    override = Coordinate(1, 32, 32)
    overridden = open_ds(store, chunks=tuple(override))
    assert Coordinate(overridden.data.chunksize) == override
    assert (overridden.chunk_shape, overridden.inner_chunk_shape) == (SHARD, override)
    np.testing.assert_array_equal(overridden[overridden.roi], data)


def test_inner_chunk_shape_of_data_that_is_not_zarr_is_the_views_chunk():
    import dask.array as da

    array = Array(da.zeros((4, 6), chunks=(2, 3)), offset=(0, 0), voxel_size=(1, 1))
    assert array.chunk_shape == array.inner_chunk_shape == Coordinate(2, 3)
    array.lazy_op(np.s_[0])
    assert array.inner_chunk_shape == Coordinate(3)
