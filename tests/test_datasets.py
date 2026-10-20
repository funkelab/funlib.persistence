from typing import Any

import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi

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


@pytest.mark.parametrize("store", stores.keys())
@pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.uint64])
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

    # test prepare_ds fails if array does not exist and mode is read
    with pytest.raises(FileNotFoundError):
        prepare_ds(
            store,
            shape,
            metadata.offset,
            metadata.voxel_size,
            metadata.axis_names,
            metadata.units,
            metadata.types,
            chunk_shape=chunk_shape,
            dtype=dtype,
            mode="r",
        )

    # test prepare_ds creates array if it does not exist and mode is write
    array = prepare_ds(
        store,
        shape,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        metadata.types,
        chunk_shape=chunk_shape,
        dtype=dtype,
        mode="w",
    )
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*shape[-3:])
    )
    assert array.voxel_size == metadata.voxel_size
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units
    assert array.types == metadata.types

    # test prepare_ds opens array if it exists and mode is read
    array = prepare_ds(
        store,
        shape,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        metadata.types,
        chunk_shape=chunk_shape,
        dtype=dtype,
        mode="r",
    )
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*shape[-3:])
    )
    assert array.voxel_size == metadata.voxel_size
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units
    assert array.types == metadata.types

    # test prepare_ds fails if array exists and is opened in read mode
    # with incompatible arguments
    with pytest.raises(PermissionError):
        array = prepare_ds(
            store,
            chunk_shape,
            metadata.offset,
            metadata.voxel_size,
            metadata.axis_names,
            metadata.units,
            metadata.types,
            chunk_shape=chunk_shape,
            dtype=dtype,
            mode="r",
        )

    # test prepare_ds overwrite existing array in write mode
    array = prepare_ds(
        store,
        chunk_shape,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        metadata.types,
        chunk_shape=chunk_shape,
        dtype=dtype,
        mode="w",
    )
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*chunk_shape[-3:])
    )
    assert array.voxel_size == metadata.voxel_size
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units
    assert array.types == metadata.types

    # test prepare_ds updates metadata existing array in "r+" or "a" mode if it exists
    array = prepare_ds(
        store,
        chunk_shape,
        metadata.offset,
        metadata.voxel_size * 2,
        metadata.axis_names,
        metadata.units,
        metadata.types,
        chunk_shape=chunk_shape,
        dtype=dtype,
        mode="r+",
    )
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*chunk_shape[-3:]) * 2
    )
    assert array.voxel_size == metadata.voxel_size * 2
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units
    assert array.types == metadata.types

    # test prepare_ds updates existing array in "r+" or "a" mode if it exists
    array = prepare_ds(
        store,
        chunk_shape,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        metadata.types,
        chunk_shape=chunk_shape,
        dtype=dtype,
        mode="a",
    )
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*chunk_shape[-3:])
    )
    assert array.voxel_size == metadata.voxel_size
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units
    assert array.types == metadata.types

    # test prepare_ds with mode "w" overwrites existing array even if compatible
    array[:] = 2
    assert np.all(np.isclose(array[:], 2))
    array = prepare_ds(
        store,
        chunk_shape,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        metadata.types,
        chunk_shape=chunk_shape,
        dtype=dtype,
        mode="w",
    )
    assert np.all(np.isclose(array[:], 0))
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*chunk_shape[-3:])
    )
    assert array.voxel_size == metadata.voxel_size
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units
    assert array.types == metadata.types


@pytest.mark.parametrize("store", stores.keys())
@pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.uint64])
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

    # test open_ds fails if array does not exist and mode is read
    with pytest.raises(FileNotFoundError):
        open_ds(
            store,
            offset=metadata.offset,
            voxel_size=metadata.voxel_size,
            axis_names=metadata.axis_names,
            units=metadata.units,
            types=metadata.types,
            mode="r",
        )

    # test open_ds creates array if it does not exist and mode is write
    array = prepare_ds(
        store,
        shape,
        offset=metadata.offset,
        voxel_size=metadata.voxel_size,
        axis_names=metadata.axis_names,
        units=metadata.units,
        types=metadata.types,
        dtype=dtype,
        mode="w",
    )
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*shape[-3:])
    )
    assert array.voxel_size == metadata.voxel_size
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units

    # test open_ds opens array if it exists and mode is read
    array = open_ds(
        store,
        offset=metadata.offset,
        voxel_size=metadata.voxel_size,
        axis_names=metadata.axis_names,
        units=metadata.units,
        mode="r",
    )
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*shape[-3:])
    )
    assert array.voxel_size == metadata.voxel_size
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units

    # test open_ds fails if array exists and is opened in read mode
    # with incompatible arguments
    array = open_ds(
        store,
        offset=(1, 2, 3),
        voxel_size=(1, 2, 3),
        axis_names=None,
        units=None,
    )


# --- sharding: chunk_shape is the WRITE unit (the shard), inner_chunk_shape the read unit -------

SHARD = Coordinate(2, 64, 32)  # what a writer covers whole; the stored object
INNER = Coordinate(2, 16, 16)  # what a reader can fetch on its own


def _sharded(store, mode="w", **over):
    kw: dict[str, Any] = dict(
        shape=(2, 64, 64),
        voxel_size=(1, 1),
        axis_names=["c^", "y", "x"],
        chunk_shape=SHARD,
        sharding=True,
        inner_chunk_shape=INNER,
        dtype=np.uint8,
        mode=mode,
    )
    kw.update(over)
    return prepare_ds(store, **kw)


@pytest.mark.parametrize("store", ["zarr", "zarr_ds", "zipped_zarr"])
def test_sharded(tmp_path, store):
    store = tmp_path / stores[store]

    # nothing to open yet: the read modes fail the same way they do unsharded
    for mode in ("r", "r+"):
        with pytest.raises(FileNotFoundError):
            _sharded(store, mode=mode)

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

    # the same layout reuses the array; so does the same WRITE UNIT with sharding unsaid
    # (a creation option only) and with no inner chunk said (don't care)
    for over in (
        {},
        {"sharding": False, "inner_chunk_shape": None},
        {"inner_chunk_shape": None},
    ):
        same = _sharded(store, mode="a", **over)
        np.testing.assert_array_equal(same[same.roi], data)
        assert same.chunk_shape == SHARD

    # another write unit, or another inner chunk, is a data incompatibility
    for over in ({"chunk_shape": (2, 32, 32)}, {"inner_chunk_shape": (2, 8, 8)}):
        for mode in ("r", "r+", "a"):
            with pytest.raises(PermissionError):
                _sharded(store, mode=mode, **over)
    np.testing.assert_array_equal(open_ds(store)[array.roi], data)

    recreated = _sharded(store, mode="w", chunk_shape=(2, 32, 32))
    assert (recreated.chunk_shape, recreated.inner_chunk_shape) == (
        Coordinate(2, 32, 32),
        INNER,
    )
    assert not np.any(recreated[recreated.roi])

    # a collapsed dimension drops out of every shape alike
    recreated.lazy_op(np.s_[0, :, :])
    assert recreated.chunk_shape == recreated.shard_shape == Coordinate(32, 32)
    assert recreated.inner_chunk_shape == Coordinate(16, 16)


def test_unsharded_is_reused_by_a_sharding_caller_with_the_same_write_unit(tmp_path):
    """`sharding` is a creation option; an existing array is judged by its write unit."""
    store = tmp_path / stores["zarr"]
    plain = prepare_ds(store, (8, 8), chunk_shape=(4, 4), mode="w")
    assert (
        plain.shard_shape is None
        and plain.chunk_shape == plain.inner_chunk_shape == (4, 4)
    )
    again = prepare_ds(store, (8, 8), chunk_shape=(4, 4), sharding=True, mode="a")
    assert again.shard_shape is None, "reused as it is; nothing recreated"
    with pytest.raises(PermissionError):
        prepare_ds(store, (8, 8), chunk_shape=(8, 8), sharding=True, mode="a")


def test_inner_chunk_defaults(tmp_path):
    """64 per axis, the whole axis where the shard is smaller, and a loud ask where 64
    does not divide a larger axis."""
    a = prepare_ds(
        tmp_path / "a.zarr",
        (2, 256, 256),
        chunk_shape=(2, 256, 128),
        sharding=True,
        dtype=np.uint8,
        mode="w",
    )
    assert a.inner_chunk_shape == Coordinate(2, 64, 64)
    assert a.chunk_shape == Coordinate(2, 256, 128)
    b = prepare_ds(
        tmp_path / "b.zarr",
        (2, 32, 32),
        chunk_shape=(2, 32, 32),
        sharding=True,
        dtype=np.uint8,
        mode="w",
    )
    assert b.inner_chunk_shape == Coordinate(2, 32, 32)
    with pytest.raises(ValueError, match="pass inner_chunk_shape explicitly"):
        prepare_ds(
            tmp_path / "c.zarr",
            (2, 100, 100),
            chunk_shape=(2, 100, 100),
            sharding=True,
            dtype=np.uint8,
            mode="w",
        )


def test_sharding_option_refusals(tmp_path):
    store = tmp_path / stores["zarr"]
    with pytest.raises(ValueError, match="only applies with sharding=True"):
        prepare_ds(
            store, (8, 8), chunk_shape=(8, 8), inner_chunk_shape=(4, 4), mode="w"
        )
    with pytest.raises(ValueError, match="needs a chunk_shape"):
        prepare_ds(store, (8, 8), sharding=True, mode="w")
    with pytest.raises(ValueError, match="whole number of inner_chunk_shape"):
        prepare_ds(
            store,
            (8, 8),
            chunk_shape=(8, 8),
            sharding=True,
            inner_chunk_shape=(3, 3),
            mode="w",
        )
    with pytest.raises(ValueError, match="differ in dimensions"):
        prepare_ds(
            store,
            (8, 8),
            chunk_shape=(8, 8),
            sharding=True,
            inner_chunk_shape=(4,),
            mode="w",
        )
    with pytest.raises(ValueError):  # sharding is zarr v3 only
        prepare_ds(
            store, (8, 8), chunk_shape=(8, 8), sharding=True, mode="w", zarr_format=2
        )
    assert not store.exists(), "every refusal came before the location was touched"


@pytest.mark.parametrize("mode", ["a", "r+"])
def test_sharded_relabel_reopens_instead_of_recreating(tmp_path, mode):
    """Data-compatible but metadata-incompatible is a RELABEL of the existing array, on the
    sharded path as on the unsharded one."""
    store = tmp_path / stores["zarr"]
    array = _sharded(store)
    array[:] = 9
    relabeled = _sharded(store, mode=mode, voxel_size=(4, 8))
    assert relabeled.voxel_size == Coordinate(4, 8)
    assert (relabeled.chunk_shape, relabeled.inner_chunk_shape) == (SHARD, INNER)
    np.testing.assert_array_equal(relabeled[relabeled.roi], 9)
    assert open_ds(store).voxel_size == Coordinate(4, 8)


def test_sharded_w_minus_refuses_an_existing_array_and_keeps_its_data(tmp_path):
    store = tmp_path / stores["zarr"]
    _sharded(store)[:] = 9
    with pytest.raises(FileExistsError):
        _sharded(store, mode="w-", voxel_size=(4, 8))
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
    assert (store / "README").exists() and not (store / "zarr.json").exists()

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
    assert not np.any(recreated[recreated.roi]), "mode 'w' starts from nothing"


def test_shard_shape_follows_collapsed_dims_and_stays_under_a_roi(tmp_path):
    """A tuple with two or more ints collapses the right dimensions (it did not: the
    in-place walk re-counted after each collapse), a Roi keeps the shard, a boolean
    mask (not writeable) drops it and chunk_shape falls back to the read chunking."""
    store = tmp_path / stores["zarr"]
    _sharded(store)
    for key, shard, inner in [
        (np.s_[0, 3], (32,), (16,)),
        (np.s_[0, :, 3], (64,), (16,)),
        (np.s_[:, 3, 5], (2,), (2,)),
    ]:
        array = open_ds(store)
        array.lazy_op(key)
        assert array.chunk_shape == array.shard_shape == Coordinate(*shard), key
        assert array.inner_chunk_shape == Coordinate(*inner), key
        assert (
            array.shard_shape.dims == array.inner_chunk_shape.dims == len(array.shape)
        )

    array = open_ds(store)
    array.lazy_op(Roi((0, 0), (32, 32)))
    assert array.shard_shape == SHARD and array.is_writeable

    array = open_ds(store)
    array.lazy_op(np.array([True, False]))  # ty: ignore[invalid-argument-type]
    assert not array.is_writeable
    assert array.shard_shape is None
    assert array.chunk_shape == array.inner_chunk_shape == Coordinate(1, 16, 16)

    array = open_ds(store)
    array.lazy_op(lambda d: d.sum(axis=0))
    assert array.shard_shape is None


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
        tmp_path / "plain.zarr", (128, 128), chunk_shape=(64, 64), mode="w"
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
    with pytest.raises(ValueError, match="only restate chunk_shape"):
        prepare_ds(
            tmp_path / "x.zarr",
            (128, 128),
            chunk_shape=(64, 64),
            inner_chunk_shape=(32, 32),
            mode="w",
        )
    with pytest.raises(ValueError, match="positive per axis"):
        prepare_ds(
            tmp_path / "x.zarr",
            (128, 128),
            chunk_shape=(64, 64),
            sharding=True,
            inner_chunk_shape=(0, 64),
            mode="w",
        )
    with pytest.raises(TypeError, match="has no shard_shape"):
        prepare_ds(
            tmp_path / "x.zarr",
            (128, 128),
            chunk_shape=(64, 64),
            shard_shape=(64, 64),
            mode="w",
        )


@pytest.mark.parametrize("mode", ["a", "r+"])
def test_a_sharding_caller_relabels_a_plain_array_in_place(tmp_path, mode):
    """`sharding` is a creation option: against an existing PLAIN array with the same write unit
    and other metadata, the array is reopened and relabelled, not recreated and not refused."""
    store = tmp_path / stores["zarr"]
    plain = prepare_ds(
        store,
        (2, 64, 64),
        chunk_shape=SHARD,
        voxel_size=(1, 1),
        axis_names=["c^", "y", "x"],
        dtype=np.uint8,
        mode="w",
    )
    plain[:] = 6
    # no explicit inner chunk: one that the plain array does not have is a data incompatibility
    relabeled = _sharded(store, mode=mode, voxel_size=(4, 8), inner_chunk_shape=None)
    assert relabeled.voxel_size == Coordinate(4, 8)
    assert relabeled.shard_shape is None, "reused as it is"
    with pytest.raises(PermissionError):
        _sharded(
            store, mode=mode, voxel_size=(4, 8)
        )  # asks for inner chunks it has not got
    assert relabeled.chunk_shape == SHARD
    np.testing.assert_array_equal(relabeled[relabeled.roi], 6)


def test_sharded_w_minus_creates_on_an_empty_location_with_creation_kwargs(tmp_path):
    from zarr.codecs import ZstdCodec

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


def test_sharded_w_minus_at_a_sub_location_and_a_rejected_kwarg_leaves_nothing(
    tmp_path,
):
    """`name=` (create_array's spelling) and `path=` (the legacy spelling) both place the
    array, on the create, reuse and relabel paths; "w-" checks emptiness at THAT location; a
    kwarg create_array rejects is refused before anything is written."""
    store = tmp_path / stores["zarr"]
    with pytest.raises(TypeError):
        _sharded(store, mode="w-", bogus=1)
    assert not store.exists(), "a rejected kwarg must not leave a placeholder"
    _sharded(store, mode="w-")  # the retry goes through

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
        nested, mode="a", name="another", voxel_size=(4, 8)
    ).voxel_size == Coordinate(4, 8)


def test_sharded_w_minus_leaves_nothing_behind_a_create_zarr_refuses(tmp_path):
    """A create zarr refuses on its VALUES, or a kwarg prepare_ds owns, is found before the
    location is touched; the corrected retry then goes through under every creating mode."""
    from zarr.codecs import ZstdCodec

    store = tmp_path / stores["zarr"]
    cases: list[dict[str, Any]] = [
        dict(inner_chunk_shape=(3, 16, 16)),  # does not divide the shard
        dict(compressors=ZstdCodec),  # the class, not an instance
        dict(chunks=(2, 16, 16)),  # prepare_ds's own parameter
        dict(overwrite=True),
    ]
    for bad in cases:
        with pytest.raises((ValueError, TypeError)):
            _sharded(store, mode="w-", **bad)
        assert not store.exists(), f"{bad} left something at the location"
    for mode in ("w-", "w", "a"):
        assert _sharded(store, mode=mode).shard_shape == SHARD, mode
        if mode == "w-":
            with pytest.raises(FileExistsError):
                _sharded(store, mode="w-")


def test_creation_kwargs_do_not_break_the_existence_probe(tmp_path):
    """An existing directory with no array in it, plus creation kwargs: the probe must say
    "no array here" and the create must proceed, in every creating mode."""
    from zarr.codecs import ZstdCodec

    for mode in ("w-", "w", "a"):
        store = tmp_path / f"{mode.replace('-', 'minus')}.zarr"
        store.mkdir()
        assert (
            _sharded(store, mode=mode, compressors=[ZstdCodec()]).shard_shape == SHARD
        ), mode


@pytest.mark.parametrize("mode", ["w-", "w", "a"])
def test_a_stray_placeholder_group_does_not_block_a_sharded_create(tmp_path, mode):
    """An EMPTY group with no attrs at the location (a "w-" placeholder whose process died
    before the create) is never data and counts as nothing there; a group with content is
    a real group and refused."""
    import zarr

    store = tmp_path / stores["zarr"]
    zarr.open_group(str(store), mode="w-", zarr_format=3)
    assert _sharded(store, mode=mode).shard_shape == SHARD
    assert open_ds(store).shard_shape == SHARD

    real = tmp_path / "real.zarr"
    zarr.open_group(str(real), mode="w-", zarr_format=3).attrs["kept"] = True
    with pytest.raises(TypeError, match="got Group"):
        _sharded(real, mode=mode)
    assert zarr.open_group(str(real), mode="r").attrs["kept"] is True


def test_unsharded_name_is_path(tmp_path):
    store = tmp_path / stores["zarr"]
    for mode in ("w-", "w", "a"):
        array = prepare_ds(store, (8, 8), chunk_shape=(4, 4), mode=mode, name="sub")
        assert array.shape == (8, 8)
        assert open_ds(store / "sub").chunk_shape == Coordinate(4, 4)
        assert prepare_ds(store, (8, 8), mode="a", path="sub").shape == (8, 8)
