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


@pytest.mark.parametrize("store", ["zarr", "zarr_ds", "zipped_zarr"])
def test_sharded(tmp_path, store):
    store = tmp_path / stores[store]
    shape = Coordinate(2, 64, 64)
    chunk_shape = Coordinate(1, 16, 16)
    shard_shape = Coordinate(2, 64, 32)
    axis_names = ["c^", "y", "x"]
    voxel_size = Coordinate(4, 4)

    # nothing to open yet: a shard_shape cannot be applied to an array that
    # does not exist, so the read modes fail the same way they do unsharded
    for mode in ("r", "r+"):
        with pytest.raises(FileNotFoundError):
            prepare_ds(
                store,
                shape,
                voxel_size=voxel_size,
                axis_names=axis_names,
                chunk_shape=chunk_shape,
                shard_shape=shard_shape,
                dtype=np.uint8,
                mode=mode,
            )

    array = prepare_ds(
        store,
        shape,
        voxel_size=voxel_size,
        axis_names=axis_names,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        dtype=np.uint8,
        mode="w",
    )
    # the shard is the stored object; the chunk stays the inner chunk
    assert array.shard_shape == shard_shape
    assert array.chunk_shape == chunk_shape
    codecs = array._source_data.metadata.to_dict()["codecs"]
    assert codecs[0]["name"] == "sharding_indexed"
    assert tuple(codecs[0]["configuration"]["chunk_shape"]) == tuple(chunk_shape)
    assert array._source_data.metadata.to_dict()["chunk_grid"]["configuration"][
        "chunk_shape"
    ] == tuple(shard_shape)

    # funlib's metadata lands on the sharded array like on any other
    assert array.voxel_size == voxel_size
    assert array.axis_names == axis_names

    # data written through funlib comes back through funlib
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    array[array.roi] = data
    reopened = open_ds(store)
    assert reopened.shard_shape == shard_shape
    assert reopened.chunk_shape == chunk_shape
    np.testing.assert_array_equal(reopened[reopened.roi], data)

    # a matching shard_shape reuses the array instead of recreating it
    same = prepare_ds(
        store,
        shape,
        voxel_size=voxel_size,
        axis_names=axis_names,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        dtype=np.uint8,
        mode="a",
    )
    np.testing.assert_array_equal(same[same.roi], data)

    # shard_shape=None means "don't care", like chunk_shape=None
    unspecified = prepare_ds(
        store,
        shape,
        voxel_size=voxel_size,
        axis_names=axis_names,
        dtype=np.uint8,
        mode="a",
    )
    assert unspecified.shard_shape == shard_shape
    np.testing.assert_array_equal(unspecified[unspecified.roi], data)

    # a different shard_shape is a data incompatibility: refused unless mode is "w"
    for mode in ("r", "r+", "a"):
        with pytest.raises(PermissionError):
            prepare_ds(
                store,
                shape,
                voxel_size=voxel_size,
                axis_names=axis_names,
                chunk_shape=chunk_shape,
                shard_shape=Coordinate(2, 32, 32),
                dtype=np.uint8,
                mode=mode,
            )
    np.testing.assert_array_equal(open_ds(store)[array.roi], data)

    recreated = prepare_ds(
        store,
        shape,
        voxel_size=voxel_size,
        axis_names=axis_names,
        chunk_shape=chunk_shape,
        shard_shape=Coordinate(2, 32, 32),
        dtype=np.uint8,
        mode="w",
    )
    assert recreated.shard_shape == Coordinate(2, 32, 32)
    assert not np.any(recreated[recreated.roi])

    # a collapsed dimension drops out of the shard shape as it does the chunk shape
    recreated.lazy_op(np.s_[0, :, :])
    assert recreated.shard_shape == Coordinate(32, 32)
    assert recreated.chunk_shape == Coordinate(16, 16)


@pytest.mark.parametrize("mode", ["a", "r+"])
def test_sharded_relabel_reopens_instead_of_recreating(tmp_path, mode):
    """Data-compatible but metadata-incompatible (a different voxel_size, say) is a RELABEL
    of the existing array in the unsharded path; the sharded path must do the same rather
    than hand the existing node to `zarr.create_array`, which refuses it."""
    store = tmp_path / stores["zarr"]
    layout = dict(chunk_shape=(2, 16, 16), shard_shape=(2, 64, 32), dtype=np.uint8)
    array = prepare_ds(
        store,
        (2, 64, 64),
        voxel_size=(1, 1),
        axis_names=["c^", "y", "x"],
        mode="w",
        **layout,
    )
    array[:] = 9

    relabeled = prepare_ds(
        store,
        (2, 64, 64),
        voxel_size=(4, 8),
        axis_names=["c^", "y", "x"],
        mode=mode,
        **layout,
    )
    assert relabeled.voxel_size == Coordinate(4, 8)
    assert relabeled.shard_shape == Coordinate(2, 64, 32)
    assert relabeled.chunk_shape == Coordinate(2, 16, 16)
    np.testing.assert_array_equal(relabeled[relabeled.roi], 9)
    assert open_ds(store).voxel_size == Coordinate(4, 8)


@pytest.mark.parametrize("store", ["zarr", "zarr_ds"])
def test_sharded_refusals(tmp_path, store):
    store = tmp_path / stores[store]

    # an unsharded array has no shard shape, so nothing sizes blocks off a phantom
    plain = prepare_ds(store, (8, 8), chunk_shape=(4, 4), mode="w")
    assert plain.shard_shape is None
    assert prepare_ds(store, (8, 8), shard_shape=None, mode="a").shard_shape is None

    # the created array reports the chunks as stored, like open_ds and the reuse
    # path do, not whatever dask would pick for an in-memory array
    assert plain.chunk_shape == Coordinate(4, 4)
    assert open_ds(store).chunk_shape == Coordinate(4, 4)
    assert prepare_ds(store, (8, 8), mode="a").chunk_shape == Coordinate(4, 4)

    # asking an unsharded array for a shard shape is a data incompatibility
    with pytest.raises(PermissionError):
        prepare_ds(store, (8, 8), chunk_shape=(4, 4), shard_shape=(8, 8), mode="a")

    # a shard must hold whole chunks
    with pytest.raises(ValueError):
        prepare_ds(store, (8, 8), chunk_shape=(3, 3), shard_shape=(8, 8), mode="w")

    # sharding is zarr v3 only
    with pytest.raises(ValueError):
        prepare_ds(
            store,
            (8, 8),
            chunk_shape=(4, 4),
            shard_shape=(8, 8),
            mode="w",
            zarr_format=2,
        )


def test_sharded_w_minus_refuses_an_existing_array_and_keeps_its_data(tmp_path):
    """ "w-" never recreates: a sharded call on an existing array refuses, like the
    unsharded path does, and the data is still there afterwards."""
    store = tmp_path / stores["zarr"]
    layout = dict(chunk_shape=(2, 16, 16), shard_shape=(2, 64, 32), dtype=np.uint8)
    array = prepare_ds(
        store,
        (2, 64, 64),
        voxel_size=(1, 1),
        axis_names=["c^", "y", "x"],
        mode="w",
        **layout,
    )
    array[:] = 9
    with pytest.raises(FileExistsError):
        prepare_ds(
            store,
            (2, 64, 64),
            voxel_size=(4, 8),
            axis_names=["c^", "y", "x"],
            mode="w-",
            **layout,
        )
    reopened = open_ds(store)
    np.testing.assert_array_equal(reopened[reopened.roi], 9)
    assert reopened.voxel_size == Coordinate(1, 1)


def test_shard_shape_is_none_after_a_callable_lazy_op(tmp_path):
    store = tmp_path / stores["zarr"]
    prepare_ds(
        store,
        (2, 64, 64),
        chunk_shape=(2, 16, 16),
        shard_shape=(2, 64, 32),
        axis_names=["c^", "y", "x"],
        dtype=np.uint8,
        mode="w",
    )
    array = open_ds(store)
    assert array.shard_shape == Coordinate(2, 64, 32)
    array.lazy_op(lambda d: d.sum(axis=0))
    assert array.shard_shape is None, "the shape changed in a way no shard describes"


def test_sharded_default_mode_creates_where_nothing_is(tmp_path):
    """The default mode, "a", on an empty location creates the sharded array."""
    store = tmp_path / stores["zarr"]
    array = prepare_ds(
        store,
        (2, 64, 64),
        chunk_shape=(2, 16, 16),
        shard_shape=(2, 64, 32),
        dtype=np.uint8,
    )
    assert array.shard_shape == Coordinate(2, 64, 32)
    assert open_ds(store).shard_shape == Coordinate(2, 64, 32)


def test_sharded_a_creates_beside_other_files_and_w_minus_refuses_them(tmp_path):
    """ "a" leaves whatever else the location holds (as the legacy path); "w-" refuses a
    non-empty location outright, stray files and orphaned shards included."""
    store = tmp_path / stores["zarr"]
    store.mkdir()
    (store / "README").write_text("not an array")
    layout = dict(chunk_shape=(2, 16, 16), shard_shape=(2, 64, 32), dtype=np.uint8)
    with pytest.raises(FileExistsError):
        prepare_ds(store, (2, 64, 64), mode="w-", **layout)
    assert (store / "README").exists() and not (store / "zarr.json").exists()

    prepare_ds(store, (2, 64, 64), mode="a", **layout)
    assert (store / "README").exists(), "mode 'a' must not wipe the location"
    assert open_ds(store).shard_shape == Coordinate(2, 64, 32)

    # orphaned shards: an array whose zarr.json is gone must not come back as data under "w-"
    array = open_ds(store, mode="r+")
    array[:] = 7
    (store / "zarr.json").unlink()
    with pytest.raises(FileExistsError):
        prepare_ds(store, (2, 64, 64), mode="w-", **layout)
    recreated = prepare_ds(store, (2, 64, 64), mode="w", **layout)
    assert not np.any(recreated[recreated.roi]), "mode 'w' starts from nothing"


def test_a_shard_needs_a_chunk_shape(tmp_path):
    with pytest.raises(ValueError, match="needs a chunk_shape"):
        prepare_ds(
            tmp_path / stores["zarr"], (2, 64, 64), shard_shape=(2, 64, 32), mode="w"
        )


def test_shard_shape_follows_collapsed_dims_and_stays_under_a_roi(tmp_path):
    """A tuple with two or more ints collapses the right dimensions (it did not: the
    in-place walk re-counted after each collapse), a Roi keeps the shard, a boolean
    mask (not writeable) drops it."""
    store = tmp_path / stores["zarr"]
    prepare_ds(
        store,
        (2, 64, 64),
        chunk_shape=(2, 16, 16),
        shard_shape=(2, 64, 32),
        voxel_size=(1, 1),
        axis_names=["c^", "y", "x"],
        dtype=np.uint8,
        mode="w",
    )
    for key, shard, chunk in [
        (np.s_[0, 3], (32,), (16,)),
        (np.s_[0, :, 3], (64,), (16,)),
        (np.s_[:, 3, 5], (2,), (2,)),
    ]:
        array = open_ds(store)
        array.lazy_op(key)
        assert array.shard_shape == Coordinate(*shard), key
        assert array.chunk_shape == Coordinate(*chunk), key
        assert array.shard_shape.dims == array.chunk_shape.dims == len(array.shape)

    array = open_ds(store)
    array.lazy_op(Roi((0, 0), (32, 32)))
    assert array.shard_shape == Coordinate(2, 64, 32)
    assert array.is_writeable

    array = open_ds(store)
    array.lazy_op(np.array([True, False]))
    assert not array.is_writeable
    assert array.shard_shape is None


def test_sharded_w_minus_creates_on_an_empty_location_with_creation_kwargs(tmp_path):
    """ "w-" on an empty location CREATES (the placeholder it uses for zarr's emptiness check
    is replaced, not tripped over), and creation kwargs that only `zarr.create_array` takes
    reach it."""
    from zarr.codecs import ZstdCodec

    store = tmp_path / stores["zarr"]
    array = prepare_ds(
        store,
        (2, 64, 64),
        chunk_shape=(2, 16, 16),
        shard_shape=(2, 64, 32),
        dtype=np.uint8,
        mode="w-",
        compressors=[ZstdCodec(level=3)],
    )
    assert array.shard_shape == Coordinate(2, 64, 32)
    array[:] = 4
    reopened = open_ds(store)
    assert reopened.shard_shape == Coordinate(2, 64, 32)
    np.testing.assert_array_equal(reopened[reopened.roi], 4)
    inner = reopened._source_data.metadata.to_dict()["codecs"][0]["configuration"][
        "codecs"
    ]
    assert any(c["name"] == "zstd" for c in inner), inner


def test_sharded_w_minus_at_a_sub_location_and_a_rejected_kwarg_leaves_nothing(
    tmp_path,
):
    """`name=` (create_array's spelling) and `path=` (the legacy spelling) both place the
    array; "w-" checks emptiness at THAT location; a kwarg create_array rejects is refused
    before anything is written, so a corrected retry is not met by a stray placeholder."""
    store = tmp_path / stores["zarr"]
    layout = dict(chunk_shape=(2, 16, 16), shard_shape=(2, 64, 32), dtype=np.uint8)

    with pytest.raises(TypeError):
        prepare_ds(store, (2, 64, 64), mode="w-", bogus=1, **layout)
    assert not store.exists(), "a rejected kwarg must not leave a placeholder"
    prepare_ds(store, (2, 64, 64), mode="w-", **layout)  # the retry goes through

    # a sub-location: the emptiness check and the create both target it (an array cannot
    # have children, so this is a second, group-rooted store)
    nested = tmp_path / "nested.zarr"
    sub = prepare_ds(nested, (2, 64, 64), mode="w-", name="sub", **layout)
    assert sub.shard_shape == Coordinate(2, 64, 32)
    assert open_ds(nested / "sub").shard_shape == Coordinate(2, 64, 32)
    with pytest.raises(FileExistsError):
        prepare_ds(nested, (2, 64, 64), mode="w-", path="sub", **layout)
    other = prepare_ds(nested, (2, 64, 64), mode="w-", path="other", **layout)
    assert other.shard_shape == Coordinate(2, 64, 32), (
        "an empty sibling location is fine"
    )
    with pytest.raises(ValueError, match="different sub-locations"):
        prepare_ds(nested, (2, 64, 64), mode="w", path="a", name="b", **layout)


def test_creation_kwargs_do_not_break_the_existence_probe(tmp_path):
    """An existing directory with no array in it, plus creation kwargs: the probe must say
    "no array here" and the create must proceed, in every creating mode."""
    from zarr.codecs import ZstdCodec

    layout = dict(chunk_shape=(2, 16, 16), shard_shape=(2, 64, 32), dtype=np.uint8)
    for mode in ("w-", "w", "a"):
        store = tmp_path / f"{mode.replace('-', 'minus')}.zarr"
        store.mkdir()
        array = prepare_ds(
            store, (2, 64, 64), mode=mode, compressors=[ZstdCodec()], **layout
        )
        assert array.shard_shape == Coordinate(2, 64, 32), mode
