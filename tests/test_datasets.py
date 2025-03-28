import numpy as np
import pytest

from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays.datasets import ArrayNotFoundError, open_ds, prepare_ds
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
def test_metadata(tmpdir, store):
    store = tmpdir / store

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
def test_helpers(tmpdir, store, dtype):
    shape = Coordinate(1, 1, 10, 20, 30)
    chunk_shape = Coordinate(2, 3, 10, 10, 10)
    store = tmpdir / store
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
    with pytest.raises(ArrayNotFoundError):
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
def test_open_ds(tmpdir, store, dtype):
    shape = Coordinate(1, 1, 10, 20, 30)
    store = tmpdir / store
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
    with pytest.raises(ArrayNotFoundError):
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
