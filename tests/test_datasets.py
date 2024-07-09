from funlib.persistence.arrays.metadata import MetaDataFormat
from funlib.persistence.arrays.datasets import open_ds, prepare_ds
from funlib.geometry import Coordinate, Roi

from zarr.errors import ArrayNotFoundError
import numpy as np

import pytest

stores = {
    "zarr": "test_array.zarr",
    "n5": "test_array.n5",
    "zarr_ds": "test_array.zarr/test_group/test_data",
    "n5_ds": "test_array.n5/test_group/test_data",
    "zipped_zarr": "test_array.zarr.zip",
    "zipped_zarr_ds": "test_array.zarr.zip/test_group/test_data",
}


@pytest.mark.parametrize("store", stores.keys())
def test_helpers(tmpdir, store):
    store = tmpdir / store
    metadata = MetaDataFormat().parse(
        {
            "offset": [100, 200, 400],
            "voxel_size": [1, 2, 3],
            "axis_names": ["sample^", "channel^", "z", "y", "x"],
            "units": ["nm", "nm", "nm"],
        }
    )
    shape = Coordinate(1, 1, 10, 20, 30)
    chunk_shape = Coordinate(2, 3, 10, 10, 10)

    # test prepare_ds fails if array does not exist and mode is read
    with pytest.raises(ArrayNotFoundError):
        prepare_ds(
            store,
            metadata.offset,
            metadata.voxel_size,
            metadata.axis_names,
            metadata.units,
            shape,
            chunk_shape,
            dtype=np.float32,
            mode="r",
        )

    # test prepare_ds creates array if it does not exist and mode is write
    array = prepare_ds(
        store,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        shape,
        chunk_shape,
        dtype=np.float32,
        mode="w",
    )
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*shape[-3:])
    )
    assert array.voxel_size == metadata.voxel_size
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units

    # test prepare_ds opens array if it exists and mode is read
    array = prepare_ds(
        store,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        shape,
        chunk_shape,
        dtype=np.float32,
        mode="r",
    )
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*shape[-3:])
    )
    assert array.voxel_size == metadata.voxel_size
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units

    # test prepare_ds fails if array exists and is opened in read mode
    # with incompatible arguments
    with pytest.raises(PermissionError):
        array = prepare_ds(
            store,
            metadata.offset,
            metadata.voxel_size,
            metadata.axis_names,
            metadata.units,
            chunk_shape,
            chunk_shape,
            dtype=np.float32,
            mode="r",
        )

    # test prepare_ds overwrite existing array in write mode
    array = prepare_ds(
        store,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        chunk_shape,
        chunk_shape,
        dtype=np.float32,
        mode="w",
    )
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*chunk_shape[-3:])
    )
    assert array.voxel_size == metadata.voxel_size
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units

    # test prepare_ds updates metadata existing array in "r+" or "a" mode if it exists
    array = prepare_ds(
        store,
        metadata.offset,
        metadata.voxel_size * 2,
        metadata.axis_names,
        metadata.units,
        chunk_shape,
        chunk_shape,
        dtype=np.float32,
        mode="r+",
    )
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*chunk_shape[-3:]) * 2
    )
    assert array.voxel_size == metadata.voxel_size * 2
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units

    # test prepare_ds updates existing array in "r+" or "a" mode if it exists
    array = prepare_ds(
        store,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        chunk_shape,
        chunk_shape,
        dtype=np.float32,
        mode="a",
    )
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*chunk_shape[-3:])
    )
    assert array.voxel_size == metadata.voxel_size
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units

    # test prepare_ds with mode "w" overwrites existing array even if compatible
    array[:] = 2
    assert np.all(np.isclose(array[:], np.ones(chunk_shape) * 2))
    array = prepare_ds(
        store,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        chunk_shape,
        chunk_shape,
        dtype=np.float32,
        mode="w",
    )
    assert np.all(np.isclose(array[:], np.ones(chunk_shape) * 0))
    assert array.roi == Roi(
        metadata.offset, metadata.voxel_size * Coordinate(*chunk_shape[-3:])
    )
    assert array.voxel_size == metadata.voxel_size
    assert array.offset == metadata.offset
    assert array.axis_names == metadata.axis_names
    assert array.units == metadata.units
