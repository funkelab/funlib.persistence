import logging
from typing import Iterable, Optional, Union

import numpy as np
import zarr
from numpy.typing import DTypeLike

from funlib.geometry import Coordinate, Roi

from .array import Array
from .metadata import MetaDataFormat, get_default_metadata_format

logger = logging.getLogger(__name__)


class ArrayNotFoundError(Exception):
    """Exception raised when an array is not found in the dataset."""

    def __init__(self, message: str = "Array not found in the dataset"):
        self.message = message
        super().__init__(self.message)


def open_ds(
    store,
    mode: str = "r",
    metadata_format: Optional[MetaDataFormat] = None,
    offset: Optional[Iterable[int]] = None,
    voxel_size: Optional[Iterable[int]] = None,
    axis_names: Optional[Iterable[str]] = None,
    units: Optional[Iterable[str]] = None,
    chunks: Optional[Union[int, Iterable[int], str]] = "strict",
    **kwargs,
) -> Array:
    """
    Open a dataset with common metadata that is useful for image processing.

    Args:

        store:

            See https://zarr.readthedocs.io/en/stable/api/convenience.html#zarr.convenience.open

        mode:

            See https://zarr.readthedocs.io/en/stable/api/convenience.html#zarr.convenience.open

        metadata_format (`MetaDataFormat`, (optional)):

            A description of the metadata format. If not provided, the default attributes will
            be used.

        offset (`Coordinate`, (optional)):

            An override for the offset of the dataset in world units.
            Useful if your metadata is not stored with your array.

        voxel_size (`Coordinate`, (optional)):

            An override for the size of one voxel in the dataset in world units.
            Useful if your metadata is not stored with your array.

        axis_names (`list`, (optional)):

            An override for the names of your axes.

        units (`str`, (optional)):

            An override for the units of your dataset.

        chunks (`Coordinate`, (optional)):

            An override for the size of the chunks in the dataset.
            The default value ("strict") will use the chunksize of the dataset.

            Otherwise, you can provide chunks in any format supported by dask.
            See https://docs.dask.org/en/stable/generated/dask.array.from_array.html
            for more information.

        kwargs:

            See additional arguments available here:
            https://zarr.readthedocs.io/en/stable/api/convenience.html#zarr.convenience.open


    Returns:

        A :class:`Array` supporting spatial indexing on your dataset.
    """

    metadata_format = (
        metadata_format
        if metadata_format is not None
        else get_default_metadata_format()
    )

    try:
        data = zarr.open(store, mode=mode, **kwargs)
    except zarr.errors.PathNotFoundError:
        raise ArrayNotFoundError(f"Nothing found at path {store}")
    metadata = metadata_format.parse(
        data.shape,
        data.attrs,
        offset=offset,
        voxel_size=voxel_size,
        axis_names=axis_names,
        units=units,
    )

    return Array(
        data,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        data.chunks if chunks == "strict" else chunks,
    )


def prepare_ds(
    store,
    shape: Iterable[int],
    offset: Optional[Coordinate] = None,
    voxel_size: Optional[Coordinate] = None,
    axis_names: Optional[Iterable[str]] = None,
    units: Optional[Iterable[str]] = None,
    chunk_shape: Optional[Iterable[int]] = None,
    dtype: DTypeLike = np.float32,
    mode: str = "r+",
    **kwargs,
) -> Array:
    """Prepare a Zarr or N5 dataset.

    Args:

        Store:

            See https://zarr.readthedocs.io/en/stable/api/creation.html#zarr.creation.open_array

        shape:

            The shape of the dataset to create. For all dimensions,
            including non-physical.

        offset:

            The offset of the dataset to prepare in world units. Only provide for physical dimensions.
            Set to all 0's by default.

        voxel_size:

            The size of one voxel in the dataset in world units. Only provide for physical dimensions.
            Set to all 1's by default.

        axis_names:

            The axis names of the dataset to create. The names of non-physical
            dimensions should end with "^". e.g. ["samples^", "channels^", "z", "y", "x"]
            Set to ["c{i}^", "d{j}"] by default. Where i, j are the index of the non-physical
            and physical dimensions respectively.

        units:

            The units of the dataset to create. Only provide for physical dimensions.
            Set to all "" by default.

        chunk_shape:

            The shape of the chunks to use in the dataset. For all dimensions,
            including non-physical.

        dtype:

            The datatype of the dataset to create.

        mode:

            The mode to open the dataset in.
            See https://zarr.readthedocs.io/en/stable/api/creation.html#zarr.creation.open_array

        kwargs:

            See additional arguments available here:
            https://zarr.readthedocs.io/en/stable/api/creation.html#zarr.creation.open_array

            Of particular interest might be `compressor` if you would like to use a different
            compression algorithm and `synchronizer` if you want to guarantee that no data
            is corrupted due to concurrent reads or writes.

    Returns:

        A :class:`Array` pointing to the newly created dataset.
    """

    n_dim = len(shape)
    spatial_dims = set(
        [
            offset.dims if offset is not None else None,
            voxel_size.dims if voxel_size is not None else None,
            len(units) if units is not None else None,
            len([n for n in axis_names if "^" not in n])
            if axis_names is not None
            else None,
        ]
    )
    spatial_dims.discard(None)
    assert (
        len(spatial_dims) <= 1
    ), "Metadata must be consistent in the number of physical dimensions defined"
    spatial_dims = spatial_dims.pop() if len(spatial_dims) > 0 else n_dim
    channel_dims = n_dim - spatial_dims

    offset = Coordinate([0] * spatial_dims) if offset is None else offset
    voxel_size = Coordinate([1] * spatial_dims) if voxel_size is None else voxel_size
    axis_names = (
        list(f"c{i}^" for i in range(channel_dims))
        + list(f"d{i}" for i in range(spatial_dims))
        if axis_names is None
        else axis_names
    )
    units = [""] * voxel_size.dims if units is None else units

    physical_shape = Coordinate(shape[-spatial_dims:])
    roi = Roi(offset, physical_shape * voxel_size)

    try:
        existing_array = open_ds(store, mode="r", **kwargs)

    except ArrayNotFoundError:
        existing_array = None

    if existing_array is not None:
        metadata_compatible = True
        data_compatible = True

        # data incompatibilities
        if existing_array.shape != shape:
            logger.info("Shapes differ: %s vs %s", existing_array.shape, shape)
            data_compatible = False

        if existing_array._source_data.chunks != chunk_shape:
            logger.info(
                "Chunk shapes differ: %s vs %s",
                existing_array._source_data.chunks,
                chunk_shape,
            )
            data_compatible = False

        if existing_array.dtype != dtype:
            logger.info("dtypes differ: %s vs %s", existing_array.dtype, dtype)
            data_compatible = False

        # metadata incompatibilities
        if existing_array.voxel_size != voxel_size:
            logger.info(
                "Voxel sizes differ: %s vs %s", existing_array.voxel_size, voxel_size
            )
            metadata_compatible = False

        if existing_array.axis_names != axis_names:
            logger.info(
                "Axis names differ: %s vs %s", existing_array.axis_names, axis_names
            )
            metadata_compatible = False

        if existing_array.units != units:
            logger.info("Units differ: %s vs %s", existing_array.units, units)
            metadata_compatible = False

        if not data_compatible:
            logger.info(
                "Existing dataset is not compatible, attempting to create a new one"
            )
            if mode != "w":
                raise PermissionError(
                    "Existing dataset is not compatible, but mode is not 'w'."
                )
        elif not metadata_compatible:
            if mode == "r":
                raise PermissionError(
                    "Existing metadata is not compatible, but mode is 'r' and the metadata can't be udpated."
                )
        else:
            if mode == "w":
                logger.info(
                    "Existing dataset is compatible, but mode is 'w' and thus the existing dataset will be deleted"
                )
            else:
                return existing_array

    # create the dataset
    try:
        ds = zarr.open_array(
            store=store,
            shape=shape,
            chunks=chunk_shape,
            dtype=dtype,
            dimension_separator="/",
            mode=mode,
            **kwargs,
        )
    except zarr.errors.ArrayNotFoundError:
        raise ArrayNotFoundError(f"Nothing found at path {store}")

    default_metadata_format = get_default_metadata_format()
    ds.attrs.put(
        {
            default_metadata_format.axis_names_attr: axis_names,
            default_metadata_format.units_attr: units,
            default_metadata_format.voxel_size_attr: voxel_size,
            default_metadata_format.offset_attr: roi.begin,
        }
    )

    # open array
    array = Array(ds, offset, voxel_size, axis_names, units)

    return array
