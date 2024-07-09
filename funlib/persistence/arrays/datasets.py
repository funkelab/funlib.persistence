from .array import Array

from funlib.geometry import Coordinate, Roi

import zarr
from zarr.errors import PathNotFoundError
import logging
from typing import Optional, Union, Iterable

from .metadata import MetaDataFormat

logger = logging.getLogger(__name__)


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
        metadata_format if metadata_format is not None else MetaDataFormat()
    )

    data = zarr.open(store, mode=mode, **kwargs)
    metadata = metadata_format.parse(
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
    offset: Coordinate,
    voxel_size: Coordinate,
    axis_names: Iterable[str],
    units: Iterable[str],
    shape: Iterable[int],
    chunk_shape: Iterable[int],
    dtype,
    mode: str = "r+",
    **kwargs,
) -> Array:
    """Prepare a Zarr or N5 dataset.

    Args:

        Store:

            See https://zarr.readthedocs.io/en/stable/api/creation.html#zarr.creation.open_array

        offset:

            The offset of the dataset to prepare in world units. Only provide for physical dimensions.

        voxel_size:

            The size of one voxel in the dataset in world units. Only provide for physical dimensions.

        axis_names:

            The axis names of the dataset to create. The names of non-physical
            dimensions should end with "^". e.g. ["samples^", "channels^", "z", "y", "x"]

        units:

            The units of the dataset to create. Only provide for physical dimensions.

        shape:

            The shape of the dataset to create. For all dimensions,
            including non-physical.

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

    physical_shape = Coordinate([s for s, n in zip(shape, axis_names) if "^" not in n])
    roi = Roi(offset, physical_shape * voxel_size)

    try:
        existing_array = open_ds(store, mode="r", **kwargs)

    except PathNotFoundError:
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
    ds = zarr.open_array(
        store=store,
        shape=shape,
        chunks=chunk_shape,
        dtype=dtype,
        dimension_separator="/",
        mode=mode,
        **kwargs,
    )
    ds.attrs.put(
        {
            "axis_names": axis_names,
            "units": units,
            "voxel_size": voxel_size,
            "offset": roi.begin,
        }
    )

    # open array
    array = Array(ds, offset, voxel_size, axis_names, units)

    return array
