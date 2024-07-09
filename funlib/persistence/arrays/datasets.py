from .array import Array

from funlib.geometry import Coordinate, Roi

import zarr
from zarr.errors import PathNotFoundError
import logging
import os
import shutil
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
    chunks: Optional[Union[int, Iterable[int], str]] = "auto",
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
            See https://docs.dask.org/en/latest/array-chunks.html
            for more information. You can use a different chunksize
            than the chunksize of your data, but this can be quite
            detrimental.

    Returns:

        A :class:`Array` supporting spatial indexing on your dataset.
    """

    metadata_format = (
        metadata_format if metadata_format is not None else MetaDataFormat()
    )

    data = zarr.open(store, mode=mode)
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
        chunks,
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
    num_channels: Optional[int] = None,
) -> Array:
    """Prepare a Zarr or N5 dataset.

    Args:

        Store:

            See https://zarr.readthedocs.io/en/stable/api/convenience.html#zarr.convenience.open

        total_roi:

            The ROI of the dataset to prepare in world units.

        voxel_size:

            The size of one voxel in the dataset in world units.

        write_size:

            The size of anticipated writes to the dataset, in world units. The
            chunk size of the dataset will be set such that ``write_size`` is a
            multiple of it. This allows concurrent writes to the dataset if the
            writes are aligned with ``write_size``.

        num_channels:

            The number of channels.

        compressor:

            The compressor to use. See `zarr.get_codec` for available options.
            Defaults to gzip level 5.

        delete:

            Whether to delete an existing dataset if it was found to be
            incompatible with the other requirements. The default is not to
            delete the dataset and raise an exception instead.

        force_exact_write_size:

            Whether to use `write_size` as-is, or to first process it with
            `get_chunk_size`.

    Returns:

        A :class:`Array` pointing to the newly created dataset.
    """

    physical_shape = Coordinate([s for s, n in zip(shape, axis_names) if "^" not in n])
    roi = Roi(offset, physical_shape * voxel_size)

    try:
        existing_array = open_ds(store, mode="r")

    except PathNotFoundError:
        existing_array = None

    if existing_array is not None:
        metadata_compatible = True
        data_compatible = True

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

        if existing_array.dtype != dtype:
            logger.info("dtypes differ: %s vs %s", existing_array.dtype, dtype)
            data_compatible = False

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
    array = open_ds(store, mode="r+")

    return array
