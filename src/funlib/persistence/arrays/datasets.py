import logging
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import zarr
from numpy.typing import DTypeLike

from funlib.geometry import Coordinate

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
    offset: Optional[Sequence[int]] = None,
    voxel_size: Optional[Sequence[int]] = None,
    axis_names: Optional[Sequence[str]] = None,
    units: Optional[Sequence[str]] = None,
    types: Optional[Sequence[str]] = None,
    chunks: Optional[Union[int, Sequence[int], str]] = "strict",
    strict_metadata: bool = False,
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

        types (`str`, (optional)):

            An override for the types of your axes. For more details see:
            https://ngff.openmicroscopy.org/latest/#axes-md

        chunks (`Coordinate`, (optional)):

            An override for the size of the chunks in the dataset.
            The default value ("strict") will use the chunksize of the dataset.

            Otherwise, you can provide chunks in any format supported by dask.
            See https://docs.dask.org/en/stable/generated/dask.array.from_array.html
            for more information.

        strict_metadata (`bool`, (optional)):

            If True, all metadata fields (offset, voxel_size, axis_names, units, types)
            must be provided either as arguments or read from dataset attributes.

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
        types=types,
        strict=strict_metadata,
    )

    return Array(
        data,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        metadata.types,
        data.chunks if chunks == "strict" else chunks,
    )


def prepare_ds(
    store,
    shape: Sequence[int],
    offset: Optional[Coordinate] = None,
    voxel_size: Optional[Coordinate] = None,
    axis_names: Optional[Sequence[str]] = None,
    units: Optional[Sequence[str]] = None,
    types: Optional[Sequence[str]] = None,
    chunk_shape: Optional[Sequence[int]] = None,
    dtype: DTypeLike = np.float32,
    mode: str = "a",
    custom_metadata: dict[str, Any] | None = None,
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

            The axis names of the dataset to create.
            Set to ["c{i}^", "d{j}"] by default. Where i, j are the index of the non-physical
            and physical dimensions respectively.

        units:

            The units of the dataset to create. Only provide for physical dimensions.
            Set to all "" by default.

        types:

            The types of the axes of the dataset to create. For more details see:
            https://ngff.openmicroscopy.org/latest/#axes-md
            If not provided, we will first fall back on to axis_names if provided
            and use "channel" for axis names ending in "^", and "space" otherwise.
            If neither are provided, we will assume all dimensions are spatial.
            Note that axis name parsing is depricated and will be removed in the
            future. Please provide types directly if you have a mix of spatial and
            non-spatial dimensions.

        chunk_shape:

            The shape of the chunks to use in the dataset. For all dimensions,
            including non-physical.

        dtype:

            The datatype of the dataset to create.

        mode:

            The mode to open the dataset in.
            See https://zarr.readthedocs.io/en/stable/api/creation.html#zarr.creation.open_array

        custom_metadata:

            A dictionary of custom metadata to add to the dataset. This will be written to the
            zarr .attrs object.

        kwargs:

            See additional arguments available here:
            https://zarr.readthedocs.io/en/stable/api/creation.html#zarr.creation.open_array

            Of particular interest might be `compressor` if you would like to use a different
            compression algorithm and `synchronizer` if you want to guarantee that no data
            is corrupted due to concurrent reads or writes.

    Returns:

        A :class:`Array` pointing to the newly created dataset.
    """

    metadata_format = get_default_metadata_format()
    given_metadata = metadata_format.parse(
        shape,
        {},
        offset=offset,
        voxel_size=voxel_size,
        axis_names=axis_names,
        units=units,
        types=types,
    )

    try:
        existing_array = open_ds(store, mode="r", **kwargs)
    except ArrayNotFoundError:
        existing_array = None

    if existing_array is not None:
        data_compatible = True

        # data incompatibilities
        if shape != existing_array.shape:
            logger.info(
                "Shapes differ: given (%s) vs parsed (%s)", shape, existing_array.shape
            )
            data_compatible = False

        if (
            chunk_shape is not None
            and chunk_shape != existing_array._source_data.chunks
        ):
            logger.info(
                "Chunk shapes differ: given (%s) vs parsed (%s)",
                chunk_shape,
                existing_array._source_data.chunks,
            )
            data_compatible = False

        if dtype != existing_array.dtype:
            logger.info(
                "dtypes differ: given (%s) vs parsed (%s)", dtype, existing_array.dtype
            )
            data_compatible = False

        metadata_compatible = True
        existing_metadata = metadata_format.parse(
            shape,
            existing_array._source_data.attrs,
        )

        # metadata incompatibilities
        if given_metadata.voxel_size != existing_metadata.voxel_size:
            logger.info(
                "Voxel sizes differ: given (%s) vs parsed (%s)",
                given_metadata.voxel_size,
                existing_metadata.voxel_size,
            )
            metadata_compatible = False

        if given_metadata.types != existing_metadata.types:
            logger.info(
                "Types differ: given (%s) vs parsed (%s)",
                given_metadata.types,
                existing_metadata.types,
            )
            metadata_compatible = False

        if given_metadata.axis_names != existing_metadata.axis_names:
            logger.info(
                "Axis names differ: given (%s) vs parsed (%s)",
                given_metadata.axis_names,
                existing_metadata.axis_names,
            )
            metadata_compatible = False

        if given_metadata.units != existing_metadata.units:
            logger.info(
                "Units differ: given (%s) vs parsed(%s)",
                given_metadata.units,
                existing_metadata.units,
            )
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
                ds = zarr.open(store, mode=mode, **kwargs)
                return Array(
                    ds,
                    existing_metadata.offset,
                    existing_metadata.voxel_size,
                    existing_metadata.axis_names,
                    existing_metadata.units,
                    existing_metadata.types,
                    ds.chunks,
                )

    combined_metadata = metadata_format.parse(
        shape,
        existing_array._source_data.attrs if existing_array is not None else {},
        offset=offset,
        voxel_size=voxel_size,
        axis_names=axis_names,
        units=units,
        types=types,
    )

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
    our_metadata = {
        default_metadata_format.axis_names_attr: combined_metadata.axis_names,
        default_metadata_format.units_attr: combined_metadata.units,
        default_metadata_format.voxel_size_attr: combined_metadata.voxel_size,
        default_metadata_format.offset_attr: combined_metadata.offset,
        default_metadata_format.types_attr: combined_metadata.types,
    }
    # check keys don't conflict
    if custom_metadata is not None:
        assert set(our_metadata.keys()).isdisjoint(custom_metadata.keys())
        our_metadata.update(custom_metadata)

    ds.attrs.put(our_metadata)

    # open array
    array = Array(ds, offset, voxel_size, axis_names, units, types)

    return array
