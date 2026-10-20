import inspect
import logging
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

import numpy as np
import zarr
import zarr.storage
from funlib.geometry import Coordinate
from numpy.typing import DTypeLike

from .array import Array
from .metadata import MetaDataFormat, get_default_metadata_format

logger = logging.getLogger(__name__)

def _location_kwargs(kwargs: dict) -> dict:
    """The kwargs that say WHERE an array is, as `zarr.open` takes them."""
    return {k: kwargs[k] for k in ("path", "storage_options", "zarr_format") if k in kwargs}


def open_ds(
    store,
    mode: Literal["r", "r+", "a", "w", "w-"] = "r",
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

    data = zarr.open(store, mode=mode, **kwargs)
    if not isinstance(data, zarr.Array):
        raise TypeError(f"Expected a zarr Array at {store}, got {type(data).__name__}")

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
    mode: Literal["r", "r+", "a", "w", "w-"] = "a",
    custom_metadata: dict[str, Any] | None = None,
    shard_shape: Optional[Sequence[int]] = None,
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

        shard_shape:

            If given, create a zarr v3 array with the sharding codec: each shard is one
            object in the store (one file, one S3 key) holding `chunk_shape`-sized chunks.
            For all dimensions, including non-physical. Every element must be a multiple
            of the corresponding `chunk_shape` element.

            A shard is the unit of writing. Writing a region that covers whole shards
            is a plain put; writing part of a shard reads the shard back, merges and
            rewrites it, so two writers touching the same shard at the same time race
            and one loses its data silently. Concurrent (e.g. blockwise) writers must
            each write regions aligned to, and a multiple of, `shard_shape`.

            `chunk_shape` keeps meaning the inner chunk, which is also what
            `Array.chunk_shape` reports; `Array.shard_shape` reports the shard.
            With a `shard_shape`, the array is created by `zarr.create_array` rather
            than `zarr.open_array`, so creation-time `kwargs` must be ones it accepts
            (`compressors`, `filters`, `serializer`, ...); opening and relabelling an
            existing array still go through `zarr.open`.

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

    # `zarr.create_array` calls the sub-location `name`, every legacy call `path`:
    # accept either, carry it as `path` from here on (the open/relabel calls take
    # that spelling) and hand it to `create_array` as `name` where that is called
    if "name" in kwargs:
        name = kwargs.pop("name")
        if kwargs.get("path", name) != name:
            raise ValueError(
                f"path={kwargs['path']!r} and name={name!r} name different sub-locations"
            )
        kwargs["path"] = name

    try:
        # only what LOCATES the array: creation kwargs (`compressors`, ...) are not
        # `zarr.open`'s, and on a location that exists without an array it would
        # hand them to `open_group` and fail with a TypeError instead of "not found"
        existing_array = open_ds(store, mode="r", **_location_kwargs(kwargs))
    except FileNotFoundError:
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

        if (
            shard_shape is not None
            and tuple(shard_shape) != existing_array._source_data.shards
        ):
            logger.info(
                "Shard shapes differ: given (%s) vs parsed (%s)",
                shard_shape,
                existing_array._source_data.shards,
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
                if not isinstance(ds, zarr.Array):
                    raise TypeError(
                        f"Expected a zarr Array at {store}, got {type(ds).__name__}"
                    )
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
    if shard_shape is None:
        ds = zarr.open_array(
            store=store,
            shape=shape,
            chunks=chunk_shape,
            dtype=dtype,
            mode=mode,
            **kwargs,
        )
    elif existing_array is not None and mode in ("a", "r+"):
        # Data-compatible but metadata-incompatible: the legacy path reopens the
        # array and rewrites its attrs below. `zarr.create_array` would refuse the
        # existing node, so reopen the same way.
        ds = zarr.open_array(store=store, mode=mode, **kwargs)
    else:
        # The legacy `zarr.open_array` cannot build a sharding codec, so a
        # sharded array goes through `zarr.create_array`, which takes `overwrite`
        # in place of `mode`. Reaching this point means there is nothing
        # compatible to reuse, so only the creating modes can proceed.
        if mode in ("r", "r+"):
            raise FileNotFoundError(
                f"No array at {store} to open in mode '{mode}', and a "
                f"shard_shape can only be applied when creating one."
            )
        if chunk_shape is None:
            # zarr guesses "auto" chunks from the array shape alone, ignoring the
            # shards, and then refuses them as not dividing the shard
            raise ValueError(
                f"shard_shape={tuple(shard_shape)} needs a chunk_shape, the chunk "
                "inside each shard."
            )
        # a sharded array is a zarr v3 array whatever the configured default format
        kwargs.setdefault("zarr_format", 3)
        create_kwargs = dict(kwargs)
        sub_path = create_kwargs.pop("path", None)
        reserved = sorted(
            set(create_kwargs)
            & {"store", "name", "shape", "dtype", "chunks", "shards", "overwrite", "data"}
        )
        if reserved:
            raise TypeError(
                f"prepare_ds sets {', '.join(reserved)} itself; pass shape, dtype, "
                "chunk_shape and shard_shape through its own parameters"
            )
        # Rejected BEFORE anything touches the store: a kwarg `zarr.create_array`
        # does not take, and anything it would refuse once it looks at the values
        # (an indivisible chunk, a codec of the wrong kind, a bad fill_value), both
        # found by making the same call against an in-memory store. Finding them
        # after the "w-" placeholder below exists would leave that placeholder
        # behind, and the probe above then meets a group and refuses every retry.
        inspect.signature(zarr.create_array).bind_partial(**create_kwargs)
        zarr.create_array(
            store=zarr.storage.MemoryStore(),
            name=sub_path,
            shape=tuple(shape),
            chunks=tuple(chunk_shape),
            shards=tuple(shard_shape),
            dtype=dtype,
            **{k: v for k, v in create_kwargs.items() if k != "storage_options"},
        )
        if mode == "w-":
            # "w-" means "create only where there is nothing": not an array, and
            # not stray files either (orphaned shards would come back as data).
            # `zarr.create_array` only refuses an existing NODE, so the emptiness
            # check is zarr's own, through a placeholder GROUP opened under "w-"
            # at the same location (it raises FileExistsError on anything there)
            # which the create below replaces.
            zarr.open_group(
                store=store,
                mode="w-",
                path=sub_path,
                storage_options=create_kwargs.get("storage_options"),
                zarr_format=create_kwargs["zarr_format"],
            )
        ds = zarr.create_array(
            store=store,
            name=sub_path,
            shape=tuple(shape),
            chunks=tuple(chunk_shape),
            shards=tuple(shard_shape),
            dtype=dtype,
            # "w" and the placeholder just written under "w-" are replaced; "a"
            # creates beside whatever else the location holds, as the legacy path
            overwrite=mode in ("w", "w-"),
            **create_kwargs,
        )

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

    # open array, chunked as stored so that `chunk_shape` reports the same
    # thing here as it does from `open_ds` and from the reuse path above
    array = Array(ds, offset, voxel_size, axis_names, units, types, ds.chunks)

    return array
