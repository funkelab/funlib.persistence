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

DEFAULT_INNER_CHUNK = 64
"""The chunk inside each shard when `prepare_ds(sharding=True)` is given none, per axis
(the whole axis where the shard is smaller)."""


def _default_inner_chunk(chunk_shape: Sequence[int]) -> tuple[int, ...]:
    inner = []
    for c in chunk_shape:
        c = int(c)
        if c <= DEFAULT_INNER_CHUNK:
            inner.append(c)
        elif c % DEFAULT_INNER_CHUNK == 0:
            inner.append(DEFAULT_INNER_CHUNK)
        else:
            raise ValueError(
                f"chunk_shape {tuple(int(x) for x in chunk_shape)}: axis extent {c} is "
                f"larger than the default inner chunk {DEFAULT_INNER_CHUNK} and not a "
                "multiple of it; pass inner_chunk_shape explicitly"
            )
    return tuple(inner)


def _is_empty_group(store, location: dict) -> bool:
    """Whether a zarr GROUP with no members and no attrs sits at the location."""
    try:
        group = zarr.open_group(store, mode="r", **location)
    except Exception:
        return False
    return not dict(group.attrs) and next(iter(group.keys()), None) is None


def _location_kwargs(kwargs: dict) -> dict:
    """The kwargs that say WHERE an array is, as `zarr.open` takes them."""
    return {
        k: kwargs[k] for k in ("path", "storage_options", "zarr_format") if k in kwargs
    }


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


def _validate_layout(
    chunk_shape: Optional[Coordinate],
    sharding: bool,
    inner_chunk_shape: Optional[Coordinate],
    kwargs: dict,
) -> Optional[tuple[int, ...]]:
    """Check the chunk arguments against each other. Returns the chunk inside each
    shard when sharding, else ``None``."""
    if "shard_shape" in kwargs:
        raise TypeError(
            "prepare_ds has no shard_shape: pass the shard as chunk_shape with "
            "sharding=True, and the chunk inside it as inner_chunk_shape"
        )
    if inner_chunk_shape is not None and any(int(c) <= 0 for c in inner_chunk_shape):
        raise ValueError(
            f"inner_chunk_shape {tuple(inner_chunk_shape)} must be positive per axis"
        )
    if not sharding:
        # unsharded, the chunk IS the inner chunk, so it may only be restated
        if inner_chunk_shape is not None and (
            chunk_shape is None or tuple(inner_chunk_shape) != tuple(chunk_shape)
        ):
            raise ValueError(
                "inner_chunk_shape only applies with sharding=True (unsharded, it may "
                "only restate chunk_shape)"
            )
        return None
    if chunk_shape is None:
        raise ValueError(
            "sharding=True needs a chunk_shape: the shard, the unit a writer covers"
        )
    inner = (
        tuple(int(c) for c in inner_chunk_shape)
        if inner_chunk_shape is not None
        else _default_inner_chunk(chunk_shape)
    )
    if len(inner) != len(tuple(chunk_shape)):
        raise ValueError(
            f"inner_chunk_shape {inner} and chunk_shape {tuple(chunk_shape)} differ "
            "in dimensions"
        )
    if any(int(c) % i != 0 for c, i in zip(chunk_shape, inner)):
        raise ValueError(
            f"chunk_shape {tuple(chunk_shape)} (the shard) must be a whole number of "
            f"inner_chunk_shape {inner} per axis"
        )
    return inner


def _normalise_sub_location(kwargs: dict) -> None:
    """`zarr.create_array` calls the sub-location `name`, every legacy call `path`:
    accept either and carry it as `path`."""
    if "name" in kwargs:
        name = kwargs.pop("name")
        path = kwargs.get("path")
        if path is not None and str(path) != str(name):
            raise ValueError(
                f"path={path!r} and name={name!r} name different sub-locations"
            )
        kwargs["path"] = name


def _find_existing(store, kwargs: dict) -> tuple[Optional[Array], bool]:
    """The array at the location, or ``None``; and whether a stray placeholder (the
    empty, attr-less group an interrupted sharded ``w-`` create leaves) is there
    instead."""
    try:
        # only what LOCATES the array: creation kwargs are not `zarr.open`'s
        return open_ds(store, mode="r", **_location_kwargs(kwargs)), False
    except FileNotFoundError:
        return None, False
    except TypeError:
        # a GROUP at the location: a real one is refused, a stray placeholder is nothing
        if not _is_empty_group(store, _location_kwargs(kwargs)):
            raise
        return None, True


def _differ(what: str, given, existing) -> bool:
    logger.info("%s differ: given (%s) vs parsed (%s)", what, given, existing)
    return True


def _data_compatible(
    existing: Array, shape, chunk_shape, inner_chunk_shape, dtype
) -> bool:
    """Whether the existing array can hold the requested one: same shape, dtype,
    write-safe chunk shape (the shard if sharded) and, if given, inner chunk shape."""
    source = existing._source_data
    write_unit = source.shards if source.shards is not None else source.chunks
    mismatch = False
    if tuple(shape) != tuple(existing.shape):
        mismatch = _differ("Shapes", tuple(shape), existing.shape)
    if chunk_shape is not None and tuple(chunk_shape) != tuple(write_unit):
        mismatch = _differ(
            "Chunk shapes (the write unit)", tuple(chunk_shape), write_unit
        )
    if inner_chunk_shape is not None and tuple(
        int(c) for c in inner_chunk_shape
    ) != tuple(source.chunks):
        mismatch = _differ(
            "Inner chunk shapes", tuple(inner_chunk_shape), source.chunks
        )
    if dtype != existing.dtype:
        mismatch = _differ("dtypes", dtype, existing.dtype)
    return not mismatch


def _metadata_compatible(given, existing) -> bool:
    mismatch = False
    for field in ("voxel_size", "types", "axis_names", "units"):
        if getattr(given, field) != getattr(existing, field):
            mismatch = _differ(field, getattr(given, field), getattr(existing, field))
    return not mismatch


def _create_sharded(
    store, shape, chunk_shape, inner, dtype, mode, stray_group: bool, kwargs: dict
) -> zarr.Array:
    """Create a zarr v3 array of ``chunk_shape`` shards holding ``inner`` chunks, through
    `zarr.create_array` (the legacy `zarr.open_array` cannot build a sharding codec)."""
    kwargs = dict(kwargs)
    kwargs.setdefault("zarr_format", 3)
    sub_path = kwargs.pop("path", None)
    reserved = sorted(set(kwargs) & {"chunks", "shards", "overwrite", "data"})
    if reserved:
        raise TypeError(
            f"prepare_ds sets {', '.join(reserved)} itself; pass shape, dtype, "
            "chunk_shape and inner_chunk_shape through its own parameters"
        )
    layout: dict[str, Any] = dict(
        name=sub_path,
        shape=tuple(shape),
        chunks=inner,
        shards=tuple(chunk_shape),
        dtype=dtype,
    )
    # Refuse a bad kwarg or value BEFORE anything touches the store, by making the
    # same call against an in-memory store: found later, it would leave the "w-"
    # placeholder below behind, and every retry would then meet a group.
    inspect.signature(zarr.create_array).bind_partial(**kwargs)
    zarr.create_array(
        store=zarr.storage.MemoryStore(),
        **layout,
        **{k: v for k, v in kwargs.items() if k != "storage_options"},
    )
    if mode == "w-" and not stray_group:
        # "w-" means nothing may be there, stray files included. `zarr.create_array`
        # only refuses an existing NODE, so the check is zarr's own, through a
        # placeholder group opened under "w-", which the create below replaces.
        zarr.open_group(
            store=store,
            mode="w-",
            path=sub_path,
            storage_options=kwargs.get("storage_options"),
            zarr_format=kwargs["zarr_format"],
        )
    return zarr.create_array(
        store=store,
        overwrite=mode in ("w", "w-") or stray_group,
        **layout,
        **kwargs,
    )


def _write_metadata(ds: zarr.Array, metadata, custom_metadata: Optional[dict]) -> None:
    fmt = get_default_metadata_format()
    attrs = {
        fmt.axis_names_attr: metadata.axis_names,
        fmt.units_attr: metadata.units,
        fmt.voxel_size_attr: metadata.voxel_size,
        fmt.offset_attr: metadata.offset,
        fmt.types_attr: metadata.types,
    }
    if custom_metadata is not None:
        assert set(attrs).isdisjoint(custom_metadata)
        attrs.update(custom_metadata)
    ds.attrs.put(attrs)


def prepare_ds(
    store,
    shape: Sequence[int],
    offset: Optional[Coordinate] = None,
    voxel_size: Optional[Coordinate] = None,
    axis_names: Optional[Sequence[str]] = None,
    units: Optional[Sequence[str]] = None,
    types: Optional[Sequence[str]] = None,
    chunk_shape: Optional[Coordinate] = None,
    dtype: DTypeLike = np.float32,
    mode: Literal["r", "r+", "a", "w", "w-"] = "a",
    custom_metadata: dict[str, Any] | None = None,
    sharding: bool = False,
    inner_chunk_shape: Optional[Coordinate] = None,
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

            The shape of the write-safe chunks to use in the dataset. For all dimensions,
            including non-physical. This is the shard shape if using shards.

        dtype:

            The datatype of the dataset to create.

        mode:

            The mode to open the dataset in.
            See https://zarr.readthedocs.io/en/stable/api/creation.html#zarr.creation.open_array

        custom_metadata:

            A dictionary of custom metadata to add to the dataset. This will be written to the
            zarr .attrs object.

        sharding:

            Create a zarr v3 array of `chunk_shape` shards, each holding `inner_chunk_shape`
            chunks. A creation option: an existing dataset is compared by its write-safe
            chunk shape, sharded or not. For more information see
            https://zarr.readthedocs.io/en/stable/user-guide/performance.html#sharding

        inner_chunk_shape:

            The efficient-read chunk shape. Can be less than the write-safe chunk shape in
            the case of sharding; defaults to `DEFAULT_INNER_CHUNK` (64) per axis there.
            Unsharded it can only restate `chunk_shape`. Given against an existing dataset
            it must match that dataset's chunks.

        kwargs:

            See additional arguments available here:
            https://zarr.readthedocs.io/en/stable/api/creation.html#zarr.creation.open_array
            (with `sharding`, a new array is created by `zarr.create_array`, which takes
            `compressors`, `filters`, `serializer`, ...).

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
    inner = _validate_layout(chunk_shape, sharding, inner_chunk_shape, kwargs)
    _normalise_sub_location(kwargs)
    existing, stray_group = _find_existing(store, kwargs)

    if existing is not None:
        existing_metadata = metadata_format.parse(shape, existing._source_data.attrs)
        if not _data_compatible(existing, shape, chunk_shape, inner_chunk_shape, dtype):
            if mode != "w":
                raise PermissionError(
                    "Existing dataset is not compatible, but mode is not 'w'."
                )
            logger.info("Existing dataset is not compatible, creating a new one")
        elif not _metadata_compatible(given_metadata, existing_metadata):
            if mode == "r":
                raise PermissionError(
                    "Existing metadata is not compatible, but mode is 'r' and the metadata can't be updated."
                )
            # the structure matches: reopened and relabelled below
        elif mode == "w":
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
        existing._source_data.attrs if existing is not None else {},
        offset=offset,
        voxel_size=voxel_size,
        axis_names=axis_names,
        units=units,
        types=types,
    )

    if not sharding:
        ds = zarr.open_array(
            store=store,
            shape=shape,
            chunks=chunk_shape,
            dtype=dtype,
            mode=mode,
            **kwargs,
        )
    elif existing is not None and mode in ("a", "r+"):
        # the structure matches and the metadata does not: reopen and relabel, as
        # the unsharded path does (`zarr.create_array` would refuse the node)
        ds = zarr.open_array(store=store, mode=mode, **kwargs)
    elif mode in ("r", "r+"):
        raise FileNotFoundError(
            f"No array at {store} to open in mode '{mode}', and sharding can only "
            "be applied when creating one."
        )
    else:
        ds = _create_sharded(
            store, shape, chunk_shape, inner, dtype, mode, stray_group, kwargs
        )

    _write_metadata(ds, combined_metadata, custom_metadata)

    # chunked as stored, so `chunk_shape` reports what it does from `open_ds`
    return Array(ds, offset, voxel_size, axis_names, units, types, ds.chunks)
