import logging
from collections.abc import Sequence
from pathlib import Path

from iohub.ngff import TransformationMeta, open_ome_zarr
from iohub.ngff.models import AxisMeta
from numpy.typing import DTypeLike

from funlib.geometry import Coordinate

from .array import Array
from .metadata import MetaData

logger = logging.getLogger(__name__)


def open_ome_ds(
    store: Path,
    name: str,
    mode: str = "r",
    **kwargs,
) -> Array:
    """
    Open an ome-zarr dataset with common metadata that is useful for indexing with physcial coordinates.

    Args:

        store:

            See https://czbiohub-sf.github.io/iohub/main/api/ngff.html#iohub.open_ome_zarr

        name:

            The name of the dataset in your ome-zarr dataset.

        mode:

            See https://zarr.readthedocs.io/en/stable/api/convenience.html#zarr.convenience.open

        kwargs:

            See additional arguments available here:
            https://czbiohub-sf.github.io/iohub/main/api/ngff.html#iohub.open_ome_zarr


    Returns:

        A :class:`Array` supporting spatial indexing on your dataset.
    """

    assert (store / name).exists(), f"Store: {store / name} does not exist!"

    ome_zarr = open_ome_zarr(store, mode=mode, **kwargs)
    axes = ome_zarr.axes
    axis_names = [axis.name for axis in axes]
    units = [axis.unit for axis in axes if axis.unit is not None]
    types = [axis.type for axis in axes if axis.type is not None]

    scale = ome_zarr.get_effective_scale(name)
    offset = ome_zarr.get_effective_translation(name)

    dataset = ome_zarr[name]

    metadata = MetaData(
        shape=dataset.shape,
        offset=offset,
        voxel_size=scale,
        axis_names=axis_names,
        units=units,
        types=types,
    )

    return Array(
        dataset,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        metadata.types,
    )


def prepare_ome_ds(
    store: Path,
    name: str,
    shape: Sequence[int],
    dtype: DTypeLike,
    chunk_shape: Sequence[int] | None = None,
    offset: Sequence[int] | None = None,
    voxel_size: Sequence[int] | None = None,
    axis_names: Sequence[str] | None = None,
    units: Sequence[str] | None = None,
    types: Sequence[str] | None = None,
    channel_names: list["str"] | None = None,
    **kwargs,
) -> Array:
    """Prepare an OME-Zarr dataset with common metadata that we use for indexing images with
    spatial coordinates.

    Args:

        Store:

            See https://czbiohub-sf.github.io/iohub/main/api/ngff.html#iohub.open_ome_zarr

        shape:

            The shape of the dataset to create. For all dimensions,
            including non-physical.

        chunk_shape:

            The shape of the chunks to use. If None, the default chunk shape
            is used.

        offset:

            The offset of the dataset in physical coordinates. If None, the
            default offset (0, ...) is used.

        voxel_size:

            The size of a voxel in physical coordinates. If None, the default
            voxel size (1, ...) is used.

        axis_names:

            The names of the axes in the dataset. If None, the default axis
            names ("d0", "d1", ...) are used.

        units:

            The units of the axes in the dataset. If None, the default units
            ("", "", ...) are used.

        types:

            The types of the axes in the dataset. If None, the default types
            ("space", "space", ...) are used.

        channel_names:

            The names of the channels in the dataset. If None, there must not be any
            channels. If channels are present and no channel names are provided an exception
            will be thrown.

        mode:

            The mode to open the dataset in.
            See https://zarr.readthedocs.io/en/stable/api/creation.html#zarr.creation.open_array

        kwargs:

            See additional arguments available here:
            https://czbiohub-sf.github.io/iohub/main/api/ngff.html#iohub.open_ome_zarr

    Returns:

        A :class:`Array` pointing to the newly created dataset.
    """

    assert not store.exists(), "Store already exists!"

    metadata = MetaData(
        Coordinate(shape),
        Coordinate(offset),
        Coordinate(voxel_size),
        list(axis_names) if axis_names is not None else None,
        list(units) if units is not None else None,
        list(types) if types is not None else None,
    )

    axis_metadata = [
        AxisMeta(name=n, type=t, unit=u)
        for n, t, u in zip(metadata.axis_names, metadata.types, metadata.ome_units)
    ]

    # create the dataset
    with open_ome_zarr(
        store, mode="w", layout="fov", axes=axis_metadata, channel_names=channel_names
    ) as ds:
        transforms = [
            TransformationMeta(type="scale", scale=metadata.ome_scale),
            TransformationMeta(type="translation", translation=metadata.ome_translate),
        ]

        ds.create_zeros(
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunk_shape,
            transform=transforms,
        )

    # open array
    array = open_ome_ds(store, name, mode="r+")

    return array
