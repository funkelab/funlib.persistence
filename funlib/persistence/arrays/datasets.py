from .array import Array

from funlib.geometry import Coordinate, Roi

import zarr
import h5py
import json
import logging
import os
import shutil
from typing import Optional, Union

logger = logging.getLogger(__name__)


def separate_store_path(store, path):
    """
    sometimes you can pass a total os path to node, leading to
    an empty('') node.path attribute.
    the correct way is to separate path to container(.n5, .zarr)
    from path to array within a container.

    Args:
        store (string): path to store
        path (string): path array/group (.n5 or .zarr)

    Returns:
        (string, string): returns regularized store and group/array path
    """
    new_store, path_prefix = os.path.split(store)
    if ".zarr" in path_prefix or ".n5" in path_prefix:
        return store, path
    return separate_store_path(new_store, os.path.join(path_prefix, path))


def access_parent(node):
    """
    Get the parent (zarr.Group) of an input zarr array(ds).


    Args:
        node (zarr.core.Array or zarr.hierarchy.Group): _description_

    Raises:
        RuntimeError: returned if the node array is in the parent group,
        or the group itself is the root group

    Returns:
        zarr.hierarchy.Group : parent group that contains input group/array
    """

    store_path, node_path = separate_store_path(node.store.path, node.path)
    if node_path == "":
        raise RuntimeError(
            f"{node.name} is in the root group of the {node.store.path} store."
        )
    else:
        return zarr.open(store=store_path, path=os.path.split(node_path)[0], mode="r")


def check_for_multiscale(group):
    """check if multiscale attribute exists in the input group and for any parent level group

    Args:
        group (zarr.hierarchy.Group): group to check

    Returns:
        tuple({}, zarr.hierarchy.Group): (multiscales attribute body, zarr group where multiscales was found)
    """
    multiscales = group.attrs.get("multiscales", None)

    if multiscales:
        return (multiscales, group)

    if group.path == "":
        return (multiscales, group)

    return check_for_multiscale(access_parent(group))


# check if voxel_size value is present in .zatts other than in multiscale attribute
def check_for_voxel_size(array, order):
    """checks specific attributes(resolution, scale,
        pixelResolution["dimensions"], transform["scale"]) for voxel size
        value in the parent directory of the input array

    Args:
        array (zarr.core.Array): array to check
        order (string): colexicographical/lexicographical order
    Raises:
        ValueError: raises value error if no voxel_size value is found

    Returns:
       [float] : returns physical size of the voxel (unitless)
    """

    voxel_size = None
    parent_group = access_parent(array)
    for item in [array, parent_group]:

        if "resolution" in item.attrs:
            return item.attrs["resolution"]
        elif "scale" in item.attrs:
            return item.attrs["scale"]
        elif "pixelResolution" in item.attrs:
            return item.attrs["pixelResolution"]["dimensions"]
        elif "transform" in item.attrs:
            # Davis saves transforms in C order regardless of underlying
            # memory format (i.e. n5 or zarr). May be explicitly provided
            # as transform.ordering
            transform_order = item.attrs["transform"].get("ordering", "C")
            voxel_size = item.attrs["transform"]["scale"]
            if transform_order != order:
                voxel_size = voxel_size[::-1]
            return voxel_size

    return voxel_size


# check if offset value is present in .zatts other than in multiscales
def check_for_offset(array, order):
    """checks specific attributes(offset, transform["translate"]) for offset
        value in the parent directory of the input array

    Args:
        array (zarr.core.Array): array to check
        order (string): colexicographical/lexicographical order
    Raises:
        ValueError: raises value error if no offset value is found

    Returns:
       [float] : returns offset of the voxel (unitless) in respect to
                the center of the coordinate system
    """
    offset = None
    parent_group = access_parent(array)
    for item in [array, parent_group]:

        if "offset" in item.attrs:
            offset = item.attrs["offset"]
            return offset

        elif "transform" in item.attrs:
            transform_order = item.attrs["transform"].get("ordering", "C")
            offset = item.attrs["transform"]["translate"]
            if transform_order != order:
                offset = offset[::-1]
            return offset

    return offset


def check_for_units(array, order):
    """checks specific attributes(units, pixelResolution["unit"] transform["units"])
        for units(nm, cm, etc.) value in the parent directory of the input array

    Args:
        array (zarr.core.Array): array to check
        order (string): colexicographical/lexicographical order
    Raises:
        ValueError: raises value error if no units value is found

    Returns:
       [string] : returns units for the voxel_size
    """

    units = None
    parent_group = access_parent(array)
    for item in [array, parent_group]:

        if "units" in item.attrs:
            return item.attrs["units"]
        elif "pixelResolution" in item.attrs:
            unit = item.attrs["pixelResolution"]["unit"]
            return [unit for _ in range(len(array.shape))]
        elif "transform" in item.attrs:
            # Davis saves transforms in C order regardless of underlying
            # memory format (i.e. n5 or zarr). May be explicitly provided
            # as transform.ordering
            transform_order = item.attrs["transform"].get("ordering", "C")
            units = item.attrs["transform"]["units"]
            if transform_order != order:
                units = units[::-1]
            return units

    if units is None:
        Warning(
            f"No units attribute was found for {type(array.store)} store. Using pixels."
        )
        return "pixels"


def check_for_attrs_multiscale(ds, multiscale_group, multiscales):
    """checks multiscale attribute of the .zarr or .n5 group
        for voxel_size(scale), offset(translation) and units values

    Args:
        ds (zarr.core.Array): input zarr Array
        multiscale_group (zarr.hierarchy.Group): the group attrs
                                                that contains multiscale
        multiscales ({}): dictionary that contains all the info necessary
                            to create multiscale resolution pyramid

    Returns:
        ([float],[float],[string]): returns (voxel_size, offset, physical units)
    """

    voxel_size = None
    offset = None
    units = None

    if multiscales is not None:
        logger.info("Found multiscales attributes")
        scale = os.path.relpath(
            separate_store_path(ds.store.path, ds.path)[1], multiscale_group.path
        )
        if isinstance(ds.store, (zarr.n5.N5Store, zarr.n5.N5FSStore)):
            for level in multiscales[0]["datasets"]:
                if level["path"] == scale:

                    voxel_size = level["transform"]["scale"]
                    offset = level["transform"]["translate"]
                    units = level["transform"]["units"]
                    return voxel_size, offset, units
        # for zarr store
        else:
            units = [item["unit"] for item in multiscales[0]["axes"]]
            for level in multiscales[0]["datasets"]:
                if level["path"].lstrip("/") == scale:
                    for attr in level["coordinateTransformations"]:
                        if attr["type"] == "scale":
                            voxel_size = attr["scale"]
                        elif attr["type"] == "translation":
                            offset = attr["translation"]
                    return voxel_size, offset, units

    return voxel_size, offset, units


def _read_attrs(ds, order="C"):
    """check n5/zarr metadata and returns voxel_size, offset, physical units,
        for the input zarr array(ds)

    Args:
        ds (zarr.core.Array): input zarr array
        order (str, optional): _description_. Defaults to "C".

    Raises:
        TypeError: incorrect data type of the input(ds) array.
        ValueError: returns value error if no multiscale attribute was found
    Returns:
        _type_: _description_
    """
    voxel_size = None
    offset = None
    units = None
    multiscales = None

    if not isinstance(ds, zarr.core.Array):
        raise TypeError(
            f"{os.path.join(ds.store.path, ds.path)} is not zarr.core.Array"
        )

    # check recursively for multiscales attribute in the zarr store tree
    multiscales, multiscale_group = check_for_multiscale(group=access_parent(ds))

    # check for attributes in .zarr group multiscale
    if not isinstance(ds.store, (zarr.n5.N5Store, zarr.n5.N5FSStore)):
        if multiscales:
            voxel_size, offset, units = check_for_attrs_multiscale(
                ds, multiscale_group, multiscales
            )

    # if multiscale attribute is missing
    if voxel_size is None:
        voxel_size = check_for_voxel_size(ds, order)
    if offset is None:
        offset = check_for_offset(ds, order)
    if units is None:
        units = check_for_units(ds, order)

    dims = len(ds.shape)
    dims = dims if dims <= 3 else 3

    if voxel_size is not None and offset is not None and units is not None:
        if order == "F" or isinstance(ds.store, (zarr.n5.N5Store, zarr.n5.N5FSStore)):
            return voxel_size[::-1], offset[::-1], units[::-1]
        else:
            return voxel_size, offset, units

    # if no voxel offset are found in transform, offset or scale, check in n5 multiscale attribute:
    if (
        isinstance(ds.store, (zarr.n5.N5Store, zarr.n5.N5FSStore))
        and multiscales != False
    ):

        voxel_size, offset, units = check_for_attrs_multiscale(
            ds, multiscale_group, multiscales
        )

    # return default value if an attribute was not found
    if voxel_size is None:
        voxel_size = (1,) * dims
        Warning(f"No voxel_size attribute was found. Using {voxel_size} as default.")
    if offset is None:
        offset = (0,) * dims
        Warning(f"No offset attribute was found. Using {offset} as default.")
    if units is None:
        units = "pixels"
        Warning(f"No units attribute was found. Using {units} as default.")

    if order == "F":
        return voxel_size[::-1], offset[::-1], units[::-1]
    else:
        return voxel_size, offset, units


def regularize_offset(voxel_size_float, offset_float):
    """
        offset is not a multiple of voxel_size. This is often due to someone defining
        offset to the point source of each array element i.e. the center of the rendered
        voxel, vs the offset to the corner of the voxel.
        apparently this can be a heated discussion. See here for arguments against
        the convention we are using: http://alvyray.com/Memos/CG/Microsoft/6_pixel.pdf

    Args:
        voxel_size_float ([float]): float voxel size list
        offset_float ([float]): float offset list
    Returns:
        (Coordinate, Coordinate)): returned offset size that is multiple of voxel size
    """
    voxel_size, offset = Coordinate(voxel_size_float), Coordinate(offset_float)

    if voxel_size is not None and (offset / voxel_size) * voxel_size != offset:

        logger.debug(
            f"Offset: {offset} being rounded to nearest voxel size: {voxel_size}"
        )
        offset = (
            (Coordinate(offset) + (Coordinate(voxel_size) / 2)) / Coordinate(voxel_size)
        ) * Coordinate(voxel_size)
        logger.debug(f"Rounded offset: {offset}")

    return Coordinate(voxel_size), Coordinate(offset)


def _read_voxel_size_offset(ds, order="C"):

    voxel_size, offset, units = _read_attrs(ds, order)

    return regularize_offset(voxel_size, offset)


def open_ds(filename: str, ds_name: str, mode: str = "r") -> Array:
    """Open a Zarr, N5, or HDF5 dataset as an :class:`Array`. If the
    dataset has attributes ``resolution`` and ``offset``, those will be
    used to determine the meta-information of the returned array.

    Args:

        filename:

            The name of the container "file" (which is a directory for Zarr and
            N5).

        ds_name:

            The name of the dataset to open.

    Returns:

        A :class:`Array` pointing to the dataset.
    """

    if filename.endswith(".zarr") or filename.endswith(".zip"):
        assert (
            not filename.endswith(".zip") or mode == "r"
        ), "Only reading supported for zarr ZipStore"

        logger.debug("opening zarr dataset %s in %s", ds_name, filename)
        try:
            ds = zarr.open(filename, mode=mode)[ds_name]
        except Exception as e:
            logger.error("failed to open %s/%s" % (filename, ds_name))
            raise e

        try:
            order = ds.attrs["order"]
        except KeyError:
            order = ds.order
        voxel_size, offset = _read_voxel_size_offset(ds, order)
        shape = Coordinate(ds.shape[-len(voxel_size) :])
        roi = Roi(offset, voxel_size * shape)

        chunk_shape = ds.chunks

        logger.debug("opened zarr dataset %s in %s", ds_name, filename)
        return Array(ds, roi, voxel_size, chunk_shape=chunk_shape)

    elif filename.endswith(".n5"):
        logger.debug("opening N5 dataset %s in %s", ds_name, filename)
        ds = zarr.open(filename, mode=mode)[ds_name]

        voxel_size, offset = _read_voxel_size_offset(ds, "F")
        shape = Coordinate(ds.shape[-len(voxel_size) :])
        roi = Roi(offset, voxel_size * shape)

        chunk_shape = ds.chunks

        logger.debug("opened N5 dataset %s in %s", ds_name, filename)
        return Array(ds, roi, voxel_size, chunk_shape=chunk_shape)

    elif filename.endswith(".h5") or filename.endswith(".hdf"):
        logger.debug("opening H5 dataset %s in %s", ds_name, filename)
        ds = h5py.File(filename, mode=mode)[ds_name]

        voxel_size, offset = _read_voxel_size_offset(ds, "C")
        shape = Coordinate(ds.shape[-len(voxel_size) :])
        roi = Roi(offset, voxel_size * shape)

        chunk_shape = ds.chunks

        logger.debug("opened H5 dataset %s in %s", ds_name, filename)
        return Array(ds, roi, voxel_size, chunk_shape=chunk_shape)

    elif filename.endswith(".json"):
        logger.debug("found JSON container spec")
        with open(filename, "r") as f:
            spec = json.load(f)

        array = open_ds(spec["container"], ds_name, mode)
        return Array(
            array.data,
            Roi(spec["offset"], spec["size"]),
            array.voxel_size,
            array.roi.begin,
            chunk_shape=array.chunk_shape,
        )

    else:
        logger.error("don't know data format of %s in %s", ds_name, filename)
        raise RuntimeError("Unknown file format for %s" % filename)


def prepare_ds(
    filename: str,
    ds_name: str,
    total_roi: Roi,
    voxel_size: Coordinate,
    dtype,
    write_roi: Optional[Roi] = None,
    write_size: Optional[Coordinate] = None,
    num_channels: Optional[int] = None,
    compressor: Union[str, dict] = "default",
    delete: bool = False,
    force_exact_write_size: bool = False,
    multiscales_metadata=False,
    units: str = "nanometer",
    axes: list = ["z", "y", "x"],
) -> Array:
    """Prepare a Zarr or N5 dataset.

    Args:

        filename:

            The name of the container "file" (which is actually a directory).

        ds_name:

            The name of the dataset to prepare.

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

    voxel_size = Coordinate(voxel_size)
    if write_size is not None:
        write_size = Coordinate(write_size)

    assert total_roi.shape.is_multiple_of(
        voxel_size
    ), "The provided ROI shape is not a multiple of voxel_size"
    assert total_roi.begin.is_multiple_of(
        voxel_size
    ), "The provided ROI offset is not a multiple of voxel_size"

    if write_roi is not None:
        logger.warning("write_roi is deprecated, please use write_size instead")

        if write_size is None:
            write_size = write_roi.shape

    if write_size is not None:
        assert write_size.is_multiple_of(
            voxel_size
        ), f"The provided write size ({write_size}) is not a multiple of voxel_size ({voxel_size})"

    if compressor == "default":
        compressor = {"id": "gzip", "level": 5}

    ds_name = ds_name.lstrip("/")

    if filename.endswith(".h5") or filename.endswith(".hdf"):
        raise RuntimeError("prepare_ds does not support HDF5 files")
    elif filename.endswith(".zarr"):
        file_format = "zarr"
    elif filename.endswith(".n5"):
        file_format = "n5"
    else:
        raise RuntimeError("Unknown file format for %s" % filename)

    if write_size is not None:
        if not force_exact_write_size:
            chunk_shape = get_chunk_shape(write_size / voxel_size)
        else:
            chunk_shape = write_size / voxel_size
    else:
        chunk_shape = None

    shape = tuple(total_roi.shape / voxel_size)

    if num_channels is not None:
        shape = (num_channels,) + shape

        if chunk_shape is not None:
            chunk_shape = Coordinate((num_channels,) + chunk_shape)
        voxel_size_with_channels = Coordinate((1,) + voxel_size)

    if not os.path.isdir(filename):
        logger.debug("Creating new %s", filename)
        os.makedirs(filename)

        zarr.open(filename, mode="w")

    if not os.path.isdir(os.path.join(filename, ds_name)):
        logger.debug(
            "Creating new %s in %s with chunk_size %s and write_size %s",
            ds_name,
            filename,
            chunk_shape,
            write_size,
        )

        if compressor is not None:
            compressor = zarr.get_codec(compressor)

        root = zarr.open(filename, mode="a")
        ds = root.create_dataset(
            ds_name,
            shape=shape,
            chunks=chunk_shape,
            dtype=dtype,
            compressor=compressor,
            overwrite=delete,
            dimension_separator="/",
        )

        if file_format == "zarr":
            ds.attrs["resolution"] = voxel_size
            ds.attrs["offset"] = total_roi.begin

            # add ome-zarr multiscales in the parent group of a newly created zarr array:
            if multiscales_metadata == True and "multiscales" not in root.attrs:
                add_multiscales_metadata(
                    root, ds_name, total_roi, voxel_size, units, axes
                )

        else:
            ds.attrs["resolution"] = voxel_size[::-1]
            ds.attrs["offset"] = total_roi.begin[::-1]

        if chunk_shape is not None:
            if num_channels is not None:
                chunk_shape = chunk_shape / voxel_size_with_channels
            else:
                chunk_shape = chunk_shape / voxel_size
        return Array(ds, total_roi, voxel_size, chunk_shape=chunk_shape)

    else:
        logger.debug("Trying to reuse existing dataset %s in %s...", ds_name, filename)
        ds = open_ds(filename, ds_name, mode="a")

        compatible = True

        if ds.shape != shape:
            logger.info("Shapes differ: %s vs %s", ds.shape, shape)
            compatible = False

        if ds.roi != total_roi:
            logger.info("ROIs differ: %s vs %s", ds.roi, total_roi)
            compatible = False

        if ds.voxel_size != voxel_size:
            logger.info("Voxel sizes differ: %s vs %s", ds.voxel_size, voxel_size)
            compatible = False

        if write_size is not None and ds.data.chunks != chunk_shape:
            logger.info("Chunk shapes differ: %s vs %s", ds.data.chunks, chunk_shape)
            compatible = False

        if dtype != ds.dtype:
            logger.info("dtypes differ: %s vs %s", ds.dtype, dtype)
            compatible = False

        if not compatible:
            if not delete:
                raise RuntimeError(
                    "Existing dataset is not compatible, please manually "
                    "delete the volume at %s/%s" % (filename, ds_name)
                )

            logger.info("Existing dataset is not compatible, creating new one")

            shutil.rmtree(os.path.join(filename, ds_name))
            return prepare_ds(
                filename=filename,
                ds_name=ds_name,
                total_roi=total_roi,
                voxel_size=voxel_size,
                dtype=dtype,
                write_size=write_size,
                num_channels=num_channels,
                compressor=compressor,
            )

        else:
            logger.info("Reusing existing dataset")
            return ds


def get_chunk_shape(block_shape):
    """Get a reasonable chunk size that divides the given block size."""

    chunk_shape = Coordinate(get_chunk_size_dim(b, 256) for b in block_shape)

    logger.debug("Setting chunk size to %s", chunk_shape)

    return chunk_shape


def get_chunk_size_dim(b, target_chunk_size):
    best_k = None
    best_target_diff = 0

    for k in range(1, b + 1):
        if ((b // k) * k) % b == 0:
            diff = abs(b // k - target_chunk_size)
            if best_k is None or diff < best_target_diff:
                best_target_diff = diff
                best_k = k

    return b // best_k


def add_multiscales_metadata(
    root: zarr.hierarchy.Group,
    ds_name: str,
    total_roi: Roi,
    voxel_size: Coordinate,
    units: str,
    axes: list,
):
    z_attrs: dict = {"multiscales": [{}]}
    z_attrs["multiscales"][0]["axes"] = [
        {"name": axis, "type": "space", "unit": units} for axis in axes
    ]
    z_attrs["multiscales"][0]["coordinateTransformations"] = [
        {"scale": [1.0, 1.0, 1.0], "type": "scale"}
    ]
    z_attrs["multiscales"][0]["datasets"] = [
        {
            "coordinateTransformations": [
                {"scale": voxel_size, "type": "scale"},
                {"translation": total_roi.begin, "type": "translation"},
            ],
            "path": ds_name,
        }
    ]

    z_attrs["multiscales"][0]["name"] = ""
    z_attrs["multiscales"][0]["version"] = "0.4"

    root.attrs.update(z_attrs)
