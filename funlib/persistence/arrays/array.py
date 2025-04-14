import logging
from collections.abc import Sequence
from functools import reduce
from typing import Any, Optional, Union

import dask.array as da
import numpy as np
from dask.array.optimization import fuse_slice
from zarr import Array as ZarrArray

from funlib.geometry import Coordinate, Roi

from .freezable import Freezable
from .lazy_ops import LazyOp
from .metadata import MetaData
from .utils import interleave

logger = logging.getLogger(__name__)


class Array(Freezable):
    """A thin wrapper around a dask array containing additional metadata such
    as the voxel size, offset, and axis names.

    Args:

        data (``np.ndarray``):

            The numpy array like object to wrap.

        offset (``Optional[Sequence[int]]``):

            The offset of the array in world units. Defaults to 0 for
            every dimension if not provided

        voxel_size (``Optional[Sequence[int]]``):

            The size of a voxel. If not provided the voxel size is
            assumed to be 1 in all dimensions.

        axis_names (``Optional[Sequence[str]]``):

            The name of each axis. If not provided, the axis names
            are given names ["c1", "c2", "c3", ...]

        units (``Optional[Sequence[str]]``):

            The units of each spatial dimension.

        types (``Optional[Sequence[str]]``):

            The type of each dimension. Can be "space", "channel", or "time".
            We treat both "space" and "time" as spatial dimensions for indexing with
            Roi and Coordinate classes in world units.
            If not provided, we fall back on the axis names and assume "channel"
            for any axis name that ends with "^" and "space" otherwise. If neither
            are provided we assume "space" for all dimensions.
            Note that the axis name parsing is depricated and will be removed in
            future versions and we highly recommend providing the types directly.

        chunks (`tuple[int]` or `str` or `int`, optional):

            See https://docs.dask.org/en/stable/generated/dask.array.from_array.html for
            details.

        lazy_op (``Optional[LazyOp]``):

            The lazy_op to use for this array. If you would like apply multiple
            lazy_ops, please look into either the `.lazy_op` method or the
            `SequentialLazyOp` class.

        strict_metadata (``bool``):

            If True, then all metadata fields must be provided, including
            offset, voxel_size, axis_names, units, and types. Metadata
            can either be passed in or read from array attributes.

    """

    data: da.Array

    def __init__(
        self,
        data,
        offset: Optional[Sequence[int]] = None,
        voxel_size: Optional[Sequence[int]] = None,
        axis_names: Optional[Sequence[str]] = None,
        units: Optional[Sequence[str]] = None,
        types: Optional[Sequence[str]] = None,
        chunks: Optional[Union[int, Sequence[int], str]] = "auto",
        lazy_op: Optional[LazyOp] = None,
        strict_metadata: bool = False,
    ):
        if not isinstance(data, da.Array):
            self.data = da.from_array(data, chunks=chunks)
        else:
            self.data = data
        self._uncollapsed_dims = [True for _ in self.data.shape]
        self._source_data = data
        self._metadata = MetaData(
            offset=Coordinate(offset) if offset is not None else None,
            voxel_size=Coordinate(voxel_size) if voxel_size is not None else None,
            axis_names=list(axis_names) if axis_names is not None else None,
            units=list(units) if units is not None else None,
            types=list(types) if types is not None else None,
            shape=self._source_data.shape,
            strict=strict_metadata,
        )

        # used for custom metadata unrelated to indexing with physical units
        # only used if not reading from zarr and there is no built in `.attrs`
        self._attrs: dict[str, Any] = {}

        if lazy_op is not None:
            self.apply_lazy_ops(lazy_op)

        lazy_ops = [] if lazy_op is None else [lazy_op]
        self.lazy_ops = lazy_ops

        self.freeze()

        self.validate(strict_metadata)

    @property
    def attrs(self) -> dict:
        """
        Return dict that can be used to store custom metadata. Will be persistent
        for zarr arrays. If reading from zarr, any existing metadata (such as
        voxel_size, axis_names, etc.) will also be exposed here.
        """
        if isinstance(self._source_data, ZarrArray):
            return self._source_data.attrs
        else:
            return self._attrs

    @property
    def chunk_shape(self) -> Coordinate:
        return Coordinate(self.data.chunksize)

    def uncollapsed_dims(self, physical: bool = False) -> list[bool]:
        """
        We support lazy slicing of arrays, such as `x = Array(np.ones(5,5,5))` and
        `x.lazy_op(np.s![0, :, :])`. This will result in the first dimension being
        collapsed and future slicing operations will need to be 2D.
        We use `_uncollapsed_dims` to keep track of which dimensions are sliceable.
        """
        if physical:
            return [
                x
                for x, t in zip(self._uncollapsed_dims, self._metadata.types)
                if t in ["space", "time"]
            ]
        else:
            return self._uncollapsed_dims

    @property
    def offset(self) -> Coordinate:
        """Get the offset of this array in world units."""
        return Coordinate(
            [
                self._metadata.offset[ii]
                for ii, uncollapsed in enumerate(self.uncollapsed_dims(physical=True))
                if uncollapsed
            ]
        )

    @property
    def voxel_size(self) -> Coordinate:
        """Get the size of a voxel in world units."""
        return Coordinate(
            [
                self._metadata.voxel_size[ii]
                for ii, uncollapsed in enumerate(self.uncollapsed_dims(physical=True))
                if uncollapsed
            ]
        )

    @property
    def units(self) -> list[str]:
        return [
            self._metadata.units[ii]
            for ii, uncollapsed in enumerate(self.uncollapsed_dims(physical=True))
            if uncollapsed
        ]

    @property
    def axis_names(self) -> list[str]:
        return [
            self._metadata.axis_names[ii]
            for ii, uncollapsed in enumerate(self.uncollapsed_dims(physical=False))
            if uncollapsed
        ]

    @property
    def types(self) -> list[str]:
        return [
            self._metadata.types[ii]
            for ii, uncollapsed in enumerate(self.uncollapsed_dims(physical=False))
            if uncollapsed
        ]

    @property
    def physical_shape(self):
        return tuple(
            self.data.shape[ii]
            for ii, axis_type in enumerate(self.types)
            if axis_type in ["space", "time"]
        )

    @property
    def roi(self):
        """
        Get the Roi associated with this data.
        """
        return Roi(
            self.offset,
            self.voxel_size * Coordinate(self.physical_shape),
        )

    @property
    def dims(self):
        return sum(self.uncollapsed_dims())

    @property
    def channel_dims(self):
        return self.dims - self.voxel_size.dims

    @property
    def spatial_dims(self):
        return self.voxel_size.dims

    @property
    def shape(self):
        """Get the shape in voxels of this array,
        This is equivalent to::

            array.data.shape
        """

        return self.data.shape

    @property
    def dtype(self):
        """Get the dtype of this array."""
        return self.data.dtype

    @property
    def is_writeable(self):
        return len(self.lazy_ops) == 0 or all(
            [
                self._is_slice(lazy_op, writeable=True) or isinstance(lazy_op, Roi)
                for lazy_op in self.lazy_ops
            ]
        )

    def apply_lazy_ops(self, lazy_op):
        if self._is_slice(lazy_op):
            if not isinstance(lazy_op, tuple):
                lazy_op = (lazy_op,)
            for ii, a in enumerate(lazy_op):
                if isinstance(a, int):
                    for i, uc in enumerate(self._uncollapsed_dims):
                        if uc:
                            if ii == 0:
                                self._uncollapsed_dims[i] = False
                                break
                            ii -= 1
            self.data = self.data[lazy_op]
        elif isinstance(lazy_op, Roi):
            assert all(self.uncollapsed_dims(physical=True)), (
                "Lazily slicing with a Roi is not yet supported with some collapsed dimensions."
            )
            assert lazy_op.dims == self.spatial_dims, (
                "Must provide a Roi with the same number of dimensions as this array."
            )
            slices = self.__slices(lazy_op, use_lazy_slices=False)
            self.data = self.data[slices]
            self._metadata._offset = lazy_op.offset
        elif callable(lazy_op):
            self.data = lazy_op(self.data)
        else:
            raise Exception(
                f"LazyOp {lazy_op} is not a supported lazy_op. "
                f"Supported lazy_ops are: {LazyOp}"
            )

    def lazy_op(self, lazy_op: LazyOp):
        """Apply an lazy operation to this array.

        Args:

            lazy_op (``LazyOp``):

                The lazy operation to apply to this array. Common lazy operations include
                slicing (e.g. `array.lazy_op(np.s_[0, :, :])`), thresholding (e.g. `array.lazy_op(lambda d: d > 0.5)`),
                or normalizing (e.g. `array.lazy_op(lambda d: d / 255)`).
                Note that the lazy operation must come in the form of a slicing operation or a callable that
                takes a dask array and returns a dask array.
                Lazy operations that would change the shape of the array may make it impossible to properly query
                the data with `Coordinate` or `Roi` objects. In these cases, it may be more appropriate to recreate
                the array with the appropriate metadata. For example:
                `sub_array = Array(array.data[::2, ::2, ::2], array.offset, array.voxel_size * 2, ...)`
        """
        self.apply_lazy_ops(lazy_op)
        self.lazy_ops.append(lazy_op)

    def __getitem__(self, key) -> np.ndarray:
        """Get a sub-array or a single value.

        Args:

            key (`class:Roi` or `class:Coordinate` or any numpy compatible key):

                The `Roi`, `Coordinate`, or numpy compatible key indicating the data
                you want to read.

        Returns:

            A numpy array containing the data requested by the key. If using a `class:Roi`
            or `class:Coordinate`, the data will be returned in world units. If using
            a numpy compatible key, the data will be sliced as a regular numpy array.
        """

        if isinstance(key, Roi):
            roi = key

            if not self.roi.contains(roi):
                raise IndexError(
                    "Requested roi %s is not contained in this array %s."
                    % (roi, self.roi)
                )

            return self.data[self.__slices(roi, use_lazy_slices=False)].compute()

        elif isinstance(key, Coordinate):
            coordinate = key

            if not self.roi.contains(coordinate):
                raise IndexError("Requested coordinate is not contained in this array.")

            index = self.__index(coordinate)
            return self.data[index].compute()

        else:
            return self.data[key].compute()

    def __setitem__(self, key, value: np.ndarray):
        """Set the data of this array within the given ROI.

        Args:

            key (`class:Roi`):

                The ROI to write to.

            value (``ndarray``):

                The value to write.
        """

        if self.is_writeable:
            if isinstance(key, Roi):
                roi = key

                if not self.roi.contains(roi):
                    raise IndexError(
                        "Requested roi %s is not contained in this array %s."
                        % (roi, self.roi)
                    )

                region_slices = self.__slices(roi)
                self._source_data[region_slices] = value
            else:
                lazy_slices = [
                    lazy_op for lazy_op in self.lazy_ops if self._is_slice(lazy_op)
                ]

                region_slices = reduce(fuse_slice, [*lazy_slices, key])

                self._source_data[region_slices] = value

            # If the source data is an in-memory numpy array, writing to the numpy
            # array does not always result in the dask array reading the new data.
            # It seems to be a caching issue. To work around this, we create a new
            # dask array from the source data.
            if isinstance(self._source_data, np.ndarray):
                self.data = da.from_array(self._source_data)

        else:
            raise RuntimeError(
                "This array is not writeable since you have applied a custom callable "
                "lazy_op that may or may not be invertable, or you have used a"
                "boolean array. Please use a list of ints to specify the axes you "
                "want if you want to write to this array."
            )

    def to_ndarray(self, roi: Roi, fill_value=0) -> np.ndarray:
        """An alternative implementation of `__getitem__` that supports
        using fill values to request data that may extend outside the
        roi covered by self.

        Args:

            roi (`class:Roi`, optional):

                If given, copy only the data represented by this ROI. This is
                equivalent to::

                    array[roi].to_ndarray()

            fill_value (scalar, optional):

                The value to use to fill in values that are outside the ROI
                provided by this data. Defaults to 0.
        """

        shape = roi.shape / self.voxel_size
        data = np.zeros(self.shape[: self.channel_dims] + shape, dtype=self.data.dtype)
        if fill_value != 0:
            data[:] = fill_value

        array = Array(data, roi.offset, self.voxel_size)

        shared_roi = self.roi.intersect(roi)

        if not shared_roi.empty:
            array[shared_roi] = self[shared_roi]

        return data

    def __slices(
        self, roi, use_lazy_slices: bool = True, check_chunk_align: bool = False
    ):
        """Get the voxel slices for the given roi."""

        voxel_roi = (roi - self.offset) / self.voxel_size

        if check_chunk_align:
            for d in range(roi.dims):
                end_of_array = roi.get_end()[d] == self.roi.get_end()[d]

                begin_align_with_chunks = voxel_roi.begin[d] % self.chunk_shape[d] == 0
                shape_align_with_chunks = voxel_roi.shape[d] % self.chunk_shape[d] == 0

                assert begin_align_with_chunks and (
                    shape_align_with_chunks or end_of_array
                ), (
                    "ROI %s (in voxels: %s) does not align with chunks of "
                    "size %s (mismatch in dimension %d)"
                    % (roi, voxel_roi, self.chunk_shape, d)
                )

        roi_slices = tuple(
            interleave(
                voxel_roi.to_slices(),
                (slice(None),) * self.channel_dims,
                [axis_type in ["space", "time"] for axis_type in self.types],
            )
        )

        lazy_slices = (
            [lazy_op for lazy_op in self.lazy_ops if self._is_slice(lazy_op)]
            if use_lazy_slices
            else []
        )

        combined_slice = reduce(fuse_slice, [*lazy_slices, roi_slices])

        return combined_slice

    def _is_slice(self, lazy_op: LazyOp, writeable: bool = False) -> bool:
        if isinstance(lazy_op, slice) or isinstance(lazy_op, int):
            return True
        elif isinstance(lazy_op, list) and all([isinstance(a, int) for a in lazy_op]):
            return True
        elif isinstance(lazy_op, tuple) and all(
            [self._is_slice(a, writeable) for a in lazy_op]
        ):
            return True
        elif (
            isinstance(lazy_op, np.ndarray)
            and lazy_op.dtype == bool
            and lazy_op.ndim == 1
        ):
            # Boolean indexing is not supported when storing regions of dask arrays
            # because dask `fuse_slice` can't combine the boolean indexing with slicing operations
            return not writeable
        return False

    def __index(self, coordinate):
        """Get the voxel slices for the given coordinate."""

        index = tuple((coordinate - self.offset) / self.voxel_size)
        if self.channel_dims > 0:
            index = (Ellipsis,) + index
        return index

    def validate(self, strict: bool = False):
        self._metadata.validate(strict)
        assert len(self.axis_names) == len(self._source_data.shape), (
            f"Axis names must be provided for every dimension. Got ({self.axis_names}) "
            f"but expected {len(self.shape)} to match the data shape: {self.shape}"
        )
        if self.chunk_shape is not None:
            assert self.chunk_shape.dims == len(self._source_data.shape), (
                f"Chunk shape ({self.chunk_shape}) must have the same "
                f"number of dimensions as the data ({self.shape})"
            )
