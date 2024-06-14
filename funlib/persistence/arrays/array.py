from funlib.geometry import Coordinate, Roi
from .freezable import Freezable
from .adapters import Adapter
import numpy as np
import dask.array as da

from typing import Optional, Iterable, Any, Union


class Array(Freezable):
    """A thin wrapper around a dask array containing additional metadata such
    as the voxel size, offset, and axis names.

    Args:

        data (``np.ndarray``):

            The numpy array like object to wrap.

        offset (``Optional[Iterable[int]]``):

            The offset of the array in world units. Defaults to 0 for
            every dimension if not provided

        voxel_size (``Optional[Iterable[int]]``):

            The size of a voxel. If not provided the voxel size is
            assumed to be 1 in all dimensions.

        axis_names (``Optional[Iterable[str]]``):

            The name of each axis. If not provided, the axis names
            are given names ["c1", "c2", "c3", ...]

        chunk_shape (`tuple`, optional):

            The size of a chunk of the underlying data container in voxels.

        adapters (``Optional[Union[Adapter, list[Adapter]]]``):

            The adapter or list of adapters to use for this array.

    """

    data: Any
    voxel_size: Coordinate
    offset: Coordinate
    axis_names: list[str]
    units: list[str]
    chunk_shape: Coordinate
    adapter: list[Adapter]

    def __init__(
        self,
        data,
        offset: Optional[Iterable[int]] = None,
        voxel_size: Optional[Iterable[int]] = None,
        axis_names: Optional[Iterable[str]] = None,
        units: Optional[Iterable[str]] = None,
        chunk_shape: Optional[Iterable[int]] = None,
        adapter: Optional[Union[Adapter, Iterable[Adapter]]] = None,
    ):
        self.data = da.from_array(data)
        self.voxel_size = (
            Coordinate(voxel_size) if voxel_size is not None else (1,) * len(data.shape)
        )
        self.offset = (
            Coordinate(offset) if offset is not None else (0,) * len(data.shape)
        )
        self.axis_names = (
            tuple(axis_names)
            if axis_names is not None
            else tuple("d{i}" for i in range(len(data.shape)))
        )
        self.units = (
            tuple(units) if units is not None else ("voxels",) * self.voxel_size.dims
        )
        self.chunk_shape = Coordinate(chunk_shape) if chunk_shape is not None else None
        self._source_data = data

        adapter = [] if adapter is None else adapter
        adapter = [adapter] if not isinstance(adapter, list) else adapter
        self.adapter = adapter

        for adapter in self.adapter:
            if not isinstance(adapter, slice):
                self.data = adapter(self.data)

        self.freeze()

        self.validate()

    @property
    def roi(self):
        """
        Get the Roi associated with this data.
        """

        return Roi(
            self.offset,
            self.voxel_size * Coordinate(self.shape[-self.voxel_size.dims :]),
        )

    @property
    def dims(self):
        return len(self.shape)

    @property
    def channel_dims(self):
        return self.dims - self.voxel_size.dims

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
        return len(self.adapter) == 0 or all(
            [
                isinstance(adapter, slice)
                or (
                    isinstance(adapter, Iterable)
                    and all([isinstance(a, slice) for a in adapter])
                )
                for adapter in self.adapter
            ]
        )

    def __getitem__(self, key) -> np.ndarray:
        """Get a sub-array or a single value.

        Args:

            key (`class:Roi` or `class:Coordinate`):

                The ROI specifying the sub-array or a coordinate for a single
                value.

        Returns:

            If ``key`` is a `class:Roi`, returns a `class:Array` that
            represents this ROI. This is a light-weight operation that does not
            access the actual data held by this array. If ``key`` is a
            `class:Coordinate`, the array value (possible multi-channel)
            closest to the coordinate is returned.
        """

        if isinstance(key, Roi):
            roi = key

            if not self.roi.contains(roi):
                raise IndexError(
                    "Requested roi %s is not contained in this array %s."
                    % (roi, self.roi)
                )

            return self.data[self.__slices(roi)].compute()

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
            roi = key

            if not self.roi.contains(roi):
                raise IndexError(
                    "Requested roi %s is not contained in this array %s."
                    % (roi, self.roi)
                )

            roi_slices = self.__slices(roi)

            self.data[roi_slices] = value

            da.store(self.data[roi_slices], self._source_data, regions=roi_slices)
        else:
            raise RuntimeError(
                "This array is not writeable since you have applied a custom callable "
                "adapter that may or may not be invertable."
            )

    def to_ndarray(self, roi, fill_value=0):
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
        data = np.zeros(
            self[self.roi].shape[: self.channel_dims] + shape, dtype=self.data.dtype
        )
        if fill_value != 0:
            data[:] = fill_value

        array = Array(data, roi.offset, self.voxel_size)

        shared_roi = self.roi.intersect(roi)

        if not shared_roi.empty:
            array[shared_roi] = self[shared_roi]

        return data

    def _combine_slices(
        self, *roi_slices: list[Union[list[slice], slice]]
    ) -> list[slice]:
        """Combine slices into a single slice."""
        roi_slices = [
            roi_slice if isinstance(roi_slice, tuple) else (roi_slice,)
            for roi_slice in roi_slices
        ]
        num_dims = max([len(roi_slice) for roi_slice in roi_slices])

        combined_slices = []
        for d in range(num_dims):
            dim_slices = [
                roi_slice[d] if len(roi_slice) > d else slice(None)
                for roi_slice in roi_slices
            ]

            slice_range = range(0, self.shape[d], 1)
            for s in dim_slices:
                slice_range = slice_range[s]
            if len(slice_range) == 0:
                return slice(0)
            elif slice_range.stop < 0:
                return slice(slice_range.start, None, slice_range.step)
            combined_slices.append(
                slice(slice_range.start, slice_range.stop, slice_range.step)
            )

        return tuple(combined_slices)

    def __slices(self, roi, check_chunk_align=False):
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

        roi_slices = (slice(None),) * self.channel_dims + voxel_roi.to_slices()

        adapter_slices = [
            adapter for adapter in self.adapter if isinstance(adapter, slice)
        ]

        combined_slice = self._combine_slices(roi_slices, *adapter_slices)

        return combined_slice

    def __index(self, coordinate):
        """Get the voxel slices for the given coordinate."""

        index = tuple((coordinate - self.offset) / self.voxel_size)
        if self.channel_dims > 0:
            index = (Ellipsis,) + index
        return index

    def validate(self):
        assert self.voxel_size.dims == self.offset.dims == len(self.units), (
            f"The number of dimensions given by the voxel size ({self.voxel_size}), "
            f"offset ({self.offset}), and units ({self.units}) must match"
        )
        assert len(self.axis_names) == len(self.shape), (
            f"Axis names must be provided for every dimension. Got ({self.axis_names})"
            f"but expected {len(self.shape)} to match the data shape: {self.shape}"
        )
        if self.chunk_shape is not None:
            assert self.chunk_shape.dims == len(self.shape), (
                f"Chunk shape ({self.chunk_shape}) must have the same "
                f"number of dimensions as the data ({self.shape})"
            )
