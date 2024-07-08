from funlib.geometry import Coordinate, Roi
from .freezable import Freezable
from .adapters import Adapter
import numpy as np
import dask.array as da
from functools import reduce
from dask.array.optimization import fuse_slice

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

        units (``Optional[Iterable[str]]``):

            The units of each spatial dimension.

        chunk_shape (`tuple`, optional):

            The size of a chunk of the underlying data container in voxels.

        adapter (``Optional[Adapter]``):

            The adapter to use for this array. If you would like apply multiple
            adapters, please look into either the `.adapt` method or the
            `SequentialAdapter` class.

    """

    data: Any
    _voxel_size: Coordinate
    _offset: Coordinate
    _axis_names: list[str]
    _units: list[str]
    chunk_shape: Coordinate
    adapter: Adapter

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
        self._uncollapsed_dims = [True for _ in self.data.shape]
        self.voxel_size = (
            voxel_size if voxel_size is not None else (1,) * len(data.shape)
        )
        self.offset = offset if offset is not None else (0,) * len(data.shape)
        self.axis_names = (
            axis_names
            if axis_names is not None
            else tuple(f"c{i}^" for i in range(self.channel_dims))
            + tuple(f"d{i}" for i in range(self.voxel_size.dims))
        )
        self.units = units if units is not None else ("",) * self.voxel_size.dims
        self.chunk_shape = Coordinate(chunk_shape) if chunk_shape is not None else None
        self._source_data = data

        if adapter is not None:
            self.apply_adapter(adapter)

        adapters = [] if adapter is None else [adapter]
        self.adapters = adapters

        self.freeze()

        self.validate()

    def uncollapsed_dims(self, physical: bool = False) -> list[bool]:
        if physical:
            return self._uncollapsed_dims[-self._voxel_size.dims :]
        else:
            return self._uncollapsed_dims

    @property
    def offset(self) -> Coordinate:
        """Get the offset of this array in world units."""
        return Coordinate(
            [
                self._offset[ii]
                for ii, uncollapsed in enumerate(self.uncollapsed_dims(physical=True))
                if uncollapsed
            ]
        )

    @offset.setter
    def offset(self, offset: Iterable[int]) -> None:
        self._offset = Coordinate(offset)

    @property
    def voxel_size(self) -> Coordinate:
        """Get the size of a voxel in world units."""
        return Coordinate(
            [
                self._voxel_size[ii]
                for ii, uncollapsed in enumerate(self.uncollapsed_dims(physical=True))
                if uncollapsed
            ]
        )

    @voxel_size.setter
    def voxel_size(self, voxel_size: Iterable[int]) -> None:
        self._voxel_size = Coordinate(voxel_size)

    @property
    def units(self) -> list[str]:
        return [
            self._units[ii]
            for ii, uncollapsed in enumerate(self.uncollapsed_dims(physical=True))
            if uncollapsed
        ]

    @units.setter
    def units(self, units: list[str]) -> None:
        self._units = list(units)

    @property
    def axis_names(self) -> list[str]:
        return [
            self._axis_names[ii]
            for ii, uncollapsed in enumerate(self.uncollapsed_dims(physical=False))
            if uncollapsed
        ]

    @axis_names.setter
    def axis_names(self, axis_names):
        self._axis_names = list(axis_names)

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
        return len(self.adapters) == 0 or all(
            [self._is_slice(adapter) for adapter in self.adapters]
        )

    def apply_adapter(self, adapter: Adapter):
        if self._is_slice(adapter):
            if not isinstance(adapter, tuple):
                adapter = (adapter,)
            for ii, a in enumerate(adapter):
                if isinstance(a, int):
                    self._uncollapsed_dims[ii] = False
            self.data = self.data[adapter]
        elif callable(adapter):
            self.data = adapter(self.data)
        else:
            raise Exception(
                f"Adapter {adapter} is not a supported adapter. "
                f"Supported adapters are: {Adapter}"
            )

    def adapt(self, adapter: Adapter):
        """Apply an adapter to this array.

        Args:

            adapter (``Adapter``):

                The adapter to apply to this array.
        """
        self.apply_adapter(adapter)
        self.adapters.append(adapter)

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

            return self.data[self.__slices(roi, use_adapters=False)].compute()

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

                roi_slices = self.__slices(roi, use_adapters=False)
                self.data[roi_slices] = value

                region_slices = self.__slices(roi)

                da.store(
                    self.data[roi_slices], self._source_data, regions=region_slices
                )
            else:
                self.data[key] = value

                adapter_slices = [
                    adapter for adapter in self.adapters if self._is_slice(adapter)
                ]

                region_slices = reduce(fuse_slice, [*adapter_slices, key])

                da.store(self.data[key], self._source_data, regions=region_slices)

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

    def __slices(self, roi, use_adapters: bool = True, check_chunk_align: bool = False):
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

        adapter_slices = (
            [adapter for adapter in self.adapters if self._is_slice(adapter)]
            if use_adapters
            else []
        )

        combined_slice = reduce(fuse_slice, [*adapter_slices, roi_slices])

        return combined_slice

    def _is_slice(self, adapter: Adapter):
        if (
            isinstance(adapter, slice)
            or isinstance(adapter, int)
            or isinstance(adapter, list)
        ):
            return True
        elif isinstance(adapter, tuple) and all([self._is_slice(a) for a in adapter]):
            return True
        elif isinstance(adapter, np.ndarray) and adapter.dtype == bool:
            return True
        return False

    def __index(self, coordinate):
        """Get the voxel slices for the given coordinate."""

        index = tuple((coordinate - self.offset) / self.voxel_size)
        if self.channel_dims > 0:
            index = (Ellipsis,) + index
        return index

    def validate(self):
        assert self._voxel_size.dims == self._offset.dims == len(self._units), (
            f"The number of dimensions given by the voxel size ({self._voxel_size}), "
            f"offset ({self._offset}), and units ({self.units}) must match"
        )
        assert len(self._axis_names) == len(self._source_data.shape), (
            f"Axis names must be provided for every dimension. Got ({self._axis_names})"
            f"but expected {len(self.shape)} to match the data shape: {self.shape}"
        )
        if self.chunk_shape is not None:
            assert self.chunk_shape.dims == len(self._source_data.shape), (
                f"Chunk shape ({self.chunk_shape}) must have the same "
                f"number of dimensions as the data ({self.shape})"
            )
