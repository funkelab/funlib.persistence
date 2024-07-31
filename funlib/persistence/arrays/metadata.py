from collections.abc import Iterable
from typing import Any, Optional

from pydantic import BaseModel

from funlib.geometry import Coordinate


class PydanticCoordinate(Coordinate):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, val_info):
        return Coordinate(*v)


class MetaDataFormat(BaseModel):
    offset_attr: str = "offset"
    voxel_size_attr: str = "voxel_size"
    axis_names_attr: str = "axis_names"
    units_attr: str = "units"
    nesting_delimiter: str = "/"
    trim_channel_transforms: bool = True

    def fetch(self, data: dict[str, Any], keys: Iterable[str]):
        current_key, *keys = keys
        try:
            current_key = int(current_key)
        except ValueError:
            pass
        if isinstance(current_key, int):
            return self.fetch(data[current_key], keys)
        if len(keys) == 0:
            return data.get(current_key, None)
        elif isinstance(data, list):
            assert current_key == "{dim}", current_key
            values = []
            for sub_data in data:
                try:
                    values.append(self.fetch(sub_data, keys))
                except KeyError:
                    values.append(None)
            return values
        else:
            return self.fetch(data[current_key], keys)

    def parse(
        self,
        shape,
        data: dict[str, Any],
        offset=None,
        voxel_size=None,
        axis_names=None,
        units=None,
    ):
        offset = (
            offset
            if offset is not None
            else self.fetch(data, self.offset_attr.split(self.nesting_delimiter))
        )
        voxel_size = (
            voxel_size
            if voxel_size is not None
            else self.fetch(data, self.voxel_size_attr.split(self.nesting_delimiter))
        )
        axis_names = (
            axis_names
            if axis_names is not None
            else self.fetch(data, self.axis_names_attr.split(self.nesting_delimiter))
        )
        units = (
            units
            if units is not None
            else self.fetch(data, self.units_attr.split(self.nesting_delimiter))
        )

        if self.trim_channel_transforms and axis_names is not None:
            channel_dims = [True if "^" in axis else False for axis in axis_names]
            if sum(channel_dims) > 0:
                if offset is not None and len(offset) == len(channel_dims):
                    offset = [
                        o
                        for o, channel_dim in zip(offset, channel_dims)
                        if not channel_dim
                    ]
                if voxel_size is not None and len(voxel_size) == len(channel_dims):
                    voxel_size = [
                        v
                        for v, channel_dim in zip(voxel_size, channel_dims)
                        if not channel_dim
                    ]
                if units is not None and len(units) == len(channel_dims):
                    units = [
                        u
                        for u, channel_dim in zip(units, channel_dims)
                        if not channel_dim
                    ]

        offset = Coordinate(offset) if offset is not None else None
        voxel_size = Coordinate(voxel_size) if voxel_size is not None else None
        axis_names = list(axis_names) if axis_names is not None else None
        units = list(units) if units is not None else None

        metadata = MetaData(
            shape=shape,
            offset=offset,
            voxel_size=voxel_size,
            axis_names=axis_names,
            units=units,
        )
        metadata.validate()

        return metadata


class MetaData:
    def __init__(
        self,
        shape: PydanticCoordinate,
        offset: Optional[PydanticCoordinate] = None,
        voxel_size: Optional[PydanticCoordinate] = None,
        axis_names: Optional[list[str]] = None,
        units: Optional[list[str]] = None,
    ):
        self._offset = offset
        self._voxel_size = voxel_size
        self._axis_names = axis_names
        self._units = units
        self.shape = shape

        self.validate()

    @property
    def offset(self) -> PydanticCoordinate:
        return (
            self._offset
            if self._offset is not None
            else Coordinate((0,) * self.physical_dims)
        )

    @property
    def voxel_size(self) -> PydanticCoordinate:
        return (
            self._voxel_size
            if self._voxel_size is not None
            else Coordinate((1,) * self.physical_dims)
        )

    @property
    def axis_names(self) -> list[str]:
        return (
            self._axis_names
            if self._axis_names is not None
            else [f"c{dim}^" for dim in range(self.channel_dims)]
            + [f"d{dim}" for dim in range(self.physical_dims)]
        )

    @property
    def units(self) -> list[str]:
        return self._units if self._units is not None else [""] * self.physical_dims

    @property
    def dims(self) -> int:
        return len(self.shape)

    @property
    def physical_dims(self) -> int:
        physical_dim_indicators = [
            len(self._units) if self._units is not None else None,
            self._voxel_size.dims if self._voxel_size is not None else None,
            self._offset.dims if self._offset is not None else None,
            len([name for name in self._axis_names if "^" not in name])
            if self._axis_names is not None
            else None,
        ]
        potential_physical_dims = set(physical_dim_indicators)
        potential_physical_dims.discard(None)
        if len(potential_physical_dims) == 0:
            return self.dims
        elif len(potential_physical_dims) == 1:
            return potential_physical_dims.pop()
        else:
            raise ValueError(
                "Given physical dimensions are ambiguous:\n"
                f"Units: {self._units}\n"
                f"Voxel size: {self._voxel_size}\n"
                f"Offset: {self._offset}\n"
                f"Axis names: {self._axis_names}"
            )

    @property
    def channel_dims(self):
        return self.dims - self.physical_dims

    def validate(self):
        assert self.dims == self.physical_dims + self.channel_dims
