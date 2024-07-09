from collections.abc import Iterable
from typing import Any

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
            return data[current_key]
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

        if self.trim_channel_transforms:
            if len(axis_names) == len(units) == len(voxel_size) == len(offset):
                offset = [o for o, axis in zip(offset, axis_names) if "^" not in axis]
                voxel_size = [
                    v for v, axis in zip(voxel_size, axis_names) if "^" not in axis
                ]
                units = [u for u, axis in zip(units, axis_names) if "^" not in axis]

        metadata = MetaData(
            offset=Coordinate(offset),
            voxel_size=Coordinate(voxel_size),
            axis_names=list(axis_names),
            units=[unit if unit is not None else "" for unit in units],
        )
        metadata.validate()

        return metadata


class MetaData(BaseModel):
    offset: PydanticCoordinate
    voxel_size: PydanticCoordinate
    axis_names: list[str]
    units: list[str]

    def validate(self):
        assert self.voxel_size.dims == self.offset.dims == len(self.units), (
            f"The number of dimensions given by the voxel size ({self.voxel_size}), "
            f"offset ({self.offset}), and units ({self.units}) must match"
        )
