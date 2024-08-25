from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import toml
from pydantic import BaseModel

from funlib.geometry import Coordinate


class MetaDataFormat(BaseModel):
    offset_attr: str = "offset"
    voxel_size_attr: str = "voxel_size"
    axis_names_attr: str = "axis_names"
    units_attr: str = "units"

    class Config:
        extra = "forbid"

    def fetch(self, data: dict[str | int, Any], keys: Sequence[str]):
        current_key: str | int
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
        data: dict[str | int, Any],
        offset=None,
        voxel_size=None,
        axis_names=None,
        units=None,
    ):
        offset = (
            offset
            if offset is not None
            else self.fetch(data, self.offset_attr.split("/"))
        )
        voxel_size = (
            voxel_size
            if voxel_size is not None
            else self.fetch(data, self.voxel_size_attr.split("/"))
        )
        axis_names = (
            axis_names
            if axis_names is not None
            else self.fetch(data, self.axis_names_attr.split("/"))
        )
        units = (
            units if units is not None else self.fetch(data, self.units_attr.split("/"))
        )

        # remove channel dimensions from offset, voxel_size and units
        if axis_names is not None:
            channel_dims = [True if "^" in axis else False for axis in axis_names]
            if sum(channel_dims) > 0:
                if offset is not None and len(offset) == len(axis_names):
                    offset = [
                        o
                        for o, channel_dim in zip(offset, channel_dims)
                        if not channel_dim
                    ]
                if voxel_size is not None and len(voxel_size) == len(axis_names):
                    voxel_size = [
                        v
                        for v, channel_dim in zip(voxel_size, channel_dims)
                        if not channel_dim
                    ]
                if units is not None and len(units) == len(axis_names):
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
        shape: Coordinate,
        offset: Optional[Coordinate] = None,
        voxel_size: Optional[Coordinate] = None,
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
    def offset(self) -> Coordinate:
        return (
            self._offset
            if self._offset is not None
            else Coordinate((0,) * self.physical_dims)
        )

    @property
    def voxel_size(self) -> Coordinate:
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
            (
                len([name for name in self._axis_names if "^" not in name])
                if self._axis_names is not None
                else None
            ),
        ]
        potential_physical_dims = set(physical_dim_indicators)
        potential_physical_dims.discard(None)
        if len(potential_physical_dims) == 0:
            return self.dims
        elif len(potential_physical_dims) == 1:
            v = potential_physical_dims.pop()
            assert v is not None
            return v
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


DEFAULT_METADATA_FORMAT = MetaDataFormat()
LOCAL_PATHS = [Path("pyproject.toml"), Path("funlib_persistence.toml")]
USER_PATHS = [
    Path.home() / ".config" / "funlib_persistence" / "funlib_persistence.toml"
]
GLOBAL_PATHS = [Path("/etc/funlib_persistence/funlib_persistence.toml")]


def read_config() -> Optional[dict]:
    config = None
    config = {}
    for path in (LOCAL_PATHS + USER_PATHS + GLOBAL_PATHS)[::-1]:
        if path.exists():
            with open(path, "r") as f:
                conf = toml.load(f)
                if path.name == "pyproject.toml":
                    conf = conf.get("tool", {}).get("funlib_persistence", {})
                config.update(conf)
    return config


def set_default_metadata_format(metadata_format: MetaDataFormat):
    global DEFAULT_METADATA_FORMAT
    DEFAULT_METADATA_FORMAT = metadata_format


def get_default_metadata_format() -> MetaDataFormat:
    global DEFAULT_METADATA_FORMAT
    return DEFAULT_METADATA_FORMAT


def configure_library():
    config = read_config()
    if config:
        set_default_metadata_format(MetaDataFormat(**config))


# Call configure_library at the start of your library initialization
configure_library()
