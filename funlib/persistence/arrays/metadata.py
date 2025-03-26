from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional
import warnings

import toml
from pydantic import BaseModel

from funlib.geometry import Coordinate


class MetaData:
    def __init__(
        self,
        shape: Coordinate,
        offset: Optional[Coordinate] = None,
        voxel_size: Optional[Coordinate] = None,
        axis_names: Optional[list[str]] = None,
        units: Optional[list[str]] = None,
        types: Optional[list[str]] = None,
        strict: bool = False,
    ):
        self._offset = offset
        self._voxel_size = voxel_size
        self._axis_names = axis_names
        self._units = units
        self._types = types
        self.shape = shape

        self.validate(strict)

    def interleave_physical(
        self, physical: Sequence[int | str], non_physical: int | str | None
    ) -> Sequence[int | str | None]:
        interleaved: list[int | str | None] = []
        physical_ind = 0
        for i, type in enumerate(self.types):
            if type in ["space", "time"]:
                interleaved.append(physical[physical_ind])
                physical_ind += 1
            else:
                interleaved.append(non_physical)
        return interleaved

    @property
    def ome_scale(self) -> Sequence[int]:
        return [
            x
            for x in self.interleave_physical(self.voxel_size, 1)
            if isinstance(x, int)
        ]

    @property
    def ome_translate(self) -> Sequence[int]:
        assert self.offset % self.voxel_size == self.voxel_size * 0, (
            "funlib.persistence only supports ome-zarr with integer multiples of voxel_size as an offset."
            f"offset: {self.offset}, voxel_size:{self.voxel_size}, offset % voxel_size: {self.offset % self.voxel_size}"
        )
        return [
            x
            for x in self.interleave_physical(self.offset / self.voxel_size, 0)
            if isinstance(x, int)
        ]

    @property
    def ome_units(self) -> list[str | None]:
        return [
            str(x) if x is not None else None
            for x in self.interleave_physical(self.units, None)
        ]

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
        if self._axis_names is not None:
            return self._axis_names
        elif self._types is not None:
            indices = {
                "channel": iter(range(self.channel_dims)),
                "space": iter(range(self.physical_dims)),
            }
            return [
                f"d{next(indices[key])}"
                if key == "space"
                else f"c{next(indices['channel'])}^"
                for key in self._types
            ]
        else:
            return [f"c{dim}^" for dim in range(self.channel_dims)] + [
                f"d{dim}" for dim in range(self.physical_dims)
            ]

    @property
    def types(self) -> list[str]:
        if self._types is not None:
            return self._types
        elif self._axis_names is not None:
            warnings.warn(
                "using axis names to define which axes are channels using "
                "a '^' is deprecated and will be removed in a future version. "
                "Please use the 'types' attribute to define the type of each axis.",
                DeprecationWarning,
                stacklevel=2,
            )
            return [
                "channel" if name.endswith("^") else "space" for name in self.axis_names
            ]
        else:
            return ["channel"] * self.channel_dims + ["space"] * self.physical_dims

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
                len(
                    [
                        axis_type
                        for axis_type in self._types
                        if axis_type in ["space", "time"]
                    ]
                )
                if self._types is not None
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
                f"Axis names: {self._axis_names}\n"
                f"Types: {self._types}"
            )

    @property
    def channel_dims(self):
        return self.dims - self.physical_dims

    def validate(self, strict: bool):
        assert (
            all([isinstance(d, int) for d in self._offset])
            if self._offset is not None
            else True
        ), f"Offset must be a sequence of ints, got {self._offset}"
        assert (
            all([isinstance(d, int) for d in self._voxel_size])
            if self._voxel_size is not None
            else True
        ), f"Voxel size must be a sequence of ints, got {self._voxel_size}"
        assert (
            all([isinstance(d, str) for d in self._axis_names])
            if self._axis_names is not None
            else True
        ), f"Axis names must be a sequence of strings, got {self._axis_names}"
        assert (
            all([isinstance(d, str) for d in self._units])
            if self._units is not None
            else True
        ), f"Units must be a sequence of strings, got {self._units}"
        assert (
            all([isinstance(d, str) for d in self._types])
            if self._types is not None
            else True
        ), (
            f"Types must be a sequence of strings, got {self._types}\n"
            "If you see ints, you are likely using an old api and passing chunk_shape as the types argument."
        )
        if strict and any(
            [
                d is None
                for d in [
                    self.offset,
                    self.voxel_size,
                    self.axis_names,
                    self.units,
                    self.types,
                ]
            ]
        ):
            raise ValueError(
                "Strict metadata parsing requires all metadata attributes to be provided.\n"
                "Got:\n"
                f"Offset: {self.offset}\n"
                f"Voxel size: {self.voxel_size}\n"
                f"Axis names: {self.axis_names}\n"
                f"Units: {self.units}\n"
                f"Types: {self.types}"
            )
        assert self.dims == self.physical_dims + self.channel_dims


class MetaDataFormat(BaseModel):
    offset_attr: str = "offset"
    voxel_size_attr: str = "voxel_size"
    axis_names_attr: str = "axis_names"
    units_attr: str = "units"
    types_attr: str = "types"

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

    def strip_channels(self, types: list[str], to_strip: list[list]) -> None:
        to_delete = [i for i, t in enumerate(types) if t not in ["space", "time"]][::-1]
        for ll in to_strip:
            if ll is not None and len(ll) == len(types):
                for i in to_delete:
                    del ll[i]

    def parse(
        self,
        shape,
        data: dict[str | int, Any],
        offset=None,
        voxel_size=None,
        axis_names=None,
        units=None,
        types=None,
        strict=False,
    ) -> MetaData:
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
        types = (
            types if types is not None else self.fetch(data, self.types_attr.split("/"))
        )
        if types is None and axis_names is not None:
            types = [
                "channel" if name.endswith("^") else "space" for name in axis_names
            ]

        # we expect offset, voxel_size, and units to only apply to time and space dimensions
        # so here we strip off values that are not space or time
        if types is not None:
            self.strip_channels(types, [offset, voxel_size, units])

        offset = Coordinate(offset) if offset is not None else None
        voxel_size = Coordinate(voxel_size) if voxel_size is not None else None
        axis_names = list(axis_names) if axis_names is not None else None
        units = list(units) if units is not None else None
        types = list(types) if types is not None else None

        metadata = MetaData(
            shape=shape,
            offset=offset,
            voxel_size=voxel_size,
            axis_names=axis_names,
            units=units,
            types=types,
            strict=strict,
        )

        return metadata


class OME_MetaDataFormat(BaseModel):
    class Config:
        extra = "forbid"

    def strip_channels(self, types: list[str], to_strip: list[list]) -> None:
        to_delete = [i for i, t in enumerate(types) if t not in ["space", "time"]][::-1]
        for ll in to_strip:
            if ll is not None and len(ll) == len(types):
                for i in to_delete:
                    del ll[i]

    def parse(
        self,
        shape,
        offset=None,
        voxel_size=None,
        axis_names=None,
        units=None,
        types=None,
        strict=False,
    ) -> MetaData:
        if types is not None:
            self.strip_channels(types, [offset, voxel_size, units])

        offset = Coordinate(offset) if offset is not None else None
        voxel_size = Coordinate(voxel_size) if voxel_size is not None else None
        axis_names = list(axis_names) if axis_names is not None else None
        units = list(units) if units is not None else None
        types = list(types) if types is not None else None

        metadata = MetaData(
            shape=shape,
            offset=offset,
            voxel_size=voxel_size,
            axis_names=axis_names,
            units=units,
            types=types,
            strict=strict,
        )

        return metadata


DEFAULT_METADATA_FORMAT = MetaDataFormat()
LOCAL_PATHS = [Path("pyproject.toml"), Path("funlib_persistence.toml")]
USER_PATHS = [
    Path.home() / ".config" / "funlib_persistence" / "funlib_persistence.toml"
]
GLOBAL_PATHS = [Path("/etc/funlib_persistence/funlib_persistence.toml")]


def read_config() -> Optional[dict]:
    config: dict[str, str] = {}
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
