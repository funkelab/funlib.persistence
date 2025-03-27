import json
from pathlib import Path

import pytest
import zarr

from funlib.geometry import Coordinate
from funlib.persistence.arrays.datasets import prepare_ds
from funlib.persistence.arrays.metadata import (
    MetaDataFormat,
    set_default_metadata_format,
)

fixtures_dir = Path(__file__).parent / "fixtures"
incomplete_metadatas = {
    "incomplete1": "metadatas/incomplete1.json",
    "incomplete2": "metadatas/incomplete2.json",
    "incomplete3": "metadatas/incomplete3.json",
}
incomplete_metadata_formats = {
    "incomplete1": MetaDataFormat(),
    "incomplete2": MetaDataFormat(),
    "incomplete3": MetaDataFormat(),
}
metadata_jsons = {
    "default": "metadatas/default.json",
    "simple": "metadatas/simple.json",
    "ome-ngff-multiscale": "metadatas/ome-ngff-multiscale.json",
}
metadata_formats = {
    "default": MetaDataFormat(),
    "simple": MetaDataFormat(
        voxel_size_attr="resolution",
        axis_names_attr="extras/axes",
        units_attr="extras/units",
    ),
    "ome-ngff-multiscale": MetaDataFormat(
        types_attr="multiscales/0/axes/{dim}/type",
        axis_names_attr="multiscales/0/axes/{dim}/name",
        units_attr="multiscales/0/axes/{dim}/unit",
        voxel_size_attr="multiscales/0/coordinateTransformations/0/scale",
        offset_attr="multiscales/0/coordinateTransformations/1/translation",
    ),
}


@pytest.fixture(params=metadata_jsons.keys())
def metadata(request):
    return metadata_formats[request.param].parse(
        (10, 2, 100, 100, 100),
        json.loads(open(fixtures_dir / metadata_jsons[request.param]).read()),
    )


def test_parse_metadata(metadata):
    assert metadata.types == ["sample", "channel", "time", "space", "space"]
    assert metadata.offset == Coordinate(100, 200, 400)
    assert metadata.voxel_size == Coordinate(1, 2, 3)
    assert metadata.axis_names == ["sample^", "channel^", "t", "y", "x"]
    assert metadata.units == ["nm", "nm", "nm"]


@pytest.fixture(params=incomplete_metadatas.keys())
def incomplete_metadata(request):
    return incomplete_metadata_formats[request.param].parse(
        (10, 2, 100, 100, 100),
        json.loads(open(fixtures_dir / incomplete_metadatas[request.param]).read()),
    )


def test_parse_incomplete_metadata(incomplete_metadata):
    assert incomplete_metadata.offset == Coordinate(0, 0, 0)
    assert incomplete_metadata.voxel_size == Coordinate(1, 1, 1)
    assert incomplete_metadata.axis_names == ["c0^", "c1^", "d0", "d1", "d2"]
    assert incomplete_metadata.units == ["", "", ""]
    assert incomplete_metadata.types == [
        "channel",
        "channel",
        "space",
        "space",
        "space",
    ]


def test_empty_metadata():
    metadata = MetaDataFormat().parse((10, 2, 100, 100, 100), {})
    assert metadata.offset == Coordinate(0, 0, 0, 0, 0)
    assert metadata.voxel_size == Coordinate(1, 1, 1, 1, 1)
    assert metadata.axis_names == ["d0", "d1", "d2", "d3", "d4"]
    assert metadata.units == ["", "", "", "", ""]
    assert metadata.types == ["space", "space", "space", "space", "space"]


def test_default_metadata_format(tmpdir):
    set_default_metadata_format(metadata_formats["simple"])
    metadata = metadata_formats["simple"].parse(
        (10, 2, 100, 100, 100),
        json.loads(open(fixtures_dir / metadata_jsons["simple"]).read()),
    )

    prepare_ds(
        tmpdir / "test.zarr/test",
        (10, 2, 100, 100, 100),
        offset=metadata.offset,
        voxel_size=metadata.voxel_size,
        axis_names=metadata.axis_names,
        units=metadata.units,
        types=metadata.types,
        dtype="float32",
        mode="w",
    )

    zarr_attrs = dict(**zarr.open(str(tmpdir / "test.zarr/test")).attrs)
    assert zarr_attrs["offset"] == [100, 200, 400]
    assert zarr_attrs["resolution"] == [1, 2, 3]
    assert zarr_attrs["extras/axes"] == ["sample^", "channel^", "t", "y", "x"]
    assert zarr_attrs["extras/units"] == ["nm", "nm", "nm"]
    assert zarr_attrs["types"] == ["sample", "channel", "time", "space", "space"]
