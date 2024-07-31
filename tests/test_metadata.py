from funlib.persistence.arrays.metadata import MetaDataFormat
from funlib.geometry import Coordinate

import pytest
import json
from pathlib import Path


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
    assert metadata.offset == Coordinate(100, 200, 400)
    assert metadata.voxel_size == Coordinate(1, 2, 3)
    assert metadata.axis_names == ["sample^", "channel^", "z", "y", "x"]
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


def test_empty_metadata():
    metadata = MetaDataFormat().parse((10, 2, 100, 100, 100), {})
    assert metadata.offset == Coordinate(0, 0, 0, 0, 0)
    assert metadata.voxel_size == Coordinate(1, 1, 1, 1, 1)
    assert metadata.axis_names == ["d0", "d1", "d2", "d3", "d4"]
    assert metadata.units == ["", "", "", "", ""]
