from funlib.persistence.arrays.metadata import MetaDataFormat
from funlib.geometry import Coordinate

import pytest
import json
from pathlib import Path


fixtures_dir = Path(__file__).parent / "fixtures"
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
        json.loads(open(fixtures_dir / metadata_jsons[request.param]).read())
    )


def test_parse_metadata(metadata):
    assert metadata.offset == Coordinate(100, 200, 400)
    assert metadata.voxel_size == Coordinate(1, 2, 3)
    assert metadata.axis_names == ["sample^", "channel^", "z", "y", "x"]
    assert metadata.units == ["nm", "nm", "nm"]
