from pathlib import Path
from textwrap import dedent
from unittest.mock import mock_open

import pytest

from funlib.persistence.arrays.metadata import (
    configure_library,
    get_default_metadata_format,
    read_config,
    set_default_metadata_format,
)

SETTINGS = """
    offset_attr = "test_offset"
    voxel_size_attr = "resolution"
    axis_names_attr = "axes"
    units_attr = "units"
    """


@pytest.fixture(
    params=[
        (Path("pyproject.toml"), "[tool.funlib_persistence]" + SETTINGS),
        (
            Path.home() / ".config" / "funlib_persistence" / "funlib_persistence.toml",
            SETTINGS,
        ),
        (Path("/etc/funlib_persistence/funlib_persistence.toml"), SETTINGS),
    ],
    ids=["local", "user", "global"],
)
def config_file(request, mocker):
    """The filesystem holds exactly one config file, at one of the three locations."""
    path, text = request.param
    mocker.patch("pathlib.Path.exists", autospec=True).side_effect = lambda p: p == path
    mocker.patch("builtins.open", new_callable=mock_open, read_data=dedent(text))
    return path


def test_read_config(config_file):
    assert read_config() == {
        "offset_attr": "test_offset",
        "voxel_size_attr": "resolution",
        "axis_names_attr": "axes",
        "units_attr": "units",
    }


def test_configure_library_sets_the_default_metadata_format(config_file):
    before = get_default_metadata_format()
    try:
        configure_library()
        configured = get_default_metadata_format()
        assert (
            configured.offset_attr,
            configured.voxel_size_attr,
            configured.axis_names_attr,
            configured.units_attr,
        ) == ("test_offset", "resolution", "axes", "units")
    finally:
        set_default_metadata_format(before)
