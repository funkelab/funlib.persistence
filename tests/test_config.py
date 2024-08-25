from pathlib import Path
from textwrap import dedent
from unittest.mock import mock_open

import pytest

from funlib.persistence.arrays.metadata import configure_library, read_config


@pytest.fixture
def mock_path_exists(mocker):
    return mocker.patch("pathlib.Path.exists", autospec=True)


@pytest.fixture
def mock_file_open(mocker):
    return mocker.patch("builtins.open", new_callable=mock_open)


def test_local_config(mock_path_exists, mock_file_open):
    mock_path_exists.side_effect = lambda p: p == Path("pyproject.toml")
    mock_file_open.return_value.read.return_value = dedent(
        """
        [tool.funlib_persistence]
        offset_attr = "test_offset"
        voxel_size_attr = "resolution"
        axis_names_attr = "axes"
        units_attr = "units"
        """
    )

    config = read_config()
    assert config == {
        "offset_attr": "test_offset",
        "voxel_size_attr": "resolution",
        "axis_names_attr": "axes",
        "units_attr": "units",
    }
    configure_library()


def test_user_config(mock_path_exists, mock_file_open):
    mock_path_exists.side_effect = (
        lambda p: p
        == Path.home() / ".config" / "funlib_persistence" / "funlib_persistence.toml"
    )
    mock_file_open.return_value.read.return_value = dedent(
        """
        offset_attr = "test_offset"
        voxel_size_attr = "resolution"
        axis_names_attr = "axes"
        units_attr = "units"
        """
    )

    config = read_config()
    assert config == {
        "offset_attr": "test_offset",
        "voxel_size_attr": "resolution",
        "axis_names_attr": "axes",
        "units_attr": "units",
    }


def test_global_config(mock_path_exists, mock_file_open):
    mock_path_exists.side_effect = lambda p: p == Path(
        "/etc/funlib_persistence/funlib_persistence.toml"
    )
    mock_file_open.return_value.read.return_value = dedent(
        """
        offset_attr = "test_offset"
        voxel_size_attr = "resolution"
        axis_names_attr = "axes"
        units_attr = "units"
        """
    )

    config = read_config()
    assert config == {
        "offset_attr": "test_offset",
        "voxel_size_attr": "resolution",
        "axis_names_attr": "axes",
        "units_attr": "units",
    }
