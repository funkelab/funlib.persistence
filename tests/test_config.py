from textwrap import dedent

import pytest

from funlib.persistence.arrays.metadata import (
    configure_library,
    get_default_metadata_format,
    read_config,
    set_default_metadata_format,
)


def _settings(offset_attr):
    return dedent(
        f"""
        offset_attr = "{offset_attr}"
        voxel_size_attr = "resolution"
        axis_names_attr = "axes"
        units_attr = "units"
        """
    )


@pytest.fixture
def write_config(tmp_path, monkeypatch):
    """A machine with no config files: an empty working directory, an empty home and
    a machine-wide path that does not exist yet. Returns a function that writes the
    config file of one location, with `offset_attr` set to the location's name."""
    cwd, home, etc = tmp_path / "cwd", tmp_path / "home", tmp_path / "etc"
    cwd.mkdir()
    (home / ".config" / "funlib_persistence").mkdir(parents=True)
    etc.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(
        "FUNLIB_PERSISTENCE_GLOBAL_CONFIG", str(etc / "funlib_persistence.toml")
    )
    files = {
        "pyproject": (cwd / "pyproject.toml", "[tool.funlib_persistence]"),
        "local": (cwd / "funlib_persistence.toml", ""),
        "user": (
            home / ".config" / "funlib_persistence" / "funlib_persistence.toml",
            "",
        ),
        "global": (etc / "funlib_persistence.toml", ""),
    }

    def write(location):
        path, header = files[location]
        path.write_text(header + _settings(location))

    return write


def test_read_config_without_files_is_empty(write_config):
    assert read_config() == {}


@pytest.mark.parametrize("location", ["pyproject", "local", "user", "global"])
def test_read_config(write_config, location):
    write_config(location)
    assert read_config() == {
        "offset_attr": location,
        "voxel_size_attr": "resolution",
        "axis_names_attr": "axes",
        "units_attr": "units",
    }


@pytest.mark.parametrize(
    "locations, winner",
    [
        (["global", "user"], "user"),
        (["user", "local"], "local"),
        (["local", "pyproject"], "pyproject"),
    ],
)
def test_read_config_precedence(write_config, locations, winner):
    for location in locations:
        write_config(location)
    assert read_config() == {
        "offset_attr": winner,
        "voxel_size_attr": "resolution",
        "axis_names_attr": "axes",
        "units_attr": "units",
    }


def test_configure_library_sets_the_default_metadata_format(write_config):
    write_config("local")
    before = get_default_metadata_format()
    try:
        configure_library()
        configured = get_default_metadata_format()
        assert (
            configured.offset_attr,
            configured.voxel_size_attr,
            configured.axis_names_attr,
            configured.units_attr,
        ) == ("local", "resolution", "axes", "units")
    finally:
        set_default_metadata_format(before)
