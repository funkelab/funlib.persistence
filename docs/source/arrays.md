# Arrays and metadata

`funlib.persistence` opens zarr, N5 and OME-Zarr datasets as an `Array`: a
dask array that knows where it sits in the world. Every spatial axis
carries an offset and a voxel size in world units, so the array can be
indexed with a `funlib.geometry.Roi` in those units rather than with voxel
indices, and two datasets at different resolutions can be read over the
same region.

```python
from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, prepare_ds

raw = open_ds("sample.zarr/raw")            # metadata comes from the attributes
raw.roi, raw.voxel_size                      # Roi in world units, Coordinate

labels = prepare_ds(
    "sample.zarr/labels",
    shape=raw.shape,
    offset=raw.offset,
    voxel_size=raw.voxel_size,
    axis_names=raw.axis_names,
    units=raw.units,
    dtype="uint64",
    mode="w",
)
region = Roi((0, 0, 0), (400, 400, 400)) + raw.offset   # world units
block = raw[region]                   # numpy, shape in voxels
labels[region] = block > 0.5
```

`Array.to_ndarray(roi, fill_value)` reads a ROI that may extend beyond the
array and pads the outside with `fill_value`. `__getitem__` and
`__setitem__` also accept ordinary numpy keys (slices, integers), which index
in voxels.

Related: [daisy](https://funkelab.github.io/daisy/) processes an `Array`
block by block, and its tutorial opens and prepares datasets with the
functions above.

## The five metadata fields

| Field | Type | Meaning |
|-------|------|---------|
| `offset` | ints, one per spatial axis | world coordinate of the first voxel |
| `voxel_size` | ints, one per spatial axis | world size of one voxel |
| `axis_names` | strings, one per axis | names such as `z`, `y`, `x`, `c^` |
| `units` | strings, one per spatial axis | world unit per axis, such as `nm` |
| `types` | strings, one per axis | `space`, `time` or `channel` |

`space` and `time` axes are spatial: they take part in ROI indexing and
have an offset, a voxel size and a unit. `channel` axes do not. When `types`
is missing, an axis whose name ends in `^` is taken as a channel and every
other axis as space; this fallback is deprecated, so store `types`.

Each field can be given to `open_ds` as an override, for datasets whose
attributes are missing or wrong. With `strict_metadata=True`, `open_ds`
raises unless every field is known, from the attributes or the arguments.
Fields that are neither stored nor given default to offset zero, voxel size
one, axis names `c0, c1, ...`, and no units.

`prepare_ds` writes the fields it is given as attributes next to the data,
in the layout described below, so the next `open_ds` reads them back.

## Where the fields are stored

A `MetaDataFormat` names the attribute each field is read from. The
defaults are the field names themselves:

```python
MetaDataFormat(
    offset_attr="offset",
    voxel_size_attr="voxel_size",
    axis_names_attr="axis_names",
    units_attr="units",
    types_attr="types",
)
```

An attribute name is a path into the dataset's attributes: `/` descends
into nested dictionaries, an integer segment indexes a list, and the
segment `{dim}` collects one value per axis from a list of per-axis
dictionaries. Three formats that the same `MetaDataFormat` can describe:

| Attributes | Attribute name | Value |
|------------|----------------|-------|
| `{"offset": [100, 200, 400]}` | `offset` | `[100, 200, 400]` |
| `{"metadata": {"offset": [100, 200, 400]}}` | `metadata/offset` | `[100, 200, 400]` |
| `{"axes": [{"offset": 0}, {"offset": 1}]}` | `axes/{dim}/offset` | `[0, 1]` |

Pass a `MetaDataFormat` to `open_ds(metadata_format=...)` for one dataset,
or change the default for the whole process with
`set_default_metadata_format`.

## Configuring the default format

The default format can also come from a TOML file, so a project whose
datasets store `resolution` instead of `voxel_size` does not have to pass a
`MetaDataFormat` on every call. At import, the library reads these files,
later entries overriding earlier ones:

1. `/etc/funlib_persistence/funlib_persistence.toml`
2. `~/.config/funlib_persistence/funlib_persistence.toml`
3. `funlib_persistence.toml` in the working directory
4. `pyproject.toml` in the working directory, table `[tool.funlib_persistence]`

So a setting in the project's `pyproject.toml` wins over the user's config,
which wins over the machine's. The keys are the `MetaDataFormat` field
names:

```toml
[tool.funlib_persistence]
offset_attr = "offset"
voxel_size_attr = "resolution"
axis_names_attr = "axes"
units_attr = "units"
```

A standalone `funlib_persistence.toml` holds the same keys at the top
level. `configure_library()` re-reads the files, for a process that changes
directory after import.
