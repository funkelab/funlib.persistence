"""Node attributes in plain arrays beside the graph they belong to.

Values computed on a graph (an embedding, a score, a label per node) do not
have to go into the graph database: the graph stays as it is, and a
64-vector per node would outweigh the graph it describes. They are stored
on their own, as memory-mapped arrays keyed by node id::

    <path>/meta.json        version, num_nodes, attrs, source
    <path>/ids.npy          uint64 (N,)   node ids, in stored order
    <path>/order.npy        int64  (N,)   argsort of ids
    <path>/node_<attr>.npy  (N,) or (N, k)

``attrs`` maps each attribute to ``{"dtype": "<f4", "size": k}``, ``size``
0 for a scalar. Rows are in the order they were written: a writer that
goes block by block over a spatially sorted graph leaves neighbours near
each other on disk, so the read of a block touches few pages. A read is by
node id, through ``order``. ``source`` is free text that says where the
values come from; the store does not interpret it.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["FORMAT_VERSION", "NodeArrays", "NodeArraysWriter", "write_node_arrays"]

FORMAT_VERSION = 1
META = "meta.json"
_COPY = 1 << 26


def _file(name: str) -> str:
    return f"node_{name}.npy"


class NodeArrays:
    """Stored node attributes, memory-mapped and read by node id."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        meta = self.path / META
        if not meta.is_file():
            raise FileNotFoundError(
                f"{self.path} holds no {META}: not a node attribute store"
            )
        self.meta: dict[str, Any] = json.loads(meta.read_text())
        if self.meta.get("version") != FORMAT_VERSION:
            raise ValueError(
                f"{self.path}: node attribute store version "
                f"{self.meta.get('version')!r}, this reader knows {FORMAT_VERSION}"
            )
        self._arrays: dict[str, np.ndarray] = {}

    @property
    def num_nodes(self) -> int:
        return int(self.meta["num_nodes"])

    @property
    def attrs(self) -> dict[str, int | None]:
        """Attribute name to vector length (None for a scalar)."""
        return {
            name: (spec["size"] or None) for name, spec in self.meta["attrs"].items()
        }

    @property
    def source(self) -> str | None:
        """What the writer recorded about where the values come from."""
        return self.meta.get("source")

    def _load(self, file: str) -> np.ndarray:
        if file not in self._arrays:
            self._arrays[file] = np.load(self.path / file, mmap_mode="r")
        return self._arrays[file]

    @property
    def ids(self) -> np.ndarray:
        """Every stored node id, in stored order."""
        return self._load("ids.npy")

    def array(self, name: str) -> np.ndarray:
        """One attribute of every node, memory-mapped, aligned with
        :attr:`ids`."""
        if name not in self.meta["attrs"]:
            raise KeyError(
                f"{self.path} holds no attribute {name!r}; it holds "
                f"{sorted(self.meta['attrs'])}"
            )
        return self._load(_file(name))

    def rows(self, ids: Any) -> tuple[np.ndarray, np.ndarray]:
        """Where each of ``ids`` is stored: ``(found, rows)``; a row is
        only meaningful where ``found``."""
        ids = np.asarray(ids, dtype=np.uint64)
        order = self._load("order.npy")
        stored = self.ids
        if len(order) == 0:
            return np.zeros(len(ids), dtype=bool), np.zeros(len(ids), dtype=np.int64)
        # ``order`` sorts the stored ids, so the search probes log(N) of
        # them per id and never reads the whole array
        lo = np.searchsorted(stored, ids, sorter=order)
        at = np.minimum(lo, len(order) - 1)
        rows = np.asarray(order[at], dtype=np.int64)
        found = (lo < len(order)) & (np.asarray(stored[rows]) == ids)
        return found, rows

    def read(
        self, ids: Any, attrs: Sequence[str] | None = None
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """The attributes of the nodes ``ids``, in their order.

        Returns ``(found, values)``. A node that was never written is not
        ``found`` and reads as NaN (0 for an integer attribute).
        """
        names = list(self.meta["attrs"]) if attrs is None else list(attrs)
        found, rows = self.rows(ids)
        # gather in row order, so a block's read walks the file forwards
        where = np.flatnonzero(found)
        walk = where[np.argsort(rows[where], kind="stable")]
        values: dict[str, np.ndarray] = {}
        for name in names:
            stored = self.array(name)
            fill = np.nan if stored.dtype.kind == "f" else 0
            out = np.full((len(found), *stored.shape[1:]), fill, dtype=stored.dtype)
            out[walk] = stored[rows[walk]]
            values[name] = out
        return found, values


class NodeArraysWriter:
    """Writes a node attribute store a block at a time.

    ::

        with NodeArraysWriter(path, {"embedding": 64}) as out:
            for ids, embedding in blocks:
                out.append(ids, embedding=embedding)

    The store appears at ``path`` only when the block is left without an
    error; until then it is a ``.partial`` directory beside it, which an
    error leaves behind for inspection and the next writer removes.
    """

    def __init__(
        self,
        path: Path | str,
        attrs: Mapping[str, int | None],
        *,
        dtypes: Mapping[str, Any] | None = None,
        source: str | None = None,
    ):
        self.path = Path(path)
        self.attrs = dict(attrs)
        if not self.attrs:
            raise ValueError("a node attribute store holds at least one attribute")
        self.dtypes = {
            name: np.dtype((dtypes or {}).get(name, "float32")).newbyteorder("<")
            for name in self.attrs
        }
        for name, dtype in self.dtypes.items():
            if dtype.kind not in "fiub":
                raise ValueError(f"attribute {name!r}: {dtype} is not a number type")
        self.source = source
        self.num_nodes = 0
        self._partial = self.path.with_name(self.path.name + ".partial")
        self._files: dict[str, Any] = {}

    def __enter__(self) -> NodeArraysWriter:
        if self._partial.exists():
            shutil.rmtree(self._partial)
        self._partial.mkdir(parents=True)
        for name in ("ids", *self.attrs):
            self._files[name] = open(self._partial / f"{name}.raw", "wb")
        return self

    def append(self, ids: Any, **values: Any) -> None:
        """Add nodes: their ids and one array per attribute, aligned."""
        ids = np.ascontiguousarray(ids, dtype="<u8")
        if set(values) != set(self.attrs):
            raise ValueError(
                f"expected the attributes {sorted(self.attrs)}, got {sorted(values)}"
            )
        blocks = {}
        for name, length in self.attrs.items():
            block = np.ascontiguousarray(values[name], dtype=self.dtypes[name])
            shape = (len(ids),) if length is None else (len(ids), length)
            if block.shape != shape:
                raise ValueError(
                    f"attribute {name!r} has shape {block.shape}, expected {shape}"
                )
            blocks[name] = block
        self._files["ids"].write(ids.tobytes())
        for name, block in blocks.items():
            self._files[name].write(block.tobytes())
        self.num_nodes += len(ids)

    def _finish(self, name: str, file: str, dtype: np.dtype, length: int | None):
        """Turn the raw bytes of one array into a ``.npy`` file."""
        shape = (self.num_nodes,) if length is None else (self.num_nodes, length)
        raw = self._partial / f"{name}.raw"
        with open(self._partial / file, "wb") as out:
            np.lib.format.write_array_header_1_0(
                out,
                {
                    "descr": np.lib.format.dtype_to_descr(dtype),
                    "fortran_order": False,
                    "shape": shape,
                },
            )
            with open(raw, "rb") as source:
                shutil.copyfileobj(source, out, _COPY)
        raw.unlink()

    def __exit__(self, kind: Any, *_: Any) -> None:
        for file in self._files.values():
            file.close()
        if kind is not None:
            return
        self._finish("ids", "ids.npy", np.dtype("<u8"), None)
        for name, length in self.attrs.items():
            self._finish(name, _file(name), self.dtypes[name], length)
        ids = np.load(self._partial / "ids.npy", mmap_mode="r")
        order = np.argsort(ids, kind="stable").astype("<i8")
        if len(ids) > 1 and (np.diff(np.asarray(ids)[order]) == 0).any():
            raise ValueError(
                f"{self.path}: a node id was written twice; a node belongs to one block"
            )
        np.save(self._partial / "order.npy", order)
        meta = {
            "version": FORMAT_VERSION,
            "num_nodes": self.num_nodes,
            "attrs": {
                name: {"dtype": self.dtypes[name].str, "size": length or 0}
                for name, length in self.attrs.items()
            },
            "source": self.source,
        }
        (self._partial / META).write_text(json.dumps(meta, indent=2) + "\n")
        if self.path.exists():
            shutil.rmtree(self.path)
        os.replace(self._partial, self.path)


def write_node_arrays(
    path: Path | str,
    ids: Any,
    values: Mapping[str, Any],
    *,
    source: str | None = None,
) -> Path:
    """Write node attributes that are in memory as one store."""
    arrays = {name: np.asarray(value) for name, value in values.items()}
    attrs = {
        name: (None if value.ndim == 1 else int(value.shape[1]))
        for name, value in arrays.items()
    }
    dtypes = {name: value.dtype for name, value in arrays.items()}
    with NodeArraysWriter(path, attrs, dtypes=dtypes, source=source) as out:
        out.append(ids, **arrays)
    return Path(path)
