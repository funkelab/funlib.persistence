"""A graph database in memory-mapped CSR (compressed sparse row) arrays.

A directory holds one plain ``.npy`` file per array:

``ids``, one array per node attribute
    One entry per node. Nodes are sorted by the Morton code of the grid cell
    their position falls in, then by id, so the nodes of a region are a few
    runs of consecutive rows.
``cell_keys``, ``cell_start``
    The occupied cells in row order and the first row of each.
``indptr``, ``nbr``, one array per edge attribute
    Row ``i``'s neighbours are ``nbr[indptr[i]:indptr[i + 1]]`` (row
    numbers). Every edge is stored under both of its ends; the edge
    attribute arrays line up with ``nbr``.

The arrays are immutable. Writes append to *segment* files, one per open
handle, so any number of processes can write at once without coordinating.
:meth:`CSRGraphDataBase.finalize` merges the segments into a new set of
arrays; leaving :meth:`CSRGraphDataBase.bulk_write_mode` in the main
process does that. Reads see the arrays of the last finalize.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from funlib.geometry import Roi
from networkx import Graph

from ..types import Vec
from .graph_database import AttributeType, GraphDataBase

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
ID_DTYPE = np.dtype("<u8")
ROW_DTYPE = np.dtype("<i8")
_PYTHON_TYPES = {"float": "float64", "int": "int64", "bool": "bool"}


class UnfinalizedWrites(RuntimeError):
    """Segments have been written since the arrays were last built."""


def _spec(attr_type: AttributeType) -> dict[str, Any]:
    """An attribute type as JSON: a numpy dtype name and a vector size."""
    size = 0
    if isinstance(attr_type, Vec):
        attr_type, size = attr_type.dtype, attr_type.size
    name = attr_type.__name__ if isinstance(attr_type, type) else str(attr_type)
    dtype = np.dtype(_PYTHON_TYPES.get(name, name))
    if dtype.kind not in "fiub":
        raise NotImplementedError(
            f"attributes of type {attr_type} are not supported by the CSR backend"
        )
    return {"dtype": dtype.str, "size": size}


def _field(spec: dict[str, Any]) -> tuple:
    dtype = np.dtype(spec["dtype"])
    return (dtype, (spec["size"],)) if spec["size"] else (dtype,)


def _missing(dtype: np.dtype) -> Any:
    return np.nan if dtype.kind == "f" else 0


def _interleave(cells: np.ndarray) -> np.ndarray:
    """Morton code of each row of non-negative cell coordinates."""
    ndims = cells.shape[1]
    bits = 63 // ndims
    if cells.size and int(cells.max()) >= 1 << bits:
        raise ValueError(
            "the graph spans more grid cells than a Morton code can hold; "
            "use a larger cell_size"
        )
    key = np.zeros(len(cells), dtype=np.uint64)
    for bit in range(bits):
        for axis in range(ndims):
            shift = np.uint64(bit * ndims + (ndims - 1 - axis))
            part = (cells[:, axis].astype(np.uint64) >> np.uint64(bit)) & np.uint64(1)
            key |= part << shift
    return key


def _gather(indptr: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions in the half-edge arrays of every half-edge of ``rows``, and
    for each the index into ``rows`` of the row it belongs to."""
    start = np.asarray(indptr[rows])
    count = np.asarray(indptr[rows + 1]) - start
    total = int(count.sum())
    owner = np.repeat(np.arange(len(rows)), count)
    first = np.cumsum(count) - count
    at = np.arange(total) - np.repeat(first, count) + np.repeat(start, count)
    return at, owner


def _last_present(
    keys: list[np.ndarray], fields: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Collapse rows with equal ``keys``. Each field takes its last present
    value among the rows of a key, in the order given.

    ``fields`` maps a name to ``(values, present)``.
    """
    n = len(keys[0])
    order = np.lexsort((np.arange(n), *reversed(keys)))
    sorted_keys = [k[order] for k in keys]
    new = np.zeros(n, dtype=bool)
    if n:
        new[0] = True
        for k in sorted_keys:
            new[1:] |= k[1:] != k[:-1]
    starts = np.flatnonzero(new)
    out_keys = [k[starts] for k in sorted_keys]
    out_fields = {}
    position = np.arange(n)
    for name, (values, present) in fields.items():
        values = values[order]
        last = np.maximum.reduceat(np.where(present[order], position, -1), starts)
        result = np.empty((len(starts), *values.shape[1:]), dtype=values.dtype)
        result[...] = _missing(values.dtype)
        found = last >= 0
        result[found] = values[last[found]]
        out_fields[name] = result
    return out_keys, out_fields


class _Segment:
    """An append-only file of ``numpy.save`` records."""

    def __init__(self, path: Path):
        self.path = path
        self.file = None

    def append(self, records: np.ndarray) -> None:
        if len(records) == 0:
            return
        if self.file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(self.path, "ab")
        np.save(self.file, records, allow_pickle=False)
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    @staticmethod
    def read(path: Path) -> list[np.ndarray]:
        records = []
        size = path.stat().st_size
        with open(path, "rb") as f:
            while f.tell() < size:
                try:
                    records.append(np.load(f, allow_pickle=False))
                except (ValueError, EOFError, OSError):
                    # a writer that died mid-record leaves a short tail
                    logger.warning("ignoring a truncated record in %s", path)
                    break
        return records


class CSRGraphDataBase(GraphDataBase):
    """A graph in memory-mapped CSR arrays under a directory.

    Arguments:

        path:

            The directory of the database.

        position_attribute:

            The node attribute that holds a node's position.

        mode:

            ``"r"`` and ``"r+"`` open an existing database, ``"w"`` creates an
            empty one, removing what is there.

        node_attrs, edge_attrs:

            Attribute names and types. A type is ``float``, ``int``, ``bool``,
            a numpy dtype name such as ``"float32"``, or a
            :class:`~funlib.persistence.types.Vec` of one of those. A float
            that was never written reads as NaN; an integer as 0.

        cell_size:

            Edge length of a grid cell, in the units of the positions: one
            number or one per dimension. Pick it a few times smaller than
            the regions that will be read.
    """

    read_modes = ("r", "r+")
    create_modes = ("w",)

    def __init__(
        self,
        path: Path | str,
        position_attribute: str = "position",
        mode: str = "r",
        directed: bool | None = None,
        total_roi: Roi | None = None,
        node_attrs: dict[str, AttributeType] | None = None,
        edge_attrs: dict[str, AttributeType] | None = None,
        cell_size: float | Iterable[float] | None = None,
    ):
        if directed:
            raise NotImplementedError("the CSR backend stores undirected graphs")
        self.path = Path(path)
        self.mode = mode
        self.directed = False
        self._arrays: dict[str, np.ndarray] | None = None
        self._segments: dict[str, _Segment] = {}
        self._bulk_depth = 0
        self.meta: dict[str, Any] = {}
        if mode in self.create_modes:
            if node_attrs is None or position_attribute not in node_attrs:
                raise ValueError(
                    f"node_attrs must define the position attribute "
                    f"{position_attribute!r}"
                )
            position = _spec(node_attrs[position_attribute])
            ndims = position["size"] or 1
            if cell_size is None:
                raise ValueError("cell_size is required to create a CSR database")
            sizes = np.broadcast_to(np.asarray(cell_size, dtype=float), (ndims,))
            if self.path.exists():
                shutil.rmtree(self.path)
            self.path.mkdir(parents=True)
            self.meta = {
                "version": FORMAT_VERSION,
                "position_attribute": position_attribute,
                "ndims": ndims,
                "node_attrs": {k: _spec(v) for k, v in node_attrs.items()},
                "edge_attrs": {k: _spec(v) for k, v in (edge_attrs or {}).items()},
                "cell_size": sizes.tolist(),
                "total_roi": None
                if total_roi is None
                else [list(total_roi.begin), list(total_roi.shape)],
                "generation": None,
                "num_nodes": 0,
                "num_edges": 0,
            }
            self._write_meta()
        elif mode in self.read_modes:
            if not (self.path / "meta.json").exists():
                raise RuntimeError(
                    f"metadata does not exist, can't open in read mode ({self.path})"
                )
            self.meta = json.loads((self.path / "meta.json").read_text())
            if self.meta["version"] != FORMAT_VERSION:
                raise RuntimeError(
                    f"{self.path} is CSR format version {self.meta['version']}, "
                    f"this library reads version {FORMAT_VERSION}"
                )
            for given, stored, what in (
                (node_attrs, self.meta["node_attrs"], "node"),
                (edge_attrs, self.meta["edge_attrs"], "edge"),
            ):
                for name, attr_type in (given or {}).items():
                    if name not in stored:
                        raise ValueError(
                            f"{what} attribute {name!r} is not in {self.path}; "
                            f"it has {sorted(stored)}"
                        )
                    if _spec(attr_type)["size"] != stored[name]["size"]:
                        raise ValueError(
                            f"{what} attribute {name!r} has a different size in "
                            f"{self.path}"
                        )
        else:
            raise ValueError(f"mode {mode!r} is not one of 'r', 'r+', 'w'")
        self.position_attribute = self.meta["position_attribute"]
        self.ndims = self.meta["ndims"]
        self.cell_size = np.asarray(self.meta["cell_size"], dtype=float)
        stored_roi = self.meta.get("total_roi")
        self.total_roi = None if stored_roi is None else Roi(*stored_roi)

    # -- metadata -----------------------------------------------------------

    def _write_meta(self) -> None:
        tmp = self.path / f"meta.json.{uuid.uuid4().hex}"
        tmp.write_text(json.dumps(self.meta, indent=2) + "\n")
        os.replace(tmp, self.path / "meta.json")

    def _reload_meta(self) -> None:
        self.meta = json.loads((self.path / "meta.json").read_text())

    def _types(self, specs: dict[str, dict]) -> dict[str, AttributeType]:
        out: dict[str, AttributeType] = {}
        for name, spec in specs.items():
            dtype = np.dtype(spec["dtype"]).name
            out[name] = Vec(dtype, spec["size"]) if spec["size"] else dtype
        return out

    @property
    def node_attrs(self) -> dict[str, AttributeType]:
        return self._types(self.meta["node_attrs"])

    @property
    def edge_attrs(self) -> dict[str, AttributeType]:
        return self._types(self.meta["edge_attrs"])

    @property
    def num_nodes(self) -> int:
        return self.meta["num_nodes"]

    @property
    def num_edges(self) -> int:
        return self.meta["num_edges"]

    def close(self) -> None:
        for segment in self._segments.values():
            segment.close()
        self._arrays = None

    # -- arrays -------------------------------------------------------------

    def _array_names(self) -> list[str]:
        return (
            ["ids", "cell_keys", "cell_start", "indptr", "nbr"]
            + [f"node_{k}" for k in self.meta["node_attrs"]]
            + [f"edge_{k}" for k in self.meta["edge_attrs"]]
        )

    def _pending(self) -> list[Path]:
        folder = self.path / "segments"
        return sorted(folder.glob("*.npys")) if folder.exists() else []

    def _open_arrays(self) -> dict[str, np.ndarray] | None:
        """The arrays of the last finalize, memory-mapped; None if nothing
        was ever finalized."""
        if self._arrays is not None:
            return self._arrays
        self._reload_meta()
        if self.meta["generation"] is None:
            return None
        folder = self.path / self.meta["generation"]
        self._arrays = {
            name: np.load(folder / f"{name}.npy", mmap_mode="r")
            for name in self._array_names()
        }
        return self._arrays

    def _readable(self) -> dict[str, np.ndarray] | None:
        if self._pending():
            raise UnfinalizedWrites(
                f"{self.path} has writes that are not in its arrays yet; call "
                "finalize() (leaving bulk_write_mode in the main process does)"
            )
        return self._open_arrays()

    # -- writing ------------------------------------------------------------

    def _segment(self, kind: str) -> _Segment:
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")
        if kind not in self._segments:
            name = f"{kind}-{os.getpid()}-{uuid.uuid4().hex}.npys"
            self._segments[kind] = _Segment(self.path / "segments" / name)
        return self._segments[kind]

    def _record_dtype(self, kind: str, attrs: list[str]) -> np.dtype:
        specs = self.meta[f"{kind}_attrs"]
        unknown = [a for a in attrs if a not in specs]
        if unknown:
            raise ValueError(f"unknown {kind} attributes {unknown}")
        keys = (
            [("id", ID_DTYPE)] if kind == "node" else [("u", ID_DTYPE), ("v", ID_DTYPE)]
        )
        return np.dtype(keys + [(a, *_field(specs[a])) for a in attrs])

    def write_node_arrays(self, ids: np.ndarray, **attrs: np.ndarray) -> None:
        """Append nodes given as arrays: ``ids`` and one array per attribute
        (the position among them)."""
        if self.position_attribute not in attrs:
            raise ValueError(f"nodes need their {self.position_attribute!r}")
        records = np.empty(len(ids), dtype=self._record_dtype("node", list(attrs)))
        records["id"] = ids
        for name, values in attrs.items():
            records[name] = values
        self._segment("nodes").append(records)
        self._after_write()

    def write_edge_arrays(
        self, u: np.ndarray, v: np.ndarray, **attrs: np.ndarray
    ) -> None:
        """Append edges given as arrays: endpoint ids and one array per
        attribute. The order of an edge's endpoints does not matter."""
        records = np.empty(len(u), dtype=self._record_dtype("edge", list(attrs)))
        u, v = np.asarray(u, dtype=ID_DTYPE), np.asarray(v, dtype=ID_DTYPE)
        records["u"], records["v"] = np.minimum(u, v), np.maximum(u, v)
        for name, values in attrs.items():
            records[name] = values
        self._segment("edges").append(records)
        self._after_write()

    def _after_write(self) -> None:
        if self._bulk_depth == 0:
            self.finalize()

    def _position(self, data: dict[str, Any]) -> np.ndarray | None:
        position = data.get(self.position_attribute)
        if position is None:
            return None
        return np.atleast_1d(np.asarray(position, dtype=float))

    def _append_rows(
        self, kind: str, rows: list[tuple[tuple, dict[str, Any]]], attrs
    ) -> None:
        """Append ``(key, data)`` rows. Rows are grouped by which attributes
        they carry, so one that lacks an attribute leaves it untouched."""
        groups: dict[tuple[str, ...], list[tuple]] = {}
        for key, data in rows:
            present = tuple(a for a in attrs if data.get(a) is not None)
            values = [tuple(data[a]) if np.ndim(data[a]) else data[a] for a in present]
            groups.setdefault(present, []).append((*key, *values))
        segment = self._segment("nodes" if kind == "node" else "edges")
        for present, values in groups.items():
            segment.append(
                np.array(values, dtype=self._record_dtype(kind, list(present)))
            )

    def bulk_write_nodes(self, nodes, roi: Roi | None = None, attributes=None) -> None:
        attrs = (
            list(attributes)
            if attributes is not None
            else list(self.meta["node_attrs"])
        )
        if self.position_attribute not in attrs:
            attrs = [self.position_attribute, *attrs]
        rows = []
        for node_id, data in nodes.items():
            position = self._position(data)
            if position is None or (
                roi is not None and not roi.contains(tuple(position))
            ):
                continue
            rows.append(((node_id,), data))
        self._append_rows("node", rows, attrs)
        self._after_write()

    def bulk_write_edges(
        self, nodes, edges, roi: Roi | None = None, attributes=None
    ) -> None:
        attrs = (
            list(attributes)
            if attributes is not None
            else list(self.meta["edge_attrs"])
        )
        rows = []
        for (u, v), data in edges.items():
            u, v = min(u, v), max(u, v)
            if roi is not None:
                position = self._position(nodes[u])
                if position is None or not roi.contains(tuple(position)):
                    continue
            rows.append(((u, v), data))
        self._append_rows("edge", rows, attrs)
        self._after_write()

    def bulk_write_graph(
        self,
        graph: Graph,
        roi: Roi | None = None,
        write_nodes: bool = True,
        write_edges: bool = True,
        node_attrs: list[str] | None = None,
        edge_attrs: list[str] | None = None,
    ) -> None:
        with self.bulk_write_mode(worker=self._bulk_depth > 0):
            if write_nodes:
                self.bulk_write_nodes(graph.nodes, roi=roi, attributes=node_attrs)
            if write_edges:
                self.bulk_write_edges(
                    graph.nodes, graph.edges, roi=roi, attributes=edge_attrs
                )

    def write_graph(
        self,
        graph: Graph,
        roi: Roi | None = None,
        write_nodes: bool = True,
        write_edges: bool = True,
        node_attrs: list[str] | None = None,
        edge_attrs: list[str] | None = None,
        fail_if_exists: bool = False,
        delete: bool = False,
    ) -> None:
        """Append ``graph``. Outside :meth:`bulk_write_mode` the arrays are
        rebuilt before this returns, which costs a pass over the whole
        database: write many blocks inside ``bulk_write_mode``."""
        if fail_if_exists or delete:
            raise NotImplementedError(
                "fail_if_exists and delete are not supported by the CSR backend"
            )
        self.bulk_write_graph(
            graph, roi, write_nodes, write_edges, node_attrs, edge_attrs
        )

    def write_nodes(
        self, nodes, roi=None, attributes=None, fail_if_exists=False, delete=False
    ):
        if fail_if_exists or delete:
            raise NotImplementedError(
                "fail_if_exists and delete are not supported by the CSR backend"
            )
        self.bulk_write_nodes(nodes, roi=roi, attributes=attributes)

    def write_edges(
        self,
        nodes,
        edges,
        roi=None,
        attributes=None,
        fail_if_exists=False,
        delete=False,
    ):
        if fail_if_exists or delete:
            raise NotImplementedError(
                "fail_if_exists and delete are not supported by the CSR backend"
            )
        self.bulk_write_edges(nodes, edges, roi=roi, attributes=attributes)

    def write_attrs(
        self,
        graph: Graph,
        roi: Roi | None = None,
        node_attrs: list[str] | None = None,
        edge_attrs: list[str] | None = None,
    ) -> None:
        """Write attribute values of nodes and edges that exist. Values are
        merged at finalize: an attribute keeps its last written value."""
        with self.bulk_write_mode(worker=self._bulk_depth > 0):
            if node_attrs:
                self.bulk_write_nodes(graph.nodes, roi=roi, attributes=node_attrs)
            if edge_attrs:
                self.bulk_write_edges(
                    graph.nodes, graph.edges, roi=roi, attributes=edge_attrs
                )

    @contextmanager
    def bulk_write_mode(self, worker=False, node_writes=True, edge_writes=True):
        """Writes inside the block only append segments. Leaving it with
        ``worker=False`` rebuilds the arrays from every segment on disk,
        those of other processes included."""
        self._bulk_depth += 1
        try:
            yield
        finally:
            self._bulk_depth -= 1
            for segment in self._segments.values():
                segment.close()
            if not worker and self._bulk_depth == 0:
                self.finalize()

    def drop_edges(self) -> None:
        """Remove every edge, keeping the nodes."""
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")
        for path in self._pending():
            if path.name.startswith("edges-"):
                path.unlink()
        arrays = self._open_arrays()
        if arrays is None or self.meta["num_edges"] == 0:
            return
        nodes = {k: np.asarray(arrays[f"node_{k}"]) for k in self.meta["node_attrs"]}
        self._build(np.asarray(arrays["ids"]), nodes, None, sort_nodes=False)

    # -- finalize -----------------------------------------------------------

    def _batches(self, kind: str, paths: list[Path]) -> list[np.ndarray]:
        prefix = f"{kind}-"
        return [
            record
            for path in paths
            if path.name.startswith(prefix)
            for record in _Segment.read(path)
        ]

    def _merge(
        self,
        kind: str,
        keys: list[str],
        existing: tuple[list[np.ndarray], dict[str, np.ndarray]] | None,
        batches: list[np.ndarray],
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        specs = self.meta[f"{kind}_attrs"]
        sizes = ([len(existing[0][0])] if existing else []) + [len(b) for b in batches]
        total = sum(sizes)
        key_arrays = [np.empty(total, dtype=ID_DTYPE) for _ in keys]
        fields = {}
        for name, spec in specs.items():
            dtype, *shape = _field(spec)
            values = np.empty((total, *(shape[0] if shape else ())), dtype=dtype)
            fields[name] = (values, np.zeros(total, dtype=bool))
        at = 0
        if existing:
            n = sizes[0]
            for target, source in zip(key_arrays, existing[0], strict=True):
                target[:n] = source
            for name, (values, present) in fields.items():
                values[:n] = existing[1][name]
                present[:n] = self._present(values[:n])
            at = n
        for batch in batches:
            n = len(batch)
            for target, key in zip(key_arrays, keys, strict=True):
                target[at : at + n] = batch[key]
            for name, (values, present) in fields.items():
                if name in (batch.dtype.names or ()):
                    values[at : at + n] = batch[name]
                    present[at : at + n] = self._present(values[at : at + n])
                else:
                    values[at : at + n] = _missing(values.dtype)
            at += n
        return _last_present(key_arrays, fields)

    @staticmethod
    def _present(values: np.ndarray) -> np.ndarray:
        if values.dtype.kind != "f":
            return np.ones(len(values), dtype=bool)
        missing = np.isnan(values)
        return ~(missing.all(axis=1) if values.ndim > 1 else missing)

    def _edge_list(self, arrays: dict[str, np.ndarray]):
        """Every edge of the arrays once, as ids, from its smaller end."""
        ids = np.asarray(arrays["ids"])
        indptr = np.asarray(arrays["indptr"])
        nbr = np.asarray(arrays["nbr"])
        owner = np.repeat(np.arange(len(ids)), np.diff(indptr))
        once = np.flatnonzero(ids[owner] < ids[nbr])
        attrs = {
            k: np.asarray(arrays[f"edge_{k}"])[once] for k in self.meta["edge_attrs"]
        }
        return [ids[owner[once]], ids[nbr[once]]], attrs

    def finalize(self) -> None:
        """Merge every segment on disk into a new set of arrays.

        A node or edge written more than once keeps, per attribute, its last
        written value. An edge with an endpoint that is not a node is
        dropped. Holds the whole graph in memory."""
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")
        for segment in self._segments.values():
            segment.close()
        self._segments = {}
        paths = self._pending()
        if not paths:
            return
        self._arrays = None
        arrays = self._open_arrays()
        node_batches = self._batches("nodes", paths)
        edge_batches = self._batches("edges", paths)

        if arrays is None:
            (ids,), nodes = self._merge("node", ["id"], None, node_batches)
            sort_nodes = True
        elif node_batches:
            old = {k: np.asarray(arrays[f"node_{k}"]) for k in self.meta["node_attrs"]}
            (ids,), nodes = self._merge(
                "node", ["id"], ([np.asarray(arrays["ids"])], old), node_batches
            )
            sort_nodes = True
        else:
            ids = np.asarray(arrays["ids"])
            nodes = {
                k: np.asarray(arrays[f"node_{k}"]) for k in self.meta["node_attrs"]
            }
            sort_nodes = False

        existing = self._edge_list(arrays) if arrays is not None else None
        if existing is not None and len(existing[0][0]) == 0:
            existing = None
        edges = None
        if existing is not None or edge_batches:
            edges = self._merge("edge", ["u", "v"], existing, edge_batches)
        self._build(ids, nodes, edges, sort_nodes)
        for path in paths:
            path.unlink()

    def _build(self, ids, nodes, edges, sort_nodes: bool) -> None:
        position = nodes[self.position_attribute].reshape(len(ids), -1).astype(float)
        if sort_nodes or self.meta.get("origin") is None:
            lowest = position.min(axis=0) if len(ids) else np.zeros(self.ndims)
            origin = np.floor(lowest / self.cell_size) * self.cell_size
        else:
            origin = np.asarray(self.meta["origin"])
        cells = np.floor((position - origin) / self.cell_size).astype(np.int64)
        keys = _interleave(cells)
        if sort_nodes:
            order = np.lexsort((ids, keys))
            ids, keys = ids[order], keys[order]
            nodes = {k: v[order] for k, v in nodes.items()}
            position = position[order]
        n = len(ids)
        first = (
            np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
            if n
            else np.zeros(0, int)
        )
        out = {
            "ids": ids.astype(ID_DTYPE),
            "cell_keys": keys[first],
            "cell_start": np.r_[first, n].astype(ROW_DTYPE),
        }
        for name, values in nodes.items():
            out[f"node_{name}"] = values

        indptr = np.zeros(n + 1, dtype=ROW_DTYPE)
        dropped = 0
        num_edges = 0
        if edges is not None and n:
            (u, v), attrs = edges
            by_id = np.argsort(ids, kind="stable")
            sorted_ids = ids[by_id]

            def rows_of(values):
                at = np.searchsorted(sorted_ids, values)
                at[at == n] = 0
                return by_id[at], sorted_ids[at] == values

            ru, known_u = rows_of(u)
            rv, known_v = rows_of(v)
            keep = known_u & known_v & (u != v)
            dropped = int(len(u) - keep.sum())
            ru, rv = ru[keep], rv[keep]
            num_edges = len(ru)
            source = np.concatenate([ru, rv])
            target = np.concatenate([rv, ru])
            by_row = np.lexsort((target, source))
            np.cumsum(np.bincount(source, minlength=n), out=indptr[1:])
            out["nbr"] = target[by_row].astype(ROW_DTYPE)
            which = np.flatnonzero(keep)[by_row % max(num_edges, 1)]
            for name, values in attrs.items():
                out[f"edge_{name}"] = values[which]
        else:
            out["nbr"] = np.zeros(0, dtype=ROW_DTYPE)
            for name, spec in self.meta["edge_attrs"].items():
                out[f"edge_{name}"] = np.zeros(0, dtype=np.dtype(spec["dtype"]))
        out["indptr"] = indptr
        if dropped:
            logger.warning(
                "%d edges dropped: an endpoint is not a node of %s", dropped, self.path
            )

        generation = f"arrays-{uuid.uuid4().hex[:12]}"
        folder = self.path / generation
        folder.mkdir()
        for name, values in out.items():
            np.save(folder / f"{name}.npy", np.ascontiguousarray(values))
        previous = self.meta.get("generation")
        self.meta.update(
            generation=generation,
            origin=origin.tolist(),
            bounds=[position.min(axis=0).tolist(), position.max(axis=0).tolist()]
            if n
            else None,
            num_nodes=n,
            num_edges=num_edges,
            dropped_edges=dropped,
        )
        self._write_meta()
        self._arrays = None
        if previous is not None:
            shutil.rmtree(self.path / previous, ignore_errors=True)

    # -- reading ------------------------------------------------------------

    def _bounds(self, roi: Roi | None) -> tuple[np.ndarray, np.ndarray] | None:
        """The box of ``roi`` with open sides at infinity; None if no bound."""
        if roi is None:
            return None
        lo = np.array([-np.inf if b is None else b for b in roi.begin], dtype=float)
        hi = np.array([np.inf if e is None else e for e in roi.end], dtype=float)
        if np.isinf(lo).all() and np.isinf(hi).all():
            return None
        return lo, hi

    def _rows_in(self, arrays: dict[str, np.ndarray], roi: Roi | None) -> np.ndarray:
        """Rows of the nodes in ``roi``, ascending."""
        n = len(arrays["ids"])
        box = self._bounds(roi)
        if box is None or n == 0:
            return np.arange(n)
        lo, hi = box
        position = arrays[f"node_{self.position_attribute}"]
        origin = np.asarray(self.meta["origin"])
        data_lo, data_hi = (np.asarray(b) for b in self.meta["bounds"])
        lo_c, hi_c = np.maximum(lo, data_lo), np.minimum(hi, data_hi)
        if (lo_c > hi_c).any():
            return np.zeros(0, dtype=ROW_DTYPE)
        a = np.floor((lo_c - origin) / self.cell_size).astype(np.int64)
        b = np.floor((hi_c - origin) / self.cell_size).astype(np.int64)
        count = int(np.prod((b - a + 1).astype(float)))
        cell_keys = arrays["cell_keys"]
        if count > max(4096, 4 * len(cell_keys)):
            candidates = np.arange(n)
        else:
            grid = np.stack(
                np.meshgrid(
                    *(np.arange(a[d], b[d] + 1) for d in range(self.ndims)),
                    indexing="ij",
                ),
                axis=-1,
            ).reshape(-1, self.ndims)
            wanted = np.sort(_interleave(grid))
            keys = np.asarray(cell_keys)
            at = np.searchsorted(keys, wanted)
            valid = at < len(keys)
            at, wanted = at[valid], wanted[valid]
            at = at[keys[at] == wanted]
            start = np.asarray(arrays["cell_start"])
            begin, end = start[at], start[at + 1]
            size = end - begin
            first = np.cumsum(size) - size
            candidates = (
                np.arange(int(size.sum()))
                - np.repeat(first, size)
                + np.repeat(begin, size)
            )
        where = np.asarray(position[candidates]).reshape(len(candidates), -1)
        inside = ((where >= lo) & (where < hi)).all(axis=1)
        return candidates[inside]

    def _node_columns(self, arrays, rows, attrs: list[str]) -> dict[str, np.ndarray]:
        out = {"id": np.asarray(arrays["ids"][rows])}
        for name in attrs:
            if name not in self.meta["node_attrs"]:
                raise ValueError(f"unknown node attribute {name!r}")
            out[name] = np.asarray(arrays[f"node_{name}"][rows])
        return out

    def read_arrays(
        self,
        roi: Roi | None = None,
        num_steps: int = 0,
        node_attrs: list[str] | None = None,
        edge_attrs: list[str] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """The nodes in ``roi``, every node within ``num_steps`` edges of
        them, and the edges among those nodes, as arrays.

        Returns ``(nodes, edges)``. ``nodes`` has ``id``, ``step`` (0 for a
        node in the region, else the step that first reached it) and the
        requested attributes; ``edges`` has ``u``, ``v`` (ids, ``u < v``,
        each edge once) and the requested attributes. All attributes are
        returned when none are named."""
        arrays = self._readable()
        node_attrs = list(self.meta["node_attrs"]) if node_attrs is None else node_attrs
        edge_attrs = list(self.meta["edge_attrs"]) if edge_attrs is None else edge_attrs
        if arrays is None:
            arrays = self._empty()
        indptr, nbr = arrays["indptr"], arrays["nbr"]
        selected = self._rows_in(arrays, roi)
        step = np.zeros(len(selected), dtype=np.int16)
        frontier = selected
        for k in range(1, num_steps + 1):
            if len(frontier) == 0:
                break
            at, _ = _gather(indptr, frontier)
            reached = np.unique(np.asarray(nbr[at]))
            frontier = reached[~np.isin(reached, selected, assume_unique=True)]
            if len(frontier) == 0:
                break
            merged = np.concatenate([selected, frontier])
            order = np.argsort(merged, kind="stable")
            selected = merged[order]
            step = np.concatenate([step, np.full(len(frontier), k, dtype=np.int16)])[
                order
            ]
        nodes = self._node_columns(arrays, selected, node_attrs)
        nodes["step"] = step
        at, owner = _gather(indptr, selected)
        other = np.asarray(nbr[at])
        where = np.searchsorted(selected, other)
        where[where == len(selected)] = 0
        inside = selected[where] == other if len(selected) else np.zeros(0, dtype=bool)
        ids = nodes["id"]
        u, v = ids[owner], ids[where]
        once = inside & (u < v)
        edges = {"u": u[once], "v": v[once]}
        for name in edge_attrs:
            edges[name] = np.asarray(arrays[f"edge_{name}"][at[once]])
        return nodes, edges

    def _empty(self) -> dict[str, np.ndarray]:
        out = {
            "ids": np.zeros(0, dtype=ID_DTYPE),
            "cell_keys": np.zeros(0, dtype=np.uint64),
            "cell_start": np.zeros(1, dtype=ROW_DTYPE),
            "indptr": np.zeros(1, dtype=ROW_DTYPE),
            "nbr": np.zeros(0, dtype=ROW_DTYPE),
        }
        for kind in ("node", "edge"):
            for name, spec in self.meta[f"{kind}_attrs"].items():
                dtype, *shape = _field(spec)
                out[f"{kind}_{name}"] = np.zeros(
                    (0, *(shape[0] if shape else ())), dtype=dtype
                )
        return out

    @staticmethod
    def _filter(columns: dict[str, np.ndarray], attr_filter: dict[str, Any] | None):
        if not attr_filter:
            return None
        mask = np.ones(len(next(iter(columns.values()))), dtype=bool)
        for name, value in attr_filter.items():
            mask &= columns[name] == value
        return mask

    @staticmethod
    def _dicts(
        columns: dict[str, np.ndarray], names: list[str]
    ) -> list[dict[str, Any]]:
        lists = {}
        for name in names:
            values = columns[name]
            if values.ndim > 1:
                lists[name] = [tuple(row) for row in values.tolist()]
            elif values.dtype.kind == "f":
                lists[name] = [None if x != x else x for x in values.tolist()]
            else:
                lists[name] = values.tolist()
        return [
            dict(zip(names, row, strict=True))
            for row in zip(*lists.values(), strict=True)
        ]

    def read_nodes(
        self,
        roi: Roi | None = None,
        attr_filter: dict[str, Any] | None = None,
        read_attrs: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """The nodes in ``roi`` as dictionaries, as the SQL backends do."""
        arrays = self._readable() or self._empty()
        read_attrs = list(self.meta["node_attrs"]) if read_attrs is None else read_attrs
        needed = list(dict.fromkeys(read_attrs + list(attr_filter or {})))
        columns = self._node_columns(arrays, self._rows_in(arrays, roi), needed)
        mask = self._filter(columns, attr_filter)
        if mask is not None:
            columns = {k: v[mask] for k, v in columns.items()}
        return self._dicts(columns, ["id", *read_attrs])

    def _edge_columns(self, arrays, rows, fetch_on_v, attrs):
        ids = arrays["ids"]
        at, owner = _gather(arrays["indptr"], rows)
        other = np.asarray(arrays["nbr"][at])
        own_id, other_id = np.asarray(ids[rows])[owner], np.asarray(ids[other])
        if fetch_on_v:
            # an edge with both ends selected comes from each of them
            where = np.searchsorted(rows, other)
            where[where == len(rows)] = 0
            both = rows[where] == other if len(rows) else np.zeros(0, dtype=bool)
            keep = ~both | (own_id < other_id)
        else:
            keep = own_id < other_id
        columns = {
            "u": np.minimum(own_id, other_id)[keep],
            "v": np.maximum(own_id, other_id)[keep],
        }
        for name in attrs:
            if name not in self.meta["edge_attrs"]:
                raise ValueError(f"unknown edge attribute {name!r}")
            columns[name] = np.asarray(arrays[f"edge_{name}"][at[keep]])
        return columns

    def read_edges(
        self,
        roi: Roi | None = None,
        nodes: list[dict[str, Any]] | None = None,
        attr_filter: dict[str, Any] | None = None,
        read_attrs: list[str] | None = None,
        fetch_on_v: bool = False,
    ) -> list[dict[str, Any]]:
        """Edges whose smaller endpoint (``u``) is in ``roi`` or in
        ``nodes``; with ``fetch_on_v`` also those whose larger one is."""
        if roi is not None and nodes is not None:
            raise ValueError(
                "read_edges does not support both roi and nodes at the same time. "
                "Pass one or the other."
            )
        arrays = self._readable() or self._empty()
        read_attrs = list(self.meta["edge_attrs"]) if read_attrs is None else read_attrs
        if nodes is not None:
            wanted = np.array([n["id"] for n in nodes], dtype=ID_DTYPE)
            rows = self._rows_of_ids(arrays, wanted)
        else:
            rows = self._rows_in(arrays, roi)
        needed = list(dict.fromkeys(read_attrs + list(attr_filter or {})))
        columns = self._edge_columns(arrays, rows, fetch_on_v, needed)
        mask = self._filter(columns, attr_filter)
        if mask is not None:
            columns = {k: v[mask] for k, v in columns.items()}
        return self._dicts(columns, ["u", "v", *read_attrs])

    def _rows_of_ids(self, arrays, wanted: np.ndarray) -> np.ndarray:
        """Rows of the given ids, ascending; ids that are not nodes are
        left out. Sorts the id column: fine for a lookup now and then."""
        ids = np.asarray(arrays["ids"])
        by_id = np.argsort(ids, kind="stable")
        at = np.searchsorted(ids[by_id], wanted)
        at[at == len(ids)] = 0
        found = (
            ids[by_id[at]] == wanted if len(ids) else np.zeros(len(wanted), dtype=bool)
        )
        return np.unique(by_id[at[found]])

    def read_graph(
        self,
        roi: Roi | None = None,
        read_edges: bool = True,
        node_attrs: list[str] | None = None,
        edge_attrs: list[str] | None = None,
        nodes_filter: dict[str, Any] | None = None,
        edges_filter: dict[str, Any] | None = None,
        fetch_on_v: bool = False,
    ) -> Graph:
        """The nodes in ``roi`` and the edges whose smaller endpoint is among
        them, as a networkx graph; with ``fetch_on_v`` also the edges whose
        larger endpoint is."""
        graph = Graph()
        nodes = self.read_nodes(roi, attr_filter=nodes_filter, read_attrs=node_attrs)
        graph.add_nodes_from((n.pop("id"), n) for n in nodes)
        if read_edges:
            if nodes_filter:
                listed = [{"id": i} for i in graph.nodes]
                edges = self.read_edges(
                    nodes=listed,
                    attr_filter=edges_filter,
                    read_attrs=edge_attrs,
                    fetch_on_v=fetch_on_v,
                )
            else:
                edges = self.read_edges(
                    roi=roi if self._bounds(roi) is not None else None,
                    attr_filter=edges_filter,
                    read_attrs=edge_attrs,
                    fetch_on_v=fetch_on_v,
                )
            graph.add_edges_from((e.pop("u"), e.pop("v"), e) for e in edges)
        return graph

    def has_edges(self, roi: Roi) -> bool:
        arrays = self._readable() or self._empty()
        rows = self._rows_in(arrays, roi)
        return bool(len(_gather(arrays["indptr"], rows)[0]))
