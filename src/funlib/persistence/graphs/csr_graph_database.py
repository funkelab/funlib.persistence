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

Node attributes that are computed on a finished graph (an embedding, a
label per node) can be kept outside the graph's directory, as *extra
fields*: see ``extra_fields`` of :class:`CSRGraphDataBase`. They are
written and finalized the same way, into a directory of their own, and the
graph's directory is not touched.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import shutil
import uuid
import zlib
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from funlib.geometry import Coordinate, Roi
from networkx import Graph

from ..types import Vec
from ._files import replace_directory, scratch_beside
from .graph_database import AttributeType, GraphDataBase

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
EXTRA_FORMAT_VERSION = 1
RESERVED_NODE_NAMES = ("id", "step", "own")
"""Names the reads give to arrays of their own (the node id, the step that
reached a node, the block's own nodes): not names of extra attributes."""
_EXTRA_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*")
"""The name of an extra attribute: letters, digits, ``_``, ``.`` and ``-``,
not starting with ``.`` or ``-``. It is part of the name of a file."""
ID_DTYPE = np.dtype("<u8")
ROW_DTYPE = np.dtype("<i8")
_PYTHON_TYPES = {"float": "float64", "int": "int64", "bool": "bool"}


class UnfinalizedWrites(RuntimeError):
    """Segments have been written since the arrays were last built."""


class GraphIsReadOnly(RuntimeError):
    """A write that would change the graph was made through a handle opened
    with ``extra_fields``, which writes extra node attributes only."""


@contextmanager
def _locked(folder: Path) -> Generator[None]:
    """Hold the lock of a directory of arrays: one finalize, or one change
    of its ``meta.json``, at a time. A second process waits."""
    with open(folder / "finalize.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


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


def _ids_crc32(ids: np.ndarray) -> int:
    """The crc32 of node ids in row order, a few million at a time."""
    checksum = 0
    for start in range(0, len(ids), 1 << 22):
        chunk = np.asarray(ids[start : start + (1 << 22)], dtype=ID_DTYPE)
        checksum = zlib.crc32(chunk.tobytes(), checksum)
    return checksum


class _ExtraFields:
    """A directory of extra node attributes of one graph::

        meta.json            version, extends, node_attrs, arrays
        node_<attr>-<x>.npy  one array per attribute, in the graph's row order
        segments/            writes that are not in the arrays yet
        finalize.lock

    ``extends`` says which graph the rows are those of: its node count,
    the name of the set of arrays it was last built into, and a checksum
    of its node ids in row order. ``arrays`` names the file of each
    attribute, or nothing for one that was declared and never finalized.
    """

    def __init__(self, path: Path):
        self.path = path
        self.meta: dict[str, Any] | None = None
        self.reload()

    def reload(self) -> None:
        file = self.path / "meta.json"
        if not file.exists():
            self.meta = None
            return
        meta = json.loads(file.read_text())
        if not isinstance(meta, dict) or "extends" not in meta:
            raise ValueError(f"{self.path} is not a directory of extra node attributes")
        if meta["version"] != EXTRA_FORMAT_VERSION:
            raise RuntimeError(
                f"{self.path} is extra-fields format version {meta['version']}, "
                f"this library reads version {EXTRA_FORMAT_VERSION}"
            )
        self.meta = meta

    @property
    def stored(self) -> dict[str, Any]:
        if self.meta is None:
            raise RuntimeError(f"{self.path} holds no extra node attributes")
        return self.meta

    def write(self) -> None:
        tmp = self.path / f"meta.json.{uuid.uuid4().hex}"
        tmp.write_text(json.dumps(self.meta, indent=2) + "\n")
        os.replace(tmp, self.path / "meta.json")

    def pending(self) -> list[Path]:
        folder = self.path / "segments"
        return sorted(folder.glob("*.npys")) if folder.exists() else []

    def record_dtype(self, attrs: Iterable[str]) -> np.dtype:
        specs = self.stored["node_attrs"]
        return np.dtype([("id", ID_DTYPE)] + [(a, *_field(specs[a])) for a in attrs])


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

        extra_fields:

            One directory, or several, that hold node attributes of this
            graph outside the graph's own directory. The graph is opened as
            it is (mode ``"r"`` or ``"r+"``); every attribute the
            directories hold is read with it, with the type its directory
            records, and written through it::

                g = CSRGraphDataBase(path, mode="r+", extra_fields="pred/modelA")
                g.write_node_arrays(ids, embedding=e, label=l)   # in bulk_write_mode
                g.read_arrays(roi, node_attrs=["position", "embedding"])

            The path is enough: no attribute is named and no type is given.
            ``meta.json`` in each directory says what it holds. Extra
            attributes are read like stored ones (same regions, steps,
            filters) and listed by :attr:`node_attrs`. An attribute that
            two of the directories hold, or that the graph itself stores,
            is refused when the graph is opened. A directory that is empty
            or does not exist is created by the first write in mode
            ``"r+"`` and is an error in mode ``"r"``. A directory inside
            the graph's directory, or one that holds it, is refused.

            **A new attribute is created by the first write that names
            it**, with the type of the values written:
            ``write_node_arrays(ids, embedding=e)`` with ``e`` a float32
            array of shape ``(N, 64)`` declares a float32 vector of 64. The
            declaration is recorded in the directory's ``meta.json`` under
            its lock, so processes that create one attribute at once agree
            on it. From then on an array written to it has that dtype and
            vector size, or the write raises; nothing is cast. The writes
            that take nodes as dictionaries (``write_nodes``,
            ``bulk_write_nodes``, ``write_attrs``) create an attribute that
            ``attributes`` names from the values the nodes carry: numbers
            of one kind (bool, int or float, stored as bool, int64 or
            float64) and one length; anything else raises, and when no node
            carries a value nothing is written and nothing is created. A
            value for an attribute that exists is stored in the recorded
            type or refused, never changed: a float for an integer or a
            bool, an integer for a bool, and a sequence of another length
            than the vector raise; a Python float takes the recorded float
            width, an integer the recorded integer width (or becomes a
            float). A call that is refused records nothing: no attribute
            of it is created. The name of an extra attribute is made of
            letters, digits, ``_``, ``.`` and ``-`` and does not start with
            ``.`` or ``-``: it is part of the name of a file. ``id``,
            ``step`` and ``own`` are not names of attributes; the reads use
            them. Node ids are integers. A value has to be in the range of
            the recorded type. ``node_attrs`` is
            never needed for an extra attribute. It may name one with a
            type, to create it up front in mode ``"r+"`` (float32 rather
            than the float64 Python numbers give), or to have the type that
            is recorded checked.

            **A new attribute goes into the one directory that is open.**
            With several directories open, a write that names an attribute
            none of them holds raises: open the graph with the directory it
            is to go into.

            **The graph is read-only through such a handle.** Writes of
            edges, of nodes with stored attributes, and ``drop_edges``
            raise :class:`GraphIsReadOnly`; nothing in the graph's
            directory is created, changed or locked. The node writes
            (``write_node_arrays``, ``write_attrs``, ``write_nodes``,
            ``bulk_write_nodes``) take extra attributes only and need no
            position, unless a ``roi`` is given: then a node with a value
            to write carries its position, and one that does not raises. A
            node that carries neither is left out: ``read_graph(roi)`` adds
            such a node for an edge that leaves the region, so a block that
            is read with its region can be written back with it.

            Writing works as for the graph: each handle appends to a
            segment file of its own in the extra directory, from any
            number of processes, and :meth:`finalize` (or leaving
            :meth:`bulk_write_mode` in the main process) merges the
            segments into the arrays of the attributes that were written.
            It reads the graph's node ids and nothing else of the graph. A
            later finalize adds to the values that are there; a node
            written twice keeps its last value. With several directories
            open every one is finalized; a directory that fails does not
            keep the others from being merged, and the errors of several
            are raised together as an ``ExceptionGroup``. A directory with
            nothing to merge, or that does not exist yet, is left alone. A
            finalize also removes arrays in the directory that its
            ``meta.json`` does not name, which a killed finalize can leave.

            Values are stored in the graph's row order, so they belong to
            one graph. The directory records that graph's node count, the
            name of its arrays and a checksum of its node ids in row order;
            opening it with a graph of another node count, or of other ids
            or another row order, raises ``ValueError``. Adding edges to
            the graph, or rebuilding it from the same nodes, keeps its
            extra fields valid.

            Limits. A node that was never written reads as NaN (0 for an
            integer or a bool, which cannot be told from a written 0). A
            written NaN counts as not written. Writing an id that is not a
            node of the graph is found at finalize, which then raises
            ``ValueError`` and changes nothing; the segments stay until
            they are removed. Finalize holds the node ids and every
            attribute it rebuilds in memory. As for the graph: finalize
            when the writers are done (a record appended to a segment while
            it is merged is lost), a record cut short by a killed writer is
            ignored with a warning, and which of two processes wrote "last"
            is decided by the names of their segment files, not by time. A
            read that returns an extra attribute raises
            :class:`UnfinalizedWrites` while its directory has segments;
            reads of other attributes do not. A handle sees the arrays of
            the finalize it last loaded: open the graph again to see what
            another process finalized since.
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
        extra_fields: Path | str | Sequence[Path | str] | None = None,
    ):
        if directed:
            raise NotImplementedError("the CSR backend stores undirected graphs")
        self.path = Path(path)
        self.mode = mode
        self.directed = False
        if isinstance(extra_fields, Mapping):
            raise TypeError(
                "extra_fields is a directory or a sequence of directories; the "
                "attributes and their types are read from each directory"
            )
        if isinstance(extra_fields, str | Path):
            extra_fields = [extra_fields]
        self.extra_fields = [Path(folder) for folder in extra_fields or ()]
        self._extra_dirs: list[_ExtraFields] = []
        self._extra: dict[str, _ExtraFields] = {}
        if self.extra_fields and mode not in self.read_modes:
            raise ValueError(
                f"mode {mode!r} creates the graph; extra_fields belong to a "
                "graph that exists and are opened with mode 'r' or 'r+'"
            )
        self._arrays: dict[str, np.ndarray] | None = None
        self._segments: dict[Any, _Segment] = {}
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
            # every argument is checked before what is at the path is removed
            node_specs = {k: _spec(v) for k, v in node_attrs.items()}
            edge_specs = {k: _spec(v) for k, v in (edge_attrs or {}).items()}
            if self.path.exists():
                shutil.rmtree(self.path)
            self.path.mkdir(parents=True)
            self.meta = {
                "version": FORMAT_VERSION,
                "position_attribute": position_attribute,
                "ndims": ndims,
                "node_attrs": node_specs,
                "edge_attrs": edge_specs,
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
                    if what == "node" and self.extra_fields and name not in stored:
                        continue  # an extra attribute: see _open_extra_fields
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
        if self.extra_fields:
            self._open_extra_fields(node_attrs or {})

    # -- extra fields -------------------------------------------------------

    @staticmethod
    def stored_extra_fields(path: Path | str) -> dict[str, AttributeType]:
        """The node attributes a directory of extra fields holds, with their
        types; empty when ``path`` is not such a directory."""
        try:
            extra = _ExtraFields(Path(path))
        except (ValueError, NotADirectoryError):
            return {}
        if extra.meta is None:
            return {}
        return CSRGraphDataBase._types(extra.meta["node_attrs"])

    def _extends(self) -> dict[str, Any]:
        """What an extra directory records of this graph."""
        ids = np.load(self.path / self.meta["generation"] / "ids.npy", mmap_mode="r")
        return {
            "num_nodes": self.meta["num_nodes"],
            "generation": self.meta["generation"],
            "ids_crc32": _ids_crc32(ids),
        }

    def _check_extends(self, extra: _ExtraFields) -> None:
        of = extra.stored["extends"]
        if of["num_nodes"] != self.meta["num_nodes"]:
            raise ValueError(
                f"{extra.path} holds attributes of a graph of {of['num_nodes']} "
                f"nodes; {self.path} has {self.meta['num_nodes']}"
            )
        # the same set of arrays is the same rows; another set (edges were
        # added, the graph was built again) is compared by its node ids
        if of["generation"] != self.meta["generation"]:
            if of["ids_crc32"] != self._extends()["ids_crc32"]:
                raise ValueError(
                    f"{extra.path} holds attributes of a graph with other nodes, "
                    f"or the same nodes in another order, than {self.path}"
                )

    def _open_extra_fields(self, given: Mapping[str, AttributeType]) -> None:
        if self.meta["generation"] is None:
            raise ValueError(
                f"{self.path} was never finalized; extra fields belong to a "
                "graph whose nodes are written"
            )
        seen: set[Path] = set()
        for folder in self.extra_fields:
            key, graph = folder.resolve(), self.path.resolve()
            if key == graph or key.is_relative_to(graph):
                raise ValueError(
                    f"{folder}: extra fields are kept outside the graph's "
                    f"directory {self.path}, which they never change"
                )
            if graph.is_relative_to(key):
                raise ValueError(
                    f"{folder} holds the graph's directory {self.path}: extra "
                    "fields are kept in a directory of their own"
                )
            if key in seen:
                raise ValueError(f"{folder} is given twice in extra_fields")
            seen.add(key)
            extra = _ExtraFields(folder)
            if extra.meta is None and self.mode == "r":
                raise ValueError(
                    f"{folder} holds no extra node attributes. Open the graph "
                    "with mode 'r+' to write the first"
                )
            self._extra_dirs.append(extra)
        self._map_extra_fields()
        # node_attrs may name extra attributes: to create them up front, or
        # to have the recorded type checked
        named = {
            name: _spec(attr_type)
            for name, attr_type in given.items()
            if name not in self.meta["node_attrs"]
        }
        new = {name: spec for name, spec in named.items() if name not in self._extra}
        if new and self.mode == "r":
            raise ValueError(
                f"node attributes {sorted(new)} are neither in {self.path} nor "
                f"in its extra fields, which hold {sorted(self._extra)}"
            )
        if new:
            self._declare_extra(new)
        for name, spec in named.items():
            self._same_spec(name, spec)

    def _map_extra_fields(self) -> None:
        """Which directory holds each extra attribute, from what the
        directories say now. A name held twice is refused."""
        found: dict[str, _ExtraFields] = {}
        for extra in self._extra_dirs:
            if extra.meta is None:
                continue
            self._check_extends(extra)
            for name in extra.meta["node_attrs"]:
                if name in RESERVED_NODE_NAMES:
                    raise ValueError(
                        f"{extra.path} holds a node attribute named {name!r}, "
                        "which the reads use for an array of their own"
                    )
                if name in self.meta["node_attrs"]:
                    raise ValueError(
                        f"node attribute {name!r} of {extra.path} is also stored "
                        f"in {self.path} itself"
                    )
                if name in found:
                    raise ValueError(
                        f"node attribute {name!r} is held by {found[name].path} "
                        f"and by {extra.path}"
                    )
                found[name] = extra
        self._extra = found

    def _same_spec(self, name: str, spec: dict[str, Any]) -> None:
        extra = self._extra[name]
        recorded = extra.stored["node_attrs"][name]
        if spec != recorded:
            raise ValueError(
                f"node attribute {name!r} is {self._types({name: recorded})[name]} "
                f"in {extra.path}, not {self._types({name: spec})[name]}"
            )

    def _check_new_extra(self, names: Iterable[str]) -> None:
        """Whether attributes of these names may be created by this handle."""
        names = sorted(names, key=str)
        if not names:
            return
        if any(not isinstance(n, str) for n in names):
            raise ValueError(f"{names} are not all names: an attribute name is a str")
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")
        reserved = [n for n in names if n in RESERVED_NODE_NAMES]
        if reserved:
            raise ValueError(
                f"{reserved} are not names of node attributes: the reads use "
                f"{list(RESERVED_NODE_NAMES)} for arrays of their own"
            )
        unfit = [n for n in names if not _EXTRA_NAME.fullmatch(n)]
        if unfit:
            raise ValueError(
                f"{unfit} are not names of extra node attributes: a name is "
                "made of letters, digits, '_', '.' and '-', and does not start "
                "with '.' or '-' (it is part of the name of a file)"
            )
        if len(self._extra_dirs) != 1:
            raise ValueError(
                f"{names} are new node attributes and "
                f"{len(self._extra_dirs)} extra directories are open: open the "
                "graph with the one directory they are to go into"
            )

    def _declare_extra(self, specs: Mapping[str, dict[str, Any]]) -> None:
        """Record new extra attributes in the one directory that is open.
        Another process may declare the same ones at the same time: the
        first to hold the lock writes them, the others find them. Either
        all of ``specs`` are recorded or, when one disagrees with what is
        recorded, none."""
        if not specs:
            return
        self._check_new_extra(specs)
        (extra,) = self._extra_dirs
        extra.path.mkdir(parents=True, exist_ok=True)
        with _locked(extra.path):
            extra.reload()
            if extra.meta is None:
                extra.meta = {
                    "version": EXTRA_FORMAT_VERSION,
                    "extends": self._extends(),
                    "node_attrs": {},
                    "arrays": {},
                }
            held = extra.meta["node_attrs"]
            for name, spec in specs.items():
                if name in held and held[name] != spec:
                    extra.reload()
                    self._map_extra_fields()
                    self._same_spec(name, spec)
            new = {name: spec for name, spec in specs.items() if name not in held}
            if new:
                held.update(new)
                extra.write()
        self._map_extra_fields()
        self._arrays = None

    def _extra_folders(self) -> list[_ExtraFields]:
        return self._extra_dirs

    def _node_specs(self) -> dict[str, dict[str, Any]]:
        """Every node attribute this handle reads: stored, then extra."""
        specs = dict(self.meta["node_attrs"])
        for name, extra in self._extra.items():
            specs[name] = extra.stored["node_attrs"][name]
        return specs

    def _refuse(self, what: str) -> None:
        if self._extra_dirs:
            raise GraphIsReadOnly(
                f"{what}: {self.path} is read-only through a handle opened with "
                "extra_fields; only extra node attributes are written, into "
                f"{[str(e.path) for e in self._extra_dirs]}. Open the graph "
                "without extra_fields to change it"
            )

    def _split_extra(
        self, attrs: Iterable[str]
    ) -> list[tuple[_ExtraFields, list[str]]]:
        """Extra attribute names by the directory that holds them."""
        attrs = list(attrs)
        return [
            (extra, [a for a in attrs if self._extra[a] is extra])
            for extra in self._extra_dirs
            if any(self._extra[a] is extra for a in attrs)
        ]

    def _new_extra_names(self, attrs: Iterable[str]) -> list[str]:
        """Refuse stored attributes and give the names that no directory
        holds yet. Nothing is recorded."""
        attrs = list(attrs)
        stored = [a for a in attrs if a in self.meta["node_attrs"]]
        if stored:
            self._refuse(f"cannot write the node attributes {stored}")
        if any(a not in self._extra for a in attrs):
            for extra in self._extra_dirs:
                extra.reload()  # another process may have created them
            self._map_extra_fields()
            # the arrays this handle loaded do not hold what was found
            self._arrays = None
        new = [a for a in attrs if a not in self._extra]
        self._check_new_extra(new)
        return new

    @staticmethod
    def _spec_of_values(name: str, values: np.ndarray) -> dict[str, Any]:
        """The type an array of values, one entry per node, declares."""
        if values.dtype.kind not in "fiub":
            raise NotImplementedError(
                f"node attribute {name!r}: attributes of type {values.dtype} are "
                "not supported by the CSR backend"
            )
        if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] < 1):
            raise ValueError(
                f"node attribute {name!r} has shape {values.shape}; expected "
                "(N,) for a scalar or (N, k) for a vector"
            )
        size = values.shape[1] if values.ndim == 2 else 0
        return {"dtype": values.dtype.newbyteorder("<").str, "size": size}

    def _write_extra_arrays(self, ids: np.ndarray, attrs: Mapping[str, Any]) -> None:
        ids = np.asarray(ids)
        if ids.ndim != 1 or (len(ids) and ids.dtype.kind not in "iu"):
            raise ValueError(
                f"ids are {ids.dtype} of shape {ids.shape}; node ids are "
                "integers, one per node"
            )
        arrays = {name: np.asarray(values) for name, values in attrs.items()}
        # every check of the call comes before anything is recorded: a write
        # that is refused declares no attribute
        new = self._new_extra_names(arrays)
        for name, values in arrays.items():
            if values.ndim == 0 or len(values) != len(ids):
                raise ValueError(
                    f"node attribute {name!r} has shape {values.shape}; "
                    f"expected one entry for each of the {len(ids)} ids"
                )
        given = {
            name: self._spec_of_values(name, values) for name, values in arrays.items()
        }
        for name, spec in given.items():
            if name not in new:
                self._same_spec(name, spec)
        self._declare_extra({name: given[name] for name in new})
        for extra, names in self._split_extra(arrays):
            records = np.empty(len(ids), dtype=extra.record_dtype(names))
            records["id"] = ids
            for name in names:
                records[name] = arrays[name]
            self._segment("nodes", extra.path).append(records)

    @staticmethod
    def _fits(name: str, value: Any, spec: dict[str, Any]) -> None:
        """Refuse a value of a node given as a dictionary that the recorded
        type would have to change: a float for an integer or a bool, an
        integer for a bool, anything that is not a number, and a sequence
        of another length than the vector. A Python number has no width, so
        a float is stored in the recorded float type and an integer in the
        recorded integer type, when it is in the range of that type."""
        given = np.asarray(value)
        dtype = np.dtype(spec["dtype"])
        allowed = {"f": "fiub", "i": "iub", "u": "iub", "b": "b"}[dtype.kind]
        shape = (spec["size"],) if spec["size"] else ()
        fits = given.dtype.kind in allowed and given.shape == shape
        if fits and dtype.kind in "iu":
            # exact Python integers: a value past int64 has dtype object
            # and was refused above
            bounds = np.iinfo(dtype.name)
            fits = all(bounds.min <= int(x) <= bounds.max for x in given.flat)
        elif fits and dtype.kind == "f":
            with np.errstate(over="ignore"):
                stored = given.astype(dtype)
            fits = bool((np.isfinite(stored) | ~np.isfinite(given)).all())
        if not fits:
            wanted = f"a vector of {spec['size']}" if spec["size"] else "one value"
            raise ValueError(
                f"node attribute {name!r} is {dtype.name}, {wanted}; "
                f"{value!r} is not stored as that without changing it"
            )

    def _write_extra_rows(self, nodes, roi: Roi | None, attributes) -> None:
        # left out, the extra attributes that exist are written; a new one
        # is named
        names = list(self._extra) if attributes is None else list(attributes)
        kept = []
        for node_id, data in nodes.items():
            if isinstance(node_id, bool) or not isinstance(node_id, int | np.integer):
                raise ValueError(f"{node_id!r} is not a node id: an id is an integer")
            if roi is not None:
                position = self._position(data)
                if position is None:
                    # a node that read_graph(roi) added for an edge leaving
                    # the roi carries nothing; one with a value to write and
                    # no position cannot be placed and is not left out
                    if all(data.get(a) is None for a in names):
                        continue
                    raise ValueError(
                        f"node {node_id} carries values and no "
                        f"{self.position_attribute!r}: with a roi, the nodes "
                        "carry their position (read them with it, or write "
                        "without a roi)"
                    )
                if not roi.contains(tuple(position)):
                    continue
            kept.append((node_id, data))

        # every check of the call comes before anything is recorded
        new = self._new_extra_names(names)
        specs: dict[str, dict[str, Any]] = {}
        for name in new:
            values = [np.asarray(d[name]) for _, d in kept if d.get(name) is not None]
            if not values:
                # no node carries it (an empty block): nothing to write, and
                # nothing to tell a type from
                continue
            kinds = {v.dtype.kind for v in values}
            shapes = {v.shape for v in values}
            if (
                len(kinds) != 1
                or len(shapes) != 1
                or not kinds <= set("fiub")
                or len(next(iter(shapes))) > 1
            ):
                raise ValueError(
                    f"node attribute {name!r} is new and its type cannot be "
                    f"told from the values of the nodes ({len(values)} "
                    f"values, kinds {sorted(kinds)}, shapes {sorted(shapes)}): "
                    "give its type in node_attrs when the graph is opened"
                )
            specs[name] = self._spec_of_values(name, np.stack(values))
        names = [a for a in names if a in self._extra or a in specs]
        for name in names:
            spec = specs.get(name) or self._extra[name].stored["node_attrs"][name]
            for _, data in kept:
                if data.get(name) is not None:
                    self._fits(name, data[name], spec)
        self._declare_extra(specs)
        for extra, own in self._split_extra(names):
            groups: dict[tuple[str, ...], list[tuple]] = {}
            for node_id, data in kept:
                present = tuple(a for a in own if data.get(a) is not None)
                if not present:
                    continue
                values = [
                    tuple(data[a]) if np.ndim(data[a]) else data[a] for a in present
                ]
                groups.setdefault(present, []).append((node_id, *values))
            for present, rows in groups.items():
                self._segment("nodes", extra.path).append(
                    np.array(rows, dtype=extra.record_dtype(present))
                )

    def _finalize_extra(self, extra: _ExtraFields) -> None:
        paths = extra.pending()
        if not paths:
            return
        extra.reload()
        self._reload_meta()
        self._check_extends(extra)
        # the node ids are all that is read of the graph
        ids = np.load(self.path / self.meta["generation"] / "ids.npy")
        n = len(ids)
        by_id = np.argsort(ids, kind="stable")
        sorted_ids = ids[by_id]
        specs = extra.stored["node_attrs"]
        files = extra.stored["arrays"]
        values: dict[str, np.ndarray] = {}
        for path in paths:
            for batch in _Segment.read(path):
                at = np.searchsorted(sorted_ids, batch["id"])
                at[at == n] = 0
                known = (
                    sorted_ids[at] == batch["id"]
                    if n
                    else np.zeros(len(batch), dtype=bool)
                )
                if not known.all():
                    some = np.unique(batch["id"][~known])[:5].tolist()
                    raise ValueError(
                        f"{path} holds values for {int((~known).sum())} ids that "
                        f"are not nodes of {self.path} (for example {some}); "
                        f"nothing was finalized in {extra.path}"
                    )
                rows = by_id[at]
                for name in (batch.dtype.names or ())[1:]:
                    if name not in specs:
                        raise ValueError(
                            f"{path} holds {name!r}, unknown to {extra.path}"
                        )
                    if name not in values:
                        values[name] = self._extra_array(extra, name, n, copy=True)
                    written = self._present(batch[name])
                    values[name][rows[written]] = batch[name][written]
        for name, array in values.items():
            file = f"node_{name}-{uuid.uuid4().hex[:12]}.npy"
            np.save(extra.path / file, array)
            files[name] = file
        extra.stored["extends"] = {
            "num_nodes": n,
            "generation": self.meta["generation"],
            "ids_crc32": _ids_crc32(ids),
        }
        extra.write()
        # the arrays that were replaced, and any that a finalize which was
        # killed before it recorded them left behind
        named = set(files.values())
        for file in extra.path.glob("node_*-*.npy"):
            if file.name not in named:
                file.unlink(missing_ok=True)
        for path in paths:
            path.unlink(missing_ok=True)

    @staticmethod
    def _extra_array(extra: _ExtraFields, name: str, n: int, copy: bool = False):
        """One extra attribute of every node, in row order. One that was
        never finalized is all missing values."""
        file = extra.stored["arrays"].get(name)
        if file is not None:
            return np.load(extra.path / file, mmap_mode=None if copy else "r")
        dtype, *shape = _field(extra.stored["node_attrs"][name])
        full = (n, *(shape[0] if shape else ()))
        if copy:
            return np.full(full, _missing(dtype), dtype=dtype)
        return np.broadcast_to(np.asarray(_missing(dtype), dtype=dtype), full)

    @staticmethod
    def is_database(path: Path | str) -> bool:
        """Whether ``path`` is the directory of a CSR graph database."""
        meta = Path(path) / "meta.json"
        if not meta.is_file():
            return False
        try:
            stored = json.loads(meta.read_text())
        except ValueError:
            return False
        return isinstance(stored, dict) and "generation" in stored

    @classmethod
    def from_arrays(
        cls,
        path: Path | str,
        ids: np.ndarray,
        node_attrs: Mapping[str, np.ndarray],
        u: np.ndarray | None = None,
        v: np.ndarray | None = None,
        edge_attrs: Mapping[str, np.ndarray] | None = None,
        *,
        cell_size: float | Iterable[float],
        position_attribute: str = "position",
        node_types: Mapping[str, AttributeType] | None = None,
        edge_types: Mapping[str, AttributeType] | None = None,
        total_roi: Roi | None = None,
    ) -> CSRGraphDataBase:
        """Create a database at ``path`` from a whole graph held in arrays,
        replacing what is there, and return it open for reading and writing.

        ``ids`` are the node ids, ``node_attrs`` one array per node attribute
        (the position among them), ``u`` and ``v`` the endpoint ids of the
        edges and ``edge_attrs`` one array per edge attribute. Every array
        has one entry per node, or per edge: an array of another length is
        refused, a single value is not spread over the graph.

        An attribute is stored in the dtype of its array, a 2-D array as a
        vector, unless ``node_types`` or ``edge_types`` names another type
        for it; the array is cast to that type.

        The database is built in a directory beside ``path`` and moved to
        ``path`` when it is complete. A call that fails, for whatever
        reason, leaves what was at ``path`` as it was.
        """
        path = Path(path)
        ids = np.asarray(ids)
        if ids.ndim != 1:
            raise ValueError(f"ids has shape {ids.shape}; expected one id per node")
        if (u is None) != (v is None):
            raise ValueError("edges need both u and v")
        if u is None and edge_attrs:
            raise ValueError("edge attributes were given without edges")
        num_edges = 0
        if u is not None and v is not None:
            u, v = np.asarray(u), np.asarray(v)
            if u.ndim != 1 or u.shape != v.shape:
                raise ValueError(
                    f"u and v have shapes {u.shape} and {v.shape}; expected one "
                    "entry per edge in each"
                )
            num_edges = len(u)

        def columns(what, given, count, named) -> tuple[dict, dict]:
            arrays = {name: np.asarray(a) for name, a in (given or {}).items()}
            unknown = sorted(set(named or {}) - set(arrays))
            if unknown:
                raise ValueError(f"types were given for {unknown}, which have no array")
            types: dict[str, AttributeType] = {}
            for name, array in arrays.items():
                if array.ndim not in (1, 2) or len(array) != count:
                    raise ValueError(
                        f"{what} attribute {name!r} has shape {array.shape}; "
                        f"expected one entry for each of the {count} {what}s"
                    )
                if named and name in named:
                    types[name] = named[name]
                elif array.dtype.kind not in "fiub":
                    raise NotImplementedError(
                        f"{what} attribute {name!r}: attributes of type "
                        f"{array.dtype} are not supported by the CSR backend"
                    )
                elif array.ndim == 1:
                    types[name] = array.dtype.name
                else:
                    types[name] = Vec(array.dtype.name, array.shape[1])
                size = _spec(types[name])["size"]
                if (array.shape[1] if array.ndim == 2 else 0) != size:
                    raise ValueError(
                        f"{what} attribute {name!r} has shape {array.shape}, which "
                        f"does not fit its type {types[name]}"
                    )
            return arrays, types

        node_arrays, node_kinds = columns("node", node_attrs, len(ids), node_types)
        edge_arrays, edge_kinds = columns("edge", edge_attrs, num_edges, edge_types)

        scratch = scratch_beside(path, "writing")
        try:
            db = cls(
                scratch,
                position_attribute=position_attribute,
                mode="w",
                total_roi=total_roi,
                node_attrs=node_kinds,
                edge_attrs=edge_kinds,
                cell_size=cell_size,
            )
            with db.bulk_write_mode():
                db.write_node_arrays(ids, **node_arrays)
                if u is not None and v is not None:
                    db.write_edge_arrays(u, v, **edge_arrays)
            if db.meta["generation"] is None:
                # no node: nothing was appended, so nothing was merged. The
                # arrays of an empty graph are built all the same, so that
                # it is a finalized graph like any other
                empty = db._empty()
                db._build(
                    empty["ids"],
                    {k: empty[f"node_{k}"] for k in db.meta["node_attrs"]},
                    None,
                    sort_nodes=True,
                )
            db.close()
        except BaseException:
            shutil.rmtree(scratch, ignore_errors=True)
            raise
        replace_directory(scratch, path)
        return cls(path, mode="r+")

    # -- metadata -----------------------------------------------------------

    def _write_meta(self) -> None:
        tmp = self.path / f"meta.json.{uuid.uuid4().hex}"
        tmp.write_text(json.dumps(self.meta, indent=2) + "\n")
        os.replace(tmp, self.path / "meta.json")

    def _reload_meta(self) -> None:
        self.meta = json.loads((self.path / "meta.json").read_text())

    @staticmethod
    def _types(specs: dict[str, dict]) -> dict[str, AttributeType]:
        out: dict[str, AttributeType] = {}
        for name, spec in specs.items():
            dtype = np.dtype(spec["dtype"]).name
            out[name] = Vec(dtype, spec["size"]) if spec["size"] else dtype
        return out

    @property
    def node_attrs(self) -> dict[str, AttributeType]:
        """The node attributes and their types: those stored in the graph,
        then the extra fields this handle was opened with."""
        return self._types(self._node_specs())

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
        arrays = {
            name: np.load(folder / f"{name}.npy", mmap_mode="r")
            for name in self._array_names()
        }
        for extra in self._extra_dirs:
            extra.reload()
        self._map_extra_fields()
        for name, extra in self._extra.items():
            arrays[f"node_{name}"] = self._extra_array(extra, name, len(arrays["ids"]))
        self._arrays = arrays
        return self._arrays

    def _readable(self) -> dict[str, np.ndarray] | None:
        if self._pending():
            raise UnfinalizedWrites(
                f"{self.path} has writes that are not in its arrays yet; call "
                "finalize() (leaving bulk_write_mode in the main process does)"
            )
        return self._open_arrays()

    # -- writing ------------------------------------------------------------

    def _segment(self, kind: str, folder: Path | None = None) -> _Segment:
        """This handle's segment of ``kind`` in the graph's directory, or in
        ``folder``, a directory of extra fields."""
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")
        if folder is None:
            self._refuse(f"cannot write {kind}")
        key = (kind, folder)
        if key not in self._segments:
            name = f"{kind}-{os.getpid()}-{uuid.uuid4().hex}.npys"
            self._segments[key] = _Segment((folder or self.path) / "segments" / name)
        return self._segments[key]

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
        (the position among them). Through a handle with ``extra_fields``:
        values of extra attributes for nodes of the graph, no position."""
        if self._extra_dirs:
            self._write_extra_arrays(ids, attrs)
            self._after_write()
            return
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
        self._refuse("cannot write edges")
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
        if self._extra_dirs:
            self._write_extra_rows(nodes, roi, attributes)
            self._after_write()
            return
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
        self._refuse("cannot write edges")
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
        if write_edges:
            self._refuse("cannot write edges (pass write_edges=False)")
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
        if edge_attrs:
            self._refuse("cannot write edge attributes")
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
        except BaseException:
            # what was written is merged; the error of the block is the one
            # that is raised, whatever the finalize has to say
            self._bulk_depth -= 1
            for segment in self._segments.values():
                segment.close()
            if not worker and self._bulk_depth == 0:
                try:
                    self.finalize()
                except Exception:
                    logger.exception(
                        "finalize of %s failed after an error in bulk_write_mode",
                        self.path,
                    )
            raise
        else:
            self._bulk_depth -= 1
            for segment in self._segments.values():
                segment.close()
            if not worker and self._bulk_depth == 0:
                self.finalize()

    def drop_edges(self) -> None:
        """Remove every edge, keeping the nodes."""
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")
        self._refuse("cannot drop edges")
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
        return np.asarray(~(missing.all(axis=1) if values.ndim > 1 else missing))

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
        dropped. Holds the whole graph in memory.

        Through a handle with ``extra_fields`` it merges the segments of
        the extra directories instead, into the arrays of the attributes
        that were written, and leaves the graph alone."""
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")
        for segment in self._segments.values():
            segment.close()
        self._segments = {}
        # one finalize at a time: a second process waits, then finds the
        # segments the first one merged gone
        if self._extra_dirs:
            # every directory is finalized; one that fails does not keep
            # the others from being merged
            errors: list[Exception] = []
            for extra in self._extra_dirs:
                if not extra.pending():
                    continue  # nothing to merge; the directory may not exist
                try:
                    with _locked(extra.path):
                        self._finalize_extra(extra)
                except Exception as error:
                    errors.append(error)
            self._arrays = None
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise ExceptionGroup(
                    f"finalize failed in {len(errors)} extra directories", errors
                )
            return
        with _locked(self.path):
            self._finalize()

    def _finalize(self) -> None:
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
            path.unlink(missing_ok=True)

    def _build(self, ids, nodes, edges, sort_nodes: bool) -> None:
        position = (
            nodes[self.position_attribute].reshape(len(ids), self.ndims).astype(float)
        )
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
        where = np.asarray(position[candidates]).reshape(len(candidates), self.ndims)
        inside = ((where >= lo) & (where < hi)).all(axis=1)
        return candidates[inside]

    def _node_columns(self, arrays, rows, attrs: list[str]) -> dict[str, np.ndarray]:
        out = {"id": np.asarray(arrays["ids"][rows])}
        specs = self._node_specs()
        for name in attrs:
            if name not in specs:
                raise ValueError(f"unknown node attribute {name!r}")
        for extra in {
            id(self._extra[a]): self._extra[a] for a in attrs if a in self._extra
        }.values():
            if extra.pending():
                raise UnfinalizedWrites(
                    f"{extra.path} has writes that are not in its arrays yet; "
                    "call finalize(), or read without its attributes"
                )
        for name in attrs:
            out[name] = np.asarray(arrays[f"node_{name}"][rows])
        return out

    def read_arrays(
        self,
        roi: Roi | None = None,
        num_steps: int = 0,
        node_attrs: list[str] | None = None,
        edge_attrs: list[str] | None = None,
        step_edges: Sequence[str] | None = None,
        edge_ends: str = "both",
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """The nodes in ``roi``, every node within ``num_steps`` edges of
        them, and the edges among those nodes, as arrays.

        ``edge_ends`` says which ends of an edge have to be among the nodes
        read for the edge to be returned:

        ``"both"`` (the default)
            The edges among the nodes read. Every ``u`` and ``v`` is an
            ``id`` in ``nodes``, so the result is a graph on its own. An
            edge that leaves the nodes read is not returned.
        ``"u"``
            The edges whose smaller-id end is among the nodes read: the
            rule of :meth:`read_edges` and :meth:`read_graph`, here and in
            the SQL backends. With ``num_steps=0`` the edges are those of
            ``read_edges(roi)``. A block sees the edges that leave it, and
            blocks that tile a graph see every edge once.
        ``"either"``
            The edges with at least one end among the nodes read: the
            rule of ``read_edges(roi, fetch_on_v=True)``.

        With ``"u"`` and ``"either"`` an edge can name a node that is not in
        ``nodes``; its attributes were not read.

        ``step_edges`` names the edge attributes that make an edge a step:
        an edge is followed when any of them is set on it (not NaN). They
        are float attributes: an integer or bool attribute has no unset
        value and is refused. ``None`` follows every edge; an empty
        sequence names no attribute, so no edge is followed. Which edges
        are returned is the matter of ``edge_ends`` alone, not of which
        were followed.

        Returns ``(nodes, edges)``. ``nodes`` has ``id``, ``step`` (0 for a
        node in the region, else the step that first reached it) and the
        requested attributes; ``edges`` has ``u``, ``v`` (ids, ``u < v``,
        each edge once) and the requested attributes. All attributes are
        returned when none are named."""
        if edge_ends not in ("both", "u", "either"):
            raise ValueError(
                f"edge_ends is {edge_ends!r}; it is one of 'both', 'u', 'either'"
            )
        arrays = self._readable()
        node_attrs = list(self._node_specs()) if node_attrs is None else node_attrs
        edge_attrs = list(self.meta["edge_attrs"]) if edge_attrs is None else edge_attrs
        if arrays is None:
            arrays = self._empty()
        for name in step_edges or ():
            if name not in self.meta["edge_attrs"]:
                raise ValueError(f"unknown edge attribute {name!r}")
            dtype = np.dtype(self.meta["edge_attrs"][name]["dtype"])
            if dtype.kind != "f":
                raise ValueError(
                    f"step_edges: edge attribute {name!r} is {dtype.name}; only "
                    "a float attribute can be unset on an edge"
                )
        indptr, nbr = arrays["indptr"], arrays["nbr"]
        selected = self._rows_in(arrays, roi)
        step = np.zeros(len(selected), dtype=np.int16)
        frontier = selected
        for k in range(1, num_steps + 1):
            if len(frontier) == 0:
                break
            at, _ = _gather(indptr, frontier)
            if step_edges is not None:
                followed = np.zeros(len(at), dtype=bool)
                for name in step_edges:
                    followed |= self._present(np.asarray(arrays[f"edge_{name}"][at]))
                at = at[followed]
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
        if edge_ends != "both":
            # the rule of read_edges, applied to the nodes read
            return nodes, self._edge_columns(
                arrays, selected, edge_ends == "either", edge_attrs
            )
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

    def read_block(
        self,
        roi: Roi,
        context: Sequence[float] | float | None = None,
        num_steps: int = 0,
        node_attrs: list[str] | None = None,
        edge_attrs: list[str] | None = None,
        step_edges: Sequence[str] | None = None,
        edge_ends: str = "both",
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """A block with its surroundings, as arrays: what a computation that
        writes one value per node of ``roi`` reads.

        The block's own nodes are those inside ``roi``. The surroundings
        are every node within ``context`` of the box (one distance or one
        per dimension, rounded up to whole units), and every node within
        ``num_steps`` edges of any of those, wherever it lies.

        Returns ``(nodes, edges)`` as :meth:`read_arrays` does, with one more
        array in ``nodes``: ``own``, true for the nodes inside ``roi``.
        ``step`` is 0 for the block's own nodes and for those the context
        reached; the steps are counted from there. ``edge_ends`` is that of
        :meth:`read_arrays`: it applies to all the nodes read, the
        surroundings included.
        """
        if num_steps < 0:
            raise ValueError("num_steps must be >= 0")
        wanted = list(self._node_specs()) if node_attrs is None else node_attrs
        if context is None:
            nodes, edges = self.read_arrays(
                roi, num_steps, wanted, edge_attrs, step_edges, edge_ends
            )
            nodes["own"] = nodes["step"] == 0
            return nodes, edges
        distance = np.broadcast_to(np.asarray(context, dtype=float), (self.ndims,))
        if not (distance >= 0).all():  # before rounding: ceil(-0.5) is -0
            raise ValueError("context must be >= 0")
        grow = np.ceil(distance)
        amount = Coordinate(int(g) for g in grow)
        position = self.position_attribute
        nodes, edges = self.read_arrays(
            roi.grow(amount, amount),
            num_steps,
            list(dict.fromkeys([*wanted, position])),
            edge_attrs,
            step_edges,
            edge_ends,
        )
        at = np.asarray(nodes[position], dtype=float).reshape(len(nodes["id"]), -1)
        own = nodes["step"] == 0
        bounds = self._bounds(roi)
        if bounds is not None:
            own &= ((at >= bounds[0]) & (at < bounds[1])).all(axis=1)
        nodes["own"] = own
        if position not in wanted:
            del nodes[position]
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
        read_attrs = list(self._node_specs()) if read_attrs is None else read_attrs
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
        ``nodes``; with ``fetch_on_v`` also those whose larger one is.
        :meth:`read_arrays` returns the same edges as arrays with
        ``edge_ends="u"`` and ``edge_ends="either"``."""
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
