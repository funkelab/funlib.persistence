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

**Writing.** The arrays of a store are never changed in place. A write is
a *segment*: an append-only file of records, one record per
:meth:`NodeArraysWriter.append`. A store is built from segments in a
scratch directory beside ``path`` and moved to ``path`` when it is
complete, so ``path`` holds a whole store or none. This is the model of
:class:`~funlib.persistence.graphs.CSRGraphDataBase`: segments, then a
finalize.

One writer
    ``with NodeArraysWriter(path, attrs) as out`` keeps its segment in a
    scratch directory of its own and builds the store when the block is
    left without an error. The store then holds what this writer appended
    and replaces what was at ``path``. Two such writers to one path do not
    touch each other's files; the one that finishes last stays.

Several writers
    ``NodeArraysWriter(path, attrs, worker=True)`` appends to a segment of
    its own in ``<path>.segments/``, in any number of processes at once,
    on any machine that sees the directory. Each ``append`` is one record,
    flushed before it returns. :func:`finalize_node_arrays`, called once
    the writers are done, merges the store at ``path`` (if there is one)
    with every segment into a new store and removes the segments. Until
    then a reader sees the store of the last finalize, or none.

Limits of the several-writers mode:

* Call :func:`finalize_node_arrays` when the writers are done. A record
  appended to a segment while that segment is being merged is lost with
  it. Two finalizes at once are safe: the second waits for the first.
* A writer that is killed leaves the records it had completed. A record
  cut short is ignored, with a warning; the block it belonged to has to
  be written again.
* A node id written twice is refused at finalize by default, and the
  segments stay. ``duplicates="last"`` keeps the last value instead, for
  a block that was run again after its first write had landed.
* Finalize copies every value once more and sorts the ids in memory
  (16 bytes per node); the values themselves are streamed.
* ``path`` is replaced by two renames. A reader that opens the store in
  the instant between them can find nothing there. A :class:`NodeArrays`
  that is open keeps reading the store it opened, whole, and has to be
  opened again to see the new one. Do not replace a store with a single
  writer while workers write to it or a finalize runs.
* Segments left by a run that was given up are merged by the next
  finalize. Remove ``<path>.segments`` to discard them.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ._files import replace_directory, scratch_beside

__all__ = [
    "FORMAT_VERSION",
    "NodeArrays",
    "NodeArraysWriter",
    "finalize_node_arrays",
    "write_node_arrays",
]

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
META = "meta.json"
_ID = "ids"  # the id field of a record; not an attribute name
_ID_DTYPE = np.dtype("<u8")
_ROWS = 1 << 20  # rows copied at a time
_LOCK = "finalize.lock"
_OPEN_ATTEMPTS = 20


def _file(name: str) -> str:
    return f"node_{name}.npy"


def _segments_dir(path: Path) -> Path:
    return path.with_name(path.name + ".segments")


class NodeArrays:
    """Stored node attributes, memory-mapped and read by node id.

    Every array is mapped when the reader is opened, so a reader holds
    one whole store: the one that was at ``path`` then. A store that
    replaces it later (a new write, a finalize) is seen by opening a
    reader again; this one keeps reading the store it opened."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        meta = self.path / META
        if not meta.is_file():
            raise FileNotFoundError(
                f"{self.path} holds no {META}: not a node attribute store"
            )
        for attempt in range(_OPEN_ATTEMPTS):
            try:
                before = os.stat(self.path)
                self.meta: dict[str, Any] = json.loads(meta.read_text())
                if self.meta.get("version") != FORMAT_VERSION:
                    raise ValueError(
                        f"{self.path}: node attribute store version "
                        f"{self.meta.get('version')!r}, this reader knows "
                        f"{FORMAT_VERSION}"
                    )
                files = ["ids.npy", "order.npy", *map(_file, self.meta["attrs"])]
                self._arrays: dict[str, np.ndarray] = {
                    file: np.load(self.path / file, mmap_mode="r") for file in files
                }
                after = os.stat(self.path)
            except FileNotFoundError:
                # the store is being replaced: its directory was moved away
                # between two of the steps above
                if attempt == _OPEN_ATTEMPTS - 1:
                    raise
                time.sleep(0.01)
                continue
            # the same directory before and after: one store, not the
            # meta.json of one and the arrays of the next
            if (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns):
                return
        raise RuntimeError(f"{self.path} kept being replaced while it was opened")

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

    @property
    def ids(self) -> np.ndarray:
        """Every stored node id, in stored order."""
        return self._arrays["ids.npy"]

    def array(self, name: str) -> np.ndarray:
        """One attribute of every node, memory-mapped, aligned with
        :attr:`ids`."""
        if name not in self.meta["attrs"]:
            raise KeyError(
                f"{self.path} holds no attribute {name!r}; it holds "
                f"{sorted(self.meta['attrs'])}"
            )
        return self._arrays[_file(name)]

    def rows(self, ids: Any) -> tuple[np.ndarray, np.ndarray]:
        """Where each of ``ids`` is stored: ``(found, rows)``; a row is
        only meaningful where ``found``."""
        ids = np.asarray(ids, dtype=np.uint64)
        order = self._arrays["order.npy"]
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


# -- segments and the build of a store ---------------------------------------


def _specs(record: np.dtype) -> dict[str, dict[str, Any]]:
    """The attributes a record dtype declares, as ``meta.json`` states them."""
    out = {}
    for name, (field, *_) in (record.fields or {}).items():
        if name != _ID:
            out[name] = {
                "dtype": field.base.str,
                "size": field.shape[0] if field.shape else 0,
            }
    return out


def _scan(path: Path) -> tuple[np.dtype | None, list[tuple[int, int]]]:
    """The whole records of a segment: their dtype and, for each, where
    its rows start in the file and how many there are."""
    size = path.stat().st_size
    record: np.dtype | None = None
    found: list[tuple[int, int]] = []
    with open(path, "rb") as f:
        while f.tell() < size:
            try:
                np.lib.format.read_magic(f)
                shape, _, dtype = np.lib.format.read_array_header_1_0(f)
                end = f.tell() + shape[0] * dtype.itemsize
            except (ValueError, EOFError, OSError):
                end = size + 1
            if end > size:
                # a writer that died mid-record leaves a short tail
                logger.warning("ignoring a truncated record in %s", path)
                break
            if record is None:
                record = dtype
            elif dtype != record:
                raise ValueError(f"{path} holds records of two kinds")
            found.append((f.tell(), shape[0]))
            f.seek(end)
    return record, found


_Scanned = list[tuple[Path, np.dtype, list[tuple[int, int]]]]


def _chunks(existing: NodeArrays | None, segments: _Scanned) -> Iterator[Any]:
    """Everything to store, a bounded number of rows at a time: the store
    that is there first, then the segments in the order given."""
    if existing is not None:
        columns = {_ID: existing.ids}
        columns.update({name: existing.array(name) for name in existing.meta["attrs"]})
        for start in range(0, existing.num_nodes, _ROWS):
            yield {k: np.asarray(v[start : start + _ROWS]) for k, v in columns.items()}
    for path, record, found in segments:
        with open(path, "rb") as f:
            for start, count in found:
                f.seek(start)
                yield np.fromfile(f, dtype=record, count=count)


def _build(
    path: Path,
    out: Path,
    existing: NodeArrays | None,
    segments: list[Path],
    source: str | None,
    duplicates: str,
) -> None:
    """Write the store of ``existing`` and ``segments`` to the new directory
    ``out`` and move it to ``path``."""
    scanned: _Scanned = []
    attrs: dict[str, dict[str, Any]] | None = (
        None if existing is None else existing.meta["attrs"]
    )
    for segment in segments:
        record, found = _scan(segment)
        if record is None:
            continue
        if attrs is None:
            attrs = _specs(record)
        elif _specs(record) != attrs:
            raise ValueError(
                f"{segment} was written with the attributes {_specs(record)}, "
                f"the rest of {path} with {attrs}"
            )
        scanned.append((segment, record, found))
    if attrs is None:
        raise FileNotFoundError(f"nothing was written to {path}")
    total = (existing.num_nodes if existing is not None else 0) + sum(
        count for _, _, found in scanned for _, count in found
    )

    out.mkdir(parents=True)
    files = {_ID: "ids.npy", **{name: _file(name) for name in attrs}}
    dtypes = {_ID: _ID_DTYPE, **{k: np.dtype(v["dtype"]) for k, v in attrs.items()}}
    tails = {
        _ID: (),
        **{k: (v["size"],) if v["size"] else () for k, v in attrs.items()},
    }

    def write(rows: int, chunks: Iterable[Any]) -> None:
        handles = {name: open(out / file, "wb") for name, file in files.items()}
        try:
            for name, handle in handles.items():
                np.lib.format.write_array_header_1_0(
                    handle,
                    {
                        "descr": np.lib.format.dtype_to_descr(dtypes[name]),
                        "fortran_order": False,
                        "shape": (rows, *tails[name]),
                    },
                )
            for chunk in chunks:
                for name, handle in handles.items():
                    np.ascontiguousarray(chunk[name], dtype=dtypes[name]).tofile(handle)
        finally:
            for handle in handles.values():
                handle.close()

    write(total, _chunks(existing, scanned))
    ids = np.load(out / "ids.npy")
    order = np.argsort(ids, kind="stable")
    again = np.flatnonzero(ids[order][1:] == ids[order][:-1])
    if len(again):
        if duplicates != "last":
            some = np.unique(ids[order][again])[:5].tolist()
            raise ValueError(
                f"{path}: a node id was written twice (for example {some}); a "
                "node belongs to one block"
            )
        # of the rows with one id the last written stays; the sort is
        # stable, so that is the last of each run of equal ids
        last = np.ones(total, dtype=bool)
        last[again] = False
        keep = np.zeros(total, dtype=bool)
        keep[order[last]] = True
        for file in files.values():
            os.rename(out / file, out / f"{file}.all")
        every = {
            name: np.load(out / f"{file}.all", mmap_mode="r")
            for name, file in files.items()
        }
        total = int(keep.sum())
        write(
            total,
            (
                {
                    name: np.asarray(values[start : start + _ROWS])[
                        keep[start : start + _ROWS]
                    ]
                    for name, values in every.items()
                }
                for start in range(0, len(keep), _ROWS)
            ),
        )
        del every
        for file in files.values():
            (out / f"{file}.all").unlink()
        ids = np.load(out / "ids.npy")
        order = np.argsort(ids, kind="stable")
    np.save(out / "order.npy", order.astype("<i8"))
    meta = {
        "version": FORMAT_VERSION,
        "num_nodes": total,
        "attrs": attrs,
        "source": source,
    }
    (out / META).write_text(json.dumps(meta, indent=2) + "\n")
    replace_directory(out, path)


@contextmanager
def _finalize_lock(folder: Path) -> Generator[bool]:
    """One finalize at a time. Yields False when ``folder`` is not there:
    no worker wrote, or another finalize merged everything in it."""
    name = folder / _LOCK
    fd = None
    while fd is None:
        try:
            fd = os.open(name, os.O_RDWR | os.O_CREAT, 0o666)
        except FileNotFoundError:
            break
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            # the finalize waited for removes the lock file it held
            current = os.fstat(fd).st_ino == os.stat(name).st_ino
        except FileNotFoundError:
            current = False
        if not current:
            os.close(fd)
            fd = None
    try:
        yield fd is not None
    finally:
        if fd is not None:
            os.close(fd)


def finalize_node_arrays(
    path: Path | str, *, source: str | None = None, duplicates: str = "raise"
) -> Path:
    """Merge what the worker writers of ``path`` appended into the store.

    The new store holds the nodes of the store at ``path``, if there is
    one, and of every segment in ``<path>.segments``; it replaces the
    store, and the segments are removed. Call it in one process when the
    writers are done (the limits are in the module documentation). With
    nothing to merge it leaves the store as it is.

    ``source`` is recorded in the new store; left out, the store keeps
    the ``source`` it had.

    ``duplicates`` says what a node id written more than once is:
    ``"raise"`` (the default) refuses it with ``ValueError`` and leaves
    the store and the segments as they were; ``"last"`` keeps the value
    written last. Segments are taken in the order their writers were
    opened, by the clocks of the machines that opened them, after the
    store that is already there.
    """
    path = Path(path)
    if duplicates not in ("raise", "last"):
        raise ValueError(f"duplicates is {duplicates!r}; it is 'raise' or 'last'")
    folder = _segments_dir(path)
    with _finalize_lock(folder) as pending:
        segments = sorted(folder.glob("*.npys")) if pending else []
        stored = (path / META).is_file()
        if not segments and not stored:
            raise FileNotFoundError(
                f"nothing was written to {path}: there is no store and no "
                f"segment in {folder}"
            )
        if segments:
            existing = NodeArrays(path) if stored else None
            if source is None and existing is not None:
                source = existing.source
            out = scratch_beside(path, "building")
            try:
                _build(path, out, existing, segments, source, duplicates)
            finally:
                shutil.rmtree(out, ignore_errors=True)
            for segment in segments:
                segment.unlink(missing_ok=True)
        if pending:
            # leave nothing beside the store, unless a writer is still there
            try:
                (folder / _LOCK).unlink()
                folder.rmdir()
            except OSError:
                pass
    return path


class NodeArraysWriter:
    """Writes a node attribute store a block at a time.

    ::

        with NodeArraysWriter(path, {"embedding": 64}) as out:
            for ids, embedding in blocks:
                out.append(ids, embedding=embedding)

    ``attrs`` maps each attribute to its vector length, ``None`` for a
    scalar; ``dtypes`` names a dtype for those that are not ``float32``.
    ``ids`` is not an attribute name.

    As above, with one writer, the store appears at ``path`` only when the
    block is left without an error, and it replaces what was there. An
    error leaves ``path`` as it was and nothing beside it.

    With ``worker=True`` the writer is one of several, in as many
    processes::

        # in each worker
        with NodeArraysWriter(path, {"embedding": 64}, worker=True) as out:
            for ids, embedding in my_blocks:
                out.append(ids, embedding=embedding)

        # in one process, when the workers are done
        finalize_node_arrays(path, source="model 3")

    Every ``append`` is then a record in a segment file of this writer in
    ``<path>.segments``, complete when ``append`` returns, and it stays
    whatever happens to the writer afterwards. Nothing changes at ``path``
    before :func:`finalize_node_arrays`, which adds the segments to the
    store that is there. Every writer of a store declares the same
    attributes. Open the writer in the process that appends to it. The
    design and its limits are in the documentation of
    :mod:`funlib.persistence.graphs.node_arrays`.
    """

    def __init__(
        self,
        path: Path | str,
        attrs: Mapping[str, int | None],
        *,
        dtypes: Mapping[str, Any] | None = None,
        source: str | None = None,
        worker: bool = False,
    ):
        self.path = Path(path)
        self.attrs = dict(attrs)
        if not self.attrs:
            raise ValueError("a node attribute store holds at least one attribute")
        if _ID in self.attrs:
            raise ValueError(
                f"{_ID!r} is not an attribute name: it names the node ids of a store"
            )
        for name, length in self.attrs.items():
            if length is not None and (isinstance(length, bool) or length < 1):
                raise ValueError(
                    f"attribute {name!r}: a vector has a length of 1 or more and "
                    f"a scalar has None, not {length!r}"
                )
        self.dtypes = {
            name: np.dtype((dtypes or {}).get(name, "float32")).newbyteorder("<")
            for name in self.attrs
        }
        for name, dtype in self.dtypes.items():
            if dtype.kind not in "fiub":
                raise ValueError(f"attribute {name!r}: {dtype} is not a number type")
        if worker and source is not None:
            raise ValueError(
                "source belongs to the store, not to one of its writers: give "
                "it to finalize_node_arrays"
            )
        self.source = source
        self.worker = worker
        self.num_nodes = 0
        self._record = np.dtype(
            [(_ID, _ID_DTYPE)]
            + [
                (name, self.dtypes[name], () if length is None else (length,))
                for name, length in self.attrs.items()
            ]
        )
        self._scratch: Path | None = None
        self._segment: Path | None = None
        self._file: Any = None

    def __enter__(self) -> NodeArraysWriter:
        if self.worker:
            meta = self.path / META
            if meta.is_file():
                stored = json.loads(meta.read_text()).get("attrs")
                if stored != _specs(self._record):
                    raise ValueError(
                        f"{self.path} holds the attributes {stored}; this writer "
                        f"declares {_specs(self._record)}"
                    )
            folder = _segments_dir(self.path)
        else:
            # a directory no other writer has: two writers to one path
            # never see each other's files
            folder = self._scratch = scratch_beside(self.path, "writing")
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{time.time_ns():020d}-{os.getpid()}-{uuid.uuid4().hex}.npys"
        self._segment = folder / name
        self._file = open(self._segment, "ab")
        # a record of no rows states the attributes, so that writers with
        # no node to write still leave a store, an empty one
        self._write(np.zeros(0, dtype=self._record))
        return self

    def _write(self, records: np.ndarray) -> None:
        np.lib.format.write_array(
            self._file, records, version=(1, 0), allow_pickle=False
        )
        self._file.flush()

    def append(self, ids: Any, **values: Any) -> None:
        """Add nodes: their ids and one array per attribute, aligned."""
        if self._file is None:
            raise RuntimeError("append belongs inside the writer's with block")
        ids = np.ascontiguousarray(ids, dtype=_ID_DTYPE)
        if set(values) != set(self.attrs):
            raise ValueError(
                f"expected the attributes {sorted(self.attrs)}, got {sorted(values)}"
            )
        records = np.empty(len(ids), dtype=self._record)
        records[_ID] = ids
        for name, length in self.attrs.items():
            block = np.asarray(values[name], dtype=self.dtypes[name])
            shape = (len(ids),) if length is None else (len(ids), length)
            if block.shape != shape:
                raise ValueError(
                    f"attribute {name!r} has shape {block.shape}, expected {shape}"
                )
            records[name] = block
        if len(records):
            self._write(records)
        self.num_nodes += len(ids)

    def __exit__(self, kind: Any, *_: Any) -> None:
        self._file.close()
        self._file = None
        if self._scratch is None or self._segment is None:
            return  # a worker: its records wait for finalize_node_arrays
        try:
            if kind is None:
                _build(
                    self.path,
                    self._scratch / "store",
                    None,
                    [self._segment],
                    self.source,
                    "raise",
                )
        finally:
            shutil.rmtree(self._scratch, ignore_errors=True)


def write_node_arrays(
    path: Path | str,
    ids: Any,
    values: Mapping[str, Any],
    *,
    source: str | None = None,
) -> Path:
    """Write node attributes that are in memory as one store, replacing
    what is at ``path``."""
    arrays = {name: np.asarray(value) for name, value in values.items()}
    attrs = {
        name: (int(value.shape[1]) if value.ndim == 2 else None)
        for name, value in arrays.items()
    }
    dtypes = {name: value.dtype for name, value in arrays.items()}
    with NodeArraysWriter(path, attrs, dtypes=dtypes, source=source) as out:
        out.append(ids, **arrays)
    return Path(path)
