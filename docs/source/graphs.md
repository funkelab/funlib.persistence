# Graph databases

A graph database stores a `networkx` graph whose nodes have positions, and
returns the subgraph inside a `Roi`:

```python
from pathlib import Path

from funlib.geometry import Roi
from funlib.persistence.graphs import SQLiteGraphDataBase
from funlib.persistence.types import Vec

db = SQLiteGraphDataBase(
    Path("fragments.db"),
    position_attribute="position",
    mode="w",
    node_attrs={"position": Vec(int, 3), "score": float},
    edge_attrs={"merge_score": float},
)
db.write_graph(graph)                    # nodes and edges of a networkx graph
sub = db[Roi((0, 0, 0), (100, 100, 100))]   # the nodes in the ROI and their edges
```

`SQLiteGraphDataBase` keeps everything in one file and suits a single
process. `PgSQLGraphDatabase` talks to a PostgreSQL server and is the one
for many workers writing at once, which is how daisy and volara use it.
Both implement `GraphDataBase`, so the code that reads and writes subgraphs
does not depend on the backend.

Node and edge attributes are declared up front with their types. A scalar
attribute is a Python type such as `int`, `float` or `str`; a fixed-length
vector such as a position is a `Vec(dtype, size)` from
`funlib.persistence.types`, stored as one column per component. The
`position_attribute` is the vector the ROI query filters on.

An edge belongs to the ROI that contains its first endpoint `u`.
`write_graph(graph, roi)` writes the nodes whose position is inside the ROI
and the edges whose `u` is, and skips the rest, so blocks that each write
their own ROI never write a node or an edge twice. `read_graph(roi)` returns
the nodes inside the ROI and the edges whose `u` is inside it, which is how
a block sees the edges that leave it; `fetch_on_v=True` adds the edges
whose `v` is inside as well. A ROI that is unbounded on every axis, or no
ROI, reads the whole graph.

## A graph in memory-mapped arrays

`CSRGraphDataBase` keeps a graph as plain `.npy` arrays under a directory,
sorted so that the nodes of a region are a few runs of consecutive rows. It
is read by memory mapping and suits graphs that are written once and then
read block by block. Attributes are numbers or vectors of numbers.

A whole graph held in arrays is written in one call, and read back as
arrays:

```python
import numpy as np
from funlib.geometry import Roi
from funlib.persistence.graphs import CSRGraphDataBase

db = CSRGraphDataBase.from_arrays(
    "fragments.csr",
    ids,                                   # (N,) node ids
    {"position": positions, "size": sizes},  # (N, 3) and (N,)
    u, v,                                  # (M,) endpoint ids
    {"merge_score": scores},               # (M,)
    cell_size=3000.0,
)
nodes, edges = db.read_arrays(Roi((0, 0, 0), (100, 100, 100)), num_steps=1)
```

`read_arrays` returns the nodes in the ROI, every node within `num_steps`
edges of them, and the edges among the nodes read. `step_edges` names the
float edge attributes that make an edge a step: an edge is followed when
one of them is set on it.

By default an edge is returned when both of its ends are among the nodes
read, so the arrays are a graph on their own. That is not the rule of
`read_graph`, which also returns the edges that leave the ROI.
`edge_ends` selects the rule:

| `edge_ends` | an edge is returned when | same edges as |
|---|---|---|
| `"both"` (default) | both ends are among the nodes read | |
| `"u"` | its smaller-id end is among the nodes read | `read_graph(roi)`, `read_edges(roi)` on every backend |
| `"either"` | at least one end is among the nodes read | `read_graph(roi, fetch_on_v=True)` |

With `"u"` or `"either"` an edge can name a node whose attributes were not
read. Code that reads one backend as arrays and another through
`read_graph` asks for `edge_ends="u"` to get the same edges from both.

`read_block` is the read of a computation that writes one value per node of
a block: the nodes of the ROI, marked `own`, with the nodes within a
distance `context` of it and within `num_steps` edges of those.

```python
nodes, edges = db.read_block(roi, context=500, num_steps=2)
values = compute(nodes, edges)[nodes["own"]]
```

`CSRGraphDataBase.is_database(path)` tells such a directory from any other.

## Values on the nodes of a graph: extra fields

Values computed on a finished graph, such as an embedding per node, do not
have to go into the graph's directory. The graph is opened with
`extra_fields`, the directory that holds them, and the extra attributes are
written and read through the database like stored ones:

```python
from funlib.persistence.graphs import CSRGraphDataBase

g = CSRGraphDataBase("fragments.csr", mode="r+", extra_fields="pred/model_a")
with g.bulk_write_mode():
    for roi in blocks:
        nodes, edges = g.read_block(roi, num_steps=2, node_attrs=["position"])
        own = nodes["own"]
        e, l = model(nodes, edges)          # float32 (N, 64) and int32 (N,)
        g.write_node_arrays(nodes["id"][own], embedding=e[own], label=l[own])

g = CSRGraphDataBase("fragments.csr", extra_fields="pred/model_a")
nodes, edges = g.read_arrays(roi, node_attrs=["position", "embedding"])
```

The path is enough. No attribute is named and no type is given: `meta.json`
in the directory records what it holds, and opening the graph with it
exposes every attribute there with its recorded type. `g.node_attrs` lists
the extra attributes after the stored ones, and `read_arrays`, `read_block`,
`read_nodes` and `read_graph` return them by name, for the same regions and
steps. `CSRGraphDataBase.stored_extra_fields(path)` tells what a directory
holds without opening a graph.

`extra_fields` also takes several directories,
`extra_fields=["pred/model_a", "pred/model_b"]`, and exposes the attributes
of all of them. An attribute that two of them hold, or that the graph itself
stores, is refused when the graph is opened. A directory that is empty or
does not exist yet is created by the first write in mode `"r+"`; in mode
`"r"` it is an error.

**A new attribute is created by the first write that names it**, with the
type of the values written: `embedding=e` with `e` a float32 array of shape
`(N, 64)` declares a float32 vector of 64. The declaration is recorded in the
directory's `meta.json` under the directory's lock, so worker processes that
create the same attribute at the same moment agree on it. From then on an
array written to the attribute has that dtype and vector size, or the write
raises `ValueError`; nothing is cast. The writes that take nodes as
dictionaries (`write_nodes`, `bulk_write_nodes`, `write_attrs`) create an
attribute named in `attributes` from the values the nodes carry, when those
are numbers of one kind (bool, int or float, stored as bool, int64 or
float64) and one length; otherwise they raise and ask for a type. Values for
an attribute that exists are cast to its type.

`node_attrs` is never needed for an extra attribute. It may name one with a
type, `node_attrs={"score": "float32"}`, to create it up front in mode
`"r+"`, for example float32 where Python numbers would give float64, or to
have the recorded type checked.

**A new attribute goes into the one directory that is open.** With several
directories open, a write that names an attribute none of them holds raises:
open the graph with the directory it is to go into. Attributes that exist
are written into the directory that holds them.

**The graph is read-only through such a handle.** Nothing in the graph's
directory is created, changed or locked. Writing edges, writing nodes with
stored attributes, and `drop_edges` raise `GraphIsReadOnly`. The node
writes (`write_node_arrays`, `write_attrs`, `write_nodes`,
`bulk_write_nodes`) take extra attributes only, for nodes that exist, and
need no position.

Writing is the database's: each handle appends to a segment file of its own
in the extra directory, and `finalize()`, or leaving `bulk_write_mode` in
the main process, merges the segments into the arrays. In a blockwise task
each worker opens the graph with the same `extra_fields` and writes inside
`bulk_write_mode(worker=True)`; the main process finalizes at the end. The
finalize rebuilds the arrays of the attributes that were written and reads
the graph's node ids, nothing else of the graph. A later finalize adds to
the values that are there, so a run can be taken up again; a node written
twice keeps its last value.

The values are stored in the graph's row order, one array per attribute, so
a region read has the graph's locality and there is no lookup by id. This
ties a directory to one graph. It records the graph's node count, the name
of the graph's arrays, and a checksum of its node ids in row order, and
opening it with a graph of another node count, other ids or another row
order raises `ValueError`. Adding edges to the graph, or building it again
from the same nodes, keeps the extra fields valid; adding nodes does not.

```
pred/model_a/meta.json                 version, extends, node_attrs, arrays
pred/model_a/node_embedding-<x>.npy    (N, 64), in the graph's row order
pred/model_a/node_label-<x>.npy        (N,)
pred/model_a/segments/                 writes not yet in the arrays
pred/model_a/finalize.lock
```

What to know:

- A node that was never written reads as NaN, and as 0 for an integer or a
  bool, which cannot be told from a written 0. Use a float, or a value
  other than 0, where the difference matters. A written NaN counts as not
  written.
- A value for an id that is not a node of the graph is found at the
  finalize, which raises `ValueError` and changes nothing. The segments
  stay; remove the one named in the error to go on.
- Finalize when the writers are done: a record appended to a segment while
  it is being merged is lost. Several finalizes at once run one after the
  other. A record cut short by a killed writer is ignored with a warning,
  and its block has to be written again.
- Which of two processes wrote a node "last" is decided by the names of
  their segment files, not by time. Within one process it is the order of
  the writes.
- A read that returns an extra attribute raises `UnfinalizedWrites` while
  its directory has segments. A read that names other attributes does not,
  so a worker reads the graph (`node_attrs=[...]` without the attribute it
  is writing) while the run is under way.
- Finalize holds the node ids and the attributes it rebuilds in memory.
- A handle sees the arrays of the finalize it last loaded; open the graph
  again to see what another process finalized since.
