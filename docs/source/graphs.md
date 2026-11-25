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
edge attributes that make an edge a step.

`read_block` is the read of a computation that writes one value per node of
a block: the nodes of the ROI, marked `own`, with the nodes within a
distance `context` of it and within `num_steps` edges of those.

```python
nodes, edges = db.read_block(roi, context=500, num_steps=2)
values = compute(nodes, edges)[nodes["own"]]
```

`CSRGraphDataBase.is_database(path)` tells such a directory from any other,
and `db.digest` identifies the content of the database.

## Values on the nodes of a graph

Values computed on a graph, such as an embedding per node, can be kept
beside the graph without rewriting it. A node attribute store is a
directory of memory-mapped arrays keyed by node id, and records the digest
of the graph the values belong to:

```python
from funlib.persistence.graphs import NodeArrays, NodeArraysWriter

with NodeArraysWriter(
    "embedding.nodes", {"embedding": 64}, graph_digest=db.digest
) as out:
    for roi in blocks:
        nodes, edges = db.read_block(roi, num_steps=2)
        own = nodes["own"]
        out.append(nodes["id"][own], embedding=model(nodes, edges)[own])

found, values = NodeArrays("embedding.nodes").read(nodes["id"])
```

A node belongs to one block: a node id written twice is refused. A node
that was never written is not `found` and reads as NaN (0 for an integer).
`write_node_arrays` writes values that are already in memory in one call.
