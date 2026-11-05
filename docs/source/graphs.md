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
