from .shared_graph_provider import SharedGraphProvider, SharedSubGraph

from funlib.geometry import Coordinate
from funlib.geometry import Roi

from networkx import Graph, DiGraph
import numpy as np
import networkx as nx

import logging
import sqlite3
import json
from pathlib import Path
from typing import Optional, Any, Iterable

logger = logging.getLogger(__name__)


class SQLiteGraphProvider(SharedGraphProvider):
    """
    Provides shared graphs stored in a SQLite Database.

    Nodes are assumed to have at least an attribute ``id``. If the have a
    position attribute (set via argument ``position_attribute``, defaults to
    ``position``), it will be used for geometric slicing (see ``__getitem__``).

    Edges are assumed to have at least attributes ``u``, ``v``.

    Arguments:

        db_file (``Path``):

            The file to store your database in.

        mode (``string``, optional):

            One of ``r+`` or ``w``. Defaults to ``r+``. ``w`` drops the
            node, edge, and meta collections.

        directed (``bool``):

            True if the graph is directed, false otherwise. If None, attempts
            to read value from existing database. If not found, defaults to
            false.

        nodes_collection (``string``):

            The name of the nodes table. Defaults to ``nodes``.
        
        edges_collection (``string``):

            The name of the edges table. Defaults to ``edges``.

        endpoint_names (``list`` or ``tuple`` with two elements):

            What keys to use for the start and end of an edge. Default is
            ['u', 'v']

        position_attribute (``string`` or list of ``string``s, optional):

            The node attribute(s) that contain position information. This will
            be used for slicing subgraphs via ``__getitem__``. If a single
            string, the attribute is assumed to be an array. If a list, each
            entry denotes the position coordinates in order (e.g.,
            `position_z`, `position_y`, `position_x`).

        node_attrs (``list`` of ``str`` or None):

            The custom attributes to store on each node.

        edge_attrs (``list`` of ``str`` or None):

            The custom attributes to store on each edge.

        ndim (``int``):

            The number of spatial dimensions for your node positions.

    """

    __node_attrs: Optional[Iterable[str]] = None
    __edge_attrs: Optional[Iterable[str]] = None

    def __init__(
        self,
        db_file: Path,
        mode: str = "r+",
        directed: Optional[bool] = None,
        total_roi: Optional[Roi] = None,
        nodes_collection: str = "nodes",
        edges_collection: str = "edges",
        endpoint_names: Optional[tuple[str, str]] = None,
        position_attribute: str = "position",
        node_attrs: Optional[list[str]] = None,
        edge_attrs: Optional[list[str]] = None,
        ndim: int = 3,
    ):
        self.db_file = db_file
        self.con = sqlite3.connect(db_file)
        self.cur = self.con.cursor()
        self.mode = mode
        self.directed = directed
        self.total_roi = total_roi
        self.nodes_collection_name = nodes_collection
        self.edges_collection_name = edges_collection
        self.endpoint_names = ("u", "v") if endpoint_names is None else endpoint_names
        self.meta_collection = self.db_file.parent / f"{self.db_file.stem}-meta.json"
        self.position_attribute = position_attribute
        self.ndim = ndim

        if isinstance(self.position_attribute, str):
            self.position_attributes = [
                self.position_attribute + f"_{i}" for i in range(ndim)
            ]
        elif isinstance(self.position_attribute, Iterable):
            self.position_attributes = list(self.position_attribute)
        else:
            raise ValueError(self.position_attribute)
        assert len(self.position_attributes) == ndim
        self.__node_attrs = node_attrs
        self.__edge_attrs = edge_attrs

        if mode == "w":
            self.drop_tables()
            self.cur.execute("""PRAGMA synchronous = OFF""")
            self.cur.execute("""PRAGMA journal_mode = OFF""")

        self.create_tables()

        if self.meta_collection.exists():
            self.__check_metadata()
        else:
            self.__set_metadata()

    @property
    def node_attrs(self) -> list[str]:
        return list(self.__node_attrs) if self.__node_attrs is not None else []

    @node_attrs.setter
    def node_attrs(self, value: Optional[Iterable[str]]) -> None:
        self.__node_attrs = value

    @property
    def edge_attrs(self) -> list[str]:
        return list(self.__edge_attrs) if self.__edge_attrs is not None else []

    @edge_attrs.setter
    def edge_attrs(self, value: Optional[Iterable[str]]) -> None:
        self.__edge_attrs = value

    def drop_tables(self) -> None:
        logger.info(
            "dropping collections %s, %s",
            self.nodes_collection_name,
            self.edges_collection_name,
        )

        self.cur.execute(f"DROP TABLE IF EXISTS {self.nodes_collection_name}")
        self.cur.execute(f"DROP TABLE IF EXISTS {self.edges_collection_name}")
        self.cur.execute("DROP TABLE IF EXISTS edge_index")

        if self.meta_collection.exists():
            self.meta_collection.unlink()

    def create_tables(self) -> None:
        position_template = "{pos_attr} REAL not null"
        columns = [
            position_template.format(pos_attr=pos_attr)
            for pos_attr in self.position_attributes
        ] + self.node_attrs
        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS "
            f"{self.nodes_collection_name}("
            "id INTEGER not null PRIMARY KEY, "
            f"{', '.join(columns)}"
            ")"
        )
        self.cur.execute(
            f"CREATE INDEX IF NOT EXISTS pos_index ON {self.nodes_collection_name}({','.join(self.position_attributes)})"
        )
        edge_columns = ["u INTEGER not null", "v INTEGER not null"] + [
            f"{edge_attr}" for edge_attr in self.edge_attrs
        ]
        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS {self.edges_collection_name}(id INTEGER not null PRIMARY KEY, "
            + f"{', '.join(edge_columns)}"
            + ")"
        )

    def roi_query(self, roi: Roi) -> str:
        query = "WHERE "
        for dim, pos_attr in enumerate(self.position_attributes):
            if dim > 0:
                query += " AND "
            if roi.begin[dim] is not None and roi.end[dim] is not None:
                query += f"{pos_attr} BETWEEN {roi.begin[dim]} and {roi.end[dim]}"
            elif roi.begin[dim] is not None:
                query += f"{pos_attr}>={roi.begin[dim]}"
            elif roi.begin[dim] is not None:
                query += f"{pos_attr}<={roi.end[dim]}"
            else:
                query = query[:-5]
        return query

    def rtree_query(self, roi: Roi) -> str:
        query = ""
        for dim, pos_attr in enumerate(self.position_attributes):
            if dim > 0:
                query += " AND "
            if roi.begin[dim] is not None and roi.end[dim] is not None:
                query += f"min{pos_attr.upper()}<={roi.end[dim]} AND max{pos_attr.upper()}>={roi.begin[dim]}"
            elif roi.begin[dim] is not None:
                query += f"max{pos_attr.upper()} >= {roi.begin[dim]}"
            elif roi.end[dim] is not None:
                query += f"min{pos_attr.upper()} <= {roi.end[dim]}"
            else:
                query = query[:-5]
        return query

    def read_nodes(
        self,
        roi: Optional[Roi] = None,
        attr_filter: Optional[dict[str, Any]] = None,
        read_attrs: Optional[list[str]] = None,
        join_collection: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return a list of nodes within roi."""

        logger.debug("Reading nodes in roi %s" % roi)
        select_statement = f"SELECT * FROM {self.nodes_collection_name} " + (
            self.roi_query(roi) if roi is not None else ""
        )

        read_attrs = (
            ["id"]
            + self.position_attributes
            + (self.node_attrs if read_attrs is None else read_attrs)
        )
        attr_filter = attr_filter if attr_filter is not None else {}
        for k, v in attr_filter.items():
            select_statement += f" AND {k}={self.__convert_to_sql(v)}"

        try:
            nodes = [
                {
                    key: val
                    for key, val in zip(
                        ["id"] + self.position_attributes + self.node_attrs, values
                    )
                    if key in read_attrs
                }
                for values in self.cur.execute(select_statement)
            ]
        except sqlite3.OperationalError as e:
            raise ValueError(select_statement) from e

        if isinstance(self.position_attribute, str):
            for data in nodes:
                data[self.position_attribute] = self.__get_node_pos(data)

        return nodes

    def num_nodes(self, roi: Roi) -> int:
        """Return the number of nodes in the roi."""

        # TODO: can be made more efficient
        return len(self.read_nodes(roi))

    def has_edges(self, roi: Roi) -> bool:
        """Returns true if there is at least one edge in the roi."""

        # TODO: can be made more efficient
        return len(self.read_edges(roi)) > 0

    def read_edges(
        self,
        roi: Optional[Roi] = None,
        nodes: Optional[list[dict[str, Any]]] = None,
        attr_filter: Optional[dict[str, Any]] = None,
        read_attrs: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Returns a list of edges within roi."""

        if nodes is None:
            nodes = self.read_nodes(roi)

        if len(nodes) == 0:
            return []

        node_ids = ", ".join([str(node["id"]) for node in nodes])
        node_condition = f"u IN ({node_ids})"
        if roi is not None:
            rtree_condition = self.rtree_query(roi)
        else:
            rtree_condition = ""

        logger.debug("Reading nodes in roi %s" % roi)
        # TODO: AND vs OR here
        desired_columns = ", ".join(["u", "v"] + self.edge_attrs)
        select_statement = (
            f"SELECT {desired_columns} FROM {self.edges_collection_name} WHERE "
            + node_condition
            + ((" AND " + rtree_condition) if len(rtree_condition) > 0 else "")
        )

        edge_attrs = ["u", "v"] + (
            self.edge_attrs if read_attrs is None else read_attrs
        )
        attr_filter = attr_filter if attr_filter is not None else {}
        for k, v in attr_filter.items():
            select_statement += f" AND {k}={self.__convert_to_sql(v)}"

        try:
            edges = [
                {
                    key: val
                    for key, val in zip(["u", "v"] + self.edge_attrs, values)
                    if key in edge_attrs
                }
                for values in self.cur.execute(select_statement)
            ]
        except sqlite3.OperationalError:
            raise ValueError(select_statement)

        return edges

    def __convert_to_sql(self, x: Any) -> str:
        if isinstance(x, str):
            return f"'{x}'"
        elif x is None:
            return "null"
        elif isinstance(x, bool):
            return f"{x}".lower()
        else:
            return str(x)

    def __get_metadata(self) -> dict[str, Any]:
        """Gets metadata out of the meta collection and returns it
        as a dictionary."""

        with open(self.meta_collection, "r") as f:
            return json.load(f)

    def __check_metadata(self):
        """Checks if the provided metadata matches the existing
        metadata in the meta collection"""

        metadata = self.__get_metadata()
        if self.directed is not None and metadata["directed"] != self.directed:
            raise ValueError(
                (
                    "Input parameter directed={} does not match"
                    "directed value {} already in stored metadata"
                ).format(self.directed, metadata["directed"])
            )
        elif self.directed is None:
            self.directed = metadata["directed"]
        if self.total_roi is not None:
            if self.total_roi.get_offset() != metadata["total_roi_offset"]:
                raise ValueError(
                    (
                        "Input total_roi offset {} does not match"
                        "total_roi offset {} already stored in metadata"
                    ).format(self.total_roi.get_offset(), metadata["total_roi_offset"])
                )
            if self.total_roi.get_shape() != metadata["total_roi_shape"]:
                raise ValueError(
                    (
                        "Input total_roi shape {} does not match"
                        "total_roi shape {} already stored in metadata"
                    ).format(self.total_roi.get_shape(), metadata["total_roi_shape"])
                )
        else:
            self.total_roi = Roi(
                metadata["total_roi_offset"], metadata["total_roi_shape"]
            )
        if self.__node_attrs is not None:
            assert self.node_attrs == metadata["node_attrs"], (
                self.node_attrs,
                metadata["node_attrs"],
            )
        else:
            self.node_attrs = metadata["node_attrs"]
        if self.__edge_attrs is not None:
            assert self.edge_attrs == metadata["edge_attrs"]
        else:
            self.edge_attrs = metadata["edge_attrs"]

    def __set_metadata(self):
        """Sets the metadata in the meta collection to the provided values"""

        if not self.directed:
            # default is False
            self.directed = self.directed if self.directed is not None else False
        if not self.total_roi:
            # default is an unbounded roi
            self.total_roi = Roi(
                (None,) * len(self.position_attributes),
                (None,) * len(self.position_attributes),
            )

        meta_data = {
            "directed": self.directed,
            "total_roi_offset": self.total_roi.offset,
            "total_roi_shape": self.total_roi.shape,
            "node_attrs": self.node_attrs,
            "edge_attrs": self.edge_attrs,
        }

        with open(self.meta_collection, "w") as f:
            json.dump(meta_data, f)

    def write_edges(
        self,
        nodes,
        edges,
        roi=None,
        attributes=None,
        fail_if_exists=False,
        fail_if_not_exists=False,
        delete=False,
    ):
        assert not (
            fail_if_exists and fail_if_not_exists
        ), "Cannot have fail_if_exists and fail_if_not_exists simultaneously"
        if delete:
            raise NotImplementedError("Delete not implemented for file backend")
        if fail_if_exists:
            raise NotImplementedError("Fail if exists not implemented for file backend")
        if fail_if_not_exists:
            raise NotImplementedError(
                "Fail if not exists not implemented for file backend"
            )
        if attributes is not None:
            raise NotImplementedError("Attributes not implemented for file backend")
        if self.mode == "r":
            raise NotImplementedError("Trying to write to read-only DB")

        columns = (
            [
                f"{prefix}{pos_attr.upper()}"
                for prefix in ["min", "max"]
                for pos_attr in self.position_attributes
            ]
            + ["u", "v"]
            + self.edge_attrs
        )
        insert_statement = (
            f"INSERT INTO {self.edges_collection_name} "
            f"({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})"
        )
        to_insert = []

        if roi is None:
            roi = Roi(
                (None,) * len(self.position_attributes),
                (None,) * len(self.position_attributes),
            )

        for (u, v), data in edges.items():
            if not self.directed:
                u, v = min(u, v), max(u, v)

            pos_u = self.__get_node_pos(nodes[u])
            pos_v = self.__get_node_pos(nodes[v])

            if not roi.contains(pos_u):
                logger.debug(
                    (
                        "Skipping edge with u {}, v {},"
                        + "and data {} because u not in roi {}"
                    ).format(u, v, data, roi)
                )
                continue

            if any([c is None for c in pos_u]) and any([c is None for c in pos_v]):
                raise ValueError(
                    f"Neither end point ({nodes[u], nodes[v]}) has a location"
                )
            elif any([c is None for c in pos_u]):
                pos_u = pos_v
            elif any([c is None for c in pos_v]):
                pos_v = pos_u
            edge_attributes = (
                [
                    func(pos_u[dim], pos_v[dim])
                    for func in [min, max]
                    for dim in range(len(pos_u))
                ]
                + [u, v]
                + [data.get(attr, None) for attr in self.edge_attrs]
            )
            to_insert.append(edge_attributes)

        if len(to_insert) == 0:
            logger.debug("No edges to insert in %s", roi)
            return

        self.cur.executemany(insert_statement, to_insert)
        self.con.commit()

    def write_nodes(
        self,
        nodes: dict[Any, dict[str, Any]],
        roi=None,
        attributes=None,
        fail_if_exists=False,
        fail_if_not_exists=False,
        delete=False,
    ):
        assert not (
            fail_if_exists and fail_if_not_exists
        ), "Cannot have fail_if_exists and fail_if_not_exists simultaneously"
        if delete:
            raise NotImplementedError("Delete not implemented for file backend")
        if fail_if_exists:
            raise NotImplementedError(
                "Fail if exists not implemented for " "file backend"
            )
        if fail_if_not_exists:
            raise NotImplementedError(
                "Fail if not exists not implemented for " "file backend"
            )
        if attributes is not None:
            raise NotImplementedError("Attributes not implemented for file backend")
        if self.mode == "r":
            raise NotImplementedError("Trying to write to read-only DB")

        logger.debug("Writing nodes in %s", roi)

        insert_statement = (
            f"INSERT OR IGNORE INTO {self.nodes_collection_name} "
            f"(id, {', '.join(self.position_attributes + self.node_attrs)}) VALUES "
            f"({', '.join(['?'] * (len(self.position_attributes) + len(self.node_attrs) + 1))})"
        )

        to_insert = []
        for node_id, data in nodes.items():
            pos = self.__get_node_pos(data)
            if roi is not None and not roi.contains(pos):
                continue
            for i, position_attribute in enumerate(self.position_attributes):
                data[position_attribute] = pos[i]
            to_insert.append(
                [node_id]
                + [
                    data.get(attr, None)
                    for attr in self.position_attributes + self.node_attrs
                ]
            )

        if len(to_insert) == 0:
            logger.debug("No nodes to insert in %s", roi)
            return

        try:
            self.cur.executemany(insert_statement, to_insert)
        except sqlite3.OperationalError as e:
            raise ValueError(insert_statement, to_insert) from e
        self.con.commit()

    def __getitem__(self, roi):
        return self.get_graph(roi)

    def get_graph(
        self,
        roi: Roi,
        nodes_filter: Optional[dict[str, Any]] = None,
        edges_filter: Optional[dict[str, Any]] = None,
        node_attrs: Optional[list[str]] = None,
        edge_attrs: Optional[list[str]] = None,
        join_collection: Optional[str] = None,
    ):
        """Return a graph within roi, optionally filtering by
        node and edge attributes.

        Arguments:

            roi:

                Get nodes and edges whose source is within this roi

            nodes_filter:
            edges_filter:

                Only return nodes/edges that have attribute=value for
                each attribute value pair in nodes/edges_filter.

            node_attrs:

                Only return these attributes for nodes. Other
                attributes will be ignored, but id and position attribute(s)
                will always be included. If None (default), return all attrs.

            edge_attrs:

                Only return these attributes for edges. Other
                attributes will be ignored, but source and target
                will always be included. If None (default), return all attrs.

            join_collection:

                Compute (left) join of the nodes collection and this
                collection using the id attribute.
                See read_nodes() for more information.

        """
        nodes = self.read_nodes(
            roi,
            attr_filter=nodes_filter,
            read_attrs=node_attrs,
            join_collection=join_collection,
        )
        edges = self.read_edges(
            roi, nodes=nodes, attr_filter=edges_filter, read_attrs=edge_attrs
        )
        u, v = self.endpoint_names
        node_list = [(n["id"], self.__remove_keys(n, ["id"])) for n in nodes]
        try:
            edge_list = [(e[u], e[v], self.__remove_keys(e, [u, v])) for e in edges]
        except KeyError as e:
            raise ValueError(edges[:5]) from e
        if self.directed:
            graph = SQLiteSubDiGraph(self, roi)
        else:
            # create the subgraph
            graph = SQLiteSubGraph(self, roi)
        graph.add_nodes_from(node_list)
        graph.add_edges_from(edge_list)
        return graph

    def __remove_keys(self, dictionary, keys):
        """Removes given keys from dictionary."""

        for key in keys:
            del dictionary[key]
        return dictionary

    def __get_node_pos(self, n: dict[str, Any]) -> Coordinate:
        if isinstance(self.position_attribute, str):
            return Coordinate(n.get(self.position_attribute, (None,) * self.ndim))
        elif isinstance(self.position_attribute, Iterable):
            return Coordinate(
                (n.get(pos_attr, None) for pos_attr in self.position_attributes)
            )

    def __create_node_collection(self):
        # TODO Create index on nodes
        """Creates the node collection, including indexes"""
        self.__open_db()
        self.__open_collections()

        if type(self.position_attribute) == list:
            self.nodes.create_index(
                [(key, ASCENDING) for key in self.position_attribute], name="position"
            )
        else:
            self.nodes.create_index([("position", ASCENDING)], name="position")

        self.nodes.create_index([("id", ASCENDING)], name="id", unique=True)


class SQLiteSharedSubGraph(SharedSubGraph):
    def __init__(self, graph_provider: SQLiteGraphProvider, roi: Roi):
        super().__init__()

        self.provider = graph_provider
        self.roi = roi

    def write_nodes(
        self,
        roi=None,
        attributes=None,
        fail_if_exists=False,
        fail_if_not_exists=False,
        delete=False,
    ):
        self.provider.write_nodes(
            self.nodes, roi, attributes, fail_if_exists, fail_if_not_exists, delete
        )

    def write_edges(
        self,
        roi=None,
        attributes=None,
        fail_if_exists=False,
        fail_if_not_exists=False,
        delete=False,
    ):
        self.provider.write_edges(
            self.nodes,
            self.edges,
            roi,
            attributes,
            fail_if_exists,
            fail_if_not_exists,
            delete,
        )

    def update_edge_attrs(self, roi=None, attributes=None):
        self.provider.write_edges(self.nodes, self.edges, roi, attributes)

    def update_node_attrs(self, roi=None, attributes=None):
        self.provider.write_nodes(self.nodes, roi, attributes)


class SQLiteSubGraph(SQLiteSharedSubGraph, Graph):
    def __init__(self, graph_provider, roi):
        # this calls the init function of the SQLite
        # SharedSubGraph,
        # because left parents come before right parents
        super().__init__(graph_provider, roi)

    def is_directed(self):
        return False


class SQLiteSubDiGraph(SQLiteSharedSubGraph, DiGraph):
    def __init__(self, graph_provider, roi):
        # this calls the init function of the SQLite
        # SharedSubGraph,
        # because left parents come before right parents
        super().__init__(graph_provider, roi)

    def is_directed(self):
        return True
