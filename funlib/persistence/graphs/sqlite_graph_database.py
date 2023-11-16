from .graph_database import GraphDataBase

from funlib.geometry import Coordinate
from funlib.geometry import Roi

from networkx import Graph, DiGraph

import logging
import sqlite3
import json
from pathlib import Path
from typing import Optional, Any, Iterable

logger = logging.getLogger(__name__)


class SQLiteGraphDataBase(GraphDataBase):
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
        self.endpoint_names = ["u", "v"] if endpoint_names is None else endpoint_names
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
            # TODO: Do we need these?
            # self.cur.execute("""PRAGMA synchronous = OFF""")
            # self.cur.execute("""PRAGMA journal_mode = OFF""")

        self.create_tables()

        if self.meta_collection.exists():
            self.__check_metadata()
        else:
            self.__set_metadata()

    def read_graph(
        self,
        roi: Optional[Roi] = None,
        read_edges: bool = True,
        node_attrs: Optional[list[str]] = None,
        edge_attrs: Optional[list[str]] = None,
        nodes_filter: Optional[dict[str, Any]] = None,
        edges_filter: Optional[dict[str, Any]] = None,
    ) -> Graph:
        if self.directed:
            graph = DiGraph()
        else:
            graph = Graph()

        nodes = self.read_nodes(
            roi,
            read_attrs=node_attrs,
            attr_filter=nodes_filter,
        )
        node_list = [(n["id"], self.__remove_keys(n, ["id"])) for n in nodes]
        graph.add_nodes_from(node_list)

        if read_edges:
            edges = self.read_edges(
                roi, nodes=nodes, read_attrs=edge_attrs, attr_filter=edges_filter
            )
            u, v = self.endpoint_names
            try:
                edge_list = [(e[u], e[v], self.__remove_keys(e, [u, v])) for e in edges]
            except KeyError as e:
                raise ValueError(edges[:5]) from e
            graph.add_edges_from(edge_list)
        return graph

    def write_attrs(
        self,
        graph: Graph,
        roi: Optional[Roi] = None,
        node_attrs: Optional[list[str]] = None,
        edge_attrs: Optional[list[str]] = None,
    ) -> None:
        self.update_nodes(
            nodes=graph.nodes(data=True),
            roi=roi,
            attributes=node_attrs,
        )
        self.update_edges(
            nodes=graph.nodes(data=True),
            edges=graph.edges(data=True),
            roi=roi,
            attributes=edge_attrs,
        )

    def write_graph(
        self,
        graph: Graph,
        roi: Optional[Roi] = None,
        write_nodes: bool = True,
        write_edges: bool = True,
        node_attrs: Optional[list[str]] = None,
        edge_attrs: Optional[list[str]] = None,
        fail_if_exists: bool = False,
        delete: bool = False,
    ) -> None:
        if write_nodes:
            self.write_nodes(
                graph.nodes,
                roi,
                attributes=node_attrs,
                fail_if_exists=fail_if_exists,
                delete=delete,
            )
        if write_edges:
            self.write_edges(
                graph.nodes,
                graph.edges,
                roi,
                attributes=edge_attrs,
                fail_if_exists=fail_if_exists,
                delete=delete,
            )

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
        edge_columns = [
            f"{self.endpoint_names[0]} INTEGER not null",
            f"{self.endpoint_names[1]} INTEGER not null",
        ] + [f"{edge_attr}" for edge_attr in self.edge_attrs]
        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS {self.edges_collection_name}("
            + f"{', '.join(edge_columns)}"
            + f", PRIMARY KEY ({self.endpoint_names[0]}, {self.endpoint_names[1]})"
            + ")"
        )

    def attr_query(self, attrs: dict[str, Any]) -> str:
        query = ""
        for attr, value in attrs.items():
            query += f"{attr}={self.__convert_to_sql(value)} AND "
        query = query[:-5]
        return query

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
                query += f"{pos_attr}<{roi.end[dim]}"
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
        select_statement = (
            f"SELECT * FROM {self.nodes_collection_name} "
            + (self.roi_query(roi) if roi is not None else "")
            + (
                f" {'WHERE' if roi is None else 'AND'} " + self.attr_query(attr_filter)
                if attr_filter is not None and len(attr_filter) > 0
                else ""
            )
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
        node_condition = f"{self.endpoint_names[0]} IN ({node_ids})"

        logger.debug("Reading nodes in roi %s" % roi)
        # TODO: AND vs OR here
        desired_columns = ", ".join(self.endpoint_names + self.edge_attrs)
        select_statement = (
            f"SELECT {desired_columns} FROM {self.edges_collection_name} WHERE "
            + node_condition
            + (
                f" AND " + self.attr_query(attr_filter)
                if attr_filter is not None and len(attr_filter) > 0
                else ""
            )
        )

        edge_attrs = self.endpoint_names + (
            self.edge_attrs if read_attrs is None else read_attrs
        )
        attr_filter = attr_filter if attr_filter is not None else {}
        for k, v in attr_filter.items():
            select_statement += f" AND {k}={self.__convert_to_sql(v)}"

        try:
            edges = [
                {
                    key: val
                    for key, val in zip(self.endpoint_names + self.edge_attrs, values)
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
        delete=False,
    ):
        if delete:
            raise NotImplementedError("Delete not implemented for file backend")
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")

        columns = self.endpoint_names + (
            self.edge_attrs if attributes is None else attributes
        )
        insert_statement = (
            f"INSERT{' OR IGNORE' if not fail_if_exists else ''} INTO {self.edges_collection_name} "
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

            if not roi.contains(pos_u):
                logger.debug(
                    (
                        f"Skipping edge with {self.endpoint_names[0]} {{}}, {self.endpoint_names[1]} {{}},"
                        + f"and data {{}} because {self.endpoint_names[0]} not in roi {{}}"
                    ).format(u, v, data, roi)
                )
                continue

            edge_attributes = [u, v] + [
                data.get(attr, None)
                for attr in (self.edge_attrs if attributes is None else attributes)
            ]
            to_insert.append(edge_attributes)

        if len(to_insert) == 0:
            logger.debug("No edges to insert in %s", roi)
            return

        self.cur.executemany(insert_statement, to_insert)
        self.con.commit()

    def update_edges(
        self,
        nodes: dict[Any, dict[str, Any]],
        edges: dict[Any, dict[str, Any]],
        roi=None,
        attributes=None,
    ):
        if self.mode == "r":
            raise NotImplementedError("Trying to write to read-only DB")

        logger.debug("Writing nodes in %s", roi)

        attrs = attributes if attributes is not None else []

        for u, v, data in edges:
            if not self.directed:
                u, v = min(u, v), max(u, v)
            if roi is not None:
                pos_u = self.__get_node_pos(nodes[u])

                if not roi.contains(pos_u):
                    logger.debug(
                        (
                            f"Skipping edge with {self.endpoint_names[0]} {{}}, {self.endpoint_names[1]} {{}},"
                            + f"and data {{}} because {self.endpoint_names[0]} not in roi {{}}"
                        ).format(u, v, data, roi)
                    )
                    continue

            values = [data.get(attr) for attr in attrs]
            setters = [f"{k}={v}" for k, v in zip(attrs, values)]
            insert_statement = (
                f"UPDATE {self.edges_collection_name} SET "
                f"{', '.join(setters)} WHERE "
                f"{self.endpoint_names[0]}={u} AND {self.endpoint_names[1]}={v}"
            )

            try:
                self.cur.execute(insert_statement)
            except sqlite3.OperationalError as e:
                raise ValueError(insert_statement) from e

        self.con.commit()

    def write_nodes(
        self,
        nodes: dict[Any, dict[str, Any]],
        roi=None,
        attributes=None,
        fail_if_exists=False,
        delete=False,
    ):
        if delete:
            raise NotImplementedError("Delete not implemented for file backend")
        if fail_if_exists:
            raise NotImplementedError(
                "Fail if exists not implemented for " "file backend"
            )
        if attributes is not None:
            raise NotImplementedError("Attributes not implemented for file backend")
        if self.mode == "r":
            raise NotImplementedError("Trying to write to read-only DB")

        logger.debug("Writing nodes in %s", roi)

        insert_statement = (
            f"INSERT{' OR IGNORE' if not fail_if_exists else ''} INTO {self.nodes_collection_name} "
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

    def update_nodes(
        self,
        nodes: dict[Any, dict[str, Any]],
        roi=None,
        attributes=None,
    ):
        if self.mode == "r":
            raise NotImplementedError("Trying to write to read-only DB")

        logger.debug("Writing nodes in %s", roi)

        attrs = attributes if attributes is not None else []

        for node, data in nodes:
            if roi is not None:
                pos_u = self.__get_node_pos(data)

                if not roi.contains(pos_u):
                    logger.debug(
                        ("Skipping node {} because it is not in roi {}").format(
                            node, roi
                        )
                    )
                    continue

            values = [data.get(attr) for attr in attrs]
            setters = [
                f"{k} = {self.__convert_to_sql(v)}" for k, v in zip(attrs, values)
            ]
            insert_statement = (
                f"UPDATE {self.nodes_collection_name} SET "
                f"{', '.join(setters)} WHERE "
                f"id={node}"
            )

            try:
                self.cur.execute(insert_statement)
            except sqlite3.OperationalError as e:
                raise ValueError(insert_statement) from e

        self.con.commit()

    def __remove_keys(self, dictionary, keys):
        """Removes given keys from dictionary."""

        return {k: v for k, v in dictionary.items() if k not in keys}

    def __get_node_pos(self, n: dict[str, Any]) -> Coordinate:
        if isinstance(self.position_attribute, str):
            return Coordinate(n.get(self.position_attribute, (None,) * self.ndim))
        elif isinstance(self.position_attribute, Iterable):
            return Coordinate(
                (n.get(pos_attr, None) for pos_attr in self.position_attributes)
            )
