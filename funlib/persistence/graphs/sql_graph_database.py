from .graph_database import GraphDataBase

from funlib.geometry import Coordinate
from funlib.geometry import Roi

from networkx import Graph, DiGraph

import logging
import json
from typing import Optional, Any, Iterable
from abc import abstractmethod

logger = logging.getLogger(__name__)


class SQLGraphDataBase(GraphDataBase):
    """Base class for SQL-based graph databases.

    Nodes must have position attributes (set via argument
    ``position_attributes``), which will be used for geometric slicing (see
    ``__getitem__`` and ``read_graph``).

    Arguments:

        position_attributes (list of ``string``s):

            The node attributes that contain position information. This will
            be used for slicing subgraphs via ``__getitem__``.

        directed (``bool``):

            True if the graph is directed, false otherwise. If None, attempts
            to read value from existing database. If not found, defaults to
            false.

        nodes_table (``string``):

            The name of the nodes table. Defaults to ``nodes``.

        edges_table (``string``):

            The name of the edges table. Defaults to ``edges``.

        endpoint_names (``list`` or ``tuple`` with two elements):

            What names to use for the columns storing the start and end of an
            edge. Default is ['u', 'v'].

        node_attrs (``list`` of ``str`` or None):

            The custom attributes to store on each node.

        edge_attrs (``list`` of ``str`` or None):

            The custom attributes to store on each edge.
    """

    _node_attrs: Optional[dict[str, type]] = None
    _edge_attrs: Optional[dict[str, type]] = None

    def __init__(
        self,
        position_attributes: list[str],
        mode: str = "r+",
        directed: Optional[bool] = None,
        total_roi: Optional[Roi] = None,
        nodes_table: str = "nodes",
        edges_table: str = "edges",
        endpoint_names: Optional[list[str]] = None,
        node_attrs: Optional[dict[str, type]] = None,
        edge_attrs: Optional[dict[str, type]] = None,
    ):
        self.position_attributes = position_attributes
        self.ndim = len(self.position_attributes)
        self.mode = mode
        self.directed = directed
        self.total_roi = total_roi
        self.nodes_table_name = nodes_table
        self.edges_table_name = edges_table
        self.endpoint_names = ["u", "v"] if endpoint_names is None else endpoint_names

        self._node_attrs = node_attrs
        self._edge_attrs = edge_attrs

        if mode == "w":
            self._drop_tables()

        self._create_tables()
        self.__init_metadata()

    @abstractmethod
    def _drop_tables(self) -> None:
        pass

    @abstractmethod
    def _create_tables(self) -> None:
        pass

    @abstractmethod
    def _store_metadata(self, metadata) -> None:
        pass

    @abstractmethod
    def _read_metadata(self) -> Optional[dict[str, Any]]:
        pass

    @abstractmethod
    def _select_query(self, query) -> Iterable[Any]:
        pass

    @abstractmethod
    def _insert_query(
        self, table, columns, values, fail_if_exists=False, commit=True
    ) -> None:
        pass

    @abstractmethod
    def _update_query(self, query, commit=True) -> None:
        pass

    @abstractmethod
    def _commit(self) -> None:
        pass

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
    def node_attrs(self) -> dict[str, type]:
        return self._node_attrs if self._node_attrs is not None else {}

    @node_attrs.setter
    def node_attrs(self, value: dict[str, type]) -> None:
        self._node_attrs = value

    @property
    def edge_attrs(self) -> dict[str, type]:
        return self._edge_attrs if self._edge_attrs is not None else {}

    @edge_attrs.setter
    def edge_attrs(self, value: dict[str, type]) -> None:
        self._edge_attrs = value

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
            f"SELECT * FROM {self.nodes_table_name} "
            + (self.__roi_query(roi) if roi is not None else "")
            + (
                f" {'WHERE' if roi is None else 'AND'} "
                + self.__attr_query(attr_filter)
                if attr_filter is not None and len(attr_filter) > 0
                else ""
            )
        )

        read_attrs = (
            ["id"]
            + self.position_attributes
            + (list(self.node_attrs.keys()) if read_attrs is None else read_attrs)
        )
        attr_filter = attr_filter if attr_filter is not None else {}
        for k, v in attr_filter.items():
            select_statement += f" AND {k}={self.__convert_to_sql(v)}"

        nodes = [
            {
                key: val
                for key, val in zip(
                    ["id"] + self.position_attributes + list(self.node_attrs.keys()),
                    values,
                )
                if key in read_attrs and val is not None
            }
            for values in self._select_query(select_statement)
        ]

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
        desired_columns = ", ".join(self.endpoint_names + list(self.edge_attrs.keys()))
        select_statement = (
            f"SELECT {desired_columns} FROM {self.edges_table_name} WHERE "
            + node_condition
            + (
                f" AND " + self.__attr_query(attr_filter)
                if attr_filter is not None and len(attr_filter) > 0
                else ""
            )
        )

        edge_attrs = self.endpoint_names + (
            list(self.edge_attrs.keys()) if read_attrs is None else read_attrs
        )
        attr_filter = attr_filter if attr_filter is not None else {}
        for k, v in attr_filter.items():
            select_statement += f" AND {k}={self.__convert_to_sql(v)}"

        edges = [
            {
                key: val
                for key, val in zip(
                    self.endpoint_names + list(self.edge_attrs.keys()), values
                )
                if key in edge_attrs
            }
            for values in self._select_query(select_statement)
        ]

        return edges

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
            raise NotImplementedError("Delete not implemented for SQL graph database")
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")

        columns = self.endpoint_names + (
            list(self.edge_attrs.keys()) if attributes is None else attributes
        )

        if roi is None:
            roi = Roi(
                (None,) * len(self.position_attributes),
                (None,) * len(self.position_attributes),
            )

        values = []
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
            values.append(edge_attributes)

        if len(values) == 0:
            logger.debug("No edges to insert in %s", roi)
            return

        self._insert_query(
            self.edges_table_name, columns, values, fail_if_exists=fail_if_exists
        )

    def update_edges(
        self,
        nodes: dict[Any, dict[str, Any]],
        edges: dict[Any, dict[str, Any]],
        roi=None,
        attributes=None,
    ):
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")

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
            update_statement = (
                f"UPDATE {self.edges_table_name} SET "
                f"{', '.join(setters)} WHERE "
                f"{self.endpoint_names[0]}={u} AND {self.endpoint_names[1]}={v}"
            )

            self._update_query(update_statement, commit=False)

        self._commit()

    def write_nodes(
        self,
        nodes: dict[Any, dict[str, Any]],
        roi=None,
        attributes=None,
        fail_if_exists=False,
        delete=False,
    ):
        if delete:
            raise NotImplementedError("Delete not implemented for SQL graph database")
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")

        logger.debug("Writing nodes in %s", roi)

        attrs = attributes if attributes is not None else list(self.node_attrs.keys())
        columns = ("id",) + tuple(self.position_attributes) + tuple(attrs)

        values = []
        for node_id, data in nodes.items():
            data = data.copy()
            pos = self.__get_node_pos(data)
            if roi is not None and not roi.contains(pos):
                continue
            for i, position_attribute in enumerate(self.position_attributes):
                data[position_attribute] = pos[i]
            values.append(
                [node_id]
                + [data.get(attr, None) for attr in self.position_attributes + attrs]
            )

        if len(values) == 0:
            logger.debug("No nodes to insert in %s", roi)
            return

        self._insert_query(self.nodes_table_name, columns, values, fail_if_exists=True)

    def update_nodes(
        self,
        nodes: dict[Any, dict[str, Any]],
        roi=None,
        attributes=None,
    ):
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")

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
            update_statement = (
                f"UPDATE {self.nodes_table_name} SET "
                f"{', '.join(setters)} WHERE "
                f"id={node}"
            )

            self._update_query(update_statement, commit=False)

        self._commit()

    def __init_metadata(self):
        metadata = self._read_metadata()

        if metadata:
            self.__check_metadata(metadata)
        else:
            metadata = self.__create_metadata()
            self._store_metadata(metadata)

    def __create_metadata(self):
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

        metadata = {
            "directed": self.directed,
            "total_roi_offset": self.total_roi.offset,
            "total_roi_shape": self.total_roi.shape,
            "node_attrs": {k: v.__name__ for k, v in self.node_attrs.items()},
            "edge_attrs": {k: v.__name__ for k, v in self.edge_attrs.items()},
        }

        return metadata

    def __check_metadata(self, metadata):
        """Checks if the provided metadata matches the existing
        metadata in the meta collection"""

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
        metadata["node_attrs"] = {k: eval(v) for k, v in metadata["node_attrs"].items()}
        metadata["edge_attrs"] = {k: eval(v) for k, v in metadata["edge_attrs"].items()}
        if self._node_attrs is not None:
            assert self.node_attrs == metadata["node_attrs"], (
                self.node_attrs,
                metadata["node_attrs"],
            )
        else:
            self.node_attrs = metadata["node_attrs"]
        if self._edge_attrs is not None:
            assert self.edge_attrs == metadata["edge_attrs"]
        else:
            self.edge_attrs = metadata["edge_attrs"]

    def __remove_keys(self, dictionary, keys):
        """Removes given keys from dictionary."""

        return {k: v for k, v in dictionary.items() if k not in keys}

    def __get_node_pos(self, n: dict[str, Any]) -> Coordinate:
        return Coordinate(
            (n.get(pos_attr, None) for pos_attr in self.position_attributes)
        )

    def __convert_to_sql(self, x: Any) -> str:
        if isinstance(x, str):
            return f"'{x}'"
        elif x is None:
            return "null"
        elif isinstance(x, bool):
            return f"{x}".lower()
        else:
            return str(x)

    def __attr_query(self, attrs: dict[str, Any]) -> str:
        query = ""
        for attr, value in attrs.items():
            query += f"{attr}={self.__convert_to_sql(value)} AND "
        query = query[:-5]
        return query

    def __roi_query(self, roi: Roi) -> str:
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
