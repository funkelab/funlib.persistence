import logging
from abc import abstractmethod
from typing import Any, Iterable, Optional

from networkx import DiGraph, Graph
from networkx.classes.reportviews import NodeView, OutEdgeView

from funlib.geometry import Coordinate, Roi

from ..types import Vec, type_to_str
from .graph_database import AttributeType, GraphDataBase

logger = logging.getLogger(__name__)


class SQLGraphDataBase(GraphDataBase):
    """Base class for SQL-based graph databases.

    Nodes must have a position attribute (set via argument
    ``position_attribute``), which will be used for geometric slicing (see
    ``__getitem__`` and ``read_graph``).

    Arguments:

        mode (``string``):

            Any of ``r`` (read-only), ``r+`` (read and allow modifications),
            or ``w`` (create new database, overwrite if exists).

        position_attribute (``string``):

            The node attribute that contains position information. This will be
            used for slicing subgraphs via ``__getitem__``.

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

    read_modes = ["r", "r+"]
    write_modes = ["r+", "w"]
    create_modes = ["w"]
    valid_modes = ["r", "r+", "w"]

    _node_attrs: Optional[dict[str, AttributeType]] = None
    _edge_attrs: Optional[dict[str, AttributeType]] = None

    def __init__(
        self,
        mode: str = "r+",
        position_attribute: Optional[str] = None,
        directed: Optional[bool] = None,
        total_roi: Optional[Roi] = None,
        nodes_table: Optional[str] = None,
        edges_table: Optional[str] = None,
        endpoint_names: Optional[list[str]] = None,
        node_attrs: Optional[dict[str, AttributeType]] = None,
        edge_attrs: Optional[dict[str, AttributeType]] = None,
    ):
        assert mode in self.valid_modes, (
            f"Mode '{mode}' not in allowed modes {self.valid_modes}"
        )
        self.mode = mode

        if mode in self.read_modes:
            self.position_attribute = position_attribute
            self.directed = directed
            self.total_roi = total_roi
            self.nodes_table_name = nodes_table
            self.edges_table_name = edges_table
            self.endpoint_names = endpoint_names
            self._node_attrs = node_attrs
            self._edge_attrs = edge_attrs
            self.ndims = None  # to be read from metadata

            metadata = self._read_metadata()
            if metadata is None:
                raise RuntimeError("metadata does not exist, can't open in read mode")
            self.__load_metadata(metadata)

        if mode in self.create_modes:
            # this is where we populate default values for the DB creation

            assert node_attrs is not None, (
                "For DB creation (mode 'w'), node_attrs is a required "
                "argument and needs to contain at least the type definition "
                "for the position attribute"
            )

            def get(value, default):
                return value if value is not None else default

            self.position_attribute = get(position_attribute, "position")

            assert self.position_attribute in node_attrs, (
                "No type information for position attribute "
                f"'{self.position_attribute}' in 'node_attrs'"
            )

            position_type = node_attrs[self.position_attribute]
            if isinstance(position_type, Vec):
                self.ndims = position_type.size
                assert self.ndims > 1, (
                    "Don't use Vecs of size 1 for the position, use the "
                    "scalar type directly instead (i.e., 'float' instead of "
                    "'Vec(float, 1)'."
                )
                # if ndims == 1, we know that we have a single scalar now
            else:
                self.ndims = 1

            self.directed = get(directed, False)
            self.total_roi = get(
                total_roi, Roi((None,) * self.ndims, (None,) * self.ndims)
            )
            self.nodes_table_name = get(nodes_table, "nodes")
            self.edges_table_name = get(edges_table, "edges")
            self.endpoint_names = get(endpoint_names, ["u", "v"])
            self._node_attrs = node_attrs  # no default, needs to be given
            self._edge_attrs = get(edge_attrs, {})

            # delete previous DB, if exists
            self._drop_tables()

            # create new DB
            self._create_tables()

            # store metadata
            metadata = self.__create_metadata()
            self._store_metadata(metadata)

    @abstractmethod
    def _drop_edges(self) -> None:
        pass

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

    def _node_attrs_to_columns(self, attrs):
        # default: each attribute maps to its own column
        return attrs

    def _columns_to_node_attrs(self, columns, attrs):
        # default: each column maps to one attribute
        return columns

    def _edge_attrs_to_columns(self, attrs):
        # default: each attribute maps to its own column
        return attrs

    def _columns_to_edge_attrs(self, columns, attrs):
        # default: each column maps to one attribute
        return columns

    def read_graph(
        self,
        roi: Optional[Roi] = None,
        read_edges: bool = True,
        node_attrs: Optional[list[str]] = None,
        edge_attrs: Optional[list[str]] = None,
        nodes_filter: Optional[dict[str, Any]] = None,
        edges_filter: Optional[dict[str, Any]] = None,
    ) -> Graph:
        graph: Graph
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
            u, v = self.endpoint_names  # type: ignore
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
            nodes=graph.nodes,
            roi=roi,
            attributes=node_attrs,
        )
        self.update_edges(
            nodes=graph.nodes,
            edges=graph.edges,
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
    def node_attrs(self) -> dict[str, AttributeType]:
        return self._node_attrs if self._node_attrs is not None else {}

    @node_attrs.setter
    def node_attrs(self, value: dict[str, AttributeType]) -> None:
        self._node_attrs = value

    @property
    def edge_attrs(self) -> dict[str, AttributeType]:
        return self._edge_attrs if self._edge_attrs is not None else {}

    @edge_attrs.setter
    def edge_attrs(self, value: dict[str, AttributeType]) -> None:
        self._edge_attrs = value

    def read_nodes(
        self,
        roi: Optional[Roi] = None,
        attr_filter: Optional[dict[str, Any]] = None,
        read_attrs: Optional[list[str]] = None,
        join_collection: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return a list of nodes within roi."""

        # attributes to read
        read_attrs = list(self.node_attrs.keys()) if read_attrs is None else read_attrs

        # corresponding column naes
        read_columns = ["id"] + self._node_attrs_to_columns(read_attrs)
        read_attrs = ["id"] + read_attrs
        read_attrs_query = ", ".join(read_columns)

        logger.debug("Reading nodes in roi %s" % roi)
        select_statement = (
            f"SELECT {read_attrs_query} FROM {self.nodes_table_name} "
            + (self.__roi_query(roi) if roi is not None else "")
            + (
                f" {'WHERE' if roi is None else 'AND'} "
                + self.__attr_query(attr_filter)
                if attr_filter is not None and len(attr_filter) > 0
                else ""
            )
        )

        attr_filter = attr_filter if attr_filter is not None else {}
        for k, v in attr_filter.items():
            select_statement += f" AND {k}={self.__convert_to_sql(v)}"

        nodes = [
            self._columns_to_node_attrs(
                {key: val for key, val in zip(read_columns, values)}, read_attrs
            )
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

        endpoint_names = self.endpoint_names
        assert endpoint_names is not None

        node_ids = ", ".join([str(node["id"]) for node in nodes])
        node_condition = f"{endpoint_names[0]} IN ({node_ids})"  # type: ignore

        logger.debug("Reading nodes in roi %s" % roi)
        # TODO: AND vs OR here
        desired_columns = ", ".join(endpoint_names + list(self.edge_attrs.keys()))  # type: ignore
        select_statement = (
            f"SELECT {desired_columns} FROM {self.edges_table_name} WHERE "
            + node_condition
            + (
                " AND " + self.__attr_query(attr_filter)
                if attr_filter is not None and len(attr_filter) > 0
                else ""
            )
        )

        edge_attrs = endpoint_names + (  # type: ignore
            list(self.edge_attrs.keys()) if read_attrs is None else read_attrs
        )
        attr_filter = attr_filter if attr_filter is not None else {}
        for k, v in attr_filter.items():
            select_statement += f" AND {k}={self.__convert_to_sql(v)}"

        edges = [
            {
                key: val
                for key, val in zip(
                    endpoint_names + list(self.edge_attrs.keys()),
                    values,  # type: ignore
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
                (None,) * self.ndims,
                (None,) * self.ndims,
            )

        values = []
        for (u, v), data in edges.items():
            if not self.directed:
                u, v = min(u, v), max(u, v)
            pos_u = self.__get_node_pos(nodes[u])

            if pos_u is None or not roi.contains(pos_u):
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
        nodes: NodeView,
        edges: OutEdgeView,
        roi=None,
        attributes=None,
    ):
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")

        logger.debug("Writing nodes in %s", roi)

        attrs = attributes if attributes is not None else []

        for u, v, data in edges(data=True):
            if not self.directed:
                u, v = min(u, v), max(u, v)
            if roi is not None:
                pos_u = self.__get_node_pos(nodes[u])

                if not roi.contains(pos_u):
                    logger.debug(
                        (
                            f"Skipping edge with {self.endpoint_names[0]} {{}}, {self.endpoint_names[1]} {{}},"  # type: ignore
                            + f"and data {{}} because {self.endpoint_names[0]} not in roi {{}}"  # type: ignore
                        ).format(u, v, data, roi)
                    )
                    continue

            values = [data.get(attr) for attr in attrs]
            setters = [f"{k}={v}" for k, v in zip(attrs, values)]
            update_statement = (
                f"UPDATE {self.edges_table_name} SET "
                f"{', '.join(setters)} WHERE "
                f"{self.endpoint_names[0]}={u} AND {self.endpoint_names[1]}={v}"  # type: ignore
            )

            self._update_query(update_statement, commit=False)

        self._commit()

    def write_nodes(
        self,
        nodes: NodeView,
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
        columns = ("id",) + tuple(attrs)

        values = []
        for node_id, data in nodes.items():
            data = data.copy()
            pos = self.__get_node_pos(data)
            if roi is not None and not roi.contains(pos):
                continue
            values.append([node_id] + [data.get(attr, None) for attr in attrs])

        if len(values) == 0:
            logger.debug("No nodes to insert in %s", roi)
            return

        self._insert_query(self.nodes_table_name, columns, values, fail_if_exists=True)

    def update_nodes(
        self,
        nodes: NodeView,
        roi=None,
        attributes=None,
    ):
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")

        logger.debug("Writing nodes in %s", roi)

        attrs = attributes if attributes is not None else []

        for node, data in nodes(data=True):
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

    def __create_metadata(self):
        """Sets the metadata in the meta collection to the provided values"""

        metadata = {
            "position_attribute": self.position_attribute,
            "directed": self.directed,
            "total_roi_offset": self.total_roi.offset,
            "total_roi_shape": self.total_roi.shape,
            "nodes_table_name": self.nodes_table_name,
            "edges_table_name": self.edges_table_name,
            "endpoint_names": self.endpoint_names,
            "node_attrs": {k: type_to_str(v) for k, v in self.node_attrs.items()},
            "edge_attrs": {k: type_to_str(v) for k, v in self.edge_attrs.items()},
            "ndims": self.ndims,
        }

        return metadata

    def __load_metadata(self, metadata):
        """Load the provided metadata into this object's attributes, check if
        it is consistent with already populated fields."""

        # simple attributes
        for attr_name in [
            "position_attribute",
            "directed",
            "nodes_table_name",
            "edges_table_name",
            "endpoint_names",
            "ndims",
        ]:
            if getattr(self, attr_name) is None:
                setattr(self, attr_name, metadata[attr_name])
            else:
                value = getattr(self, attr_name)
                assert value == metadata[attr_name], (
                    f"Attribute {attr_name} is already set to {value} for this "
                    "object, but disagrees with the stored metadata value of "
                    f"{metadata[attr_name]}"
                )

        # special attributes

        total_roi = Roi(metadata["total_roi_offset"], metadata["total_roi_shape"])
        if self.total_roi is None:
            self.total_roi = total_roi
        else:
            assert self.total_roi == total_roi, (
                f"Attribute total_roi is already set to {self.total_roi} for "
                "this object, but disagrees with the stored metadata value of "
                f"{total_roi}"
            )

        node_attrs = {k: eval(v) for k, v in metadata["node_attrs"].items()}
        edge_attrs = {k: eval(v) for k, v in metadata["edge_attrs"].items()}
        if self._node_attrs is None:
            self.node_attrs = node_attrs
        else:
            assert self.node_attrs == node_attrs, (
                f"Attribute node_attrs is already set to {self.node_attrs} for "
                "this object, but disagrees with the stored metadata value of "
                f"{node_attrs}"
            )
        if self._edge_attrs is None:
            self.edge_attrs = edge_attrs
        else:
            assert self.edge_attrs == edge_attrs, (
                f"Attribute edge_attrs is already set to {self.edge_attrs} for "
                "this object, but disagrees with the stored metadata value of "
                f"{edge_attrs}"
            )

    def __remove_keys(self, dictionary, keys):
        """Removes given keys from dictionary."""

        return {k: v for k, v in dictionary.items() if k not in keys}

    def __get_node_pos(self, n: dict[str, Any]) -> Optional[Coordinate]:
        try:
            return Coordinate(n[self.position_attribute])  # type: ignore
        except KeyError:
            return None

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
        pos_attr = self.position_attribute
        for dim in range(self.ndims):  # type: ignore
            if dim > 0:
                query += " AND "
            if roi.begin[dim] is not None and roi.end[dim] is not None:
                query += (
                    f"{pos_attr}[{dim + 1}] BETWEEN {roi.begin[dim]} and {roi.end[dim]}"
                )
            elif roi.begin[dim] is not None:
                query += f"{pos_attr}[{dim + 1}]>={roi.begin[dim]}"
            elif roi.begin[dim] is not None:
                query += f"{pos_attr}[{dim + 1}]<{roi.end[dim]}"
            else:
                query = query[:-5]
        return query
