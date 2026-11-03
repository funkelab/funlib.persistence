import logging
from abc import abstractmethod
from typing import Any, Iterable, Optional

import numpy as np
from funlib.geometry import Coordinate, Roi
from networkx import DiGraph, Graph
from networkx.classes.reportviews import NodeView, OutEdgeView

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
            metadata = self._read_metadata()
            if metadata is None:
                raise RuntimeError("metadata does not exist, can't open in read mode")
            self.__load_metadata(
                metadata,
                position_attribute=position_attribute,
                directed=directed,
                total_roi=total_roi,
                nodes_table=nodes_table,
                edges_table=edges_table,
                endpoint_names=endpoint_names,
                node_attrs=node_attrs,
                edge_attrs=edge_attrs,
            )

        if mode in self.create_modes:
            # this is where we populate default values for the DB creation

            assert node_attrs is not None, (
                "For DB creation (mode 'w'), node_attrs is a required "
                "argument and needs to contain at least the type definition "
                "for the position attribute"
            )

            def get(value, default):
                return value if value is not None else default

            self.position_attribute: str = get(position_attribute, "position")

            assert self.position_attribute in node_attrs, (
                "No type information for position attribute "
                f"'{self.position_attribute}' in 'node_attrs'"
            )

            position_type = node_attrs[self.position_attribute]
            if isinstance(position_type, Vec):
                self.ndims: int = position_type.size
                assert self.ndims > 1, (
                    "Don't use Vecs of size 1 for the position, use the "
                    "scalar type directly instead (i.e., 'float' instead of "
                    "'Vec(float, 1)'."
                )
                # if ndims == 1, we know that we have a single scalar now
            else:
                self.ndims = 1

            self.directed: bool = get(directed, False)
            self.total_roi: Roi = get(
                total_roi, Roi((None,) * self.ndims, (None,) * self.ndims)
            )
            self.nodes_table_name: str = get(nodes_table, "nodes")
            self.edges_table_name: str = get(edges_table, "edges")
            self.endpoint_names: list[str] = get(endpoint_names, ["u", "v"])
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

    @abstractmethod
    def _bulk_insert(self, table, columns, rows) -> None:
        """Insert rows using a backend-optimized bulk method.

        Args:
            table: Table name.
            columns: Column names.
            rows: Iterable of row lists (Python values, not formatted strings).
        """
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
        fetch_on_v: bool = False,
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
            # When a nodes_filter is used, the filtered node set is narrower
            # than the ROI. Fall back to the nodes path so edges only connect
            # nodes that passed the filter. Otherwise use the faster ROI path.
            if nodes_filter:
                edges = self.read_edges(
                    nodes=nodes,
                    read_attrs=edge_attrs,
                    attr_filter=edges_filter,
                    fetch_on_v=fetch_on_v,
                )
            else:
                # We use ROI to query edges ro avoid serializing a list of
                # node IDs into the SQL query.

                # A fully unbounded ROI (all None in shape) provides no
                # filtering, so treat it as None to fetch all edges.
                effective_roi = roi
                if roi is not None and all(s is None for s in roi.shape):
                    effective_roi = None
                edges = self.read_edges(
                    roi=effective_roi,
                    read_attrs=edge_attrs,
                    attr_filter=edges_filter,
                    fetch_on_v=fetch_on_v,
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

    def bulk_write_graph(
        self,
        graph: Graph,
        roi: Optional[Roi] = None,
        write_nodes: bool = True,
        write_edges: bool = True,
        node_attrs: Optional[list[str]] = None,
        edge_attrs: Optional[list[str]] = None,
    ) -> None:
        if write_nodes:
            self.bulk_write_nodes(graph.nodes, roi=roi, attributes=node_attrs)
        if write_edges:
            self.bulk_write_edges(
                graph.nodes, graph.edges, roi=roi, attributes=edge_attrs
            )

    def bulk_write_nodes(self, nodes, roi=None, attributes=None):
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")

        attrs = attributes if attributes is not None else list(self.node_attrs.keys())
        columns = ["id"] + list(attrs)

        def rows():
            for node_id, data in nodes.items():
                pos = self.__get_node_pos(data)
                if roi is not None and not roi.contains(pos):
                    continue
                yield [node_id] + [data.get(attr, None) for attr in attrs]

        self._bulk_insert(self.nodes_table_name, columns, rows())

    def bulk_write_edges(self, nodes, edges, roi=None, attributes=None):
        if self.mode == "r":
            raise RuntimeError("Trying to write to read-only DB")

        u_name, v_name = self.endpoint_names
        attrs = attributes if attributes is not None else list(self.edge_attrs.keys())
        columns = [u_name, v_name] + list(attrs)

        def rows():
            for (u, v), data in edges.items():
                if not self.directed:
                    u, v = min(u, v), max(u, v)
                if roi is not None:
                    pos_u = self.__get_node_pos(nodes[u])
                    if pos_u is None or not roi.contains(pos_u):
                        continue
                yield [u, v] + [data.get(attr, None) for attr in attrs]

        self._bulk_insert(self.edges_table_name, columns, rows())

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

    # Node ids per ``WHERE id IN (...)`` clause. The ids are inlined as
    # literals, so a bounded batch keeps each statement well inside any
    # backend's statement length limit.
    _READ_NODES_BY_ID_BATCH = 50_000

    def read_nodes(
        self,
        roi: Optional[Roi] = None,
        attr_filter: Optional[dict[str, Any]] = None,
        read_attrs: Optional[list[str]] = None,
        join_collection: Optional[str] = None,
        ids: Optional[Iterable[int]] = None,
    ) -> list[dict[str, Any]]:
        """Return a list of nodes within ``roi``, or the nodes with the given
        ``ids``, or all nodes.

        Arguments:

            roi (``Roi`` or ``None``):

                Only return nodes whose position lies in this roi.

            ids (iterable of ``int`` or ``None``):

                Only return the nodes with these ids. Ids that are not in the
                database are silently absent from the result, so the caller can
                look up which of a set of ids exist. The order of the result is
                not guaranteed to match the order of ``ids``. Cannot be
                combined with ``roi``.

            attr_filter, read_attrs:

                As for ``read_graph``.
        """

        if roi is not None and ids is not None:
            raise ValueError(
                "read_nodes does not support both roi and ids at the same time. "
                "Pass one or the other."
            )

        # attributes to read
        read_attrs = list(self.node_attrs.keys()) if read_attrs is None else read_attrs

        # corresponding column naes
        read_columns = ["id"] + self._node_attrs_to_columns(read_attrs)
        read_attrs = ["id"] + read_attrs
        read_attrs_query = ", ".join(read_columns)

        nodes: list[dict[str, Any]] = []
        for statement in self._node_select_statements(
            read_attrs_query, roi, ids, attr_filter
        ):
            nodes.extend(
                self._columns_to_node_attrs(
                    {key: val for key, val in zip(read_columns, values)}, read_attrs
                )
                for values in self._select_query(statement)
            )
        return nodes

    def _node_select_statements(
        self,
        columns_query: str,
        roi: Optional[Roi],
        ids: Optional[Iterable[int]],
        attr_filter: Optional[dict[str, Any]],
    ) -> list[str]:
        """The SELECT statement(s) for ``read_nodes``/``read_nodes_arrays``:
        one per batch of ids, else exactly one."""
        attr_clause = (
            self.__attr_query(attr_filter)
            if attr_filter is not None and len(attr_filter) > 0
            else None
        )

        def statement(where_clauses: list[str]) -> str:
            stmt = f"SELECT {columns_query} FROM {self.nodes_table_name}"
            if where_clauses:
                stmt += " WHERE " + " AND ".join(where_clauses)
            return stmt

        if ids is not None:
            id_list = [int(i) for i in ids]
            logger.debug("Reading %d nodes by id", len(id_list))
            batch = self._READ_NODES_BY_ID_BATCH
            statements = []
            for start in range(0, len(id_list), batch):
                chunk = ", ".join(str(i) for i in id_list[start : start + batch])
                clauses = [f"id IN ({chunk})"]
                if attr_clause is not None:
                    clauses.append(attr_clause)
                statements.append(statement(clauses))
            return statements

        logger.debug("Reading nodes in roi %s" % roi)
        clauses = []
        if roi is not None:
            clauses.append(self.__roi_query(roi).replace("WHERE ", "", 1))
        if attr_clause is not None:
            clauses.append(attr_clause)
        return [statement([c for c in clauses if c])]

    def read_nodes_arrays(
        self,
        roi: Optional[Roi] = None,
        ids: Optional[Iterable[int]] = None,
        attr_filter: Optional[dict[str, Any]] = None,
        read_attrs: Optional[list[str]] = None,
        chunk_size: int = 20_000,
    ) -> dict[str, np.ndarray]:
        """Columnar version of ``read_nodes``: the same selection, returned as
        ``{"id": int64[n], attr: array[n] or array[n, d], ...}`` instead of a
        list of dicts. Numeric attributes are float64 with NaN for a missing
        value (``Vec`` attributes as ``(n, d)``), anything else is an object
        array. Rows are consumed in chunks so peak memory is the arrays plus
        one chunk of rows, not one dict per node."""
        read_attrs = list(self.node_attrs.keys()) if read_attrs is None else read_attrs
        columns = ["id"] + self._node_attrs_to_columns(read_attrs)
        statements = self._node_select_statements(
            ", ".join(columns), roi, ids, attr_filter
        )
        converters, members = self._column_plan(
            columns, read_attrs, self.node_attrs, ("id",)
        )
        raw = self._fetch_columns(statements, converters, chunk_size)
        return self._assemble_columns(
            raw, columns, read_attrs, self.node_attrs, ("id",), members
        )

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
        fetch_on_v: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Returns a list of edges within roi, connected to provided nodes, or all edges.

        Args:
            fetch_on_v: If True, also match edges where the v endpoint is in the
                node list or ROI. If False (default), only match on u.

        Raises:
            ValueError: If both roi and nodes are provided.
        """

        endpoint_names = self.endpoint_names
        # Columns to select from the edge table (T1)
        edge_table_cols = endpoint_names + list(self.edge_attrs.keys())
        select_statement = self._edge_select_statement(
            edge_table_cols, roi, nodes, attr_filter, fetch_on_v
        )
        if select_statement is None:
            return []

        # Define the keys for the result dictionaries
        all_edge_keys = endpoint_names + list(self.edge_attrs.keys())

        # Define which keys to keep based on read_attrs
        final_edge_keys = endpoint_names + (
            list(self.edge_attrs.keys()) if read_attrs is None else read_attrs
        )
        edges = [
            {
                key: val
                for key, val in zip(all_edge_keys, values)
                if key in final_edge_keys
            }
            for values in self._select_query(select_statement)
        ]

        return edges

    def read_edges_arrays(
        self,
        roi: Optional[Roi] = None,
        nodes: Optional[list[dict[str, Any]]] = None,
        attr_filter: Optional[dict[str, Any]] = None,
        read_attrs: Optional[list[str]] = None,
        fetch_on_v: bool = False,
        chunk_size: int = 20_000,
    ) -> dict[str, np.ndarray]:
        """Columnar version of ``read_edges``: the same selection, returned as
        ``{u: int64[n], v: int64[n], attr: float64[n], ...}`` instead of a
        list of dicts. Numeric attributes are float64 with NaN for a missing
        value (``Vec`` attributes as ``(n, d)``), anything else is an object
        array. Only the requested attributes are selected. Rows are consumed
        in chunks, so peak memory is the arrays plus one chunk of rows rather
        than one dict per edge (roughly 10-20x less for a plain weighted
        edge)."""
        endpoint_names = self.endpoint_names
        read_attrs = list(self.edge_attrs.keys()) if read_attrs is None else read_attrs
        columns = list(endpoint_names) + self._edge_attrs_to_columns(read_attrs)
        statement = self._edge_select_statement(
            columns, roi, nodes, attr_filter, fetch_on_v
        )
        statements = [] if statement is None else [statement]
        ids = tuple(endpoint_names)
        converters, members = self._column_plan(
            columns, read_attrs, self.edge_attrs, ids
        )
        raw = self._fetch_columns(statements, converters, chunk_size)
        return self._assemble_columns(
            raw, columns, read_attrs, self.edge_attrs, ids, members
        )

    def _edge_select_statement(
        self,
        edge_table_cols: list[str],
        roi: Optional[Roi],
        nodes: Optional[list[dict[str, Any]]],
        attr_filter: Optional[dict[str, Any]],
        fetch_on_v: bool,
    ) -> Optional[str]:
        """The SELECT statement for ``read_edges``/``read_edges_arrays``, or
        ``None`` when the selection is empty by construction."""
        if roi is not None and nodes is not None:
            raise ValueError(
                "read_edges does not support both roi and nodes at the same time. "
                "Pass one or the other."
            )

        endpoint_names = self.endpoint_names

        # 1. Determine the base SELECT statement and WHERE clause
        desired_columns = ", ".join(edge_table_cols)

        # Base query starts with selecting all columns from the edges table
        select_statement = f"SELECT {desired_columns} FROM {self.edges_table_name}"
        where_clauses = []
        using_join = False

        if nodes is not None:
            # Case 1: Filter by explicit list of nodes
            if len(nodes) == 0:
                return None

            node_ids = ", ".join([str(node["id"]) for node in nodes])
            if fetch_on_v:
                where_clauses.append(
                    f"({endpoint_names[0]} IN ({node_ids})"
                    f" OR {endpoint_names[1]} IN ({node_ids}))"
                )
            else:
                where_clauses.append(f"{endpoint_names[0]} IN ({node_ids})")

        elif roi is not None:
            # Case 2: Filter by ROI using INNER JOIN
            using_join = True
            node_id_column = "id"

            edge_cols = ", ".join([f"T1.{col}" for col in edge_table_cols])
            roi_condition = self.__roi_query(roi).replace("WHERE ", "", 1)

            def join_on(endpoint: str) -> str:
                return (
                    f"SELECT {edge_cols} "
                    f"FROM {self.edges_table_name} AS T1 "
                    f"INNER JOIN {self.nodes_table_name} AS T2 "
                    f"ON T1.{endpoint} = T2.{node_id_column} "
                )

            # ``fetch_on_v`` used to join ``ON T1.u = T2.id OR T1.v = T2.id``.
            # An OR in the join condition defeats the endpoint indexes and
            # makes every read a full scan of the edge table, so it is
            # expressed as the UNION of two single-endpoint joins instead
            # (UNION also removes the duplicates DISTINCT used to).
            join_statements = [join_on(endpoint_names[0])]
            if fetch_on_v:
                join_statements.append(join_on(endpoint_names[1]))
            if roi_condition:
                where_clauses.append(roi_condition)

        # Case 3: Both nodes and roi are None — fetch all edges

        # 2. Add Attribute Filter to WHERE clauses
        if attr_filter is not None and len(attr_filter) > 0:
            if using_join:
                # Qualify each attribute with T1 for the JOIN case
                parts = [
                    f"T1.{k}={self.__convert_to_sql(v)}" for k, v in attr_filter.items()
                ]
                where_clauses.append(" AND ".join(parts))
            else:
                where_clauses.append(self.__attr_query(attr_filter))

        # 3. Finalize the SELECT statement
        where = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        if using_join:
            select_statement = " UNION ".join(stmt + where for stmt in join_statements)
        else:
            select_statement += where

        logger.debug(f"Reading edges with query: {select_statement}")
        return select_statement

    # -- columnar reads ----------------------------------------------------

    @staticmethod
    def _column_converter(attr_type, size: Optional[int] = None):
        """Converter from one chunk of raw column values (a Python list) to an
        array: ids/int64 for ``None`` type, numeric -> float64 with NaN for
        None (``(chunk, size)`` when the column holds sequences), else object."""
        if attr_type is None:

            def ids(chunk):
                return np.array(chunk, dtype=np.int64)

            return ids
        numeric = attr_type in (int, float, bool) or (
            isinstance(attr_type, Vec) and attr_type.dtype in (int, float, bool)
        )
        if numeric and size is not None:
            width: int = size  # bound here: narrowing does not reach the closure

            def vec(chunk):
                filled = [[np.nan] * width if x is None else list(x) for x in chunk]
                return np.array(filled, dtype=np.float64).reshape(-1, width)

            return vec
        if numeric:

            def scalar(chunk):
                return np.array(
                    [np.nan if x is None else x for x in chunk], dtype=np.float64
                )

            return scalar

        def obj(chunk):
            arr = np.empty(len(chunk), dtype=object)
            arr[:] = chunk
            return arr

        return obj

    def _fetch_columns(
        self, statements: list[str], converters: list, chunk_size: int
    ) -> list[list[np.ndarray]]:
        """Run ``statements`` and collect their rows column-wise. Every chunk
        of rows is converted to arrays immediately, so the Python row objects
        of at most one chunk are alive at a time."""
        n_columns = len(converters)
        columns: list[list[np.ndarray]] = [[] for _ in range(n_columns)]
        for statement in statements:
            cursor = self._select_query(statement)
            fetchmany = getattr(cursor, "fetchmany", None)
            if fetchmany is None:  # a materialised sequence of rows
                rows_all = list(cursor)
                chunks = (
                    rows_all[i : i + chunk_size]
                    for i in range(0, len(rows_all), chunk_size)
                )
            else:
                chunks = iter(lambda: fetchmany(chunk_size), [])
            for rows in chunks:
                if not rows:
                    break
                for j in range(n_columns):
                    columns[j].append(converters[j]([row[j] for row in rows]))
                del rows
        return columns

    def _column_plan(
        self,
        columns: list[str],
        attrs: list[str],
        attr_types: dict[str, AttributeType],
        id_columns: tuple[str, ...],
    ) -> tuple[list, dict[str, list[str]]]:
        """Per selected column, how to convert it; and per attribute, which
        selected columns make it up (several for a ``Vec`` stored exploded)."""
        converters = []
        members: dict[str, list[str]] = {}
        col_type: dict[str, tuple] = {}
        for name in id_columns:
            col_type[name] = (None, None)
        for attr in attrs:
            attr_type = attr_types[attr]
            if isinstance(attr_type, Vec):
                exploded = [f"{attr}_{d}" for d in range(attr_type.size)]
                if all(c in columns for c in exploded):
                    members[attr] = exploded
                    for c in exploded:
                        col_type[c] = (attr_type, None)
                else:
                    members[attr] = [attr]
                    col_type[attr] = (attr_type, attr_type.size)
            else:
                members[attr] = [attr]
                col_type[attr] = (attr_type, None)
        for name in columns:
            converters.append(self._column_converter(*col_type[name]))
        return converters, members

    @staticmethod
    def _assemble_columns(
        raw: list[list[np.ndarray]],
        columns: list[str],
        attrs: list[str],
        attr_types: dict[str, AttributeType],
        id_columns: tuple[str, ...],
        members: dict[str, list[str]],
    ) -> dict[str, np.ndarray]:
        by_name = dict(zip(columns, raw))

        def cat(name, empty):
            chunks = by_name[name]
            return np.concatenate(chunks) if chunks else empty

        out: dict[str, np.ndarray] = {}
        for name in id_columns:
            out[name] = cat(name, np.zeros(0, dtype=np.int64))
        for attr in attrs:
            attr_type = attr_types[attr]
            if isinstance(attr_type, Vec) and len(members[attr]) > 1:
                parts = [cat(c, np.zeros(0, dtype=np.float64)) for c in members[attr]]
                out[attr] = np.stack(parts, axis=1)
            elif isinstance(attr_type, Vec):
                out[attr] = cat(attr, np.zeros((0, attr_type.size), dtype=np.float64))
            else:
                numeric = attr_type in (int, float, bool)
                out[attr] = cat(
                    attr, np.zeros(0, dtype=np.float64 if numeric else object)
                )
        return out

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

        self._insert_query(
            self.nodes_table_name, columns, values, fail_if_exists=fail_if_exists
        )

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

            setters = []
            for attr in attrs:
                value = data.get(attr)
                columns = self._node_attrs_to_columns([attr])
                if len(columns) == 1:
                    setters.append(f"{columns[0]} = {self.__convert_to_sql(value)}")
                else:
                    # attr is stored across multiple columns (e.g. a Vec in the
                    # SQLite backend -> attr_0, attr_1, ...); expand the value to
                    # one assignment per column instead of writing the whole
                    # sequence into a single column (which produced invalid SQL).
                    components = [None] * len(columns) if value is None else list(value)
                    setters.extend(
                        f"{col} = {self.__convert_to_sql(component)}"
                        for col, component in zip(columns, components)
                    )
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

    def __load_metadata(
        self,
        metadata,
        position_attribute: Optional[str] = None,
        directed: Optional[bool] = None,
        total_roi: Optional[Roi] = None,
        nodes_table: Optional[str] = None,
        edges_table: Optional[str] = None,
        endpoint_names: Optional[list[str]] = None,
        node_attrs: Optional[dict[str, AttributeType]] = None,
        edge_attrs: Optional[dict[str, AttributeType]] = None,
    ):
        """Load the provided metadata into this object's attributes, check if
        user-provided overrides are consistent with stored metadata."""

        # For each simple attribute, use metadata as the source of truth.
        # If the user also provided a value, check consistency.
        overrides: dict[str, Any] = {
            "position_attribute": position_attribute,
            "directed": directed,
            "nodes_table_name": nodes_table,
            "edges_table_name": edges_table,
            "endpoint_names": endpoint_names,
            "ndims": None,  # ndims is never user-provided
        }
        for attr_name, override in overrides.items():
            stored = metadata[attr_name]
            if override is not None:
                assert override == stored, (
                    f"Attribute {attr_name} was given as {override}, but "
                    f"disagrees with the stored metadata value of {stored}"
                )
            setattr(self, attr_name, stored)

        # total_roi
        stored_roi = Roi(metadata["total_roi_offset"], metadata["total_roi_shape"])
        if total_roi is not None:
            assert total_roi == stored_roi, (
                f"Attribute total_roi was given as {total_roi}, but "
                f"disagrees with the stored metadata value of {stored_roi}"
            )
        self.total_roi = stored_roi

        # node_attrs / edge_attrs
        stored_node_attrs = {k: eval(v) for k, v in metadata["node_attrs"].items()}
        stored_edge_attrs = {k: eval(v) for k, v in metadata["edge_attrs"].items()}
        if node_attrs is not None:
            assert node_attrs == stored_node_attrs, (
                f"Attribute node_attrs was given as {node_attrs}, but "
                f"disagrees with the stored metadata value of {stored_node_attrs}"
            )
        self._node_attrs = stored_node_attrs
        if edge_attrs is not None:
            assert edge_attrs == stored_edge_attrs, (
                f"Attribute edge_attrs was given as {edge_attrs}, but "
                f"disagrees with the stored metadata value of {stored_edge_attrs}"
            )
        self._edge_attrs = stored_edge_attrs

    def __remove_keys(self, dictionary, keys):
        """Removes given keys from dictionary."""

        return {k: v for k, v in dictionary.items() if k not in keys}

    def __get_node_pos(self, n: dict[str, Any]) -> Optional[Coordinate]:
        try:
            return Coordinate(n[self.position_attribute])
        except KeyError:
            return None

    def __convert_to_sql(self, x: Any) -> str:
        # normalize numpy values to their native python equivalents; all
        # other types pass through untouched
        if isinstance(x, np.ndarray):
            x = x.tolist()
        elif isinstance(x, np.generic):
            x = x.item()
        if isinstance(x, str):
            return f"'{x}'"
        elif x is None:
            return "null"
        elif isinstance(x, bool):
            return f"{x}".lower()
        elif isinstance(x, (list, tuple)):
            # a native array column (e.g. a Vec stored as a single array column
            # in the PostgreSQL backend) -> array literal '{v0,v1,...}'
            return "'{" + ",".join(map(str, x)) + "}'"
        else:
            return str(x)

    def __attr_query(self, attrs: dict[str, Any]) -> str:
        query = ""
        for attr, value in attrs.items():
            query += f"{attr}={self.__convert_to_sql(value)} AND "
        query = query[:-5]
        return query

    def __roi_query(self, roi: Roi) -> str:
        """``WHERE <position inside roi>`` for the bounded dimensions of
        ``roi``, or the empty string when every dimension is unbounded."""
        pos_attr = self.position_attribute
        conditions = []
        for dim in range(self.ndims):
            if roi.begin[dim] is not None:
                conditions.append(f"{pos_attr}[{dim + 1}]>={roi.begin[dim]}")
            if roi.end[dim] is not None:
                conditions.append(f"{pos_attr}[{dim + 1}]<{roi.end[dim]}")
        if not conditions:
            return ""
        return "WHERE " + " AND ".join(conditions)

    def print_summary(self):
        raise ValueError("Not implemented for base SQLGraphDataBase")
