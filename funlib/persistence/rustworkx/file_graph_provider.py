from .shared_graph_provider import SharedGraphProvider

from funlib.geometry import Coordinate
from funlib.geometry import Roi

import sqlite3

import json
import logging
import os
from typing import Optional, Any, Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


class FileGraphProvider(SharedGraphProvider):
    """Provides shared graphs stored in sqlite file.

    Nodes are assumed to have at least an attribute ``id`` and a position
    attribute (set via argument ``position_attribute``, defaults to
    ``position``).

    Edges are assumed to have at least attributes ``u``, ``v``.

    Arguments:

        db_file (``Path``):

            The path to the graph container sqlite db.

        mode (``string``, optional):

            One of ``r``, ``r+``, or ``w``. Defaults to ``r+``. ``w`` drops the
            node, edge, and meta collections.

        directed (``bool``):

            True if the graph is directed, false otherwise

        nodes_collection (``string``):
        edges_collection (``string``):

            Names of the nodes and edges collections, should they differ from
            ``nodes`` and ``edges``.

        position_attribute (``string`` or list of ``string``s, optional):

            The node attribute(s) that contain position information. This will
            be used for slicing subgraphs via ``__getitem__``. If a single
            string, the attribute is assumed to be an array. If a list, each
            entry denotes the position coordinates in order (e.g.,
            `position_z`, `position_y`, `position_x`).
    """

    def __init__(
        self,
        db_file: Path,
        mode: str = "r+",
        directed: Optional[bool] = None,
        total_roi: Optional[Roi] = None,
        nodes_collection: str = "nodes",
        edges_collection: str = "edges",
        position_attributes: Iterable[str] = "zyx",
        node_attrs: Optional[Iterable[str]] = None,
        edge_attrs: Optional[Iterable[str]] = None,
    ):
        self.db_file = db_file
        self.con = sqlite3.connect(db_file)
        self.cur = self.con.cursor()
        self.mode = mode
        self.directed = directed
        self.total_roi = total_roi
        self.nodes_collection_name = nodes_collection
        self.edges_collection_name = edges_collection
        self.meta_collection = self.db_file.parent / f"{self.db_file.stem}-meta.json"
        self.position_attributes = list(position_attributes)
        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs

        if mode == "w":
            self.drop_tables()

        self.create_tables()

        if os.path.exists(self.meta_collection):
            self.__check_metadata()
        else:
            self.__set_metadata()

    @property
    def node_attrs(self) -> list[str]:
        return self.__node_attrs if self.__node_attrs is not None else []

    @node_attrs.setter
    def node_attrs(self, value: Iterable[str]) -> None:
        self.__node_attrs = value

    @property
    def edge_attrs(self) -> list[str]:
        return self.__edge_attrs if self.__edge_attrs is not None else []

    @edge_attrs.setter
    def edge_attrs(self, value: Iterable[str]) -> None:
        self.__edge_attrs = value

    def drop_tables(self) -> None:
        logger.info(
            "dropping collections %s, %s",
            self.nodes_collection_name,
            self.edges_collection_name,
        )

        self.cur.execute(f"DROP TABLE IF EXISTS {self.nodes_collection_name}")
        self.cur.execute(f"DROP TABLE IF EXISTS {self.edges_collection_name}")

        if self.meta_collection.exists():
            self.meta_collection.unlink()

    def create_tables(self) -> None:
        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS "
            f"{self.nodes_collection_name}"
            f"(id, {', '.join(self.position_attributes + self.node_attrs)})"
        )
        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS "
            f"{self.edges_collection_name}({', '.join(['u', 'v'] + self.edge_attrs)})"
        )

    def roi_query(self, roi: Roi) -> str:
        query = "WHERE "
        for dim, pos_attr in enumerate(self.position_attributes):
            if dim > 0:
                query += " AND "
            query += f"{pos_attr} BETWEEN {roi.begin[dim]} and {roi.end[dim]}"
        return query

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
        return self.read_graph(roi)

    def read_nodes(self, roi: Roi) -> list[dict[str, Any]]:
        """Return a list of nodes within roi."""

        logger.debug("Reading nodes in roi %s" % roi)
        select_statement = (
            f"SELECT * FROM {self.nodes_collection_name} "  # + self.roi_query(roi)
        )
        nodes = [
            {
                key: val
                for key, val in zip(
                    ["id"] + self.position_attributes + self.node_attrs, values
                )
            }
            for values in self.cur.execute(select_statement)
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
        self, roi: Roi, nodes: Optional[list[dict[str, Any]]] = None
    ) -> list[dict[str, Any]]:
        """Returns a list of edges within roi."""

        if nodes is None:
            nodes = self.read_nodes(roi)

        if len(nodes) == 0:
            return {}

        node_ids = ", ".join([str(node["id"]) for node in nodes])

        logger.debug("Reading nodes in roi %s" % roi)
        # TODO: AND vs OR here
        select_statement = (
            f"SELECT * FROM {self.edges_collection_name} "
            + f"WHERE u IN ({node_ids}) AND v IN ({node_ids})"
        )

        edges = [
            {key: val for key, val in zip(["u", "v"] + self.edge_attrs, values)}
            for values in self.cur.execute(select_statement)
        ]

        return edges

    def __getitem__(self, roi: Roi):
        return self.get_graph(roi)

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
            # default is false
            self.directed = False
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

    def __get_node_pos(self, n: dict[str, Any]) -> Coordinate:
        return Coordinate(
            (n.get(pos_attr, None) for pos_attr in self.position_attributes)
        )

    def __convert_to_sql(self, x: Any) -> str:
        if isinstance(x, str):
            return f"'{x}'"
        elif x is None:
            return "null"
        else:
            return str(x)

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

        insert_statement = (
            f"INSERT INTO {self.edges_collection_name} "
            f"({', '.join(['u', 'v'] + self.edge_attrs)}) VALUES "
        )
        num_entries = 0

        for u, v, data in edges:
            if roi is not None:
                pos_u = self.__get_node_pos(nodes[u])
                pos_v = self.__get_node_pos(nodes[v])
                if not roi.contains(pos_u) and not roi.contains(pos_v):
                    continue

            if num_entries > 0:
                insert_statement += ", "
            edge_attributes = [nodes[u]["id"], nodes[v]["id"]] + [
                data.get(attr, None) for attr in self.edge_attrs
            ]
            edge_attributes = [self.__convert_to_sql(x) for x in edge_attributes]
            insert_statement += f"({', '.join(edge_attributes)})"
            num_entries += 1

        if num_entries == 0:
            logger.debug("No edges to insert in %s", roi)
            return

        self.cur.execute(insert_statement)
        self.con.commit()

    def write_nodes(
        self,
        nodes: list[dict[str:Any]],
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
            f"INSERT INTO {self.nodes_collection_name} "
            f"(id, {', '.join(self.position_attributes + self.node_attrs)}) VALUES "
        )

        num_entries = 0
        for data in nodes:
            pos = self.__get_node_pos(data)
            if roi is not None and not roi.contains(pos):
                continue
            if num_entries > 0:
                insert_statement += ", "
            id_pos_str = ", ".join(
                self.__convert_to_sql(data.get(attr, None))
                for attr in ["id"] + self.position_attributes + self.node_attrs
            )
            insert_statement += f"({id_pos_str})"
            num_entries += 1

        if num_entries == 0:
            logger.debug("No nodes to insert in %s", roi)
            return

        try:
            self.cur.execute(insert_statement)
        except sqlite3.OperationalError as e:
            raise ValueError(insert_statement) from e
        self.con.commit()

    def __contains(self, roi, node):
        """Determines if the given node is inside the given roi"""
        node_data = self.node[node]

        # Some nodes are outside of the originally requested ROI (they have
        # been pulled in by edges leaving the ROI). These nodes have no
        # attributes, so we can't perform an inclusion test. However, we
        # know they are outside of the subgraph ROI, and therefore also
        # outside of 'roi', whatever it is.
        if "position" not in node_data:
            return False

        return roi.contains(Coordinate(node_data["position"]))

    def is_directed(self):
        raise NotImplementedError("not implemented in %s" % self.name())
