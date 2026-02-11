import logging
from abc import ABC, abstractmethod
from typing import Optional

from networkx import Graph

from funlib.geometry import Roi

from ..types import Vec

logger = logging.getLogger(__name__)


AttributeType = type | str | Vec


class GraphDataBase(ABC):
    """
    Interface for graph databases that supports slicing to retrieve
    and write subgraphs.

    Implementations should support the following interactions::

        # graph_db is a GraphDataBase

        # slicing with ROI to extract a subgraph
        sub_graph = graph_db[Roi((0, 0, 0), (10, 10, 10))]

        # sub_graph is a networkx Graph or DiGraph

        # write graphs
        graph_db.write_graph(sub_graph, Roi((0, 0, 0), (10, 10, 10)))

    """

    def __getitem__(self, roi) -> Graph:
        return self.read_graph(roi)

    @property
    @abstractmethod
    def node_attrs(self) -> dict[str, AttributeType]:
        """
        Return the node attributes supported by the database.
        """
        pass

    @property
    @abstractmethod
    def edge_attrs(self) -> dict[str, AttributeType]:
        """
        Return the edge attributes supported by the database.
        """
        pass

    @abstractmethod
    def read_graph(
        self,
        roi: Optional[Roi] = None,
        read_edges: bool = True,
        node_attrs: Optional[list[str]] = None,
        edge_attrs: Optional[list[str]] = None,
    ) -> Graph:
        """
        Read a graph from the database for a given roi.

        Arguments:

            roi (``Roi`` or ``None``):

                The region of interest to read. If ``None``, read the entire graph.

            read_edges (``bool``):

                If ``True``, read the edges of the graph. If ``False``, only read the nodes.

            node_attrs (``list[str]`` or ``None``):

                If not ``None``, only read the given node attributes.

            edge_attrs (``list[str]`` or ``None``):

                If not ``None``, only read the given edge attributes.

        """
        pass

    @abstractmethod
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
        """
        Write a graph from the database for a given roi.

        Arguments:

            graph (``Graph`` or ``DiGraph``):

                The graph to write.

            roi (``Roi`` or ``None``):

                The region of interest to write to. If ``None``, write the entire graph.

            write_nodes (``bool``):

                If ``True``, write the nodes of the graph. If ``False``, only write the edges.

            write_edges (``bool``):

                If ``True``, write the edges of the graph. If ``False``, only write the nodes.
                Edges will only be written if `u` is contained in the roi.

            node_attrs (``list[str]`` or ``None``):

                If not ``None``, only write the given node attributes.

            edge_attrs (``list[str]`` or ``None``):

                If not ``None``, only write the given edge attributes.

            fail_if_exists (``bool``):

                If ``True``, fail if a node or edge in the graph already exists in the database.

            delete (``bool``):

                If ``True``, delete the graph (possibly within the roi) in the database before writing.

        """
        pass

    @abstractmethod
    def write_attrs(
        self,
        graph: Graph,
        roi: Optional[Roi] = None,
        node_attrs: Optional[list[str]] = None,
        edge_attrs: Optional[list[str]] = None,
    ) -> None:
        """
        Alias call to write_graph with write_nodes and write_edges set to False.
        """
        pass
