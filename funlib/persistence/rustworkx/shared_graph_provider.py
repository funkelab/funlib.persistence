from funlib.geometry import Roi

from rustworkx import PyGraph

from abc import ABC, abstractmethod
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class SharedGraphProvider(ABC):
    """Interface for shared graph providers that supports slicing to retrieve
    subgraphs.

    Implementations should support the following interactions::

        # provider is a SharedGraphProvider

        # slicing with ROI to extract a subgraph
        sub_graph = provider[Roi((0, 0, 0), (10, 10, 10))]

        # sub_graph should inherit from an implementation of
        SharedSubGraph, and either Graph or DiGraph

        # write nodes
        sub_graph.write_nodes()

        # write edges
        sub_graph.write_edges()
    """

    @abstractmethod
    def __getitem__(self, roi: Optional[Roi] = None) -> PyGraph:
        pass

    @abstractmethod
    def read_nodes(self, roi: Optional[Roi] = None) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def read_edges(self, roi: Optional[Roi] = None) -> list[dict[str, Any]]:
        pass

    def read_graph(self, roi: Optional[Roi] = None) -> PyGraph:
        nodes = self.read_nodes(roi)
        edges = self.read_edges(roi)
        graph = PyGraph()
        node_ids = graph.add_nodes_from(nodes)
        node_mapping = {
            node_attrs["id"]: node_id for node_id, node_attrs in zip(node_ids, nodes)
        }
        # TODO: What to do with edges whose endpoints are not in the roi
        graph.add_edges_from(
            [
                (
                    node_mapping[edge_attrs["u"]],
                    node_mapping[edge_attrs["v"]],
                    edge_attrs,
                )
                for edge_attrs in edges
            ]
        )
        return graph

    @abstractmethod
    def write_edges(
        self,
        edges,
        roi: Optional[Roi] = None,
        attributes: Optional[list[str]] = None,
        fail_if_exists: bool = False,
        fail_if_not_exists: bool = False,
        delete: bool = False,
    ):
        """Write edges and their attributes.
        Args:
            roi(`class:Roi`):
                Restrict the write to the given ROI

            attributes(`class:list`):
                Only write the given attributes. If None, write all attributes.

            fail_if_exists:
                If true, throw error if edge with same u,v already exists
                in back end.

            fail_if_not_exists:
                If true, throw error if edge with same u,v does not already
                exist in back end.

            delete:
                If true, delete edges in ROI in back end that do not exist
                in subgraph.

        """
        pass

    @abstractmethod
    def write_nodes(
        self,
        nodes,
        roi: Optional[Roi] = None,
        attributes: Optional[list[str]] = None,
        fail_if_exists: bool = False,
        fail_if_not_exists: bool = False,
        delete: bool = False,
    ):
        """Write nodes and their attributes.
        Args:
            roi(`class:Roi`):
                Restrict the write to the given ROI

            attributes(`class:list`):
                Only write the given attributes. If None, write all attributes.

            fail_if_exists:
                If true, throw error if node with same id already exists in
                back end, while still performing all other valid writes.

            fail_if_not_exists:
                If true, throw error if node with same id does not already
                exist in back end, while still performing all other
                valid writes.

            delete:
                If true, delete nodes in ROI in back end that do not exist
                in subgraph.

        """
        pass

    def write_graph(self, graph: PyGraph, roi: Optional[Roi] = None):
        nodes = graph.nodes()
        edge_list = graph.edge_list()
        edges = graph.edges()
        self.write_nodes(nodes, roi)
        self.write_edges(nodes, list(zip(*zip(*edge_list), edges)), roi)


class SubGraph:
    def __init__(self, graph: PyGraph, graph_provider: SharedGraphProvider):
        self.__graph = graph
        self.__graph_provider = graph_provider

    def __getattr__(self, attr):
        return getattr(self.graph, attr)

    @property
    def graph_provider(self) -> SharedGraphProvider:
        return self.__graph_provider

    @property
    def graph(self) -> PyGraph:
        return self.__graph
