from daisy.client import Client
from daisy.task import Task
from daisy import run_blockwise
from funlib.geometry import Roi


from queue import Empty
import multiprocessing
import logging
import time

logger = logging.getLogger(__name__)


class SharedGraphProvider(object):
    """Interface for shared graph providers that supports slicing to retrieve
    subgraphs.

    Implementations should support the following interactions::

        # provider is a SharedGraphProvider

        # slicing with ROI to extract a subgraph
        sub_graph = provider[daisy.Roi((0, 0, 0), (10, 10, 10))]

        # sub_graph should inherit from an implementation of
        SharedSubGraph, and either Graph or DiGraph

        # write nodes
        sub_graph.write_nodes()

        # write edges
        sub_graph.write_edges()
    """

    def __getitem__(self, roi):
        raise RuntimeError("not implemented in %s" % self.name())

    def name(self):
        return type(self).__name__


class SharedSubGraph:
    def write_edges(
        self,
        roi=None,
        attributes=None,
        fail_if_exists=False,
        fail_if_not_exists=False,
        delete=False,
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
        raise RuntimeError("not implemented in %s" % self.name())

    def write_nodes(
        self,
        roi=None,
        attributes=None,
        fail_if_exists=False,
        fail_if_not_exists=False,
        delete=False,
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
        raise RuntimeError("not implemented in %s" % self.name())

    def get_connected_components(self):
        """Returns a list of connected components from the nodes and edges
        in the subgraph. For directed graphs, weak connectivity is sufficient.
        """
        raise RuntimeError("not implemented in %s" % self.name())

    def name(self):
        return type(self).__name__

