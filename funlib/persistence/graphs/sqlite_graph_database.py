from .sql_graph_database import SQLGraphDataBase

from funlib.geometry import Roi

import logging
import sqlite3
import json
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


class SQLiteGraphDataBase(SQLGraphDataBase):
    def __init__(
        self,
        db_file: Path,
        position_attributes: list[str],
        mode: str = "r+",
        directed: Optional[bool] = None,
        total_roi: Optional[Roi] = None,
        nodes_table: str = "nodes",
        edges_table: str = "edges",
        endpoint_names: Optional[tuple[str, str]] = None,
        node_attrs: Optional[list[str]] = None,
        edge_attrs: Optional[list[str]] = None,
    ):
        self.db_file = db_file
        self.meta_collection = self.db_file.parent / f"{self.db_file.stem}-meta.json"
        self.con = sqlite3.connect(db_file)
        self.cur = self.con.cursor()

        super().__init__(
            position_attributes,
            mode=mode,
            directed=directed,
            total_roi=total_roi,
            nodes_table=nodes_table,
            edges_table=edges_table,
            endpoint_names=endpoint_names,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
        )

    def _init_metadata(self):
        if self.meta_collection.exists():
            self.__check_metadata()
        else:
            self.__set_metadata()

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

    def _drop_tables(self) -> None:
        logger.info(
            "dropping collections %s, %s",
            self.nodes_table_name,
            self.edges_table_name,
        )

        self.cur.execute(f"DROP TABLE IF EXISTS {self.nodes_table_name}")
        self.cur.execute(f"DROP TABLE IF EXISTS {self.edges_table_name}")
        self.cur.execute("DROP TABLE IF EXISTS edge_index")

        if self.meta_collection.exists():
            self.meta_collection.unlink()

    def _create_tables(self) -> None:
        position_template = "{pos_attr} REAL not null"
        columns = [
            position_template.format(pos_attr=pos_attr)
            for pos_attr in self.position_attributes
        ] + self.node_attrs
        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS "
            f"{self.nodes_table_name}("
            "id INTEGER not null PRIMARY KEY, "
            f"{', '.join(columns)}"
            ")"
        )
        self.cur.execute(
            f"CREATE INDEX IF NOT EXISTS pos_index ON {self.nodes_table_name}({','.join(self.position_attributes)})"
        )
        edge_columns = [
            f"{self.endpoint_names[0]} INTEGER not null",
            f"{self.endpoint_names[1]} INTEGER not null",
        ] + [f"{edge_attr}" for edge_attr in self.edge_attrs]
        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS {self.edges_table_name}("
            + f"{', '.join(edge_columns)}"
            + f", PRIMARY KEY ({self.endpoint_names[0]}, {self.endpoint_names[1]})"
            + ")"
        )

    def _select_query(self, query):
        try:
            return self.cur.execute(query)
        except sqlite3.OperationalError as e:
            raise ValueError(query) from e

    def _insert_query(self, table, columns, values, fail_if_exists=False, commit=True):
        insert_statement = (
            f"INSERT{' OR IGNORE' if not fail_if_exists else ''} INTO {table} "
            f"({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})"
        )
        self.cur.executemany(insert_statement, values)

        if commit:
            self.con.commit()

    def _update_query(self, query, commit=True):
        try:
            self.cur.execute(query)
        except sqlite3.OperationalError as e:
            raise ValueError(query) from e

        if commit:
            self.con.commit()

    def _commit(self):
        self.con.commit()

    def __get_metadata(self) -> dict[str, Any]:
        """Gets metadata out of the meta collection and returns it
        as a dictionary."""

        with open(self.meta_collection, "r") as f:
            return json.load(f)
