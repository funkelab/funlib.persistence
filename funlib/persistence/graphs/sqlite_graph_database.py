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
        endpoint_names: Optional[list[str]] = None,
        node_attrs: Optional[dict[str, type]] = None,
        edge_attrs: Optional[dict[str, type]] = None,
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
        ] + list(self.node_attrs.keys())
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
        ] + [f"{edge_attr}" for edge_attr in self.edge_attrs.keys()]
        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS {self.edges_table_name}("
            + f"{', '.join(edge_columns)}"
            + f", PRIMARY KEY ({self.endpoint_names[0]}, {self.endpoint_names[1]})"
            + ")"
        )

    def _store_metadata(self, metadata):
        with open(self.meta_collection, "w") as f:
            json.dump(metadata, f)

    def _read_metadata(self) -> Optional[dict[str, Any]]:
        if not self.meta_collection.exists():
            return None

        with open(self.meta_collection, "r") as f:
            return json.load(f)

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
