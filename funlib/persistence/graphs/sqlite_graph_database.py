from .sql_graph_database import SQLGraphDataBase, AttributeType
from ..types import Vec

from funlib.geometry import Roi

import logging
import sqlite3
import json
import re
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


class SQLiteGraphDataBase(SQLGraphDataBase):
    def __init__(
        self,
        db_file: Path,
        position_attribute: str,
        mode: str = "r+",
        directed: Optional[bool] = None,
        total_roi: Optional[Roi] = None,
        nodes_table: str = "nodes",
        edges_table: str = "edges",
        endpoint_names: Optional[list[str]] = None,
        node_attrs: Optional[dict[str, AttributeType]] = None,
        edge_attrs: Optional[dict[str, AttributeType]] = None,
    ):
        self.db_file = db_file
        self.meta_collection = self.db_file.parent / f"{self.db_file.stem}-meta.json"
        self.con = sqlite3.connect(db_file)
        self.cur = self.con.cursor()

        super().__init__(
            mode=mode,
            position_attribute=position_attribute,
            directed=directed,
            total_roi=total_roi,
            nodes_table=nodes_table,
            edges_table=edges_table,
            endpoint_names=endpoint_names,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
        )

        # in SQLite, array types are stored in individual columns
        self.node_array_columns = {
            attr: [
                f"{attr}_{d}" for d in range(attr_type.size)
            ]
            for attr, attr_type in self.node_attrs.items()
            if isinstance(attr_type, Vec)
        }
        self.edge_array_columns = {
            attr: [
                f"{attr}_{d}" for d in range(attr_type.size)
            ]
            for attr, attr_type in self.edge_attrs.items()
            if isinstance(attr_type, Vec)
        }

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
        columns = list(self.node_attrs.keys())
        columns.remove(self.position_attribute)
        position_columns = [
            self.position_attribute + f"_{d}"
            for d in range(self.ndims)
        ]
        columns += [ p + " REAL NOT NULL" for p in position_columns ]
        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS "
            f"{self.nodes_table_name}("
            "id INTEGER not null PRIMARY KEY, "
            f"{', '.join(columns)}"
            ")"
        )
        self.cur.execute(
            f"CREATE INDEX IF NOT EXISTS pos_index ON {self.nodes_table_name}({','.join(position_columns)})"
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

        # replace array_name[1] with array_name_0
        #                   ^^^
        #             Yes, that's not a typo
        #
        # If SQL dialects allow array element access, they start counting at 1.
        # We don't want that, we start counting at 0 like normal people.
        query = re.sub(r'\[(\d+)\]', lambda m: "_" + str(int(m.group(1)) - 1), query)

        try:
            return self.cur.execute(query)
        except sqlite3.OperationalError as e:
            raise ValueError(query) from e

    def _insert_query(self, table, columns, values, fail_if_exists=False, commit=True):

        # explode array attributes into multiple columns

        exploded_values = []
        for row in values:
            exploded_columns = []
            exploded_row_values = []
            for column, value in zip(columns, row):
                if column in self.node_array_columns:
                    for c, v in zip(self.node_array_columns[column], value):
                        exploded_columns.append(c)
                        exploded_row_values.append(v)
                else:
                    exploded_columns.append(column)
                    exploded_row_values.append(value)
            exploded_values.append(exploded_row_values)

        columns = exploded_columns
        values = exploded_values

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

    def _node_attrs_to_columns(self, attrs):
        columns = []
        for attr in attrs:
            attr_type = self.node_attrs[attr]
            if isinstance(attr_type, Vec):
                columns += [
                    f"{attr}_{d}" for d in range(attr_type.size)
                ]
            else:
                columns.append(attr)
        return columns

    def _columns_to_node_attrs(self, columns, query_attrs):
        attrs = {}
        for attr in query_attrs:
            if attr in self.node_array_columns:
                value = tuple(
                    columns[f"{attr}_{d}"]
                    for d in range(self.node_attrs[attr].size)
                )
            else:
                value = columns[attr]
            attrs[attr] = value
        return attrs

    def _edge_attrs_to_columns(self, attrs):
        columns = []
        for attr in attrs:
            attr_type = self.edge_attrs[attr]
            if isinstance(attr_type, Vec):
                columns += [
                    f"{attr}_{d}" for d in range(attr_type.size)
                ]
            else:
                columns.append(attr)
        return columns

    def _columns_to_edge_attrs(self, columns, query_attrs):
        attrs = {}
        for attr in query_attrs:
            if attr in self.edge_array_columns:
                value = tuple(
                    columns[f"{attr}_{d}"]
                    for d in range(self.edge_attrs[attr].size)
                )
            else:
                value = columns[attr]
            attrs[attr] = value
        return attrs
