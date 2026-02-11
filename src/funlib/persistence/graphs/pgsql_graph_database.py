import json
import logging
from collections.abc import Iterable
from typing import Any, Optional

import psycopg2

from funlib.geometry import Roi

from ..types import Vec
from .sql_graph_database import SQLGraphDataBase

logger = logging.getLogger(__name__)


class PgSQLGraphDatabase(SQLGraphDataBase):
    def __init__(
        self,
        position_attribute: str,
        db_name: str,
        db_host: str = "localhost",
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        db_port: Optional[int] = None,
        mode: str = "r+",
        directed: Optional[bool] = None,
        total_roi: Optional[Roi] = None,
        nodes_table: str = "nodes",
        edges_table: str = "edges",
        endpoint_names: Optional[list[str]] = None,
        node_attrs: Optional[dict[str, type]] = None,
        edge_attrs: Optional[dict[str, type]] = None,
    ):
        self.db_host = db_host
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_port = db_port

        connection = psycopg2.connect(
            host=db_host,
            database="postgres",
            user=db_user,
            password=db_password,
            port=db_port,
        )
        connection.autocommit = True
        cur = connection.cursor()
        try:
            cur.execute(f"CREATE DATABASE {db_name}")
        except psycopg2.errors.DuplicateDatabase:
            # DB already exists, moving on...
            connection.rollback()
            pass
        self.connection = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port,
        )
        # TODO: remove once tests pass:
        # self.connection.autocommit = True
        self.cur = self.connection.cursor()

        super().__init__(
            mode=mode,
            position_attribute=position_attribute,
            directed=directed,
            total_roi=total_roi,
            nodes_table=nodes_table,
            edges_table=edges_table,
            endpoint_names=endpoint_names,
            node_attrs=node_attrs,  # type: ignore
            edge_attrs=edge_attrs,  # type: ignore
        )

    def _drop_edges(self) -> None:
        logger.info("dropping edges table %s", self.edges_table_name)
        self.__exec(f"DROP TABLE IF EXISTS {self.edges_table_name}")
        self._commit()

    def _drop_tables(self) -> None:
        logger.info(
            "dropping tables %s, %s",
            self.nodes_table_name,
            self.edges_table_name,
        )

        self.__exec(f"DROP TABLE IF EXISTS {self.nodes_table_name}")
        self.__exec(f"DROP TABLE IF EXISTS {self.edges_table_name}")
        self.__exec("DROP TABLE IF EXISTS metadata")
        self._commit()

    def _create_tables(self) -> None:
        columns = self.node_attrs.keys()
        types = [self.__sql_type(t) for t in self.node_attrs.values()]
        column_types = [f"{c} {t}" for c, t in zip(columns, types)]
        self.__exec(
            f"CREATE TABLE IF NOT EXISTS "
            f"{self.nodes_table_name}("
            "id BIGINT not null PRIMARY KEY, "
            f"{', '.join(column_types)}"
            ")"
        )
        self.__exec(
            f"CREATE INDEX IF NOT EXISTS pos_index ON "
            f"{self.nodes_table_name}({self.position_attribute})"
        )

        columns = list(self.edge_attrs.keys())  # type: ignore
        types = list([self.__sql_type(t) for t in self.edge_attrs.values()])
        column_types = [f"{c} {t}" for c, t in zip(columns, types)]
        endpoint_names = self.endpoint_names
        assert endpoint_names is not None
        self.__exec(
            f"CREATE TABLE IF NOT EXISTS {self.edges_table_name}("
            f"{endpoint_names[0]} BIGINT not null, "  # type: ignore
            f"{endpoint_names[1]} BIGINT not null, "
            f"{' '.join([c + ',' for c in column_types])}"
            f"PRIMARY KEY ({endpoint_names[0]}, {endpoint_names[1]})"
            ")"
        )

        self._commit()

    def _store_metadata(self, metadata) -> None:
        self.__exec("DROP TABLE IF EXISTS metadata")
        self.__exec("CREATE TABLE metadata (value VARCHAR)")
        self._insert_query(
            "metadata", ["value"], [[json.dumps(metadata)]], fail_if_exists=True
        )

    def _read_metadata(self) -> Optional[dict[str, Any]]:
        try:
            self.__exec("SELECT value FROM metadata")
        except psycopg2.errors.UndefinedTable:
            self.connection.rollback()
            return None

        result = self.cur.fetchone()
        if result is not None:
            metadata = result[0]
            return json.loads(metadata)

        return None

    def _select_query(self, query) -> Iterable[Any]:
        self.__exec(query)
        return self.cur

    def _insert_query(
        self, table, columns, values, fail_if_exists=False, commit=True
    ) -> None:
        values_str = (
            "VALUES ("
            + "), (".join(
                [", ".join([self.__sql_value(v) for v in value]) for value in values]
            )
            + ")"
        )
        # TODO: fail_if_exists is the default if UNIQUE was used to create the
        # table, we need to update if fail_if_exists==False
        insert_statement = f"INSERT INTO {table}({', '.join(columns)}) " + values_str
        self.__exec(insert_statement)

        if commit:
            self.connection.commit()

    def _update_query(self, query, commit=True) -> None:
        self.__exec(query)

        if commit:
            self.connection.commit()

    def _commit(self) -> None:
        self.connection.commit()

    def __exec(self, query):
        try:
            return self.cur.execute(query)
        except:
            self.connection.rollback()
            raise

    def __sql_value(self, value):
        if isinstance(value, str):
            return f"'{value}'"
        if isinstance(value, Iterable):
            return f"array[{','.join([self.__sql_value(v) for v in value])}]"
        elif value is None:
            return "NULL"
        else:
            return str(value)

    def __sql_type(self, type):
        if isinstance(type, Vec):
            return self.__sql_type(type.dtype) + f"[{type.size}]"
        try:
            return {bool: "BOOLEAN", int: "INTEGER", str: "VARCHAR", float: "REAL"}[
                type
            ]
        except ValueError:
            raise NotImplementedError(
                f"attributes of type {type} are not yet supported"
            )
