import io
import json
import logging
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any, Optional

import psycopg2
from funlib.geometry import Roi
from psycopg2 import sql

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
        connection.close()
        self.connection = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port,
        )
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

    def close(self):
        if not self.connection.closed:
            self.connection.close()

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
        if not values:
            return

        values_str = (
            "VALUES ("
            + "), (".join(
                [", ".join([self.__sql_value(v) for v in value]) for value in values]
            )
            + ")"
        )
        insert_statement = f"INSERT INTO {table}({', '.join(columns)}) " + values_str
        if not fail_if_exists:
            insert_statement += " ON CONFLICT DO NOTHING"
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
            return {bool: "BOOLEAN", int: "BIGINT", str: "VARCHAR", float: "REAL"}[type]
        except ValueError:
            raise NotImplementedError(
                f"attributes of type {type} are not yet supported"
            )

    def print_summary(self, schema="public", limit=None):
        self._commit()

        def fmt_bytes(n):
            for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
                if n < 1024 or unit == "PB":
                    return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
                n /= 1024

        # Basic DB info (read-only)
        self.cur.execute("SELECT current_database(), current_user, version();")
        db, user, version = self.cur.fetchone()
        print(f"DB: {db} | User: {user}")
        print(version.split("\n")[0])
        print("-" * 80)

        q = sql.SQL(
            """
            SELECT
            c.relname AS table_name,
            c.relkind,
            COALESCE(c.reltuples::bigint, 0) AS est_rows,
            pg_total_relation_size(c.oid) AS total_bytes
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s
            AND c.relkind IN ('r','p')  -- r=table, p=partitioned table
            ORDER BY pg_total_relation_size(c.oid) DESC, c.relname
        """
        )
        if limit is not None:
            q = q + sql.SQL(" LIMIT %s")
            self.cur.execute(q, (schema, limit))
        else:
            self.cur.execute(q, (schema,))
        rows = self.cur.fetchall()
        if not rows:
            print(f"No tables found in schema={schema!r}")
            return

        # Pretty print
        name_w = min(max(len(r[0]) for r in rows), 60)
        header = f"{'table':<{name_w}}  {'kind':<4}  {'est_rows':>12}  {'size':>10}"
        print(header)
        print("-" * len(header))

        kind_map = {"r": "tbl", "p": "part"}
        for name, relkind, est_rows, total_bytes in rows:
            print(
                f"{name:<{name_w}}  {kind_map.get(relkind, relkind):<4}  {est_rows:>12,}  {fmt_bytes(total_bytes):>10}"
            )

        print("-" * 80)
        print(f"Tables shown: {len(rows)} (schema={schema!r})")

    @contextmanager
    def bulk_write_mode(self, worker=False, node_writes=True, edge_writes=True):
        nodes = self.nodes_table_name
        edges = self.edges_table_name
        endpoint_names = self.endpoint_names
        assert endpoint_names is not None

        if not worker:
            if node_writes:
                self.__exec("DROP INDEX IF EXISTS pos_index")
                self.__exec(
                    f"ALTER TABLE {nodes} DROP CONSTRAINT IF EXISTS {nodes}_pkey"
                )
            if edge_writes:
                self.__exec(
                    f"ALTER TABLE {edges} DROP CONSTRAINT IF EXISTS {edges}_pkey"
                )
            self._commit()

        self.__exec("SET synchronous_commit TO OFF")
        self._commit()

        try:
            yield
        finally:
            self.__exec("SET synchronous_commit TO ON")
            self._commit()

            if not worker:
                logger.info("Re-creating indexes and constraints...")
                if node_writes:
                    self.__exec(
                        f"ALTER TABLE {nodes} "
                        f"ADD CONSTRAINT {nodes}_pkey PRIMARY KEY (id)"
                    )
                    self.__exec(
                        f"CREATE INDEX IF NOT EXISTS pos_index ON "
                        f"{nodes}({self.position_attribute})"
                    )
                if edge_writes:
                    self.__exec(
                        f"ALTER TABLE {edges} "
                        f"ADD CONSTRAINT {edges}_pkey "
                        f"PRIMARY KEY ({endpoint_names[0]}, {endpoint_names[1]})"
                    )
                self._commit()

    def _bulk_insert(self, table, columns, rows) -> None:
        def format_gen():
            for row in rows:
                formatted = []
                for val in row:
                    if val is None:
                        formatted.append(r"\N")
                    elif isinstance(val, (list, tuple)):
                        formatted.append(f"{{{','.join(map(str, val))}}}")
                    else:
                        formatted.append(str(val))
                yield "\t".join(formatted) + "\n"

        self._stream_copy(table, columns, format_gen())
        self._commit()

    def _stream_copy(self, table_name, columns, data_generator):
        """
        Consumes a generator of strings and sends them to Postgres via COPY.
        Uses a chunked buffer to keep memory usage stable.
        """
        # Tune this size (in bytes). 10MB - 50MB is usually a sweet spot.
        BATCH_SIZE = 50 * 1024 * 1024

        buffer = io.StringIO()
        current_size = 0

        # Helper to flush buffer to DB
        def flush():
            buffer.seek(0)
            self.cur.copy_from(buffer, table_name, columns=columns, null=r"\N")
            buffer.truncate(0)
            buffer.seek(0)

        for line in data_generator:
            buffer.write(line)
            current_size += len(line)

            if current_size >= BATCH_SIZE:
                flush()
                current_size = 0

        # Flush remaining
        if current_size > 0:
            flush()
