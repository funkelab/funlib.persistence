"""PostgreSQL-specific tests for opening graph databases."""

import uuid

import psycopg2
import pytest

from funlib.persistence.graphs import PgSQLGraphDatabase
from funlib.persistence.types import Vec

from .conftest import _psql_connect_kwargs, can_connect_to_psql

pytestmark = pytest.mark.skipif(
    not can_connect_to_psql(), reason="Cannot connect to psql database"
)


def _admin_cursor():
    kwargs = _psql_connect_kwargs()
    kwargs["dbname"] = "postgres"
    connection = psycopg2.connect(**kwargs)
    connection.autocommit = True
    return connection, connection.cursor()


def _open(db_name, mode, **kwargs):
    connect_kwargs = _psql_connect_kwargs()
    kwargs.setdefault("db_user", connect_kwargs.get("user"))
    kwargs.setdefault("db_password", connect_kwargs.get("password"))
    return PgSQLGraphDatabase(
        position_attribute="position",
        db_name=db_name,
        db_host=connect_kwargs.get("host", "localhost"),
        db_port=connect_kwargs.get("port"),
        mode=mode,
        node_attrs={"position": Vec(float, 3)} if mode == "w" else None,
        **kwargs,
    )


def test_open_existing_db_without_createdb_privilege():
    connection, cur = _admin_cursor()
    cur.execute(
        "SELECT rolsuper OR rolcreaterole FROM pg_roles WHERE rolname = current_user"
    )
    if not cur.fetchone()[0]:
        connection.close()
        pytest.skip("test needs a role that can create roles")

    suffix = uuid.uuid4().hex[:8]
    db_name = f"pytest_existing_{suffix}"
    role = f"pytest_reader_{suffix}"
    password = uuid.uuid4().hex
    try:
        _open(db_name, "w").close()
        cur.execute(f"CREATE ROLE {role} LOGIN NOCREATEDB PASSWORD %s", (password,))
        graph_db_kwargs = _psql_connect_kwargs()
        graph_db_kwargs["dbname"] = db_name
        graph_db = psycopg2.connect(**graph_db_kwargs)
        with graph_db, graph_db.cursor() as graph_cur:
            graph_cur.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {role}")
        graph_db.close()

        provider = _open(db_name, "r", db_user=role, db_password=password)
        try:
            graph = provider.read_graph()
            assert graph.number_of_nodes() == 0
        finally:
            provider.close()
    finally:
        cur.execute(f"DROP DATABASE IF EXISTS {db_name}")
        cur.execute(f"DROP ROLE IF EXISTS {role}")
        connection.close()
