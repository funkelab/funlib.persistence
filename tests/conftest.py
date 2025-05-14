import os
from contextlib import contextmanager
from pathlib import Path

import psycopg2
import pytest
from psycopg2 import OperationalError

from funlib.persistence.graphs import PgSQLGraphDatabase, SQLiteGraphDataBase


def _psql_connect_kwargs():
    """Build psycopg2 connection kwargs from environment variables."""
    kwargs = {"dbname": "pytest"}
    if os.environ.get("PGHOST"):
        kwargs["host"] = os.environ["PGHOST"]
    if os.environ.get("PGUSER"):
        kwargs["user"] = os.environ["PGUSER"]
    if os.environ.get("PGPASSWORD"):
        kwargs["password"] = os.environ["PGPASSWORD"]
    if os.environ.get("PGPORT"):
        kwargs["port"] = int(os.environ["PGPORT"])
    return kwargs


# Attempt to connect to the server (using the default 'postgres' database
# which always exists, since the test database may not exist yet).
def can_connect_to_psql():
    try:
        kwargs = _psql_connect_kwargs()
        kwargs["dbname"] = "postgres"
        conn = psycopg2.connect(**kwargs)
        conn.close()
        return True
    except OperationalError:
        return False


# Conditionally mark the "psql" parameter
psql_param = (
    pytest.param(
        "psql", marks=pytest.mark.skip(reason="Cannot connect to psql database")
    )
    if not can_connect_to_psql()
    else "psql"
)


@pytest.fixture(
    params=(
        pytest.param("sqlite"),
        psql_param,
    )
)
def provider_factory(request, tmp_path):

    @contextmanager
    def sqlite_provider_factory(
        mode, directed=None, total_roi=None, node_attrs=None, edge_attrs=None
    ):
        provider = SQLiteGraphDataBase(
            tmp_path / "test_sqlite_graph.db",
            position_attribute="position",
            mode=mode,
            directed=directed,
            total_roi=total_roi,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
        )
        try:
            yield provider
        finally:
            provider.close()

    @contextmanager
    def psql_provider_factory(
        mode, directed=None, total_roi=None, node_attrs=None, edge_attrs=None
    ):
        connect_kwargs = _psql_connect_kwargs()
        provider = PgSQLGraphDatabase(
            position_attribute="position",
            db_name="pytest",
            db_host=connect_kwargs.get("host", "localhost"),
            db_user=connect_kwargs.get("user"),
            db_password=connect_kwargs.get("password"),
            db_port=connect_kwargs.get("port"),
            mode=mode,
            directed=directed,
            total_roi=total_roi,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
        )
        try:
            yield provider
        finally:
            provider.close()

    if request.param == "sqlite":
        yield sqlite_provider_factory
    elif request.param == "psql":
        yield psql_provider_factory
    else:
        raise ValueError()


@pytest.fixture(params=["standard", "bulk"])
def write_method(request):
    return request.param
