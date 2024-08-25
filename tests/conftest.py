from pathlib import Path

import psycopg2
import pytest
from psycopg2 import OperationalError

from funlib.persistence.graphs import PgSQLGraphDatabase, SQLiteGraphDataBase


# Attempt to connect to the default database
def can_connect_to_psql():
    try:
        conn = psycopg2.connect(
            dbname="pytest",
        )
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
def provider_factory(request, tmpdir):
    # provides a factory function to generate graph provider
    # can provide either mongodb graph provider or file graph provider
    # if file graph provider, will generate graph in a temporary directory
    # to avoid artifacts

    tmpdir = Path(tmpdir)

    def sqlite_provider_factory(
        mode, directed=None, total_roi=None, node_attrs=None, edge_attrs=None
    ):
        return SQLiteGraphDataBase(
            tmpdir / "test_sqlite_graph.db",
            position_attribute="position",
            mode=mode,
            directed=directed,
            total_roi=total_roi,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
        )

    def psql_provider_factory(
        mode, directed=None, total_roi=None, node_attrs=None, edge_attrs=None
    ):
        return PgSQLGraphDatabase(
            position_attribute="position",
            db_name="pytest",
            mode=mode,
            directed=directed,
            total_roi=total_roi,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
        )

    if request.param == "sqlite":
        yield sqlite_provider_factory
    elif request.param == "psql":
        yield psql_provider_factory
    else:
        raise ValueError()
