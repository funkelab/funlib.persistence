from funlib.persistence.graphs import SQLiteGraphDataBase, PgSQLGraphDatabase

import pytest
import pymongo

from pathlib import Path


@pytest.fixture(params=(pytest.param("sqlite"), pytest.param("psql")))
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
