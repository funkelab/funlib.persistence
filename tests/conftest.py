from funlib.persistence.graphs import (
    FileGraphProvider,
    MongoDbGraphProvider,
    SQLiteGraphProvider,
)

import pytest
import pymongo

from pathlib import Path


def mongo_db_available():
    client = pymongo.MongoClient(serverSelectionTimeoutMS=1000)
    try:
        client.admin.command("ping")
        return True
    except pymongo.errors.ConnectionFailure:
        return False


@pytest.fixture(
    params=(
        pytest.param(
            "files",
            marks=pytest.mark.xfail(reason="FileProvider not fully implemented!"),
        ),
        pytest.param(
            "mongo",
            marks=pytest.mark.skipif(
                not mongo_db_available(), reason="MongoDB not available!"
            ),
        ),
        pytest.param("sqlite"),
    )
)
def provider_factory(request, tmpdir):
    # provides a factory function to generate graph provider
    # can provide either mongodb graph provider or file graph provider
    # if file graph provider, will generate graph in a temporary directory
    # to avoid artifacts

    tmpdir = Path(tmpdir)

    def mongo_provider_factory(
        mode, directed=None, total_roi=None, node_attrs=None, edge_attrs=None
    ):
        return MongoDbGraphProvider(
            "test_mongo_graph", mode=mode, directed=directed, total_roi=total_roi
        )

    def file_provider_factory(
        mode, directed=None, total_roi=None, node_attrs=None, edge_attrs=None
    ):
        return FileGraphProvider(
            tmpdir / "test_file_graph.db",
            chunk_size=(10, 10, 10),
            mode=mode,
            directed=directed,
            total_roi=total_roi,
            # node_attrs=node_attrs,
            # edge_attrs=edge_attrs,
        )

    def sqlite_provider_factory(
        mode, directed=None, total_roi=None, node_attrs=None, edge_attrs=None
    ):
        return SQLiteGraphProvider(
            tmpdir / "test_sqlite_graph.db",
            mode=mode,
            directed=directed,
            total_roi=total_roi,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
        )

    if request.param == "mongo":
        yield mongo_provider_factory
    elif request.param == "sqlite":
        yield sqlite_provider_factory
    elif request.param == "files":
        yield file_provider_factory
    else:
        raise ValueError()
