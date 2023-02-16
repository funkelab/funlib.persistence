from funlib.persistence.graphs import FileGraphProvider, MongoDbGraphProvider

import pytest
import pymongo


def mongo_db_available():
    client = pymongo.MongoClient(serverSelectionTimeoutMS=1000)
    try:
        client.admin.command("ping")
        return True
    except pymongo.errors.ConnectionFailure:
        return False


@pytest.fixture(
    params=(
        "files",
        pytest.param(
            "mongo",
            marks=pytest.mark.skipif(
                not mongo_db_available(), reason="MongoDB not available!"
            ),
        ),
    )
)
def provider_factory(request, tmpdir):
    # provides a factory function to generate graph provider
    # can provide either mongodb graph provider or file graph provider
    # if file graph provider, will generate graph in a temporary directory
    # to avoid artifacts

    def mongo_provider_factory(mode, directed=None, total_roi=None):
        return MongoDbGraphProvider(
            "test_mongo_graph", mode=mode, directed=directed, total_roi=total_roi
        )

    def file_provider_factory(mode, directed=None, total_roi=None):
        return FileGraphProvider(
            tmpdir / "test_file_graph",
            chunk_size=(10, 10, 10),
            mode=mode,
            directed=directed,
            total_roi=total_roi,
        )

    if request.param == "mongo":
        return mongo_provider_factory
    else:
        return file_provider_factory

