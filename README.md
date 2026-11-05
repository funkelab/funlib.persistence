[![tests](https://github.com/funkelab/funlib.persistence/actions/workflows/tests.yaml/badge.svg)](https://github.com/funkelab/funlib.persistence/actions/workflows/tests.yaml)
[![lint](https://github.com/funkelab/funlib.persistence/actions/workflows/lint.yaml/badge.svg)](https://github.com/funkelab/funlib.persistence/actions/workflows/lint.yaml)
[![publish](https://github.com/funkelab/funlib.persistence/actions/workflows/publish.yaml/badge.svg)](https://pypi.org/project/funlib.persistence/)

# funlib.persistence
Interfaces for data (arrays and graphs) and storage formats (databases and file formats)

## Install

```bash
pip install funlib.persistence    # or: uv add funlib.persistence
```

## Contributing

The commands below are the ones CI runs, so a green local run means a green PR.

```bash
git clone https://github.com/funkelab/funlib.persistence
cd funlib.persistence
uv sync                                             # Python, dependencies, dev tools (from uv.lock)
uv run pytest                                       # tests with coverage
uv run pytest --no-cov tests/<file>.py              # one file, without the coverage floor
uv run ruff check . && uv run ruff format --check . # lint and formatting
uv run ty check src tests                           # types
```

The graph-database tests need a PostgreSQL server (CI starts one as a service, `PGUSER=postgres`); they are skipped when none is reachable.

`uv.lock` pins the development environment; the `lock-upgrade` workflow refreshes it monthly and opens a pull request.
