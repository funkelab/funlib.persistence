# API

## Arrays

`Array`, `open_ds`, `prepare_ds`, `open_ome_ds` and `prepare_ome_ds` are
importable from `funlib.persistence`.

```{eval-rst}
.. autoclass:: funlib.persistence.arrays.array.Array
   :members:
   :undoc-members:
   :special-members: __getitem__, __setitem__

.. autofunction:: funlib.persistence.arrays.datasets.open_ds

.. autofunction:: funlib.persistence.arrays.datasets.prepare_ds

.. autofunction:: funlib.persistence.arrays.ome_datasets.open_ome_ds

.. autofunction:: funlib.persistence.arrays.ome_datasets.prepare_ome_ds
```

## Metadata

```{eval-rst}
.. autopydantic_model:: funlib.persistence.arrays.metadata.MetaDataFormat
   :members:

.. autofunction:: funlib.persistence.arrays.metadata.read_config

.. autofunction:: funlib.persistence.arrays.metadata.configure_library

.. autofunction:: funlib.persistence.arrays.metadata.get_default_metadata_format

.. autofunction:: funlib.persistence.arrays.metadata.set_default_metadata_format

.. autoclass:: funlib.persistence.types.Vec
   :members:
```

## Graphs

The graph databases are importable from `funlib.persistence.graphs`.

```{eval-rst}
.. autoclass:: funlib.persistence.graphs.graph_database.GraphDataBase
   :members:
   :special-members: __getitem__

.. autoclass:: funlib.persistence.graphs.sql_graph_database.SQLGraphDataBase
   :members:
   :show-inheritance:

.. autoclass:: funlib.persistence.graphs.sqlite_graph_database.SQLiteGraphDataBase
   :members:
   :show-inheritance:

.. autoclass:: funlib.persistence.graphs.pgsql_graph_database.PgSQLGraphDatabase
   :members:
   :show-inheritance:
```
