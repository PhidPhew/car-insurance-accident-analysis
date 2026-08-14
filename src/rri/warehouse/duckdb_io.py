"""DuckDB connection + table-write helpers shared by every dim/fact builder.

Kept deliberately thin: builders return a finished ``pl.DataFrame`` and this
module's only job is getting that frame into ``warehouse.duckdb`` as a table,
so builder modules stay testable without touching a database at all.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from config.settings import settings
from rri.utils.logging import get_logger

log = get_logger(__name__)


def get_connection(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the warehouse DuckDB file."""
    resolved = db_path or settings.warehouse_db_path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(resolved))


def write_table(con: duckdb.DuckDBPyConnection, table_name: str, df: pl.DataFrame) -> None:
    """Create (or replace) ``table_name`` in ``con`` from a polars DataFrame.

    Always ``CREATE OR REPLACE`` -- the warehouse build is idempotent (N9):
    re-running it against the same staging inputs must produce byte-identical
    tables, not append to what's already there.
    """
    con.register("_incoming_df", df)
    con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _incoming_df')
    con.unregister("_incoming_df")
    log.info("wrote_table", table=table_name, rows=df.height, cols=df.width)
