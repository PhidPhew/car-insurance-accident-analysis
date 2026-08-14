"""Load the exported warehouse (`data/40_exports/`) into a MySQL 8 instance.

Course submission requirement (CLAUDE.md "Submission extras") -- the warehouse
must be openable/runnable outside DuckDB too. Connection settings come from
environment variables (never hard-coded, CLAUDE.md's "never hard-code a
path/credential" spirit):

    RRI_MYSQL_HOST      default "127.0.0.1"
    RRI_MYSQL_PORT      default 3306
    RRI_MYSQL_USER      default "root"
    RRI_MYSQL_PASSWORD  default "" (empty)
    RRI_MYSQL_DATABASE  default "rri_warehouse"

Run: ``uv run python scripts/load_to_mysql.py``

Requires the target MySQL server to have ``local_infile`` enabled (a server
global, not something this script can set) -- run
``SET GLOBAL local_infile = 1;`` as an admin first if loads fail with
"the used command is not allowed".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from rri.utils.logging import configure_logging, get_logger  # noqa: E402
from rri.warehouse.export import TABLE_PRIMARY_KEYS  # noqa: E402

log = get_logger(__name__)


def _connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=os.environ.get("RRI_MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("RRI_MYSQL_PORT", "3306")),
        user=os.environ.get("RRI_MYSQL_USER", "root"),
        password=os.environ.get("RRI_MYSQL_PASSWORD", ""),
        database=os.environ.get("RRI_MYSQL_DATABASE", "rri_warehouse"),
        local_infile=True,
        autocommit=False,
    )


def create_database_if_missing(connection: pymysql.connections.Connection, database: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
        cursor.execute(f"USE `{database}`")


def run_schema(connection: pymysql.connections.Connection, schema_sql_path: Path) -> None:
    statements = schema_sql_path.read_text(encoding="utf-8").split(";")
    with connection.cursor() as cursor:
        for statement in statements:
            cleaned = statement.strip()
            if cleaned and not cleaned.startswith("--"):
                cursor.execute(cleaned)
    connection.commit()
    log.info("mysql_schema_applied", path=str(schema_sql_path))


def load_table_csv(
    connection: pymysql.connections.Connection, table_name: str, csv_path: Path
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(f"TRUNCATE TABLE `{table_name}`")
        cursor.execute(
            f"""
            LOAD DATA LOCAL INFILE '{csv_path}'
            INTO TABLE `{table_name}`
            FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
            LINES TERMINATED BY '\\n'
            IGNORE 1 ROWS
            """
        )
    connection.commit()
    log.info("mysql_table_loaded", table=table_name, csv=str(csv_path))


def main() -> None:
    configure_logging()
    export_dir = settings.exports_dir
    schema_sql_path = export_dir / "schema_mysql.sql"
    if not schema_sql_path.exists():
        raise FileNotFoundError(
            f"{schema_sql_path} not found -- run `uv run python -m rri.warehouse.export` first"
        )

    database = os.environ.get("RRI_MYSQL_DATABASE", "rri_warehouse")
    connection = _connect()
    try:
        create_database_if_missing(connection, database)
        run_schema(connection, schema_sql_path)
        for table_name in TABLE_PRIMARY_KEYS:
            csv_path = export_dir / f"{table_name}.csv"
            if csv_path.exists():
                load_table_csv(connection, table_name, csv_path)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
