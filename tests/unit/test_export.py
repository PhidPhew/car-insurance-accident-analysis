"""Unit tests for the warehouse export module (Phase 4 Step 6)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest

from rri.warehouse.export import export_table_files, generate_mysql_schema


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    df = pl.DataFrame(
        {
            "dim_vehicle_sk": [1, 2],
            "vehicle_group": ["motorcycle", "pickup"],
            "has_normalized_exposure": [True, True],
            "avg_days": [1.5, 2.5],
        }
    )
    connection.register("_df", df)
    connection.execute('CREATE TABLE "dim_vehicle" AS SELECT * FROM _df')
    return connection


class TestExportTableFiles:
    def test_writes_parquet_and_csv(self, con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
        export_table_files(con, "dim_vehicle", tmp_path)
        assert (tmp_path / "dim_vehicle.parquet").exists()
        assert (tmp_path / "dim_vehicle.csv").exists()

        roundtrip = pl.read_parquet(tmp_path / "dim_vehicle.parquet")
        assert roundtrip.height == 2
        assert "vehicle_group" in roundtrip.columns


class TestGenerateMysqlSchema:
    def test_includes_create_table_and_primary_key(self, con: duckdb.DuckDBPyConnection) -> None:
        sql = generate_mysql_schema(con, ["dim_vehicle"])
        assert "CREATE TABLE IF NOT EXISTS `dim_vehicle`" in sql
        assert "`vehicle_group` TEXT" in sql
        assert "`has_normalized_exposure` BOOLEAN" in sql
        assert "`avg_days` DOUBLE" in sql

    def test_no_primary_key_clause_for_unknown_table(self, con: duckdb.DuckDBPyConnection) -> None:
        # dim_vehicle's real PK column is `vehicle_sk`; this fixture table
        # uses `dim_vehicle_sk` instead, which TABLE_PRIMARY_KEYS won't
        # match -- no PRIMARY KEY clause should be emitted rather than one
        # referencing a column that doesn't exist.
        sql = generate_mysql_schema(con, ["dim_vehicle"])
        assert "PRIMARY KEY" not in sql
