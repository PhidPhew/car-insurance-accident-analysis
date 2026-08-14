"""Unit tests for the ER diagram generator (Phase 4 Step 6)."""

from __future__ import annotations

import duckdb
import polars as pl
import pytest
from scripts.generate_er_diagram import generate_er_diagram_markdown


@pytest.fixture
def warehouse_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    dims = {
        "dim_date": pl.DataFrame(
            {"date_sk": [1], "date": [None]}, schema={"date_sk": pl.Int32, "date": pl.Date}
        ),
        "dim_geography": pl.DataFrame({"geography_sk": [1], "province_th": ["x"]}),
        "dim_vehicle": pl.DataFrame({"vehicle_sk": [1], "vehicle_group": ["motorcycle"]}),
        "dim_accident_context": pl.DataFrame(
            {"accident_context_sk": [1], "cause_group": ["speeding"]}
        ),
        "dim_time_of_day": pl.DataFrame({"time_of_day_sk": [1], "hour": [0]}),
        "dim_weather_band": pl.DataFrame({"weather_band_sk": [1], "weather_band": ["clear"]}),
        "dim_policy_type": pl.DataFrame({"policy_type_sk": [1], "cover_class": ["class_1"]}),
        "dim_customer_segment": pl.DataFrame({"customer_segment_sk": [1], "age_band": ["18-25"]}),
        "fact_accident": pl.DataFrame({"accident_sk": [1], "date_sk": [1]}),
        "fact_underwriting_monthly": pl.DataFrame({"underwriting_sk": [1], "date_sk": [1]}),
    }
    for table_name, df in dims.items():
        con.register("_df", df)
        con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM _df')
    return con


class TestGenerateErDiagramMarkdown:
    def test_contains_mermaid_block(self, warehouse_con: duckdb.DuckDBPyConnection) -> None:
        markdown = generate_er_diagram_markdown(warehouse_con)
        assert "```mermaid" in markdown
        assert "erDiagram" in markdown

    def test_every_table_has_an_entity_block(
        self, warehouse_con: duckdb.DuckDBPyConnection
    ) -> None:
        markdown = generate_er_diagram_markdown(warehouse_con)
        for table_name in [
            "dim_date",
            "dim_geography",
            "dim_vehicle",
            "dim_accident_context",
            "dim_time_of_day",
            "dim_weather_band",
            "dim_policy_type",
            "dim_customer_segment",
            "fact_accident",
            "fact_underwriting_monthly",
        ]:
            assert f"{table_name} {{" in markdown

    def test_primary_keys_marked(self, warehouse_con: duckdb.DuckDBPyConnection) -> None:
        markdown = generate_er_diagram_markdown(warehouse_con)
        assert "date_sk PK" in markdown
        assert "accident_sk PK" in markdown

    def test_every_relationship_present(self, warehouse_con: duckdb.DuckDBPyConnection) -> None:
        markdown = generate_er_diagram_markdown(warehouse_con)
        assert 'dim_date ||--o{ fact_accident : "date_sk"' in markdown
        assert 'dim_policy_type ||--o{ fact_underwriting_monthly : "policy_type_sk"' in markdown
