"""Unit tests for `rri.insight.facts` (CLAUDE.md Phase 8, Step 1).

Builds a tiny synthetic warehouse in an in-memory DuckDB connection with just
the tables `build_facts` reads (`fact_accident`, `dim_date`, `dim_geography`,
`fact_underwriting_monthly`) rather than running the full warehouse build
pipeline -- keeps these tests fast and focused on the facts contract itself.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from rri.insight.facts import (
    MIN_ACCIDENTS_FOR_RANKING,
    Fact,
    InsightScope,
    build_facts,
)


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")

    connection.execute(
        """
        CREATE TABLE dim_date AS
        SELECT * FROM (VALUES
            (20240101, DATE '2024-01-01'),
            (20240615, DATE '2024-06-15'),
            (20241231, DATE '2024-12-31'),
            (20250101, DATE '2025-01-01'),
            (20250615, DATE '2025-06-15'),
            (20251231, DATE '2025-12-31')
        ) AS t(date_sk, date)
        """
    )

    connection.execute(
        """
        CREATE TABLE dim_geography AS
        SELECT * FROM (VALUES
            (-1, 'Unknown'),
            (1, 'Bangkok'),
            (2, 'Chiang Mai')
        ) AS t(geography_sk, province_en)
        """
    )

    # 2025: 100 accidents/10 fatalities/50 injuries in Bangkok (sk=1).
    # 2024 (prior year): 80 accidents/5 fatalities/40 injuries in Bangkok.
    # Chiang Mai (sk=2) gets a small, below-ranking-floor 2025 sample.
    connection.execute(
        """
        CREATE TABLE fact_accident AS
        SELECT * FROM (VALUES
            (20250615, 1, 100.0, 10.0, 50.0),
            (20240615, 1, 80.0, 5.0, 40.0),
            (20250615, 2, 5.0, 1.0, 2.0)
        ) AS t(date_sk, geography_sk, accident_count, fatality_count, total_injury_count)
        """
    )

    connection.execute(
        """
        CREATE TABLE fact_underwriting_monthly AS
        SELECT * FROM (VALUES
            (20250615, 1, 1000.0, 12000.0, 500000.0, 300000.0),
            (20240615, 1, 900.0, 10800.0, 450000.0, 250000.0)
        ) AS t(date_sk, geography_sk, claim_count, policies_in_force,
                earned_premium, incurred_claim_amount)
        """
    )
    return connection


def test_build_facts_national_scope_has_expected_keys(con: duckdb.DuckDBPyConnection) -> None:
    scope = InsightScope(
        label="Test national 2025", start_date=date(2025, 1, 1), end_date=date(2025, 12, 31)
    )
    payload = build_facts(con, scope)
    keys = {f.key for f in payload.facts}

    assert "accident_count" in keys
    assert "severity_index_coarse" in keys
    assert "accident_count_yoy_pct_change" in keys
    assert "loss_ratio" in keys
    assert "claim_frequency" in keys


def test_accident_count_sums_only_in_scope_window(con: duckdb.DuckDBPyConnection) -> None:
    scope = InsightScope(
        label="Bangkok 2025",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        geography_sk=1,
    )
    payload = build_facts(con, scope)
    accident_count = payload.as_dict()["accident_count"]
    assert accident_count.value == 100.0


def test_severity_index_coarse_matches_hand_computed_value(
    con: duckdb.DuckDBPyConnection,
) -> None:
    from config.severity_index_params import W_F_COARSE, W_I_COARSE

    scope = InsightScope(
        label="Bangkok 2025",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        geography_sk=1,
    )
    payload = build_facts(con, scope)
    expected = (10.0 * W_F_COARSE + 50.0 * W_I_COARSE) / 100.0
    actual = payload.as_dict()["severity_index_coarse"].value
    assert actual == pytest.approx(expected, rel=1e-6)


def test_yoy_pct_change_matches_hand_computed_value(con: duckdb.DuckDBPyConnection) -> None:
    scope = InsightScope(
        label="Bangkok 2025",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        geography_sk=1,
    )
    payload = build_facts(con, scope)
    delta = payload.as_dict()["accident_count_yoy_pct_change"].value
    # 100 (2025) vs 80 (2024) -> +25%
    assert delta == pytest.approx(25.0, rel=1e-6)


def test_top_province_ranking_excludes_below_floor_provinces(
    con: duckdb.DuckDBPyConnection,
) -> None:
    scope = InsightScope(
        label="Test national 2025", start_date=date(2025, 1, 1), end_date=date(2025, 12, 31)
    )
    payload = build_facts(con, scope)
    descriptions = " ".join(f.description for f in payload.facts)
    # Chiang Mai has only 5 accidents, below MIN_ACCIDENTS_FOR_RANKING -- must
    # never appear in a top-risk-province fact even though it's the only
    # other province in the fixture.
    assert MIN_ACCIDENTS_FOR_RANKING > 5
    assert "Chiang Mai" not in descriptions
    assert "Bangkok" in descriptions


def test_province_scope_never_produces_top_province_facts(
    con: duckdb.DuckDBPyConnection,
) -> None:
    scope = InsightScope(
        label="Bangkok 2025",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        geography_sk=1,
    )
    payload = build_facts(con, scope)
    assert not any(f.key.startswith("top_risk_province_") for f in payload.facts)


def test_fact_payload_as_dict_round_trips(con: duckdb.DuckDBPyConnection) -> None:
    scope = InsightScope(
        label="Test national 2025", start_date=date(2025, 1, 1), end_date=date(2025, 12, 31)
    )
    payload = build_facts(con, scope)
    as_dict = payload.as_dict()
    for fact in payload.facts:
        assert as_dict[fact.key] is fact


def test_fact_is_a_typed_record_with_stable_key_value_unit() -> None:
    fact = Fact(key="accident_count", value=42.0, unit="accidents", description="test")
    assert fact.key == "accident_count"
    assert fact.value == 42.0
    assert fact.unit == "accidents"
