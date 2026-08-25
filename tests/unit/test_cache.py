"""Unit tests for `rri.insight.cache` (CLAUDE.md Phase 8, Step 4).

Mocks `generate_insight` -- never hits the live API. Uses the same tiny
in-memory DuckDB pattern as `test_facts.py`, extended with an `is_current`
column on `dim_geography` since `_top_province_scopes` filters on it.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import duckdb
import pytest

from rri.insight.cache import (
    ALL_YEARS_END,
    ALL_YEARS_START,
    LATEST_COMPLETE_YEAR_END,
    LATEST_COMPLETE_YEAR_START,
    build_insight_cache,
    load_cache,
    save_cache,
)
from rri.insight.facts import FactsPayload
from rri.insight.generate import CitedFact, GeneratedInsight


@pytest.fixture
def con(tmp_path) -> duckdb.DuckDBPyConnection:  # type: ignore[no-untyped-def]
    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE dim_date AS
        SELECT * FROM (VALUES
            (20190101, DATE '2019-01-01'),
            (20250615, DATE '2025-06-15'),
            (20241231, DATE '2024-12-31'),
            (20251231, DATE '2025-12-31')
        ) AS t(date_sk, date)
        """
    )
    connection.execute(
        """
        CREATE TABLE dim_geography AS
        SELECT * FROM (VALUES
            (-1, 'Unknown', true),
            (1, 'Bangkok', true),
            (2, 'Chiang Mai', true)
        ) AS t(geography_sk, province_en, is_current)
        """
    )
    connection.execute(
        """
        CREATE TABLE fact_accident AS
        SELECT * FROM (VALUES
            (20250615, 1, 100.0, 10.0, 50.0),
            (20250615, 2, 200.0, 5.0, 20.0)
        ) AS t(date_sk, geography_sk, accident_count, fatality_count, total_injury_count)
        """
    )
    connection.execute(
        """
        CREATE TABLE fact_underwriting_monthly AS
        SELECT * FROM (VALUES
            (20250615, 1, 1000.0, 12000.0, 500000.0, 300000.0)
        ) AS t(date_sk, geography_sk, claim_count, policies_in_force,
                earned_premium, incurred_claim_amount)
        """
    )
    return connection


def _fake_generate_insight(
    facts: FactsPayload, api_key: str | None = None, model: str | None = None
) -> GeneratedInsight:
    return GeneratedInsight(
        narrative=f"Fake narrative for {facts.scope_label} with {facts.facts[0].value}.",
        self_reported_citations=[
            CitedFact(fact_key=facts.facts[0].key, value_as_used=str(facts.facts[0].value))
        ],
        model="claude-sonnet-5",
        scope_label=facts.scope_label,
    )


def test_national_scope_windows_match_module_constants() -> None:
    assert date(2025, 1, 1) == LATEST_COMPLETE_YEAR_START
    assert date(2025, 12, 31) == LATEST_COMPLETE_YEAR_END
    assert date(2019, 1, 1) == ALL_YEARS_START
    assert date(2025, 12, 31) == ALL_YEARS_END


def test_build_insight_cache_includes_both_national_scopes(
    con: duckdb.DuckDBPyConnection,
) -> None:
    with patch("rri.insight.cache.generate_insight", side_effect=_fake_generate_insight):
        cache = build_insight_cache(con)

    keys = {e.scope_key for e in cache.entries}
    assert "national_2025" in keys
    assert "national_2019_2025" in keys


def test_build_insight_cache_includes_province_scopes_from_top_risk(
    con: duckdb.DuckDBPyConnection,
) -> None:
    with patch("rri.insight.cache.generate_insight", side_effect=_fake_generate_insight):
        cache = build_insight_cache(con)

    province_entries = [e for e in cache.entries if e.scope_key.startswith("province_")]
    assert len(province_entries) >= 1
    assert all(e.geography_sk is not None for e in province_entries)


def test_build_insight_cache_entries_carry_verification_records(
    con: duckdb.DuckDBPyConnection,
) -> None:
    with patch("rri.insight.cache.generate_insight", side_effect=_fake_generate_insight):
        cache = build_insight_cache(con)

    for entry in cache.entries:
        assert entry.verification.total_numbers >= 0
        assert 0.0 <= entry.verification.hallucination_rate <= 1.0
        assert entry.insight.narrative


def test_save_and_load_cache_round_trips(con: duckdb.DuckDBPyConnection, tmp_path) -> None:  # type: ignore[no-untyped-def]
    with patch("rri.insight.cache.generate_insight", side_effect=_fake_generate_insight):
        cache = build_insight_cache(con)

    path = save_cache(cache, tmp_path / "insights_cache.json")
    loaded = load_cache(path)

    assert loaded.generated_at == cache.generated_at
    assert {e.scope_key for e in loaded.entries} == {e.scope_key for e in cache.entries}
    by_key = loaded.by_scope_key()
    assert by_key["national_2025"].insight.narrative == (
        cache.by_scope_key()["national_2025"].insight.narrative
    )
