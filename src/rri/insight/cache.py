"""Fixed-scope insight cache (CLAUDE.md Phase 8, Step 4).

Generates AI insight narratives + verification records for a fixed,
documented set of scopes and writes both to one committed JSON file
(`data/60_insights/insights_cache.json`) so the dashboard can show a real
audit result -- narrative and verification badge together -- without
calling the Anthropic API at serve time. Streamlit Community Cloud has no
persistent storage and no key baked into the public repo, so the default
path is always this cache; a live "regenerate" is opt-in (see
`dashboard/lib/insights.py`).

Fixed scope set:
  - ``national_2025``       -- national overview, the latest complete year
  - ``national_2019_2025``  -- national overview, the full multi-year window
  - ``province_<slug>_2025`` -- one per province the national_2025 scope's
    own top-risk-province facts surfaced (so the cache lines up with what
    the Executive Overview alert list and Spatial Risk ranking already show)
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
from pydantic import BaseModel

from config.settings import settings
from rri.insight.facts import FactsPayload, InsightScope, build_facts
from rri.insight.generate import GeneratedInsight, generate_insight
from rri.insight.verify import VerificationRecord, log_verification, verify_insight
from rri.utils.logging import get_logger

log = get_logger(__name__)

#: 2026 is a P7.11 incomplete-year holdout (~4 months of data) -- excluded
#: here for the same reason the dashboard's own date filter defaults its
#: trend view to end at 2025-12-31.
LATEST_COMPLETE_YEAR_START = date(2025, 1, 1)
LATEST_COMPLETE_YEAR_END = date(2025, 12, 31)

#: 2019 is the project's first year (CLAUDE.md §9: kept deliberately, the
#: only pre-COVID baseline) through the same complete-year cutoff.
ALL_YEARS_START = date(2019, 1, 1)
ALL_YEARS_END = date(2025, 12, 31)

CACHE_PATH = settings.insights_dir / "insights_cache.json"


class CachedInsight(BaseModel):
    """One fixed scope's generated narrative + its verification record,
    ready for the dashboard to render directly -- no re-verification at
    serve time."""

    scope_key: str
    scope_label: str
    geography_sk: int | None
    start_date: date
    end_date: date
    generated_at: str
    insight: GeneratedInsight
    verification: VerificationRecord


class InsightCache(BaseModel):
    generated_at: str
    entries: list[CachedInsight]

    def by_scope_key(self) -> dict[str, CachedInsight]:
        return {e.scope_key: e for e in self.entries}


def _national_scopes() -> list[tuple[str, InsightScope]]:
    return [
        (
            "national_2025",
            InsightScope(
                label="National overview, 2025",
                start_date=LATEST_COMPLETE_YEAR_START,
                end_date=LATEST_COMPLETE_YEAR_END,
            ),
        ),
        (
            "national_2019_2025",
            InsightScope(
                label="National overview, 2019-2025",
                start_date=ALL_YEARS_START,
                end_date=ALL_YEARS_END,
            ),
        ),
    ]


def _province_name_from_top_risk_fact(description: str) -> str:
    """`facts.py`'s top-risk-province description reads:
    "Rank {N} riskiest province by severity_index_coarse in the scope
    period: {province_en} (based on {count} accidents)" -- pull the name
    back out rather than adding a second, redundant field to Fact just for
    this one caller."""
    after_colon = description.split(": ", 1)[1]
    return after_colon.split(" (based on")[0]


def _top_province_scopes(
    con: duckdb.DuckDBPyConnection, national_facts: FactsPayload
) -> list[tuple[str, InsightScope]]:
    """One province-level scope per province the national_2025 scope's own
    top-risk-province facts named -- keeps the cache aligned with whatever
    the dashboard's own alert list / ranking already surfaces, rather than a
    hand-picked list that can drift from the live data."""
    province_names = [
        _province_name_from_top_risk_fact(f.description)
        for f in national_facts.facts
        if f.key.startswith("top_risk_province_") and f.key.endswith("_severity_index")
    ]
    scopes: list[tuple[str, InsightScope]] = []
    for name in province_names:
        row = con.execute(
            "SELECT geography_sk FROM dim_geography WHERE province_en = ? AND is_current",
            [name],
        ).fetchone()
        if row is None:
            log.warning("insight.cache.province_not_found", province=name)
            continue
        slug = name.lower().replace(" ", "_")
        scopes.append(
            (
                f"province_{slug}_2025",
                InsightScope(
                    label=f"{name}, 2025",
                    start_date=LATEST_COMPLETE_YEAR_START,
                    end_date=LATEST_COMPLETE_YEAR_END,
                    geography_sk=int(row[0]),
                ),
            )
        )
    return scopes


def _generate_and_verify(
    scope_key: str,
    scope: InsightScope,
    facts: FactsPayload,
    api_key: str | None,
) -> CachedInsight:
    insight = generate_insight(facts, api_key=api_key)
    verification = verify_insight(insight, facts)
    log_verification(verification)
    log.info(
        "insight.cache.entry_built",
        scope_key=scope_key,
        hallucination_rate=round(verification.hallucination_rate, 4),
    )
    return CachedInsight(
        scope_key=scope_key,
        scope_label=scope.label,
        geography_sk=scope.geography_sk,
        start_date=scope.start_date,
        end_date=scope.end_date,
        generated_at=datetime.now(UTC).isoformat(),
        insight=insight,
        verification=verification,
    )


def build_insight_cache(con: duckdb.DuckDBPyConnection, api_key: str | None = None) -> InsightCache:
    """Generate + verify insights for the fixed scope set, in order."""
    entries: list[CachedInsight] = []
    facts_by_key: dict[str, FactsPayload] = {}

    for scope_key, scope in _national_scopes():
        facts = build_facts(con, scope)
        facts_by_key[scope_key] = facts
        entries.append(_generate_and_verify(scope_key, scope, facts, api_key))

    for scope_key, scope in _top_province_scopes(con, facts_by_key["national_2025"]):
        facts = build_facts(con, scope)
        entries.append(_generate_and_verify(scope_key, scope, facts, api_key))

    return InsightCache(generated_at=datetime.now(UTC).isoformat(), entries=entries)


def save_cache(cache: InsightCache, path: Path | None = None) -> Path:
    resolved = path or CACHE_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(cache.model_dump_json(indent=2), encoding="utf-8")
    return resolved


def load_cache(path: Path | None = None) -> InsightCache:
    resolved = path or CACHE_PATH
    return InsightCache.model_validate_json(resolved.read_text(encoding="utf-8"))
