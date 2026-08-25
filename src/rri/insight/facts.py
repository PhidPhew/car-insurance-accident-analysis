"""The numbers contract (CLAUDE.md Phase 8, Step 1).

Every number an AI-generated insight narrative is allowed to mention must
come from a :class:`Fact` produced here. Nothing downstream computes a new
number -- `generate.py` only turns these facts into prose, and `verify.py`
checks that the prose said nothing else.

Reads `warehouse.duckdb` directly (not through `dashboard/lib/data_access.py`,
which depends on Streamlit) -- CLAUDE.md N2: all logic lives in `src/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import duckdb
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field

from config.settings import settings
from config.severity_index_params import W_F_COARSE, W_I_COARSE
from rri.utils.logging import get_logger

log = get_logger(__name__)

#: A province-month cell (or a province total) needs at least this many
#: accidents before its severity_index_coarse is trustworthy enough to rank
#: -- same calibration dashboard/lib/data_access.py uses for its alert list,
#: re-derived here rather than imported (that module depends on streamlit).
MIN_ACCIDENTS_FOR_RANKING = 30

#: How many top-ranked provinces to expose as facts for a national scope.
TOP_N_PROVINCES = 3

#: fact_underwriting_monthly's actual availability window (CLAUDE.md §12
#: decision log, 2026-08-14) -- portfolio facts are clipped to this range.
UNDERWRITING_WINDOW_START = date(2022, 1, 1)
UNDERWRITING_WINDOW_END = date(2025, 12, 1)


class Fact(BaseModel):
    """One verifiable number, with a stable key an LLM can cite and a
    verifier can look up again."""

    key: str
    value: float
    unit: str
    description: str


class FactsPayload(BaseModel):
    """The complete, closed set of numbers available for one insight scope."""

    scope_label: str
    facts: list[Fact] = Field(default_factory=list)

    def as_dict(self) -> dict[str, Fact]:
        return {f.key: f for f in self.facts}


@dataclass(frozen=True)
class InsightScope:
    """What an insight narrative is about: a place and a time window.

    ``geography_sk`` of ``None`` means national (every province, Unknown
    Member excluded -- same convention as `dashboard/lib/data_access.py`).
    """

    label: str
    start_date: date
    end_date: date
    geography_sk: int | None = None

    def prior_year(self) -> InsightScope:
        return InsightScope(
            label=f"{self.label} (prior year)",
            start_date=self.start_date - relativedelta(years=1),
            end_date=self.end_date - relativedelta(years=1),
            geography_sk=self.geography_sk,
        )


def get_connection(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    """Open the warehouse read-only -- insight generation never writes to it."""
    resolved = db_path or str(settings.warehouse_db_path)
    return duckdb.connect(resolved, read_only=True)


def _accident_aggregate(
    con: duckdb.DuckDBPyConnection, scope: InsightScope
) -> tuple[float, float, float]:
    geo_filter = "AND f.geography_sk = ?" if scope.geography_sk is not None else ""
    params: list[object] = [scope.start_date, scope.end_date]
    if scope.geography_sk is not None:
        params.append(scope.geography_sk)
    row = con.execute(
        f"""
        SELECT
            COALESCE(SUM(f.accident_count), 0)::DOUBLE AS accident_count,
            COALESCE(SUM(f.fatality_count), 0)::DOUBLE AS fatality_count,
            COALESCE(SUM(f.total_injury_count), 0)::DOUBLE AS total_injury_count
        FROM fact_accident f
        JOIN dim_date d ON f.date_sk = d.date_sk
        WHERE d.date BETWEEN ? AND ?
          {geo_filter}
        """,
        params,
    ).fetchone()
    assert row is not None
    return float(row[0]), float(row[1]), float(row[2])


def _severity_index_coarse(
    fatality_count: float, total_injury_count: float, accident_count: float
) -> float | None:
    if not accident_count:
        return None
    return (fatality_count * W_F_COARSE + total_injury_count * W_I_COARSE) / accident_count


def _portfolio_aggregate(
    con: duckdb.DuckDBPyConnection, scope: InsightScope
) -> tuple[float | None, float | None]:
    """Returns (loss_ratio, claim_frequency), clipped to the underwriting
    simulation's real availability window. None if the scope has no overlap
    with that window."""
    window_start = max(scope.start_date, UNDERWRITING_WINDOW_START)
    window_end = min(scope.end_date, UNDERWRITING_WINDOW_END)
    if window_start > window_end:
        return None, None

    geo_filter = "AND u.geography_sk = ?" if scope.geography_sk is not None else ""
    params: list[object] = [window_start, window_end]
    if scope.geography_sk is not None:
        params.append(scope.geography_sk)
    row = con.execute(
        f"""
        SELECT
            COALESCE(SUM(u.claim_count), 0)::DOUBLE AS claim_count,
            COALESCE(SUM(u.policies_in_force), 0)::DOUBLE AS policy_months,
            COALESCE(SUM(u.incurred_claim_amount), 0)::DOUBLE AS incurred_claim_amount,
            COALESCE(SUM(u.earned_premium), 0)::DOUBLE AS earned_premium
        FROM fact_underwriting_monthly u
        JOIN dim_date d ON u.date_sk = d.date_sk
        WHERE d.date BETWEEN ? AND ?
          {geo_filter}
        """,
        params,
    ).fetchone()
    assert row is not None
    claim_count, policy_months, incurred_claim_amount, earned_premium = row
    policy_years = policy_months / 12 if policy_months else 0.0
    claim_frequency = (claim_count / policy_years) if policy_years else None
    loss_ratio = (incurred_claim_amount / earned_premium) if earned_premium else None
    return loss_ratio, claim_frequency


def _pct_delta(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return (current - prior) / prior * 100


def _top_provinces(con: duckdb.DuckDBPyConnection, scope: InsightScope) -> list[Fact]:
    """Top-N riskiest provinces by severity_index_coarse in scope, subject to
    a minimum-accident-count floor -- only meaningful for a national scope."""
    if scope.geography_sk is not None:
        return []
    df = con.execute(
        """
        SELECT
            g.province_en,
            SUM(f.accident_count)::DOUBLE AS accident_count,
            SUM(f.fatality_count)::DOUBLE AS fatality_count,
            SUM(f.total_injury_count)::DOUBLE AS total_injury_count
        FROM fact_accident f
        JOIN dim_date d ON f.date_sk = d.date_sk
        JOIN dim_geography g ON f.geography_sk = g.geography_sk
        WHERE d.date BETWEEN ? AND ?
          AND g.geography_sk != -1
        GROUP BY 1
        """,
        [scope.start_date, scope.end_date],
    ).fetchdf()
    df["severity_index_coarse"] = (
        df["fatality_count"] * W_F_COARSE + df["total_injury_count"] * W_I_COARSE
    ) / df["accident_count"].replace(0, None)
    eligible = df[df["accident_count"] >= MIN_ACCIDENTS_FOR_RANKING]
    top = eligible.sort_values("severity_index_coarse", ascending=False).head(TOP_N_PROVINCES)

    facts: list[Fact] = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        province_en = str(row["province_en"])
        accident_count = float(row["accident_count"])
        severity_index = float(row["severity_index_coarse"])
        facts.append(
            Fact(
                key=f"top_risk_province_{rank}_severity_index",
                value=round(severity_index, 4),
                unit="severity index (weighted fatalities+injuries per accident)",
                description=(
                    f"Rank {rank} riskiest province by severity_index_coarse in the "
                    f"scope period: {province_en} (based on {int(accident_count)} accidents)"
                ),
            )
        )
        facts.append(
            Fact(
                key=f"top_risk_province_{rank}_accident_count",
                value=accident_count,
                unit="accidents",
                description=f"Accident count for the rank {rank} riskiest province, {province_en}",
            )
        )
    return facts


def build_facts(con: duckdb.DuckDBPyConnection, scope: InsightScope) -> FactsPayload:
    """Build the complete, closed set of facts for one insight scope."""
    accident_count, fatality_count, total_injury_count = _accident_aggregate(con, scope)
    severity_index = _severity_index_coarse(fatality_count, total_injury_count, accident_count)

    prior = scope.prior_year()
    prior_accident_count, prior_fatality_count, prior_total_injury_count = _accident_aggregate(
        con, prior
    )
    prior_severity_index = _severity_index_coarse(
        prior_fatality_count, prior_total_injury_count, prior_accident_count
    )

    loss_ratio, claim_frequency = _portfolio_aggregate(con, scope)

    facts: list[Fact] = [
        Fact(
            key="accident_count",
            value=accident_count,
            unit="accidents",
            description=(
                f"Total accidents in scope ({scope.label}, {scope.start_date} to {scope.end_date})"
            ),
        ),
        Fact(
            key="fatality_count",
            value=fatality_count,
            unit="fatalities",
            description=f"Total fatalities in scope ({scope.label})",
        ),
        Fact(
            key="total_injury_count",
            value=total_injury_count,
            unit="injuries",
            description=f"Total injuries (all severities) in scope ({scope.label})",
        ),
    ]
    if severity_index is not None:
        facts.append(
            Fact(
                key="severity_index_coarse",
                value=round(severity_index, 4),
                unit="severity index (weighted fatalities+injuries per accident)",
                description=(
                    "severity_index_coarse for the scope period: "
                    "(fatalities * W_F_COARSE + total_injuries * W_I_COARSE) / accidents"
                ),
            )
        )

    accident_count_delta = _pct_delta(accident_count, prior_accident_count)
    if accident_count_delta is not None:
        facts.append(
            Fact(
                key="accident_count_yoy_pct_change",
                value=round(accident_count_delta, 2),
                unit="percent",
                description=(
                    "Year-over-year percent change in accident count vs. the same "
                    "period one year earlier"
                ),
            )
        )
    severity_index_delta = _pct_delta(severity_index, prior_severity_index)
    if severity_index_delta is not None:
        facts.append(
            Fact(
                key="severity_index_coarse_yoy_pct_change",
                value=round(severity_index_delta, 2),
                unit="percent",
                description=(
                    "Year-over-year percent change in severity_index_coarse vs. "
                    "the same period one year earlier"
                ),
            )
        )

    if loss_ratio is not None:
        facts.append(
            Fact(
                key="loss_ratio",
                value=round(loss_ratio * 100, 2),
                unit="percent",
                description=(
                    "SiamShield simulated-portfolio loss ratio (incurred claims / earned "
                    "premium) over the underwriting simulation's available window intersected "
                    "with the scope period"
                ),
            )
        )
    if claim_frequency is not None:
        facts.append(
            Fact(
                key="claim_frequency",
                value=round(claim_frequency, 4),
                unit="claims per policy-year",
                description="SiamShield simulated-portfolio claim frequency over the same window",
            )
        )

    facts.extend(_top_provinces(con, scope))

    log.info(
        "insight.facts.built",
        scope=scope.label,
        n_facts=len(facts),
        start_date=str(scope.start_date),
        end_date=str(scope.end_date),
    )
    return FactsPayload(scope_label=scope.label, facts=facts)
