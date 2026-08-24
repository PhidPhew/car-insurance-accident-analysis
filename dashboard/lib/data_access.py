"""Read-only warehouse access for the dashboard (CLAUDE.md Phase 7 rule:
dashboard code reads from `warehouse.duckdb` and never recomputes cleaning,
features, or training).

Every KPI here is aggregated straight from `fact_accident` /
`fact_underwriting_monthly` at query time — `severity_index_coarse` reuses
the exact weights `rri.features.severity_index` was calibrated with
(CLAUDE.md §9), and `loss_ratio`/`frequency`/`pure_premium` are computed
here rather than stored, per CLAUDE.md §8 ("storing a ratio in a fact table
breaks roll-up arithmetic").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import duckdb
import pandas as pd
import streamlit as st
import structlog
from dateutil.relativedelta import relativedelta

from config.settings import settings
from config.severity_index_params import W_F_COARSE, W_I_COARSE

logger = structlog.get_logger(__name__)

#: Portfolio simulation only exists for this window (CLAUDE.md §12 decision
#: log, 2026-08-14) -- the exposure back-cast's growth-curve source has no
#: data before 2022-01, and 2026 is a P7.11 incomplete-year holdout.
UNDERWRITING_WINDOW_START = date(2022, 1, 1)
UNDERWRITING_WINDOW_END = date(2025, 12, 1)

#: 2026 is a P7.11 incomplete-year holdout ("exclude from trend analysis;
#: use as holdout") -- the date filter defaults to the last complete year so
#: trend/YoY comparisons aren't skewed by a partial year. Users can still
#: extend the filter into 2026 manually.
DEFAULT_TREND_END = date(2025, 12, 31)

#: Alert-list calibration: a province needs at least this many accidents in
#: the selected window to be eligible for the "over threshold" flag, so a
#: tiny sample (e.g. 3 accidents, 1 severe) can't produce a noisy spike.
#: Flag threshold is mean + 1 standard deviation of severity_index_coarse
#: across eligible provinces in the current filter scope -- a relative,
#: filter-scoped bar rather than an invented absolute cutoff.
MIN_ACCIDENTS_FOR_ALERT = 30
ALERT_STD_MULTIPLIER = 1.0

#: Cleaned rainfall staging output (Phase 2) -- see .gitignore for why this is
#: committed. Not a warehouse table (CLAUDE.md §8's star schema has no rainfall
#: dimension, a real design gap); read directly here since it's already-cleaned
#: pipeline output, not raw data and not a dashboard-side recomputation.
RAINFALL_PATH = settings.staging_dir / "rainfall.parquet"

AGENCY_LABELS = {
    "กรมทางหลวง": "Highways",
    "กรมทางหลวงชนบท": "Rural Roads",
    "การทางพิเศษแห่งประเทศไทย": "Expressway",
}


@st.cache_resource(show_spinner=False)
def get_connection() -> duckdb.DuckDBPyConnection:
    logger.info("dashboard.warehouse.connect", path=str(settings.warehouse_db_path))
    return duckdb.connect(str(settings.warehouse_db_path), read_only=True)


@dataclass(frozen=True)
class Filters:
    start_date: date
    end_date: date
    geography_sks: tuple[int, ...]
    vehicle_sks: tuple[int, ...]
    agencies: tuple[str, ...]

    def prior_year(self) -> Filters:
        return Filters(
            start_date=self.start_date - relativedelta(years=1),
            end_date=self.end_date - relativedelta(years=1),
            geography_sks=self.geography_sks,
            vehicle_sks=self.vehicle_sks,
            agencies=self.agencies,
        )

    def underwriting_window(self) -> tuple[date, date] | None:
        """Clip to the simulated portfolio's actual availability. Returns
        None if the filter's range has no overlap with it."""
        month_floor = date(self.start_date.year, self.start_date.month, 1)
        start = max(month_floor, UNDERWRITING_WINDOW_START)
        end = min(self.end_date, UNDERWRITING_WINDOW_END)
        if start > end:
            return None
        return start, end


@st.cache_data(show_spinner=False)
def list_provinces() -> pd.DataFrame:
    con = get_connection()
    return con.execute(
        "SELECT geography_sk, province_en, province_th, region "
        "FROM dim_geography ORDER BY (geography_sk = -1), province_en"
    ).fetchdf()


@st.cache_data(show_spinner=False)
def list_vehicle_groups() -> pd.DataFrame:
    con = get_connection()
    return con.execute(
        "SELECT vehicle_sk, vehicle_group, vehicle_group_label "
        "FROM dim_vehicle ORDER BY (vehicle_sk = -1), vehicle_group_label"
    ).fetchdf()


@st.cache_data(show_spinner=False)
def list_agencies() -> list[str]:
    con = get_connection()
    rows = con.execute("SELECT DISTINCT agency FROM fact_accident ORDER BY 1").fetchall()
    return [r[0] for r in rows]


@st.cache_data(show_spinner=False)
def accident_date_bounds() -> tuple[date, date]:
    con = get_connection()
    row = con.execute(
        "SELECT MIN(d.date), MAX(d.date) FROM fact_accident f "
        "JOIN dim_date d ON f.date_sk = d.date_sk WHERE d.date_sk != -1"
    ).fetchone()
    assert row is not None
    return row[0], row[1]


def _accident_aggregate(con: duckdb.DuckDBPyConnection, f: Filters) -> dict[str, float]:
    row = con.execute(
        """
        SELECT
            COALESCE(SUM(f.accident_count), 0)::DOUBLE AS accident_count,
            COALESCE(SUM(f.fatality_count), 0)::DOUBLE AS fatality_count,
            COALESCE(SUM(f.total_injury_count), 0)::DOUBLE AS total_injury_count
        FROM fact_accident f
        JOIN dim_date d ON f.date_sk = d.date_sk
        WHERE d.date BETWEEN ? AND ?
          AND f.geography_sk = ANY(?)
          AND f.vehicle_sk = ANY(?)
          AND f.agency = ANY(?)
        """,
        [f.start_date, f.end_date, list(f.geography_sks), list(f.vehicle_sks), list(f.agencies)],
    ).fetchone()
    assert row is not None
    accident_count, fatality_count, total_injury_count = row
    severity_index_coarse = (
        (fatality_count * W_F_COARSE + total_injury_count * W_I_COARSE) / accident_count
        if accident_count
        else None
    )
    fatalities_per_1000 = (fatality_count / accident_count * 1000) if accident_count else None
    return {
        "accident_count": accident_count,
        "fatality_count": fatality_count,
        "severity_index_coarse": severity_index_coarse,
        "fatalities_per_1000_accidents": fatalities_per_1000,
    }


def _portfolio_aggregate(con: duckdb.DuckDBPyConnection, f: Filters) -> dict[str, float | None]:
    window = f.underwriting_window()
    if window is None:
        return {
            "claim_frequency": None,
            "pure_premium": None,
            "loss_ratio": None,
        }
    start, end = window
    row = con.execute(
        """
        SELECT
            COALESCE(SUM(f.claim_count), 0)::DOUBLE AS claim_count,
            COALESCE(SUM(f.policies_in_force), 0)::DOUBLE AS policy_months,
            COALESCE(SUM(f.incurred_claim_amount), 0)::DOUBLE AS incurred_claim_amount,
            COALESCE(SUM(f.earned_premium), 0)::DOUBLE AS earned_premium
        FROM fact_underwriting_monthly f
        JOIN dim_date d ON f.date_sk = d.date_sk
        WHERE d.date BETWEEN ? AND ?
          AND f.geography_sk = ANY(?)
          AND f.vehicle_sk = ANY(?)
        """,
        [start, end, list(f.geography_sks), list(f.vehicle_sks)],
    ).fetchone()
    assert row is not None
    claim_count, policy_months, incurred_claim_amount, earned_premium = row
    # `policies_in_force` is SiamShield's own simulated book (§7.3); `claim_count`
    # was generated per-row as policies_in_force * CLAIM_FREQUENCY/12 (build
    # module docstring), so summing policies_in_force across the selected months
    # gives the matching policy-months denominator -- NOT `exposure_vehicle_months`,
    # which is the whole province's real fleet and deliberately not the basis for
    # claim_count (fact_underwriting_monthly.py: "informational... NOT the basis").
    # Verified: SUM(claim_count)/(SUM(policies_in_force)/12) = 0.0597 over the full
    # table, recovering the calibrated CLAIM_FREQUENCY=0.06 almost exactly.
    policy_years = policy_months / 12 if policy_months else 0.0
    return {
        "claim_frequency": (claim_count / policy_years) if policy_years else None,
        "pure_premium": (incurred_claim_amount / policy_years) if policy_years else None,
        "loss_ratio": (incurred_claim_amount / earned_premium) if earned_premium else None,
    }


def _pct_delta(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return (current - prior) / prior * 100


@st.cache_data(show_spinner=False)
def kpi_summary(f: Filters) -> dict[str, dict[str, float | None]]:
    """Current-period KPIs + year-over-year deltas, all traced to the warehouse."""
    con = get_connection()
    prior_year_filters = f.prior_year()
    current = {**_accident_aggregate(con, f), **_portfolio_aggregate(con, f)}
    prior = {
        **_accident_aggregate(con, prior_year_filters),
        **_portfolio_aggregate(con, prior_year_filters),
    }
    deltas = {k: _pct_delta(current[k], prior.get(k)) for k in current}
    return {"current": current, "prior": prior, "delta_pct": deltas}


@st.cache_data(show_spinner=False)
def monthly_trend(f: Filters) -> pd.DataFrame:
    con = get_connection()
    df = con.execute(
        """
        SELECT
            date_trunc('month', d.date) AS month,
            SUM(f.accident_count) AS accident_count,
            SUM(f.fatality_count) AS fatality_count,
            SUM(f.total_injury_count) AS total_injury_count
        FROM fact_accident f
        JOIN dim_date d ON f.date_sk = d.date_sk
        WHERE d.date BETWEEN ? AND ?
          AND f.geography_sk = ANY(?)
          AND f.vehicle_sk = ANY(?)
          AND f.agency = ANY(?)
        GROUP BY 1
        ORDER BY 1
        """,
        [f.start_date, f.end_date, list(f.geography_sks), list(f.vehicle_sks), list(f.agencies)],
    ).fetchdf()
    df["severity_index_coarse"] = (
        df["fatality_count"] * W_F_COARSE + df["total_injury_count"] * W_I_COARSE
    ) / df["accident_count"].replace(0, pd.NA)
    return df


#: BQ1 "riskiest province x month" needs its own (smaller) sample-size floor
#: than the province-level alert list -- a single province-month cell has far
#: fewer accidents than a province's full-range total, so MIN_ACCIDENTS_FOR_ALERT
#: (calibrated for the province-level view) would exclude nearly every cell.
MIN_ACCIDENTS_FOR_PROVINCE_MONTH = 5


@st.cache_data(show_spinner=False)
def province_summary(f: Filters) -> pd.DataFrame:
    """One row per province: accident/severity aggregates joined with
    dim_geography's tourism and flood/rain attributes (BQ1, BQ4).

    `centroid_lat`/`centroid_lon` average the province's own valid accident
    coordinates -- there is no committed province-boundary GeoJSON in this
    project, so the map (page 2) uses these data-derived centroids for a
    bubble map rather than a true polygon choropleth (documented there)."""
    con = get_connection()
    df = con.execute(
        """
        SELECT
            g.geography_sk,
            g.province_en,
            g.province_th,
            g.region,
            g.tourist_attraction_count,
            g.flood_pct_high_risk_tambons,
            g.r35mm_avg_days_2014_2023,
            SUM(f.accident_count)::DOUBLE AS accident_count,
            SUM(f.fatality_count)::DOUBLE AS fatality_count,
            SUM(f.total_injury_count)::DOUBLE AS total_injury_count,
            AVG(CASE WHEN f.is_geo_valid THEN f.latitude END) AS centroid_lat,
            AVG(CASE WHEN f.is_geo_valid THEN f.longitude END) AS centroid_lon
        FROM fact_accident f
        JOIN dim_date d ON f.date_sk = d.date_sk
        JOIN dim_geography g ON f.geography_sk = g.geography_sk
        WHERE d.date BETWEEN ? AND ?
          AND f.geography_sk = ANY(?)
          AND f.vehicle_sk = ANY(?)
          AND f.agency = ANY(?)
          AND g.geography_sk != -1
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        """,
        [f.start_date, f.end_date, list(f.geography_sks), list(f.vehicle_sks), list(f.agencies)],
    ).fetchdf()
    df["severity_index_coarse"] = (
        df["fatality_count"] * W_F_COARSE + df["total_injury_count"] * W_I_COARSE
    ) / df["accident_count"].replace(0, pd.NA)
    return df


@st.cache_data(show_spinner=False)
def province_exposure_adjusted_rate(f: Filters) -> pd.DataFrame:
    """Accidents per 1,000 exposed vehicles, by province -- the exposure-adjusted
    view for the spatial risk map. Uses fact_underwriting_monthly's real-fleet
    `exposure_vehicle_months` (not SiamShield's own book), clipped to its actual
    availability window (2022-01 to 2025-12) intersected with the filter range."""
    window = f.underwriting_window()
    if window is None:
        return pd.DataFrame(columns=["geography_sk", "accidents_per_1000_vehicles"])
    start, end = window
    con = get_connection()
    return con.execute(
        """
        WITH exposure AS (
            SELECT u.geography_sk, SUM(u.exposure_vehicle_months) / 12 AS vehicle_years
            FROM fact_underwriting_monthly u
            JOIN dim_date d ON u.date_sk = d.date_sk
            WHERE d.date BETWEEN ? AND ?
              AND u.geography_sk = ANY(?)
              AND u.vehicle_sk = ANY(?)
            GROUP BY 1
        ),
        accidents AS (
            SELECT f.geography_sk, SUM(f.accident_count) AS accident_count
            FROM fact_accident f
            JOIN dim_date d ON f.date_sk = d.date_sk
            WHERE d.date BETWEEN ? AND ?
              AND f.geography_sk = ANY(?)
              AND f.vehicle_sk = ANY(?)
              AND f.agency = ANY(?)
            GROUP BY 1
        )
        SELECT
            e.geography_sk,
            (a.accident_count::DOUBLE / NULLIF(e.vehicle_years, 0)) * 1000
                AS accidents_per_1000_vehicles
        FROM exposure e
        JOIN accidents a ON e.geography_sk = a.geography_sk
        """,
        [
            start,
            end,
            list(f.geography_sks),
            list(f.vehicle_sks),
            start,
            end,
            list(f.geography_sks),
            list(f.vehicle_sks),
            list(f.agencies),
        ],
    ).fetchdf()


@st.cache_data(show_spinner=False)
def accident_points(f: Filters, geography_sk: int, limit: int = 2000) -> pd.DataFrame:
    """Individual accident coordinates for one province -- the BQ1 drill-down
    from province-level aggregates down to actual accident locations."""
    con = get_connection()
    return con.execute(
        """
        SELECT
            f.latitude,
            f.longitude,
            d.date,
            f.fatality_count,
            f.total_injury_count,
            v.vehicle_group_label,
            ac.cause_group
        FROM fact_accident f
        JOIN dim_date d ON f.date_sk = d.date_sk
        JOIN dim_vehicle v ON f.vehicle_sk = v.vehicle_sk
        JOIN dim_accident_context ac ON f.accident_context_sk = ac.accident_context_sk
        WHERE f.geography_sk = ?
          AND f.is_geo_valid
          AND d.date BETWEEN ? AND ?
          AND f.vehicle_sk = ANY(?)
          AND f.agency = ANY(?)
        ORDER BY d.date DESC
        LIMIT ?
        """,
        [
            geography_sk,
            f.start_date,
            f.end_date,
            list(f.vehicle_sks),
            list(f.agencies),
            limit,
        ],
    ).fetchdf()


@st.cache_data(show_spinner=False)
def top_province_months(f: Filters, limit: int = 15) -> pd.DataFrame:
    """Riskiest province x month cells by severity_index_coarse, restricted to
    cells with enough accidents to be meaningful (BQ1: "is it driven by
    frequency or severity?")."""
    con = get_connection()
    df = con.execute(
        """
        SELECT
            g.province_en,
            g.region,
            date_trunc('month', d.date) AS month,
            SUM(f.accident_count)::DOUBLE AS accident_count,
            SUM(f.fatality_count)::DOUBLE AS fatality_count,
            SUM(f.total_injury_count)::DOUBLE AS total_injury_count
        FROM fact_accident f
        JOIN dim_date d ON f.date_sk = d.date_sk
        JOIN dim_geography g ON f.geography_sk = g.geography_sk
        WHERE d.date BETWEEN ? AND ?
          AND f.geography_sk = ANY(?)
          AND f.vehicle_sk = ANY(?)
          AND f.agency = ANY(?)
          AND g.geography_sk != -1
        GROUP BY 1, 2, 3
        """,
        [f.start_date, f.end_date, list(f.geography_sks), list(f.vehicle_sks), list(f.agencies)],
    ).fetchdf()
    df["severity_index_coarse"] = (
        df["fatality_count"] * W_F_COARSE + df["total_injury_count"] * W_I_COARSE
    ) / df["accident_count"].replace(0, pd.NA)
    eligible = df[df["accident_count"] >= MIN_ACCIDENTS_FOR_PROVINCE_MONTH]
    return eligible.sort_values("severity_index_coarse", ascending=False).head(limit)


@dataclass(frozen=True)
class ProvinceAlerts:
    threshold: float | None
    flagged: pd.DataFrame


@st.cache_data(show_spinner=False)
def provinces_over_threshold(f: Filters) -> ProvinceAlerts:
    con = get_connection()
    df = con.execute(
        """
        SELECT
            g.geography_sk,
            g.province_en,
            g.region,
            SUM(f.accident_count)::DOUBLE AS accident_count,
            SUM(f.fatality_count)::DOUBLE AS fatality_count,
            SUM(f.total_injury_count)::DOUBLE AS total_injury_count
        FROM fact_accident f
        JOIN dim_date d ON f.date_sk = d.date_sk
        JOIN dim_geography g ON f.geography_sk = g.geography_sk
        WHERE d.date BETWEEN ? AND ?
          AND f.geography_sk = ANY(?)
          AND f.vehicle_sk = ANY(?)
          AND f.agency = ANY(?)
          AND g.geography_sk != -1
        GROUP BY 1, 2, 3
        """,
        [f.start_date, f.end_date, list(f.geography_sks), list(f.vehicle_sks), list(f.agencies)],
    ).fetchdf()

    df["severity_index_coarse"] = (
        df["fatality_count"] * W_F_COARSE + df["total_injury_count"] * W_I_COARSE
    ) / df["accident_count"].replace(0, pd.NA)

    eligible = df[df["accident_count"] >= MIN_ACCIDENTS_FOR_ALERT]
    if eligible.empty or eligible["severity_index_coarse"].std(ddof=0) == 0:
        return ProvinceAlerts(threshold=None, flagged=eligible.iloc[0:0])

    mean = eligible["severity_index_coarse"].mean()
    std = eligible["severity_index_coarse"].std(ddof=0)
    threshold = mean + ALERT_STD_MULTIPLIER * std
    flagged = eligible[eligible["severity_index_coarse"] > threshold].sort_values(
        "severity_index_coarse", ascending=False
    )
    return ProvinceAlerts(threshold=threshold, flagged=flagged)


@st.cache_data(show_spinner=False)
def rainfall_vs_accidents(f: Filters) -> pd.DataFrame:
    """One row per province x month in scope: accident_count, severity_index_coarse,
    and that month's rain_lead0 (nowcast rainfall, lead-0 of the forecast pivot) --
    BQ2's rainfall-vs-frequency view. Joined by (province, year, month), the same
    known-future convention `rri.features.frequency_severity` established."""
    con = get_connection()
    df = con.execute(
        """
        WITH accidents AS (
            SELECT
                g.province_id, g.province_en, g.region,
                d.year, d.month,
                SUM(f.accident_count)::DOUBLE AS accident_count,
                SUM(f.fatality_count)::DOUBLE AS fatality_count,
                SUM(f.total_injury_count)::DOUBLE AS total_injury_count
            FROM fact_accident f
            JOIN dim_date d ON f.date_sk = d.date_sk
            JOIN dim_geography g ON f.geography_sk = g.geography_sk
            WHERE d.date BETWEEN ? AND ?
              AND f.geography_sk = ANY(?)
              AND f.vehicle_sk = ANY(?)
              AND f.agency = ANY(?)
              AND g.geography_sk != -1
            GROUP BY 1, 2, 3, 4, 5
        ),
        rain AS (
            SELECT
                Province_ID AS province_id,
                CAST(Initial_Year AS INTEGER) AS year,
                CAST(Initial_Month AS INTEGER) AS month,
                rain_lead0
            FROM read_parquet(?)
        )
        SELECT a.*, r.rain_lead0
        FROM accidents a
        JOIN rain r ON a.province_id = r.province_id AND a.year = r.year AND a.month = r.month
        """,
        [
            f.start_date,
            f.end_date,
            list(f.geography_sks),
            list(f.vehicle_sks),
            list(f.agencies),
            str(RAINFALL_PATH),
        ],
    ).fetchdf()
    df["severity_index_coarse"] = (
        df["fatality_count"] * W_F_COARSE + df["total_injury_count"] * W_I_COARSE
    ) / df["accident_count"].replace(0, pd.NA)
    return df


@st.cache_data(show_spinner=False)
def weather_band_vs_forecast_rain(f: Filters) -> pd.DataFrame:
    """Average forecast rainfall (rain_lead0, for the accident's own province+month)
    grouped by the REPORTED weather condition at accident time -- a cross-check of
    whether the rainfall forecast and ground-truth accident reports agree
    directionally (BQ2)."""
    con = get_connection()
    return con.execute(
        """
        WITH acc AS (
            SELECT
                w.weather_band, w.is_hazardous,
                g.province_id, d.year, d.month
            FROM fact_accident f
            JOIN dim_date d ON f.date_sk = d.date_sk
            JOIN dim_geography g ON f.geography_sk = g.geography_sk
            JOIN dim_weather_band w ON f.weather_band_sk = w.weather_band_sk
            WHERE d.date BETWEEN ? AND ?
              AND f.geography_sk = ANY(?)
              AND f.vehicle_sk = ANY(?)
              AND f.agency = ANY(?)
              AND g.geography_sk != -1
              AND w.weather_band_sk != -1
        ),
        rain AS (
            SELECT
                Province_ID AS province_id,
                CAST(Initial_Year AS INTEGER) AS year,
                CAST(Initial_Month AS INTEGER) AS month,
                rain_lead0
            FROM read_parquet(?)
        )
        SELECT
            acc.weather_band,
            acc.is_hazardous,
            AVG(rain.rain_lead0) AS avg_forecast_rain_mm,
            COUNT(*)::DOUBLE AS accident_count
        FROM acc
        JOIN rain ON acc.province_id = rain.province_id
            AND acc.year = rain.year AND acc.month = rain.month
        GROUP BY 1, 2
        ORDER BY avg_forecast_rain_mm DESC
        """,
        [
            f.start_date,
            f.end_date,
            list(f.geography_sks),
            list(f.vehicle_sks),
            list(f.agencies),
            str(RAINFALL_PATH),
        ],
    ).fetchdf()


@st.cache_data(show_spinner=False)
def monthly_trend_with_calendar(f: Filters) -> pd.DataFrame:
    """Monthly accident trend (like `monthly_trend`) plus calendar overlay flags
    for BQ3 (Songkran/New Year "7 dangerous days" windows, COVID disruption) --
    joined from dim_date's own boolean columns, not recomputed."""
    con = get_connection()
    df = con.execute(
        """
        SELECT
            date_trunc('month', d.date) AS month,
            SUM(f.accident_count)::DOUBLE AS accident_count,
            SUM(f.fatality_count)::DOUBLE AS fatality_count,
            SUM(f.total_injury_count)::DOUBLE AS total_injury_count,
            bool_or(d.is_songkran_7days) AS has_songkran,
            bool_or(d.is_newyear_7days) AS has_newyear,
            max(CASE d.covid_disruption_level
                WHEN 'severe' THEN 2 WHEN 'partial' THEN 1 ELSE 0 END) AS covid_level
        FROM fact_accident f
        JOIN dim_date d ON f.date_sk = d.date_sk
        WHERE d.date BETWEEN ? AND ?
          AND f.geography_sk = ANY(?)
          AND f.vehicle_sk = ANY(?)
          AND f.agency = ANY(?)
        GROUP BY 1
        ORDER BY 1
        """,
        [f.start_date, f.end_date, list(f.geography_sks), list(f.vehicle_sks), list(f.agencies)],
    ).fetchdf()
    df["severity_index_coarse"] = (
        df["fatality_count"] * W_F_COARSE + df["total_injury_count"] * W_I_COARSE
    ) / df["accident_count"].replace(0, pd.NA)
    return df


@st.cache_data(show_spinner=False)
def holiday_vs_normal_comparison(f: Filters) -> pd.DataFrame:
    """Accident count and severity, split by holiday-period type (BQ3):
    Songkran 7 days, New Year 7 days, vs. every other day in scope."""
    con = get_connection()
    return con.execute(
        """
        SELECT
            CASE
                WHEN d.is_songkran_7days THEN 'Songkran (7 days)'
                WHEN d.is_newyear_7days THEN 'New Year (7 days)'
                ELSE 'Other days'
            END AS period,
            COUNT(*)::DOUBLE AS accident_count,
            SUM(f.fatality_count)::DOUBLE AS fatality_count,
            SUM(f.total_injury_count)::DOUBLE AS total_injury_count,
            COUNT(DISTINCT d.date)::DOUBLE AS n_days
        FROM fact_accident f
        JOIN dim_date d ON f.date_sk = d.date_sk
        WHERE d.date BETWEEN ? AND ?
          AND f.geography_sk = ANY(?)
          AND f.vehicle_sk = ANY(?)
          AND f.agency = ANY(?)
          AND d.date_sk != -1
        GROUP BY 1
        """,
        [f.start_date, f.end_date, list(f.geography_sks), list(f.vehicle_sks), list(f.agencies)],
    ).fetchdf()


@st.cache_data(show_spinner=False)
def portfolio_segment_matrix(f: Filters) -> pd.DataFrame:
    """Vehicle group x region matrix (BQ5): loss ratio + severity, one row per
    cell. Portfolio measures (loss_ratio, earned_premium, incurred_claim_amount)
    clipped to fact_underwriting_monthly's real availability window; severity
    measures from fact_accident over the full filter date range."""
    con = get_connection()
    window = f.underwriting_window()
    portfolio = pd.DataFrame(
        columns=[
            "vehicle_group",
            "region",
            "earned_premium",
            "incurred_claim_amount",
            "claim_count",
        ]
    )
    if window is not None:
        start, end = window
        portfolio = con.execute(
            """
            SELECT
                v.vehicle_group_label AS vehicle_group,
                g.region,
                SUM(u.earned_premium)::DOUBLE AS earned_premium,
                SUM(u.incurred_claim_amount)::DOUBLE AS incurred_claim_amount,
                SUM(u.claim_count)::DOUBLE AS claim_count
            FROM fact_underwriting_monthly u
            JOIN dim_date d ON u.date_sk = d.date_sk
            JOIN dim_vehicle v ON u.vehicle_sk = v.vehicle_sk
            JOIN dim_geography g ON u.geography_sk = g.geography_sk
            WHERE d.date BETWEEN ? AND ?
              AND u.geography_sk = ANY(?)
              AND u.vehicle_sk = ANY(?)
              AND g.geography_sk != -1
              AND v.vehicle_sk != -1
            GROUP BY 1, 2
            """,
            [start, end, list(f.geography_sks), list(f.vehicle_sks)],
        ).fetchdf()

    accidents = con.execute(
        """
        SELECT
            v.vehicle_group_label AS vehicle_group,
            g.region,
            SUM(f.accident_count)::DOUBLE AS accident_count,
            SUM(f.fatality_count)::DOUBLE AS fatality_count,
            SUM(f.total_injury_count)::DOUBLE AS total_injury_count
        FROM fact_accident f
        JOIN dim_date d ON f.date_sk = d.date_sk
        JOIN dim_vehicle v ON f.vehicle_sk = v.vehicle_sk
        JOIN dim_geography g ON f.geography_sk = g.geography_sk
        WHERE d.date BETWEEN ? AND ?
          AND f.geography_sk = ANY(?)
          AND f.vehicle_sk = ANY(?)
          AND f.agency = ANY(?)
          AND g.geography_sk != -1
          AND v.vehicle_sk != -1
        GROUP BY 1, 2
        """,
        [f.start_date, f.end_date, list(f.geography_sks), list(f.vehicle_sks), list(f.agencies)],
    ).fetchdf()
    accidents["severity_index_coarse"] = (
        accidents["fatality_count"] * W_F_COARSE + accidents["total_injury_count"] * W_I_COARSE
    ) / accidents["accident_count"].replace(0, pd.NA)

    merged = accidents.merge(portfolio, on=["vehicle_group", "region"], how="left")
    merged["loss_ratio"] = merged["incurred_claim_amount"] / merged["earned_premium"].replace(
        0, pd.NA
    )
    # claim_count is a per-cell round() in fact_underwriting_monthly's build code
    # (policies_in_force * CLAIM_FREQUENCY/12, cast to int): for a vehicle group
    # whose average per-cell policies_in_force is under ~1 (Heavy Truck/Bus, Other
    # -- CLAUDE.md §7.2, no real registration source), nearly every cell rounds to
    # 0 claims, so incurred_claim_amount can be structurally zero across the whole
    # scope. That is a rounding-cascade artifact of the simulation, not a real "0%
    # loss ratio" -- null it out here rather than let it read as risk-free.
    merged.loc[merged["claim_count"].fillna(0) == 0, "loss_ratio"] = pd.NA
    return merged
