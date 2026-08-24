"""Pricing Simulator support (BQ7) — loads the persisted LightGBM frequency/
severity models and predicts a what-if risk scenario.

**Why this bypasses `rri.models.persistence`**: that module imports every
model tier at the top of the file, including the PyTorch-based GRU/MLP
(`rri.models.gru_forecast`, `rri.models.mlp_entity_embeddings`). Importing it
in the same process as LightGBM reliably segfaults — the documented libomp
conflict from CLAUDE.md's Phase 6 log ("PyTorch and LightGBM both bundle
their own libomp runtime... reliably segfaults inside LightGBM's Dataset
construction"), confirmed directly here too (`import rri.models.persistence`
alone crashes the interpreter with SIGSEGV, no GRU/MLP code even called).
Streamlit runs the whole dashboard in one long-lived process, so the
test-suite's "two separate pytest processes" fix doesn't apply — the fix
here is importing `rri.models.lightgbm_baseline` directly and loading the
`.joblib` files ourselves, which never pulls torch in. Still the same
persisted artifacts and the same `predict_lightgbm_*` functions Phase 5
built — just not routed through the aggregating module.

**Why LightGBM, not GLM**: the GLM lives in the same `persistence.py` file,
so it carries the identical import-time risk. LightGBM also won the
frequency Gini (0.773 vs. 0.751, CLAUDE.md §12 Phase 5 checkpoint).

**Why only 3 vehicle groups**: Heavy Truck/Bus and Other have no real
vehicle-registration exposure source (CLAUDE.md §7.2) — the frequency model
was never trained on rows for these two groups at all (`frequency_training_
rows` filters to non-null `log_exposure`), so predicting for them would be
extrapolating outside the model's training distribution with no real offset
to use. Restricting the simulator to the 3 normalized groups is the
technically correct scope, not just a simplification.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from config.settings import settings
from dashboard.lib.data_access import RAINFALL_PATH, get_connection
from rri.features.frequency_severity import _real_geography_with_tourism_tier
from rri.models.lightgbm_baseline import (
    FittedLgbmFrequency,
    FittedLgbmSeverity,
    predict_lightgbm_frequency,
    predict_lightgbm_severity,
)

NORMALIZED_VEHICLE_GROUPS = ["motorcycle", "passenger_car", "pickup"]

#: Illustrative lapse-risk heuristic -- NOT fitted to any data (no lapse model
#: exists in this project; CLAUDE.md never scoped one). Published motor-
#: insurance price-elasticity studies commonly cite a few percentage points of
#: extra lapse per large premium swing; these are mid-range illustrative
#: figures in that spirit, same "Option B" convention as the Phase 4
#: simulation constants -- not independently sourced, and labeled as such on
#: the page.
BASE_LAPSE_RATE = 0.05
LAPSE_SENSITIVITY_PER_100PCT = 0.15


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[FittedLgbmFrequency, FittedLgbmSeverity]:
    freq = joblib.load(settings.models_dir / "lightgbm_frequency.joblib")
    sev = joblib.load(settings.models_dir / "lightgbm_severity.joblib")
    return freq, sev


@st.cache_data(show_spinner=False)
def geography_attributes() -> pd.DataFrame:
    """Real provinces with region/flood/ev/r35mm attributes + the same
    tourism-tier bucketing `rri.features.frequency_severity` uses (reused
    directly, not reimplemented)."""
    con = get_connection()
    dim_geo = pl.from_pandas(
        con.execute("SELECT * FROM dim_geography WHERE geography_sk != -1").fetchdf()
    )
    # `_real_geography_with_tourism_tier` selects a fixed column set for the
    # modeling frame that doesn't include province_en (display-only, not a
    # model feature) -- join it back in for the UI.
    tiered = _real_geography_with_tourism_tier(dim_geo)
    with_english = tiered.join(
        dim_geo.select("geography_sk", "province_en"), on="geography_sk", how="left"
    )
    return with_english.to_pandas()


@st.cache_data(show_spinner=False)
def historical_month_rain(month: int) -> pd.DataFrame:
    """Average rain_lead0 per province for a given calendar month, across all years."""
    con = get_connection()
    return con.execute(
        """
        SELECT Province_ID AS province_id, AVG(rain_lead0) AS rain_lead0
        FROM read_parquet(?)
        WHERE CAST(Initial_Month AS INTEGER) = ?
        GROUP BY 1
        """,
        [str(RAINFALL_PATH), month],
    ).fetchdf()


@st.cache_data(show_spinner=False)
def historical_month_calendar(month: int) -> dict[str, float | bool]:
    """n_holiday_days / n_long_weekend_days averages + has_songkran/has_newyear
    for a calendar month, averaged across all real years in dim_date."""
    con = get_connection()
    row = con.execute(
        """
        SELECT
            AVG(n_holiday_days) AS n_holiday_days,
            AVG(n_long_weekend_days) AS n_long_weekend_days,
            bool_or(has_songkran) AS has_songkran,
            bool_or(has_newyear) AS has_newyear
        FROM (
            SELECT
                year,
                SUM(CASE WHEN is_public_holiday THEN 1 ELSE 0 END) AS n_holiday_days,
                SUM(CASE WHEN is_long_weekend THEN 1 ELSE 0 END) AS n_long_weekend_days,
                bool_or(is_songkran_7days) AS has_songkran,
                bool_or(is_newyear_7days) AS has_newyear
            FROM dim_date
            WHERE date_sk != -1 AND month = ?
            GROUP BY year
        )
        """,
        [month],
    ).fetchone()
    assert row is not None
    return {
        "n_holiday_days": float(row[0]),
        "n_long_weekend_days": float(row[1]),
        "has_songkran": bool(row[2]),
        "has_newyear": bool(row[3]),
    }


@st.cache_data(show_spinner=False)
def season_for_month(month: int) -> str:
    con = get_connection()
    row = con.execute(
        "SELECT season FROM dim_date WHERE month = ? AND date_sk != -1 LIMIT 1", [month]
    ).fetchone()
    return str(row[0]) if row else "unknown"


@st.cache_data(show_spinner=False)
def latest_exposure_all() -> pd.DataFrame:
    """Latest available exposure_vehicle_months per (province, vehicle_group) --
    the same real-registration back-cast the frequency model's log_exposure
    offset was trained on (see fact_underwriting_monthly.exposure_vehicle_months
    docstring: same §7.1 back-cast source)."""
    con = get_connection()
    return con.execute(
        """
        SELECT province_th, vehicle_group, exposure_vehicle_months
        FROM (
            SELECT
                g.province_th, v.vehicle_group, u.exposure_vehicle_months,
                ROW_NUMBER() OVER (
                    PARTITION BY g.province_th, v.vehicle_group ORDER BY d.date DESC
                ) AS rn
            FROM fact_underwriting_monthly u
            JOIN dim_geography g ON u.geography_sk = g.geography_sk
            JOIN dim_vehicle v ON u.vehicle_sk = v.vehicle_sk
            JOIN dim_date d ON u.date_sk = d.date_sk
            WHERE v.vehicle_group = ANY(?)
        )
        WHERE rn = 1
        """,
        [NORMALIZED_VEHICLE_GROUPS],
    ).fetchdf()


@st.cache_data(show_spinner=False)
def current_segment_economics(province_th: str, vehicle_group: str) -> dict[str, float] | None:
    """Current simulated premium/claims for one (province, vehicle_group), all
    cover classes combined, over the full underwriting window."""
    con = get_connection()
    row = con.execute(
        """
        SELECT
            SUM(u.earned_premium)::DOUBLE AS earned_premium,
            SUM(u.incurred_claim_amount)::DOUBLE AS incurred_claim_amount,
            SUM(u.policies_in_force)::DOUBLE AS policy_months
        FROM fact_underwriting_monthly u
        JOIN dim_geography g ON u.geography_sk = g.geography_sk
        JOIN dim_vehicle v ON u.vehicle_sk = v.vehicle_sk
        WHERE g.province_th = ? AND v.vehicle_group = ?
        """,
        [province_th, vehicle_group],
    ).fetchone()
    if row is None or not row[2]:
        return None
    earned_premium, incurred_claim_amount, policy_months = row
    return {
        "earned_premium": earned_premium,
        "incurred_claim_amount": incurred_claim_amount,
        "policy_months": policy_months,
        "current_annual_premium_per_policy": earned_premium / (policy_months / 12),
        "current_loss_ratio": (incurred_claim_amount / earned_premium) if earned_premium else None,
    }


def _feature_frame(
    provinces: pd.DataFrame,
    vehicle_groups: list[str],
    month: int,
    rain_by_province: dict[str, float],
    exposure: pd.DataFrame,
) -> pd.DataFrame:
    calendar = historical_month_calendar(month)
    season = season_for_month(month)
    rows = []
    for _, prov in provinces.iterrows():
        for vg in vehicle_groups:
            exp_match = exposure[
                (exposure["province_th"] == prov["province_th"]) & (exposure["vehicle_group"] == vg)
            ]
            exposure_val = (
                float(exp_match["exposure_vehicle_months"].iloc[0]) if not exp_match.empty else None
            )
            if not exposure_val or exposure_val <= 0:
                continue
            rain_val = rain_by_province.get(prov["province_id"], np.nan)
            rows.append(
                {
                    "province_th": prov["province_th"],
                    "vehicle_group": vg,
                    "region": prov["region"],
                    "season": season,
                    "tourism_tier": prov["tourism_tier"],
                    "has_songkran": calendar["has_songkran"],
                    "has_newyear": calendar["has_newyear"],
                    "rain_lead0": rain_val,
                    "rain_lead1": rain_val,
                    "rain_lead2": rain_val,
                    "rain_lead3": rain_val,
                    "rain_lead4": rain_val,
                    "rain_lead5": rain_val,
                    "flood_pct_high_risk_tambons": prov["flood_pct_high_risk_tambons"],
                    "ev_share": prov["ev_share"],
                    "r35mm_avg_days_2014_2023": prov["r35mm_avg_days_2014_2023"],
                    "n_holiday_days": calendar["n_holiday_days"],
                    "n_long_weekend_days": calendar["n_long_weekend_days"],
                    "log_exposure": np.log(exposure_val),
                    "exposure_vehicle_months": exposure_val,
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ScenarioResult:
    predicted_frequency: float
    predicted_severity: float
    risk_score: float
    portfolio_avg_risk_score: float
    risk_multiplier: float


@st.cache_data(show_spinner=False)
def run_scenario(
    province_th: str, vehicle_group: str, month: int, rain_override: float | None
) -> ScenarioResult | None:
    """Predict frequency/severity for one (province, vehicle_group, month, rain)
    scenario, and compare against the portfolio-average predicted risk for that
    same month (every normalized province x vehicle_group cell, each at its own
    historical rainfall) to get a relative risk multiplier for repricing."""
    provinces = geography_attributes()
    exposure = latest_exposure_all()
    rain_hist = historical_month_rain(month)
    rain_by_province = dict(zip(rain_hist["province_id"], rain_hist["rain_lead0"], strict=True))

    prov_row = provinces[provinces["province_th"] == province_th]
    if prov_row.empty:
        return None
    province_id = prov_row.iloc[0]["province_id"]
    scenario_rain = (
        rain_override if rain_override is not None else rain_by_province.get(province_id, 50.0)
    )

    portfolio_frame = _feature_frame(
        provinces, NORMALIZED_VEHICLE_GROUPS, month, rain_by_province, exposure
    )
    if portfolio_frame.empty:
        return None

    scenario_frame = portfolio_frame[
        (portfolio_frame["province_th"] == province_th)
        & (portfolio_frame["vehicle_group"] == vehicle_group)
    ].copy()
    if scenario_frame.empty:
        return None
    rain_cols = ["rain_lead0", "rain_lead1", "rain_lead2", "rain_lead3", "rain_lead4", "rain_lead5"]
    scenario_frame[rain_cols] = scenario_rain

    # risk_score is a PER-VEHICLE rate (predicted accidents / that cell's own
    # exposure) x severity, not raw predicted count x severity -- Bangkok's
    # predicted accident count is naturally huge next to a small province's
    # purely because its registered fleet is huge, which would make a
    # count-based "risk score" mostly measure province size, not per-policy
    # risk. Dividing by exposure_vehicle_months first is what makes the
    # multiplier a fair per-policy repricing factor.
    # The portfolio reference is exposure-weighted (total predicted claims-cost
    # / total exposure), not a plain mean of each cell's own rate -- an
    # unweighted mean over-represents small provinces (many cells with tiny,
    # noisy exposure denominators pull the average up), which would make
    # every populous province look artificially "safer than average" purely
    # from that skew. Weighting by exposure is the standard actuarial way to
    # get a book-wide average rate.
    freq_model, sev_model = load_models()
    portfolio_freq = predict_lightgbm_frequency(freq_model, pl.from_pandas(portfolio_frame))
    portfolio_sev = predict_lightgbm_severity(sev_model, pl.from_pandas(portfolio_frame))
    portfolio_exposure = portfolio_frame["exposure_vehicle_months"].to_numpy()
    portfolio_risk = float(np.sum(portfolio_freq * portfolio_sev) / np.sum(portfolio_exposure))

    scenario_freq = predict_lightgbm_frequency(freq_model, pl.from_pandas(scenario_frame))
    scenario_sev = predict_lightgbm_severity(sev_model, pl.from_pandas(scenario_frame))
    scenario_rate = float(scenario_freq[0]) / float(
        scenario_frame["exposure_vehicle_months"].iloc[0]
    )
    risk_score = scenario_rate * float(scenario_sev[0])

    return ScenarioResult(
        predicted_frequency=float(scenario_freq[0]),
        predicted_severity=float(scenario_sev[0]),
        risk_score=risk_score,
        portfolio_avg_risk_score=portfolio_risk,
        risk_multiplier=risk_score / portfolio_risk if portfolio_risk else float("nan"),
    )


def lapse_risk_estimate(premium_change_pct: float) -> float:
    """Illustrative heuristic only -- see module docstring."""
    raw = BASE_LAPSE_RATE + LAPSE_SENSITIVITY_PER_100PCT * (premium_change_pct / 100)
    return max(0.0, min(raw, 0.6))
