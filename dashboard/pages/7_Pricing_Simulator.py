"""Page 7 — Pricing Simulator (BQ7).

If we reprice per the model, how much does the portfolio improve, and what
lapse risk does that create? Calls the persisted LightGBM frequency/severity
models (CLAUDE.md Phase 7 rule — see dashboard.lib.pricing's module
docstring for why this bypasses rri.models.persistence).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from dashboard.lib.formatting import fmt_money_thb, fmt_pct
from dashboard.lib.pricing import (
    current_segment_economics,
    geography_attributes,
    historical_month_rain,
    lapse_risk_estimate,
    run_scenario,
)
from dashboard.lib.theme import (
    MUTED_INK,
    bq_tag,
    inject_theme,
    kpi_card_html,
    page_header,
    render_kpi_row,
)

st.set_page_config(page_title="Pricing Simulator | SiamShield", page_icon="🧮", layout="wide")
inject_theme()
page_header(
    "Pricing Simulator",
    "If we reprice per the model, how much does the portfolio improve, and "
    "what lapse risk does that create?",
)

bq_tag("BQ7")
st.caption(
    "SiamShield's current simulated premium is flat by cover class only — it does "
    "not vary by province, vehicle risk, or weather (docs/simulation_assumptions.md). "
    "This simulator asks: what if it did?"
)

provinces = geography_attributes().sort_values("province_en")
vehicle_labels = {
    "motorcycle": "Motorcycle",
    "passenger_car": "Passenger Car / Van",
    "pickup": "Pickup Truck",
}
month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip

col1, col2, col3 = st.columns(3)
with col1:
    province_en = st.selectbox("Province", provinces["province_en"].tolist(), index=0)
with col2:
    vehicle_group = st.selectbox(
        "Vehicle group", list(vehicle_labels.keys()), format_func=lambda k: vehicle_labels[k]
    )
with col3:
    month = st.selectbox("Month", list(range(1, 13)), format_func=lambda m: month_names[m - 1])

st.caption(
    "Scoped to Motorcycle, Passenger Car/Van, and Pickup Truck — the only groups "
    "with a real vehicle-registration exposure source the frequency model was "
    "trained on (CLAUDE.md §7.2; see dashboard.lib.pricing's module docstring)."
)

province_id = provinces.loc[provinces["province_en"] == province_en, "province_id"].iloc[0]
province_th = provinces.loc[provinces["province_en"] == province_en, "province_th"].iloc[0]
hist_rain = historical_month_rain(month)
hist_match = hist_rain[hist_rain["province_id"] == province_id]
default_rain = float(hist_match["rain_lead0"].iloc[0]) if not hist_match.empty else 50.0

rain_override = st.slider(
    f"Rainfall scenario for {month_names[month - 1]} (mm)",
    min_value=0.0,
    max_value=800.0,
    value=round(default_rain, 1),
    step=10.0,
    help=f"Historical average for {province_en} in {month_names[month - 1]}: "
    f"{default_rain:.0f}mm. Drag to test a wetter/drier scenario.",
)

with st.spinner("Running the model..."):
    result = run_scenario(province_th, vehicle_group, month, rain_override)
    economics = current_segment_economics(province_th, vehicle_group)

if result is None or economics is None:
    st.warning("No data available for this province/vehicle group combination.")
    st.stop()

st.markdown("#### Model output")
cards = [
    kpi_card_html("Predicted accidents/month", f"{result.predicted_frequency:.1f}", None),
    kpi_card_html("Predicted severity index", f"{result.predicted_severity:.2f}", None),
    kpi_card_html(
        "Risk vs. portfolio average",
        f"{result.risk_multiplier:.2f}x",
        None,
        caveat="Exposure-weighted per-vehicle rate x severity, relative to the "
        "book-wide average for this month.",
    ),
]
render_kpi_row(cards)

st.markdown("#### Repricing impact")
current_premium = economics["current_annual_premium_per_policy"]
current_loss_ratio = economics["current_loss_ratio"]
repriced_premium = current_premium * result.risk_multiplier
repriced_loss_ratio = (
    current_loss_ratio / result.risk_multiplier
    if current_loss_ratio is not None and result.risk_multiplier
    else None
)
premium_change_pct = (result.risk_multiplier - 1) * 100
lapse_risk = lapse_risk_estimate(premium_change_pct)

cards2 = [
    kpi_card_html("Current premium/policy (annual)", fmt_money_thb(current_premium), None),
    kpi_card_html(
        "Repriced premium/policy (annual)",
        fmt_money_thb(repriced_premium),
        premium_change_pct,
        caveat="vs. current, this segment/scenario",
    ),
    kpi_card_html(
        "Current loss ratio", fmt_pct(current_loss_ratio, 1) if current_loss_ratio else "N/A", None
    ),
    kpi_card_html(
        "Repriced loss ratio",
        fmt_pct(repriced_loss_ratio, 1) if repriced_loss_ratio else "N/A",
        None,
        caveat="Holding claims experience fixed, premium repriced by the risk multiplier.",
    ),
    kpi_card_html(
        "Estimated lapse risk",
        fmt_pct(lapse_risk, 0),
        None,
        higher_is_worse=True,
        caveat="Illustrative heuristic, not fitted to data — see dashboard.lib."
        "pricing's module docstring.",
    ),
]
render_kpi_row(cards2)

st.caption(
    f"<span style='color:{MUTED_INK}'>Model: LightGBM frequency/severity (Phase 5, "
    "won the frequency Gini vs. the GLM baseline). Current premium/loss ratio: "
    "fact_underwriting_monthly, full 2022-01 to 2025-12 window, all cover classes "
    "combined for this (province, vehicle group).</span>",
    unsafe_allow_html=True,
)
