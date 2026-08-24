"""Page 5 — Portfolio Segments (BQ5).

Which vehicle type loses the most money, and how should portfolio mix
change? Every number traced to `fact_accident` / `fact_underwriting_monthly`
via `dashboard.lib.data_access` (CLAUDE.md Phase 7 rule).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.lib.data_access import Filters, portfolio_segment_matrix
from dashboard.lib.filters import render_filters
from dashboard.lib.theme import (
    GRIDLINE,
    MUTED_INK,
    PRIMARY_INK,
    STATUS_CRITICAL,
    STATUS_GOOD,
    bq_tag,
    inject_theme,
    page_header,
)

st.set_page_config(page_title="Portfolio Segments | SiamShield", page_icon="📊", layout="wide")
inject_theme()
page_header(
    "Portfolio Segments",
    "Which vehicle type loses the most money, and how should portfolio mix change?",
)

f: Filters = render_filters()

no_data = not (f.geography_sks and f.vehicle_sks and f.agencies)
if no_data:
    st.info("Select at least one option in every sidebar filter to see data.")
    st.stop()

bq_tag("BQ5")
matrix = portfolio_segment_matrix(f)
if matrix.empty:
    st.info("No accidents in the selected filters.")
    st.stop()

st.markdown("#### Vehicle group x region")
metric = st.radio(
    "Heatmap metric",
    ["Loss ratio", "Severity index (coarse)"],
    horizontal=True,
    key="segment_heatmap_metric",
)
metric_col = "loss_ratio" if metric == "Loss ratio" else "severity_index_coarse"

pivot = matrix.pivot_table(index="region", columns="vehicle_group", values=metric_col)
if pivot.dropna(how="all").empty:
    st.info(
        "No data for this metric in scope."
        + (
            " Loss ratio needs the simulated portfolio window (2022-01 to 2025-12)."
            if metric_col == "loss_ratio"
            else ""
        )
    )
else:
    text = pivot.map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
    fig = px.imshow(
        pivot,
        color_continuous_scale="Blues" if metric_col == "severity_index_coarse" else "Oranges",
        labels={"color": metric, "x": "Vehicle group", "y": "Region"},
        aspect="auto",
    )
    fig.update_traces(text=text.values, texttemplate="%{text}")
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font={"color": PRIMARY_INK, "family": "sans-serif"},
    )
    st.plotly_chart(fig, width="stretch")

if matrix["vehicle_group"].str.startswith(("Heavy Truck", "Other")).any():
    st.caption(
        "⚠️ Heavy Truck / Bus and Other have no real vehicle-registration exposure "
        "source (CLAUDE.md §7.2). Their simulated policies_in_force per cell is so "
        "small that claim_count rounds to 0 almost everywhere, making their loss "
        "ratio structurally undefined (shown as n/a) rather than a real number —"
        ' not "risk-free," just too sparse for this simulated portfolio to price.'
    )

st.markdown("#### Loss ratio by vehicle group")
st.caption(
    "Aggregated across all regions in scope. Red = losing money (loss ratio > "
    "breakeven, dashed line); green = profitable at the simulated premium level."
)
by_vehicle = (
    matrix.groupby("vehicle_group", as_index=False)
    .agg(
        earned_premium=("earned_premium", "sum"),
        incurred_claim_amount=("incurred_claim_amount", "sum"),
        claim_count=("claim_count", "sum"),
        accident_count=("accident_count", "sum"),
        fatality_count=("fatality_count", "sum"),
        total_injury_count=("total_injury_count", "sum"),
    )
    .sort_values("earned_premium", ascending=False)
)
by_vehicle["loss_ratio"] = by_vehicle["incurred_claim_amount"] / by_vehicle[
    "earned_premium"
].replace(0, pd.NA)
# See portfolio_segment_matrix's docstring note: a vehicle group whose simulated
# claim_count rounds to 0 in every cell (Heavy Truck/Bus, Other) has a
# structurally-zero incurred_claim_amount that is a rounding artifact, not a real
# "0% loss ratio" -- exclude rather than show it as risk-free.
excluded = by_vehicle[by_vehicle["claim_count"].fillna(0) == 0]["vehicle_group"].tolist()
by_vehicle = by_vehicle[by_vehicle["claim_count"].fillna(0) > 0].dropna(subset=["loss_ratio"])
if excluded:
    st.caption(
        f"⚠️ Excluded from this chart: {', '.join(excluded)} — their simulated "
        "claim count rounds to 0 in every province-month cell (too few simulated "
        "policies per cell), making incurred claims structurally zero rather than "
        "a real risk-free result. See the heatmap's own caveat above."
    )
if by_vehicle.empty:
    st.info("No loss-ratio data in this filter scope (needs the 2022-01 to 2025-12 window).")
else:
    by_vehicle["status"] = by_vehicle["loss_ratio"].apply(
        lambda v: "Losing money" if v > 1.0 else "Profitable"
    )
    fig_bar = px.bar(
        by_vehicle.sort_values("loss_ratio", ascending=True),
        x="loss_ratio",
        y="vehicle_group",
        color="status",
        color_discrete_map={"Losing money": STATUS_CRITICAL, "Profitable": STATUS_GOOD},
        orientation="h",
        labels={"loss_ratio": "Loss ratio", "vehicle_group": "", "status": ""},
        text_auto=".0%",
    )
    fig_bar.add_vline(x=1.0, line_dash="dash", line_color=MUTED_INK)
    fig_bar.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font={"color": PRIMARY_INK, "family": "sans-serif"},
        xaxis={"gridcolor": GRIDLINE, "tickformat": ".0%"},
        yaxis={"gridcolor": GRIDLINE},
    )
    st.plotly_chart(fig_bar, width="stretch")

st.caption(
    f"<span style='color:{MUTED_INK}'>Loss ratio: fact_underwriting_monthly "
    "(simulated portfolio), 2022-01 to 2025-12 only. Severity index: fact_accident, "
    "full filter date range (severity_index_coarse, CLAUDE.md §9).</span>",
    unsafe_allow_html=True,
)
