"""Page 1 — Executive Overview (highest-polish page, first impression).

6 KPI cards, a monthly trend line, and a "provinces to watch" alert list —
every number traced to `fact_accident` / `fact_underwriting_monthly` via
`dashboard.lib.data_access`. No hardcoded figures (CLAUDE.md Phase 7 rule).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.graph_objects as go
import streamlit as st

from dashboard.lib.data_access import (
    UNDERWRITING_WINDOW_END,
    UNDERWRITING_WINDOW_START,
    Filters,
    kpi_summary,
    monthly_trend,
    provinces_over_threshold,
)
from dashboard.lib.filters import render_filters
from dashboard.lib.formatting import fmt_index, fmt_int, fmt_money_thb, fmt_pct
from dashboard.lib.insights import render_insight_panel
from dashboard.lib.theme import (
    BLUE,
    GRIDLINE,
    MUTED_INK,
    PRIMARY_INK,
    inject_theme,
    kpi_card_html,
    page_header,
    render_kpi_row,
)

st.set_page_config(page_title="Executive Overview | SiamShield", page_icon="🚗", layout="wide")
inject_theme()
page_header(
    "Executive Overview",
    "Headline risk and portfolio KPIs across every business question (BQ1–BQ8).",
)

f: Filters = render_filters()

no_data = not (f.geography_sks and f.vehicle_sks and f.agencies)
if no_data:
    st.info("Select at least one option in every sidebar filter to see data.")
    st.stop()

summary = kpi_summary(f)
current, delta = summary["current"], summary["delta_pct"]

portfolio_note = None
if current["claim_frequency"] is None:
    portfolio_note = (
        f"Simulated portfolio covers {UNDERWRITING_WINDOW_START:%Y-%m} to "
        f"{UNDERWRITING_WINDOW_END:%Y-%m} only — no overlap with the selected range."
    )

cards = [
    kpi_card_html("Accident Count", fmt_int(current["accident_count"]), delta["accident_count"]),
    kpi_card_html(
        "Claim Frequency",
        fmt_pct(current["claim_frequency"], 2),
        delta["claim_frequency"],
        caveat=portfolio_note,
    ),
    kpi_card_html(
        "Severity Index (coarse)",
        fmt_index(current["severity_index_coarse"]),
        delta["severity_index_coarse"],
        caveat="Fatalities + injuries per accident, weighted (CLAUDE.md §9). All years.",
    ),
    kpi_card_html(
        "Fatality Rate",
        f"{current['fatalities_per_1000_accidents']:.1f} / 1,000"
        if current["fatalities_per_1000_accidents"] is not None
        else "N/A",
        delta["fatalities_per_1000_accidents"],
        caveat="Fatalities per 1,000 accidents.",
    ),
    kpi_card_html(
        "Pure Premium",
        fmt_money_thb(current["pure_premium"]),
        delta["pure_premium"],
        caveat=portfolio_note or "THB per policy-year, simulated portfolio.",
    ),
    kpi_card_html(
        "Loss Ratio",
        fmt_pct(current["loss_ratio"], 1),
        delta["loss_ratio"],
        caveat=portfolio_note,
    ),
]
render_kpi_row(cards)

render_insight_panel(title="AI Insight — National overview, 2025", scope_key="national_2025")
st.caption(
    "This narrative and its verification are fixed to the national, full-2025 scope "
    "(CLAUDE.md Phase 8) — independent of the sidebar filters above."
)

st.markdown("#### Monthly accident trend")
trend = monthly_trend(f)
if trend.empty:
    st.info("No accidents in the selected filters.")
else:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend["month"],
            y=trend["accident_count"],
            mode="lines",
            line={"color": BLUE, "width": 2},
            hovertemplate="%{x|%b %Y}<br>%{y:,.0f} accidents<extra></extra>",
            name="Accidents",
        )
    )
    fig.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font={"color": PRIMARY_INK, "family": "sans-serif"},
        xaxis={"gridcolor": GRIDLINE, "showgrid": False},
        yaxis={"gridcolor": GRIDLINE, "title": "Accidents / month"},
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "2026 has ~4 months of data (P7.11 holdout) — extend the date filter to see it; "
        "excluded from the default range so it doesn't read as a mid-year drop."
    )

st.markdown("#### Provinces to watch")
alerts = provinces_over_threshold(f)
threshold, flagged = alerts.threshold, alerts.flagged
if threshold is None or flagged.empty:
    st.success("No provinces currently exceed the alert threshold in this filter scope.")
else:
    st.caption(
        f"Flagged: severity index above the mean + 1 standard deviation among provinces "
        f"with ≥30 accidents in scope (threshold: {threshold:.2f})."
    )
    display = flagged[["province_en", "region", "accident_count", "severity_index_coarse"]].copy()
    display.columns = ["Province", "Region", "Accidents", "Severity Index"]
    display["Accidents"] = display["Accidents"].map(lambda v: f"{v:,.0f}")
    display["Severity Index"] = display["Severity Index"].map(lambda v: f"{v:.2f}")
    st.dataframe(display, hide_index=True, width="stretch")

st.caption(
    f"<span style='color:{MUTED_INK}'>Accident measures: fact_accident, all years in range. "
    "Portfolio measures (claim frequency, pure premium, loss ratio): fact_underwriting_monthly, "
    "simulated per docs/simulation_assumptions.md.</span>",
    unsafe_allow_html=True,
)
