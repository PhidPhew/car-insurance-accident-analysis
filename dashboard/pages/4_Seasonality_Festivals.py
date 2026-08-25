"""Page 4 — Seasonality & Festivals (BQ3).

How much do holiday periods (Songkran, New Year, the "7 dangerous days") raise
case counts, and is severity different? Every number traced to `fact_accident`
via `dashboard.lib.data_access` (CLAUDE.md Phase 7 rule).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.lib.data_access import (
    Filters,
    holiday_vs_normal_comparison,
    monthly_trend_with_calendar,
)
from dashboard.lib.filters import render_filters
from dashboard.lib.formatting import fmt_index
from dashboard.lib.theme import (
    BLUE,
    GRIDLINE,
    MUTED_INK,
    PRIMARY_INK,
    STATUS_CRITICAL,
    bq_tag,
    inject_theme,
    page_header,
)

st.set_page_config(page_title="Seasonality & Festivals | SiamShield", page_icon="🎉", layout="wide")
inject_theme()
page_header(
    "Seasonality & Festivals",
    "How much do holiday periods raise case counts, and is severity different?",
)

f: Filters = render_filters()

no_data = not (f.geography_sks and f.vehicle_sks and f.agencies)
if no_data:
    st.info("Select at least one option in every sidebar filter to see data.")
    st.stop()

bq_tag("BQ3")
st.markdown("#### Monthly accident trend, with festival and COVID windows marked")
st.caption(
    'Orange bands: Songkran "7 dangerous days" (April). Purple bands: New Year '
    '"7 dangerous days" (Dec–Jan). Red band: the COVID-severe disruption window '
    "(Apr–Jun 2020) — marked explicitly so it isn't misread as a normal seasonal dip."
)
trend = monthly_trend_with_calendar(f)
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
    for _, row in trend[trend["has_songkran"]].iterrows():
        month_start = row["month"]
        month_end = month_start + pd.Timedelta(days=31)
        fig.add_vrect(x0=month_start, x1=month_end, fillcolor="#eb6834", opacity=0.15, line_width=0)
    for _, row in trend[trend["has_newyear"]].iterrows():
        month_start = row["month"]
        month_end = month_start + pd.Timedelta(days=31)
        fig.add_vrect(x0=month_start, x1=month_end, fillcolor="#4a3aa7", opacity=0.15, line_width=0)
    covid_rows = trend[trend["covid_level"] >= 1]
    if not covid_rows.empty:
        fig.add_vrect(
            x0=covid_rows["month"].min(),
            x1=covid_rows["month"].max() + pd.Timedelta(days=31),
            fillcolor=STATUS_CRITICAL,
            opacity=0.12,
            line_width=0,
            annotation_text="COVID disruption",
            annotation_position="top left",
        )
    fig.update_layout(
        height=340,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font={"color": PRIMARY_INK, "family": "sans-serif"},
        xaxis={"gridcolor": GRIDLINE, "showgrid": False},
        yaxis={"gridcolor": GRIDLINE, "title": "Accidents / month"},
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

st.markdown("#### Holiday periods vs. every other day")
comparison = holiday_vs_normal_comparison(f)
if comparison.empty:
    st.info("No accidents in the selected filters.")
else:
    comparison["accidents_per_day"] = comparison["accident_count"] / comparison["n_days"]
    comparison["severity_index_coarse"] = (
        comparison["fatality_count"] * 40.0 + comparison["total_injury_count"] * 2.15
    ) / comparison["accident_count"]
    order = ["Songkran (7 days)", "New Year (7 days)", "Other days"]
    comparison["period"] = pd.Categorical(comparison["period"], categories=order, ordered=True)
    comparison = comparison.sort_values("period")

    col1, col2, col3 = st.columns(3)
    for col, (_, row) in zip([col1, col2, col3], comparison.iterrows(), strict=True):
        with col:
            st.markdown(f"**{row['period']}**")
            st.metric("Accidents / day", f"{row['accidents_per_day']:.1f}")
            st.caption(f"Severity index: {fmt_index(row['severity_index_coarse'])}")

    baseline = comparison.loc[comparison["period"] == "Other days", "accidents_per_day"]
    if not baseline.empty and baseline.iloc[0] > 0:
        base = baseline.iloc[0]
        for _, row in comparison.iterrows():
            if row["period"] == "Other days":
                continue
            lift = (row["accidents_per_day"] / base - 1) * 100
            st.caption(
                f"{row['period']}: {lift:+.0f}% accidents/day vs. an ordinary day in this scope."
            )

st.caption(
    f"<span style='color:{MUTED_INK}'>All measures: fact_accident. Holiday windows: "
    "dim_date.is_songkran_7days / is_newyear_7days. COVID window: "
    "dim_date.covid_disruption_level ('partial' or 'severe').</span>",
    unsafe_allow_html=True,
)
