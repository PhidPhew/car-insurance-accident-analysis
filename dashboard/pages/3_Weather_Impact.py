"""Page 3 — Weather Impact (BQ2).

How much do rain and flood-prone areas raise risk, and what premium loading
does that justify? Every number traced to `fact_accident` + the cleaned
rainfall staging output via `dashboard.lib.data_access` (CLAUDE.md Phase 7
rule — see data_access.RAINFALL_PATH's docstring for why that one file is
read directly instead of from the warehouse).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.express as px
import streamlit as st

from dashboard.lib.data_access import (
    Filters,
    province_summary,
    rainfall_vs_accidents,
    weather_band_vs_forecast_rain,
)
from dashboard.lib.filters import render_filters
from dashboard.lib.theme import (
    GRIDLINE,
    MUTED_INK,
    PRIMARY_INK,
    STATUS_CRITICAL,
    bq_tag,
    inject_theme,
    page_header,
)

st.set_page_config(page_title="Weather Impact | SiamShield", page_icon="🌧️", layout="wide")
inject_theme()
page_header(
    "Weather Impact",
    "How much do rain and flood-prone areas raise risk, and what premium loading "
    "does that justify?",
)

f: Filters = render_filters()

no_data = not (f.geography_sks and f.vehicle_sks and f.agencies)
if no_data:
    st.info("Select at least one option in every sidebar filter to see data.")
    st.stop()

bq_tag("BQ2")
st.markdown("#### Rainfall vs. accident frequency")
st.caption(
    "Each point is one province-month in scope. Rainfall is the forecast lead-0 "
    "figure (rri.features.frequency_severity's own known-future convention) for "
    "that month, in millimetres."
)
rain_df = rainfall_vs_accidents(f)
if rain_df.empty:
    st.info("No overlapping province-months in the selected filters.")
else:
    fig = px.scatter(
        rain_df,
        x="rain_lead0",
        y="accident_count",
        trendline="ols",
        trendline_color_override=STATUS_CRITICAL,
        opacity=0.45,
        labels={"rain_lead0": "Forecast rainfall (mm)", "accident_count": "Accident count"},
        color_discrete_sequence=["#2a78d6"],
    )
    fig.update_traces(marker={"size": 6})
    fig.update_layout(
        height=400,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font={"color": PRIMARY_INK, "family": "sans-serif"},
        xaxis={"gridcolor": GRIDLINE},
        yaxis={"gridcolor": GRIDLINE},
    )
    st.plotly_chart(fig, width="stretch")
    corr = rain_df[["rain_lead0", "accident_count"]].corr().iloc[0, 1]
    st.caption(f"Pearson r (rainfall vs. accident count, province-month level): {corr:.2f}")

st.markdown("#### The r35mm finding: fewer, more severe crashes where heavy rain is common")
st.caption(
    "r35mm = average yearly days with ≥35mm rainfall, 2014–2023 (dim_geography, static "
    "per province). Phase 5's SHAP analysis found this flips sign between frequency "
    "(decreases) and severity (increases) — heavy-rain provinces see fewer crashes, but "
    "the ones that happen are worse. Two views below test that at province level."
)
summary = province_summary(f)
eligible = summary[summary["accident_count"] >= 30].dropna(subset=["r35mm_avg_days_2014_2023"])
if eligible.empty:
    st.info("Not enough provinces in scope (≥30 accidents, non-null r35mm) to show this.")
else:
    col1, col2 = st.columns(2)
    with col1:
        fig_freq = px.scatter(
            eligible,
            x="r35mm_avg_days_2014_2023",
            y="accident_count",
            trendline="ols",
            trendline_color_override=STATUS_CRITICAL,
            hover_name="province_en",
            log_y=True,
            labels={
                "r35mm_avg_days_2014_2023": "Heavy-rain days/year (2014-2023 avg)",
                "accident_count": "Accident count (frequency, log scale)",
            },
        )
        fig_freq.update_traces(marker={"color": "#2a78d6", "size": 8})
        fig_freq.update_layout(
            height=360,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            plot_bgcolor="#fcfcfb",
            paper_bgcolor="#fcfcfb",
            font={"color": PRIMARY_INK, "family": "sans-serif"},
            xaxis={"gridcolor": GRIDLINE},
            yaxis={"gridcolor": GRIDLINE},
            title="Frequency",
        )
        st.plotly_chart(fig_freq, width="stretch")
        r_freq = eligible[["r35mm_avg_days_2014_2023", "accident_count"]].corr().iloc[0, 1]
        st.caption(f"Pearson r = {r_freq:.2f}")
    with col2:
        fig_sev = px.scatter(
            eligible,
            x="r35mm_avg_days_2014_2023",
            y="severity_index_coarse",
            trendline="ols",
            trendline_color_override=STATUS_CRITICAL,
            hover_name="province_en",
            labels={
                "r35mm_avg_days_2014_2023": "Heavy-rain days/year (2014-2023 avg)",
                "severity_index_coarse": "Severity index (coarse)",
            },
        )
        fig_sev.update_traces(marker={"color": "#eb6834", "size": 8})
        fig_sev.update_layout(
            height=360,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            plot_bgcolor="#fcfcfb",
            paper_bgcolor="#fcfcfb",
            font={"color": PRIMARY_INK, "family": "sans-serif"},
            xaxis={"gridcolor": GRIDLINE},
            yaxis={"gridcolor": GRIDLINE},
            title="Severity",
        )
        st.plotly_chart(fig_sev, width="stretch")
        r_sev = eligible[["r35mm_avg_days_2014_2023", "severity_index_coarse"]].corr().iloc[0, 1]
        st.caption(f"Pearson r = {r_sev:.2f}")

st.caption(
    "These are simple, unconditional province-level correlations — they will not "
    "necessarily match Phase 5's SHAP finding exactly, since SHAP measures each "
    "feature's effect *holding the model's other features constant* (region, season, "
    "tourism, etc.), while a raw scatter does not control for anything. Report the "
    "SHAP result, not this correlation, as the controlled finding."
)

st.markdown("#### Reported weather vs. forecast rainfall cross-check")
st.caption(
    "Average forecast rainfall (that accident's own province+month) grouped by the "
    "REPORTED weather condition at the moment of the accident — a sanity check that "
    "the two independent data sources agree directionally. Hazardous conditions "
    "(rain, low visibility, severe weather) in red."
)
cross = weather_band_vs_forecast_rain(f)
if cross.empty:
    st.info("No data in the selected filters.")
else:
    cross["Condition"] = cross["weather_band"].str.replace("_", " ").str.title()
    fig_cross = px.bar(
        cross,
        x="Condition",
        y="avg_forecast_rain_mm",
        color="is_hazardous",
        color_discrete_map={True: STATUS_CRITICAL, False: "#2a78d6"},
        labels={
            "avg_forecast_rain_mm": "Avg. forecast rainfall (mm)",
            "is_hazardous": "Hazardous condition",
        },
        text_auto=".0f",
    )
    fig_cross.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font={"color": PRIMARY_INK, "family": "sans-serif"},
        xaxis={"gridcolor": GRIDLINE, "title": None},
        yaxis={"gridcolor": GRIDLINE},
    )
    st.plotly_chart(fig_cross, width="stretch")

st.caption(
    f"<span style='color:{MUTED_INK}'>Rainfall: cleaned staging output "
    "(data/10_staging/rainfall.parquet, Phase 2), joined by province + month — not "
    "yet a warehouse dimension (a documented design gap). All other measures: "
    "fact_accident, aggregated over the selected filter scope.</span>",
    unsafe_allow_html=True,
)
