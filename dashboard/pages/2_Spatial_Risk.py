"""Page 2 — Spatial Risk (BQ1, BQ4).

BQ1: which province x month is riskiest, and is it driven by frequency or
severity? BQ4: do tourism-dense provinces carry a different risk profile?
Every number traced to `fact_accident` via `dashboard.lib.data_access` —
no hardcoded figures (CLAUDE.md Phase 7 rule).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.express as px
import streamlit as st

from dashboard.lib.data_access import (
    Filters,
    accident_points,
    province_exposure_adjusted_rate,
    province_summary,
    top_province_months,
)
from dashboard.lib.filters import render_filters
from dashboard.lib.insights import render_insight_panel
from dashboard.lib.theme import (
    GRIDLINE,
    MUTED_INK,
    PRIMARY_INK,
    REGION_COLOR_MAP,
    bq_tag,
    inject_theme,
    page_header,
)
from rri.insight.facts import InsightScope

st.set_page_config(page_title="Spatial Risk | SiamShield", page_icon="🗺️", layout="wide")
inject_theme()
page_header(
    "Spatial Risk",
    "Which provinces and months carry the most risk, and does tourism density change the picture?",
)

f: Filters = render_filters()

no_data = not (f.geography_sks and f.vehicle_sks and f.agencies)
if no_data:
    st.info("Select at least one option in every sidebar filter to see data.")
    st.stop()

summary = province_summary(f)
if summary.empty:
    st.info("No accidents in the selected filters.")
    st.stop()

bq_tag("BQ1")
st.markdown("#### Province risk map")
st.caption(
    "Bubble map at each province's own accident centroid (mean lat/lon of its real "
    "accidents) — this project has no committed province-boundary GeoJSON, so this "
    "is a data-derived centroid map, not a polygon choropleth. Click a province to "
    "drill down to its actual accident locations below."
)

map_metric = st.radio(
    "Map metric",
    ["Accident count (frequency)", "Severity index (coarse)", "Accidents per 1,000 vehicles"],
    horizontal=True,
    key="spatial_map_metric",
)

map_df = summary.dropna(subset=["centroid_lat", "centroid_lon"]).copy()
if map_metric == "Accidents per 1,000 vehicles":
    rate = province_exposure_adjusted_rate(f)
    if rate.empty:
        st.info(
            "No exposure-adjusted rate available — the simulated portfolio only "
            "covers 2022-01 to 2025-12, with no overlap with the selected date range."
        )
        map_df = map_df.iloc[0:0]
    else:
        map_df = map_df.merge(rate, on="geography_sk", how="inner")
        color_col, color_label = "accidents_per_1000_vehicles", "Accidents / 1,000 vehicles"
elif map_metric == "Severity index (coarse)":
    color_col, color_label = "severity_index_coarse", "Severity index"
else:
    color_col, color_label = "accident_count", "Accident count"

if not map_df.empty:
    hover_data = {"accident_count": ":,", "centroid_lat": False, "centroid_lon": False}
    if color_col != "accident_count":
        hover_data[color_col] = ":.2f"
    fig_map = px.scatter_map(
        map_df,
        lat="centroid_lat",
        lon="centroid_lon",
        size="accident_count",
        color=color_col,
        hover_name="province_en",
        hover_data=hover_data,
        custom_data=["geography_sk", "province_en"],
        color_continuous_scale="Blues",
        labels={color_col: color_label, "accident_count": "Accident count"},
        center={"lat": 13.0, "lon": 101.3},
        zoom=4.9,
        height=480,
    )
    fig_map.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        map_style="carto-positron",
        coloraxis_colorbar={"title": color_label},
    )
    map_event = st.plotly_chart(
        fig_map,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="province_map_select",
    )

    selected_sk: int | None = None
    selected_name: str | None = None
    points = map_event.selection.points if map_event and map_event.selection else []
    if points:
        selected_sk = int(points[0]["customdata"][0])
        selected_name = str(points[0]["customdata"][1])

    st.markdown("#### Drill-down: accident locations")
    if selected_sk is None:
        st.info("Click a province on the map above to see its individual accident locations.")
    else:
        detail = accident_points(f, selected_sk)
        st.caption(
            f"{selected_name}: {len(detail):,} accidents shown "
            f"(most recent {len(detail):,} in scope, capped at 2,000)."
        )
        if detail.empty:
            st.info("No geolocated accidents for this province in the current filters.")
        else:
            fig_detail = px.scatter_map(
                detail,
                lat="latitude",
                lon="longitude",
                color="fatality_count",
                hover_data=["date", "vehicle_group_label", "cause_group", "total_injury_count"],
                color_continuous_scale="Reds",
                zoom=7,
                height=420,
            )
            fig_detail.update_layout(
                margin={"l": 0, "r": 0, "t": 0, "b": 0}, map_style="carto-positron"
            )
            st.plotly_chart(fig_detail, width="stretch")

        slug = selected_name.lower().replace(" ", "_") if selected_name else ""
        render_insight_panel(
            title=f"AI Insight — {selected_name}",
            scope_key=f"province_{slug}_2025",
            live_scope=InsightScope(
                label=f"{selected_name}, {f.start_date} to {f.end_date}",
                start_date=f.start_date,
                end_date=f.end_date,
                geography_sk=selected_sk,
            ),
        )
        st.caption(
            "Cached AI insights are pre-built for the province, 2025 scope (CLAUDE.md "
            "Phase 8) — 'Regenerate live' (if available) covers the exact sidebar date "
            "range instead."
        )

st.markdown("#### Frequency vs. severity, by province")
st.caption(
    "Each point is one province in the selected scope. Size = accident count, "
    "color = region. A province in the top-right is high on both frequency "
    "and severity; top-left is few accidents but severe when they happen."
)
fig = px.scatter(
    summary,
    x="accident_count",
    y="severity_index_coarse",
    size="accident_count",
    color="region",
    hover_name="province_en",
    labels={
        "accident_count": "Accident count (frequency)",
        "severity_index_coarse": "Severity index (coarse)",
        "region": "Region",
    },
    color_discrete_map=REGION_COLOR_MAP,
)
fig.update_layout(
    height=440,
    margin={"l": 10, "r": 10, "t": 10, "b": 10},
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    font={"color": PRIMARY_INK, "family": "sans-serif"},
    xaxis={"gridcolor": GRIDLINE},
    yaxis={"gridcolor": GRIDLINE},
)
st.plotly_chart(fig, width="stretch")

st.markdown("#### Province ranking")
rank_col1, rank_col2 = st.columns(2)
with rank_col1:
    st.caption("Top 10 by accident count (frequency)")
    by_freq = summary.sort_values("accident_count", ascending=False).head(10)
    disp = by_freq[["province_en", "region", "accident_count", "severity_index_coarse"]].copy()
    disp.columns = ["Province", "Region", "Accidents", "Severity Index"]
    disp["Accidents"] = disp["Accidents"].map(lambda v: f"{v:,.0f}")
    disp["Severity Index"] = disp["Severity Index"].map(lambda v: f"{v:.2f}")
    st.dataframe(disp, hide_index=True, width="stretch")
with rank_col2:
    st.caption("Top 10 by severity index (min 30 accidents in scope)")
    eligible = summary[summary["accident_count"] >= 30]
    by_sev = eligible.sort_values("severity_index_coarse", ascending=False).head(10)
    disp2 = by_sev[["province_en", "region", "accident_count", "severity_index_coarse"]].copy()
    disp2.columns = ["Province", "Region", "Accidents", "Severity Index"]
    disp2["Accidents"] = disp2["Accidents"].map(lambda v: f"{v:,.0f}")
    disp2["Severity Index"] = disp2["Severity Index"].map(lambda v: f"{v:.2f}")
    st.dataframe(disp2, hide_index=True, width="stretch")

st.markdown("#### Riskiest province x month cells")
st.caption(
    "Top 15 individual (province, month) combinations by severity index, "
    "restricted to cells with ≥5 accidents so a single bad month in a tiny "
    "sample can't dominate the list."
)
top_cells = top_province_months(f, limit=15)
if top_cells.empty:
    st.info("No province-month cell in scope meets the minimum accident count.")
else:
    disp3 = top_cells[
        ["province_en", "region", "month", "accident_count", "severity_index_coarse"]
    ].copy()
    disp3.columns = ["Province", "Region", "Month", "Accidents", "Severity Index"]
    disp3["Month"] = disp3["Month"].dt.strftime("%Y-%m")
    disp3["Accidents"] = disp3["Accidents"].map(lambda v: f"{v:,.0f}")
    disp3["Severity Index"] = disp3["Severity Index"].map(lambda v: f"{v:.2f}")
    st.dataframe(disp3, hide_index=True, width="stretch")

bq_tag("BQ4")
st.markdown("#### Tourism density vs. risk")
tourism = summary[summary["accident_count"] >= 30]
fig2 = px.scatter(
    tourism,
    x="tourist_attraction_count",
    y="severity_index_coarse",
    size="accident_count",
    color="region",
    hover_name="province_en",
    labels={
        "tourist_attraction_count": "Tourist attractions (count, TAT)",
        "severity_index_coarse": "Severity index (coarse)",
        "region": "Region",
    },
    color_discrete_map=REGION_COLOR_MAP,
)
fig2.update_layout(
    height=400,
    margin={"l": 10, "r": 10, "t": 10, "b": 10},
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    font={"color": PRIMARY_INK, "family": "sans-serif"},
    xaxis={"gridcolor": GRIDLINE},
    yaxis={"gridcolor": GRIDLINE},
    showlegend=False,
)
st.plotly_chart(fig2, width="stretch")
corr = tourism[["tourist_attraction_count", "severity_index_coarse"]].corr().iloc[0, 1]
st.caption(
    f"Pearson correlation (tourist attraction count vs. severity index, provinces "
    f"with ≥30 accidents in scope): r = {corr:.2f}. This is a province-level "
    "(ecological) correlation only — no per-accident tourist-vs-local flag exists "
    "in the source data."
)

st.caption(
    f"<span style='color:{MUTED_INK}'>All measures: fact_accident, aggregated "
    "over the selected filter scope. Severity index uses severity_index_coarse "
    "(CLAUDE.md §9, all years).</span>",
    unsafe_allow_html=True,
)
