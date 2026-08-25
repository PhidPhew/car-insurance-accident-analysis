"""Page 8 — EV Analysis (BQ8) + Data Quality.

Do high-EV-share provinces show higher severity? Is a separate EV rating
factor warranted? Plus a data-quality panel showing the rigor behind every
number on this dashboard. Every number traced to `fact_accident` /
`dim_geography` via `dashboard.lib.data_access`, or to a committed report
artifact (CLAUDE.md Phase 7 rule).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.express as px
import streamlit as st

from dashboard.lib.data_access import (
    AGENCY_LABELS,
    Filters,
    agency_coverage,
    data_quality_gate2,
    ev_share_summary,
)
from dashboard.lib.filters import render_filters
from dashboard.lib.theme import (
    GRIDLINE,
    MUTED_INK,
    PRIMARY_INK,
    REGION_COLOR_MAP,
    bq_tag,
    inject_theme,
    page_header,
)

st.set_page_config(
    page_title="EV Analysis & Data Quality | SiamShield", page_icon="🔋", layout="wide"
)
inject_theme()
page_header(
    "EV Analysis & Data Quality",
    "Do high-EV-share provinces show higher severity? Plus: the rigor behind "
    "every number on this dashboard.",
)

f: Filters = render_filters()

no_data = not (f.geography_sks and f.vehicle_sks and f.agencies)
if no_data:
    st.info("Select at least one option in every sidebar filter to see data.")
    st.stop()

bq_tag("BQ8")
st.markdown("#### EV share vs. severity — an ecological comparison only")
st.warning(
    "**This is NOT a causal EV effect.** Phase 5's SHAP analysis found `ev_share` "
    "behaves as a province-identity / urbanization proxy (r=0.83 vs. total "
    "registered vehicles, concentrated in Bangkok/Phuket/Chiang Mai/Songkhla) — "
    "not a genuine EV risk signal. `fact_accident` has no per-vehicle fuel type "
    "(limitation L5), so this can only ever be a **province-level (ecological) "
    "comparison**, never a per-vehicle one. Do not read this chart as "
    '"EVs are safer" or "EVs are riskier."',
    icon="⚠️",
)

ev_data = ev_share_summary(f)
eligible = ev_data[ev_data["accident_count"] >= 30]
if eligible.empty:
    st.info("Not enough provinces in scope (≥30 accidents) to show this.")
else:
    fig = px.scatter(
        eligible,
        x="ev_share",
        y="severity_index_coarse",
        size="accident_count",
        color="region",
        hover_name="province_en",
        labels={
            "ev_share": "EV share of registered vehicles",
            "severity_index_coarse": "Severity index (coarse)",
            "region": "Region",
        },
        color_discrete_map=REGION_COLOR_MAP,
    )
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font={"color": PRIMARY_INK, "family": "sans-serif"},
        xaxis={"gridcolor": GRIDLINE, "tickformat": ".0%"},
        yaxis={"gridcolor": GRIDLINE},
    )
    st.plotly_chart(fig, width="stretch")
    corr = eligible[["ev_share", "severity_index_coarse"]].corr().iloc[0, 1]
    st.caption(
        f"Pearson r = {corr:.2f} (province-level, ≥30 accidents in scope). "
        "**Conclusion: no separate EV rating factor is warranted from this "
        "project's data** — the honest limitation (no per-vehicle fuel type) "
        "means this question can't be answered at the resolution BQ8 asks for."
    )

st.divider()
st.markdown("#### Data Quality")
st.caption(
    "19 verified data-quality problems (P1–P7, docs/data_quality_findings.md) were "
    "found and fixed in Phases 1–2. Reconciliation is checked at two separate "
    "pipeline stages so ingest loss and dedup aggressiveness can never hide behind "
    "one collapsed number (CLAUDE.md §6)."
)

gate2 = data_quality_gate2()
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Gate 1 — post-Extract, pre-dedup**")
    st.caption(
        "Documented at build time (reports/cleaning_log.csv) — not independently "
        "re-queryable from the built warehouse, since raw duplicates are collapsed "
        "before load."
    )
    st.metric("Accidents", "161,307")
    st.metric("Fatalities", "21,814")
with col2:
    st.markdown("**Gate 2 — post-Load, post-dedup**")
    st.caption("Live from the warehouse right now (fact_accident).")
    st.metric("Accidents", f"{gate2['accident_count']:,.0f}")
    st.metric("Fatalities", f"{gate2['fatality_count']:,.0f}")

st.markdown("#### Reporting agency coverage (P7.13 — a stated limitation, not fixable)")
coverage = agency_coverage()
coverage["share"] = coverage["accident_count"] / coverage["accident_count"].sum()
coverage["Agency"] = coverage["agency"].map(lambda a: AGENCY_LABELS.get(a, a))
fig_cov = px.bar(
    coverage.sort_values("share"),
    x="share",
    y="Agency",
    orientation="h",
    text_auto=".1%",
    labels={"share": "Share of all accidents", "Agency": ""},
    color_discrete_sequence=["#2a78d6"],
)
fig_cov.update_traces(marker_color="#2a78d6")
fig_cov.update_layout(
    height=240,
    margin={"l": 10, "r": 10, "t": 10, "b": 10},
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    font={"color": PRIMARY_INK, "family": "sans-serif"},
    xaxis={"gridcolor": GRIDLINE, "tickformat": ".0%"},
    yaxis={"gridcolor": GRIDLINE},
)
st.plotly_chart(fig_cov, width="stretch")
st.caption(
    "Highways reports the large majority of accidents nationally; Rural Roads and "
    "Expressway cover far less of the road network. This is a real reporting-coverage "
    "bias, not something this project can correct — use the sidebar's agency filter "
    "to see past it on any page."
)

with st.expander("Problem categories (P1–P7) fixed in Phases 1–2"):
    st.markdown(
        """
- **P1** — Three date formats mixed in one column, detected per source file
- **P2** — Two time formats (clock string vs. Excel day fraction)
- **P3** — Schema drift: 4 genuine twin-column pairs coalesced
- **P4** — Inconsistent categories (120/152/89 distinct values) — committed mapping tables
- **P5** — Duplicates: 82 exact rows + 1,702 excess rows from 1,641 `ACC_CODE` groups
- **P6** — Province names mismatched across sources — resolved, 100% match verified
- **P7** — Twelve smaller issues: invalid coordinates, inconsistent BOM, header typos,
  Buddhist/Gregorian years, rainfall's 6-rows-per-province-month shape, ID overflow,
  injury-total mismatches, null/unparseable report dates, incomplete 2026 data,
  registration nulls, and the agency coverage bias above
        """
    )

st.caption(
    f"<span style='color:{MUTED_INK}'>Full write-ups: docs/data_quality_findings.md. "
    "Gate 2 + agency coverage: live from fact_accident. Gate 1: documented pipeline "
    "result (reports/cleaning_log.csv).</span>",
    unsafe_allow_html=True,
)
