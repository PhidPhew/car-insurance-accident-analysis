"""Page 6 — Forecast (BQ6).

Which provinces will spike in the next 6 months? How much reserve and
staffing is needed? Every number traced to `fact_accident` via
`dashboard.lib.data_access` (CLAUDE.md Phase 7 rule).

**Honesty note (see data_access.forecast_series's docstring)**: Phase 6's
GRU did not beat a seasonal-naive baseline on this data (Poisson deviance
14.64 vs. 7.40) — this page forecasts with the validated-better
seasonal-naive method, not a live GRU inference pipeline, and shows a wide,
explicitly-labeled heuristic uncertainty band rather than a false-precision
model confidence interval.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.graph_objects as go
import streamlit as st

from config.settings import settings
from dashboard.lib.data_access import Filters, forecast_series
from dashboard.lib.filters import render_filters
from dashboard.lib.theme import (
    BLUE,
    GRIDLINE,
    MUTED_INK,
    PRIMARY_INK,
    STATUS_WARNING,
    bq_tag,
    inject_theme,
    page_header,
)

st.set_page_config(page_title="Forecast | SiamShield", page_icon="📈", layout="wide")
inject_theme()
page_header(
    "Forecast",
    "Which provinces will spike in the next 6 months? How much reserve and staffing is needed?",
)

f: Filters = render_filters()

no_data = not (f.geography_sks and f.vehicle_sks and f.agencies)
if no_data:
    st.info("Select at least one option in every sidebar filter to see data.")
    st.stop()

bq_tag("BQ6")
st.markdown("#### 6-month accident forecast")
st.info(
    "**Model used: seasonal-naive**, not the GRU. Phase 6 tested a global GRU "
    "against a seasonal-naive baseline (forecast month *m* with the actual count "
    "from month *m − 12*) and the GRU **lost** (Poisson deviance 14.64 vs. 7.40) — "
    "539 windows was too thin for a sequence model to beat a strongly seasonal "
    "series. Reporting that honestly means forecasting with the method that "
    "actually validated better, not the more sophisticated-looking one.",
    icon="ℹ️",
)
st.caption(
    "Uncertainty band = ± this scope's own historical year-over-year variability — "
    "a heuristic spread, not a fitted model confidence interval."
)

series = forecast_series(f)
if series.empty:
    st.info("No accidents in the selected filters.")
else:
    history = series[series["kind"] == "history"]
    forecast = series[series["kind"] == "forecast"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["month"],
            y=history["accident_count"],
            mode="lines",
            line={"color": BLUE, "width": 2},
            name="Actual",
            hovertemplate="%{x|%b %Y}<br>%{y:,.0f} accidents<extra></extra>",
        )
    )
    if not forecast.empty:
        bridge_x = [history["month"].iloc[-1], *forecast["month"]]
        bridge_y = [history["accident_count"].iloc[-1], *forecast["accident_count"]]
        fig.add_trace(
            go.Scatter(
                x=bridge_x,
                y=bridge_y,
                mode="lines+markers",
                line={"color": STATUS_WARNING, "width": 2, "dash": "dash"},
                name="Forecast (seasonal-naive)",
                hovertemplate="%{x|%b %Y}<br>%{y:,.0f} accidents (forecast)<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[*forecast["month"], *forecast["month"][::-1]],
                y=[*forecast["upper"], *forecast["lower"][::-1]],
                fill="toself",
                fillcolor="rgba(250, 178, 25, 0.18)",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                name="Uncertainty band",
            )
        )
    fig.update_layout(
        height=380,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font={"color": PRIMARY_INK, "family": "sans-serif"},
        xaxis={"gridcolor": GRIDLINE, "showgrid": False},
        yaxis={"gridcolor": GRIDLINE, "title": "Accidents / month"},
        legend={"orientation": "h", "y": 1.1},
    )
    st.plotly_chart(fig, width="stretch")

    if not forecast.empty:
        total_forecast = forecast["accident_count"].sum()
        prior_6 = history["accident_count"].tail(6).sum()
        st.caption(
            f"Forecast total (next 6 months): {total_forecast:,.0f} accidents, vs. "
            f"{prior_6:,.0f} in the previous 6 months on record "
            f"({(total_forecast / prior_6 - 1) * 100:+.0f}%)."
            if prior_6
            else ""
        )

st.markdown("#### Province risk profiles (embeddings)")
st.caption(
    "From Phase 6's MLP entity-embedding severity classifier: provinces close "
    "together learned similar risk profiles, independent of any hand-picked "
    "feature. t-SNE projection of 77 real province embeddings (English labels — "
    "no verified Thai-font fallback for matplotlib in this environment)."
)
tsne_path = settings.reports_dir / "province_embeddings_tsne.png"
if tsne_path.exists():
    st.image(str(tsne_path), width="stretch")
else:
    st.info(f"Report image not found at {tsne_path}.")

st.caption(
    f"<span style='color:{MUTED_INK}'>Forecast: fact_accident, full warehouse "
    "history for the selected province/vehicle/agency scope. Embeddings: "
    "reports/province_embeddings_tsne.png (Phase 6 artifact).</span>",
    unsafe_allow_html=True,
)
