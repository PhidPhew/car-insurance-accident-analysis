"""SiamShield Road Risk Intelligence dashboard — entry point.

Reads only from the built warehouse (`data/30_warehouse/warehouse.duckdb`)
and calls persisted models' `predict()` interface (`rri.models.persistence`)
— it never recomputes cleaning, features, or training (CLAUDE.md Phase 7).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `dashboard.lib.*` imports when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from dashboard.lib.data_access import accident_date_bounds
from dashboard.lib.theme import inject_theme, page_header

st.set_page_config(
    page_title="SiamShield Road Risk Intelligence",
    page_icon="🚗",
    layout="wide",
)
inject_theme()

page_header(
    "SiamShield Insurance — Road Risk Intelligence",
    "Weather-aware motor risk analytics: location × time × vehicle type × weather, "
    "built on real accident data (2019–2026) and OIC-calibrated simulated policy figures.",
)

st.markdown(
    """
This dashboard answers SiamShield's 8 core business questions from the warehouse
built in Phases 1–6 (real accidents, simulated policy portfolio, and 7 trained
models). Use the sidebar to open a page.

**Available now**
- **Executive Overview** — the 6 headline KPIs, trend, and provinces to watch.

**Coming in later batches**: Spatial Risk, Weather Impact, Seasonality & Festivals,
Portfolio Segments, Forecast, Pricing Simulator, EV Analysis + Data Quality.
"""
)

start, end = accident_date_bounds()
st.caption(
    f"Warehouse data: {start:%Y-%m-%d} to {end:%Y-%m-%d} · "
    "159,605 accident records · simulated portfolio 2022-01 to 2025-12."
)
