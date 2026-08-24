"""Shared filter sidebar, built once and reused on every page.

Widget keys are fixed strings so `st.session_state` carries the selection
across page switches within a session (Streamlit's native multipage
behaviour) — a user filtering to "North + motorcycle" on one page keeps
that scope when they navigate to the next.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from dashboard.lib.data_access import (
    AGENCY_LABELS,
    Filters,
    accident_date_bounds,
    list_agencies,
    list_provinces,
    list_vehicle_groups,
)


def render_filters() -> Filters:
    provinces = list_provinces()
    vehicles = list_vehicle_groups()
    agencies = list_agencies()
    min_date, max_date = accident_date_bounds()

    st.sidebar.header("Filters")

    default_end = min(max_date, date(2025, 12, 31))
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, default_end),
        min_value=min_date,
        max_value=max_date,
        key="filter_date_range",
        help="Defaults to the last complete year (2026 has ~4 months of data "
        "and is excluded from trend comparisons by default — extend the range "
        "to include it).",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        # Mid-selection (only the start date picked so far) — hold the previous
        # complete range rather than crashing on an incomplete widget value.
        start_date, end_date = min_date, default_end

    province_options = provinces["province_en"].tolist()
    selected_provinces = st.sidebar.multiselect(
        "Province",
        options=province_options,
        default=province_options,
        key="filter_provinces",
    )

    vehicle_options = vehicles["vehicle_group_label"].tolist()
    selected_vehicles = st.sidebar.multiselect(
        "Vehicle group",
        options=vehicle_options,
        default=vehicle_options,
        key="filter_vehicles",
    )

    agency_options = [AGENCY_LABELS.get(a, a) for a in agencies]
    agency_label_to_raw = dict(zip(agency_options, agencies, strict=True))
    selected_agency_labels = st.sidebar.multiselect(
        "Reporting agency",
        options=agency_options,
        default=agency_options,
        key="filter_agencies",
        help="Highways reports 92.6% of accidents; Rural Roads and Expressway "
        "cover far less of the road network (CLAUDE.md limitation L1). Use this "
        "to see past that coverage bias.",
    )

    if not selected_provinces or not selected_vehicles or not selected_agency_labels:
        st.sidebar.warning("Select at least one option in every filter to see data.")

    geography_sks = tuple(
        int(sk)
        for sk in provinces.loc[provinces["province_en"].isin(selected_provinces), "geography_sk"]
    )
    vehicle_sks = tuple(
        int(sk)
        for sk in vehicles.loc[
            vehicles["vehicle_group_label"].isin(selected_vehicles), "vehicle_sk"
        ]
    )
    selected_agencies = tuple(agency_label_to_raw[label] for label in selected_agency_labels)

    return Filters(
        start_date=start_date,
        end_date=end_date,
        geography_sks=geography_sks,
        vehicle_sks=vehicle_sks,
        agencies=selected_agencies,
    )
