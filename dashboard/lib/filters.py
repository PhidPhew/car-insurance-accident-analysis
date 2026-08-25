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

    # Two independent single-date widgets, deliberately NOT one range
    # st.date_input. Streamlit's range date_input is a BaseWeb
    # DateRangePicker with a client-side (frontend) defect: clicking the
    # currently-highlighted start boundary of an already-complete range
    # silently swaps in the OTHER endpoint as the new start and jumps the
    # calendar to a different month -- confirmed via a live Playwright
    # browser session AND reproduced in a minimal repro with zero relation
    # to this app's code (a bare `st.date_input` with no `value=` argument
    # at all, in an otherwise-empty script). It is not something any
    # Python-side widget invocation can fix -- AppTest cannot see it either,
    # since AppTest drives session_state directly and never exercises the
    # real frontend calendar. Do not "simplify" this back into one range
    # widget; see tests/unit/test_filters.py's docstring for the same note.
    #
    # Seeding each key's session_state once (never re-passing `value=` on
    # every rerun) is still the idiomatic pattern -- kept here for the same
    # reason it mattered for the range widget, even though single-date
    # inputs aren't known to have the swap defect.
    if "filter_start_date" not in st.session_state:
        st.session_state["filter_start_date"] = min_date
    if "filter_end_date" not in st.session_state:
        st.session_state["filter_end_date"] = default_end

    date_col1, date_col2 = st.sidebar.columns(2)
    with date_col1:
        picked_start = st.date_input(
            "Start date",
            min_value=min_date,
            max_value=max_date,
            key="filter_start_date",
        )
    with date_col2:
        picked_end = st.date_input(
            "End date",
            min_value=min_date,
            max_value=max_date,
            key="filter_end_date",
            help="Defaults to the last complete year (2026 has ~4 months of data "
            "and is excluded from trend comparisons by default — extend the range "
            "to include it).",
        )

    if picked_start <= picked_end:
        start_date, end_date = picked_start, picked_end
        st.session_state["_last_complete_date_range"] = (start_date, end_date)
    else:
        # Two independent inputs means an invalid order (end before start) is
        # now reachable and is this code's responsibility to handle -- hold
        # the last valid range and tell the user why, rather than crash or
        # silently invert/clamp their explicit picks.
        st.sidebar.warning(
            f"Start date ({picked_start}) is after end date ({picked_end}) — "
            "holding the last valid range until this is corrected."
        )
        start_date, end_date = st.session_state.get(
            "_last_complete_date_range", (min_date, default_end)
        )

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
