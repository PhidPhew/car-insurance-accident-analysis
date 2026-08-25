"""Regression tests for the sidebar date filter (`dashboard/lib/filters.py`).

**Design note for whoever touches this file next:** `render_filters()` uses
TWO independent single-date `st.date_input` widgets ("Start date" / "End
date"), not one `st.date_input` range widget, and that is deliberate --
do not "simplify" it back to a single range widget.

The range widget (`st.date_input(..., value=(start, end))`) has a
**client-side** defect in Streamlit's BaseWeb `DateRangePicker`: clicking the
currently-highlighted start boundary of an already-complete range silently
swaps in the OTHER endpoint as the new start and jumps the calendar to a
different month, producing a garbage selection. This was confirmed via a
live Playwright browser session against the running dashboard, and then
reproduced in a *minimal* single-widget repro script with zero relation to
this app's code (no custom CSS, no other widgets, `value=` argument omitted
entirely) -- proving it is inherent to Streamlit's frontend component, not
something fixable by changing how the widget is invoked from Python.

**AppTest cannot see that bug at all.** AppTest drives `st.session_state`
directly and never renders or clicks the real frontend calendar, so an
AppTest suite against the old range widget would stay green while the
browser-observed bug persisted -- false confidence for anyone who "fixes"
this back to a range widget and reruns only `pytest`. Two single-date
widgets sidestep the defective component entirely, which is why AppTest
*is* appropriate for regression-testing this version: with no range
calendar involved, the remaining failure modes (default seeding, an
invalid start > end order, cross-page session persistence) are pure
Python/session_state logic that AppTest exercises faithfully.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[2] / "dashboard" / "pages" / "1_Executive_Overview.py")


def _widget_dates(at: AppTest) -> tuple[date, date]:
    start = at.date_input(key="filter_start_date").value
    end = at.date_input(key="filter_end_date").value
    assert isinstance(start, date)
    assert isinstance(end, date)
    return start, end


def test_initial_render_has_two_single_date_widgets_not_a_range_widget() -> None:
    at = AppTest.from_file(PAGE)
    at.run()
    assert at.exception == []

    date_widgets = at.date_input
    assert len(date_widgets) == 2, (
        "expected exactly two single-date widgets (Start date, End date) -- "
        "if this now finds one widget, someone reverted to the range "
        "st.date_input, which reintroduces the browser-only swap bug this "
        "test suite cannot detect on its own (see module docstring)"
    )
    for widget in date_widgets:
        assert isinstance(widget.value, date), (
            f"widget {widget.label!r} returned {type(widget.value)}, expected a "
            "single date -- a tuple return means this is a range widget again"
        )


def test_default_dates_match_expected_bounds() -> None:
    at = AppTest.from_file(PAGE)
    at.run()

    start, end = _widget_dates(at)
    assert start == date(2019, 1, 1)
    # default_end = min(max_date, 2025-12-31); the warehouse's own max_date
    # covers into 2026, so this is exactly 2025-12-31 (CLAUDE.md P7.11 holdout).
    assert end == date(2025, 12, 31)


def test_start_after_end_holds_last_valid_range_and_warns() -> None:
    """Two independent inputs make an invalid order reachable -- this is now
    render_filters()'s responsibility to handle explicitly (per the project
    owner's requirement), not something a single range widget could even
    produce."""
    at = AppTest.from_file(PAGE)
    at.run()

    # Establish a known-valid range first.
    at.session_state["filter_start_date"] = date(2024, 3, 1)
    at.session_state["filter_end_date"] = date(2024, 6, 30)
    at.run()
    assert at.exception == []
    assert at.session_state["_last_complete_date_range"] == (
        date(2024, 3, 1),
        date(2024, 6, 30),
    )

    # Now set an invalid order: start after end.
    at.session_state["filter_start_date"] = date(2024, 12, 1)
    at.session_state["filter_end_date"] = date(2024, 1, 1)
    at.run()

    assert at.exception == []
    warnings = [w.value for w in at.sidebar.warning]
    assert any("after end date" in w for w in warnings), (
        f"expected an inline warning about the invalid order, got: {warnings}"
    )
    # The held range must be the last VALID one, not the global default and
    # not a silent swap/clamp of the invalid picks.
    assert at.session_state["_last_complete_date_range"] == (
        date(2024, 3, 1),
        date(2024, 6, 30),
    )


def test_valid_order_after_invalid_recovers_cleanly() -> None:
    at = AppTest.from_file(PAGE)
    at.run()

    at.session_state["filter_start_date"] = date(2024, 12, 1)
    at.session_state["filter_end_date"] = date(2024, 1, 1)
    at.run()
    assert any("after end date" in w.value for w in at.sidebar.warning)

    at.session_state["filter_start_date"] = date(2023, 1, 1)
    at.session_state["filter_end_date"] = date(2023, 12, 31)
    at.run()

    assert at.exception == []
    assert not any("after end date" in w.value for w in at.sidebar.warning)
    assert at.session_state["_last_complete_date_range"] == (
        date(2023, 1, 1),
        date(2023, 12, 31),
    )


def test_date_selection_persists_across_session_state_like_other_filters() -> None:
    """Matches the module's own stated contract (docstring of filters.py):
    widget keys are fixed strings so a selection persists across page
    switches within a session."""
    at = AppTest.from_file(PAGE)
    at.run()

    at.session_state["filter_start_date"] = date(2022, 5, 1)
    at.session_state["filter_end_date"] = date(2022, 8, 31)
    at.run()

    assert at.session_state["filter_start_date"] == date(2022, 5, 1)
    assert at.session_state["filter_end_date"] == date(2022, 8, 31)

    # Simulate switching pages: a fresh AppTest run against a different page
    # can't share in-process session_state, but within one running app
    # instance (the real multipage scenario) the same session_state dict is
    # reused across page scripts -- assert the keys survive another rerun of
    # the same script unchanged, which is the mechanism multipage navigation
    # relies on.
    at.run()
    assert at.session_state["filter_start_date"] == date(2022, 5, 1)
    assert at.session_state["filter_end_date"] == date(2022, 8, 31)
