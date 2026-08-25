"""Regression tests for `dashboard/pages/4_Seasonality_Festivals.py`.

**The bug:** the holiday-vs-normal comparison panel hardcoded
`col1, col2, col3 = st.columns(3)` and zipped it (`strict=True`) against
`comparison.iterrows()`. `comparison` groups accidents into up to 3 periods
(Songkran / New Year / Other days), but a narrow date filter routinely
excludes one or both holiday windows -- e.g. any single-month range that
isn't April or December produces exactly 1 row ("Other days" only). That
raised `ValueError: zip() argument 2 is shorter than argument 1` at runtime
on the deployed app.

The deployed app's *default* filter range (2019-01-01 to 2025-12-31, all
provinces/vehicles/agencies) happens to produce exactly 3 rows -- which is
why this shipped without being caught by a smoke test of the default view.
The crash only appears once a user narrows the date range, which is exactly
what CLAUDE.md Phase 7's own filter sidebar invites them to do. These tests
cover the default (control, must still pass) plus several narrower ranges
that are real, easy-to-reach reproductions of 1-row and 2-row cases.

Fix: `st.columns(len(comparison))` -- column count now always matches row
count, including the 0-row case (already separately guarded by the
`comparison.empty` branch above it).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(
    Path(__file__).resolve().parents[2] / "dashboard" / "pages" / "4_Seasonality_Festivals.py"
)


def _run_with_date_range(start: date, end: date) -> AppTest:
    at = AppTest.from_file(PAGE)
    at.run()
    at.session_state["filter_start_date"] = start
    at.session_state["filter_end_date"] = end
    at.run()
    return at


def test_deployed_default_range_renders_without_exception() -> None:
    """Control case: the deployed app's actual default filter (full history,
    all provinces/vehicles/agencies) -- produces 3 periods, must keep working."""
    at = AppTest.from_file(PAGE)
    at.run()
    assert at.exception == []
    assert len(at.columns) >= 3


def test_single_ordinary_month_produces_one_period_no_crash() -> None:
    """June 2025 contains neither Songkran (April) nor New Year (Dec-Jan) --
    exactly the 1-row case that crashed with a hardcoded st.columns(3)."""
    at = _run_with_date_range(date(2025, 6, 1), date(2025, 6, 30))
    assert at.exception == []


def test_april_only_produces_two_periods_no_crash() -> None:
    """April 2025 contains Songkran + ordinary days, but no New Year window --
    exactly the 2-row case."""
    at = _run_with_date_range(date(2025, 4, 1), date(2025, 4, 30))
    assert at.exception == []


def test_december_only_produces_two_periods_no_crash() -> None:
    """December 2025 contains New Year + ordinary days, but no Songkran window."""
    at = _run_with_date_range(date(2025, 12, 1), date(2025, 12, 31))
    assert at.exception == []


def test_single_day_produces_one_period_no_crash() -> None:
    at = _run_with_date_range(date(2025, 6, 15), date(2025, 6, 15))
    assert at.exception == []


def test_column_count_always_matches_row_count() -> None:
    """Guards the actual fix, not just 'no exception' -- the number of
    st.columns() calls in the comparison panel must equal len(comparison)
    for every scope, not a hardcoded 3."""
    for start, end, expected_periods in [
        (date(2019, 1, 1), date(2025, 12, 31), 3),
        (date(2025, 6, 1), date(2025, 6, 30), 1),
        (date(2025, 4, 1), date(2025, 4, 30), 2),
    ]:
        at = _run_with_date_range(start, end)
        assert at.exception == []
        metrics = at.metric
        assert len(metrics) == expected_periods, (
            f"range {start}..{end}: expected {expected_periods} period metric(s), "
            f"got {len(metrics)}"
        )
