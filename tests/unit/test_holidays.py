import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import requests

from rri.io import holidays as holidays_module
from rri.io.holidays import (
    build_daily_flags,
    classify_holiday_type,
    cross_check_existing_file,
    cross_check_logicspark_2023,
    fetch_holidays_package,
    fetch_logicspark_year,
    validate_holiday_count,
    validate_new_year,
    validate_songkran,
)


class TestClassifyHolidayType:
    def test_substitution_marker_thai(self) -> None:
        holiday_type, is_sub = classify_holiday_type(
            "Chakri Day (Substitution)", "ชดเชยวันจักรี", is_government_only=False
        )
        assert holiday_type == "substitution_holiday"
        assert is_sub is True

    def test_substitution_marker_english(self) -> None:
        holiday_type, is_sub = classify_holiday_type(
            "Songkran Festival (in lieu)", "", is_government_only=False
        )
        assert holiday_type == "substitution_holiday"
        assert is_sub is True

    def test_special_bridge_holiday(self) -> None:
        holiday_type, is_sub = classify_holiday_type(
            "Bridge Public Holiday", "", is_government_only=False
        )
        assert holiday_type == "special_holiday"
        assert is_sub is False

    def test_government_only_holiday(self) -> None:
        holiday_type, is_sub = classify_holiday_type(
            "Royal Ploughing Ceremony", "", is_government_only=True
        )
        assert holiday_type == "government_holiday"
        assert is_sub is False

    def test_ordinary_public_holiday(self) -> None:
        holiday_type, is_sub = classify_holiday_type(
            "New Year's Day", "วันขึ้นปีใหม่", is_government_only=False
        )
        assert holiday_type == "public_holiday"
        assert is_sub is False


class TestFetchHolidaysPackage:
    def test_2020_songkran_is_postponed_not_hidden(self) -> None:
        """Historical fact, not a bug: 2020 Songkran moved to Jul/Sep due to COVID-19."""
        entries = fetch_holidays_package([2020])
        dates = {e["date"] for e in entries}

        assert "2020-04-13" not in dates
        assert "2020-04-14" not in dates
        assert "2020-04-15" not in dates
        assert "2020-07-27" in dates or "2020-09-04" in dates

    def test_excludes_childrens_day_and_labor_day(self) -> None:
        entries = fetch_holidays_package([2023])
        names = " ".join(e["name_en"].lower() for e in entries)

        assert "children's day" not in names
        assert "labor day" not in names
        assert "labour day" not in names

    def test_coinciding_holidays_combined_into_one_row(self) -> None:
        """2023-06-03: Visakha Bucha coincided with the Queen's birthday -- one row, not two."""
        entries = fetch_holidays_package([2023])
        june_3_entries = [e for e in entries if e["date"] == "2023-06-03"]

        assert len(june_3_entries) == 1
        assert "/" in june_3_entries[0]["name_en"]

    def test_new_years_day_present(self) -> None:
        entries = fetch_holidays_package([2021])
        assert any(e["date"] == "2021-01-01" for e in entries)


class TestValidateHolidayCount:
    def test_within_range_is_none(self) -> None:
        entries = [{"date": f"2020-01-{i:02d}"} for i in range(1, 16)]  # 15 entries
        assert validate_holiday_count(2020, entries) is None

    def test_below_range_flags(self) -> None:
        entries = [{"date": "2020-01-01"}]
        assert validate_holiday_count(2020, entries) is not None

    def test_above_range_flags(self) -> None:
        entries = [{"date": f"2020-01-{i:02d}"} for i in range(1, 27)]  # 26 entries
        assert validate_holiday_count(2020, entries) is not None


class TestValidateSongkran:
    def test_all_three_days_present_passes(self) -> None:
        entries = [{"date": "2025-04-13"}, {"date": "2025-04-14"}, {"date": "2025-04-15"}]
        assert validate_songkran(2025, entries) is None

    def test_missing_day_fails(self) -> None:
        entries = [{"date": "2025-04-13"}, {"date": "2025-04-14"}]
        assert validate_songkran(2025, entries) is not None

    def test_2020_postponement_is_flagged(self) -> None:
        entries = [{"date": "2020-07-27"}]
        message = validate_songkran(2020, entries)
        assert message is not None
        assert "2020-04-13" in message


class TestValidateNewYear:
    def test_jan_1_present_passes(self) -> None:
        assert validate_new_year(2025, [{"date": "2025-01-01"}]) is None

    def test_jan_1_missing_fails(self) -> None:
        assert validate_new_year(2025, [{"date": "2025-01-02"}]) is not None


class TestBuildDailyFlags:
    """Synthetic 10-day calendar: Fri Jan 3 - Sun Jan 12, 2025.

    Jan 1 (Wed) and Jan 13 (Mon) are outside the window on purpose, to
    check the function doesn't need lookahead/lookbehind beyond its range.
    A holiday block: Sat Jan 4 - Mon Jan 6 (weekend + one holiday Monday)
    -- a 3-day non-working run, so Fri Jan 3 should be a holiday eve.
    """

    START = date(2025, 1, 3)
    END = date(2025, 1, 12)
    PUBLIC_HOLIDAYS = {date(2025, 1, 6)}  # a Monday
    BANK_HOLIDAYS: set[date] = set()

    def _flags_by_date(self) -> dict[date, Any]:
        flags = build_daily_flags(
            self.START, self.END, self.PUBLIC_HOLIDAYS, self.BANK_HOLIDAYS, {2025}
        )
        return {f.date: f for f in flags}

    def test_covers_every_day_in_range(self) -> None:
        flags = self._flags_by_date()
        assert len(flags) == 10
        assert date(2025, 1, 3) in flags
        assert date(2025, 1, 12) in flags

    def test_weekend_detection(self) -> None:
        flags = self._flags_by_date()
        assert flags[date(2025, 1, 4)].is_weekend is True  # Saturday
        assert flags[date(2025, 1, 6)].is_weekend is False  # Monday (holiday, not weekend)
        assert flags[date(2025, 1, 3)].is_weekend is False  # Friday

    def test_three_day_run_is_long_weekend(self) -> None:
        flags = self._flags_by_date()
        assert flags[date(2025, 1, 4)].is_long_weekend is True
        assert flags[date(2025, 1, 5)].is_long_weekend is True
        assert flags[date(2025, 1, 6)].is_long_weekend is True
        assert flags[date(2025, 1, 3)].is_long_weekend is False
        assert flags[date(2025, 1, 7)].is_long_weekend is False

    def test_holiday_eve_is_the_working_day_before_the_run(self) -> None:
        flags = self._flags_by_date()
        assert flags[date(2025, 1, 3)].is_holiday_eve is True
        assert flags[date(2025, 1, 4)].is_holiday_eve is False  # already non-working

    def test_days_to_next_holiday_counts_down_to_zero(self) -> None:
        flags = self._flags_by_date()
        assert flags[date(2025, 1, 3)].days_to_next_holiday == 3
        assert flags[date(2025, 1, 5)].days_to_next_holiday == 1
        assert flags[date(2025, 1, 6)].days_to_next_holiday == 0
        assert flags[date(2025, 1, 7)].days_to_next_holiday is None  # no more holidays in range

    def test_bank_holiday_is_none_outside_known_years(self) -> None:
        flags = build_daily_flags(
            self.START, self.END, self.PUBLIC_HOLIDAYS, self.BANK_HOLIDAYS, set()
        )
        assert all(f.is_bank_holiday is None for f in flags)

    def test_songkran_and_newyear_windows(self) -> None:
        flags = build_daily_flags(date(2025, 4, 10), date(2025, 4, 18), set(), set(), set())
        by_date = {f.date: f for f in flags}
        assert by_date[date(2025, 4, 11)].is_songkran_7days is True
        assert by_date[date(2025, 4, 17)].is_songkran_7days is True
        assert by_date[date(2025, 4, 10)].is_songkran_7days is False
        assert by_date[date(2025, 4, 18)].is_songkran_7days is False

        newyear_flags = build_daily_flags(date(2025, 12, 27), date(2026, 1, 5), set(), set(), set())
        by_date2 = {f.date: f for f in newyear_flags}
        assert by_date2[date(2025, 12, 29)].is_newyear_7days is True
        assert by_date2[date(2026, 1, 4)].is_newyear_7days is True
        assert by_date2[date(2025, 12, 28)].is_newyear_7days is False
        assert by_date2[date(2026, 1, 5)].is_newyear_7days is False


class TestFetchLogicsparkYear:
    def test_returns_none_on_request_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_error(*args: object, **kwargs: object) -> None:
            raise requests.RequestException("boom")

        monkeypatch.setattr(requests, "get", raise_error)
        assert fetch_logicspark_year(2023) is None

    def test_returns_none_on_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeResponse:
            status_code = 404

        monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse())
        assert fetch_logicspark_year(2019) is None

    def test_returns_parsed_json_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeResponse:
            status_code = 200

            def json(self) -> list[dict[str, str]]:
                return [{"date": "2023-01-01", "nameEn": "New Year's Day"}]

        monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse())
        result = fetch_logicspark_year(2023)
        assert result == [{"date": "2023-01-01", "nameEn": "New Year's Day"}]


class TestCrossCheckLogicspark2023:
    def test_no_mismatch_when_dates_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            holidays_module, "fetch_logicspark_year", lambda year: [{"date": "2023-01-01"}]
        )
        package_entries = [{"date": "2023-01-01"}]
        assert cross_check_logicspark_2023(package_entries) == []

    def test_reports_dates_only_in_each_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            holidays_module,
            "fetch_logicspark_year",
            lambda year: [{"date": "2023-01-01"}, {"date": "2023-05-01"}],
        )
        package_entries = [{"date": "2023-01-01"}, {"date": "2023-12-29"}]
        messages = cross_check_logicspark_2023(package_entries)

        assert any("2023-05-01" in m for m in messages)
        assert any("2023-12-29" in m for m in messages)

    def test_reports_when_fetch_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(holidays_module, "fetch_logicspark_year", lambda year: None)
        messages = cross_check_logicspark_2023([{"date": "2023-01-01"}])
        assert len(messages) == 1
        assert "could not fetch" in messages[0]


class TestCrossCheckExistingFile:
    def test_reports_missing_and_extra_dates(self, tmp_path: Path) -> None:
        existing_file = tmp_path / "public_holidays_all.json"
        existing_file.write_text(
            json.dumps(
                {
                    "years": [
                        {
                            "year": 2025,
                            "holidays": [{"date": "2025-01-01"}, {"date": "2025-02-12"}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        package_entries = [{"date": "2025-01-01"}, {"date": "2025-05-09"}]
        messages = cross_check_existing_file(package_entries, existing_file)

        assert any("2025-02-12" in m and "but not fetched" in m for m in messages)
        assert any("2025-05-09" in m and "not in" in m for m in messages)

    def test_no_messages_when_identical(self, tmp_path: Path) -> None:
        existing_file = tmp_path / "public_holidays_all.json"
        existing_file.write_text(
            json.dumps({"years": [{"year": 2025, "holidays": [{"date": "2025-01-01"}]}]}),
            encoding="utf-8",
        )

        assert cross_check_existing_file([{"date": "2025-01-01"}], existing_file) == []
