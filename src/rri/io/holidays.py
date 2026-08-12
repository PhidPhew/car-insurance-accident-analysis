"""Thai public holiday backfill for 2019-2024, and dim_date-ready daily flags.

``public_holidays_all.json`` / ``thai_bank_holidays_all.json`` only cover
2025-2026, but accident data spans 2019-2026 (CLAUDE.md Task 2). This module
fetches the missing years from external sources rather than inventing dates
itself, then derives the calendar flags Task 2 asks for.

Source strategy (CLAUDE.md's stated fallback order, adapted to what each
source actually offers):

1. GitHub ``logicspark/thailand-holidays`` -- as of writing, this repo's own
   README says it has only ever published one year of data (2023). It
   cannot serve as the primary source for 2019-2024 as a whole, so it is
   used here only as a cross-check against source 2 for the one year it
   covers (see ``cross_check_logicspark_2023``).
2. The ``holidays`` Python package (``holidays.Thailand``) -- the primary
   source for all of 2019-2024. It deterministically derives Thai
   lunar-calendar holidays (Buddhist observances, royal birthdays, etc.)
   for a requested year range with no network dependency, so backfilling
   with a single consistent method keeps N9 (determinism) intact and
   doesn't add an external-service runtime dependency for 5 of the 6
   backfilled years.

If source 2 raises or returns nothing, this module raises -- it does not
invent a holiday date.

The package doesn't expose the exact ``type`` taxonomy the real
DGA-sourced 2025/2026 files use (public_holiday / substitution_holiday /
special_holiday / government_holiday), so entries are classified into it
by name-string heuristics (see ``classify_holiday_type``). This is a
documented limitation: backfilled years won't carry the same
cabinet-resolution-specific Thai notes the real gazette files have (e.g.
"ตรงกับวันอาทิตย์").
"""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
from holidays.countries.thailand import Thailand as ThailandHolidays

from rri.utils.logging import get_logger

LOGICSPARK_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/logicspark/thailand-holidays/main/download/"
    "th-holiday-{year}.json"
)

SUBSTITUTION_MARKERS = ("ชดเชย", "in lieu")
SPECIAL_MARKERS = ("วันหยุดพิเศษ", "bridge", "special")

# The `holidays` package's "public" category is broader than the real
# public_holidays_all.json: it also includes observance/private-sector days
# (National Children's Day, National Labor Day) that a cross-check against
# the real 2025-2026 file shows are absent there. public_holidays_all.json
# is the government-office holiday calendar, not the full labour-law
# calendar, so these are excluded to keep 2019-2024 consistent in scope
# with 2025-2026.
EXCLUDED_NAME_MARKERS_EN = ("children's day", "labor day", "labour day")

MIN_EXPECTED_HOLIDAYS_PER_YEAR = 13
MAX_EXPECTED_HOLIDAYS_PER_YEAR = 25
SONGKRAN_DAYS = (13, 14, 15)
LONG_HOLIDAY_MIN_DAYS = 3

# 2020: Songkran (13-15 April) was postponed nationwide by Cabinet
# Resolution due to COVID-19 and rescheduled as substitute holidays on
# 27 Jul and 4/7 Sep 2020. This is real history, not a fetch error -- the
# "Songkran must be 13-15 April" validation is expected to fail for this
# one year and should be logged as a documented exception, not a bug.
KNOWN_SONGKRAN_EXCEPTIONS = {
    2020: "Postponed nationwide due to COVID-19 (Cabinet Resolution); "
    "rescheduled to 27 Jul and 4/7 Sep 2020"
}

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def classify_holiday_type(
    name_en: str, name_th: str, *, is_government_only: bool
) -> tuple[str, bool]:
    """Map a holiday name to CLAUDE.md's type taxonomy via name heuristics.

    Returns (type, is_substitution). `is_government_only` should be True
    when the date came from the `holidays` package's "government" category
    but not its "public" category (e.g. Royal Ploughing Ceremony Day).
    """
    combined = f"{name_en} {name_th}".lower()
    if any(marker.lower() in combined for marker in SUBSTITUTION_MARKERS):
        return "substitution_holiday", True
    if is_government_only:
        return "government_holiday", False
    if any(marker.lower() in combined for marker in SPECIAL_MARKERS):
        return "special_holiday", False
    return "public_holiday", False


def _is_excluded(name_en: str) -> bool:
    lowered = name_en.lower()
    return any(marker in lowered for marker in EXCLUDED_NAME_MARKERS_EN)


def fetch_holidays_package(years: list[int]) -> list[dict[str, Any]]:
    """Fetch Thai holidays for `years` from the `holidays` package.

    One dict per calendar date (a date with two coinciding holidays -- e.g.
    Visakha Bucha falling on the Queen's birthday -- produces ONE entry
    with both names joined by " / "), matching the real
    public_holidays_all.json convention (see e.g. its 2025-12-05 entry,
    which combines three titles into one row). Field shape: date, name_th,
    name_en, type, is_substitution, note.

    Excludes names matching EXCLUDED_NAME_MARKERS_EN (see its docstring).
    """
    # `holidays` ships py.typed but HolidayBase.__init__ itself isn't fully
    # annotated, so mypy --strict flags these as untyped calls regardless.
    public_en = ThailandHolidays(  # type: ignore[no-untyped-call]
        years=years, language="en_US", categories=("public",)
    )
    public_th = ThailandHolidays(  # type: ignore[no-untyped-call]
        years=years, language="th", categories=("public",)
    )
    gov_en = ThailandHolidays(  # type: ignore[no-untyped-call]
        years=years, language="en_US", categories=("government",)
    )
    gov_th = ThailandHolidays(  # type: ignore[no-untyped-call]
        years=years, language="th", categories=("government",)
    )

    all_dates = sorted(set(public_en.keys()) | set(gov_en.keys()))

    entries: list[dict[str, Any]] = []
    for d in all_dates:
        is_government_only = d not in public_en
        source_en = gov_en if is_government_only else public_en
        source_th = gov_th if is_government_only else public_th

        names_en = [n.strip() for n in source_en[d].split(";")]
        names_th = [n.strip() for n in source_th[d].split(";")]
        if len(names_en) != len(names_th):
            # Fall back to English names for both rather than misaligning
            # Thai/English pairs across a coincidence day.
            names_th = names_en

        kept = [(en, th) for en, th in zip(names_en, names_th, strict=True) if not _is_excluded(en)]
        if not kept:
            continue

        combined_en = " / ".join(en for en, _ in kept)
        combined_th = " / ".join(th for _, th in kept)
        # A day counts as a substitution/government holiday only if ALL of
        # its coinciding names are -- a mixed day (e.g. a regular holiday
        # coinciding with a substitution) is classified by its first name,
        # matching how the real file's own combined entries read.
        holiday_type, is_substitution = classify_holiday_type(
            kept[0][0], kept[0][1], is_government_only=is_government_only
        )
        entries.append(
            {
                "date": d.isoformat(),
                "name_th": combined_th,
                "name_en": combined_en,
                "type": holiday_type,
                "is_substitution": is_substitution,
                "note": None,
            }
        )
    return entries


def fetch_logicspark_year(year: int, timeout: float = 10.0) -> list[dict[str, Any]] | None:
    """Fetch one year of logicspark/thailand-holidays raw JSON, or None if unavailable.

    The repo has only ever published 2023 data (per its own README), so
    None is the expected return for every other year, not an error.
    """
    url = LOGICSPARK_URL_TEMPLATE.format(year=year)
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        log.warning("logicspark_fetch_failed", year=year, error=str(exc))
        return None
    if response.status_code != 200:
        return None
    result: list[dict[str, Any]] = response.json()
    return result


def cross_check_logicspark_2023(package_entries: list[dict[str, Any]]) -> list[str]:
    """Compare source 1's 2023 dates against source 2's. Returns mismatch messages."""
    logicspark_data = fetch_logicspark_year(2023)
    if logicspark_data is None:
        return ["logicspark/thailand-holidays: could not fetch 2023 data for cross-check"]

    logicspark_dates = {entry["date"] for entry in logicspark_data}
    package_dates = {e["date"] for e in package_entries if e["date"].startswith("2023")}

    messages = []
    only_in_logicspark = sorted(logicspark_dates - package_dates)
    only_in_package = sorted(package_dates - logicspark_dates)
    if only_in_logicspark:
        messages.append(f"2023 dates only in logicspark: {only_in_logicspark}")
    if only_in_package:
        messages.append(f"2023 dates only in holidays-package: {only_in_package}")
    return messages


def cross_check_existing_file(
    package_entries: list[dict[str, Any]], existing_file: Path
) -> list[str]:
    """Compare freshly-fetched dates against the real DGA-sourced file for its years."""
    existing = json.loads(existing_file.read_text(encoding="utf-8-sig"))
    existing_dates_by_year: dict[int, set[str]] = defaultdict(set)
    for year_block in existing["years"]:
        for h in year_block["holidays"]:
            existing_dates_by_year[year_block["year"]].add(h["date"])

    messages = []
    for year, existing_dates in existing_dates_by_year.items():
        fetched_dates = {e["date"] for e in package_entries if e["date"].startswith(str(year))}
        missing = sorted(existing_dates - fetched_dates)
        extra = sorted(fetched_dates - existing_dates)
        if missing:
            messages.append(f"{year}: in {existing_file.name} but not fetched: {missing}")
        if extra:
            messages.append(f"{year}: fetched but not in {existing_file.name}: {extra}")
    return messages


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_holiday_count(year: int, entries: list[dict[str, Any]]) -> str | None:
    count = len(entries)
    if not (MIN_EXPECTED_HOLIDAYS_PER_YEAR <= count <= MAX_EXPECTED_HOLIDAYS_PER_YEAR):
        return (
            f"{year}: {count} holidays, outside the expected "
            f"{MIN_EXPECTED_HOLIDAYS_PER_YEAR}-{MAX_EXPECTED_HOLIDAYS_PER_YEAR} range"
        )
    return None


def validate_songkran(year: int, entries: list[dict[str, Any]]) -> str | None:
    dates = {e["date"] for e in entries}
    expected = {date(year, 4, day).isoformat() for day in SONGKRAN_DAYS}
    missing = sorted(expected - dates)
    if missing:
        return f"{year}: Songkran (13-15 April) missing dates: {missing}"
    return None


def validate_new_year(year: int, entries: list[dict[str, Any]]) -> str | None:
    dates = {e["date"] for e in entries}
    if date(year, 1, 1).isoformat() not in dates:
        return f"{year}: New Year's Day (1 Jan) missing"
    return None


# --------------------------------------------------------------------------
# Daily flags (for the future dim_date build in Phase 3)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DailyFlags:
    date: date
    is_public_holiday: bool
    is_bank_holiday: bool | None
    is_weekend: bool
    is_long_weekend: bool
    is_songkran_7days: bool
    is_newyear_7days: bool
    is_holiday_eve: bool
    days_to_next_holiday: int | None


def _non_working_runs(days: list[date], non_working: set[date]) -> dict[date, tuple[date, date]]:
    """Map each non-working day to the (start, end) of its consecutive run.

    Working days are absent from the returned dict.
    """
    runs: dict[date, tuple[date, date]] = {}
    i = 0
    n = len(days)
    while i < n:
        if days[i] not in non_working:
            i += 1
            continue
        j = i
        while j + 1 < n and days[j + 1] in non_working:
            j += 1
        run_start, run_end = days[i], days[j]
        for k in range(i, j + 1):
            runs[days[k]] = (run_start, run_end)
        i = j + 1
    return runs


def _days_to_next_holiday(
    day: date, public_holidays: set[date], sorted_public_holidays: list[date]
) -> int | None:
    if day in public_holidays:
        return 0
    idx = bisect.bisect_right(sorted_public_holidays, day)
    if idx >= len(sorted_public_holidays):
        return None
    return (sorted_public_holidays[idx] - day).days


def build_daily_flags(
    start: date,
    end: date,
    public_holidays: set[date],
    bank_holidays: set[date],
    bank_holiday_known_years: set[int],
) -> list[DailyFlags]:
    """Derive per-day flags for every date in [start, end] inclusive.

    `bank_holiday_known_years` marks which years actually have real bank
    holiday source data (2025-2026 per CLAUDE.md) - is_bank_holiday is
    None (not False) for any other year, since absence of data is not the
    same as absence of a bank holiday.
    """
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    non_working = {d for d in days if d.weekday() >= 5 or d in public_holidays}
    runs = _non_working_runs(days, non_working)
    sorted_public = sorted(public_holidays)

    results: list[DailyFlags] = []
    for d in days:
        run = runs.get(d)
        is_long_weekend = run is not None and (run[1] - run[0]).days + 1 >= LONG_HOLIDAY_MIN_DAYS

        next_day = d + timedelta(days=1)
        next_run = runs.get(next_day)
        is_holiday_eve = (
            d not in non_working
            and next_run is not None
            and next_run[0] == next_day
            and (next_run[1] - next_run[0]).days + 1 >= LONG_HOLIDAY_MIN_DAYS
        )

        is_songkran = d.month == 4 and d.day in range(11, 18)
        is_newyear = (d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 4)
        is_bank_holiday = d in bank_holidays if d.year in bank_holiday_known_years else None

        results.append(
            DailyFlags(
                date=d,
                is_public_holiday=d in public_holidays,
                is_bank_holiday=is_bank_holiday,
                is_weekend=d.weekday() >= 5,
                is_long_weekend=is_long_weekend,
                is_songkran_7days=is_songkran,
                is_newyear_7days=is_newyear,
                is_holiday_eve=is_holiday_eve,
                days_to_next_holiday=_days_to_next_holiday(d, public_holidays, sorted_public),
            )
        )
    return results
