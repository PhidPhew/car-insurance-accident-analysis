"""Backfill Thai public holidays for 2019-2024 and derive dim_date-ready flags.

Run: ``uv run python scripts/backfill_holidays.py``

Writes two files:

* ``config/holidays_2019_2026.json`` -- same nested structure as the real
  ``public_holidays_all.json``, with 2019-2024 backfilled (see
  ``src/rri/io/holidays.py`` for the source strategy) and 2025-2026 copied
  verbatim from the real file (unchanged, since it's the authoritative
  DGA-sourced data).
* ``config/holiday_flags_2019_2026.csv`` -- one row per calendar day from
  2019-01-01 to 2026-12-31 with the derived flags CLAUDE.md Task 2 asks
  for (is_long_weekend, is_songkran_7days, days_to_next_holiday, ...),
  plus is_private_sector_holiday (Labour Day, tracked separately from
  is_public_holiday since it's absent from the DGA calendar but a real
  non-working day for most of the workforce) and covid_disruption_level
  (none/partial/severe, sourced date ranges in src/rri/io/holidays.py --
  April 2020 was under nationwide curfew, not just a postponed Songkran).
  Building `dim_date` itself is Phase 3's job; this is the validated
  input Phase 3 will consume.

Exits non-zero if a "must pass" validation fails (CLAUDE.md Task 2): all 8
years present, Songkran on 13-15 April, New Year's Day on 1 Jan. A
holiday count outside the expected 13-25 range, or a mismatch against the
external cross-check sources, is logged as a warning (not fatal) since a
real government year can legitimately fall outside that heuristic range
(ad-hoc cabinet "bridge day" resolutions vary year to year).
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from rri.io.holidays import (  # noqa: E402
    KNOWN_SONGKRAN_EXCEPTIONS,
    build_daily_flags,
    cross_check_existing_file,
    cross_check_logicspark_2023,
    fetch_holidays_package,
    fetch_private_sector_holiday_dates,
    validate_holiday_count,
    validate_new_year,
    validate_songkran,
)
from rri.utils.logging import configure_logging, get_logger  # noqa: E402

BACKFILL_YEARS = list(range(2019, 2025))  # 2019-2024
EXISTING_YEARS = [2025, 2026]
FULL_RANGE_START = date(2019, 1, 1)
FULL_RANGE_END = date(2026, 12, 31)

log = get_logger(__name__)


def load_existing_json(filename: str) -> dict[str, Any]:
    path = settings.raw_dir / filename
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))
    return data


def assemble_combined_holidays(
    backfilled_by_year: dict[int, list[dict[str, Any]]],
    existing_public: dict[str, Any],
) -> dict[str, Any]:
    """Build the config/holidays_2019_2026.json structure."""
    years_blocks: list[dict[str, Any]] = []
    for year in BACKFILL_YEARS:
        entries = backfilled_by_year[year]
        years_blocks.append(
            {
                "year": year,
                "year_buddhist_era": year + 543,
                "total_holidays": len(entries),
                "holidays": sorted(entries, key=lambda e: e["date"]),
            }
        )
    for year_block in existing_public["years"]:
        if year_block["year"] in EXISTING_YEARS:
            years_blocks.append(year_block)
    years_blocks.sort(key=lambda b: b["year"])

    return {
        "source": (
            "2019-2024: derived via the `holidays` Python package "
            "(holidays.Thailand), cross-checked against "
            "logicspark/thailand-holidays for 2023. "
            "2025-2026: Royal Thai Government Gazette / DGA (unchanged from "
            "public_holidays_all.json)."
        ),
        "description": "Thai public holidays 2019-2026 (2019-2024 backfilled for Phase 1)",
        "last_updated": date.today().isoformat(),
        "available_years": [year for b in years_blocks for year in [b["year"]]],
        "years": years_blocks,
    }


def write_daily_flags_csv(path: Path, flags: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "is_public_holiday",
        "is_bank_holiday",
        "is_private_sector_holiday",
        "is_weekend",
        "is_long_weekend",
        "is_songkran_7days",
        "is_newyear_7days",
        "is_holiday_eve",
        "days_to_next_holiday",
        "covid_disruption_level",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in flags:
            writer.writerow(
                [
                    row.date.isoformat(),
                    row.is_public_holiday,
                    "" if row.is_bank_holiday is None else row.is_bank_holiday,
                    row.is_private_sector_holiday,
                    row.is_weekend,
                    row.is_long_weekend,
                    row.is_songkran_7days,
                    row.is_newyear_7days,
                    row.is_holiday_eve,
                    "" if row.days_to_next_holiday is None else row.days_to_next_holiday,
                    row.covid_disruption_level,
                ]
            )


def main() -> int:
    configure_logging()
    hard_failures: list[str] = []
    warnings: list[str] = []

    all_years = BACKFILL_YEARS + EXISTING_YEARS
    package_entries = fetch_holidays_package(all_years)
    log.info("holidays_package_fetched", years=all_years, total_entries=len(package_entries))

    backfilled_by_year: dict[int, list[dict[str, Any]]] = {
        year: [e for e in package_entries if e["date"].startswith(str(year))]
        for year in BACKFILL_YEARS
    }

    for year in BACKFILL_YEARS:
        entries = backfilled_by_year[year]
        if (msg := validate_songkran(year, entries)) is not None:
            if year in KNOWN_SONGKRAN_EXCEPTIONS:
                warnings.append(f"{msg} -- documented exception: {KNOWN_SONGKRAN_EXCEPTIONS[year]}")
            else:
                hard_failures.append(msg)
        if (msg := validate_new_year(year, entries)) is not None:
            hard_failures.append(msg)
        if (msg := validate_holiday_count(year, entries)) is not None:
            warnings.append(msg)

    warnings.extend(cross_check_logicspark_2023(package_entries))

    existing_public = load_existing_json("public_holidays_all.json")
    existing_bank = load_existing_json("thai_bank_holidays_all.json")
    warnings.extend(
        cross_check_existing_file(package_entries, settings.raw_dir / "public_holidays_all.json")
    )

    present_years = set(BACKFILL_YEARS) | {b["year"] for b in existing_public["years"]}
    missing_years = sorted(set(range(2019, 2027)) - present_years)
    if missing_years:
        hard_failures.append(f"missing years in final output: {missing_years}")

    combined = assemble_combined_holidays(backfilled_by_year, existing_public)
    out_path = settings.config_dir / "holidays_2019_2026.json"
    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("holidays_json_written", path=str(out_path), years=len(combined["years"]))

    public_holiday_dates = {
        date.fromisoformat(h["date"]) for block in combined["years"] for h in block["holidays"]
    }
    bank_holiday_dates = {
        date.fromisoformat(h["date"]) for block in existing_bank["years"] for h in block["holidays"]
    }
    bank_holiday_known_years = {block["year"] for block in existing_bank["years"]}
    private_sector_holiday_dates = fetch_private_sector_holiday_dates(all_years)
    log.info(
        "private_sector_holidays_fetched",
        dates=sorted(d.isoformat() for d in private_sector_holiday_dates),
    )

    flags = build_daily_flags(
        FULL_RANGE_START,
        FULL_RANGE_END,
        public_holiday_dates,
        bank_holiday_dates,
        bank_holiday_known_years,
        private_sector_holiday_dates,
    )
    flags_path = settings.config_dir / "holiday_flags_2019_2026.csv"
    write_daily_flags_csv(flags_path, flags)
    log.info("holiday_flags_csv_written", path=str(flags_path), rows=len(flags))

    for message in warnings:
        log.warning("holiday_backfill_warning", message=message)
    if hard_failures:
        for message in hard_failures:
            log.error("holiday_backfill_hard_failure", message=message)
        return 1

    log.info("holiday_backfill_complete", years=sorted(present_years), warning_count=len(warnings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
