"""Independently validate config/mappings/tat_province_id_crosswalk.csv.

Run: ``uv run python scripts/validate_tat_crosswalk.py``

Deliberately does NOT use postcode logic again -- the crosswalk was built
from postcodes, so re-checking it with postcodes would only catch bugs in
the postcode reference itself, not bugs in how the crosswalk was built or
applied. Three independent checks instead:

1. Attraction counts per province must be plausible (tourism-heavy
   provinces high, small inland provinces low) -- prints top 10 / bottom 10
   for a human to eyeball, since "plausible" isn't a hard threshold.
2. Counts across all 77 provinces plus the excluded junk bucket ("1") must
   sum to exactly 1,731 (the file's total record count) -- if they don't,
   some records were silently dropped or double-counted.
3. Five well-known attractions are spot-checked by name against the
   province their record resolves to.

Exits non-zero and prints exactly what failed if any check fails, per the
project owner's explicit instruction: report a failing check, don't adjust
the crosswalk to make it pass.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from rri.utils.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)

EXPECTED_TOTAL_RECORDS = 1731
JUNK_PROVINCE_ID = "1"

# (attraction name substring, expected province_th) -- chosen as
# unambiguous, well-known landmarks rather than generic names.
#
# Note: วัดพระแก้ว / the Grand Palace (arguably Thailand's single most
# famous attraction) is NOT in this dataset under any name variant tried
# (Wat Phra Kaew, พระบรมมหาราชวัง, แก้วมรกต) -- nor is Wat Arun or Chatuchak
# Market. This file reads as a tourism-*service* directory (spas, farms,
# boathouses) more than a comprehensive monuments list; logged as a
# separate coverage observation, not a crosswalk failure, since there's no
# record to mis-assign.
SPOT_CHECKS: list[tuple[str, str]] = [
    ("เยาวราช", "กรุงเทพมหานคร"),  # Yaowarat (Bangkok Chinatown)
    ("ดอยสุเทพ", "เชียงใหม่"),  # Doi Suthep
    ("เกาะพีพี", "กระบี่"),  # Phi Phi Islands (administratively Krabi)
    ("อยุธยา", "พระนครศรีอยุธยา"),  # anything named after Ayutthaya itself
    ("หาดป่าตอง", "ภูเก็ต"),  # Patong Beach
]


def load_tat_records() -> list[dict[str, Any]]:
    path = settings.raw_dir / "tat_tourist_attractions.json"
    with path.open(encoding="utf-8-sig") as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def load_crosswalk() -> dict[str, str]:
    path = settings.mappings_dir / "tat_province_id_crosswalk.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["att_province_id"]: row["province_id"] for row in csv.DictReader(f)}


def load_master_names() -> dict[str, str]:
    path = settings.config_dir / "province_master.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["province_id"]: row["province_th"] for row in csv.DictReader(f)}


def check_total_count(records: list[dict[str, Any]], crosswalk: dict[str, str]) -> list[str]:
    failures = []
    actual_total = len(records)
    if actual_total != EXPECTED_TOTAL_RECORDS:
        failures.append(
            f"file has {actual_total} records, expected {EXPECTED_TOTAL_RECORDS} "
            f"(EXPECTED_TOTAL_RECORDS constant may be stale)"
        )

    matched = sum(1 for r in records if r.get("ATT_PROVINCE_ID", "").strip() in crosswalk)
    junk = sum(1 for r in records if r.get("ATT_PROVINCE_ID", "").strip() == JUNK_PROVINCE_ID)
    accounted_for = matched + junk
    if accounted_for != actual_total:
        failures.append(
            f"matched ({matched}) + junk ({junk}) = {accounted_for}, "
            f"but file has {actual_total} records -- some records are unaccounted for"
        )
    log.info(
        "tat_count_check",
        total=actual_total,
        matched=matched,
        junk=junk,
        accounted_for=accounted_for,
    )
    return failures


def print_attraction_counts_by_province(
    records: list[dict[str, Any]], crosswalk: dict[str, str], master_names: dict[str, str]
) -> None:
    province_counts: Counter[str] = Counter()
    for record in records:
        pid = record.get("ATT_PROVINCE_ID", "").strip()
        province_id = crosswalk.get(pid)
        if province_id:
            province_counts[master_names[province_id]] += 1

    ranked = province_counts.most_common()
    log.info("attraction_counts_top_10", top_10=ranked[:10])
    log.info("attraction_counts_bottom_10", bottom_10=ranked[-10:])


def run_spot_checks(
    records: list[dict[str, Any]], crosswalk: dict[str, str], master_names: dict[str, str]
) -> list[str]:
    failures = []
    for name_substring, expected_province in SPOT_CHECKS:
        matches = [r for r in records if name_substring in r.get("ATT_NAME_TH", "")]
        if not matches:
            failures.append(f"spot check: no attraction found matching {name_substring!r}")
            continue
        record = matches[0]
        pid = record.get("ATT_PROVINCE_ID", "").strip()
        province_id = crosswalk.get(pid)
        actual_province = master_names.get(province_id) if province_id else None
        status = "ok" if actual_province == expected_province else "MISMATCH"
        log.info(
            "spot_check",
            status=status,
            attraction=record.get("ATT_NAME_TH", ""),
            expected=expected_province,
            actual=actual_province,
        )
        if actual_province != expected_province:
            failures.append(
                f"spot check failed: {record.get('ATT_NAME_TH', '')!r} expected "
                f"{expected_province!r}, crosswalk resolved to {actual_province!r}"
            )
    return failures


def main() -> int:
    configure_logging()
    records = load_tat_records()
    crosswalk = load_crosswalk()
    master_names = load_master_names()

    failures: list[str] = []
    failures.extend(check_total_count(records, crosswalk))
    print_attraction_counts_by_province(records, crosswalk, master_names)
    failures.extend(run_spot_checks(records, crosswalk, master_names))

    if failures:
        for message in failures:
            log.error("tat_crosswalk_validation_failed", message=message)
        return 1

    log.info("tat_crosswalk_validation_passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
