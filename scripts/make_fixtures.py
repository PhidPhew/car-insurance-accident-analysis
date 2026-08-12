"""Generate small, defect-preserving fixture files for the test suite.

Run: ``uv run python scripts/make_fixtures.py``

CLAUDE.md N1 makes ``data/00_raw/`` read-only and off-limits to git, so the
test suite cannot depend on it directly. This script samples ~50 rows per
source into ``tests/fixtures/`` — but *stratified*, not random: it first
selects rows known to carry each documented data-quality defect (mixed date
formats, twin columns, out-of-bounds coordinates, ...), then tops up with a
seeded random sample to reach the target size. A plain random sample would
statistically miss the rare defects (e.g. the twin columns are only ~5.4%
populated) and produce fixtures that cannot catch the bugs they exist to
catch.

This script is fixture-generation *tooling*, not the Phase 2 cleaning
pipeline: its format-detection heuristics are deliberately simple — they
only need to *find* example rows, not *correct* them. Real cleaning logic
belongs in ``src/rri/cleaning/`` (Phase 2).

For the accidents file specifically, every selection condition below is
re-verified against the written fixture at the end of this script, and the
script exits non-zero (with every unmet condition listed) if any is missed
— a fixture that doesn't contain the defect can't catch the bug.
"""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

# `config` lives at the repo root, not under src/, so a direct script
# invocation (`python scripts/make_fixtures.py`) needs the repo root added
# to sys.path explicitly — it isn't part of the installed `rri` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from rri.utils.logging import configure_logging, get_logger  # noqa: E402

TARGET_ROWS_PER_SOURCE = 50

ACCIDENT_TWIN_COLUMNS = [
    "รถยนต์นั่งส่วนบุคคล/รถยนต์นั่งสาธารณะ",
    "รถบรรทุกมากกว่า6ล้อไม่เกิน10ล้อ",
    "รถบรรทุกมากกว่า10ล้อ(รถพ่วง)",
    "บริเวณที่เกิดเหตุอื่น",
    "อื่นๆ",
]
ACCIDENT_SPEEDING_VARIANTS = [
    "ขับรถเร็วเกินอัตรากำหนด",
    "ขับรถเร็วเกินอัตราที่กำหนด",
]
ACCIDENT_SOURCE_FILES = [f"accident{year}.csv" for year in range(2019, 2027)]

RowT = TypeVar("RowT")

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------


def to_float(value: str | None) -> float | None:
    """Parse a raw CSV string as a float, returning None for blank/unparseable."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a raw CSV as a list of dicts, preserving column order and raw strings."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write rows back out, re-adding the BOM (utf-8-sig) since the source has one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_up_random(
    rows: list[RowT],
    selected: set[int],
    target_total: int,
    rng: random.Random,
) -> set[int]:
    """Fill `selected` up to `target_total` indices with a seeded random sample."""
    if len(selected) >= target_total or len(selected) >= len(rows):
        return selected
    remaining = [i for i in range(len(rows)) if i not in selected]
    need = min(target_total - len(selected), len(remaining))
    return selected | set(rng.sample(remaining, need))


def first_index_where(
    rows: list[dict[str, str]], predicate: Callable[[dict[str, str]], bool]
) -> int | None:
    return next((i for i, r in enumerate(rows) if predicate(r)), None)


def column_equals(column: str, value: str) -> Callable[[dict[str, str]], bool]:
    """Build a `first_index_where` predicate for `row[column].strip() == value`.

    A plain closure factory rather than a loop-local lambda, since a lambda
    capturing a loop variable is a classic late-binding bug and mypy cannot
    infer the parameter type of a lambda with a default-arg workaround.
    """

    def predicate(row: dict[str, str]) -> bool:
        return row.get(column, "").strip() == value

    return predicate


# --------------------------------------------------------------------------
# Accidents — the one source with an explicit, self-verified checklist
# --------------------------------------------------------------------------


def is_ambiguous_dmy(date_str: str) -> bool:
    """True if a D/M/Y-ish string has both leading components <= 12 (format is undecidable)."""
    match = re.match(r"^(\d{1,2})/(\d{1,2})/\d{2,4}$", date_str.strip())
    if not match:
        return False
    a, b = int(match.group(1)), int(match.group(2))
    return a <= 12 and b <= 12


def is_excel_fraction_time(value: str) -> bool:
    """True if a time value looks like an Excel day-fraction (e.g. 0.375 = 09:00)."""
    stripped = value.strip()
    if not stripped or ":" in stripped or "/" in stripped:
        return False
    parsed = to_float(stripped)
    return parsed is not None and 0.0 <= parsed < 1.0


def find_out_of_bounds_coord_indices(rows: list[dict[str, str]]) -> set[int]:
    """Rows where LATITUDE is outside [5,21] or LONGITUDE is outside [97,106] (Thailand bounds)."""
    indices: set[int] = set()
    for i, row in enumerate(rows):
        lat = to_float(row.get("LATITUDE"))
        lon = to_float(row.get("LONGITUDE"))
        lat_bad = lat is not None and not (5 <= lat <= 21)
        lon_bad = lon is not None and not (97 <= lon <= 106)
        if lat_bad or lon_bad:
            indices.add(i)
    return indices


def select_accident_indices(rows: list[dict[str, str]], rng: random.Random) -> set[int]:
    selected: set[int] = set()

    by_source: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_source[row["source_file"]].append(i)
    for indices in by_source.values():
        selected.update(indices[:5])

    ambiguous = [i for i, r in enumerate(rows) if is_ambiguous_dmy(r.get("วันที่เกิดเหตุ", ""))]
    selected.update(ambiguous[:5])

    for col in ACCIDENT_TWIN_COLUMNS:
        non_null = [i for i, r in enumerate(rows) if r.get(col, "").strip()]
        selected.update(non_null[:5])

    # ALL out-of-bounds coordinate rows, not a fixed count: CLAUDE.md P7 quotes
    # 5 lat-invalid + 10 lon-invalid rows, but they overlap in the real file
    # (verified: union is 10, not 15) — see the discrepancy note this script
    # logs at the end. We include every row the real file actually has, so
    # the fixture is correct regardless of which number CLAUDE.md records.
    selected.update(find_out_of_bounds_coord_indices(rows))

    frac_time = [i for i, r in enumerate(rows) if is_excel_fraction_time(r.get("เวลา", ""))]
    selected.update(frac_time[:3])

    null_acc = [i for i, r in enumerate(rows) if not r.get("ACC_CODE", "").strip()]
    selected.update(null_acc[:5])

    positions: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        code = row.get("ACC_CODE", "").strip()
        if code:
            positions[code].append(i)
    dup_pair = next((idxs for idxs in positions.values() if len(idxs) >= 2), None)
    if dup_pair:
        selected.update(dup_pair[:2])

    for variant in ACCIDENT_SPEEDING_VARIANTS:
        idx = first_index_where(rows, column_equals("มูลเหตุสันนิษฐาน", variant))
        if idx is not None:
            selected.add(idx)

    bangkok_idx = first_index_where(rows, lambda r: r.get("จังหวัด", "").strip() == "กรุงเทพมหานคร")
    if bangkok_idx is not None:
        selected.add(bangkok_idx)

    target_total = TARGET_ROWS_PER_SOURCE * len(by_source)
    return top_up_random(rows, selected, target_total, rng)


def verify_accident_fixture(rows: list[dict[str, str]], real_out_of_bounds_count: int) -> list[str]:
    """Re-check every condition against the written fixture. Empty list == all satisfied."""
    failures: list[str] = []

    counts_by_source: dict[str, int] = defaultdict(int)
    for row in rows:
        counts_by_source[row["source_file"]] += 1
    for source in ACCIDENT_SOURCE_FILES:
        count = counts_by_source.get(source, 0)
        if count < 5:
            failures.append(f"accidents: source_file={source} has {count} rows, need >=5")

    ambiguous = sum(1 for r in rows if is_ambiguous_dmy(r.get("วันที่เกิดเหตุ", "")))
    if ambiguous < 5:
        failures.append(f"accidents: only {ambiguous} ambiguous-date rows, need >=5")

    for col in ACCIDENT_TWIN_COLUMNS:
        non_null = sum(1 for r in rows if r.get(col, "").strip())
        if non_null < 5:
            failures.append(
                f"accidents: twin column {col!r} has {non_null} non-null rows, need >=5"
            )

    out_of_bounds = len(find_out_of_bounds_coord_indices(rows))
    if out_of_bounds < real_out_of_bounds_count:
        failures.append(
            f"accidents: only {out_of_bounds} out-of-bounds coordinate rows in fixture, "
            f"need all {real_out_of_bounds_count} from the source"
        )

    frac_time = sum(1 for r in rows if is_excel_fraction_time(r.get("เวลา", "")))
    if frac_time < 3:
        failures.append(f"accidents: only {frac_time} Excel-fraction time rows, need >=3")

    null_acc = sum(1 for r in rows if not r.get("ACC_CODE", "").strip())
    if null_acc < 5:
        failures.append(f"accidents: only {null_acc} null ACC_CODE rows, need >=5")

    code_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        code = row.get("ACC_CODE", "").strip()
        if code:
            code_counts[code] += 1
    if not any(c >= 2 for c in code_counts.values()):
        failures.append("accidents: no duplicated ACC_CODE pair present")

    for variant in ACCIDENT_SPEEDING_VARIANTS:
        if not any(r.get("มูลเหตุสันนิษฐาน", "").strip() == variant for r in rows):
            failures.append(f"accidents: missing speeding-cause variant {variant!r}")

    if not any(r.get("จังหวัด", "").strip() == "กรุงเทพมหานคร" for r in rows):
        failures.append("accidents: missing a plain-Thai กรุงเทพมหานคร province row")

    return failures


def make_accident_fixture(rng: random.Random) -> list[str]:
    """Build and verify the accidents fixture. Returns any verification failures."""
    source_path = settings.raw_dir / "thailand_accidents_2019_2026_combined.csv"
    fieldnames, rows = read_csv_rows(source_path)
    log.info("accidents_source_loaded", rows=len(rows), source=source_path.name)

    real_out_of_bounds_count = len(find_out_of_bounds_coord_indices(rows))
    documented_count = 15  # CLAUDE.md P7: "lat outside 5-21: 5 rows; lon outside 97-106: 10 rows"
    if real_out_of_bounds_count != documented_count:
        log.warning(
            "claude_md_discrepancy",
            problem="P7 out-of-bounds coordinates",
            documented_count=documented_count,
            actual_union_count=real_out_of_bounds_count,
            note="the 5 lat-invalid and 10 lon-invalid rows overlap in the real file",
        )

    selected_indices = select_accident_indices(rows, rng)
    fixture_rows = [rows[i] for i in sorted(selected_indices)]

    out_path = settings.fixtures_dir / source_path.name
    write_csv_rows(out_path, fieldnames, fixture_rows)
    log.info("accidents_fixture_written", rows=len(fixture_rows), path=str(out_path))

    _, written_rows = read_csv_rows(out_path)
    failures = verify_accident_fixture(written_rows, real_out_of_bounds_count)
    return failures


# --------------------------------------------------------------------------
# Other CSV sources — targeted "must include" rows + random top-up
# --------------------------------------------------------------------------


def make_generic_csv_fixture(
    filename: str,
    must_include: Callable[[list[dict[str, str]]], set[int]],
    rng: random.Random,
) -> None:
    source_path = settings.raw_dir / filename
    fieldnames, rows = read_csv_rows(source_path)

    selected = must_include(rows)
    selected = top_up_random(rows, selected, TARGET_ROWS_PER_SOURCE, rng)
    fixture_rows = [rows[i] for i in sorted(selected)]

    out_path = settings.fixtures_dir / filename
    write_csv_rows(out_path, fieldnames, fixture_rows)
    log.info("csv_fixture_written", source=filename, rows=len(fixture_rows), path=str(out_path))


def select_rainfall_must_include(rows: list[dict[str, str]]) -> set[int]:
    selected: set[int] = set()
    # A complete lead-0..5 group for one (Forecast_Year, Forecast_Month,
    # Province_ID) — preserves the "6 rows per province-month" pivot quirk.
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        key = (row["Forecast_Year"], row["Foreast_Month"], row["Province_ID"])
        groups[key].append(i)
    full_group = next((idxs for idxs in groups.values() if len(idxs) == 6), None)
    if full_group:
        selected.update(full_group)
    bangkok = first_index_where(
        rows, lambda r: r.get("Province_NameTH", "").strip() == "กรุงเทพมหานคร"
    )
    if bangkok is not None:
        selected.add(bangkok)
    return selected


def select_flood_must_include(rows: list[dict[str, str]]) -> set[int]:
    selected: set[int] = set()
    # Bangkok is the one province stored WITHOUT the "จ." prefix in this file
    # — a spelling inconsistency worth preserving, not just sampling around.
    bangkok = first_index_where(rows, lambda r: r.get("PROV_T", "").strip() == "กรุงเทพมหานคร")
    if bangkok is not None:
        selected.add(bangkok)
    prefixed = first_index_where(rows, lambda r: r.get("PROV_T", "").strip().startswith("จ."))
    if prefixed is not None:
        selected.add(prefixed)
    return selected


def select_r35mm_must_include(rows: list[dict[str, str]]) -> set[int]:
    selected: set[int] = set()
    bangkok = first_index_where(rows, lambda r: r.get("Province_Name", "").strip() == "Bangkok")
    if bangkok is not None:
        selected.add(bangkok)
    pre_2014 = first_index_where(rows, lambda r: int(r["Year"]) < 2014)
    if pre_2014 is not None:
        selected.add(pre_2014)
    in_usable_range = first_index_where(rows, lambda r: 2014 <= int(r["Year"]) <= 2023)
    if in_usable_range is not None:
        selected.add(in_usable_range)
    return selected


def select_stt_car_fuel_must_include(rows: list[dict[str, str]]) -> set[int]:
    selected: set[int] = set()
    for province in ("กรุงเทพมหานคร", "กระบี่"):
        idx = first_index_where(rows, column_equals("จังหวัด", province))
        if idx is not None:
            selected.add(idx)
    both_null = [
        i
        for i, r in enumerate(rows)
        if not r.get("ประเภทเชื้อเพลิง", "").strip() and not r.get("จำนวนรถ", "").strip()
    ]
    selected.update(both_null[:5])
    return selected


def select_car_registered_must_include(rows: list[dict[str, str]]) -> set[int]:
    selected: set[int] = set()
    years = sorted({r["ปี พ.ศ."] for r in rows})
    for year in (years[0], years[-1]):
        idx = first_index_where(rows, column_equals("ปี พ.ศ.", year))
        if idx is not None:
            selected.add(idx)
    for category in ("รถยนต์", "รถขนส่ง"):
        idx = first_index_where(rows, column_equals("ประเภทกฏหมาย", category))
        if idx is not None:
            selected.add(idx)
    return selected


# --------------------------------------------------------------------------
# JSON sources
# --------------------------------------------------------------------------


def copy_whole_json(filename: str) -> None:
    """Holiday files are tiny (~15KB) — copy wholesale rather than sample."""
    source_path = settings.raw_dir / filename
    out_path = settings.fixtures_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_path, out_path)
    log.info("json_fixture_copied_whole", source=filename, path=str(out_path))


def make_tat_fixture(rng: random.Random) -> None:
    source_path = settings.raw_dir / "tat_tourist_attractions.json"
    with source_path.open(encoding="utf-8-sig") as f:
        records: list[dict[str, Any]] = json.load(f)

    selected: set[int] = set()

    # ID-overflow quirk: ATT_ID is a 17-digit string; ATT_SUBDISTRICT_ID is a
    # float-formatted string like "9637.0".
    floaty_subdistrict = [
        i for i, r in enumerate(records) if str(r.get("ATT_SUBDISTRICT_ID", "")).endswith(".0")
    ]
    selected.update(floaty_subdistrict[:5])

    # Embedded "lat, lon" string quirk in ATT_LOCATION.
    has_location = [
        i
        for i, r in enumerate(records)
        if isinstance(r.get("ATT_LOCATION"), str) and "," in r["ATT_LOCATION"]
    ]
    selected.update(has_location[:5])

    # Spread of distinct ATT_PROVINCE_ID values (the file has 78 distinct
    # values against Thailand's 77 provinces — Task 3 investigates which one
    # is spurious; this fixture just needs enough diversity to test against).
    seen_provinces: set[str] = set()
    for i, r in enumerate(records):
        province_id = str(r.get("ATT_PROVINCE_ID", ""))
        if province_id and province_id not in seen_provinces:
            seen_provinces.add(province_id)
            selected.add(i)
        if len(seen_provinces) >= 20:
            break

    selected = top_up_random(records, selected, TARGET_ROWS_PER_SOURCE, rng)
    fixture_records = [records[i] for i in sorted(selected)]

    out_path = settings.fixtures_dir / source_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fixture_records, f, ensure_ascii=False, indent=2)
        f.write("\n")  # end-of-file-fixer (pre-commit) expects a trailing newline
    log.info("tat_fixture_written", rows=len(fixture_records), path=str(out_path))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main() -> int:
    configure_logging()
    rng = random.Random(settings.random_seed)

    failures = make_accident_fixture(rng)

    make_generic_csv_fixture(
        "thailand_rainfall_2021_2026_combined.csv", select_rainfall_must_include, rng
    )
    make_generic_csv_fixture("thailand_flood_risk_areas.csv", select_flood_must_include, rng)
    make_generic_csv_fixture("thailand_heavy_rain_R35mm.csv", select_r35mm_must_include, rng)
    make_generic_csv_fixture("stt_car_fuel_at_25690228.csv", select_stt_car_fuel_must_include, rng)
    make_generic_csv_fixture(
        "car_registered_new_2565_2569_combined.csv", select_car_registered_must_include, rng
    )

    copy_whole_json("public_holidays_all.json")
    copy_whole_json("thai_bank_holidays_all.json")
    make_tat_fixture(rng)

    if failures:
        log.error("fixture_verification_failed", failures=failures)
        for message in failures:
            log.error("unmet_condition", message=message)
        return 1

    log.info("fixture_generation_complete", target_rows_per_source=TARGET_ROWS_PER_SOURCE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
