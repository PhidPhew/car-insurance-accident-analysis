"""Build config/mappings/tat_province_id_crosswalk.csv.

Run: ``uv run python scripts/build_tat_province_crosswalk.py``

tat_tourist_attractions.json has no province *name* field at all -- only
`ATT_PROVINCE_ID`, using TAT's own internal numbering ("1001".."1077"), which
does not correspond to the standard geocode `normalize_province` otherwise
relies on (config/province_master.csv's `province_id`, e.g. "10" for
Bangkok). Without this crosswalk, TAT cannot be joined to the province
dimension at all.

CLAUDE.md Task 3 says TAT has 78 distinct province values against Thailand's
77 provinces and asks what the 78th one actually is. Investigation (see
docs/ai_usage_log.md for the full account, including a methodology
correction after the first version of this script shipped a real bug):

- 77 of the 78 distinct ATT_PROVINCE_ID values follow the "10XX" pattern and
  are matched to a province via the *mode* of their records' ATT_POSTCODE,
  resolved through a real postcode->province lookup table (see
  `fetch_postcode_to_province`) rather than assuming a postcode's leading
  2 digits equal the province geocode. That assumption is wrong for
  Bangkok/Samut Prakan/Nonthaburi/Pathum Thani/Ayutthaya/Ang Thong/Lopburi/
  Sing Buri/Chai Nat/Saraburi (geocodes 10-19): Samut Prakan has no leading
  postal digit of its own -- its postcodes (10270, 10280, 10290, 10540,
  10560, ...) fall inside Bangkok's 10xxx block -- which shifts every
  subsequent province's postcode prefix one behind its geocode across that
  entire block. A full-postcode lookup avoids this; a 2-digit prefix does
  not.
- The 2 ids with no usable postcode anywhere in their records (1005, 1045)
  were hand-identified by web-searching their attraction names (Wat Nong
  Pling -> Kamphaeng Phet; Nam Tok Nang Non, Kudchum district -> Yasothon).
- The 78th value is the literal string "1" (13 records). Its record names
  are junk/placeholder data ("test", "test 13120844", "Hello world!" x3),
  not a real 78th province -- a data-quality defect in the source (an
  unset/default province_id sentinel), not a geography. It is excluded
  from this crosswalk and reported as unmatched.

Independent validation (not more postcode logic, per the project owner's
explicit instruction) lives in scripts/validate_tat_crosswalk.py: attraction
counts per province must look plausible, must sum to exactly 1,731 across
all provinces plus the excluded junk bucket, and 5 well-known attractions
are spot-checked by name.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from rri.utils.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)

JUNK_PROVINCE_ID = "1"

# kongvut/thai-province-data: an open, actively-maintained dataset of Thai
# provinces/districts/subdistricts with postal codes, cross-checked against
# known facts before trusting it (Ayutthaya=13000, Ang Thong=14000, Bangkok
# 10200 vs. Samut Prakan 10270/10280/10290/10540/10560 all confirmed
# correct -- see docs/ai_usage_log.md).
THAI_PROVINCE_DATA_BASE = (
    "https://raw.githubusercontent.com/kongvut/thai-province-data/master/data/raw"
)

# The 2 ids with no usable postcode data anywhere in their records -- hand
# resolved via web search on their attraction names (see module docstring).
MANUAL_OVERRIDES: dict[str, str] = {
    "1005": "62",  # Wat Nong Pling -> Kamphaeng Phet
    "1045": "35",  # Nam Tok Nang Non, Kudchum district -> Yasothon
}


def fetch_postcode_to_province(timeout: float = 20.0) -> dict[str, set[str]]:
    """Fetch a real Thai postcode -> province_th lookup (see module docstring for source)."""
    provinces_resp = requests.get(f"{THAI_PROVINCE_DATA_BASE}/provinces.json", timeout=timeout)
    provinces_resp.raise_for_status()
    districts_resp = requests.get(f"{THAI_PROVINCE_DATA_BASE}/districts.json", timeout=timeout)
    districts_resp.raise_for_status()
    subdistricts_resp = requests.get(
        f"{THAI_PROVINCE_DATA_BASE}/sub_districts.json", timeout=timeout
    )
    subdistricts_resp.raise_for_status()

    province_name_by_id = {p["id"]: p["name_th"] for p in provinces_resp.json()}
    province_id_by_district = {d["id"]: d["province_id"] for d in districts_resp.json()}

    postcode_to_province: dict[str, set[str]] = defaultdict(set)
    for sub in subdistricts_resp.json():
        province_id = province_id_by_district.get(sub["district_id"])
        if province_id is None:
            continue
        postcode_to_province[str(sub["zip_code"])].add(province_name_by_id[province_id])
    return postcode_to_province


def load_master_th_to_id() -> dict[str, str]:
    path = settings.config_dir / "province_master.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["province_th"]: row["province_id"] for row in csv.DictReader(f)}


def load_tat_records() -> list[dict[str, Any]]:
    path = settings.raw_dir / "tat_tourist_attractions.json"
    with path.open(encoding="utf-8-sig") as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def build_crosswalk(
    records: list[dict[str, Any]],
    postcode_to_province: dict[str, set[str]],
    master_th_to_id: dict[str, str],
) -> list[dict[str, str]]:
    by_pid_provinces: dict[str, list[str]] = defaultdict(list)
    all_pids: set[str] = set()
    for record in records:
        pid = record.get("ATT_PROVINCE_ID", "").strip()
        if not pid:
            continue
        all_pids.add(pid)
        postcode = record.get("ATT_POSTCODE", "").strip()
        provinces = postcode_to_province.get(postcode)
        if provinces and len(provinces) == 1:
            by_pid_provinces[pid].append(next(iter(provinces)))

    rows: list[dict[str, str]] = []
    for pid in sorted(all_pids):
        if pid == JUNK_PROVINCE_ID:
            continue
        if pid in MANUAL_OVERRIDES:
            rows.append(
                {
                    "att_province_id": pid,
                    "province_id": MANUAL_OVERRIDES[pid],
                    "matched_via": "manual_lookup",
                }
            )
            continue

        resolved_provinces = by_pid_provinces.get(pid, [])
        if not resolved_provinces:
            log.warning("tat_pid_unmatched_no_postcode", att_province_id=pid)
            continue

        mode_province_th, mode_count = Counter(resolved_provinces).most_common(1)[0]
        province_id = master_th_to_id.get(mode_province_th)
        if province_id is None:
            log.warning(
                "tat_pid_resolved_province_not_in_master",
                att_province_id=pid,
                resolved_province=mode_province_th,
            )
            continue
        confidence = mode_count / len(resolved_provinces)
        if confidence < 0.7:
            log.warning(
                "tat_pid_ambiguous_postcode_mode",
                att_province_id=pid,
                confidence=round(confidence, 2),
            )
        rows.append(
            {
                "att_province_id": pid,
                "province_id": province_id,
                "matched_via": "postcode_lookup",
            }
        )
    return rows


def main() -> int:
    configure_logging()
    master_th_to_id = load_master_th_to_id()
    records = load_tat_records()
    postcode_to_province = fetch_postcode_to_province()
    log.info("postcode_reference_fetched", distinct_postcodes=len(postcode_to_province))

    rows = build_crosswalk(records, postcode_to_province, master_th_to_id)

    province_id_counts = Counter(r["province_id"] for r in rows)
    duplicates = {pid: count for pid, count in province_id_counts.items() if count > 1}
    missing = sorted(set(master_th_to_id.values()) - set(province_id_counts))
    if duplicates:
        log.error("tat_crosswalk_has_duplicate_province_id", duplicates=duplicates)
    if missing:
        log.error("tat_crosswalk_missing_province_ids", missing=missing)

    out_path = settings.mappings_dir / "tat_province_id_crosswalk.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["att_province_id", "province_id", "matched_via"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info(
        "tat_crosswalk_written",
        path=str(out_path),
        rows=len(rows),
        expected=77,
        junk_excluded=JUNK_PROVINCE_ID,
    )
    if len(rows) != 77 or duplicates or missing:
        log.error("tat_crosswalk_incomplete_or_inconsistent", rows=len(rows), expected=77)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
