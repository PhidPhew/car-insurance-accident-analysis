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
docs/ai_usage_log.md):

- 77 of the 78 distinct ATT_PROVINCE_ID values follow the "10XX" pattern.
  Thai postal codes' first two digits closely mirror the standard province
  geocode for most provinces (verified: postcode 81xxx co-occurs with
  ATT_PROVINCE_ID 1002, and geocode 81 = Krabi; postcode 71xxx co-occurs with
  1003, and geocode 71 = Kanchanaburi; etc.) so each "10XX" id was mapped to
  a province via the *mode* of its records' ATT_POSTCODE prefix. This
  resolved 75 of the 77 "10XX" ids outright. The remaining 2 ids
  (1005, 1045) had no records with a usable postcode; their two example
  records' ATT_LOCATION coordinates and ATT_NAME_TH were identified by name
  via web search (Wat Nong Pling -> Kamphaeng Phet; Nam Tok Nang Non,
  Kudchum district -> Yasothon) and hand-confirmed.
- The 78th value is the literal string "1" (13 records). Its record names
  are junk/placeholder data ("test", "test 13120844", "Hello world!" x3),
  not a real 78th province -- this is a data-quality defect in the source
  (an unset/default province_id sentinel), not a geography. It is excluded
  from this crosswalk and reported as unmatched.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from rri.utils.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)

JUNK_PROVINCE_ID = "1"

# The 2 ids with no usable postcode data anywhere in their records -- hand
# resolved via web search on their attraction names (see module docstring).
MANUAL_OVERRIDES: dict[str, str] = {
    "1005": "62",  # Wat Nong Pling -> Kamphaeng Phet
    "1045": "35",  # Nam Tok Nang Non, Kudchum district -> Yasothon
}


def load_master_codes() -> set[str]:
    path = settings.config_dir / "province_master.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["province_id"] for row in csv.DictReader(f)}


def load_tat_records() -> list[dict[str, Any]]:
    path = settings.raw_dir / "tat_tourist_attractions.json"
    with path.open(encoding="utf-8-sig") as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def build_crosswalk(records: list[dict[str, Any]], master_codes: set[str]) -> list[dict[str, str]]:
    by_pid_postcodes: dict[str, list[str]] = defaultdict(list)
    all_pids: set[str] = set()
    for record in records:
        pid = record.get("ATT_PROVINCE_ID", "").strip()
        if not pid:
            continue
        all_pids.add(pid)
        postcode = record.get("ATT_POSTCODE", "").strip()
        if postcode.isdigit() and len(postcode) >= 2:
            by_pid_postcodes[pid].append(postcode[:2])

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

        postcodes = by_pid_postcodes.get(pid, [])
        if not postcodes:
            log.warning("tat_pid_unmatched_no_postcode", att_province_id=pid)
            continue

        mode_code, mode_count = Counter(postcodes).most_common(1)[0]
        if mode_code not in master_codes:
            log.warning(
                "tat_pid_postcode_not_a_valid_geocode",
                att_province_id=pid,
                mode_code=mode_code,
            )
            continue
        confidence = mode_count / len(postcodes)
        if confidence < 0.7:
            log.warning(
                "tat_pid_ambiguous_postcode_mode",
                att_province_id=pid,
                confidence=round(confidence, 2),
            )
        rows.append(
            {
                "att_province_id": pid,
                "province_id": mode_code,
                "matched_via": "postcode_mode",
            }
        )
    return rows


def main() -> int:
    configure_logging()
    master_codes = load_master_codes()
    records = load_tat_records()
    rows = build_crosswalk(records, master_codes)

    out_path = settings.mappings_dir / "tat_province_id_crosswalk.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["att_province_id", "province_id", "matched_via"])
        writer.writeheader()
        writer.writerows(rows)

    log.info(
        "tat_crosswalk_written",
        path=str(out_path),
        rows=len(rows),
        expected=77,
        junk_excluded=JUNK_PROVINCE_ID,
    )
    if len(rows) != 77:
        log.error("tat_crosswalk_incomplete", rows=len(rows), expected=77)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
