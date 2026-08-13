"""Regression guard for config/mappings/tat_province_id_crosswalk.csv.

A postcode-prefix-based build of this crosswalk once silently duplicated
Bangkok's province_id onto Samut Prakan's ATT_PROVINCE_ID (and cascaded
into Nonthaburi/Pathum Thani/... down the whole 10-19 geocode block) --
see docs/ai_usage_log.md. These tests would have caught that: a crosswalk
where two different source ids resolve to the same province, or where any
of the 77 provinces has no id resolving to it at all, is wrong regardless
of how it was built.
"""

import csv
from collections import Counter
from pathlib import Path

from config.settings import settings

EXPECTED_ROWS = 77


def _load_crosswalk() -> list[dict[str, str]]:
    path = settings.mappings_dir / "tat_province_id_crosswalk.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_master_province_ids() -> set[str]:
    path = settings.config_dir / "province_master.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["province_id"] for row in csv.DictReader(f)}


class TestTatCrosswalkIntegrity:
    def test_has_exactly_77_rows(self) -> None:
        rows = _load_crosswalk()
        assert len(rows) == EXPECTED_ROWS

    def test_no_two_att_province_ids_share_a_province(self) -> None:
        rows = _load_crosswalk()
        counts = Counter(r["province_id"] for r in rows)
        duplicates = {pid: n for pid, n in counts.items() if n > 1}
        assert duplicates == {}, (
            f"province_id assigned to more than one ATT_PROVINCE_ID: {duplicates}"
        )

    def test_every_master_province_has_exactly_one_att_province_id(self) -> None:
        rows = _load_crosswalk()
        master_ids = _load_master_province_ids()
        crosswalk_ids = {r["province_id"] for r in rows}
        assert crosswalk_ids == master_ids

    def test_known_bangkok_samut_prakan_trap_is_resolved_correctly(self) -> None:
        # The specific pair that broke the original postcode-prefix
        # heuristic: Samut Prakan's postcodes fall inside Bangkok's 10xxx
        # postal block, so a 2-digit-prefix match cannot tell them apart.
        rows = {r["att_province_id"]: r["province_id"] for r in _load_crosswalk()}
        assert rows["1001"] == "10"  # Bangkok
        assert rows["1059"] == "11"  # Samut Prakan


class TestTatCrosswalkFilePath:
    def test_crosswalk_file_exists(self) -> None:
        path = settings.mappings_dir / "tat_province_id_crosswalk.csv"
        assert isinstance(path, Path)
        assert path.exists()
