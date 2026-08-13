"""Build config/mappings/{cause,crash_type,road_geometry}_groups.csv (Phase 2 Step 2).

Run: ``uv run python scripts/build_taxonomy_mappings.py``

Reads every distinct raw value (with its row count) for the 3 taxonomy columns
directly from the Step 1 staging output, classifies each with hand-written
keyword rules below, and writes one committed mapping CSV per taxonomy:
``raw_value,canonical_group,row_count``.

The rules here are the *drafting* step -- CLAUDE.md is explicit that
normalization proposes candidates and a committed table is the source of
truth, never automatic fuzzy matching as the final authority. Every value's
assigned group in the output was read and sanity-checked by hand before this
script's rules were treated as final (see docs/data_quality_findings.md P4
for the review notes); a maintainer changing raw category text upstream
should re-run this script and re-review the diff, not trust it blindly.

Road geometry note (P3/P5): the raw values are compound
``{road_type}+{slope_descriptor}`` (e.g. "ทางตรง+ไม่มีความลาดชัน"). The
canonical group keeps only the road-type half -- the slope descriptor is
exactly what the 2026-only secondary column (`บริเวณที่เกิดเหตุอื่น`)
disagrees with the primary column on for the same event (P3), and what
caused 74% of P5's duplicate-pair "conflicts" that turned out to be the same
concept. Grouping by road type alone is what makes both taxonomies resolve
to the same canonical value.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl  # noqa: E402

from config.settings import settings  # noqa: E402
from rri.utils.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def classify_cause(value: str) -> str:
    """Classify one raw `มูลเหตุสันนิษฐาน` (cause) value into a canonical group."""
    v = value

    def has(*needles: str) -> bool:
        return any(n in v for n in needles)

    if has("เร็วเกิน"):
        return "speeding"
    if has(
        "เปลี่ยนช่องทาง",
        "ผิดช่องทาง",
        "เข้าผิดช่อง",
        "คร่อมเส้นแบ่ง",
        "ชิดซ้ายมากเกินไป",
        "ชิดขวามาก",
        "ช่องทางเดินรถซ้ายสุด",
    ):
        return "improper_lane_position"
    if has("ตัดหน้า", "กระชั้นชิด") and not has("เบรค", "เบรก"):
        return "following_too_close_or_cutting"
    if has("ไม่ให้สิทธิ", "ไม่ยอมให้รถที่มีสิทธิ์"):
        return "right_of_way_violation"
    if has("หลับใน", "เมื่อยล้า"):
        return "drowsy_or_fatigue"
    if has("ยางเสื่อม", "ยางแตก", "ยางรถยนต์ชำรุด"):
        return "tire_defect"
    if has("เมาสุรา", "แอลกอฮอล์", "สารออกฤทธิ์ต่อจิต"):
        return "alcohol_or_substance"
    if has("ฝ่าฝืนสัญญาณไฟ", "ฝ่าฝืนป้ายหยุด"):
        return "signal_or_sign_violation"
    if has("ถนนลื่น", "คราบสะสมบนผิวถนน"):
        return "slippery_road_surface"
    if has("ถนนชำรุด", "ผิวถนนสภาพแย่", "ถนนแคบ", "ความต่างระดับของไหล่ทาง"):
        return "damaged_road_surface"
    if has("ไม่คุ้นเคย", "ไม่ชำนาญ", "ไม่คุ้นชิน"):
        return "driver_inexperience"
    if has("แซง"):
        return "illegal_overtaking"
    if has("กีดขวาง", "บดบัง"):
        return "obstacle_or_blocked_view"
    if has("ย้อนศร", "ย้อนทิศทาง"):
        return "wrong_way_driving"
    if has("สูญเสียการควบคุม", "เสียการควบคุม"):
        return "loss_of_control"
    if has("ไม่ให้สัญญาณ"):
        return "no_turn_signal_given"
    if has("บรรทุกเกิน", "บรรทุกสูงเกิน"):
        return "overloading"
    if has("ทางโค้งอันตราย", "ลักษณะของถนน"):
        return "dangerous_road_geometry"
    if has(
        "แสงสว่างไม่เพียงพอ",
        "ระยะการมองเห็นไม่เพียงพอ",
        "ระยะมองเห็นจำกัด",
        "ไฟฟ้าแสงสว่างชำรุด",
        "ไฟส่องสว่างชำรุด",
        "ไม่เห็นเสาไฟ",
        "ไม่เปิดไฟ",
        "ไม่ใช้แสงสว่าง",
    ):
        return "insufficient_lighting_or_visibility"
    if has("โรคประจำตัว", "ไร้ความสามารถทางประสาท", "ปัญหาทางสายตา"):
        return "driver_health_condition"
    if has("โทรศัพท์", "ละสายตา", "สิ่งรบกวน"):
        return "driver_distraction"
    if has("กะทันหัน") and not has("เปลี่ยนช่องทาง"):
        return "sudden_braking"
    if has("ห้ามล้อ", "เบรค", "เบรก", "พวงมาลัย", "บังคับเลี้ยว", "บังขับเลี้ยว"):
        return "brake_or_steering_defect"
    if has(
        "เครื่องยนต์",
        "ระบบไฟฟ้าของยานพาหนะ",
        "ระบบไฟฟ้ขัดข้อง",
        "ระบบไฟขัดข้องในห้องเครื่อง",
        "อุปกรณ์นำทางชำรุด",
        "หางพ่วงหลุด",
        "ฝาตู้เก็บของ",
        "ความสูงเกินกำหนด",
        "อุปกรณ์ยานพาหนะบกพร่อง",
    ):
        return "vehicle_mechanical_defect"
    if has("ขับขี่ช้าเกิน", "เคลื่อนที่ช้า"):
        return "slow_moving_vehicle"
    if has("ซ่อม/สร้างบนสายทาง", "ทำงานบนถนน"):
        return "road_work"
    if has("ไม่มีมูลเหตุสันนิษฐาน"):
        return "no_cause_reported"
    if v == "อื่นๆ":
        return "other"
    return "other"


def classify_crash_type(value: str) -> str:
    """Classify one raw `ลักษณะการเกิดเหตุ` (crash type) value into a canonical group.

    Almost every long-tail value here is `อื่นๆ (free-text elaboration)` -- the
    source system's own "other" bucket with a note appended -- so any value
    starting with `อื่นๆ` collapses to "other" directly, which accounts for
    most of the 152 -> ~15 reduction.
    """
    v = value

    def has(*needles: str) -> bool:
        return any(n in v for n in needles)

    if v.startswith("อื่นๆ"):
        return "other"
    if has("พลิกคว่ำ", "ตกถนน", "เสียหลัก"):
        return "rollover_or_run_off_road"
    if has("ชนท้าย", "ถูกชนท้าย"):
        return "rear_end_collision"
    if has("ชนในทิศทางตรงกันข้าม", "ประสานงา"):
        return "head_on_collision"
    if has("ชนสิ่งกีดขวาง", "ชนสิ่งของ", "ชนป้าย", "ชนเกาะกลาง"):
        return "collision_with_obstacle"
    if has("ชนคนเดินเท้า"):
        return "pedestrian_collision"
    if has("ชนสัตว์"):
        return "animal_collision"
    if has("ชนเป็นมุมบริเวณทางแยก", "รถทางตรงชน", "รถเลี้ยว"):
        return "junction_angle_collision"
    if has("ชนด้านข้าง", "เฉี่ยวชนรถข้างเคียง", "เปลี่ยนช่องจราจร"):
        return "side_swipe_collision"
    if has("ชนขณะแซง", "ชนรถฝั่งตรงข้ามขณะแซง", "แซงไม่พ้น", "แซงขวา"):
        return "overtaking_collision"
    if has("เลี้ยว/ถอยชน", "ชนท้ายรถขณะเลี้ยวกลับรถ", "ชนรถที่ออกจากที่จอด", "ถอยหลังชน"):
        return "turning_or_reversing_collision"
    if has("ไม่มีรายละเอียด"):
        return "no_detail_reported"
    if has("ตกจากรถ"):
        return "fall_from_vehicle"
    if has("ชนรถไฟ", "ทางรถไฟ"):
        return "railway_collision"
    return "other"


def classify_road_geometry(value: str) -> str:
    """Classify one raw `บริเวณที่เกิดเหตุ` / `บริเวณที่เกิดเหตุอื่น` value by road TYPE.

    Values are `{road_type}+{slope_descriptor}` -- only the road-type half
    determines the group; the slope descriptor is dropped (see module
    docstring for why: it's what the two taxonomies disagree on for the
    same event, not a genuine road-type difference).

    Deliberately matches keywords against the FULL raw value, not a
    `road_type = value.split("+")[...]` prefix: "ทางแยกรูปตัว +" (a
    "+"-shaped / 4-way junction) has its own literal "+" as PART of the
    road-type name, so a naive split on "+" corrupts it (e.g.
    "ทางแยกรูปตัว ++ที่ราบ" has TWO "+" characters, only the second is a
    real separator -- found via a 938-row group silently landing in
    "other" before this was caught and fixed).
    """

    def has(*needles: str) -> bool:
        return any(n in value for n in needles)

    if has("ทางโค้งหักศอก"):
        return "sharp_curve"
    if has("ทางโค้งกว้าง"):
        return "wide_curve"
    if has("ทางโค้งปกติ"):
        return "normal_curve"
    if has("ทางตรง"):
        return "straight_road"
    if has("ทางแยกรูปตัว T", "ทางสามแยก (T)", "ทางสามแยก(T)"):
        return "t_junction"
    if has("ทางแยกรูปตัว Y", "ทางสามแยก (Y)", "ทางสามแยก(Y)"):
        return "y_junction"
    if has("ทางแยกรูปตัว +", "ทางสี่แยก"):
        return "cross_junction"
    if has("สะพาน"):
        return "bridge"
    if has("ทางลอด", "ทางยกระดับ"):
        return "underpass_or_overpass"
    if has("วงเวียน"):
        return "roundabout"
    if has("จุดกลับรถต่างระดับ", "ทางแยกต่างระดับ"):
        return "interchange_or_grade_separated_uturn"
    if has("เชื่อมเข้า", "เข้าพื้นที่", "เข้าสถานศึกษา"):
        return "entrance_area"
    if has("ทางคนเดินเท้า", "ทางม้าลาย"):
        return "pedestrian_crossing"
    return "other"


TAXONOMIES: list[tuple[str, str, str, list[str]]] = [
    # (output filename, description, primary column, extra columns sharing the taxonomy)
    ("cause_groups.csv", "cause", "มูลเหตุสันนิษฐาน", []),
    ("crash_type_groups.csv", "crash type", "ลักษณะการเกิดเหตุ", []),
    (
        "road_geometry_groups.csv",
        "road geometry",
        "บริเวณที่เกิดเหตุ",
        ["บริเวณที่เกิดเหตุอื่น"],
    ),
]

CLASSIFIERS = {
    "มูลเหตุสันนิษฐาน": classify_cause,
    "ลักษณะการเกิดเหตุ": classify_crash_type,
    "บริเวณที่เกิดเหตุ": classify_road_geometry,
    "บริเวณที่เกิดเหตุอื่น": classify_road_geometry,
}


def main() -> None:
    configure_logging()
    df = pl.read_parquet(settings.staging_dir / "accidents.parquet")

    for out_filename, description, primary_col, extra_cols in TAXONOMIES:
        classifier = CLASSIFIERS[primary_col]
        counts: dict[str, int] = {}
        for col in [primary_col, *extra_cols]:
            vc = df[col].drop_nulls().value_counts()
            for raw_value, count in vc.iter_rows():
                counts[raw_value] = counts.get(raw_value, 0) + int(count)

        triples = sorted(
            ((raw_value, classifier(raw_value), count) for raw_value, count in counts.items()),
            key=lambda t: (-t[2], t[0]),
        )
        rows = [
            {"raw_value": raw_value, "canonical_group": group, "row_count": count}
            for raw_value, group, count in triples
        ]

        out_path = settings.mappings_dir / out_filename
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["raw_value", "canonical_group", "row_count"])
            writer.writeheader()
            writer.writerows(rows)

        group_count = len({r["canonical_group"] for r in rows})
        log.info(
            "taxonomy_mapping_written",
            taxonomy=description,
            path=str(out_path),
            raw_values=len(rows),
            canonical_groups=group_count,
        )


if __name__ == "__main__":
    main()
