"""Unit tests for `rri.features.severity_classification` (CLAUDE.md §10 L1/L2)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from rri.features.severity_classification import (
    YEAR_END,
    YEAR_START,
    build_severity_classification_frame,
)
from rri.warehouse.dim_accident_context import build_dim_accident_context
from rri.warehouse.dim_date import build_dim_date
from rri.warehouse.dim_geography import build_dim_geography
from rri.warehouse.dim_time_of_day import build_dim_time_of_day
from rri.warehouse.dim_vehicle import build_dim_vehicle
from rri.warehouse.dim_weather_band import build_dim_weather_band
from rri.warehouse.fact_accident import build_fact_accident


def _holiday_flags_csv(tmp_path: Path) -> Path:
    dates = [date(2019, 6, 1), date(2020, 3, 15), date(2021, 8, 1), date(2025, 3, 15)]
    rows = [
        f"{d.isoformat()},False,False,False,False,False,False,False,False,10,none" for d in dates
    ]
    path = tmp_path / "holiday_flags.csv"
    path.write_text(
        "date,is_public_holiday,is_bank_holiday,is_private_sector_holiday,is_weekend,"
        "is_long_weekend,is_songkran_7days,is_newyear_7days,is_holiday_eve,"
        "days_to_next_holiday,covid_disruption_level\n" + "\n".join(rows) + "\n"
    )
    return path


def _empty_geography_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    flood = pl.DataFrame(
        {"Month": [], "TAMBON_T": [], "PROV_T": [], "RISK": []},
        schema={"Month": pl.Utf8, "TAMBON_T": pl.Utf8, "PROV_T": pl.Utf8, "RISK": pl.Utf8},
    )
    r35mm = pl.DataFrame(
        {"Province_Name": [], "Year": [], "R35mm (Day)": []},
        schema={"Province_Name": pl.Utf8, "Year": pl.Utf8, "R35mm (Day)": pl.Utf8},
    )
    stt = pl.DataFrame(
        {
            "ประเภทรถ": ["รย. 12 รถจักรยานยนต์"],
            "จังหวัด": ["กรุงเทพมหานคร"],
            "ประเภทเชื้อเพลิง": ["เบนซิน"],
            "จำนวนรถ": ["1000"],
        }
    )
    return flood, r35mm, stt


def _accidents_fixture() -> pl.DataFrame:
    # 3 rows, in this exact order (accident_sk = row index + 1):
    #   1. 2019, has_injury_breakdown=False -> excluded (no fine breakdown, P7/§6)
    #   2. 2021, has_injury_breakdown=True, fatality>0 -> "fatal", in window
    #   3. 2025, has_injury_breakdown=True, minor only -> "minor", in window
    return pl.DataFrame(
        {
            "ACC_CODE": ["A1", "A2", "A3"],
            "รหัสสายทาง": ["31", "32", "33"],
            "KM": ["1.0", "2.0", "3.0"],
            "หน่วยงาน": ["กรมทางหลวง"] * 3,
            "จังหวัด": ["กรุงเทพมหานคร"] * 3,
            "รถคันที่1": ["รถจักรยานยนต์"] * 3,
            "สภาพอากาศ": ["แจ่มใส", "ฝนตก", "แจ่มใส"],
            "มูลเหตุสันนิษฐาน": ["เมาสุรา", "ขับรถเร็วเกินอัตรา", "หลับใน"],
            "ลักษณะการเกิดเหตุ": ["ชนท้าย", "ชนประสานงา", "ตกถนน"],
            "บริเวณที่เกิดเหตุ": ["ทางตรง", "ทางโค้ง", "ทางแยก"],
            "มูลเหตุสันนิษฐาน_group": ["drunk_driving", "speeding", "drowsy_driving"],
            "ลักษณะการเกิดเหตุ_group": ["rear_end_collision", "head_on_collision", "run_off_road"],
            "บริเวณที่เกิดเหตุ_group": ["straight_road", "curve", "intersection"],
            "บริเวณที่เกิดเหตุอื่น_group": [None, None, None],
            "accident_date": [date(2019, 6, 1), date(2021, 8, 1), date(2025, 3, 15)],
            "report_date": [date(2019, 6, 2), date(2021, 8, 2), date(2025, 3, 16)],
            "accident_time": [None, None, None],
            "ผู้เสียชีวิต": ["0", "1", "0"],
            "ผู้บาดเจ็บสาหัส": ["0", "0", "0"],
            "ผู้บาดเจ็บเล็กน้อย": ["1", "0", "2"],
            "รวมจำนวนผู้บาดเจ็บ": ["1", "0", "2"],
            "รถที่เกิดเหตุ": ["1", "1", "1"],
            "คนเดินเท้า": ["0", "0", "0"],
            "has_injury_breakdown": [False, True, True],
            "date_certainty": ["certain"] * 3,
            "is_geo_valid": [True] * 3,
            "LATITUDE": [13.75] * 3,
            "LONGITUDE": [100.5] * 3,
        }
    ).with_columns(pl.col("accident_time").cast(pl.Time))


#: (severity_classification_frame, raw accidents_staging fixture)
Built = tuple[pl.DataFrame, pl.DataFrame]


@pytest.fixture
def built(tmp_path: Path) -> Built:
    accidents = _accidents_fixture()
    dim_date = build_dim_date(_holiday_flags_csv(tmp_path))
    flood, r35mm, stt = _empty_geography_inputs()
    tat_path = tmp_path / "tat.json"
    tat_path.write_text("[]")
    dim_geography = build_dim_geography(flood, r35mm, stt, tat_path)
    dim_vehicle = build_dim_vehicle()
    dim_weather_band = build_dim_weather_band()
    vehicle_mapping_path = Path(__file__).parents[2] / "config" / "mappings" / "vehicle_groups.csv"

    fact_accident = build_fact_accident(
        accidents,
        dim_date,
        dim_geography,
        dim_vehicle,
        build_dim_accident_context(accidents),
        build_dim_time_of_day(),
        dim_weather_band,
        vehicle_mapping_path,
    )

    frame = build_severity_classification_frame(
        accidents, fact_accident, dim_date, dim_geography, dim_vehicle, dim_weather_band
    )
    return frame, accidents


class TestBuildSeverityClassificationFrame:
    def test_excludes_rows_without_injury_breakdown(self, built: Built) -> None:
        frame, _ = built
        # accident_sk 1 (2019, has_injury_breakdown=False) must not appear.
        assert frame.height == 2
        assert 1 not in frame["accident_sk"].to_list()

    def test_window_bounds_match_module_constants(self) -> None:
        assert YEAR_START == 2020
        assert YEAR_END == 2025

    def test_severity_class_fatal_takes_priority(self, built: Built) -> None:
        frame, _ = built
        row = frame.filter(pl.col("accident_sk") == 2)
        assert row["severity_class"][0] == "fatal"

    def test_severity_class_minor(self, built: Built) -> None:
        frame, _ = built
        row = frame.filter(pl.col("accident_sk") == 3)
        assert row["severity_class"][0] == "minor"

    def test_raw_high_cardinality_columns_present(self, built: Built) -> None:
        frame, _ = built
        row = frame.filter(pl.col("accident_sk") == 2)
        assert row["cause_raw"][0] == "ขับรถเร็วเกินอัตรา"
        assert row["crash_type_raw"][0] == "ชนประสานงา"
        assert row["road_geometry_raw"][0] == "ทางโค้ง"
        assert row["road_code"][0] == "32"

    def test_weather_band_present_not_leakage_here(self, built: Built) -> None:
        # Unlike the frequency/severity frame, weather is a legitimate
        # attribute of the event being classified here (see module docstring).
        frame, _ = built
        assert "weather_band" in frame.columns
        assert frame["weather_band"].null_count() == 0

    def test_covid_disruption_level_carried_as_feature(self, built: Built) -> None:
        frame, _ = built
        assert "covid_disruption_level" in frame.columns
