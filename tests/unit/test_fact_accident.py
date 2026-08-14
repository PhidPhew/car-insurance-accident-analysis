"""Unit tests for the `fact_accident` builder (CLAUDE.md §8, Phase 3 Step 2)."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import polars as pl
import pytest

from rri.warehouse.dim_accident_context import build_dim_accident_context
from rri.warehouse.dim_date import build_dim_date
from rri.warehouse.dim_geography import build_dim_geography
from rri.warehouse.dim_time_of_day import build_dim_time_of_day
from rri.warehouse.dim_vehicle import build_dim_vehicle
from rri.warehouse.dim_weather_band import build_dim_weather_band
from rri.warehouse.fact_accident import build_fact_accident
from rri.warehouse.keys import UNKNOWN_SK


def _holiday_flags_csv(tmp_path: Path) -> Path:
    path = tmp_path / "holiday_flags.csv"
    rows = [
        "2020-12-15,False,False,False,False,False,False,False,False,10,none",
        "2023-04-12,False,False,False,False,False,False,False,False,10,none",
        "2023-04-13,False,False,False,False,False,False,False,False,9,none",
    ]
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
        {"ประเภทรถ": [], "จังหวัด": [], "ประเภทเชื้อเพลิง": [], "จำนวนรถ": []},
        schema={
            "ประเภทรถ": pl.Utf8,
            "จังหวัด": pl.Utf8,
            "ประเภทเชื้อเพลิง": pl.Utf8,
            "จำนวนรถ": pl.Utf8,
        },
    )
    return flood, r35mm, stt


def _accidents_fixture() -> pl.DataFrame:
    """4 rows: 1 clean, 1 null-province, 1 null-vehicle/weather, 1 null accident_date."""
    return pl.DataFrame(
        {
            "ACC_CODE": ["A1", "A2", None, "A4"],
            "รหัสสายทาง": ["31", "212", "41", None],
            "KM": ["10.6", "268.45", None, "5.0"],
            "หน่วยงาน": ["กรมทางหลวง", "กรมทางหลวง", "กรมทางหลวงชนบท", "กรมทางหลวง"],
            "จังหวัด": ["กรุงเทพมหานคร", None, "เชียงใหม่", "เชียงใหม่"],
            "รถคันที่1": ["รถจักรยานยนต์", "รถปิคอัพบรรทุก 4 ล้อ", None, "รถจักรยานยนต์"],
            "สภาพอากาศ": ["แจ่มใส", "ฝนตก", None, "แจ่มใส"],
            "มูลเหตุสันนิษฐาน_group": ["speeding", "speeding", None, "speeding"],
            "ลักษณะการเกิดเหตุ_group": ["rear_end_collision"] * 4,
            "บริเวณที่เกิดเหตุ_group": ["straight_road"] * 4,
            "บริเวณที่เกิดเหตุอื่น_group": [None, None, None, None],
            "accident_date": [date(2023, 4, 12), date(2023, 4, 12), date(2023, 4, 13), None],
            "report_date": [
                date(2023, 4, 13),
                date(2023, 4, 14),
                date(2023, 4, 15),
                date(2020, 12, 15),
            ],
            "accident_time": [time(8, 0), time(14, 30), time(23, 0), time(9, 0)],
            "ผู้เสียชีวิต": ["0", "1", "0", "0"],
            "ผู้บาดเจ็บสาหัส": ["1", "0", None, "0"],
            "ผู้บาดเจ็บเล็กน้อย": ["0", "0", None, "0"],
            "รวมจำนวนผู้บาดเจ็บ": ["1", "0", "2", "0"],
            "รถที่เกิดเหตุ": ["1", "2", "2", "1"],
            "คนเดินเท้า": ["0", "0", None, "0"],
            "has_injury_breakdown": [True, True, False, True],
            "date_certainty": ["certain", "certain", "certain", "certain"],
            "is_geo_valid": [True, True, True, True],
            "LATITUDE": [13.75, 17.6, None, 18.7],
            "LONGITUDE": [100.5, 104.4, None, 98.9],
        }
    )


@pytest.fixture
def built_fact(tmp_path: Path) -> pl.DataFrame:
    accidents = _accidents_fixture()
    dim_date = build_dim_date(_holiday_flags_csv(tmp_path))
    flood, r35mm, stt = _empty_geography_inputs()
    tat_path = tmp_path / "tat.json"
    tat_path.write_text("[]")
    dim_geography = build_dim_geography(flood, r35mm, stt, tat_path)
    dim_vehicle = build_dim_vehicle()
    dim_accident_context = build_dim_accident_context(accidents)
    dim_time_of_day = build_dim_time_of_day()
    dim_weather_band = build_dim_weather_band()

    vehicle_mapping_path = Path(__file__).parents[2] / "config" / "mappings" / "vehicle_groups.csv"

    return build_fact_accident(
        accidents,
        dim_date,
        dim_geography,
        dim_vehicle,
        dim_accident_context,
        dim_time_of_day,
        dim_weather_band,
        vehicle_mapping_path,
    )


class TestBuildFactAccident:
    def test_row_count_matches_input_exactly_no_fanout(self, built_fact: pl.DataFrame) -> None:
        # Regression test: dim_accident_context's Unknown Member (sk=-1) once
        # shared a join key with a real combo row, fanning every matching
        # fact row out into 2 output rows.
        assert built_fact.height == 4
        assert built_fact["accident_sk"].n_unique() == 4

    def test_fatality_sum_preserved(self, built_fact: pl.DataFrame) -> None:
        assert built_fact["fatality_count"].sum() == 1

    def test_no_orphan_fks(self, built_fact: pl.DataFrame) -> None:
        # Every FK column must be non-null (Unknown Member fallback, never a bare null).
        fk_columns = [
            "date_sk",
            "geography_sk",
            "vehicle_sk",
            "accident_context_sk",
            "time_of_day_sk",
            "weather_band_sk",
        ]
        for col in fk_columns:
            assert built_fact[col].null_count() == 0

    def test_null_province_falls_back_to_unknown_member(self, built_fact: pl.DataFrame) -> None:
        row = built_fact.filter(pl.col("acc_code") == "A2")
        assert row["geography_sk"][0] == UNKNOWN_SK

    def test_null_vehicle_and_weather_fall_back_to_unknown_member(
        self, built_fact: pl.DataFrame
    ) -> None:
        row = built_fact.filter(pl.col("road_code") == "41")
        assert row["vehicle_sk"][0] == UNKNOWN_SK
        assert row["weather_band_sk"][0] == UNKNOWN_SK

    def test_null_accident_date_falls_back_to_unknown_member(
        self, built_fact: pl.DataFrame
    ) -> None:
        row = built_fact.filter(pl.col("road_code").is_null())
        assert row["date_sk"][0] == UNKNOWN_SK
        assert row["reporting_lag_days"][0] is None

    def test_reporting_lag_days_computed(self, built_fact: pl.DataFrame) -> None:
        row = built_fact.filter(pl.col("acc_code") == "A1")
        assert row["reporting_lag_days"][0] == 1

    def test_degenerate_dims_carried_through(self, built_fact: pl.DataFrame) -> None:
        row = built_fact.filter(pl.col("acc_code") == "A1")
        assert row["road_code"][0] == "31"
        assert row["km_marker"][0] == pytest.approx(10.6)
        assert row["agency"][0] == "กรมทางหลวง"

    def test_accident_count_is_always_one(self, built_fact: pl.DataFrame) -> None:
        assert built_fact["accident_count"].to_list() == [1, 1, 1, 1]
