"""Unit tests for Phase 3 dimension builders (CLAUDE.md §8)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from rri.warehouse.dim_accident_context import build_dim_accident_context
from rri.warehouse.dim_customer_segment import build_dim_customer_segment
from rri.warehouse.dim_date import build_dim_date, date_expr_to_sk
from rri.warehouse.dim_geography import build_dim_geography
from rri.warehouse.dim_policy_type import build_dim_policy_type
from rri.warehouse.dim_time_of_day import build_dim_time_of_day
from rri.warehouse.dim_vehicle import build_dim_vehicle
from rri.warehouse.dim_weather_band import WEATHER_BAND_BY_RAW_VALUE, build_dim_weather_band
from rri.warehouse.keys import UNKNOWN_SK


def _has_unique_unknown_member(df: pl.DataFrame, sk_column: str) -> bool:
    unknown_rows = df.filter(pl.col(sk_column) == UNKNOWN_SK)
    return unknown_rows.height == 1


class TestDimDate:
    def test_date_sk_does_not_overflow_i8(self) -> None:
        # Regression test: dt.month()/dt.day() are i8 in polars, so
        # month * 100 silently overflowed before an explicit cast --
        # 2019-09-01 computed as 20189877 instead of 20190901.
        expr_df = pl.DataFrame({"date": [date(2019, 9, 1)]}).select(
            date_expr_to_sk("date").alias("sk")
        )
        assert expr_df["sk"][0] == 20190901

    def test_build_dim_date_has_one_row_per_day_plus_unknown(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "holiday_flags.csv"
        csv_path.write_text(
            "date,is_public_holiday,is_bank_holiday,is_private_sector_holiday,is_weekend,"
            "is_long_weekend,is_songkran_7days,is_newyear_7days,is_holiday_eve,"
            "days_to_next_holiday,covid_disruption_level\n"
            "2019-01-01,True,False,False,False,False,False,True,False,0,none\n"
            "2019-01-02,False,False,False,False,False,False,True,False,48,none\n"
            "2019-01-03,False,False,False,False,False,False,False,False,47,none\n"
        )
        df = build_dim_date(csv_path)
        assert df.height == 4  # 3 real days + Unknown Member
        assert _has_unique_unknown_member(df, "date_sk")
        assert df["date_sk"].n_unique() == df.height

    def test_unknown_member_has_null_date(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "holiday_flags.csv"
        csv_path.write_text(
            "date,is_public_holiday,is_bank_holiday,is_private_sector_holiday,is_weekend,"
            "is_long_weekend,is_songkran_7days,is_newyear_7days,is_holiday_eve,"
            "days_to_next_holiday,covid_disruption_level\n"
            "2019-01-01,True,False,False,False,False,False,True,False,0,none\n"
        )
        df = build_dim_date(csv_path)
        unknown = df.filter(pl.col("date_sk") == UNKNOWN_SK)
        assert unknown["date"][0] is None


class TestDimTimeOfDay:
    def test_24_hours_plus_unknown(self) -> None:
        df = build_dim_time_of_day()
        assert df.height == 25
        assert _has_unique_unknown_member(df, "time_of_day_sk")
        assert sorted(df.filter(pl.col("time_of_day_sk") != UNKNOWN_SK)["hour"].to_list()) == list(
            range(24)
        )

    def test_late_night_and_night_are_flagged(self) -> None:
        df = build_dim_time_of_day()
        night_hours = df.filter(pl.col("is_night"))["hour"].to_list()
        assert 2 in night_hours  # late_night
        assert 22 in night_hours  # night
        assert 12 not in night_hours  # midday


class TestDimWeatherBand:
    def test_covers_every_raw_value_seen_in_staging(self) -> None:
        # The 7 non-null สภาพอากาศ values observed in data/10_staging/accidents.parquet.
        observed_raw_values = {
            "แจ่มใส",
            "ฝนตก",
            "มีหมอก/ควัน/ฝุ่น",
            "มืดครึ้ม",
            "อื่นๆ",
            "ภัยธรรมชาติ เช่น พายุ น้ำท่วม",
            "ดินถล่ม",
        }
        assert observed_raw_values <= WEATHER_BAND_BY_RAW_VALUE.keys()

    def test_bands_plus_unknown(self) -> None:
        df = build_dim_weather_band()
        assert df.height == 6  # 5 bands + Unknown
        assert _has_unique_unknown_member(df, "weather_band_sk")
        assert set(WEATHER_BAND_BY_RAW_VALUE.values()) == set(
            df.filter(pl.col("weather_band_sk") != UNKNOWN_SK)["weather_band"].to_list()
        )


class TestDimVehicle:
    def test_five_groups_plus_unknown(self) -> None:
        df = build_dim_vehicle()
        assert df.height == 6
        assert _has_unique_unknown_member(df, "vehicle_sk")

    def test_exposure_normalization_flag_matches_claude_md_s7_2(self) -> None:
        df = build_dim_vehicle()
        normalized = set(df.filter(pl.col("has_normalized_exposure"))["vehicle_group"].to_list())
        assert normalized == {"motorcycle", "passenger_car", "pickup"}


class TestDimAccidentContext:
    def _accidents_fixture(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "มูลเหตุสันนิษฐาน_group": ["speeding", "speeding", None],
                "ลักษณะการเกิดเหตุ_group": ["rear_end_collision", "rear_end_collision", None],
                "บริเวณที่เกิดเหตุ_group": ["straight_road", "straight_road", None],
                "บริเวณที่เกิดเหตุอื่น_group": [None, None, None],
            }
        )

    def test_grain_is_distinct_combos_plus_unknown(self) -> None:
        df = build_dim_accident_context(self._accidents_fixture())
        # 2 real combos (rows 1&2 collapse, row 3's all-null is its own combo) + Unknown Member
        assert df.height == 3
        assert _has_unique_unknown_member(df, "accident_context_sk")

    def test_null_group_becomes_unknown_text_not_dropped(self) -> None:
        df = build_dim_accident_context(self._accidents_fixture())
        all_null_combo = df.filter(
            (pl.col("accident_context_sk") != UNKNOWN_SK) & (pl.col("cause_group") == "Unknown")
        )
        assert all_null_combo.height == 1


class TestDimGeography:
    def _flood_fixture(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Month": ["6", "6", "9"],
                "TAMBON_T": ["ต.เอ", "ต.บี", "ต.เอ"],
                "PROV_T": ["จ.นนทบุรี", "จ.นนทบุรี", "จ.นนทบุรี"],
                "RISK": ["เสี่ยงสูง", "เสี่ยงต่ำ", "เสี่ยงสูง"],
            }
        )

    def _r35mm_fixture(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Province_Name": ["Nonthaburi"] * 3,
                "Year": ["2014", "2020", "2010"],  # 2010 must be excluded (outside 2014-2023)
                "R35mm (Day)": ["10", "20", "999"],
            }
        )

    def _stt_car_fuel_fixture(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ประเภทรถ": [
                    "รย. 12 รถจักรยานยนต์",
                    "รย. 1 รถยนต์นั่งส่วนบุคคลไม่เกิน 7 คน",
                    "รย. 1 รถยนต์นั่งส่วนบุคคลไม่เกิน 7 คน",
                ],
                "จังหวัด": ["นนทบุรี", "นนทบุรี", "นนทบุรี"],
                "ประเภทเชื้อเพลิง": [None, "ไฟฟ้า", "เบนซิน"],
                "จำนวนรถ": ["60", "10", "30"],
            }
        )

    def test_77_provinces_plus_unknown(self, tmp_path: Path) -> None:
        tat_path = tmp_path / "tat.json"
        tat_path.write_text("[]")
        df = build_dim_geography(
            self._flood_fixture(), self._r35mm_fixture(), self._stt_car_fuel_fixture(), tat_path
        )
        assert df.height == 78
        assert _has_unique_unknown_member(df, "geography_sk")

    def test_phuket_ranong_flood_unavailable_not_dropped(self, tmp_path: Path) -> None:
        tat_path = tmp_path / "tat.json"
        tat_path.write_text("[]")
        df = build_dim_geography(
            self._flood_fixture(), self._r35mm_fixture(), self._stt_car_fuel_fixture(), tat_path
        )
        for province in ("ภูเก็ต", "ระนอง"):
            row = df.filter(pl.col("province_th") == province)
            assert row.height == 1
            assert row["flood_data_available"][0] is False
            assert row["flood_pct_high_risk_tambons"][0] is None

    def test_r35mm_excludes_years_outside_2014_2023(self, tmp_path: Path) -> None:
        tat_path = tmp_path / "tat.json"
        tat_path.write_text("[]")
        df = build_dim_geography(
            self._flood_fixture(), self._r35mm_fixture(), self._stt_car_fuel_fixture(), tat_path
        )
        row = df.filter(pl.col("province_th") == "นนทบุรี")
        assert row["r35mm_avg_days_2014_2023"][0] == pytest.approx(15.0)  # mean(10, 20), not 999

    def test_ev_and_motorcycle_share(self, tmp_path: Path) -> None:
        tat_path = tmp_path / "tat.json"
        tat_path.write_text("[]")
        df = build_dim_geography(
            self._flood_fixture(), self._r35mm_fixture(), self._stt_car_fuel_fixture(), tat_path
        )
        row = df.filter(pl.col("province_th") == "นนทบุรี")
        # total=100 (60 motorcycle + 10 ev-car + 30 petrol-car); ev=10/100; motorcycle=60/100
        assert row["ev_share"][0] == pytest.approx(0.10)
        assert row["motorcycle_share"][0] == pytest.approx(0.60)


class TestDimPolicyTypeAndCustomerSegmentScaffolds:
    def test_policy_type_scaffold_is_unknown_member_only(self) -> None:
        df = build_dim_policy_type()
        assert df.height == 1
        assert df["policy_type_sk"][0] == UNKNOWN_SK

    def test_customer_segment_scaffold_is_unknown_member_only(self) -> None:
        df = build_dim_customer_segment()
        assert df.height == 1
        assert df["customer_segment_sk"][0] == UNKNOWN_SK
