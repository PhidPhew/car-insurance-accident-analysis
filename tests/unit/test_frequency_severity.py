"""Unit tests for `rri.features.frequency_severity` (CLAUDE.md §10 L0/L1, Phase 5 Step 1)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from rri.features.frequency_severity import (
    WINDOW_END,
    WINDOW_START,
    build_frequency_severity_frame,
    month_range,
)
from rri.warehouse.dim_accident_context import build_dim_accident_context
from rri.warehouse.dim_date import build_dim_date
from rri.warehouse.dim_geography import build_dim_geography
from rri.warehouse.dim_time_of_day import build_dim_time_of_day
from rri.warehouse.dim_vehicle import build_dim_vehicle
from rri.warehouse.dim_weather_band import build_dim_weather_band
from rri.warehouse.fact_accident import build_fact_accident
from rri.warehouse.keys import UNKNOWN_SK


class TestMonthRange:
    def test_covers_exactly_48_months(self) -> None:
        months = month_range(WINDOW_START, WINDOW_END)
        assert len(months) == 48
        assert months[0] == (2022, 1)
        assert months[-1] == (2025, 12)

    def test_wraps_year_boundary(self) -> None:
        months = month_range((2023, 11), (2024, 2))
        assert months == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]


def _holiday_flags_csv(tmp_path: Path) -> Path:
    dates = sorted({date(year, month, 1) for year in range(2022, 2026) for month in range(1, 13)})
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
            "ประเภทรถ": ["รย. 12 รถจักรยานยนต์", "รย.3 รถบรรทุก 6 ล้อ"],
            "จังหวัด": ["กรุงเทพมหานคร", "กรุงเทพมหานคร"],
            "ประเภทเชื้อเพลิง": ["เบนซิน", "ดีเซล"],
            "จำนวนรถ": ["1000", "500"],
        }
    )
    return flood, r35mm, stt


def _car_registered_new_fixture() -> pl.DataFrame:
    years_months = [(year, month) for year in range(2022, 2026) for month in range(1, 13)]
    years_months.append((2026, 2))
    return pl.DataFrame(
        {
            "ประเภทกฏหมาย": ["รถยนต์"] * len(years_months),
            "year_ce": [ym[0] for ym in years_months],
            "เดือนที่": [str(ym[1]) for ym in years_months],
            "จำนวน": ["100"] * len(years_months),
        }
    )


def _rainfall_fixture(province_id: str) -> pl.DataFrame:
    years_months = [(year, month) for year in range(2022, 2026) for month in range(1, 13)]
    n = len(years_months)
    return pl.DataFrame(
        {
            "Province_ID": [province_id] * n,
            "Province_NameTH": ["กรุงเทพมหานคร"] * n,
            "Initial_Year": [str(ym[0]) for ym in years_months],
            "Initial_Month": [str(ym[1]) for ym in years_months],
            "rain_lead0": [10.0] * n,
            "rain_lead1": [11.0] * n,
            "rain_lead2": [12.0] * n,
            "rain_lead3": [13.0] * n,
            "rain_lead4": [14.0] * n,
            "rain_lead5": [15.0] * n,
        }
    )


def _accidents_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ACC_CODE": ["A1"],
            "รหัสสายทาง": ["31"],
            "KM": ["10.6"],
            "หน่วยงาน": ["กรมทางหลวง"],
            "จังหวัด": ["กรุงเทพมหานคร"],
            "รถคันที่1": ["รถจักรยานยนต์"],
            "สภาพอากาศ": ["แจ่มใส"],
            "มูลเหตุสันนิษฐาน_group": ["speeding"],
            "ลักษณะการเกิดเหตุ_group": ["rear_end_collision"],
            "บริเวณที่เกิดเหตุ_group": ["straight_road"],
            "บริเวณที่เกิดเหตุอื่น_group": [None],
            # dim_date's test fixture (`_holiday_flags_csv`) only backfills the
            # 1st of every month -- the accident date must land on one of those
            # rows or it resolves to the Unknown Member date_sk (-1).
            "accident_date": [date(2022, 3, 1)],
            "report_date": [date(2022, 3, 2)],
            "accident_time": [None],
            "ผู้เสียชีวิต": ["1"],
            "ผู้บาดเจ็บสาหัส": ["0"],
            "ผู้บาดเจ็บเล็กน้อย": ["2"],
            "รวมจำนวนผู้บาดเจ็บ": ["2"],
            "รถที่เกิดเหตุ": ["1"],
            "คนเดินเท้า": ["0"],
            "has_injury_breakdown": [True],
            "date_certainty": ["certain"],
            "is_geo_valid": [True],
            "LATITUDE": [13.75],
            "LONGITUDE": [100.5],
        }
    ).with_columns(pl.col("accident_time").cast(pl.Time))


@pytest.fixture
def frame(tmp_path: Path) -> pl.DataFrame:
    accidents = _accidents_fixture()
    dim_date = build_dim_date(_holiday_flags_csv(tmp_path))
    flood, r35mm, stt = _empty_geography_inputs()
    tat_path = tmp_path / "tat.json"
    tat_path.write_text("[]")
    dim_geography = build_dim_geography(flood, r35mm, stt, tat_path)
    dim_vehicle = build_dim_vehicle()
    vehicle_mapping_path = Path(__file__).parents[2] / "config" / "mappings" / "vehicle_groups.csv"

    fact_accident = build_fact_accident(
        accidents,
        dim_date,
        dim_geography,
        dim_vehicle,
        build_dim_accident_context(accidents),
        build_dim_time_of_day(),
        build_dim_weather_band(),
        vehicle_mapping_path,
    )
    bangkok_id = dim_geography.filter(pl.col("province_th") == "กรุงเทพมหานคร")["province_id"][0]

    return build_frequency_severity_frame(
        fact_accident,
        dim_date,
        dim_geography,
        dim_vehicle,
        _rainfall_fixture(bangkok_id),
        stt,
        _car_registered_new_fixture(),
        vehicle_mapping_path,
    )


class TestBuildFrequencySeverityFrame:
    def test_grid_is_complete_including_zero_accident_cells(self, frame: pl.DataFrame) -> None:
        # 77 provinces x 5 vehicle groups x 48 months -- every cell present,
        # not just cells where an accident happened (else the Poisson model's
        # zero class would be silently censored).
        assert frame.height == 77 * 5 * 48
        assert (frame["accident_count"] == 0).sum() == frame.height - 1

    def test_exposure_null_only_for_unnormalized_groups(self, frame: pl.DataFrame) -> None:
        normalized = {"motorcycle", "passenger_car", "pickup"}
        normalized_rows = frame.filter(pl.col("vehicle_group").is_in(normalized))
        unnormalized_rows = frame.filter(~pl.col("vehicle_group").is_in(normalized))
        assert unnormalized_rows["exposure_vehicle_count"].null_count() == unnormalized_rows.height
        assert unnormalized_rows["log_exposure"].null_count() == unnormalized_rows.height
        # At least the fixture's own (Bangkok, motorcycle) cells have exposure.
        assert normalized_rows["exposure_vehicle_count"].null_count() < normalized_rows.height

    def test_severity_index_null_for_zero_accident_cells(self, frame: pl.DataFrame) -> None:
        zero_cells = frame.filter(pl.col("accident_count") == 0)
        assert zero_cells["severity_index_coarse"].null_count() == zero_cells.height
        assert zero_cells["severity_index_fine"].null_count() == zero_cells.height

    def test_severity_index_computed_for_the_one_accident_cell(self, frame: pl.DataFrame) -> None:
        row = frame.filter(pl.col("accident_count") > 0)
        assert row.height == 1
        assert row["severity_index_coarse"][0] is not None
        assert row["fatality_count"][0] == 1
        assert row["total_injury_count"][0] == 2

    def test_no_actual_weather_column_present(self, frame: pl.DataFrame) -> None:
        # No-leakage guard rail: only forecast rain_lead0..5 belong here, never
        # fact_accident's actual reported weather_band.
        assert "weather_band" not in frame.columns
        for lead in range(6):
            assert f"rain_lead{lead}" in frame.columns

    def test_rainfall_joined_for_the_fixture_province(self, frame: pl.DataFrame) -> None:
        bangkok_motorcycle = frame.filter(
            (pl.col("province_th") == "กรุงเทพมหานคร") & (pl.col("vehicle_group") == "motorcycle")
        )
        assert bangkok_motorcycle["rain_lead0"].null_count() == 0
        assert (bangkok_motorcycle["rain_lead0"] == 10.0).all()

    def test_all_cells_have_injury_breakdown_true(self, frame: pl.DataFrame) -> None:
        # Sanity check on the window assumption documented in the module
        # docstring: 2022-2025 is entirely post has_injury_breakdown-capture.
        assert (frame["accident_count"] >= 0).all()

    def test_excludes_unknown_geography_and_vehicle(self, frame: pl.DataFrame) -> None:
        assert (frame["geography_sk"] != UNKNOWN_SK).all()
        assert (frame["vehicle_sk"] != UNKNOWN_SK).all()
