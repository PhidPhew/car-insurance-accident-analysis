"""Unit tests for `fact_underwriting_monthly` (CLAUDE.md §8, Phase 4 Step 5)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from config.simulation_params import CLAIM_FREQUENCY
from rri.warehouse.dim_accident_context import build_dim_accident_context
from rri.warehouse.dim_date import build_dim_date
from rri.warehouse.dim_geography import build_dim_geography
from rri.warehouse.dim_policy_type import build_dim_policy_type
from rri.warehouse.dim_time_of_day import build_dim_time_of_day
from rri.warehouse.dim_vehicle import build_dim_vehicle
from rri.warehouse.dim_weather_band import build_dim_weather_band
from rri.warehouse.fact_accident import build_fact_accident
from rri.warehouse.fact_underwriting_monthly import (
    WINDOW_END,
    WINDOW_START,
    _window_months,
    build_fact_underwriting_monthly,
)
from rri.warehouse.keys import UNKNOWN_SK


def _holiday_flags_csv(tmp_path: Path) -> Path:
    path = tmp_path / "holiday_flags.csv"
    # The 1st of every month across the full production window (dim_date
    # needs a row for build_fact_underwriting_monthly's month-key lookup),
    # plus the specific accident dates the fixture below uses.
    dates = sorted(
        {date(year, month, 1) for year in range(2022, 2026) for month in range(1, 13)}
        | {date(2022, 1, 1), date(2022, 1, 2)}
    )
    rows = [
        f"{d.isoformat()},False,False,False,False,False,False,False,False,10,none" for d in dates
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
        {
            "ประเภทรถ": ["รย. 12 รถจักรยานยนต์"],
            "จังหวัด": ["กรุงเทพมหานคร"],
            "ประเภทเชื้อเพลิง": ["เบนซิน"],
            "จำนวนรถ": ["1000"],
        }
    )
    return flood, r35mm, stt


def _car_registered_new_fixture() -> pl.DataFrame:
    # One row per month across the full production window (2022-01..2025-12)
    # plus the 2026-02 snapshot month -- build_fact_underwriting_monthly
    # back-casts every one of those months, so a sparse fixture would raise.
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


def _accidents_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ACC_CODE": ["A1", "A2"],
            "รหัสสายทาง": ["31", "212"],
            "KM": ["10.6", "268.45"],
            "หน่วยงาน": ["กรมทางหลวง", "กรมทางหลวง"],
            "จังหวัด": ["กรุงเทพมหานคร", "กรุงเทพมหานคร"],
            "รถคันที่1": ["รถจักรยานยนต์", "รถจักรยานยนต์"],
            "สภาพอากาศ": ["แจ่มใส", "ฝนตก"],
            "มูลเหตุสันนิษฐาน_group": ["speeding", "speeding"],
            "ลักษณะการเกิดเหตุ_group": ["rear_end_collision"] * 2,
            "บริเวณที่เกิดเหตุ_group": ["straight_road"] * 2,
            "บริเวณที่เกิดเหตุอื่น_group": [None, None],
            "accident_date": [date(2022, 1, 1), date(2022, 1, 2)],
            "report_date": [date(2022, 1, 2), date(2022, 1, 3)],
            "accident_time": [None, None],
            "ผู้เสียชีวิต": ["1", "0"],
            "ผู้บาดเจ็บสาหัส": ["0", "1"],
            "ผู้บาดเจ็บเล็กน้อย": ["0", "0"],
            "รวมจำนวนผู้บาดเจ็บ": ["0", "1"],
            "รถที่เกิดเหตุ": ["1", "1"],
            "คนเดินเท้า": ["0", "0"],
            "has_injury_breakdown": [True, True],
            "date_certainty": ["certain", "certain"],
            "is_geo_valid": [True, True],
            "LATITUDE": [13.75, 13.76],
            "LONGITUDE": [100.5, 100.6],
        }
    ).with_columns(pl.col("accident_time").cast(pl.Time))


#: (fact_underwriting_monthly, dim_geography, dim_vehicle, dim_policy_type)
BuiltFixture = tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]


@pytest.fixture
def built(tmp_path: Path) -> BuiltFixture:
    accidents = _accidents_fixture()
    dim_date = build_dim_date(_holiday_flags_csv(tmp_path))
    flood, r35mm, stt = _empty_geography_inputs()
    tat_path = tmp_path / "tat.json"
    tat_path.write_text("[]")
    dim_geography = build_dim_geography(flood, r35mm, stt, tat_path)
    dim_vehicle = build_dim_vehicle()
    dim_policy_type = build_dim_policy_type(
        members=[
            {"cover_class": "class_1", "cover_class_label": "Class 1", "is_comprehensive": True},
            {"cover_class": "class_3", "cover_class_label": "Class 3", "is_comprehensive": False},
        ]
    )
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

    fact_uw = build_fact_underwriting_monthly(
        fact_accident,
        dim_date,
        dim_geography,
        dim_vehicle,
        dim_policy_type,
        stt,
        _car_registered_new_fixture(),
        vehicle_mapping_path,
    )
    return fact_uw, dim_geography, dim_vehicle, dim_policy_type


class TestWindowMonths:
    def test_covers_exactly_48_months(self) -> None:
        months = _window_months()
        assert len(months) == 48
        assert months[0] == WINDOW_START == (2022, 1)
        assert months[-1] == WINDOW_END == (2025, 12)


class TestBuildFactUnderwritingMonthly:
    def test_row_count_is_months_times_real_dim_combos(self, built: BuiltFixture) -> None:
        fact_uw, dim_geography, dim_vehicle, dim_policy_type = built
        n_geo = dim_geography.filter(pl.col("geography_sk") != UNKNOWN_SK).height
        n_veh = dim_vehicle.filter(pl.col("vehicle_sk") != UNKNOWN_SK).height
        n_pt = dim_policy_type.filter(pl.col("policy_type_sk") != UNKNOWN_SK).height
        assert fact_uw.height == 48 * n_geo * n_veh * n_pt

    def test_no_null_fks_or_measures(self, built: BuiltFixture) -> None:
        # exposure_vehicle_months is the one intentional exception -- null
        # for heavy_truck_bus/other, which have no registration source
        # (CLAUDE.md §7.2), covered explicitly by
        # test_exposure_null_only_for_unnormalized_vehicle_groups below.
        fact_uw, *_ = built
        for col in fact_uw.columns:
            if col == "exposure_vehicle_months":
                continue
            assert fact_uw[col].null_count() == 0, f"{col} has nulls"

    def test_exposure_null_only_for_unnormalized_vehicle_groups(self, built: BuiltFixture) -> None:
        # The fixture's stt_car_fuel only has snapshot data for (Bangkok,
        # motorcycle) -- compare that one cell against heavy_truck_bus
        # (structurally unnormalized, no registration source at all,
        # CLAUDE.md §7.2), which must be null everywhere.
        fact_uw, dim_geography, dim_vehicle, _dim_policy_type = built
        bangkok_sk = dim_geography.filter(pl.col("province_th") == "กรุงเทพมหานคร")["geography_sk"][
            0
        ]
        motorcycle_sk = dim_vehicle.filter(pl.col("vehicle_group") == "motorcycle")["vehicle_sk"][0]
        heavy_sk = dim_vehicle.filter(pl.col("vehicle_group") == "heavy_truck_bus")["vehicle_sk"][0]

        bangkok_motorcycle = fact_uw.filter(
            (pl.col("geography_sk") == bangkok_sk) & (pl.col("vehicle_sk") == motorcycle_sk)
        )
        assert bangkok_motorcycle.height == 48 * 2  # 48 months x 2 cover classes in the fixture
        assert bangkok_motorcycle["exposure_vehicle_months"].null_count() == 0

        heavy_rows = fact_uw.filter(pl.col("vehicle_sk") == heavy_sk)
        assert heavy_rows["exposure_vehicle_months"].null_count() == heavy_rows.height

    def test_claim_count_derives_from_policies_not_raw_accident_count(
        self, built: BuiltFixture
    ) -> None:
        # Regression test: claim_count was originally derived from a single
        # month's raw accident count in one (province, vehicle_group) cell --
        # far too sparse a signal, and decoupled in scale from the
        # exposure-derived policies_in_force, producing a loss ratio 3
        # orders of magnitude below the CLAUDE.md §7.3 55-75% sanity band.
        fact_uw, *_ = built
        expected = (fact_uw["policies_in_force"] * (CLAIM_FREQUENCY / 12.0)).round().cast(pl.Int64)
        assert (fact_uw["claim_count"] == expected).all()

    def test_policies_and_exposure_never_negative(self, built: BuiltFixture) -> None:
        fact_uw, *_ = built
        assert (fact_uw["policies_in_force"] >= 0).all()
        assert (fact_uw["exposure_vehicle_months"] >= 0).fill_null(True).all()

    def test_underwriting_sk_unique(self, built: BuiltFixture) -> None:
        fact_uw, *_ = built
        assert fact_uw["underwriting_sk"].n_unique() == fact_uw.height
