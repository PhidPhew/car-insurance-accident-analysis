"""Unit tests for the exposure back-cast (CLAUDE.md §7.1)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rri.features.exposure import (
    SNAPSHOT_MONTH,
    SNAPSHOT_YEAR,
    backcast_monthly_exposure,
    national_cumulative_registrations,
    province_vehicle_group_snapshot_counts,
)


class TestNationalCumulativeRegistrations:
    def _fixture(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ประเภทกฏหมาย": ["รถยนต์", "รถยนต์", "รถยนต์", "รถขนส่ง"],
                "year_ce": [2022, 2022, 2022, 2022],
                "เดือนที่": ["1", "2", "3", "1"],
                "จำนวน": ["100", "50", "25", "999"],
            }
        )

    def test_cumulative_sum_and_legal_category_filter(self) -> None:
        result = national_cumulative_registrations(self._fixture()).sort("year", "month")
        assert result["national_cum"].to_list() == [100, 150, 175]  # excludes รถขนส่ง row

    def test_null_count_treated_as_zero(self) -> None:
        df = pl.DataFrame(
            {
                "ประเภทกฏหมาย": ["รถยนต์", "รถยนต์"],
                "year_ce": [2022, 2022],
                "เดือนที่": ["1", "2"],
                "จำนวน": ["100", None],
            }
        )
        result = national_cumulative_registrations(df).sort("year", "month")
        assert result["national_cum"].to_list() == [100, 100]


class TestProvinceVehicleGroupSnapshotCounts:
    def test_aggregates_by_normalized_province_and_vehicle_group(self, tmp_path: Path) -> None:
        mapping_path = tmp_path / "vehicle_groups.csv"
        mapping_path.write_text(
            "raw_value,canonical_group\nรย. 12 รถจักรยานยนต์,motorcycle\n"
            "รย. 1 รถยนต์นั่งส่วนบุคคลไม่เกิน 7 คน,passenger_car\n"
        )
        df = pl.DataFrame(
            {
                "ประเภทรถ": [
                    "รย. 12 รถจักรยานยนต์",
                    "รย. 12 รถจักรยานยนต์",
                    "รย. 1 รถยนต์นั่งส่วนบุคคลไม่เกิน 7 คน",
                ],
                "จังหวัด": ["Bangkok", "กรุงเทพมหานคร", "กรุงเทพมหานคร"],
                "จำนวนรถ": ["100", "50", "200"],
            }
        )
        result = province_vehicle_group_snapshot_counts(df, mapping_path)
        motorcycle_row = result.filter(pl.col("vehicle_group") == "motorcycle")
        # Both "Bangkok" and "กรุงเทพมหานคร" normalize to the same canonical province
        assert motorcycle_row["snapshot_vehicle_count"][0] == 150


class TestBackcastMonthlyExposure:
    def _growth_curve(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "year": [2022, 2022, SNAPSHOT_YEAR],
                "month": [1, 6, SNAPSHOT_MONTH],
                "national_cum": [100.0, 150.0, 200.0],
            }
        )

    def test_ratio_applied_correctly(self) -> None:
        snapshot = pl.DataFrame(
            {
                "province_th": ["X"],
                "vehicle_group": ["motorcycle"],
                "snapshot_vehicle_count": [1000],
            }
        )
        result = backcast_monthly_exposure(snapshot, self._growth_curve(), 2022, 6)
        # 150/200 of the snapshot count
        assert result["exposure_vehicle_count"][0] == pytest.approx(750.0)

    def test_raises_on_missing_target_month(self) -> None:
        snapshot = pl.DataFrame(
            {
                "province_th": ["X"],
                "vehicle_group": ["motorcycle"],
                "snapshot_vehicle_count": [1000],
            }
        )
        with pytest.raises(ValueError, match="no national growth-curve data|no data"):
            backcast_monthly_exposure(snapshot, self._growth_curve(), 1999, 1)
