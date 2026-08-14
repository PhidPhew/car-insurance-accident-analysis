"""Exposure back-cast (CLAUDE.md §7.1) -- `stt_car_fuel`'s single 2026-02-28 snapshot,
projected across months via `car_registered_new`'s national monthly new-registration curve.

    vehicles(province, group, month) ~= vehicles(province, group, snapshot)
        * national_cum(month) / national_cum(snapshot)

`national_cum` accumulates monthly new registrations (`รถยนต์` legal category only, to
match `stt_car_fuel`'s own scope, CLAUDE.md §7.2) starting from the first month
`car_registered_new` has data for (2022-01) -- there is no earlier data to accumulate
from, so this back-cast is only valid for months on/after 2022-01, which is why
`fact_underwriting_monthly`'s window starts there rather than matching `fact_accident`'s
full 2019-2026 range (see `rri.warehouse.fact_underwriting_monthly` module docstring).

**Assumption (CLAUDE.md §7.1, stated as a limitation):** every province grows at the
national curve's rate. There is no province-level registration time series to check this
against, so it is a simplification, not a validated province-specific model.

Only the 3 vehicle groups `stt_car_fuel` actually registers (motorcycle, passenger_car,
pickup -- CLAUDE.md §7.2) get a real snapshot count here. `heavy_truck_bus`/`other` are
absent from the returned frame -- callers must not invent a number for them.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from rri.cleaning.province import normalize_province
from rri.cleaning.taxonomy import apply_taxonomy, load_taxonomy_mapping

#: `stt_car_fuel_at_25690228.csv` -- the snapshot this whole module back-casts from/to.
SNAPSHOT_YEAR = 2026
SNAPSHOT_MONTH = 2


def national_cumulative_registrations(car_registered_new_df: pl.DataFrame) -> pl.DataFrame:
    """Monthly cumulative new-registration total, `รถยนต์` legal category only."""
    return (
        car_registered_new_df.filter(pl.col("ประเภทกฏหมาย") == "รถยนต์")
        .with_columns(
            pl.col("year_ce").cast(pl.Int32).alias("year"),
            pl.col("เดือนที่").cast(pl.Int32).alias("month"),
            pl.col("จำนวน").cast(pl.Int64, strict=False).fill_null(0).alias("_count"),
        )
        .group_by(["year", "month"])
        .agg(pl.col("_count").sum().alias("_monthly_total"))
        .sort(["year", "month"])
        .with_columns(pl.col("_monthly_total").cum_sum().alias("national_cum"))
        .select("year", "month", "national_cum")
    )


def province_vehicle_group_snapshot_counts(
    stt_car_fuel_df: pl.DataFrame, vehicle_mapping_path: Path
) -> pl.DataFrame:
    """Province x vehicle_group vehicle counts at the snapshot date (3 normalized groups only)."""
    mapping = load_taxonomy_mapping(vehicle_mapping_path)
    df = stt_car_fuel_df.with_columns(
        pl.col("จำนวนรถ").cast(pl.Int64, strict=False).fill_null(0).alias("_count"),
        pl.col("จังหวัด").map_elements(normalize_province, return_dtype=pl.Utf8).alias("province_th"),
    ).filter(pl.col("province_th").is_not_null())
    df, _log = apply_taxonomy(df, "ประเภทรถ", mapping, new_column="vehicle_group")

    return (
        df.filter(pl.col("vehicle_group").is_not_null())
        .group_by(["province_th", "vehicle_group"])
        .agg(pl.col("_count").sum().alias("snapshot_vehicle_count"))
    )


def backcast_monthly_exposure(
    snapshot_counts: pl.DataFrame, growth_curve: pl.DataFrame, year: int, month: int
) -> pl.DataFrame:
    """Back-cast `snapshot_counts` to `year`/`month` via the national growth-curve ratio."""
    snapshot_cum_rows = growth_curve.filter(
        (pl.col("year") == SNAPSHOT_YEAR) & (pl.col("month") == SNAPSHOT_MONTH)
    )
    if snapshot_cum_rows.height == 0:
        raise ValueError(
            f"growth curve has no data for the snapshot month {SNAPSHOT_YEAR}-{SNAPSHOT_MONTH:02d}"
        )
    target_rows = growth_curve.filter((pl.col("year") == year) & (pl.col("month") == month))
    if target_rows.height == 0:
        raise ValueError(f"growth curve has no data for {year}-{month:02d}")

    ratio = target_rows["national_cum"][0] / snapshot_cum_rows["national_cum"][0]
    return snapshot_counts.with_columns(
        (pl.col("snapshot_vehicle_count") * ratio).alias("exposure_vehicle_count"),
        pl.lit(year).cast(pl.Int32).alias("year"),
        pl.lit(month).cast(pl.Int32).alias("month"),
    ).select("province_th", "vehicle_group", "year", "month", "exposure_vehicle_count")
