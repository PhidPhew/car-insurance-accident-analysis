"""Frequency/severity modeling frame (CLAUDE.md §10 L0/L1, Phase 5 Step 1).

Grain: **one province x one month x one vehicle_group**. Built entirely from
`fact_accident` (CLAUDE.md N5 -- never `fact_underwriting_monthly`, the simulated
portfolio) plus conformed dimension attributes and two feature sources joined
directly from staging (`rainfall.parquet`) and the §7.1 exposure back-cast.

**Window: 2022-01 through 2025-12 (48 months)**, not `fact_accident`'s full
2019-2026 range. Same reason as `fact_underwriting_monthly` (CLAUDE.md §12 decision
log, 2026-08-14): the exposure back-cast's growth-curve source
(`car_registered_new`) only has data from 2022-01 onward, and `log(exposure)` is
the frequency model's offset -- there is nothing to compute it from before that
month. 2026 is additionally a P7.11 incomplete-year holdout.

**Complete grid, not accidents-only rows**: every (province, month, vehicle_group)
combination in the window gets a row, including zero-accident cells, via a cross
join + left join + fill 0. Building only from cells where an accident happened
would silently censor the Poisson model's zero class -- the whole point of a
frequency model is learning the rate at which *nothing* happens too.

**Vehicle groups: all 5, but exposure is null for the 2 unnormalized ones**
(`heavy_truck_bus`/`other`) -- CLAUDE.md §7.2, `stt_car_fuel` only registers
motorcycle/passenger_car/pickup. `log_exposure` is therefore also null for those
rows, and the Poisson GLM/LightGBM frequency models (Step 2/3) must filter to
non-null exposure before fitting -- fabricating a normalized denominator for the
other two groups is exactly what §7.2 rules out. The severity target
(`severity_index_coarse`/`_fine`) doesn't need an exposure offset, so severity
rows keep all 5 groups.

**No leakage** (CLAUDE.md Step 1 guard rail): `rain_lead0..5` are joined by each
cell's own (province, year, month) -- CLAUDE.md's own determination is that these
are legitimately known-future forecast values, not realized same-month weather, so
they're safe features here. `fact_accident.weather_band` (the *actual* reported
weather at each accident) is deliberately never joined into this frame -- that
would be true same-month leakage for a model whose job is estimating this month's
risk before it's observed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from rri.features.exposure import (
    backcast_monthly_exposure,
    national_cumulative_registrations,
    province_vehicle_group_snapshot_counts,
)
from rri.features.severity_index import add_severity_index_coarse, add_severity_index_fine
from rri.warehouse.keys import UNKNOWN_SK

WINDOW_START = (2022, 1)
WINDOW_END = (2025, 12)

#: Tourism tiers computed as terciles over the 77 real provinces' own
#: `tourist_attraction_count` distribution -- CLAUDE.md §5 explicitly has no
#: land-area/population source to build a true density-based tier from, so a
#: relative (data-driven) tier is the documented stand-in, same spirit as
#: `dim_geography`'s own `tourist_attraction_count` decision.
_TOURISM_TIERS = ("low", "medium", "high")


def month_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) tuples from `start` to `end`."""
    year, month = start
    months: list[tuple[int, int]] = []
    while (year, month) <= end:
        months.append((year, month))
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months


def _real_geography_with_tourism_tier(dim_geography: pl.DataFrame) -> pl.DataFrame:
    real = dim_geography.filter(pl.col("geography_sk") != UNKNOWN_SK).select(
        "geography_sk",
        "province_id",
        "province_th",
        "region",
        "flood_pct_high_risk_tambons",
        "flood_peak_month",
        "flood_data_available",
        "r35mm_avg_days_2014_2023",
        "ev_share",
        "tourist_attraction_count",
    )
    counts = real["tourist_attraction_count"].to_numpy()
    breakpoints = np.quantile(counts, [1 / 3, 2 / 3])
    tiers = np.select(
        [counts <= breakpoints[0], counts <= breakpoints[1]],
        [_TOURISM_TIERS[0], _TOURISM_TIERS[1]],
        default=_TOURISM_TIERS[2],
    )
    return real.with_columns(pl.Series("tourism_tier", tiers, dtype=pl.Utf8))


def _accident_aggregates(
    fact_accident: pl.DataFrame, dim_date: pl.DataFrame, months: list[tuple[int, int]]
) -> pl.DataFrame:
    date_lookup = dim_date.filter(pl.col("date_sk") != UNKNOWN_SK).select(
        "date_sk", "year", "month"
    )
    # `months` always spans whole calendar years (WINDOW_START/END are Jan/Dec),
    # so filtering on year alone is equivalent to filtering on the exact
    # (year, month) pairs -- no partial-year boundary to worry about here.
    year_set = {y for y, _ in months}
    joined = (
        fact_accident.filter(
            (pl.col("date_sk") != UNKNOWN_SK)
            & (pl.col("geography_sk") != UNKNOWN_SK)
            & (pl.col("vehicle_sk") != UNKNOWN_SK)
        )
        .join(date_lookup, on="date_sk", how="inner")
        .filter(pl.col("year").is_in(year_set))
    )
    agg = joined.group_by(["year", "month", "geography_sk", "vehicle_sk"]).agg(
        pl.len().alias("accident_count"),
        pl.col("fatality_count").sum().alias("fatality_count"),
        pl.col("serious_injury_count").sum().alias("serious_injury_count"),
        pl.col("minor_injury_count").sum().alias("minor_injury_count"),
        pl.col("total_injury_count").sum().alias("total_injury_count"),
        pl.col("has_injury_breakdown").all().alias("has_injury_breakdown"),
    )
    unbroken = agg.filter(~pl.col("has_injury_breakdown")).height
    if unbroken:
        raise AssertionError(
            f"{unbroken} cells in the {WINDOW_START}-{WINDOW_END} window have "
            "has_injury_breakdown=false -- window assumption violated"
        )
    return agg


def _exposure_frame(
    stt_car_fuel_df: pl.DataFrame,
    car_registered_new_df: pl.DataFrame,
    vehicle_mapping_path: Path,
    months: list[tuple[int, int]],
) -> pl.DataFrame:
    snapshot_counts = province_vehicle_group_snapshot_counts(stt_car_fuel_df, vehicle_mapping_path)
    growth_curve = national_cumulative_registrations(car_registered_new_df)
    frames = [
        backcast_monthly_exposure(snapshot_counts, growth_curve, year, month)
        for year, month in months
    ]
    return pl.concat(frames)


def _rainfall_features(rainfall_df: pl.DataFrame) -> pl.DataFrame:
    return rainfall_df.select(
        pl.col("Province_ID").alias("province_id"),
        pl.col("Initial_Year").cast(pl.Int64).alias("year"),
        pl.col("Initial_Month").cast(pl.Int64).alias("month"),
        "rain_lead0",
        "rain_lead1",
        "rain_lead2",
        "rain_lead3",
        "rain_lead4",
        "rain_lead5",
    )


def _month_calendar_features(dim_date: pl.DataFrame, months: list[tuple[int, int]]) -> pl.DataFrame:
    year_set = {y for y, _ in months}
    real_dates = dim_date.filter(
        (pl.col("date_sk") != UNKNOWN_SK) & (pl.col("year").is_in(year_set))
    )
    return real_dates.group_by(["year", "month"]).agg(
        pl.col("season").first(),
        pl.col("covid_disruption_level").first(),
        pl.col("is_public_holiday").sum().alias("n_holiday_days"),
        pl.col("is_long_weekend").sum().alias("n_long_weekend_days"),
        pl.col("is_songkran_7days").any().alias("has_songkran"),
        pl.col("is_newyear_7days").any().alias("has_newyear"),
        pl.col("days_to_next_holiday").mean().alias("avg_days_to_next_holiday"),
    )


def build_frequency_severity_frame(
    fact_accident: pl.DataFrame,
    dim_date: pl.DataFrame,
    dim_geography: pl.DataFrame,
    dim_vehicle: pl.DataFrame,
    rainfall_df: pl.DataFrame,
    stt_car_fuel_df: pl.DataFrame,
    car_registered_new_df: pl.DataFrame,
    vehicle_mapping_path: Path,
) -> pl.DataFrame:
    """Build the finished province x month x vehicle_group frame."""
    months = month_range(WINDOW_START, WINDOW_END)
    months_df = pl.DataFrame(
        {"year": [y for y, _ in months], "month": [m for _, m in months]},
        schema={"year": pl.Int64, "month": pl.Int64},
    )

    real_geography = _real_geography_with_tourism_tier(dim_geography)
    real_vehicle = dim_vehicle.filter(pl.col("vehicle_sk") != UNKNOWN_SK).select(
        "vehicle_sk", "vehicle_group", "has_normalized_exposure"
    )

    grid = real_geography.join(real_vehicle, how="cross").join(months_df, how="cross")

    accident_agg = _accident_aggregates(fact_accident, dim_date, months)
    grid = grid.join(
        accident_agg.drop("has_injury_breakdown"),
        on=["year", "month", "geography_sk", "vehicle_sk"],
        how="left",
    ).with_columns(
        pl.col("accident_count").fill_null(0),
        pl.col("fatality_count").fill_null(0),
        pl.col("serious_injury_count").fill_null(0),
        pl.col("minor_injury_count").fill_null(0),
        pl.col("total_injury_count").fill_null(0),
        pl.lit(True).alias("has_injury_breakdown"),
    )

    exposure = _exposure_frame(stt_car_fuel_df, car_registered_new_df, vehicle_mapping_path, months)
    grid = grid.join(
        exposure.select("province_th", "vehicle_group", "year", "month", "exposure_vehicle_count"),
        on=["province_th", "vehicle_group", "year", "month"],
        how="left",
    )
    # `stt_car_fuel` has a handful of stray heavy_truck_bus/other rows (~0.06% of
    # its total count -- likely edge-case legal-category overlap, not real
    # commercial-fleet coverage, CLAUDE.md §7.2). `dim_vehicle.has_normalized_exposure`
    # is this project's already-documented Phase 3/4 decision on which groups have a
    # real registration denominator; honor it here rather than let that noise leak
    # a false normalized exposure into the 2 groups §7.2 says have none.
    grid = grid.with_columns(
        pl.when(pl.col("has_normalized_exposure"))
        .then(pl.col("exposure_vehicle_count"))
        .otherwise(None)
        .alias("exposure_vehicle_count")
    )
    grid = grid.with_columns(
        pl.when(pl.col("exposure_vehicle_count") > 0)
        .then(pl.col("exposure_vehicle_count").log())
        .otherwise(None)
        .alias("log_exposure")
    )

    rainfall = _rainfall_features(rainfall_df)
    grid = grid.join(rainfall, on=["province_id", "year", "month"], how="left")

    calendar = _month_calendar_features(dim_date, months)
    grid = grid.join(calendar, on=["year", "month"], how="left")

    grid = add_severity_index_coarse(grid)
    grid = add_severity_index_fine(grid)
    grid = grid.with_columns(
        pl.when(pl.col("accident_count") > 0)
        .then(pl.col("severity_index_coarse"))
        .otherwise(None)
        .alias("severity_index_coarse"),
        pl.when(pl.col("accident_count") > 0)
        .then(pl.col("severity_index_fine"))
        .otherwise(None)
        .alias("severity_index_fine"),
    )

    return grid.select(
        "geography_sk",
        "province_id",
        "province_th",
        "region",
        "vehicle_sk",
        "vehicle_group",
        "has_normalized_exposure",
        "year",
        "month",
        "season",
        "accident_count",
        "fatality_count",
        "serious_injury_count",
        "minor_injury_count",
        "total_injury_count",
        "severity_index_coarse",
        "severity_index_fine",
        "exposure_vehicle_count",
        "log_exposure",
        "rain_lead0",
        "rain_lead1",
        "rain_lead2",
        "rain_lead3",
        "rain_lead4",
        "rain_lead5",
        "flood_pct_high_risk_tambons",
        "flood_peak_month",
        "flood_data_available",
        "r35mm_avg_days_2014_2023",
        "ev_share",
        "tourist_attraction_count",
        "tourism_tier",
        "n_holiday_days",
        "n_long_weekend_days",
        "has_songkran",
        "has_newyear",
        "avg_days_to_next_holiday",
        "covid_disruption_level",
    ).sort(["province_th", "vehicle_group", "year", "month"])
