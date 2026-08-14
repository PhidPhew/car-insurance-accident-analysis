"""`dim_date` builder (CLAUDE.md §8, Phase 3 Step 1). Type 0 -- a calendar never changes.

One row per calendar day across the full analysis window
(2019-01-01..2026-12-31, the exact range Phase 1's holiday backfill already
covers -- `config/holiday_flags_2019_2026.csv`). That file supplies every
holiday/COVID flag Phase 1 derived; this module only adds calendar
attributes (year/month/quarter/day-of-week/`season`) that weren't part of
Phase 1's scope.

`season` assumption (not sourced from any raw file, so stated here): the
commonly cited 3-season Thai calendar -- hot Mar-May, rainy Jun-Oct, cool
Nov-Feb. TMD's own boundaries float by a few weeks year to year; this uses
fixed month boundaries for determinism (N9).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from config.settings import settings
from rri.warehouse.keys import prepend_unknown_member

_SEASON_BY_MONTH = {
    3: "hot",
    4: "hot",
    5: "hot",
    6: "rainy",
    7: "rainy",
    8: "rainy",
    9: "rainy",
    10: "rainy",
    11: "cool",
    12: "cool",
    1: "cool",
    2: "cool",
}

_DAY_NAMES = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday",
}

_MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def date_expr_to_sk(date_col: str = "date") -> pl.Expr:
    """`YYYYMMDD` int expression for `date_col` -- the one true `date_sk` formula.

    `dt.month()`/`dt.day()` return `i8` in polars, which silently overflows
    when multiplied by 100/10_000 before promotion (caught once here --
    2019-09-01 computed as 20189877, not 20190901). Every FK derivation
    (this module, `fact_accident`) must route through this function rather
    than re-deriving the arithmetic, so that bug can't reappear in a second
    place.
    """
    return (
        pl.col(date_col).dt.year() * 10_000
        + pl.col(date_col).dt.month().cast(pl.Int32) * 100
        + pl.col(date_col).dt.day().cast(pl.Int32)
    )


def build_dim_date(holiday_flags_path: Path | None = None) -> pl.DataFrame:
    """Build the finished `dim_date` table, including the Unknown Member row."""
    path = holiday_flags_path or (settings.config_dir / "holiday_flags_2019_2026.csv")
    df = pl.read_csv(path, try_parse_dates=True)

    df = df.with_columns(
        date_expr_to_sk("date").alias("date_sk"),
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.quarter().alias("quarter"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.month().replace_strict(_MONTH_NAMES).alias("month_name"),
        pl.col("date").dt.day().alias("day"),
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("date").dt.weekday().replace_strict(_DAY_NAMES).alias("day_name"),
        pl.col("date").dt.week().alias("week_of_year"),
        pl.col("date").dt.month().replace_strict(_SEASON_BY_MONTH).alias("season"),
    ).select(
        "date_sk",
        "date",
        "year",
        "quarter",
        "month",
        "month_name",
        "day",
        "day_of_week",
        "day_name",
        "week_of_year",
        "season",
        "is_public_holiday",
        "is_bank_holiday",
        "is_private_sector_holiday",
        "is_weekend",
        "is_long_weekend",
        "is_songkran_7days",
        "is_newyear_7days",
        "is_holiday_eve",
        "days_to_next_holiday",
        "covid_disruption_level",
    )

    df = df.sort("date_sk")

    return prepend_unknown_member(
        df,
        "date_sk",
        overrides={
            "year": -1,
            "month": -1,
            "day": -1,
            "month_name": "Unknown",
            "day_name": "Unknown",
            "season": "Unknown",
            "covid_disruption_level": "unknown",
            "is_public_holiday": False,
            "is_bank_holiday": False,
            "is_private_sector_holiday": False,
            "is_weekend": False,
            "is_long_weekend": False,
            "is_songkran_7days": False,
            "is_newyear_7days": False,
            "is_holiday_eve": False,
        },
    )
