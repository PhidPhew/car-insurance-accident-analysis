"""`dim_time_of_day` builder (CLAUDE.md §8, Phase 3 Step 1). Type 0 -- 24 hours + bands.

One row per hour of day (0-23), bucketed into the bands SiamShield's
business questions actually use (BQ1/BQ3 talk about time-of-day risk, not
minute-level precision). Band boundaries are a stated design choice, not
derived from data:

- late_night   00:00-03:59
- early_morning 04:00-06:59
- morning      07:00-09:59
- midday       10:00-13:59
- afternoon    14:00-16:59
- evening      17:00-19:59
- night        20:00-23:59
"""

from __future__ import annotations

import polars as pl

from rri.warehouse.keys import prepend_unknown_member

_BAND_BY_HOUR = {
    0: "late_night",
    1: "late_night",
    2: "late_night",
    3: "late_night",
    4: "early_morning",
    5: "early_morning",
    6: "early_morning",
    7: "morning",
    8: "morning",
    9: "morning",
    10: "midday",
    11: "midday",
    12: "midday",
    13: "midday",
    14: "afternoon",
    15: "afternoon",
    16: "afternoon",
    17: "evening",
    18: "evening",
    19: "evening",
    20: "night",
    21: "night",
    22: "night",
    23: "night",
}

_IS_NIGHT_BAND = frozenset({"late_night", "night"})


def build_dim_time_of_day() -> pl.DataFrame:
    """Build the finished `dim_time_of_day` table, including the Unknown Member row."""
    hours = pl.DataFrame({"hour": list(range(24))}, schema={"hour": pl.Int64})
    df = hours.with_columns(
        pl.col("hour").alias("time_of_day_sk"),
        pl.col("hour").replace_strict(_BAND_BY_HOUR).alias("time_band"),
    ).with_columns(pl.col("time_band").is_in(list(_IS_NIGHT_BAND)).alias("is_night"))

    return prepend_unknown_member(
        df,
        "time_of_day_sk",
        overrides={"hour": -1, "time_band": "Unknown", "is_night": False},
    )
