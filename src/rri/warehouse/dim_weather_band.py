"""`dim_weather_band` builder (CLAUDE.md §8, Phase 3 Step 1). Type 0 -- rainfall/visibility bands.

`fact_accident`'s grain includes "one reported weather condition" -- the
officer-reported `สภาพอากาศ` free-text on each accident row, which has
exactly 7 non-null values in staging. This module buckets those 7 into
6 business bands (a stated design choice, not derived from data):

- clear           แจ่มใส
- rain            ฝนตก
- low_visibility  มีหมอก/ควัน/ฝุ่น, มืดครึ้ม (fog/smoke/dust and overcast/dark
                   both impair visibility the same way for driving risk)
- severe_weather  ภัยธรรมชาติ เช่น พายุ น้ำท่วม, ดินถล่ม (storm/flood/landslide)
- other           อื่นๆ

The dimension's grain is one row per band (not per raw value) -- multiple
raw strings collapse onto the same band, which is what `fact_accident`'s FK
actually needs.
"""

from __future__ import annotations

import polars as pl

from rri.warehouse.keys import prepend_unknown_member

#: raw `สภาพอากาศ` value -> band. Single source of truth; `fact_accident` imports
#: this same dict rather than re-deriving the mapping.
WEATHER_BAND_BY_RAW_VALUE: dict[str, str] = {
    "แจ่มใส": "clear",
    "ฝนตก": "rain",
    "มีหมอก/ควัน/ฝุ่น": "low_visibility",
    "มืดครึ้ม": "low_visibility",
    "ภัยธรรมชาติ เช่น พายุ น้ำท่วม": "severe_weather",
    "ดินถล่ม": "severe_weather",
    "อื่นๆ": "other",
}

_HAZARDOUS_BANDS = frozenset({"rain", "low_visibility", "severe_weather"})

_BAND_ORDER = ("clear", "rain", "low_visibility", "severe_weather", "other")


def build_dim_weather_band() -> pl.DataFrame:
    """Build the finished `dim_weather_band` table, including the Unknown Member row."""
    df = pl.DataFrame({"weather_band": list(_BAND_ORDER)}).with_columns(
        pl.col("weather_band").is_in(list(_HAZARDOUS_BANDS)).alias("is_hazardous")
    )
    df = df.with_row_index("weather_band_sk", offset=1).with_columns(
        pl.col("weather_band_sk").cast(pl.Int64)
    )

    return prepend_unknown_member(
        df,
        "weather_band_sk",
        overrides={"weather_band": "Unknown", "is_hazardous": False},
    )
