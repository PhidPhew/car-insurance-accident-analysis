"""`fact_accident` builder (CLAUDE.md §8, Phase 3 Step 2). Grain: one road accident.

Reads the Phase 2 staging frame (`data/10_staging/accidents.parquet`, 159,605
rows -- Gate 2) and resolves every dimension FK by joining a normalized key
against the already-built dim tables (Step 1 must run first). CLAUDE.md N7:
a row is never dropped for a missing/unmatched dimension value -- every FK
that fails to resolve coalesces to `sk = -1` (Unknown Member) instead.

`agency` (`หน่วยงาน`) is carried as a degenerate attribute, not a dimension
FK, despite CLAUDE.md §8 listing `agency_sk` in the fact's FK list -- the
§8 "Dimensions (8)" enumeration (and this phase's own build instructions,
which say "build all 8 dimensions" and name exactly the 8 built in Step 1)
never lists a `dim_agency` table. With only 3 possible values, a degenerate
column serves the same purpose (an agency filter in the dashboard, per
P7.13) without a phantom 9th dimension. Flagged for a CLAUDE.md §8 fix.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from rri.cleaning.province import normalize_province
from rri.cleaning.taxonomy import apply_taxonomy, load_taxonomy_mapping
from rri.warehouse.dim_accident_context import CONTEXT_COLUMN_RENAME, CONTEXT_COLUMNS
from rri.warehouse.dim_date import date_expr_to_sk
from rri.warehouse.dim_weather_band import WEATHER_BAND_BY_RAW_VALUE
from rri.warehouse.keys import UNKNOWN_SK


def _weather_band_for(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    return WEATHER_BAND_BY_RAW_VALUE.get(raw_value)


def _resolve_fk(
    df: pl.DataFrame,
    key_col: str,
    dim: pl.DataFrame,
    dim_key_col: str,
    dim_sk_col: str,
    fk_name: str,
) -> pl.DataFrame:
    """Left-join `df[key_col]` onto `dim[dim_key_col]`, coalescing an unmatched join to sk=-1.

    Joins through isolated internal column names (`_join_key`/`_join_sk`) so
    this works even when `dim_key_col`/`dim_sk_col`/`fk_name` collide (e.g.
    `dim_date`, where the natural key and the surrogate key are the same
    column) without one alias silently clobbering another.
    """
    joined = df.join(
        dim.select(pl.col(dim_key_col).alias("_join_key"), pl.col(dim_sk_col).alias("_join_sk")),
        left_on=key_col,
        right_on="_join_key",
        how="left",
    )
    return joined.with_columns(pl.col("_join_sk").fill_null(UNKNOWN_SK).alias(fk_name)).drop(
        "_join_sk"
    )


def build_fact_accident(
    accidents_df: pl.DataFrame,
    dim_date: pl.DataFrame,
    dim_geography: pl.DataFrame,
    dim_vehicle: pl.DataFrame,
    dim_accident_context: pl.DataFrame,
    dim_time_of_day: pl.DataFrame,
    dim_weather_band: pl.DataFrame,
    vehicle_mapping_path: Path,
) -> pl.DataFrame:
    """Build the finished `fact_accident` table (no Unknown Member row -- facts don't get one)."""
    vehicle_mapping = load_taxonomy_mapping(vehicle_mapping_path)

    df = accidents_df.with_columns(
        pl.col("จังหวัด")
        .map_elements(normalize_province, return_dtype=pl.Utf8)
        .alias("_province_th"),
        pl.col("สภาพอากาศ")
        .map_elements(_weather_band_for, return_dtype=pl.Utf8)
        .alias("_weather_band"),
        date_expr_to_sk("accident_date").alias("_date_sk"),
        pl.col("accident_time").dt.hour().alias("_hour"),
        *[pl.col(c).fill_null("Unknown").alias(f"_{c}") for c in CONTEXT_COLUMNS],
    )
    df, _vehicle_log = apply_taxonomy(df, "รถคันที่1", vehicle_mapping, new_column="_vehicle_group")

    valid_date_sks = dim_date["date_sk"].to_list()
    df = df.with_columns(
        pl.when(pl.col("_date_sk").is_in(valid_date_sks))
        .then(pl.col("_date_sk"))
        .otherwise(UNKNOWN_SK)
        .alias("date_sk")
    )
    df = _resolve_fk(
        df, "_province_th", dim_geography, "province_th", "geography_sk", "geography_sk"
    )
    df = _resolve_fk(df, "_vehicle_group", dim_vehicle, "vehicle_group", "vehicle_sk", "vehicle_sk")
    df = _resolve_fk(df, "_hour", dim_time_of_day, "hour", "time_of_day_sk", "time_of_day_sk")
    df = _resolve_fk(
        df, "_weather_band", dim_weather_band, "weather_band", "weather_band_sk", "weather_band_sk"
    )

    context_key_expr = pl.concat_str(
        [pl.col(f"_{c}") for c in CONTEXT_COLUMNS], separator="||"
    ).alias("_context_key")
    dim_context_key_expr = pl.concat_str(
        [pl.col(c) for c in CONTEXT_COLUMN_RENAME.values()], separator="||"
    ).alias("_context_key")
    df = df.with_columns(context_key_expr)
    # Exclude the Unknown Member (sk=-1) from the join candidates: its
    # all-"Unknown" key is identical to the real combo (sk=1) built from
    # source rows where all 4 group columns were null -- joining against
    # both would fan out every such fact row into 2 output rows. _resolve_fk
    # already coalesces a genuinely unmatched key to -1, so dropping it here
    # costs nothing.
    dim_context_keyed = dim_accident_context.filter(
        pl.col("accident_context_sk") != UNKNOWN_SK
    ).with_columns(dim_context_key_expr)
    df = _resolve_fk(
        df,
        "_context_key",
        dim_context_keyed,
        "_context_key",
        "accident_context_sk",
        "accident_context_sk",
    )

    df = df.with_columns(
        pl.col("ACC_CODE").alias("acc_code"),
        pl.col("รหัสสายทาง").alias("road_code"),
        pl.col("KM").cast(pl.Float64, strict=False).alias("km_marker"),
        pl.col("หน่วยงาน").alias("agency"),
        pl.lit(1).cast(pl.Int64).alias("accident_count"),
        pl.col("ผู้เสียชีวิต").cast(pl.Int64, strict=False).alias("fatality_count"),
        pl.col("ผู้บาดเจ็บสาหัส").cast(pl.Int64, strict=False).alias("serious_injury_count"),
        pl.col("ผู้บาดเจ็บเล็กน้อย").cast(pl.Int64, strict=False).alias("minor_injury_count"),
        pl.col("รวมจำนวนผู้บาดเจ็บ").cast(pl.Int64, strict=False).alias("total_injury_count"),
        pl.col("รถที่เกิดเหตุ").cast(pl.Int64, strict=False).alias("vehicle_involved_count"),
        pl.col("คนเดินเท้า").cast(pl.Int64, strict=False).alias("pedestrian_count"),
        (pl.col("report_date") - pl.col("accident_date"))
        .dt.total_days()
        .alias("reporting_lag_days"),
    )

    df = df.with_row_index("accident_sk", offset=1).with_columns(
        pl.col("accident_sk").cast(pl.Int64)
    )

    return df.select(
        "accident_sk",
        "acc_code",
        "road_code",
        "km_marker",
        "date_sk",
        "time_of_day_sk",
        "geography_sk",
        "vehicle_sk",
        "accident_context_sk",
        "weather_band_sk",
        "agency",
        "accident_count",
        "fatality_count",
        "serious_injury_count",
        "minor_injury_count",
        "total_injury_count",
        "vehicle_involved_count",
        "pedestrian_count",
        "reporting_lag_days",
        "has_injury_breakdown",
        "date_certainty",
        "is_geo_valid",
        pl.col("LATITUDE").alias("latitude"),
        pl.col("LONGITUDE").alias("longitude"),
    )
