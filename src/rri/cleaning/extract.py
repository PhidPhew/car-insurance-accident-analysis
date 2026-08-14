"""Phase 2 Step 1: extract raw CSVs to staging parquet + structural fixes.

Structural fixes here don't depend on row identity across rows (no dedup,
no cross-row category harmonization) -- see `rri.cleaning.taxonomy` (Step 2)
and `rri.cleaning.dates` (Step 3) for those. `data/00_raw/` is read-only
(N1); every function here takes a DataFrame already read into memory and
returns a new one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from rri.contracts.raw import TWIN_COLUMN_PAIRS

#: Standard null-value tokens for every raw CSV in this project (matches Phase 1 profiling).
RAW_NULL_VALUES = ["", "NA", "N/A", "-"]


def read_raw_csv(path: Path) -> pl.DataFrame:
    """Read one raw CSV with the correct BOM-aware encoding (P7.2) and standard null tokens.

    Always reads every column as string (`infer_schema_length=0`) -- casting to a typed
    dtype is each fix function's job, not the reader's, so a column that looks numeric but
    contains a placeholder like `"//"` (P7.10) isn't silently coerced to null at read time.
    """
    return pl.read_csv(
        path,
        encoding=csv_encoding_for(path.name),
        infer_schema_length=0,
        null_values=RAW_NULL_VALUES,
    )


# P7.2: BOM present on 5 of 6 raw CSVs -- stt_car_fuel is the one exception
# (see CLAUDE.md P7.2 / docs/data_quality_findings.md).
BOM_CSV_FILENAMES = frozenset(
    {
        "thailand_accidents_2019_2026_combined.csv",
        "thailand_rainfall_2021_2026_combined.csv",
        "thailand_flood_risk_areas.csv",
        "thailand_heavy_rain_R35mm.csv",
        "car_registered_new_2565_2569_combined.csv",
    }
)
NO_BOM_CSV_FILENAMES = frozenset({"stt_car_fuel_at_25690228.csv"})


def csv_encoding_for(filename: str) -> str:
    """Return the correct read encoding for a raw CSV filename (P7.2).

    Raises `ValueError` for an unrecognized filename rather than silently
    guessing -- a new raw file must be explicitly classified here first.
    """
    if filename in BOM_CSV_FILENAMES:
        return "utf-8-sig"
    if filename in NO_BOM_CSV_FILENAMES:
        return "utf-8"
    raise ValueError(f"unknown raw CSV filename: {filename!r} -- classify it in extract.py first")


def compute_row_hash(df: pl.DataFrame, seed: int = 42) -> pl.Series:
    """Deterministic per-row hash (N9) over every column's current values.

    Used as `_row_hash` for a stable, reproducible dedup tie-break (Step 4)
    -- not a content-addressed identity that survives column changes, since
    it is computed fresh at whatever point in the pipeline it's called.
    """
    return (
        df.hash_rows(seed=seed)
        .map_elements(lambda h: f"{h:016x}", return_dtype=pl.Utf8)
        .rename("_row_hash")
    )


def add_extract_metadata(df: pl.DataFrame, source_file: str, ingested_at: datetime) -> pl.DataFrame:
    """Add `_source_file`, `_ingested_at`, `_row_hash` to a freshly-read raw frame."""
    with_meta = df.with_columns(
        pl.lit(source_file).alias("_source_file"),
        pl.lit(ingested_at).alias("_ingested_at"),
    )
    return with_meta.with_columns(compute_row_hash(with_meta))


def rename_headers(df: pl.DataFrame, renames: dict[str, str]) -> pl.DataFrame:
    """Rename headers present in `df`; a mapping key absent from `df` is a no-op, not an error.

    Callers pass the exact known-typo mapping (e.g. `{"Foreast_Month": "Forecast_Month"}`,
    P7.3) rather than this function guessing anything.
    """
    applicable = {old: new for old, new in renames.items() if old in df.columns}
    return df.rename(applicable) if applicable else df


@dataclass(frozen=True)
class TwinColumnCoalesceLog:
    """How many rows a twin-column coalesce (P3) pulled from each side."""

    old_column: str
    new_column: str
    total_rows: int
    rows_from_old: int
    rows_from_new: int
    rows_still_null: int


def coalesce_twin_columns(
    df: pl.DataFrame, pairs: tuple[tuple[str, str], ...] = TWIN_COLUMN_PAIRS
) -> tuple[pl.DataFrame, list[TwinColumnCoalesceLog]]:
    """Coalesce each proven zero-overlap twin-column pair (P3) into the old column's name.

    Only the 4 pairs proven safe in Phase 1 (`rri.contracts.raw.TWIN_COLUMN_PAIRS`) are
    coalesced by default -- the 5th pair (`บริเวณที่เกิดเหตุ`/`บริเวณที่เกิดเหตุอื่น`) is
    deliberately excluded there already, since Phase 1 proved it is NOT a twin pair.
    Raises if a pair has any overlap (both non-null on the same row) -- that would mean
    reality no longer matches the Phase 1 finding this function is built on, and coalescing
    would silently discard the "new" column's value.
    """
    out = df
    logs = []
    for old_col, new_col in pairs:
        overlap = int((df[old_col].is_not_null() & df[new_col].is_not_null()).sum())
        if overlap != 0:
            raise ValueError(
                f"twin-column pair ({old_col!r}, {new_col!r}) has {overlap} overlapping rows -- "
                "Phase 1's zero-overlap finding no longer holds for this data, do not coalesce"
            )
        rows_from_old = int(df[old_col].is_not_null().sum())
        rows_from_new = int(df[new_col].is_not_null().sum())
        out = out.with_columns(pl.coalesce(pl.col(old_col), pl.col(new_col)).alias(old_col)).drop(
            new_col
        )
        logs.append(
            TwinColumnCoalesceLog(
                old_column=old_col,
                new_column=new_col,
                total_rows=df.height,
                rows_from_old=rows_from_old,
                rows_from_new=rows_from_new,
                rows_still_null=df.height - rows_from_old - rows_from_new,
            )
        )
    return out, logs


def convert_buddhist_year(
    df: pl.DataFrame, column: str, new_column: str = "year_ce"
) -> pl.DataFrame:
    """Add a Gregorian-year column (`CE = BE - 543`, P7.4) alongside the original BE column.

    Non-destructive: the raw Buddhist-era column is kept as-is for traceability.
    """
    return df.with_columns((pl.col(column).cast(pl.Int64) - 543).alias(new_column))


def pivot_rainfall_leads(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot the rainfall file's 6 lead-time rows per (initial period, province) into columns.

    P7.5: this is NOT deduplication -- every one of the 6 rows per group is a genuinely
    distinct forecast (lead 0..5 months ahead issued from the same initial period), so all
    6 values must survive as `rain_lead0`..`rain_lead5`, not be dropped or averaged.
    `lead` is computed from the year/month pair, not assumed to already be 0..5 in file order.
    """
    with_lead = df.with_columns(
        (
            (pl.col("Forecast_Year").cast(pl.Int64) - pl.col("Initial_Year").cast(pl.Int64)) * 12
            + (pl.col("Forecast_Month").cast(pl.Int64) - pl.col("Initial_Month").cast(pl.Int64))
        ).alias("lead"),
        pl.col("Rainfall").cast(pl.Float64).alias("Rainfall"),
    )
    pivoted = with_lead.pivot(
        on="lead",
        index=["Initial_Year", "Initial_Month", "Province_ID", "Province_NameTH"],
        values="Rainfall",
    )
    lead_renames = {str(i): f"rain_lead{i}" for i in range(6)}
    return pivoted.rename({old: new for old, new in lead_renames.items() if old in pivoted.columns})
