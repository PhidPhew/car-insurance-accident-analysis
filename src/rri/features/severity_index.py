"""Severity Index -- two versions, not one (CLAUDE.md §9, decided 2026-08-14).

`accident2019.csv` never captured the serious/minor injury breakdown (P7/§6), so a
single three-tier severity formula is uncomputable for 19,194 rows. Rather than drop
the year (2019 is the project's only pre-COVID baseline), two measures are computed:

- `severity_index_coarse` = `(fatalities * W_F_COARSE + total_injuries * W_I_COARSE)
  / accidents` -- computable for every row, all years. Primary measure for anything
  comparing across years (BQ1, BQ3, BQ6).
- `severity_index_fine` = `(fatalities * SEVERITY_WEIGHT_FATAL + serious_injuries *
  SEVERITY_WEIGHT_SERIOUS + minor_injuries * SEVERITY_WEIGHT_MINOR) / accidents` --
  computable only where `has_injury_breakdown` is true (2020-2025 in-range rows; 2026
  is an incomplete holdout year, P7.11, so it is excluded from the correlation check
  below even though `has_injury_breakdown` is also true there).

Both formulas divide by `accidents` (the grain's own `accident_count`) so they apply
unchanged whether `accidents` is 1 (a single `fact_accident` row) or an aggregate
count (a province x month x vehicle_group cell).
"""

from __future__ import annotations

import polars as pl

from config.severity_index_params import W_F_COARSE, W_I_COARSE
from config.simulation_params import (
    SEVERITY_WEIGHT_FATAL,
    SEVERITY_WEIGHT_MINOR,
    SEVERITY_WEIGHT_SERIOUS,
)

#: The fine index's documented valid window (CLAUDE.md §9) -- 2020 is the first year
#: with a captured injury breakdown, 2025 the last complete year (2026 is a P7.11
#: holdout, not excluded by `has_injury_breakdown` alone).
FINE_INDEX_YEAR_START = 2020
FINE_INDEX_YEAR_END = 2025


def add_severity_index_coarse(
    df: pl.DataFrame,
    *,
    fatality_col: str = "fatality_count",
    total_injury_col: str = "total_injury_count",
    accident_col: str = "accident_count",
) -> pl.DataFrame:
    """Add `severity_index_coarse`, computable for every row (all years)."""
    return df.with_columns(
        (
            (pl.col(fatality_col) * W_F_COARSE + pl.col(total_injury_col) * W_I_COARSE)
            / pl.col(accident_col)
        ).alias("severity_index_coarse")
    )


def add_severity_index_fine(
    df: pl.DataFrame,
    *,
    fatality_col: str = "fatality_count",
    serious_injury_col: str = "serious_injury_count",
    minor_injury_col: str = "minor_injury_count",
    accident_col: str = "accident_count",
    has_breakdown_col: str = "has_injury_breakdown",
) -> pl.DataFrame:
    """Add `severity_index_fine`, null where `has_injury_breakdown` is false (Gate B, §8)."""
    fine_score = (
        pl.col(fatality_col) * SEVERITY_WEIGHT_FATAL
        + pl.col(serious_injury_col) * SEVERITY_WEIGHT_SERIOUS
        + pl.col(minor_injury_col) * SEVERITY_WEIGHT_MINOR
    ) / pl.col(accident_col)
    return df.with_columns(
        pl.when(pl.col(has_breakdown_col))
        .then(fine_score)
        .otherwise(None)
        .alias("severity_index_fine")
    )


def severity_index_correlation(fact_accident_with_year: pl.DataFrame) -> dict[str, float | int]:
    """Pearson correlation of coarse vs. fine index, record-level, 2020-2025 (CLAUDE.md §9).

    `fact_accident_with_year` must have a `year` column (join `dim_date` first) and
    already carry both index columns (`add_severity_index_coarse`/`_fine`). Restricted
    to `has_injury_breakdown = true` rows within `FINE_INDEX_YEAR_START..END` -- the
    only rows where both indices are simultaneously defined.
    """
    window = fact_accident_with_year.filter(
        pl.col("has_injury_breakdown"),
        pl.col("year").is_between(FINE_INDEX_YEAR_START, FINE_INDEX_YEAR_END),
    )
    correlation = window.select(
        pl.corr("severity_index_coarse", "severity_index_fine").alias("r")
    ).item()
    return {
        "pearson_r": float(correlation),
        "n_rows": window.height,
        "year_start": FINE_INDEX_YEAR_START,
        "year_end": FINE_INDEX_YEAR_END,
    }
