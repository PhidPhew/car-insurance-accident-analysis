"""Surrogate-key + Unknown Member helpers shared by every dimension builder.

CLAUDE.md N7: a fact row must never be dropped for a missing dimension --
every dimension gets exactly one Unknown Member row at ``sk = -1`` for a
fact's FK to point at instead.
"""

from __future__ import annotations

from typing import Any

import polars as pl

UNKNOWN_SK: int = -1


def assign_surrogate_keys(df: pl.DataFrame, sk_column: str, *, start: int = 1) -> pl.DataFrame:
    """Add an ``Int64`` surrogate key column, numbered ``start, start+1, ...`` in row order.

    Caller must sort ``df`` into its final deterministic order (N9) first --
    this only numbers whatever order it's handed.
    """
    return df.with_row_index(sk_column, offset=start).with_columns(pl.col(sk_column).cast(pl.Int64))


def prepend_unknown_member(
    df: pl.DataFrame, sk_column: str, overrides: dict[str, Any]
) -> pl.DataFrame:
    """Prepend one Unknown Member row (``sk = -1``) built from ``df``'s own schema.

    Every column defaults to null; pass ``overrides`` for columns that need a
    real value (e.g. a "Unknown" label, or a boolean flag set to a sentinel).
    """
    unknown_row = {col: overrides.get(col) for col in df.columns}
    unknown_row[sk_column] = UNKNOWN_SK
    unknown_df = pl.DataFrame([unknown_row], schema=df.schema)
    return pl.concat([unknown_df, df], how="vertical")
