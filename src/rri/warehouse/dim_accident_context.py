"""`dim_accident_context` builder (CLAUDE.md §8, Phase 3 Step 1). Type 1 -- junk dimension.

Grain: one row per distinct combination of the 4 harmonized group columns
Phase 2 already produced on the accidents staging frame --
`มูลเหตุสันนิษฐาน_group` (cause), `ลักษณะการเกิดเหตุ_group` (crash type),
`บริเวณที่เกิดเหตุ_group` / `บริเวณที่เกิดเหตุอื่น_group` (primary / secondary
road-geometry location -- CLAUDE.md P3: proven NOT a twin pair, so both are
carried as separate attributes rather than coalesced).

This is a junk dimension (Kimball terminology): rather than a full cross
product of the 4 columns' distinct values, it holds only the ~1,700 combos
that actually occur in the data, built directly from the staging frame so
every fact row is guaranteed a match -- a null in any of the 4 source
columns becomes the literal value `"Unknown"` *within its own combo* before
grouping, so it's a normal dimension member, not a dropped row (N7).
"""

from __future__ import annotations

import polars as pl

from rri.warehouse.keys import assign_surrogate_keys, prepend_unknown_member

CONTEXT_COLUMNS = (
    "มูลเหตุสันนิษฐาน_group",
    "ลักษณะการเกิดเหตุ_group",
    "บริเวณที่เกิดเหตุ_group",
    "บริเวณที่เกิดเหตุอื่น_group",
)

CONTEXT_COLUMN_RENAME = {
    "มูลเหตุสันนิษฐาน_group": "cause_group",
    "ลักษณะการเกิดเหตุ_group": "crash_type_group",
    "บริเวณที่เกิดเหตุ_group": "road_geometry_group",
    "บริเวณที่เกิดเหตุอื่น_group": "road_geometry_other_group",
}


def build_dim_accident_context(accidents_df: pl.DataFrame) -> pl.DataFrame:
    """Build `dim_accident_context` from the staging accidents frame's realized combos."""
    combos = (
        accidents_df.select(list(CONTEXT_COLUMNS))
        .with_columns([pl.col(c).fill_null("Unknown") for c in CONTEXT_COLUMNS])
        .unique()
        .rename(CONTEXT_COLUMN_RENAME)
        .sort(list(CONTEXT_COLUMN_RENAME.values()))
    )

    combos = assign_surrogate_keys(combos, "accident_context_sk")

    return prepend_unknown_member(
        combos,
        "accident_context_sk",
        overrides={
            "cause_group": "Unknown",
            "crash_type_group": "Unknown",
            "road_geometry_group": "Unknown",
            "road_geometry_other_group": "Unknown",
        },
    )
