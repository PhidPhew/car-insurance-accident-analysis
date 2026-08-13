"""Phase 2 Step 2: apply committed category-taxonomy mappings (CLAUDE.md P4).

Must run BEFORE dedup (Step 4) -- CLAUDE.md P5: 74% of duplicate-pair
conflicts on `บริเวณที่เกิดเหตุ` are taxonomy-drift synonyms across two
overlapping vocabularies, not genuine disagreement, and deduping first
would resolve them as conflicts instead of collapsing them first.

The committed CSV tables in `config/mappings/` (built by
`scripts/build_taxonomy_mappings.py`, hand-reviewed) are the source of
truth -- this module only applies them, it never re-derives a mapping.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class TaxonomyMapping:
    """A loaded raw-value -> canonical-group lookup table."""

    raw_to_group: dict[str, str]

    def group_for(self, raw_value: str | None) -> str | None:
        """Canonical group for a raw value, or None if null or not in the committed table."""
        if raw_value is None:
            return None
        return self.raw_to_group.get(raw_value)


def load_taxonomy_mapping(path: Path) -> TaxonomyMapping:
    """Load a committed `raw_value,canonical_group[,row_count]` CSV."""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return TaxonomyMapping({row["raw_value"]: row["canonical_group"] for row in reader})


@dataclass(frozen=True)
class TaxonomyApplyLog:
    """How a taxonomy mapping was applied to one column (N4 audit trail)."""

    column: str
    total_rows: int
    rows_mapped: int
    rows_null: int
    rows_unmapped: int
    unmapped_values: tuple[str, ...]


def apply_taxonomy(
    df: pl.DataFrame, column: str, mapping: TaxonomyMapping, new_column: str | None = None
) -> tuple[pl.DataFrame, TaxonomyApplyLog]:
    """Add a `{column}_group` column (or `new_column`) derived from `column` via `mapping`.

    A raw value present in the data but absent from the committed mapping
    table becomes null in the output (N7: never drop a row for this) --
    NOT an error, since a future re-run against new data could see a
    genuinely novel category value. The returned log surfaces exactly which
    values were unmapped so a maintainer updates the committed CSV, rather
    than this silently mapping an unknown category to some existing group.
    """
    out_col = new_column or f"{column}_group"
    raw_values = df[column].to_list()
    groups = [mapping.group_for(v) for v in raw_values]
    result = df.with_columns(pl.Series(out_col, groups, dtype=pl.Utf8))

    unmapped_raw_values = [
        v for v, g in zip(raw_values, groups, strict=True) if v is not None and g is None
    ]
    rows_null = sum(1 for v in raw_values if v is None)
    rows_unmapped = len(unmapped_raw_values)
    rows_mapped = len(raw_values) - rows_null - rows_unmapped

    log = TaxonomyApplyLog(
        column=column,
        total_rows=len(raw_values),
        rows_mapped=rows_mapped,
        rows_null=rows_null,
        rows_unmapped=rows_unmapped,
        unmapped_values=tuple(sorted(set(unmapped_raw_values))),
    )
    return result, log
