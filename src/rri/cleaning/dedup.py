"""Phase 2 Step 4: deduplicate accidents by ACC_CODE (CLAUDE.md P5).

Must run AFTER Step 2 (taxonomy harmonization) and Step 3 (date resolution)
-- see `rri.cleaning.taxonomy` for why harmonize-before-dedup matters, and
`rri.cleaning.dates` for `date_certainty`, which this module's tie-break
depends on.

Tie-break order (decided, CLAUDE.md P5 + owner instruction), applied only
within a group (rows sharing the same non-null `ACC_CODE`):
1. Prefer `date_certainty == "certain"` over `"ambiguous"`.
2. If tied, prefer the row with fewer nulls (across the caller-specified
   business columns -- not the pipeline's own derived/metadata columns).
3. If still tied, prefer a non-null `ACC_CODE` (a no-op for this dataset,
   since a group's key is by definition non-null within the group -- kept
   for parity with the stated rule in case this is reused where grouping
   semantics differ).
4. If still tied, keep the first row by `_row_hash` (deterministic, N9).

Rows with a null `ACC_CODE` are never deduplicated (nothing to group them
by) and pass through unchanged, matching Phase 1's `compute_dedup_impact`.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from rri.utils.logging import get_logger
from rri.utils.profiling import check_within_group_agreement

log = get_logger(__name__)

#: The 4 casualty columns P5 proved agree 100% within every ACC_CODE group.
CASUALTY_COLUMNS = (
    "ผู้เสียชีวิต",
    "ผู้บาดเจ็บสาหัส",
    "ผู้บาดเจ็บเล็กน้อย",
    "รวมจำนวนผู้บาดเจ็บ",
)


def assert_casualty_agreement(
    df: pl.DataFrame,
    group_key_column: str = "ACC_CODE",
    casualty_columns: tuple[str, ...] = CASUALTY_COLUMNS,
) -> None:
    """Fail loudly if any ACC_CODE group disagrees on a casualty count (CLAUDE.md P5).

    P5 proved this holds for all 1,641 real duplicate groups -- a violation here means
    reality no longer matches that finding, and dedup must not silently proceed picking
    an arbitrary casualty total in that case.
    """
    for column in casualty_columns:
        result = check_within_group_agreement(df, [group_key_column], column)
        if result.conflict_count > 0:
            raise ValueError(
                f"{result.conflict_count} {group_key_column} groups disagree on {column!r} -- "
                "CLAUDE.md P5 proved 0 conflicts on the real data; do not dedup until this "
                "is understood, casualty totals must never be silently picked"
            )


@dataclass(frozen=True)
class DedupExample:
    """One collapsed group's before/after, for the N4 audit trail."""

    group_key: str
    rows_in_group: int
    kept_certainty: str
    dropped_certainties: tuple[str, ...]


@dataclass(frozen=True)
class DedupResult:
    rows_before: int
    rows_after: int
    groups_collapsed: int
    examples: tuple[DedupExample, ...]


def deduplicate_accidents(
    df: pl.DataFrame,
    columns_for_null_count: list[str],
    group_key_column: str = "ACC_CODE",
    certainty_column: str = "date_certainty",
    row_hash_column: str = "_row_hash",
    example_count: int = 5,
) -> tuple[pl.DataFrame, DedupResult]:
    """Collapse each `group_key_column` duplicate group to 1 row via the tie-break above.

    `columns_for_null_count` should be the original business columns (not this
    pipeline's own derived/metadata columns) -- completeness is judged on the source
    data, not on columns every row has identically by construction.

    Calls `assert_casualty_agreement` first and raises before touching anything if it
    fails -- dedup must never silently proceed past a broken assumption it depends on.
    """
    assert_casualty_agreement(df, group_key_column)

    keyed = df.filter(pl.col(group_key_column).is_not_null())
    unkeyed = df.filter(pl.col(group_key_column).is_null())

    group_sizes = keyed.group_by(group_key_column).agg(pl.len().alias("_group_size"))
    duplicate_keys = group_sizes.filter(pl.col("_group_size") > 1)[group_key_column].to_list()

    null_count_expr = pl.sum_horizontal(
        [pl.col(c).is_null().cast(pl.Int32) for c in columns_for_null_count]
    )
    ranked = keyed.with_columns(
        (pl.col(certainty_column) != "certain").cast(pl.Int8).alias("_certain_rank"),
        null_count_expr.alias("_null_count"),
        pl.col(group_key_column).is_null().cast(pl.Int8).alias("_acc_code_null_rank"),
    ).sort(["_certain_rank", "_null_count", "_acc_code_null_rank", row_hash_column])

    examples = []
    if duplicate_keys:
        sample_keys = duplicate_keys[:example_count]
        example_frame = ranked.filter(pl.col(group_key_column).is_in(sample_keys))
        for key, group in example_frame.group_by(group_key_column, maintain_order=True):
            certainties = group[certainty_column].to_list()
            examples.append(
                DedupExample(
                    group_key=str(key[0]),
                    rows_in_group=len(certainties),
                    kept_certainty=certainties[0],
                    dropped_certainties=tuple(certainties[1:]),
                )
            )

    deduped = ranked.unique(subset=[group_key_column], keep="first", maintain_order=True).drop(
        ["_certain_rank", "_null_count", "_acc_code_null_rank"]
    )

    result_df = pl.concat([deduped, unkeyed], how="vertical")

    dedup_result = DedupResult(
        rows_before=df.height,
        rows_after=result_df.height,
        groups_collapsed=len(duplicate_keys),
        examples=tuple(examples),
    )
    log.info(
        "accidents_deduplicated",
        rows_before=dedup_result.rows_before,
        rows_after=dedup_result.rows_after,
        groups_collapsed=dedup_result.groups_collapsed,
        examples=[
            {
                "group_key": e.group_key,
                "rows_in_group": e.rows_in_group,
                "kept_certainty": e.kept_certainty,
                "dropped_certainties": e.dropped_certainties,
            }
            for e in dedup_result.examples
        ],
    )
    return result_df, dedup_result
