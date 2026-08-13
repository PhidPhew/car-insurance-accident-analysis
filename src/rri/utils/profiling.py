"""Generic, source-agnostic profiling functions for raw data files.

Phase 1 is observe-only (CLAUDE.md Task 4): these functions describe what is
in a file -- size, encoding, per-column stats, duplicate counts, null
correlation between columns, mixed date/time formats -- they never clean,
fix, or drop anything. Cleaning logic belongs in ``src/rri/cleaning/``
(Phase 2).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, TypeVar

import polars as pl

T = TypeVar("T")

# --------------------------------------------------------------------------
# File-level metadata
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FileMetadata:
    filename: str
    size_bytes: int
    encoding: str
    has_bom: bool
    delimiter: str | None
    line_ending: str


def detect_line_ending(path: Path) -> str:
    """Return "CRLF", "LF", or "mixed" based on the file's raw bytes."""
    raw = path.read_bytes()
    has_crlf = b"\r\n" in raw
    has_lone_lf = b"\n" in raw.replace(b"\r\n", b"")
    if has_crlf and has_lone_lf:
        return "mixed"
    if has_crlf:
        return "CRLF"
    return "LF"


def detect_delimiter(path: Path, sample_lines: int = 5) -> str | None:
    """Guess a CSV delimiter from the most common candidate in the first few lines.

    Returns None for non-delimited (e.g. JSON) files.
    """
    candidates = (",", ";", "\t", "|")
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        lines = [f.readline() for _ in range(sample_lines)]
    counts = {c: sum(line.count(c) for line in lines) for c in candidates}
    best = max(counts, key=lambda c: counts[c])
    return best if counts[best] > 0 else None


def profile_file_metadata(path: Path) -> FileMetadata:
    raw_head = path.read_bytes()[:4]
    has_bom = raw_head.startswith(b"\xef\xbb\xbf")
    is_json = path.suffix.lower() == ".json"
    return FileMetadata(
        filename=path.name,
        size_bytes=path.stat().st_size,
        encoding="utf-8-sig" if has_bom else "utf-8",
        has_bom=has_bom,
        delimiter=None if is_json else detect_delimiter(path),
        line_ending=detect_line_ending(path),
    )


# --------------------------------------------------------------------------
# Per-column profiling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    null_count: int
    null_pct: float
    distinct_count: int
    sample_values: tuple[str, ...]
    min_value: str | None
    max_value: str | None


def profile_column(df: pl.DataFrame, column: str) -> ColumnProfile:
    series = df.get_column(column)
    total = len(series)
    null_count = series.null_count()
    non_null = series.drop_nulls()

    sample = tuple(str(v) for v in non_null.head(5).to_list())

    min_value: str | None = None
    max_value: str | None = None
    if non_null.dtype.is_numeric() and len(non_null) > 0:
        min_value = str(non_null.min())
        max_value = str(non_null.max())

    return ColumnProfile(
        name=column,
        dtype=str(series.dtype),
        null_count=null_count,
        null_pct=(null_count / total * 100) if total else 0.0,
        distinct_count=non_null.n_unique(),
        sample_values=sample,
        min_value=min_value,
        max_value=max_value,
    )


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateProfile:
    full_row_duplicate_count: int
    candidate_key_columns: tuple[str, ...] | None
    candidate_key_duplicate_group_count: int
    candidate_key_duplicate_row_count: int
    candidate_key_null_count: int


def profile_duplicates(
    df: pl.DataFrame, candidate_key_columns: list[str] | None = None
) -> DuplicateProfile:
    full_row_dupes = len(df) - len(df.unique())

    if not candidate_key_columns:
        return DuplicateProfile(
            full_row_duplicate_count=full_row_dupes,
            candidate_key_columns=None,
            candidate_key_duplicate_group_count=0,
            candidate_key_duplicate_row_count=0,
            candidate_key_null_count=0,
        )

    key_null_count = df.filter(
        pl.any_horizontal([pl.col(c).is_null() for c in candidate_key_columns])
    ).height
    non_null_keys = df.drop_nulls(subset=candidate_key_columns)
    group_sizes = non_null_keys.group_by(candidate_key_columns).len()
    dup_groups = group_sizes.filter(pl.col("len") > 1)

    return DuplicateProfile(
        full_row_duplicate_count=full_row_dupes,
        candidate_key_columns=tuple(candidate_key_columns),
        candidate_key_duplicate_group_count=dup_groups.height,
        candidate_key_duplicate_row_count=int(dup_groups["len"].sum() or 0),
        candidate_key_null_count=key_null_count,
    )


# --------------------------------------------------------------------------
# Null-correlated column groups (schema-drift / twin-column signal)
# --------------------------------------------------------------------------


def find_null_correlated_column_groups(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    min_combined_coverage_pct: float = 50.0,
) -> list[tuple[str, str]]:
    """Find PAIRS of columns never populated on the same row that together cover most of it.

    This is the twin-column / schema-drift candidate signal (CLAUDE.md P3):
    an old column name used through one range of years and a new name used
    from some point on are never simultaneously populated, but their null
    percentages need not be similar or even both high -- one twin can be
    populated in most rows (the years it was in use) while the other is
    populated in only a few percent (the newer name's years).

    Reports pairs, not merged groups: several unrelated columns can be
    mutually exclusive with each other too (e.g. a set of one-hot vehicle-
    type indicator columns, where a row has exactly one type set) without
    being genuine schema-drift twins -- merging every zero-overlap column
    into one clique would bury the real signal in noise, and greedy
    clique-building is order-dependent and can pull in unrelated columns
    (verified against the real accidents file: a naive clique approach
    pulled ACC_CODE into the vehicle-type twin-column group purely because
    ACC_CODE happens to be null in the same year-range as one of the "new
    name" columns, despite being a structurally unrelated identifier
    column). Like the province alias table, this proposes CANDIDATE pairs
    for human review -- it does not claim certainty.
    """
    target_columns = columns or df.columns
    total = len(df)
    if total == 0:
        return []

    candidates = [col for col in target_columns if df.get_column(col).null_count() > 0]
    non_null_masks = {col: df.get_column(col).is_not_null() for col in candidates}

    pairs: list[tuple[str, str]] = []
    for i, col_a in enumerate(candidates):
        for col_b in candidates[i + 1 :]:
            overlap = int((non_null_masks[col_a] & non_null_masks[col_b]).sum())
            if overlap != 0:
                continue
            coverage_pct = int((non_null_masks[col_a] | non_null_masks[col_b]).sum()) / total * 100
            if coverage_pct >= min_combined_coverage_pct:
                pairs.append((col_a, col_b))
    return pairs


# --------------------------------------------------------------------------
# Mixed date/time format detection (observe-only: counts, never parses-to-fix)
# --------------------------------------------------------------------------

_SLASH_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$")


def classify_date_string(value: str) -> str:
    """Classify one raw date string. Never attempts to resolve it to a real date.

    Returns one of: "empty", "excel_serial", "slash_ambiguous" (both leading
    components <= 12), "slash_unambiguous" (one component > 12, so DMY vs.
    MDY is determinable), "other".
    """
    stripped = value.strip()
    if not stripped:
        return "empty"

    match = _SLASH_DATE_RE.match(stripped)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        return "slash_ambiguous" if a <= 12 and b <= 12 else "slash_unambiguous"

    try:
        as_float = float(stripped)
    except ValueError:
        return "other"
    if as_float > 0:
        return "excel_serial"
    return "other"


def profile_date_column_formats(values: list[str]) -> dict[str, int]:
    """Count how many values fall into each `classify_date_string` category."""
    counter: Counter[str] = Counter(classify_date_string(v) for v in values)
    return dict(counter)


def classify_slash_date_side_evidence(value: str) -> str | None:
    """Which side of a `a/b/y` string is the unambiguous >12 evidence on, if any?

    Returns "mdy_evidence" (second component > 12, so the first must be the
    month), "dmy_evidence" (first component > 12, so the first must be the
    day), "both_over_12" (both > 12 -- not a valid calendar date on either
    reading), "ambiguous" (neither > 12), or None for values that are not a
    plain two-slash date. Used to settle a per-file DMY-vs-MDY decision by
    checking whether ANY row contributes evidence for the losing side, not
    just counting the winning side.
    """
    stripped = value.strip()
    match = _SLASH_DATE_RE.match(stripped)
    if not match:
        return None
    a, b = int(match.group(1)), int(match.group(2))
    if a > 12 and b > 12:
        return "both_over_12"
    if a > 12:
        return "dmy_evidence"
    if b > 12:
        return "mdy_evidence"
    return "ambiguous"


def profile_slash_date_side_evidence(values: list[str]) -> dict[str, int]:
    """Count how many values fall into each `classify_slash_date_side_evidence` category."""
    counter: Counter[str] = Counter()
    for v in values:
        category = classify_slash_date_side_evidence(v)
        if category is not None:
            counter[category] += 1
    return dict(counter)


_TIME_COLON_RE = re.compile(r"^\d{1,2}:\d{2}$")


def classify_time_string(value: str) -> str:
    """Classify one raw time string: "empty" / "clock_string" / "excel_fraction" / "other"."""
    stripped = value.strip()
    if not stripped:
        return "empty"
    if _TIME_COLON_RE.match(stripped):
        return "clock_string"
    try:
        as_float = float(stripped)
    except ValueError:
        return "other"
    if 0.0 <= as_float < 1.0:
        return "excel_fraction"
    return "other"


def profile_time_column_formats(values: list[str]) -> dict[str, int]:
    counter: Counter[str] = Counter(classify_time_string(v) for v in values)
    return dict(counter)


# --------------------------------------------------------------------------
# Nested JSON structure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JsonStructureProfile:
    root_type: str
    record_count: int | None
    keys: tuple[str, ...]
    array_valued_keys: tuple[str, ...]
    nested_object_keys: tuple[str, ...]
    max_depth: int


def _max_depth(value: Any, current: int = 0) -> int:
    if isinstance(value, dict) and value:
        return max(_max_depth(v, current + 1) for v in value.values())
    if isinstance(value, list) and value:
        return max(_max_depth(v, current) for v in value)
    return current


def profile_json_structure(data: Any) -> JsonStructureProfile:
    """Describe a parsed JSON document's shape: is it a list of records? What keys/nesting?"""
    if isinstance(data, list):
        record_count = len(data)
        sample = data[0] if data else {}
    elif isinstance(data, dict):
        record_count = None
        sample = data
    else:
        return JsonStructureProfile(
            root_type=type(data).__name__,
            record_count=None,
            keys=(),
            array_valued_keys=(),
            nested_object_keys=(),
            max_depth=0,
        )

    keys = tuple(sample.keys()) if isinstance(sample, dict) else ()
    array_keys = tuple(k for k in keys if isinstance(sample.get(k), list))
    nested_object_keys = tuple(k for k in keys if isinstance(sample.get(k), dict))

    return JsonStructureProfile(
        root_type=type(data).__name__,
        record_count=record_count,
        keys=keys,
        array_valued_keys=array_keys,
        nested_object_keys=nested_object_keys,
        max_depth=_max_depth(data),
    )


def load_json_file(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Full-file profile (CSV)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FileProfile:
    metadata: FileMetadata
    row_count: int
    column_count: int
    columns: tuple[ColumnProfile, ...]
    duplicates: DuplicateProfile
    null_correlated_pairs: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def profile_csv_file(path: Path, candidate_key_columns: list[str] | None = None) -> FileProfile:
    """Full profile of one CSV file: metadata + every column + duplicates + null-correlation."""
    metadata = profile_file_metadata(path)
    df = pl.read_csv(
        path,
        encoding="utf8-lossy",
        infer_schema_length=0,  # read everything as string -- profiling should not guess types
        null_values=["", "NA", "N/A", "-"],
    )

    columns = tuple(profile_column(df, col) for col in df.columns)
    duplicates = profile_duplicates(df, candidate_key_columns)
    null_pairs = tuple(find_null_correlated_column_groups(df))

    return FileProfile(
        metadata=metadata,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        duplicates=duplicates,
        null_correlated_pairs=null_pairs,
    )


# --------------------------------------------------------------------------
# Duplicate-group deep-dive (P5-style investigation: is a column consistent
# within a group of rows sharing a key, and what happens if the group is
# collapsed to one row?)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupAgreementResult:
    """For groups of rows sharing `group_keys`, does `check_column` hold one value per group?"""

    column: str
    groups_checked: int
    agree_count: int
    conflict_count: int

    @property
    def agree_pct(self) -> float:
        return (self.agree_count / self.groups_checked * 100) if self.groups_checked else 0.0


def check_within_group_agreement(
    df: pl.DataFrame, group_keys: list[str], check_column: str
) -> GroupAgreementResult:
    """Among groups (by `group_keys`) with more than one row, does `check_column` agree?

    Rows where any `group_keys` column is null are excluded first: polars'
    `group_by` treats all nulls as one group, so e.g. grouping by ACC_CODE
    without filtering would silently lump every row with a missing
    ACC_CODE into a single giant "duplicate group" -- almost certainly a
    conflict, and not a meaningful one.

    A null value in `check_column` counts as distinct from a populated one
    (so a group where one row has a value and another is null on
    `check_column` is reported as a conflict, not an agreement) -- this
    deliberately does not distinguish "genuinely different values" from
    "one row is just less complete"; use two columns' worth of
    investigation for that nuance.
    """
    keyed = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in group_keys]))
    groups = keyed.group_by(group_keys).agg(
        pl.col(check_column).n_unique().alias("n_unique"), pl.len().alias("n_rows")
    )
    multi_row = groups.filter(pl.col("n_rows") > 1)
    agree = int(multi_row.filter(pl.col("n_unique") == 1).height)
    conflict = int(multi_row.filter(pl.col("n_unique") > 1).height)
    return GroupAgreementResult(
        column=check_column,
        groups_checked=multi_row.height,
        agree_count=agree,
        conflict_count=conflict,
    )


@dataclass(frozen=True)
class DedupImpact:
    """Row count and column sums before/after collapsing duplicate `group_key` groups to 1 row."""

    group_key: str
    rows_before: int
    rows_after: int
    sums_before: dict[str, float]
    sums_after: dict[str, float]


def compute_dedup_impact(df: pl.DataFrame, group_key: str, sum_columns: list[str]) -> DedupImpact:
    """Compare row counts and column sums before/after keeping 1 row per non-null `group_key`.

    Rows where `group_key` is null are never deduplicated (there is nothing
    to group them by) and pass through unchanged into the "after" total.
    Which row is kept ("first", by input order) is arbitrary and only valid
    as a total when `sum_columns` are known to agree within every group --
    this does not choose a tie-break rule for columns that can disagree.
    """
    keyed = df.filter(pl.col(group_key).is_not_null())
    unkeyed = df.filter(pl.col(group_key).is_null())
    deduped_keyed = keyed.unique(subset=[group_key], keep="first", maintain_order=True)
    after = pl.concat([deduped_keyed, unkeyed])

    def sums(frame: pl.DataFrame) -> dict[str, float]:
        return {
            col: float(frame.get_column(col).cast(pl.Float64, strict=False).fill_null(0).sum())
            for col in sum_columns
        }

    return DedupImpact(
        group_key=group_key,
        rows_before=len(df),
        rows_after=len(after),
        sums_before=sums(df),
        sums_after=sums(after),
    )


# --------------------------------------------------------------------------
# Column-relationship hypothesis checks (P3-style: is column B really a
# rename of column A, or does it serve some other role?)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueMatchWhenBothPopulated:
    """When two columns are both populated, how often does one hold a given value?"""

    rows_both_populated: int
    matching_value_count: int

    @property
    def matching_pct(self) -> float:
        return (
            (self.matching_value_count / self.rows_both_populated * 100)
            if self.rows_both_populated
            else 0.0
        )


def check_value_when_both_populated(
    df: pl.DataFrame, primary: str, secondary: str, expected_value: str
) -> ValueMatchWhenBothPopulated:
    """Among rows where both `primary` and `secondary` are populated, how often does
    `primary` equal `expected_value`?

    Built to test the hypothesis "`secondary` is a free-text detail field
    used only when `primary` is set to some catch-all value" -- if that's
    true, `expected_value` (the catch-all, e.g. "other") should dominate.
    """
    both = df.filter(pl.col(primary).is_not_null() & pl.col(secondary).is_not_null())
    if both.height == 0:
        # An empty frame can have an inferred dtype (e.g. Null) that doesn't
        # support `==` against a string -- nothing to match either way.
        return ValueMatchWhenBothPopulated(rows_both_populated=0, matching_value_count=0)
    matching = int((both.get_column(primary) == expected_value).sum())
    return ValueMatchWhenBothPopulated(
        rows_both_populated=both.height, matching_value_count=matching
    )


def population_rate_by_group(df: pl.DataFrame, group_col: str, target_col: str) -> pl.DataFrame:
    """For each value of `group_col`, how many rows total, and how many have `target_col` populated?

    Used to check whether a column is populated consistently across time
    (e.g. by year) or only within a specific slice -- the signal for
    "this isn't an old-name/new-name rename, it's new in one period only."
    """
    return (
        df.group_by(group_col)
        .agg(
            pl.len().alias("total_rows"),
            pl.col(target_col).is_not_null().sum().alias("populated_rows"),
        )
        .sort(group_col)
    )


# --------------------------------------------------------------------------
# Conflicting-group extraction (which duplicate groups disagree on a
# column, and what are the actual conflicting values -- for direct
# inspection rather than a single agree/conflict count)
# --------------------------------------------------------------------------


def find_conflicting_groups(
    df: pl.DataFrame, group_keys: list[str], target_column: str
) -> list[tuple[tuple[str, ...], list[str]]]:
    """For groups (by `group_keys`, excluding null keys) with >1 distinct non-null
    `target_column` value, return each group's key and its sorted distinct values.

    Unlike `check_within_group_agreement` (which only counts), this returns
    the actual conflicting values so they can be inspected directly -- e.g.
    to check whether "conflicting" values are genuinely different or just
    differently-worded synonyms.
    """
    keyed = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in group_keys]))
    conflicts: list[tuple[tuple[str, ...], list[str]]] = []
    for key_values, group in keyed.group_by(group_keys, maintain_order=True):
        non_null = [v for v in group.get_column(target_column).to_list() if v is not None]
        distinct = sorted(set(non_null))
        if len(distinct) > 1:
            conflicts.append((tuple(str(k) for k in key_values), distinct))
    return conflicts


def sample_deterministic(items: list[T], n: int, seed: int) -> list[T]:
    """Deterministically sample up to `n` items (CLAUDE.md N9: reproducible, seeded)."""
    rng = random.Random(seed)
    return rng.sample(items, min(n, len(items)))


# Diagnostic-only heuristic for measuring how much of a "conflict" rate is
# actually taxonomy/formatting noise, NOT a Phase 2 cleaning rule. Rules
# below were derived by inspecting a real sample of conflicting
# บริเวณที่เกิดเหตุ pairs (CLAUDE.md P5/P3 investigation): they collapse
# known synonym pairs and formatting variants so genuinely different
# values can be told apart from differently-worded identical ones.
_LOCATION_SYNONYM_SUBSTITUTIONS = (
    ("ไม่มีความลาดชัน", "ที่ราบ"),  # "no slope" == "flat/level"
    ("พานิชย์", "พาณิชย์"),  # spelling variant of "commercial"
    ("หรือ", "/"),  # "X หรือ Y" (X or Y) ~= "X/Y"
)


def normalize_for_synonym_check(value: str) -> str:
    """Collapse known synonym/formatting variants for a same-concept comparison.

    NOT a canonical cleaning function -- purely diagnostic, to measure
    whether raw "conflicts" between two values are the same concept
    differently phrased, before deciding whether a tie-break rule or a
    taxonomy harmonization step is the right fix.
    """
    normalized = re.sub(r"\s+", "", value.strip())
    for old, new in _LOCATION_SYNONYM_SUBSTITUTIONS:
        normalized = normalized.replace(old, new)
    normalized = normalized.split("/")[0]  # drop an elaboration/alt-taxonomy suffix
    normalized = normalized.replace("ทางเชื่อมเข้า", "")
    return normalized


def count_resolved_by_normalization(
    conflicts: list[tuple[tuple[str, ...], list[str]]],
) -> tuple[int, int]:
    """Among conflicting groups, how many collapse to one value under `normalize_for_synonym_check`?

    Returns (resolved_count, still_conflicting_count).
    """
    resolved = 0
    still_conflicting = 0
    for _, values in conflicts:
        normalized_values = {normalize_for_synonym_check(v) for v in values}
        if len(normalized_values) == 1:
            resolved += 1
        else:
            still_conflicting += 1
    return resolved, still_conflicting


# --------------------------------------------------------------------------
# P1/P5 duplicate-date investigation: telling a genuine parser bug apart
# from a data-entry error (transposed day/month digits) within known
# duplicate groups (P5). Diagnostic only -- `parse_slash_date` is not the
# Phase 2 cleaning parser, it exists to test candidate dates during
# profiling.
# --------------------------------------------------------------------------


def parse_slash_date(value: str, month_first: bool) -> date | None:
    """Parse a `d/m/y` (or `m/d/y`, per `month_first`) string into a `date`.

    Returns None for anything that is not a plain two-slash date, or that
    doesn't form a valid calendar date (e.g. day=31 in a 30-day month).
    Diagnostic only: this does not attempt to detect the format itself
    (see `classify_date_string` / P1's per-file detection) -- the caller
    must already know which side is the month.
    """
    match = _SLASH_DATE_RE.match(value.strip())
    if not match:
        return None
    first, second = int(match.group(1)), int(match.group(2))
    year = int(match.group(3))
    if year < 100:
        year += 2000
    month, day = (first, second) if month_first else (second, first)
    try:
        return date(year, month, day)
    except ValueError:
        return None


@dataclass(frozen=True)
class DuplicateDateStringConsistency:
    """Safety net 4a: duplicate rows sharing an IDENTICAL raw date string must parse identically.

    Deliberately scoped narrow. Rows in a duplicate group (by `group_key`)
    can legitimately carry DIFFERENT raw date strings that are digit
    transpositions of each other (a data-entry error -- see
    `find_transposed_date_string_pairs`); those are expected to parse to two
    different dates and are not a parser bug. This check only looks at
    groups where the raw string is identical across every row: for those, a
    deterministic parser must always produce one date, so any mismatch here
    is a real parser or format-detection bug, not a keying error.
    """

    groups_checked: int
    mismatched_groups: int
    mismatched_group_keys: list[Any]


def check_duplicate_date_string_consistency(
    df: pl.DataFrame, group_key: str, raw_date_column: str, parsed_date_column: str
) -> DuplicateDateStringConsistency:
    """Among duplicate groups (by `group_key`) with one raw date string, does the parsed date agree?

    Returns the count of same-raw-string groups checked and any that
    disagree after parsing.
    """
    keyed = df.filter(pl.col(group_key).is_not_null())
    groups = keyed.group_by(group_key).agg(
        pl.col(raw_date_column).n_unique().alias("n_raw"),
        pl.col(parsed_date_column).n_unique().alias("n_parsed"),
        pl.len().alias("n_rows"),
    )
    identical_raw = groups.filter((pl.col("n_rows") > 1) & (pl.col("n_raw") == 1))
    mismatched = identical_raw.filter(pl.col("n_parsed") > 1)
    return DuplicateDateStringConsistency(
        groups_checked=int(identical_raw.height),
        mismatched_groups=int(mismatched.height),
        mismatched_group_keys=mismatched[group_key].to_list(),
    )


def find_transposed_date_string_pairs(
    df: pl.DataFrame, group_key: str, raw_date_column: str
) -> list[Any]:
    """Duplicate groups (by `group_key`) whose raw date strings disagree in exactly two
    values that are digit transpositions of each other (`a/b/y` vs. `b/a/y`, same year).

    This is a data-entry problem, not a date-format problem: once a file's
    format is fixed (P1), transposed day/month digits are mathematically
    guaranteed to parse to two different dates. That is why they must NOT be
    fed into `check_duplicate_date_string_consistency` (which would flag all
    of them as parser failures) -- they belong in a separate
    `date_ambiguous_pairs` report instead, resolved by
    `discriminate_transposed_date_pairs`.
    """
    keyed = df.filter(pl.col(group_key).is_not_null())
    groups = keyed.group_by(group_key).agg(
        pl.col(raw_date_column).n_unique().alias("n_raw"), pl.len().alias("n_rows")
    )
    candidate_keys = groups.filter((pl.col("n_rows") > 1) & (pl.col("n_raw") == 2))[
        group_key
    ].to_list()

    transposed_keys = []
    for key in candidate_keys:
        raws = keyed.filter(pl.col(group_key) == key)[raw_date_column].unique().to_list()
        if len(raws) != 2:
            continue
        match_a = _SLASH_DATE_RE.match(raws[0].strip())
        match_b = _SLASH_DATE_RE.match(raws[1].strip())
        if not match_a or not match_b:
            continue
        a1, b1, y1 = match_a.group(1), match_a.group(2), match_a.group(3)
        a2, b2, y2 = match_b.group(1), match_b.group(2), match_b.group(3)
        if a1 == b2 and b1 == a2 and y1 == y2:
            transposed_keys.append(key)
    return transposed_keys


@dataclass(frozen=True)
class TransposedDatePairResolution:
    """One transposed-date duplicate group and which candidate date(s) survive safety net #2."""

    group_key: Any
    candidate_dates: list[date]
    valid_dates: list[date]

    @property
    def outcome(self) -> str:
        """Returns 'resolved_unique', 'both_valid' (undecidable), or 'neither_valid' (flag)."""
        if len(self.valid_dates) == 1:
            return "resolved_unique"
        if len(self.valid_dates) == 2:
            return "both_valid"
        return "neither_valid"


@dataclass(frozen=True)
class TransposedDatePairDiscrimination:
    """Resolution outcomes for a set of transposed-date duplicate pairs.

    See `discriminate_transposed_date_pairs`.
    """

    resolutions: list[TransposedDatePairResolution]

    @property
    def resolved_unique(self) -> int:
        return sum(1 for r in self.resolutions if r.outcome == "resolved_unique")

    @property
    def both_valid(self) -> int:
        return sum(1 for r in self.resolutions if r.outcome == "both_valid")

    @property
    def neither_valid(self) -> int:
        return sum(1 for r in self.resolutions if r.outcome == "neither_valid")


def discriminate_transposed_date_pairs(
    df: pl.DataFrame,
    group_key: str,
    group_keys_to_resolve: list[Any],
    raw_date_column: str,
    raw_report_date_column: str,
    month_first: bool,
    max_report_lag_days: int | dict[str, int] = 90,
    threshold_group_column: str | None = None,
) -> TransposedDatePairDiscrimination:
    """Resolve transposed-date duplicate groups using safety net #2 as a discriminator.

    For each group in `group_keys_to_resolve`, each distinct raw date string
    is one candidate date. A candidate is valid only if EVERY row carrying
    that raw string satisfies both safety-net-#2 conditions against its own
    report date (`accident_date <= report_date`, gap <= threshold) -- so a
    3-row group (one candidate date repeated across two rows) requires both
    of those rows to agree, not just one.

    `max_report_lag_days` is normally a single global threshold. Pass a dict
    instead (mapping a value of `threshold_group_column`, e.g. agency, to
    its own threshold) to use a subgroup-specific bound -- appropriate when
    the legitimate lag distribution varies sharply by subgroup (CLAUDE.md:
    agency p99 ranges from 235 to 1,219 days). When a dict is given,
    `threshold_group_column` is required and every row's value in that
    column must be a key in the dict.

    If exactly one candidate is valid, that is the resolved true date. If
    both are valid, the pair is genuinely undecidable from this evidence. If
    neither is valid, something else is wrong with the row and it must be
    flagged rather than resolved either way.
    """
    if isinstance(max_report_lag_days, dict) and threshold_group_column is None:
        raise ValueError("threshold_group_column is required when max_report_lag_days is a dict")

    resolutions = []
    for key in group_keys_to_resolve:
        sub = df.filter(pl.col(group_key) == key)
        candidate_valid: dict[date, bool] = {}
        for raw_value in sub[raw_date_column].unique().to_list():
            rows = sub.filter(pl.col(raw_date_column) == raw_value)
            candidate = parse_slash_date(raw_value, month_first=month_first)
            if candidate is None:
                continue
            valid = True
            threshold_values: list[Any] = (
                rows[threshold_group_column].to_list()
                if threshold_group_column is not None
                else [None] * rows.height
            )
            for accident_raw, report_raw, threshold_group_value in zip(
                rows[raw_date_column].to_list(),
                rows[raw_report_date_column].to_list(),
                threshold_values,
                strict=True,
            ):
                accident_dt = parse_slash_date(accident_raw, month_first=month_first)
                report_dt = parse_slash_date(report_raw, month_first=month_first)
                if accident_dt is None or report_dt is None:
                    valid = False
                    break
                gap = (report_dt - accident_dt).days
                threshold = (
                    max_report_lag_days[threshold_group_value]
                    if isinstance(max_report_lag_days, dict)
                    else max_report_lag_days
                )
                if gap < 0 or gap > threshold:
                    valid = False
                    break
            candidate_valid[candidate] = valid
        valid_dates = [d for d, ok in candidate_valid.items() if ok]
        resolutions.append(
            TransposedDatePairResolution(
                group_key=key,
                candidate_dates=list(candidate_valid.keys()),
                valid_dates=valid_dates,
            )
        )
    return TransposedDatePairDiscrimination(resolutions=resolutions)


# --------------------------------------------------------------------------
# Report-lag distribution (report_date - accident_date): validates the
# 90-day threshold used by safety net #2 and `discriminate_transposed_date_pairs`
# against real data, rather than trusting the value it was designed with.
# --------------------------------------------------------------------------

#: Per-file date format, as established in P1 (CLAUDE.md) by counting
#: unambiguous (a component > 12) rows -- "dmy", "mdy", or "excel_serial".
#: This is the single settled source of truth for per-file format, reused
#: here for lag profiling and by the Task 5 data contracts to assert the
#: same fact.
SOURCE_FILE_DATE_FORMAT: dict[str, str] = {
    "accident2019.csv": "dmy",
    "accident2020.csv": "mdy",
    "accident2021.csv": "mdy",
    "accident2022.csv": "mdy",
    "accident2023.csv": "mdy",
    "accident2024.csv": "excel_serial",
    "accident2025.csv": "excel_serial",
    "accident2026.csv": "dmy",
}

EXCEL_SERIAL_ORIGIN = date(1899, 12, 30)


def parse_excel_serial_date(value: str) -> date | None:
    """Parse an Excel serial-date string (days since 1899-12-30) into a `date`."""
    try:
        serial = float(value.strip())
    except ValueError:
        return None
    if serial <= 0:
        return None
    return EXCEL_SERIAL_ORIGIN + timedelta(days=int(serial))


def parse_accident_date(value: str, date_format: str) -> date | None:
    """Parse a raw date string using a fixed per-file format label.

    `date_format` must be "dmy", "mdy", or "excel_serial" (see
    `SOURCE_FILE_DATE_FORMAT`). Returns None for anything that doesn't parse
    under that specific format rather than falling back to another one --
    per-file format is fixed by design (P1), so a value that doesn't fit is
    a row-level anomaly to flag elsewhere, not something to guess around
    here.
    """
    if date_format == "excel_serial":
        return parse_excel_serial_date(value)
    if date_format == "dmy":
        return parse_slash_date(value, month_first=False)
    if date_format == "mdy":
        return parse_slash_date(value, month_first=True)
    raise ValueError(f"unknown date_format: {date_format!r}")


def compute_report_lag_days(
    df: pl.DataFrame,
    source_file_column: str,
    accident_date_column: str,
    report_date_column: str,
    file_date_formats: dict[str, str],
) -> pl.DataFrame:
    """Parse both date columns per `source_file`'s known format and add a `lag_days` column.

    Returns one row per input row where BOTH dates parsed successfully under
    that file's format (`df.height` before minus after gives the
    unparseable count). `lag_days` is `report_date - accident_date` in days
    and can be negative (report predating the accident -- a different
    anomaly than "long lag", see `summarize_report_lag`).
    """
    frames = []
    for source_file, date_format in file_date_formats.items():
        sub = df.filter(pl.col(source_file_column) == source_file)
        if sub.height == 0:
            continue
        lag_days: list[int | None] = []
        for accident_raw, report_raw in zip(
            sub[accident_date_column].to_list(), sub[report_date_column].to_list(), strict=True
        ):
            accident_dt = (
                parse_accident_date(accident_raw, date_format) if accident_raw is not None else None
            )
            report_dt = (
                parse_accident_date(report_raw, date_format) if report_raw is not None else None
            )
            if accident_dt is None or report_dt is None:
                lag_days.append(None)
            else:
                lag_days.append((report_dt - accident_dt).days)
        frames.append(sub.with_columns(pl.Series("lag_days", lag_days, dtype=pl.Int64)))
    if not frames:
        return df.head(0).with_columns(pl.lit(None, dtype=pl.Int64).alias("lag_days"))
    return pl.concat(frames).filter(pl.col("lag_days").is_not_null())


@dataclass(frozen=True)
class ReportLagProfile:
    """Distribution of `report_date - accident_date` (days) across parseable rows.

    Percentiles/median/max are computed on non-negative lags only -- a
    negative lag (report predates the accident) is a different failure mode
    (a date is simply wrong), not evidence about how long legitimate
    reporting takes, and would distort the "how high should the upper bound
    be" question this profile exists to answer.
    """

    rows_parsed: int
    rows_negative_lag: int
    median_days: float
    p95_days: float
    p99_days: float
    max_days: float
    count_over_90_days: int


def count_rows_over_lag_threshold(
    df_with_lag: pl.DataFrame, threshold_days: int, lag_column: str = "lag_days"
) -> int:
    """How many non-negative-lag rows exceed `threshold_days`? Used to test candidate thresholds."""
    return int(
        df_with_lag.filter((pl.col(lag_column) >= 0) & (pl.col(lag_column) > threshold_days)).height
    )


def count_rows_over_group_thresholds(
    df_with_lag: pl.DataFrame,
    group_column: str,
    thresholds_by_group: dict[str, int],
    lag_column: str = "lag_days",
) -> int:
    """Like `count_rows_over_lag_threshold`, but each `group_column` value uses its own threshold.

    Every group present in `df_with_lag` must be a key in `thresholds_by_group`
    -- this raises `KeyError` on an unmapped group rather than silently
    skipping or falling back to a default, so a new/unexpected group value
    can't be quietly excluded from the count.
    """
    total = 0
    for group_value in df_with_lag[group_column].unique().to_list():
        threshold = thresholds_by_group[group_value]
        subset = df_with_lag.filter(pl.col(group_column) == group_value)
        total += count_rows_over_lag_threshold(subset, threshold, lag_column)
    return total


def summarize_report_lag(
    df_with_lag: pl.DataFrame, lag_column: str = "lag_days"
) -> ReportLagProfile:
    """Summarize a `lag_days` column produced by `compute_report_lag_days`."""
    negative = df_with_lag.filter(pl.col(lag_column) < 0)
    non_negative = df_with_lag.filter(pl.col(lag_column) >= 0)
    lags = non_negative[lag_column]
    if non_negative.height == 0:
        return ReportLagProfile(
            rows_parsed=df_with_lag.height,
            rows_negative_lag=negative.height,
            median_days=0.0,
            p95_days=0.0,
            p99_days=0.0,
            max_days=0.0,
            count_over_90_days=0,
        )
    return ReportLagProfile(
        rows_parsed=df_with_lag.height,
        rows_negative_lag=negative.height,
        median_days=float(lags.median()),  # type: ignore[arg-type]
        p95_days=float(lags.quantile(0.95, interpolation="linear")),  # type: ignore[arg-type]
        p99_days=float(lags.quantile(0.99, interpolation="linear")),  # type: ignore[arg-type]
        max_days=float(lags.max()),  # type: ignore[arg-type]
        count_over_90_days=int(non_negative.filter(pl.col(lag_column) > 90).height),
    )


def report_lag_quantiles_by_group(
    df_with_lag: pl.DataFrame, group_col: str, lag_column: str = "lag_days"
) -> pl.DataFrame:
    """Median/p95/p99/max lag per `group_col` value (non-negative lags only), plus row count.

    Used to check whether the lag distribution differs meaningfully by
    agency or by year -- a single global threshold is only appropriate if
    it doesn't.
    """
    non_negative = df_with_lag.filter(pl.col(lag_column) >= 0)
    return (
        non_negative.group_by(group_col)
        .agg(
            pl.len().alias("n"),
            pl.col(lag_column).median().alias("median_days"),
            pl.col(lag_column).quantile(0.95).alias("p95_days"),
            pl.col(lag_column).quantile(0.99).alias("p99_days"),
            pl.col(lag_column).max().alias("max_days"),
        )
        .sort(group_col)
    )
