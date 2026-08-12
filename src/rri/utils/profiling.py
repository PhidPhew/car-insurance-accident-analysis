"""Generic, source-agnostic profiling functions for raw data files.

Phase 1 is observe-only (CLAUDE.md Task 4): these functions describe what is
in a file -- size, encoding, per-column stats, duplicate counts, null
correlation between columns, mixed date/time formats -- they never clean,
fix, or drop anything. Cleaning logic belongs in ``src/rri/cleaning/``
(Phase 2).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

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
