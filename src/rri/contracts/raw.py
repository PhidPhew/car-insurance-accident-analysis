"""Data contracts for the raw accidents source file (Phase 1 Task 5).

Contracts here encode facts PROVEN during Phase 1 profiling (CLAUDE.md
P1-P7), never facts assumed at design time -- see the ai_usage_log entry
on the 90-day report-lag threshold for why that distinction matters. Where
the raw file is known to violate an expectation (out-of-bounds
coordinates, mismatched injury totals), the check asserts the clean
expectation and is expected to fail on exactly the rows P7 already
quantified -- Phase 2 cleans, this module only observes and reports.

Deliberately NOT asserted: a single date format on `วันที่เกิดเหตุ` /
`วันที่รายงาน`. Three formats coexist by design across the 8 source files
(P1) -- the column-level schema only requires the column to be present and
non-null. Per-file format correctness is checked separately, at the
dataframe level, against the exact per-file assignment profiling already
proved (`check_source_file_date_formats`).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandera.polars as pa
import polars as pl

from rri.cleaning.province import load_canonical_province_names, match_province_values
from rri.utils.profiling import SOURCE_FILE_DATE_FORMAT, profile_slash_date_side_evidence

# --------------------------------------------------------------------------
# Reconciliation gates (CLAUDE.md S8) -- testable constants, not prose.
# --------------------------------------------------------------------------

GATE_1_ACCIDENT_COUNT = 161_307
GATE_1_FATALITY_COUNT = 21_814

GATE_2_ACCIDENT_COUNT = 159_605
GATE_2_FATALITY_COUNT = 21_497


@dataclass(frozen=True)
class GateCheckResult:
    label: str
    expected: int
    actual: int

    @property
    def passed(self) -> bool:
        return self.expected == self.actual


def _sum_numeric_column(df: pl.DataFrame, column: str) -> int:
    return int(df[column].cast(pl.Float64, strict=False).fill_null(0).sum())


def check_gate1_reconciliation(df: pl.DataFrame) -> list[GateCheckResult]:
    """Gate 1 (post-Extract, pre-dedup): proves nothing was lost reading the raw files in."""
    return [
        GateCheckResult("gate1_accident_count", GATE_1_ACCIDENT_COUNT, df.height),
        GateCheckResult(
            "gate1_fatality_count", GATE_1_FATALITY_COUNT, _sum_numeric_column(df, "ผู้เสียชีวิต")
        ),
    ]


def check_gate2_reconciliation(deduped_df: pl.DataFrame) -> list[GateCheckResult]:
    """Gate 2 (post-Load, post-dedup): proves dedup removed exactly what was intended.

    Not run against the raw extract by this module -- the raw file is
    pre-dedup by definition (see P5). Phase 2's warehouse-build step calls
    this once the ACC_CODE dedup rule (still undecided) is implemented.
    """
    return [
        GateCheckResult("gate2_accident_count", GATE_2_ACCIDENT_COUNT, deduped_df.height),
        GateCheckResult(
            "gate2_fatality_count",
            GATE_2_FATALITY_COUNT,
            _sum_numeric_column(deduped_df, "ผู้เสียชีวิต"),
        ),
    ]


# --------------------------------------------------------------------------
# Twin columns (CLAUDE.md P3) -- the 4 pairs PROVEN to be genuine renames
# (zero row-overlap, matching population rates). Deliberately excludes
# บริเวณที่เกิดเหตุ / บริเวณที่เกิดเหตุอื่น -- proven NOT a twin pair (a
# 2026-only parallel taxonomy instead, see CLAUDE.md P3).
# --------------------------------------------------------------------------

TWIN_COLUMN_PAIRS: tuple[tuple[str, str], ...] = (
    ("รถยนต์นั่งส่วนบุคคล", "รถยนต์นั่งส่วนบุคคล/รถยนต์นั่งสาธารณะ"),
    ("รถบรรทุกไม่เกิน10ล้อ", "รถบรรทุกมากกว่า6ล้อไม่เกิน10ล้อ"),
    ("รถบรรทุกมากกว่า10ล้อ", "รถบรรทุกมากกว่า10ล้อ(รถพ่วง)"),
    ("รถอื่นๆ", "อื่นๆ"),
)


@dataclass(frozen=True)
class TwinColumnCheckResult:
    old_column: str
    new_column: str
    overlap_row_count: int

    @property
    def passed(self) -> bool:
        return self.overlap_row_count == 0


def check_twin_column_pairs(df: pl.DataFrame) -> list[TwinColumnCheckResult]:
    """Each proven twin-column pair must never be populated on the same row (CLAUDE.md P3)."""
    results = []
    for old_col, new_col in TWIN_COLUMN_PAIRS:
        overlap = int((df[old_col].is_not_null() & df[new_col].is_not_null()).sum())
        results.append(TwinColumnCheckResult(old_col, new_col, overlap))
    return results


# --------------------------------------------------------------------------
# Per-file date format (CLAUDE.md P1) -- each file's opposite-format
# evidence must be exactly zero, exactly as verified when the per-file
# assumption was settled (see the "zero DMY evidence" finding).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DateFormatCheckResult:
    source_file: str
    expected_format: str
    counter_evidence_count: int

    @property
    def passed(self) -> bool:
        return self.counter_evidence_count == 0


def check_source_file_date_formats(
    df: pl.DataFrame, date_column: str = "วันที่เกิดเหตุ"
) -> list[DateFormatCheckResult]:
    """For each source_file, the OPPOSITE format's evidence must be exactly zero (CLAUDE.md P1)."""
    results = []
    for source_file, expected_format in SOURCE_FILE_DATE_FORMAT.items():
        values = df.filter(pl.col("source_file") == source_file)[date_column].drop_nulls().to_list()
        evidence = profile_slash_date_side_evidence(values)
        if expected_format == "dmy":
            counter = evidence.get("mdy_evidence", 0) + evidence.get("both_over_12", 0)
        elif expected_format == "mdy":
            counter = evidence.get("dmy_evidence", 0) + evidence.get("both_over_12", 0)
        else:  # excel_serial -- no slash-form date is expected at all
            counter = sum(evidence.values())
        results.append(DateFormatCheckResult(source_file, expected_format, counter))
    return results


# --------------------------------------------------------------------------
# Province normalization (CLAUDE.md P6, Task 3) -- 77 canonical provinces,
# zero orphan (unmatched, non-null) raw province values.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvinceContractResult:
    canonical_province_count: int
    orphan_values: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.canonical_province_count == 77 and len(self.orphan_values) == 0


def check_province_normalization(
    df: pl.DataFrame, province_column: str = "จังหวัด"
) -> ProvinceContractResult:
    match = match_province_values("accidents_raw", df[province_column].to_list())
    return ProvinceContractResult(
        canonical_province_count=len(load_canonical_province_names()),
        orphan_values=match.unmatched_values,
    )


# --------------------------------------------------------------------------
# Injury-total cross-column check (CLAUDE.md P7): serious + minor == total.
# Not a Column-level check (it spans 3 columns) -- reported as its own
# fact, flagged rather than silently reconciled per P7's explicit rule.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InjuryTotalCheckResult:
    mismatched_row_count: int
    total_rows: int

    @property
    def mismatch_rate(self) -> float:
        return self.mismatched_row_count / self.total_rows if self.total_rows else 0.0


def check_injury_totals(df: pl.DataFrame) -> InjuryTotalCheckResult:
    serious = df["ผู้บาดเจ็บสาหัส"].cast(pl.Float64, strict=False).fill_null(0)
    minor = df["ผู้บาดเจ็บเล็กน้อย"].cast(pl.Float64, strict=False).fill_null(0)
    total = df["รวมจำนวนผู้บาดเจ็บ"].cast(pl.Float64, strict=False).fill_null(0)
    mismatched = int(((serious + minor) != total).sum())
    return InjuryTotalCheckResult(mismatched_row_count=mismatched, total_rows=df.height)


# --------------------------------------------------------------------------
# Column-level schema. Deliberately loose on the date columns (see module
# docstring above); coordinate bounds are asserted because P7 already
# quantified exactly which/how many rows violate them (5 lat, 10 lon).
# --------------------------------------------------------------------------


def _in_thailand_lat(value: str | None) -> bool:
    if value is None:
        return True
    try:
        lat = float(value)
    except ValueError:
        return False
    return 5 <= lat <= 21


def _in_thailand_lon(value: str | None) -> bool:
    if value is None:
        return True
    try:
        lon = float(value)
    except ValueError:
        return False
    return 97 <= lon <= 106


ACCIDENTS_RAW_SCHEMA = pa.DataFrameSchema(
    {
        "ACC_CODE": pa.Column(str, nullable=True),  # 17.3% null by design (P5) -- not a PK
        "source_file": pa.Column(str, checks=pa.Check.isin(list(SOURCE_FILE_DATE_FORMAT))),
        # Present and non-null ONLY -- deliberately no format/shape check.
        # Three date formats coexist across files by design (P1).
        "วันที่เกิดเหตุ": pa.Column(str, nullable=False),
        "วันที่รายงาน": pa.Column(str, nullable=False),
        "จังหวัด": pa.Column(str, nullable=True),  # 1,969 null rows -- routed to Unknown Member (N7)
        "LATITUDE": pa.Column(
            str, nullable=True, checks=pa.Check(_in_thailand_lat, element_wise=True)
        ),
        "LONGITUDE": pa.Column(
            str, nullable=True, checks=pa.Check(_in_thailand_lon, element_wise=True)
        ),
    },
    strict=False,  # the raw file has 45 columns; this contract only pins down verified ones
)


# --------------------------------------------------------------------------
# Combined report
# --------------------------------------------------------------------------

_EMPTY_FAILURE_CASES = pl.DataFrame(
    {"failure_case": [], "column": [], "check": [], "index": []},
    schema={"failure_case": pl.Utf8, "column": pl.Utf8, "check": pl.Utf8, "index": pl.Int64},
)


@dataclass(frozen=True)
class AccidentsContractReport:
    """Everything a run of the accidents-raw contract found -- pass and fail alike."""

    column_schema_failure_cases: pl.DataFrame  # empty if the column schema passed outright
    gate1: list[GateCheckResult]
    twin_columns: list[TwinColumnCheckResult]
    date_formats: list[DateFormatCheckResult]
    province: ProvinceContractResult
    injury_totals: InjuryTotalCheckResult


def run_accidents_raw_contract(df: pl.DataFrame) -> AccidentsContractReport:
    """Run every accidents-raw check and return a full report, pass and fail alike.

    Never raises on a known/expected failure (e.g. the 10 out-of-bounds
    coordinate rows, P7) -- the caller decides what to do with the report;
    this module only observes.
    """
    try:
        ACCIDENTS_RAW_SCHEMA.validate(df, lazy=True)
        failure_cases = _EMPTY_FAILURE_CASES
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases.select("failure_case", "column", "check", "index")

    return AccidentsContractReport(
        column_schema_failure_cases=failure_cases,
        gate1=check_gate1_reconciliation(df),
        twin_columns=check_twin_column_pairs(df),
        date_formats=check_source_file_date_formats(df),
        province=check_province_normalization(df),
        injury_totals=check_injury_totals(df),
    )
