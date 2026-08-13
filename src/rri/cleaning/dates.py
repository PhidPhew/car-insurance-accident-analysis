"""Phase 2 Step 3: resolve accident/report dates (CLAUDE.md P1) -- CHECKPOINT step.

Reuses Phase 1's diagnostic building blocks from `rri.utils.profiling`
(`SOURCE_FILE_DATE_FORMAT`, `parse_accident_date`,
`find_transposed_date_string_pairs`, `check_duplicate_date_string_consistency`)
rather than re-deriving any of them -- this module adds only what Phase 1
designed but did not implement: actually parsing every row's date, and the
per-agency reporting-lag *likelihood* discriminator for the 448 known
day/month-transposed duplicate pairs (replacing the old binary
pass/fail threshold test, which left too many pairs "undecidable").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from rri.utils.profiling import parse_accident_date, parse_slash_date

# --------------------------------------------------------------------------
# Per-file date parsing (safety nets #1-#3 depend on this)
# --------------------------------------------------------------------------


def parse_dates(
    df: pl.DataFrame,
    file_date_formats: dict[str, str],
    source_file_column: str = "source_file",
    accident_date_raw_column: str = "วันที่เกิดเหตุ",
    report_date_raw_column: str = "วันที่รายงาน",
    accident_date_column: str = "accident_date",
    report_date_column: str = "report_date",
) -> pl.DataFrame:
    """Parse accident/report dates per `source_file`'s known format (P1).

    A row whose `source_file` isn't a key in `file_date_formats` raises
    `KeyError` -- a new/unclassified source file must be added to the
    format table explicitly, not silently skipped. A raw value that fails
    to parse under its file's format becomes null (e.g. the accident2022
    `"//"` report-date placeholder, or a genuinely malformed string) --
    not an error, since accident_date parsing must never be blocked by an
    unrelated report_date problem.
    """
    source_files = df[source_file_column].to_list()
    accident_raw = df[accident_date_raw_column].to_list()
    report_raw = df[report_date_raw_column].to_list()

    accident_dates: list[date | None] = []
    report_dates: list[date | None] = []
    for source_file, a_raw, r_raw in zip(source_files, accident_raw, report_raw, strict=True):
        date_format = file_date_formats[source_file]
        accident_dates.append(
            parse_accident_date(a_raw, date_format) if a_raw is not None else None
        )
        report_dates.append(parse_accident_date(r_raw, date_format) if r_raw is not None else None)

    return df.with_columns(
        pl.Series(accident_date_column, accident_dates, dtype=pl.Date),
        pl.Series(report_date_column, report_dates, dtype=pl.Date),
    )


# --------------------------------------------------------------------------
# Safety net #1: parsed year must equal the existing year column
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class YearMatchResult:
    """Safety net #1: parsed `accident_date.year` vs. the existing year column."""

    rows_checked: int
    rows_unparseable: int
    rows_mismatched: int
    # each example: (ACC_CODE, parsed_year, year_column value)
    mismatch_examples: tuple[tuple[str | None, int, str], ...]


def check_year_match(
    df: pl.DataFrame,
    accident_date_column: str = "accident_date",
    year_column: str = "ปีที่เกิดเหตุ",
    acc_code_column: str = "ACC_CODE",
    max_examples: int = 20,
) -> YearMatchResult:
    """Every parsed accident year must equal the raw year column -- a mismatch is fatal (P1)."""
    parsed_dates = df[accident_date_column].to_list()
    year_strings = df[year_column].to_list()
    acc_codes = df[acc_code_column].to_list()

    unparseable = 0
    mismatches: list[tuple[str | None, int, str]] = []
    for acc_code, parsed_dt, year_str in zip(acc_codes, parsed_dates, year_strings, strict=True):
        if parsed_dt is None:
            unparseable += 1
            continue
        if str(parsed_dt.year) != str(year_str):
            mismatches.append((acc_code, parsed_dt.year, str(year_str)))

    return YearMatchResult(
        rows_checked=len(parsed_dates),
        rows_unparseable=unparseable,
        rows_mismatched=len(mismatches),
        mismatch_examples=tuple(mismatches[:max_examples]),
    )


# --------------------------------------------------------------------------
# Safety net #3: day-of-month histogram (a wrong parse spikes days 1-12)
# --------------------------------------------------------------------------


def day_of_month_histogram(
    df: pl.DataFrame, accident_date_column: str = "accident_date"
) -> dict[int, int]:
    """Count of accidents by day-of-month (1-31), for the safety-net-#3 shape check."""
    days = [d.day for d in df[accident_date_column].to_list() if d is not None]
    histogram = {day: 0 for day in range(1, 32)}
    for day in days:
        histogram[day] += 1
    return histogram


# --------------------------------------------------------------------------
# Per-agency likelihood discriminator for the 448 transposed-date pairs
# (replaces the old binary safety-net-#2 pass/fail test, CLAUDE.md P1)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LikelihoodResolution:
    """One transposed-date duplicate group's resolution by the likelihood discriminator."""

    group_key: str
    candidate_dates: tuple[date, ...]
    winner_date: date | None
    loser_dates: tuple[date, ...]
    margin_days: float | None
    outcome: str  # "resolved" or "ambiguous"


def resolve_transposed_pairs_by_likelihood(
    df: pl.DataFrame,
    group_key_column: str,
    group_keys: list[str],
    accident_date_raw_column: str,
    report_date_raw_column: str,
    agency_column: str,
    month_first: bool,
    agency_median_lag_days: dict[str, float],
    agency_max_lag_days: dict[str, float],
) -> list[LikelihoodResolution]:
    """Resolve each transposed-date group by comparing each candidate's lag to its agency median.

    For each of the (usually 2) distinct raw date strings in a group, every row carrying
    that string contributes one lag (`report_date - candidate_date`, using that row's own
    report date). A candidate is *viable* only if every one of its lags is sane
    (`0 <= lag <= agency_max_lag_days[agency]` -- a wide sanity ceiling, not the tight
    per-agency p99 used for Gate A). Among viable candidates, the one whose mean deviation
    from `agency_median_lag_days[agency]` is smallest wins -- always, not just when it
    clears some pass/fail bar, which is what lets this resolve nearly all pairs the old
    binary test left "both valid, undecided" on. `margin_days` records
    `deviation(loser) - deviation(winner)` as a confidence signal (larger = more decisive)
    when there were 2 viable candidates to compare; None when there was only one.

    `outcome` is "ambiguous" only when NO candidate has even one sane lag to compare --
    genuinely no evidence to decide from, not merely a close call.
    """
    resolutions = []
    for key in group_keys:
        sub = df.filter(pl.col(group_key_column) == key)
        candidate_scores: dict[date, float] = {}
        candidate_viable: dict[date, bool] = {}
        for raw_value in sub[accident_date_raw_column].unique().to_list():
            rows = sub.filter(pl.col(accident_date_raw_column) == raw_value)
            candidate = parse_slash_date(raw_value, month_first=month_first)
            if candidate is None:
                continue
            deviations = []
            viable = True
            for report_raw, agency in zip(
                rows[report_date_raw_column].to_list(), rows[agency_column].to_list(), strict=True
            ):
                report_dt = (
                    parse_slash_date(report_raw, month_first=month_first) if report_raw else None
                )
                if report_dt is None:
                    viable = False
                    break
                lag = (report_dt - candidate).days
                if lag < 0 or lag > agency_max_lag_days[agency]:
                    viable = False
                    break
                deviations.append(abs(lag - agency_median_lag_days[agency]))
            candidate_viable[candidate] = viable
            if viable and deviations:
                candidate_scores[candidate] = sum(deviations) / len(deviations)

        viable_candidates = sorted(candidate_scores.items(), key=lambda item: item[1])
        all_candidates = tuple(candidate_viable.keys())

        if not viable_candidates:
            resolutions.append(
                LikelihoodResolution(key, all_candidates, None, all_candidates, None, "ambiguous")
            )
        elif len(viable_candidates) == 1:
            winner_date = viable_candidates[0][0]
            losers = tuple(c for c in all_candidates if c != winner_date)
            resolutions.append(
                LikelihoodResolution(key, all_candidates, winner_date, losers, None, "resolved")
            )
        else:
            winner_date, winner_dev = viable_candidates[0]
            _, loser_dev = viable_candidates[1]
            losers = tuple(c for c in all_candidates if c != winner_date)
            resolutions.append(
                LikelihoodResolution(
                    key, all_candidates, winner_date, losers, loser_dev - winner_dev, "resolved"
                )
            )
    return resolutions


def apply_date_certainty(
    df: pl.DataFrame,
    resolutions: list[LikelihoodResolution],
    group_key_column: str,
    accident_date_column: str = "accident_date",
    certainty_column: str = "date_certainty",
) -> pl.DataFrame:
    """Add `date_certainty` ("certain" / "ambiguous"), default "certain" for every row.

    Only rows belonging to a resolved transposed-date group are touched: the row holding
    the winning candidate date is "certain" (Step 4's dedup keeps it); every row holding a
    losing or unresolved candidate is "ambiguous" (Step 4 prefers the "certain" row within
    the same duplicate group, so the ambiguous one is naturally dropped by dedup, not by a
    guess made here).
    """
    ambiguous_keys_and_dates: set[tuple[str, date]] = set()
    for resolution in resolutions:
        if resolution.outcome == "ambiguous":
            for candidate in resolution.candidate_dates:
                ambiguous_keys_and_dates.add((resolution.group_key, candidate))
        else:
            for loser in resolution.loser_dates:
                ambiguous_keys_and_dates.add((resolution.group_key, loser))

    group_keys = df[group_key_column].to_list()
    accident_dates = df[accident_date_column].to_list()
    certainty = [
        "ambiguous" if (key, dt) in ambiguous_keys_and_dates else "certain"
        for key, dt in zip(group_keys, accident_dates, strict=True)
    ]
    return df.with_columns(pl.Series(certainty_column, certainty, dtype=pl.Utf8))
