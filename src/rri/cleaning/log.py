"""Cleaning-log accounting shared by every Phase 2 step (CLAUDE.md N4).

Every cleaning rule reports how many rows it touched, from what to what --
this module is just the shared record shape and CSV writer so each step
doesn't reinvent it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl


@dataclass(frozen=True)
class CleaningLogEntry:
    """One cleaning rule's effect, for the `reports/cleaning_log.csv` audit trail."""

    step: str
    rule: str
    rows_before: int
    rows_affected: int
    description: str


def cleaning_log_to_frame(entries: list[CleaningLogEntry]) -> pl.DataFrame:
    """Assemble log entries into a DataFrame in the order they were recorded."""
    if not entries:
        return pl.DataFrame(
            schema={
                "step": pl.Utf8,
                "rule": pl.Utf8,
                "rows_before": pl.Int64,
                "rows_affected": pl.Int64,
                "description": pl.Utf8,
            }
        )
    return pl.DataFrame([asdict(e) for e in entries])


def write_cleaning_log(entries: list[CleaningLogEntry], path: str) -> None:
    """Write the accumulated cleaning log to `path` as CSV."""
    cleaning_log_to_frame(entries).write_csv(path)
