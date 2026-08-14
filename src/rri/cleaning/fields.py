"""Phase 2 Step 5: remaining field-level fixes (CLAUDE.md P2, P7).

Three independent, unrelated fixes bundled in one step because each is a
single-column, no-imputation cleanup: coordinates, time format, and an
injury-breakdown-availability flag. None of these invent a value the source
data didn't provide -- N7-style, unknowns become explicit unknown markers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import polars as pl

from rri.utils.logging import get_logger
from rri.utils.profiling import classify_time_string

log = get_logger(__name__)

LAT_BOUNDS = (5.0, 21.0)
LON_BOUNDS = (97.0, 106.0)


@dataclass(frozen=True)
class CoordinateCleanLog:
    rows_total: int
    rows_lat_out_of_bounds: int
    rows_lon_out_of_bounds: int
    rows_is_geo_valid_false: int


def clean_coordinates(
    df: pl.DataFrame,
    lat_column: str = "LATITUDE",
    lon_column: str = "LONGITUDE",
    lat_bounds: tuple[float, float] = LAT_BOUNDS,
    lon_bounds: tuple[float, float] = LON_BOUNDS,
) -> tuple[pl.DataFrame, CoordinateCleanLog]:
    """Null out-of-bounds coordinates (CLAUDE.md P7) and add `is_geo_valid`.

    Rows are never dropped. A coordinate already null stays null. A
    coordinate present but outside Thailand's bounding box is nulled (the
    original bad value is not recoverable, so there is nothing to guess);
    `is_geo_valid` is false whenever either coordinate is missing or was
    out of bounds, true only when both are present and in range.
    """
    lat = pl.col(lat_column).cast(pl.Float64, strict=False)
    lon = pl.col(lon_column).cast(pl.Float64, strict=False)

    lat_in_bounds = lat.is_not_null() & lat.is_between(*lat_bounds)
    lon_in_bounds = lon.is_not_null() & lon.is_between(*lon_bounds)

    stats = df.select(
        rows_lat_out_of_bounds=(lat.is_not_null() & ~lat_in_bounds).sum(),
        rows_lon_out_of_bounds=(lon.is_not_null() & ~lon_in_bounds).sum(),
        rows_is_geo_valid_false=(~(lat_in_bounds & lon_in_bounds)).sum(),
    ).row(0, named=True)

    cleaned = df.with_columns(
        pl.when(lat_in_bounds).then(lat).otherwise(None).alias(lat_column),
        pl.when(lon_in_bounds).then(lon).otherwise(None).alias(lon_column),
        (lat_in_bounds & lon_in_bounds).alias("is_geo_valid"),
    )

    coord_log = CoordinateCleanLog(
        rows_total=df.height,
        rows_lat_out_of_bounds=stats["rows_lat_out_of_bounds"],
        rows_lon_out_of_bounds=stats["rows_lon_out_of_bounds"],
        rows_is_geo_valid_false=stats["rows_is_geo_valid_false"],
    )
    log.info(
        "coordinates_cleaned",
        rows_total=coord_log.rows_total,
        rows_lat_out_of_bounds=coord_log.rows_lat_out_of_bounds,
        rows_lon_out_of_bounds=coord_log.rows_lon_out_of_bounds,
        rows_is_geo_valid_false=coord_log.rows_is_geo_valid_false,
    )
    return cleaned, coord_log


def parse_time_value(raw: str | None) -> time | None:
    """Parse one raw time string into a `datetime.time`, or None if unparseable.

    Two coexisting formats (CLAUDE.md P2): `"21:10"` clock strings, and
    Excel day-fraction floats (`"0.375"` == 09:00). Uses
    `classify_time_string` (Phase 1) to tell them apart rather than
    re-deriving the same detection logic.
    """
    if raw is None:
        return None
    category = classify_time_string(raw)
    stripped = raw.strip()
    if category == "clock_string":
        hour_str, minute_str = stripped.split(":")
        hour, minute = int(hour_str), int(minute_str)
        if hour > 23 or minute > 59:
            return None
        return time(hour=hour, minute=minute)
    if category == "excel_fraction":
        total_minutes = round(float(stripped) * 24 * 60)
        hour, minute = divmod(total_minutes, 60)
        if hour > 23:
            return None
        return time(hour=hour, minute=minute)
    return None


@dataclass(frozen=True)
class TimeCleanLog:
    rows_total: int
    rows_parsed: int
    rows_unparseable: int
    rows_literal_midnight: int


def convert_times(
    df: pl.DataFrame,
    time_column: str = "เวลา",
    new_column: str = "accident_time",
    literal_midnight_flag_column: str = "time_is_literal_midnight",
) -> tuple[pl.DataFrame, TimeCleanLog]:
    """Add a parsed `accident_time` column plus a literal-midnight flag (CLAUDE.md P2).

    The "0:00" values were investigated against `เวลาที่รายงาน` (report
    time) before this function was written: across the post-dedup data,
    all 215 rows with a literal "0:00" accident time have a *non-null*,
    normally-distributed report time (no clustering at "0:00", no nulls)
    -- there is no evidence these are a shared "time unknown" placeholder
    convention. Absent that evidence, nulling them would be a guess in the
    other direction, so the parsed value is kept as real midnight and only
    flagged via `literal_midnight_flag_column` for downstream transparency
    (N7-style: expose the ambiguity, don't silently resolve it either way).
    """
    raw = df[time_column].cast(pl.Utf8)
    parsed = raw.map_elements(parse_time_value, return_dtype=pl.Time)
    is_literal_midnight = raw.str.strip_chars() == "0:00"

    stats = (
        pl.DataFrame({"parsed": parsed, "is_literal_midnight": is_literal_midnight})
        .select(
            rows_parsed=pl.col("parsed").is_not_null().sum(),
            rows_unparseable=pl.col("parsed").is_null().sum(),
            rows_literal_midnight=pl.col("is_literal_midnight").fill_null(False).sum(),
        )
        .row(0, named=True)
    )

    cleaned = df.with_columns(
        parsed.alias(new_column),
        is_literal_midnight.fill_null(False).alias(literal_midnight_flag_column),
    )

    time_log = TimeCleanLog(
        rows_total=df.height,
        rows_parsed=stats["rows_parsed"],
        rows_unparseable=stats["rows_unparseable"],
        rows_literal_midnight=stats["rows_literal_midnight"],
    )
    log.info(
        "times_converted",
        rows_total=time_log.rows_total,
        rows_parsed=time_log.rows_parsed,
        rows_unparseable=time_log.rows_unparseable,
        rows_literal_midnight=time_log.rows_literal_midnight,
    )
    return cleaned, time_log


@dataclass(frozen=True)
class InjuryBreakdownLog:
    rows_total: int
    rows_without_breakdown: int


def add_injury_breakdown_flag(
    df: pl.DataFrame,
    source_file_column: str = "source_file",
    no_breakdown_source_file: str = "accident2019.csv",
    new_column: str = "has_injury_breakdown",
) -> tuple[pl.DataFrame, InjuryBreakdownLog]:
    """Flag rows that never had a serious/minor injury split (CLAUDE.md P7).

    `accident2019.csv` never collected `ผู้บาดเจ็บสาหัส`/`ผู้บาดเจ็บเล็กน้อย`
    (100% null that year) -- only the total. This is a file-level
    column-omission, not row-level corruption, so the fix is a flag, not
    an imputed split. `total_injury_count`/`fatality_count` are untouched
    by this function and stay intact for every year.
    """
    has_breakdown = pl.col(source_file_column) != no_breakdown_source_file
    rows_without_breakdown = df.filter(~has_breakdown).height

    cleaned = df.with_columns(has_breakdown.alias(new_column))

    breakdown_log = InjuryBreakdownLog(
        rows_total=df.height,
        rows_without_breakdown=rows_without_breakdown,
    )
    log.info(
        "injury_breakdown_flagged",
        rows_total=breakdown_log.rows_total,
        rows_without_breakdown=breakdown_log.rows_without_breakdown,
    )
    return cleaned, breakdown_log
