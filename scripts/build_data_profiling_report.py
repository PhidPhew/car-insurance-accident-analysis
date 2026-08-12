"""Build reports/data_profiling_report.html.

Run: ``uv run python scripts/build_data_profiling_report.py``

Phase 1 is observe-only (CLAUDE.md Task 4): this profiles every raw source
file -- size/encoding/delimiter, per-column stats, duplicates, mixed date
formats, null-correlated column pairs, JSON structure -- to confirm the 17
documented data-quality problems (CLAUDE.md §6) are real and match the
recorded numbers. It never cleans or fixes anything; that is Phase 2.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from rri.utils.logging import configure_logging, get_logger  # noqa: E402
from rri.utils.profiling import (  # noqa: E402
    ColumnProfile,
    FileProfile,
    JsonStructureProfile,
    load_json_file,
    profile_csv_file,
    profile_date_column_formats,
    profile_json_structure,
    profile_time_column_formats,
)

log = get_logger(__name__)

CSV_SOURCES: list[tuple[str, list[str] | None]] = [
    ("thailand_accidents_2019_2026_combined.csv", ["ACC_CODE", "ปีที่เกิดเหตุ"]),
    ("thailand_rainfall_2021_2026_combined.csv", None),
    ("thailand_flood_risk_areas.csv", ["GEOCODE", "Month"]),
    ("thailand_heavy_rain_R35mm.csv", ["ProvinceID", "Year"]),
    ("stt_car_fuel_at_25690228.csv", None),
    ("car_registered_new_2565_2569_combined.csv", None),
]
JSON_SOURCES = [
    "public_holidays_all.json",
    "thai_bank_holidays_all.json",
    "tat_tourist_attractions.json",
]


def render_column_table(columns: tuple[ColumnProfile, ...]) -> str:
    rows = []
    for c in columns:
        sample = html.escape(", ".join(c.sample_values[:3]))
        minmax = f"{c.min_value} .. {c.max_value}" if c.min_value is not None else ""
        flag = ' class="warn"' if c.null_pct >= 90 else ""
        rows.append(
            f"<tr{flag}><td>{html.escape(c.name)}</td><td>{html.escape(c.dtype)}</td>"
            f"<td>{c.null_pct:.1f}%</td><td>{c.distinct_count}</td>"
            f"<td>{sample}</td><td>{html.escape(minmax)}</td></tr>"
        )
    return (
        "<table><tr><th>Column</th><th>dtype</th><th>%null</th>"
        "<th>distinct</th><th>sample</th><th>min..max</th></tr>" + "".join(rows) + "</table>"
    )


def render_file_section(profile: FileProfile) -> str:
    m = profile.metadata
    dup = profile.duplicates
    key_info = ""
    if dup.candidate_key_columns:
        key_info = f"""
        <p>Candidate key <code>{html.escape(", ".join(dup.candidate_key_columns))}</code>:
        {dup.candidate_key_duplicate_group_count} duplicate groups
        ({dup.candidate_key_duplicate_row_count} rows), {dup.candidate_key_null_count}
        rows with a null key component.</p>"""

    pairs_info = ""
    if profile.null_correlated_pairs:
        pairs_list = "".join(
            f"<li>{html.escape(a)} &harr; {html.escape(b)}</li>"
            for a, b in profile.null_correlated_pairs
        )
        pairs_info = f"""
        <p>{len(profile.null_correlated_pairs)} null-correlated column pairs
        (candidate schema-drift / twin-column signal -- proposed for human
        review, not certain):</p><ul>{pairs_list}</ul>"""

    return f"""
    <section>
    <h2>{html.escape(m.filename)}</h2>
    <p>{m.size_bytes:,} bytes &middot; encoding: {m.encoding}
    (BOM: {m.has_bom}) &middot; delimiter: <code>{html.escape(m.delimiter or "")}</code>
    &middot; line endings: {m.line_ending} &middot;
    {profile.row_count:,} rows &times; {profile.column_count} columns</p>
    <p>Full-row duplicates: {profile.duplicates.full_row_duplicate_count}</p>
    {key_info}
    {render_column_table(profile.columns)}
    {pairs_info}
    </section>"""


def render_json_section(filename: str, structure: JsonStructureProfile) -> str:
    record_count_display = (
        structure.record_count if structure.record_count is not None else "n/a (single object)"
    )
    array_keys_display = html.escape(", ".join(structure.array_valued_keys) or "none")
    nested_keys_display = html.escape(", ".join(structure.nested_object_keys) or "none")
    return f"""
    <section>
    <h2>{html.escape(filename)}</h2>
    <p>root type: {structure.root_type} &middot;
    record count: {record_count_display}
    &middot; max nesting depth: {structure.max_depth}</p>
    <p>top-level keys: <code>{html.escape(", ".join(structure.keys))}</code></p>
    <p>array-valued keys: <code>{array_keys_display}</code>
    &middot; nested-object keys: <code>{nested_keys_display}</code></p>
    </section>"""


def render_date_format_section(df: pl.DataFrame) -> str:
    rows = []
    for source in sorted(df["source_file"].unique().to_list()):
        sub = df.filter(pl.col("source_file") == source)
        date_values = [v for v in sub["วันที่เกิดเหตุ"].to_list() if v is not None]
        time_values = [v for v in sub["เวลา"].to_list() if v is not None]
        date_counts = profile_date_column_formats(date_values)
        time_counts = profile_time_column_formats(time_values)
        rows.append(
            f"<tr><td>{html.escape(source)}</td><td>{date_counts}</td><td>{time_counts}</td></tr>"
        )
    return (
        "<section><h2>Mixed date/time formats — accidents file, per source_file</h2>"
        "<p>Confirms CLAUDE.md P1 (date) and P2 (time). "
        '"slash_unambiguous" means one component &gt; 12 so DMY vs MDY is decidable from '
        'that row alone; "slash_ambiguous" means both components &le; 12 (format must be '
        "inferred from the file, not the row).</p>"
        "<table><tr><th>source_file</th><th>date format counts</th>"
        "<th>time format counts</th></tr>" + "".join(rows) + "</table></section>"
    )


def main() -> int:
    configure_logging()

    sections: list[str] = []
    accidents_df: pl.DataFrame | None = None

    for filename, key_cols in CSV_SOURCES:
        path = settings.raw_dir / filename
        profile = profile_csv_file(path, candidate_key_columns=key_cols)
        sections.append(render_file_section(profile))
        log.info(
            "csv_profiled",
            file=filename,
            rows=profile.row_count,
            cols=profile.column_count,
            full_row_dupes=profile.duplicates.full_row_duplicate_count,
            null_pairs=len(profile.null_correlated_pairs),
        )
        if filename.startswith("thailand_accidents"):
            accidents_df = pl.read_csv(
                path,
                encoding="utf8-lossy",
                infer_schema_length=0,
                null_values=["", "NA", "N/A", "-"],
            )

    assert accidents_df is not None
    sections.append(render_date_format_section(accidents_df))

    for filename in JSON_SOURCES:
        path = settings.raw_dir / filename
        data = load_json_file(path)
        structure = profile_json_structure(data)
        sections.append(render_json_section(filename, structure))
        log.info(
            "json_profiled",
            file=filename,
            record_count=structure.record_count,
            max_depth=structure.max_depth,
        )

    report_html = f"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<title>Data profiling report</title>
<style>
  body {{ font-family: 'Tahoma', 'Noto Sans Thai', sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ccc; }}
  h2 {{ padding-bottom: 0.3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #ccc; padding: 0.3rem 0.5rem; text-align: left; }}
  th, td {{ vertical-align: top; }}
  th {{ background: #f0f0f0; }}
  tr.warn {{ background: #fff3cd; }}
  section {{ margin-bottom: 1.5rem; }}
  code {{ background: #f5f5f5; padding: 0.1rem 0.3rem; }}
</style>
</head>
<body>
<h1>Data profiling report</h1>
<p>Phase 1, Task 4 — observe-only. Generated by
<code>scripts/build_data_profiling_report.py</code>. Rows highlighted
yellow have &ge;90% nulls (not necessarily a problem -- see CLAUDE.md P3,
these can be legitimate twin columns).</p>
{"".join(sections)}
</body>
</html>
"""
    out_path = settings.reports_dir / "data_profiling_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Empty template interpolations (e.g. a file with no candidate key) leave
    # whitespace-only lines; strip trailing whitespace per line so the file
    # is clean without depending on the pre-commit hook to fix it.
    normalized = "\n".join(line.rstrip() for line in report_html.splitlines()) + "\n"
    out_path.write_text(normalized, encoding="utf-8")
    log.info("data_profiling_report_written", path=str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
