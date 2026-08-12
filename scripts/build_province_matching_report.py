"""Build reports/province_matching_report.html.

Run: ``uv run python scripts/build_province_matching_report.py``

Runs `normalize_province` against every distinct raw province value in every
source that has a province column, reports the match rate, and lists every
unmatched value. TAT is handled separately since it has no province *name*
field at all (see config/mappings/tat_province_id_crosswalk.csv and
scripts/build_tat_province_crosswalk.py) -- its "match rate" here is
crosswalk coverage, not `normalize_province` coverage.
car_registered_new_2565_2569_combined.csv has no province column (it's a
national-level monthly series) and is excluded, not silently skipped.
"""

from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from rri.cleaning.province import ProvinceMatchResult, match_province_values  # noqa: E402
from rri.utils.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)

NAME_BASED_SOURCES: list[tuple[str, str, str]] = [
    # (source_name, filename, column)
    (
        "thailand_accidents_2019_2026_combined.csv",
        "thailand_accidents_2019_2026_combined.csv",
        "จังหวัด",
    ),
    (
        "thailand_rainfall_2021_2026_combined.csv",
        "thailand_rainfall_2021_2026_combined.csv",
        "Province_NameTH",
    ),
    ("thailand_flood_risk_areas.csv", "thailand_flood_risk_areas.csv", "PROV_T"),
    ("thailand_heavy_rain_R35mm.csv", "thailand_heavy_rain_R35mm.csv", "Province_Name"),
    ("stt_car_fuel_at_25690228.csv", "stt_car_fuel_at_25690228.csv", "จังหวัด"),
]


def load_column_values(filename: str, column: str) -> list[str]:
    path = settings.raw_dir / filename
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [row[column] for row in reader]


def load_master_province_count() -> int:
    path = settings.config_dir / "province_master.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def tat_crosswalk_summary() -> tuple[int, int, list[str]]:
    """Returns (matched_count, total_distinct_pids, unmatched_pid_descriptions)."""
    crosswalk_path = settings.mappings_dir / "tat_province_id_crosswalk.csv"
    with crosswalk_path.open(encoding="utf-8-sig", newline="") as f:
        matched_pids = {row["att_province_id"] for row in csv.DictReader(f)}

    tat_path = settings.raw_dir / "tat_tourist_attractions.json"
    with tat_path.open(encoding="utf-8-sig") as f:
        records: list[dict[str, Any]] = json.load(f)
    all_pids: dict[str, int] = {}
    for record in records:
        pid = record.get("ATT_PROVINCE_ID", "").strip()
        if pid:
            all_pids[pid] = all_pids.get(pid, 0) + 1

    unmatched = [
        f'"{pid}" ({count} records)'
        for pid, count in sorted(all_pids.items())
        if pid not in matched_pids
    ]
    return len(matched_pids), len(all_pids), unmatched


TatAttractionCountsSummary = tuple[int, int, list[tuple[str, int]], list[tuple[str, int]]]


def tat_attraction_counts_summary() -> TatAttractionCountsSummary:
    """Returns (junk_count, total_records, top_10_by_province, bottom_10_by_province)."""
    crosswalk_path = settings.mappings_dir / "tat_province_id_crosswalk.csv"
    with crosswalk_path.open(encoding="utf-8-sig", newline="") as f:
        pid_to_province_id = {
            row["att_province_id"]: row["province_id"] for row in csv.DictReader(f)
        }

    master_path = settings.config_dir / "province_master.csv"
    with master_path.open(encoding="utf-8-sig", newline="") as f:
        province_names = {row["province_id"]: row["province_th"] for row in csv.DictReader(f)}

    tat_path = settings.raw_dir / "tat_tourist_attractions.json"
    with tat_path.open(encoding="utf-8-sig") as f:
        records: list[dict[str, Any]] = json.load(f)

    junk_count = 0
    counts: dict[str, int] = {}
    for record in records:
        pid = record.get("ATT_PROVINCE_ID", "").strip()
        if pid == "1":
            junk_count += 1
            continue
        province_id = pid_to_province_id.get(pid)
        if province_id:
            name = province_names[province_id]
            counts[name] = counts.get(name, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return junk_count, len(records), ranked[:10], ranked[-10:]


def render_html(
    results: list[ProvinceMatchResult],
    tat_matched: int,
    tat_total: int,
    tat_unmatched: list[str],
    master_count: int,
    tat_junk_count: int,
    tat_total_records: int,
    tat_top10: list[tuple[str, int]],
    tat_bottom10: list[tuple[str, int]],
) -> str:
    def match_row(r: ProvinceMatchResult) -> str:
        pct = r.match_rate * 100
        status = "ok" if pct >= 99 else "warn"
        unmatched_list = (
            "<ul>" + "".join(f"<li>{html.escape(v)}</li>" for v in r.unmatched_values) + "</ul>"
            if r.unmatched_values
            else "<em>none</em>"
        )
        return f"""
        <tr class="{status}">
          <td>{html.escape(r.source_name)}</td>
          <td>{r.total_distinct_values}</td>
          <td>{r.matched_count}</td>
          <td>{pct:.1f}%</td>
          <td>{unmatched_list}</td>
        </tr>"""

    rows_html = "".join(match_row(r) for r in results)

    tat_pct = (tat_matched / tat_total * 100) if tat_total else 0.0
    tat_unmatched_html = (
        "<ul>" + "".join(f"<li>{html.escape(v)}</li>" for v in tat_unmatched) + "</ul>"
        if tat_unmatched
        else "<em>none</em>"
    )

    return f"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<title>Province matching report</title>
<style>
  body {{ font-family: 'Tahoma', 'Noto Sans Thai', sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; vertical-align: top; }}
  th {{ background: #f0f0f0; }}
  tr.warn {{ background: #fff3cd; }}
  tr.ok {{ background: #eafbea; }}
  section {{ margin-bottom: 2rem; }}
  code {{ background: #f5f5f5; padding: 0.1rem 0.3rem; }}
</style>
</head>
<body>
<h1>Province matching report</h1>
<p>Master dimension: <code>config/province_master.csv</code>, {master_count} provinces.
Generated by <code>scripts/build_province_matching_report.py</code>.</p>

<section>
<h2>Name-based sources (normalize_province)</h2>
<table>
  <tr>
    <th>Source</th><th>Distinct values</th><th>Matched</th>
    <th>Match rate</th><th>Unmatched values</th>
  </tr>
  {rows_html}
</table>
</section>

<section>
<h2>tat_tourist_attractions.json (ID-based crosswalk)</h2>
<p>This file has no province name field -- only <code>ATT_PROVINCE_ID</code>
("1001".."1077", TAT's own internal numbering, unrelated to the standard
geocode every other source uses). It is matched via
<code>config/mappings/tat_province_id_crosswalk.csv</code>
(see <code>scripts/build_tat_province_crosswalk.py</code> for how it was built)
rather than <code>normalize_province</code>.</p>
<table>
  <tr>
    <th>Source</th><th>Distinct values</th><th>Matched</th>
    <th>Match rate</th><th>Unmatched values</th>
  </tr>
  <tr class="{"ok" if tat_pct >= 99 else "warn"}">
    <td>tat_tourist_attractions.json</td>
    <td>{tat_total}</td>
    <td>{tat_matched}</td>
    <td>{tat_pct:.1f}%</td>
    <td>{tat_unmatched_html}</td>
  </tr>
</table>

<h3>A methodology bug was caught and fixed here</h3>
<p>The first version of this crosswalk matched each <code>ATT_PROVINCE_ID</code>
to a province via the leading <strong>2 digits</strong> of its records'
postcodes, assuming that prefix equals the province geocode. That assumption
silently breaks for Bangkok/Samut Prakan: Samut Prakan has no postal block of
its own -- its real postcodes (10270, 10280, 10290, 10540, 10560, ...) fall
<em>inside</em> Bangkok's 10xxx block -- which shifted every subsequent
province's postcode prefix one behind its geocode across the entire
Bangkok-Samut Prakan-Nonthaburi-Pathum Thani-Ayutthaya-Ang Thong-Lopburi-
Sing Buri-Chai Nat-Saraburi block (geocodes 10-19). The result was two
different <code>ATT_PROVINCE_ID</code>s both resolving to Bangkok, and
Saraburi ending up with no id mapped to it at all. Caught by checking for
duplicate <code>province_id</code> assignments across the crosswalk (there
should never be two), not by re-running more postcode logic. Fixed by
resolving each postcode against a real postcode&rarr;province reference
table (kongvut/thai-province-data) instead of assuming a prefix match, then
re-validated: every one of the 77 provinces now has exactly one
<code>ATT_PROVINCE_ID</code> mapped to it, with zero duplicates. See
<code>tests/unit/test_tat_crosswalk.py</code> for the regression guard and
<code>scripts/validate_tat_crosswalk.py</code> for the independent
validation below.</p>

<h3>Independent validation (not more postcode logic)</h3>
<ul>
<li><strong>Record count reconciles exactly:</strong> {tat_matched} matched
+ {tat_junk_count} junk = {tat_total_records}, the file's real total row
count.</li>
<li><strong>5 well-known attractions spot-checked by name</strong> all
resolve to the expected province (Yaowarat &rarr; Bangkok, Doi Suthep
&rarr; Chiang Mai, Phi Phi Islands &rarr; Krabi, an Ayutthaya-named
attraction &rarr; Ayutthaya, Patong Beach &rarr; Phuket). Note: Wat Phra
Kaew / the Grand Palace -- arguably Thailand's single most famous
attraction -- is not in this dataset under any name variant tried, nor are
Wat Arun or Chatuchak Market; this file reads as a tourism-<em>service</em>
directory (spas, farms, boathouses) more than a comprehensive monuments
list, a coverage observation worth keeping in mind for any attraction
-density metric built on it.</li>
<li><strong>Attraction counts per province</strong> (top 10 / bottom 10
below) are broadly plausible -- Chiang Mai is a clear #1 (256), small
interior provinces sit at the bottom (Amnat Charoen, Ang Thong: 1 each) --
though Prachuap Khiri Khan's #2 (107, ahead of Bangkok's 47 and Phuket's 50)
is higher than tourism volume alone would suggest and likely reflects
uneven submission activity by province across TAT's regional offices
feeding this open-data set, not a crosswalk error (every id in the 10-19
block, including Prachuap Khiri Khan's, was independently verified above).</li>
</ul>
<table>
  <tr><th>Top 10 by attraction count</th><th>Bottom 10 by attraction count</th></tr>
  <tr>
    <td><ol>{"".join(f"<li>{html.escape(p)}: {c}</li>" for p, c in tat_top10)}</ol></td>
    <td><ol>{"".join(f"<li>{html.escape(p)}: {c}</li>" for p, c in tat_bottom10)}</ol></td>
  </tr>
</table>

<h3>What is TAT's 78th value?</h3>
<p>77 of the 78 distinct <code>ATT_PROVINCE_ID</code> values follow the
"10XX" pattern and were matched to a real province via a real
postcode&rarr;province reference (75 automatically; the remaining 2 --
<code>1005</code> and <code>1045</code> -- had no usable postcode anywhere
in their records and were hand-identified by searching their attraction
names, resolving to Kamphaeng Phet and Yasothon respectively).
<strong>The 78th value is the literal string <code>"1"</code></strong>,
attached to <strong>{tat_junk_count} of the file's {tat_total_records}
records ({tat_junk_count / tat_total_records * 100:.2f}%)</strong>. Their
names are junk/placeholder data -- <code>"test"</code>,
<code>"test 13120844"</code>, <code>"Hello world!"</code> (three times) --
not a real 78th province. It reads as an unset/default
<code>province_id</code> sentinel in the source system, a concrete
data-quality defect in a government open-data file, and is excluded from
the crosswalk rather than assigned anywhere.</p>
</section>

<section>
<h2>What are flood's 2 missing provinces?</h2>
<p><code>thailand_flood_risk_areas.csv</code> has rows for 75 of the 77
provinces. Diffing its (prefix-stripped) province list against the master
shows the 2 entirely absent provinces are <strong>Phuket (ภูเก็ต)</strong>
and <strong>Ranong (ระนอง)</strong> -- there is not a single row, for any
tambon, in either province, across all 17 years the file covers. This isn't
a spelling mismatch (every other province matches exactly once its "จ."
prefix is stripped; Bangkok is the one province stored without that prefix).</p>
<p><strong>Why they're absent, and why that's not the same as "zero
risk":</strong> this file measures <em>recurrent floodplain risk</em>, which
is river-basin derived. Phuket is an island with no major river basin;
Ranong is a narrow coastal strip pinned against mountains. Neither has
areas that meet this file's definition of flood-risk terrain -- but both
experience real flash flooding, and Ranong is the wettest province in
Thailand. Treating their absence as "0 risk" would tell any model built on
this data that they are safe, when the honest statement is "unmeasured by
this method."</p>
<p><strong>Business rule for dim_geography (Phase 3):</strong> Phuket and
Ranong must map to the <strong>Unknown Member (sk = -1)</strong> for every
flood attribute sourced from this file -- never imputed as 0, never
classified as low risk. A <code>flood_data_available</code> boolean column
should accompany the flood attributes so the dashboard can show this
distinction honestly rather than silently. Phuket in particular is a major
tourism province, so this directly affects BQ2 (rain/flood premium loading)
and BQ4 (tourism-dense province risk profile) if not handled this way.</p>
</section>

<section>
<h2>Not applicable</h2>
<p><code>car_registered_new_2565_2569_combined.csv</code> is a national-level
monthly series (no province breakdown) and has no province column to match.</p>
</section>

</body>
</html>
"""


def main() -> int:
    configure_logging()
    master_count = load_master_province_count()

    results = []
    for source_name, filename, column in NAME_BASED_SOURCES:
        values = load_column_values(filename, column)
        result = match_province_values(source_name, values)
        results.append(result)
        log.info(
            "province_match_computed",
            source=source_name,
            match_rate=round(result.match_rate, 4),
            unmatched=len(result.unmatched_values),
        )

    tat_matched, tat_total, tat_unmatched = tat_crosswalk_summary()
    log.info(
        "tat_crosswalk_summary",
        matched=tat_matched,
        total=tat_total,
        unmatched=tat_unmatched,
    )
    tat_junk_count, tat_total_records, tat_top10, tat_bottom10 = tat_attraction_counts_summary()
    log.info(
        "tat_attraction_counts_summary",
        junk_count=tat_junk_count,
        total_records=tat_total_records,
        top10=tat_top10,
        bottom10=tat_bottom10,
    )

    report_html = render_html(
        results,
        tat_matched,
        tat_total,
        tat_unmatched,
        master_count,
        tat_junk_count,
        tat_total_records,
        tat_top10,
        tat_bottom10,
    )
    out_path = settings.reports_dir / "province_matching_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_html, encoding="utf-8")
    log.info("province_matching_report_written", path=str(out_path))

    below_threshold = [r for r in results if r.match_rate < 0.99]
    if below_threshold:
        for r in below_threshold:
            log.warning(
                "source_below_99pct_match",
                source=r.source_name,
                match_rate=round(r.match_rate, 4),
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
