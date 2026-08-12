# CLAUDE.md — Project Context (read this at the start of every session)

> This is the project's persistent memory. We work phase-by-phase; each new session
> starts with no memory of the last one, so this file carries the context.
> **Update §12 (Status) at the end of every phase.**

---

## 1. What this project is

**Name:** Weather-Aware Road Risk Intelligence for Motor Insurance
**Course:** Data Mining (69-1) — Group Assignment #1: Mini Data Warehouse & Analytics Dashboard
**Team:** 4 Computer Engineering students, Kasetsart University

**Business framing:** We act as a fictional motor insurer, **"SiamShield Insurance"**.
Today it prices with crude tiers (Bangkok vs. upcountry + vehicle type), causing **adverse selection**:
low-risk areas overpay and leave; high-risk areas underpay and pile in → underwriting loss.

**System goal:** an end-to-end pipeline `raw data → ETL → star-schema DW → ML → dashboard`
answering how real risk varies by **location × time × vehicle type × weather**, plus a 6-month-ahead forecast.

---

## 2. Non-negotiables

| # | Rule | Why |
|---|---|---|
| **N1** | `data/00_raw/` is **read-only**. Never modify it. | Course requirement: the process must be re-runnable without manual data edits |
| **N2** | **All logic lives in `src/`, never in notebooks.** Notebooks only import and call `src`. | Notebooks have hidden state, can't be diffed or tested — but we still need them so the professor can follow each step |
| **N3** | **Every cleaning rule has a unit test.** | This is what shows real software engineering, not a one-off script |
| **N4** | **Every data fix must log how many rows changed, from what to what.** | Feeds the Before/After table in the report (Data Quality = 20% of the grade) |
| **N5** | **ML trains only on real data (`fact_accident`), never on the simulated portfolio.** | Otherwise the model just reads back our own assumptions — circular reasoning |
| **N6** | **Time-based split only** (train ≤2023 / valid 2024 / test 2025). No random K-fold. | Time-series leakage |
| **N7** | **Never drop a fact row because a dimension is missing** → use Unknown Member (`sk = -1`). | DW totals must always reconcile to the raw files |
| **N8** | **No `print()`. Use `structlog`.** | Production-grade practice |
| **N9** | **Everything deterministic** — `random_seed = 42` wherever randomness appears. | Re-runs must reproduce |
| **N10** | **Language:** code, variables, commits, docstrings → English. Dashboard UI, report text, notebook markdown → **Thai**. | End users and the grader are Thai |

---

## 3. Agreed stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11+ | |
| Dependencies | `uv` (fallback: venv + pip) | fast, reproducible lock file |
| DataFrames | Polars primary, Pandas only where needed | much faster on 161k+ rows |
| **Warehouse** | **DuckDB** (`data/30_warehouse/warehouse.duckdb`) | embedded OLAP, single file, easy to share |
| Submission extras | `schema_mysql.sql` + `load_to_mysql.py` + Parquet/CSV exports | course requires source files be openable/runnable |
| Validation | `pandera` | schemas declared as code, checked automatically |
| ML baseline | `statsmodels` (Poisson / Gamma GLM) | the actual actuarial standard — our comparison line |
| ML main | `lightgbm` | |
| Neural nets | `pytorch` (MLP + entity embeddings, GRU) | |
| Explainability | `shap` | insurance is regulated — unexplainable = unusable |
| Dashboard | **Streamlit**, Thai UI → Streamlit Community Cloud | can call models and the Claude API live; Power BI cannot |
| LLM insight | Anthropic API (`claude-sonnet-4-6`) | |
| CLI | `typer` | |
| Logging | `structlog` | |
| Tests | `pytest` + `pytest-cov` | |
| Lint / types | `ruff` + `mypy` | |
| Hooks | `pre-commit` | |
| CI | GitHub Actions (ruff + mypy + pytest on every push) | |

**Environments:** develop in VS Code now; later port to Google Colab with Google Drive mounted.
**Therefore: never hard-code a path.** All paths resolve through `config/settings.py`.

---

## 4. Project structure

```
road-risk-intelligence/
├── CLAUDE.md                     ← this file
├── README.md · pyproject.toml · Makefile
├── .pre-commit-config.yaml · .github/workflows/ci.yml
│
├── config/
│   ├── settings.py               ← ALL paths live here (local + Colab aware)
│   ├── province_master.csv       ← 77 provinces (Phase 1)
│   └── mappings/                 ← hand-editable, version-controlled
│       ├── province_aliases.csv
│       ├── cause_groups.csv
│       ├── crash_type_groups.csv
│       ├── road_geometry_groups.csv
│       └── vehicle_groups.csv
│
├── data/
│   ├── 00_raw/                   ← 🔒 READ-ONLY
│   ├── 10_staging/               ← cleaned parquet
│   ├── 20_intermediate/
│   ├── 30_warehouse/             ← warehouse.duckdb
│   └── 40_exports/               ← parquet / csv / sql for submission
│
├── src/rri/                      ← rri = Road Risk Intelligence
│   ├── io/          ← one reader per source
│   ├── cleaning/    ← one module per problem family
│   ├── contracts/   ← pandera schemas
│   ├── warehouse/   ← dim / fact builders
│   ├── features/
│   ├── models/
│   ├── insight/     ← Claude API + verifier
│   └── utils/       ← logging, profiling, thai_text
│
├── notebooks/       ← narrative for the grader (import from src only)
│   01_data_profiling · 02_data_cleaning · 03_warehouse_build
│   04_portfolio_simulation · 05_modeling_baseline
│   06_modeling_neural · 07_insight_verification
│
├── tests/           ← mirrors src/
├── dashboard/       ← Streamlit app
└── reports/         ← html/png/csv artifacts for the written report
```

---

## 5. Data sources (final)

### 🟢 Full use
| File | Role |
|---|---|
| `thailand_accidents_2019_2026_combined.csv` | **Core** → `fact_accident`. 161,307 rows × 45 cols |
| `stt_car_fuel_at_25690228.csv` | **Exposure.** Cumulative registered vehicles by province × type × fuel. 44.3M vehicles, 77 provinces, snapshot 2026-02-28 |
| `thailand_rainfall_2021_2026_combined.csv` | **Rainfall forecast, lead 0–5 months** (the project's key differentiator) |
| `thailand_flood_risk_areas.csv` | Aggregate tambon → province: % high-risk tambons, peak month |
| `public_holidays_all.json` | `dim_date` — **only has 2025–2026, must backfill 2019–2024** |
| `thai_bank_holidays_all.json` | Extra boolean column in `dim_date` |
| `tat_tourist_attractions.json` | Tourist-attraction density per province (nested JSON) |

### 🟡 Partial use
| File | Used for |
|---|---|
| `car_registered_new_2565_2569_combined.csv` | **Important:** national monthly growth curve used to **back-cast** per-province vehicle counts (see §7.1), plus vehicle-type taxonomy |
| `thailand_heavy_rain_R35mm.csv` | Years **2014–2023 only** → one column in `dim_geography` |

### 🔴 Dropped — do not use
- `banks.json` — no join key to anything
- `tat_tourist_attractions_clean.csv` — exact duplicate of the JSON (1,731 rows both)

---

## 6. Known data-quality problems (17, all verified against the real files)

Fix in this order.

### 🔴 P1 — Three date formats mixed in one column (fix before anything else)

| source_file | Format | Unambiguous rows | Ambiguous rows |
|---|---|---|---|
| accident2019 | **DD/MM/YYYY** | 11,846 | 7,348 |
| accident2020 | **MM/DD/YYYY** | 12,856 | 8,382 |
| accident2021 | **MM/DD/YYYY** | 11,751 | 8,311 |
| accident2022 | **MM/DD/YYYY** | 12,431 | 8,135 |
| accident2023 | **MM/DD/YYYY** | 14,578 | 9,263 |
| accident2024 | **Excel serial** (e.g. `45746`) | 23,952 | 0 |
| accident2025 | **Excel serial** | 23,715 | 0 |
| accident2026 | **DD/MM/YYYY** | 4,760 | 3,978 |

**Approach:** detect format **per file** (not per row) using rows where one component > 12, then apply that
format to the ambiguous rows in the same file. Excel serials use `origin = 1899-12-30`.
Verified: no single file mixes DMY and MDY.

**Phase 1 Task 4 found one exception the per-file rule alone would miss:** `accident2020.csv`
(otherwise pure MM/DD/YYYY text) has exactly 1 row (`ACC_CODE=3810244`) where `วันที่เกิดเหตุ` is an
Excel serial (`44178` → 2020-12-13, plausible: 2 days before its `วันที่รายงาน` of 12/15/2020). Immaterial
to the 161,307-row total, but it's the concrete case the 3-layer safety net below exists to catch —
a blanket per-file format assumption would silently mis-parse this one row.

⚠️ **The per-file-format assumption itself was stress-tested (owner-requested) using P5's 448
day/month-transposed duplicate pairs (see P5) — read that analysis before touching Q1/P1 code.**
All 448 pairs sit in the same `source_file` as each other (100%, all in `accident2023.csv`), which is
the branch that would normally mean "the assumption is broken, redesign." **The assumption survives —
settled, not just evidence:** if `accident2023.csv` genuinely mixed DMY and MDY rows, that mixing would
show up spread across the whole file's unambiguous evidence, not confined to duplicate pairs — some
rows would have a **first** component > 12 (DMY evidence). It has zero. Verified directly:
`accident2023.csv`'s 23,841 date values split **14,578 with second-component > 12 (MDY evidence) / 0
with first-component > 12 (DMY evidence) / 0 with both / 9,263 ambiguous (neither)**. Zero
counter-evidence across the entire file, not just a majority — the file is uniformly MDY, full stop.
The 448 transposed pairs don't change this: every one has *both* components ≤ 12 in *both* strings —
structurally, digit-transposition can only occur inside the already-ambiguous range, so it can never
produce DMY evidence and never contaminates the unambiguous-row vote. The swap pattern itself
(`a/b/y` ↔ `b/a/y`, exact, 100% of the 448) is a re-keying transcription error, not a file that
secretly mixes formats.

🔴 **Correction to an earlier draft of safety net #4:** it was first written as "both rows of a known
P5 duplicate pair must resolve to the same calendar date after parsing." That is wrong and must not be
implemented as written. Given the file is uniformly MDY (above), both rows of a transposed pair parse
as MDY by construction — one is `month=a day=b`, the other `month=b day=a` — so they are **mathematically
guaranteed to produce different dates**. A rule requiring them to match would flag all 448 pairs as
parser failures every single time, and anyone trusting that signal would "fix" a parser that is actually
correct. The real problem in these 448 pairs isn't the date format — it's that someone keyed the same
event twice and swapped the day/month fields the second time. Split into two separate, non-conflicting
concepts:

- **Safety net #4a (the real parser check):** among P5's known duplicate pairs, restrict to the subset
  where the **raw date string is identical** across the group (1,193 of the 1,641 ACC_CODE duplicate
  groups). For those, a deterministic parser must always produce one date — any mismatch is a genuine
  parser/format-detection bug. Verified against the real file: **1,193 groups checked, 0 mismatches.**
  Implemented as `check_duplicate_date_string_consistency()` in `src/rri/utils/profiling.py`.
- **`date_ambiguous_pairs` report (not a parser check):** the remaining 448 groups, where the two rows'
  raw strings are exact digit-transpositions of each other, are a **data-entry problem**. They belong in
  their own report, resolved separately (below) — never fed into the 4a identical-string check.
  Detected via `find_transposed_date_string_pairs()`.

✅ **Completeness check:** 1,193 (identical-raw-string groups, 4a) + 448 (transposed groups,
`date_ambiguous_pairs`) = **1,641 — exactly matches the total ACC_CODE duplicate-group count from P5.**
Every duplicate group is accounted for by exactly one branch, with no overlap and nothing left over.
This proves the split is exhaustive over the real data, not a sample or an approximation.

⚠️ **The 90-day threshold in safety net #2 was invented at design time, before looking at any data —
it had no empirical basis and turned out to be badly wrong.** Profiled `report_date − accident_date`
across all 8 files (`compute_report_lag_days()` / `summarize_report_lag()` /
`report_lag_quantiles_by_group()` in `src/rri/utils/profiling.py`), 161,224 of 161,307 rows parseable
(83 rows fail to parse under their file's format and are excluded, immaterial):

| Stat | Value |
|---|---|
| Median | 45 days |
| p95 | 284 days |
| p99 | **398 days** |
| Max | 1,254 days |
| Rows with gap > 90 days | **46,179 / 161,224 = 28.6%** |
| Rows with negative gap (report before accident — a different anomaly, not addressed here) | 2,696 |

**28.6% of all rows would fail a hard 90-day gate.** That is not a tail of outliers — treating gap > 90
as invalid at that scale would silently reject a large slice of genuinely valid data, exactly the
failure mode flagged when this was raised: "a threshold that's too tight will reject valid data at
scale."

**The distribution differs materially by both year and agency** — a single global number is a real
simplification, noted here rather than hidden:
- **By year (`source_file`):** p95 ranges from 57 days (`accident2026`, still-incomplete year) to 368
  days (`accident2020`); p99 from 72 to 450. Earlier years consistently show longer lags than 2024–2026,
  suggesting either slower reporting/finalization workflows in earlier years or a data-capture change —
  not investigated further here.
- **By agency (`หน่วยงาน`):** Rural Roads Dept. (`กรมทางหลวงชนบท`) is tightest — p95 = 110, p99 = 235.
  Highways Dept. (`กรมทางหลวง`, 92.6% of all rows) sits in the middle — p95 = 286, p99 = 393. Expressway
  Authority (`การทางพิเศษแห่งประเทศไทย`) is far wider — **p95 = 986, p99 = 1,219** — over three years.
  This is consistent with the known agency coverage-bias limitation (P7) and suggests `report_date` may
  mean something closer to "case finalized" for this agency rather than "initially filed." **Not
  addressed now — a per-agency threshold is a reasonable Phase 2 refinement, not implemented here.**

**Decision: safety net #2's upper bound is set to 398 days (p99 of the observed distribution), not
90.** This accepts ~1% of legitimate rows sitting past the bound by design (that is what a p99 threshold
means) in exchange for not rejecting the 28.6% that a 90-day bound would have wrongly caught. This is a
single global number despite the agency/year variance above — documented as a simplification, not a
final answer.

**Resolving the 448 — use safety net #2 (accident_date ≤ report_date, gap ≤ 398 days) as a discriminator.**
For each pair, both candidate dates (from the two transposed raw strings) are tested against that rule
using each row's own report date; a candidate is valid only if every row carrying that raw string
passes (this correctly covers the 23 groups that have 3 rows — 2 distinct dates, one repeated — not
just the simple 2-row case). Implemented as `discriminate_transposed_date_pairs()`, run against the real
`accident2023.csv` data, **re-run after the threshold correction — the split shifted as expected:**

| Outcome | Count @90 days (invented) | Count @398 days (p99, data-derived) | Meaning |
|---|---|---|---|
| Resolved uniquely | 222 | **138** | exactly one candidate passes → that's the true date, the other row is the keying error |
| Both valid | 196 | **310** | genuinely undecidable from this evidence — flag, don't guess |
| Neither valid | 30 | **0** | — |
| **Total** | **448** | **448** | matches `find_transposed_date_string_pairs()` exactly (425 two-row + 23 three-row groups) |

**All 30 "neither valid" cases at the old threshold turn out to be legitimately long reporting lags, not
corrupted rows** — confirms the concern that motivated re-checking the threshold in the first place: at
398 days every one of them has at least one valid candidate. The cost of the wider, correct threshold is
weaker discriminating power (fewer pairs resolve uniquely — 138 vs. 222 — because a 398-day window makes
both candidates plausible more often), not a bug; a looser, honest bound necessarily discriminates less
than a tight, wrong one appeared to.

**Phase 2 design note (not implemented yet — for whoever builds the real parser):** for the 310 groups
(620 rows) that safety net #2 cannot resolve, add a `date_certainty` field (`certain` / `ambiguous`) to
`fact_accident` rather than silently picking a date. 448 of 23,841 rows in `accident2023.csv` is ~1.9% —
small, but a date wrong by months is exactly the kind of error that would distort BQ3 (holiday-period
risk analysis), so it must be flagged, not guessed.

**Mandatory safety net (4 layers, #4 corrected to 4a):**
1. Parsed year **must equal the existing `ปีที่เกิดเหตุ` column** — a single mismatch fails the pipeline
2. `accident_date ≤ report_date`, and the gap must be ≤ **398 days** (p99 of the observed distribution —
   see above; NOT 90, which was an invented design-time guess that would reject 28.6% of all rows)
3. Day-of-month histogram must be flat — a wrong parse shows a spike on days 1–12
4a. **Duplicate-group rows sharing an identical raw date string must parse to the same calendar date**
   (day resolution). Rows in a duplicate group whose raw strings are transposed digits are expected to
   parse differently and are handled by the `date_ambiguous_pairs` / `date_certainty` process above, not
   this check.

### 🔴 P2 — Two time formats
String `21:10` mixed with Excel day fractions `0.375` (=09:00), `0.625` (=15:00).
If it parses as a float in [0,1), multiply by 24. `0:00` appears 219 times — may be a
"time unknown" placeholder; check against the report-time column before treating it as midnight.

### 🔴 P3 — Schema drift (twin columns)
| Old name | New name | Null % |
|---|---|---|
| `รถยนต์นั่งส่วนบุคคล` | `รถยนต์นั่งส่วนบุคคล/รถยนต์นั่งสาธารณะ` | 94.6 |
| `รถบรรทุกไม่เกิน10ล้อ` | `รถบรรทุกมากกว่า6ล้อไม่เกิน10ล้อ` | 94.6 |
| `รถบรรทุกมากกว่า10ล้อ` | `รถบรรทุกมากกว่า10ล้อ(รถพ่วง)` | 94.6 |
| `บริเวณที่เกิดเหตุ` | `บริเวณที่เกิดเหตุอื่น` | 94.8 |
| `รถอื่นๆ` | `อื่นๆ` | 94.6 |

⚠️ **Do not drop these for being >90% null.** They are null *in blocks by source year*, not at
random — different years used different column names. **Coalesce them** — but see the correction
below, this applies to only 4 of the 5 rows in the table above, not all 5.

✅ **Confirmed old-name/new-name pairs, safe to coalesce (Phase 1 Task 4, exhaustive check, all
161,307 rows):** the 3 vehicle-type pairs and `รถอื่นๆ`/`อื่นๆ` — **zero row-overlap** in every
case, and identical population rates across all four (82.7% old / 5.4% new). These are genuinely one
field renamed at a fixed cutover point.

⚠️ **`บริเวณที่เกิดเหตุ` / `บริเวณที่เกิดเหตุอื่น` is NOT a twin-column pair — the whole row above is
wrong for this one.** Two hypotheses were checked and refuted (Phase 1 Task 4, owner-requested
follow-up):

1. ~~Old name replaced by new name~~ — **false.** `บริเวณที่เกิดเหตุ` is populated across *every*
   year 2019–2026 (91.7% overall, 90.2% even within 2026 alone). It was never phased out.
2. ~~`บริเวณที่เกิดเหตุอื่น` is a free-text detail field used when the primary column is "อื่นๆ"
   (other)~~ — **false.** Checked all 7,606 rows where both are populated: **0.0%** have
   `บริเวณที่เกิดเหตุ = "อื่นๆ"`. The primary column holds ordinary values
   (`ทางตรง+ไม่มีความลาดชัน`, `ทางโค้งกว้าง+ไม่มีความลาดชัน`, ...) in every one of those rows.

**What the data actually shows:** `บริเวณที่เกิดเหตุอื่น` is populated **only in `accident2026.csv`**
(8,462 of that file's 8,738 rows; 0 rows in any other year) — appearing *alongside*, not instead of,
the primary column (7,606 of those rows have both). Paired values describe the same road-geometry
concept with an overlapping-but-different taxonomy (e.g. primary `ทางตรง+ไม่มีความลาดชัน` "straight,
no slope" vs. secondary `ทางตรง+ที่ราบ` "straight, flat/level" on the same row). This reads as the
source system introducing a second, more granular classification scheme starting in the 2026 file,
captured in parallel with the original — **not** a rename, **not** an "other" detail field. 276 rows
have only the primary value, 856 have only the secondary. **Unresolved — needs Phase 2 investigation
(possibly asking the source agency what changed in 2026), not a plain coalesce and not automatic
routing into `dim_accident_context` as an "other" detail either.**

**This is the same phenomenon as P5's `บริเวณที่เกิดเหตุ` "74% conflict rate"** (see P5) — two
overlapping-but-different classification taxonomies for the same road-geometry concept, one showing
up as a cross-column pair (here) and one showing up as within-column disagreement between duplicate
rows (P5). Both point to the same underlying fix: harmonize the taxonomy before either coalescing
this pair or deduplicating P5's rows.

### 🟠 P4 — Inconsistent categories
`มูลเหตุสันนิษฐาน` has 120 distinct values, should collapse to ~20.
`ลักษณะการเกิดเหตุ` 152 → ~15. `บริเวณที่เกิดเหตุ` 89 → ~12.
Example: `ขับรถเร็วเกินอัตรากำหนด` (99,967) and `ขับรถเร็วเกินอัตราที่กำหนด` (14,636) are the
same thing — combined 114,603 (71% of all cases).
Use committed mapping tables. **Never rely on automatic fuzzy matching alone** — it will merge
"speeding" with "reckless driving", which are different causes.

### 🟠 P5 — Duplicates
- 82 fully identical rows → safe to drop
- 1,702 excess rows from duplicate `ACC_CODE` (1,641 duplicate groups, 3,343 rows total involved,
  3,343 − 1,641 = 1,702) — numbers verified exactly against Phase 1 Task 4 profiling
- `ACC_CODE` is null in 17.3% of rows → **cannot be the primary key**; use a surrogate key

⚠️ **Correction (Phase 1 Task 4):** this entry originally said the duplication was cross-year reuse
and that `(ACC_CODE, year)` is therefore the real key. **Verified false: zero `ACC_CODE` values span
more than one year.** The composite key does not resolve the duplication at all — grouping by
`ACC_CODE` alone and by `(ACC_CODE, year)` produce the identical 1,641 duplicate groups.

**Full investigation (owner-requested follow-up), all 1,641 groups checked exhaustively, not sampled:**

- **Agency:** 100% same-agency (1,641 / 1,641). Zero cross-agency pairs. This rules out systemic
  double-reporting (e.g. a highway/rural-road junction reported by two agencies) — every duplicate
  is one agency's own record appearing twice.
- **Casualty counts:** 100% agree (1,641 / 1,641) across all four columns (`ผู้เสียชีวิต`,
  `ผู้บาดเจ็บสาหัส`, `ผู้บาดเจ็บเล็กน้อย`, `รวมจำนวนผู้บาดเจ็บ`) — **zero conflicts.** Province, road,
  KM, and year are also 100% identical within every group. So the numeric reconciliation figures are
  never ambiguous, regardless of which row in a group is eventually kept.
- **What *does* differ:** free-text description columns, and — in 448 of the 1,641 groups (27%) —
  the accident date. Every one of those 448 date differences (checked exhaustively, not a subset) is
  a **pure day/month digit transposition** under the file's own established format (e.g. one row
  reads `4/12/2023`, the duplicate reads `12/4/2023`, both from `accident2023.csv` which is MDY —
  April 12 vs. December 4). This is a specific, identifiable transcription-error pattern, not random
  date corruption, and it strengthens rather than weakens the "same event, entered twice" reading.
  **All 448 pairs sit in the same file (`accident2023.csv`) — see the P1 stress-test this triggered,
  above.**
- **Free-text conflict rate varies sharply by column** (of the 1,641 groups): `บริเวณที่เกิดเหตุ`
  true conflicts in 1,216 (74%) — "prefer the more complete row" **will not work** for this column,
  most disagreements are non-null vs. non-null; `ลักษณะการเกิดเหตุ` true conflicts in only 66 (4%),
  969 (59%) are a simple null-vs-populated gap — "most complete" works here; `มูลเหตุสันนิษฐาน` true
  conflicts in 340 (21%), 1,172 (71%) identical; `วันที่รายงาน` true conflicts in 544 (33%).

⚠️ **Correction #2 (owner-requested follow-up): most of `บริเวณที่เกิดเหตุ`'s 74% "conflict" rate is
not content disagreement — it's the same taxonomy-drift phenomenon as P3's rejected pair.** Sampled
20 conflicting pairs directly: every single one was a synonym or elaboration-suffix pair (e.g.
`ไม่มีความลาดชัน` "no slope" vs. `ที่ราบ` "flat/level" — same concept, different wording; or
`X` vs. `X/ที่ราบ` — base value vs. base + an added detail). Quantified across all 1,216 conflicting
groups with a simple normalization (collapse known synonyms, strip whitespace, drop an elaboration
suffix after `/`): **85.1% (1,035/1,216) collapse to one value** — and most of the remaining 14.9%
are still the same kind of pattern under slightly different phrasing the simple heuristic didn't
catch (e.g. `ที่ลาดชัน` vs. `โค้งดิ่ง/ที่ลาดชัน`), not genuinely different locations.
**Consequence: order matters. Harmonize the location taxonomy *before* deduplicating, not after** —
picking a per-row tie-break rule first would silently discard rows that actually agree once the
vocabulary is unified, and "which taxonomy wins" is itself a design decision for whoever builds
`dim_accident_context`, not something to bury inside a dedup rule.

**Fatality reconciliation — two separate gates, not one number (owner-requested framing):**
- **Gate 1 — post-Extract, pre-dedup (161,307 rows): 21,814 fatalities.** Proves nothing was lost
  reading the raw files in. This is the number currently recorded below in §8.
- **Gate 2 — post-Load, post-dedup (159,605 rows): 21,497 fatalities** (317 fewer, exact — casualty
  counts agree 100% within every duplicate group, so this number does not depend on which row of a
  group is kept; only the free-text tie-break, still undecided, affects anything downstream of
  fatalities). Proves dedup removed exactly the intended rows, no more and no less.
- **Both gates are recorded in §8 and must stay separate** — they catch different classes of error
  (ingest loss vs. over/under-aggressive dedup) and collapsing them to a single target loses one of
  the two checks.

### 🟠 P6 — Province names don't match across sources
Accidents `ลพบุรี` · flood `จ.นนทบุรี` (prefix) · R35mm `Bangkok` (English) · registration `กระบี่`
Province counts: accidents 77 ✅ · rainfall 77 ✅ · registration 77 ✅ · **flood 75** ❌ · **TAT 78** ❌

**Resolved in Phase 1 Task 3** (`config/mappings/`, `reports/province_matching_report.html`):
- **Flood's missing 2** are Phuket and Ranong — not a spelling issue, the file has zero rows for
  either province across all 17 years. Flood risk here is river-basin derived; Phuket is an island
  with no major basin, Ranong is a narrow coastal strip against mountains — neither has terrain
  meeting this file's definition, but both have real flash-flood risk (Ranong is Thailand's wettest
  province). **Locked-down rule for Phase 3 `dim_geography`:** these two must map to the Unknown
  Member (`sk = -1`) for every flood attribute — never impute 0, never classify as low risk. Add a
  `flood_data_available` boolean so the dashboard shows this honestly. Phuket is a major tourism
  province, so this directly affects BQ2 and BQ4 if skipped.
- **TAT's 78th value** is the literal string `"1"` (13 of 1,731 records, 0.75%) — junk/placeholder
  data (`"test"`, `"Hello world!"` ×3), not a real province. TAT also has no province *name* field
  at all, only its own internal `ATT_PROVINCE_ID` numbering (unrelated to the standard geocode) —
  crosswalked via `config/mappings/tat_province_id_crosswalk.csv`.

### 🟡 P7 — Remaining
| Problem | Evidence | Fix |
|---|---|---|
| Coordinates outside Thailand | lat outside 5–21 in 5 rows, lon outside 97–106 in 10 rows, overlapping in 5 rows → 10 distinct rows affected | null + `is_geo_valid` flag |
| BOM `\ufeff` | every CSV | `encoding='utf-8-sig'` |
| Typo in header | `Foreast_Month` | rename map |
| Buddhist vs Gregorian year | registration 2565–2569 | `CE = BE − 543` |
| Rainfall has 6 rows per province-month | lead 0–5 | ⚠️ **not duplicates** — pivot to `rain_lead0..5` |
| TAT coordinates embedded in a string | `"7.376891, 99.6734393"` | split, cast, validate bounds |
| ID overflow | `ATT_ID` is 17 digits; `9637.0` float/int mix | read everything as string first |
| Injury totals | check `serious + minor == total` | flag mismatches, don't silently fix |
| Incomplete periods | 2026 has 8,738 rows (~4 months) | exclude from trends; use as holdout |
| Registration nulls | 697 rows with both count and fuel null | fill 0 (province has none of that type) |
| **Coverage bias** | Highways Dept 149,327 (92.6%) · Rural Roads 9,302 · Expressway 2,678 | **cannot be fixed** → document as a limitation, expose an agency filter in the dashboard |

### ⛔ Deliberately NOT fixed (equally important to document)
- **Don't impute missing injury counts** — inventing casualty numbers corrupts the core measure
- **Don't remove severe outliers** (one crash killed 23 people) — tail events are exactly what
  bankrupts insurers; they must stay
- **Don't try to correct agency coverage bias** — state it as a limitation instead

---

## 7. Data limitations requiring stated assumptions

### 7.1 Exposure is a single snapshot
`stt_car_fuel` is a snapshot at 2026-02-28, but accidents span 7 years.

Implement in `src/rri/features/exposure.py`:
```
vehicles(province, year) ≈ vehicles(province, 2026) × [ national_cum(year) / national_cum(2026) ]
```
where `national_cum` accumulates from `car_registered_new_2565_2569` (national, monthly).
**Assumption:** all provinces grow at a similar rate → must be written up as a limitation.

### 7.2 Exposure excludes commercial transport vehicles
`stt_car_fuel` covers only `ประเภทกฎหมาย = รถยนต์` — no 6/10-wheel trucks, trailers, or large buses.
It still covers **125,429 / 161,307 = 78%** of accident cases (pickups + cars + motorcycles).
Heavy-vehicle groups use raw counts (unnormalized) and must be clearly flagged in the dashboard.

### 7.3 Simulated policy portfolio
| Component | Status |
|---|---|
| Accidents / severity / location / time | 🟢 **100% real** |
| Claim counts | 🟢 derived from real accidents × OIC claim rates |
| Claim cost per case | 🟡 real severity × published OIC reference costs |
| Policy counts / premium / cover class / customer mix | 🔴 **simulated** (calibrated to OIC market shares) |

**Rule:** ML trains on `fact_accident` only. **Never bake a weather→claim relationship into the
simulator** — the model would just rediscover our own assumption.
Requires `docs/simulation_assumptions.md`, attached to the report.

---

## 8. Warehouse design

**2 fact tables (earns bonus points) + 8 dimensions, star schema.**

### `fact_accident` — 100% real
- **Grain:** *one road accident, on one date and time, in one province, on one road section, with a
  first vehicle of one type, under one cause and one reported weather condition*
- Type: transaction fact
- PK `accident_sk` · degenerate dims: `acc_code`, `road_code`, `km_marker`
- FKs: `date_sk`, `time_of_day_sk`, `geography_sk`, `vehicle_sk`, `accident_context_sk`,
  `weather_band_sk`, `agency_sk`
- Additive measures: `accident_count` (=1), `fatality_count`, `serious_injury_count`,
  `minor_injury_count`, `total_injury_count`, `vehicle_involved_count`, `pedestrian_count`,
  `reporting_lag_days`
- ⚠️ `total_injury_count` is derived — never sum it alongside its two components
- ⚠️ `LATITUDE`/`LONGITUDE` are non-additive — never sum or naively average

**Reconciliation — two separate gates (owner-requested framing, see P5). Both must be checked; they
catch different classes of error, so collapsing them to one number loses a check:**

- **Gate 1 — post-Extract (full file, 2019–2026, pre-dedup):** 161,307 accidents · 21,814 fatalities ·
  134,313 total injuries · 248,825 vehicles involved. Proves nothing was lost reading the raw files in.
- **Gate 2 — post-Load (post-dedup, 1 row per `ACC_CODE` group):** 159,605 accidents (1,702 fewer) ·
  21,497 fatalities (317 fewer, exact — casualty counts agree 100% within every duplicate group, so
  this number does not depend on which row of a group is kept). Injury/vehicle totals post-dedup not
  yet computed. Proves dedup removed exactly the intended rows, no more and no less.

Neither gate "is the real target" to the exclusion of the other — Extract and Load are different
pipeline stages with different failure modes to catch. Do not silently collapse to one number.

### `fact_underwriting_monthly`
- **Grain:** *one month × one province × one vehicle group × one cover class* (~92,400 rows)
- Type: periodic snapshot
- Measures: `policies_in_force` (**semi-additive** — cannot sum across months),
  `exposure_vehicle_months`, `earned_premium`, `claim_count`, `incurred_claim_amount`,
  `paid_claim_amount`
- **Derived at query time, never stored:** `loss_ratio`, `frequency`, `severity`, `pure_premium`
  (storing a ratio in a fact table breaks roll-up arithmetic)

### Dimensions (8)
`dim_date` (incl. `is_songkran_7days`, `is_newyear_7days`, `is_long_weekend`, `days_to_next_holiday`) ·
`dim_geography` (SCD2: flood risk, attraction density, R35mm, **EV share**) ·
`dim_vehicle` · `dim_accident_context` · `dim_time_of_day` · `dim_weather_band` ·
`dim_policy_type` · `dim_customer_segment`

**Conformed:** `dim_date` and `dim_geography` are shared by both facts (Kimball bus architecture).
**Never join fact to fact directly** — aggregate each to a common grain first (fan trap).

---

## 9. Business questions (8)

| # | Question |
|---|---|
| BQ1 | Which province × month is riskiest, and is it driven by frequency or severity? |
| BQ2 | How much do rain and flood-prone areas raise risk, and what premium loading does that justify? |
| BQ3 | How much do holiday periods (Songkran, New Year, the "7 dangerous days") raise case counts, and is severity different? |
| BQ4 | Do tourism-dense provinces carry a different risk profile? |
| BQ5 | Which vehicle type loses the most money, and how should portfolio mix change? |
| BQ6 | **Which provinces will spike in the next 6 months?** How much reserve and staffing is needed? |
| BQ7 | If we reprice per the model, how much does the portfolio improve, and what lapse risk does that create? |
| **BQ8** | **Do high-EV-share provinces show higher severity? Is a separate EV rating factor warranted?** |

**Core measures:** Accident Count · Claim Frequency · Severity Index · Fatality Rate ·
Pure Premium · Loss Ratio

---

## 10. ML stack

| Tier | Model | Target |
|---|---|---|
| L0 | Poisson GLM (offset = log exposure) / Gamma GLM | frequency / severity — **the actuarial standard baseline** |
| L1 | LightGBM (`objective=poisson` / `gamma`) | frequency / severity |
| L1 | LightGBM classifier | severity class (152,569 record-level rows) |
| L2 | **MLP + entity embeddings** (PyTorch) | severity class — justified by high cardinality: 120 causes, 152 crash types, **thousands of road codes** |
| L3 | **Global GRU** (weights shared across 77 provinces) | 6-month-ahead case counts, using **rainfall forecast lead 0–5 as a known-future covariate** |
| L4 | Claude API + **code-based verifier** | insight narrative, every number checked against the DW (0.5% tolerance) |

**Core principle:** `Pure Premium = Frequency × Severity`. Frequency is a count (Poisson-like,
zero-heavy); severity is money (Gamma, right-skewed with a long tail). Different distributions →
**never use a single model to predict total claim cost.**

**Metrics:** frequency → Poisson deviance (not RMSE) · severity → Gamma deviance / MAE ·
risk ranking → **Gini / lift curve** · classification → macro-F1 + confusion matrix

**Expectation:** LightGBM will probably beat the neural net. **That is not a failure** — the report
should say "we tested three approaches and chose on evidence."

---

## 11. Computer-engineering rigor checklist

> A pretty dashboard alone reads as data-analyst work. These are what make it engineering.

- [ ] Git with conventional commits (`feat:`, `fix:`, `test:`, `docs:`)
- [ ] Properly separated package modules with explicit public APIs
- [ ] **Type hints everywhere**, `mypy --strict` clean
- [ ] **Pytest covering every cleaning rule**, ≥80% coverage in `src/rri/cleaning/`
- [ ] **Data contracts declared as code** (pandera), not eyeballed in a notebook
- [ ] Structured logging (`structlog`) with a `run_id` on every pipeline run
- [ ] Typer CLI: `python -m rri run --phase etl --mode full|incremental`
- [ ] Makefile: `setup / test / lint / etl / warehouse / dashboard`
- [ ] **GitHub Actions CI** on every push (screenshot the green badge for the report)
- [ ] pre-commit hooks
- [ ] **Idempotency test** — running the pipeline twice produces identical output hashes
- [ ] **Incremental load** via watermark
- [ ] Notebooks contain zero business logic

---

## 12. Status (update at the end of every phase)

| Phase | Work | Status | Notes |
|---|---|---|---|
| 1 | Scaffold + holiday backfill + province master + profiling | ⬜ not started | |
| 2 | ETL: extract + clean (17 problems) | ⬜ | |
| 3 | Build warehouse + DQ gates | ⬜ | |
| 4 | Simulated policy portfolio | ⬜ | |
| 5 | Features + GLM / LightGBM baseline | ⬜ | |
| 6 | Neural net + forecast | ⬜ | |
| 7 | Streamlit dashboard (Thai UI) | ⬜ | |
| 8 | AI insight + verifier | ⬜ | |
| 9 | Report + video + submission | ⬜ | |

**Decision log**

| Date | Topic | Decision | Reason |
|---|---|---|---|
| 2026-08-11 | Database | DuckDB + MySQL export | OLAP, single file, easy to share |
| 2026-08-11 | Dashboard | Streamlit, Thai UI | can call ML and Claude API live |
| 2026-08-11 | Dropped data | `banks.json`, `tat_*_clean.csv` | no join key / exact duplicate |
| 2026-08-11 | Portfolio | Simulated, but claims derived from real accidents | avoids circular reasoning |
