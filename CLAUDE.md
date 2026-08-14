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

## 6. Known data-quality problems (19, all verified against the real files)

Full investigation write-ups, corrections, and reasoning for every problem below are in
[docs/data_quality_findings.md](docs/data_quality_findings.md) — **read that before touching any
related Phase 2 code.** This section keeps only a one-line summary per problem plus the numbers
Phase 2 code implements directly against (reconciliation gates, per-agency thresholds, the P1
format-per-file table).

Fix in this order: P1 → P2 → P3 → P4 → P5 → P6 → P7.

| # | Problem | Fix direction |
|---|---|---|
| P1 | Three date formats mixed in one column, detected **per file** not per row | mandatory safety net below; per-agency thresholds, not one global number |
| P2 | Two time formats — clock string vs. Excel day fraction; `0:00` ×219 may be "unknown" | parse float in [0,1) as fraction × 24; check `0:00` against report time |
| P3 | Schema drift — 4 genuine twin-column pairs; 5th pair (`บริเวณที่เกิดเหตุ`/`บริเวณที่เกิดเหตุอื่น`) is **not** a twin, still unresolved | coalesce the 4 (zero row-overlap, verified); Phase 2 investigates the 5th |
| P4 | Inconsistent categories — 120/152/89 distinct values across 3 columns | committed mapping tables; never fuzzy-match alone |
| P5 | Duplicates — 82 exact rows + 1,702 excess rows from 1,641 `ACC_CODE` groups; `ACC_CODE` null 17.3% | surrogate key; harmonize taxonomy **before** dedup (order matters, see findings doc) |
| P6 | Province names mismatched across sources (flood 75, TAT 78 vs. 77 canonical) | resolved — `config/mappings/`, 100% match verified (Phuket/Ranong → Unknown Member for flood; TAT junk value documented) |
| P7.1 | Coordinates outside Thailand, 10 rows | null + `is_geo_valid` flag |
| P7.2 | BOM present on 5 of 6 CSVs, not all (corrected from an earlier wrong claim) | detect BOM per file, don't hard-code `utf-8-sig` |
| P7.3 | Header typo `Foreast_Month` | rename map |
| P7.4 | Buddhist vs. Gregorian year (registration file) | `CE = BE − 543` |
| P7.5 | Rainfall has 6 rows per province-month | **not duplicates** — pivot to `rain_lead0..5` |
| P7.6 | TAT coordinates embedded in a string | split, cast, validate bounds |
| P7.7 | ID overflow — `ATT_ID` 17 digits, float/int mix | read everything as string first |
| P7.8 | Injury totals mismatch, 9,872 rows (6.1%) — 100% in `accident2019.csv`, breakdown never captured that year | flag, don't impute — **policy: two severity indices, see §9 Measures** |
| P7.9 | `วันที่รายงาน` null in 78 rows (74 in incomplete `accident2026.csv`, 4 in `accident2023.csv`) | contract asserts non-null and fails per-row; don't loosen to pass |
| P7.10 | `วันที่รายงาน` = literal `"//"` (not null, not a date) in 4 rows, `accident2022.csv` | treat as unparseable, same as null |
| P7.11 | Incomplete periods — 2026 has ~4 months of data | exclude from trend analysis; use as holdout |
| P7.12 | Registration nulls — 697 rows, both count and fuel null | fill 0 (province has none of that type) |
| P7.13 | Agency coverage bias — Highways 92.6% / Rural Roads 5.8% / Expressway 1.6% | cannot be fixed → document as a limitation, expose an agency filter in the dashboard |

### Deliberately NOT fixed
- **Don't impute missing injury counts** — inventing casualty numbers corrupts the core measure
- **Don't remove severe outliers** (one crash killed 23 people) — tail events are exactly what
  bankrupts insurers; they must stay
- **Don't try to correct agency coverage bias** — state it as a limitation instead

### Numbers Phase 2 code implements directly against

**P1 per-file date format** (detect per `source_file`, not per row — see findings doc for the
"zero DMY evidence" proof this is sound):

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

**Reconciliation gates** (two separate checks, never collapsed to one number — see P5 in the
findings doc for why): **Gate 1** (post-Extract, pre-dedup) = 161,307 accidents · 21,814 fatalities.
**Gate 2** (post-Load, post-dedup) = 159,605 accidents (1,702 fewer) · 21,497 fatalities (317 fewer).

**Per-agency report-lag thresholds** (safety net #2's upper bound — p99 of each agency's own
distribution, not one global number; see P1 in the findings doc for why one number is wrong):

| Agency | n | p99 threshold (days) |
|---|---|---|
| กรมทางหลวง (Highways, 92.6%) | 149,318 | 393 |
| กรมทางหลวงชนบท (Rural Roads) | 9,302 | 235 |
| การทางพิเศษแห่งประเทศไทย (Expressway) | 2,604 | 1,219 |

**Mandatory safety net (P1, 4 layers):**
1. Parsed year must equal the existing `ปีที่เกิดเหตุ` column
2. `accident_date ≤ report_date`, gap ≤ the row's own agency's p99 threshold (above)
3. Day-of-month histogram must be flat
4a. Duplicate-group rows sharing an identical raw date string must parse to the same date

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
- `has_injury_breakdown` (boolean) — **false for all `accident2019.csv` rows** (`serious_injury_count`/
  `minor_injury_count` never captured that year, see P7/§6), true elsewhere. Makes the 2019 gap
  explicit on every row rather than letting a query silently average across it. Drives which severity
  index is computable — see §9 Measures.
- ⚠️ `total_injury_count` is derived — never sum it alongside its two components
- ⚠️ `LATITUDE`/`LONGITUDE` are non-additive — never sum or naively average

**Two gate systems, don't conflate the names:**
- **DQ Gates A/B** — *when* a check runs. Gate A = row-level validity at ingest (e.g. safety net #2's
  per-agency report-lag threshold, §6). Gate B = completeness/referential checks after transform/load
  (e.g. the 83-row unparsed-row reconciliation, §6). **Gate B rule: `severity_index_fine` must never be
  computed for a row where `has_injury_breakdown` is false** — a pipeline check, not a query-time filter.
- **Reconciliation Gates 1/2** — *row-count totals* at two pipeline stages, never collapsed to one
  number (different failure modes: ingest loss vs. over/under-aggressive dedup):
  - **Gate 1** (post-Extract, pre-dedup, full 2019–2026 file): 161,307 accidents · 21,814 fatalities ·
    134,313 total injuries · 248,825 vehicles involved.
  - **Gate 2** (post-Load, post-dedup, 1 row per `ACC_CODE` group): 159,605 accidents (1,702 fewer) ·
    21,497 fatalities (317 fewer, exact — casualty counts agree 100% within every duplicate group).
    Injury/vehicle totals post-dedup not yet computed.

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

### Measures

**Accident Count · Claim Frequency · Fatality Rate · Pure Premium · Loss Ratio** — as usual.

**Severity Index — two versions, not one (decided 2026-08-14).** P7/§6 found `accident2019.csv`
never captured the serious/minor injury breakdown — only fatalities and the total. A single
three-tier severity formula is uncomputable for all 19,194 rows of 2019. Rather than drop the year:

- **`severity_index_coarse`** = `(fatalities × W_f + total_injuries × W_i) / accidents` —
  computable for **all years, 2019–2026**. **Primary measure for all cross-year work** (BQ1, BQ3,
  BQ6) — anything comparing across years uses this one.
- **`severity_index_fine`** = the original three-tier formula (fatalities / serious injuries / minor
  injuries weighted separately) — computable **2020–2026 only**. Use where the extra resolution
  matters, and **always state the restricted window** when reporting it.
  (Weight values for both — `W_f`, `W_i`, and the fine formula's per-tier weights — are calibrated
  against OIC reference severity costs in Phase 5, §7.3; not invented here.)
- **Why keep 2019 rather than drop it:** it is the project's only pre-COVID year. Dropping it would
  leave every remaining baseline (2020 on) contaminated by pandemic-era travel/reporting effects with
  nothing un-affected to compare against.
- **`has_injury_breakdown`** (boolean, `fact_accident`, §8) makes the gap explicit on every row.
  **Gate B**: `severity_index_fine` is never computed where this is false (§8).
- **Phase 5 task**: compute both indices over the 2020–2025 window (where both are available) and
  report their correlation. This quantifies how much resolution `severity_index_coarse` loses —
  turning "we use two measures" from a compromise into a documented, measured methodological choice.

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
| 1 | Scaffold + holiday backfill + province master + profiling + data contracts | ✅ **approved** | Tasks 0-5 complete: git/scaffold, holiday backfill, province master (77, 100% matched), full raw-file profiling (`notebooks/01_data_profiling.ipynb`), `src/rri/contracts/raw.py` (pandera). 19 P1-P7 problems verified against real data (grew from 17 as this phase's own verification passes surfaced 2 new ones — the `"//"` report-date placeholder and the BOM claim error); several corrected from their original draft (P1 safety net #4, P5 root cause, P3 pair 5, the 90-day→per-agency threshold). Full write-ups moved to `docs/data_quality_findings.md`; §6 is now a summary. Nothing cleaned yet — Phase 2 starts from here. |
| 2 | ETL: extract + clean (17 problems) | ✅ **approved** | `notebooks/02_data_cleaning.ipynb`, all logic in `src/rri/cleaning/` (`extract.py`, `taxonomy.py`, `dates.py`, `dedup.py`, `fields.py`), every rule unit-tested (99% coverage on `src/rri/cleaning/`, 345 tests total). Step 1: parquet extract with BOM-aware encoding, header rename, the 4 proven twin-column pairs coalesced (8,738 rows each), Buddhist→CE year, rainfall pivoted to `rain_lead0..5`. Step 2: committed taxonomy tables (`config/mappings/{cause,crash_type,road_geometry}_groups.csv`) applied before dedup — `บริเวณที่เกิดเหตุ` duplicate-pair conflict rate dropped 74.1%→3.2% (1,216→52 of 1,641 groups) as P5 predicted. Step 3 (owner checkpoint): per-file date parsing + 4-layer safety net; the 448 P1 transposed-date pairs resolved via a **two-stage discriminator** (hard chronological filter against every report date in the group, THEN likelihood breaks ties) after the owner caught a bug in the first draft (a single-stage likelihood-only version silently could have picked a logically-impossible post-report date) — final split 321 resolved / 127 ambiguous (112 both-hard-valid + 15 neither), `date_certainty` column added, 336/448 groups had a candidate eliminated for violating accident≤report. Step 4: dedup by `ACC_CODE` (certain > ambiguous → fewer nulls → `_row_hash`), casualty agreement asserted before collapsing; 1,641 groups → 159,605 rows, fatalities 21,814→21,497, exact match. Step 5: coordinates (5 lat / 8 lon nulled post-dedup, `is_geo_valid`), time format (`accident_time`, 0 unparseable; the 215 literal `"0:00"` values investigated against report time — no placeholder evidence found, kept as real midnight + flagged via `time_is_literal_midnight`), `has_injury_breakdown=false` for all 19,194 `accident2019.csv` rows (no split imputed). Step 6: Gate 1 (161,307/21,814) and Gate 2 (159,605/21,497) both pass exactly; raw-file contract re-run reconfirms Phase 1's baseline (93 known schema failures = 78 null report-date + 5 lat + 10 lon, 0 province orphans, 9,872 injury mismatches, all unchanged); row-count conservation 161,307 = 159,605 + 1,702 confirmed; the P1/P7 83-row date-parseability gap re-confirmed exactly (82 null `report_date` + 1 null `accident_date`) on cleaned data; before/after day-of-month histogram saved to `reports/day_of_month_histogram_before_after.png`; `reports/cleaning_log.csv` written; `data/00_raw/` confirmed untouched. |
| 3 | Build warehouse + DQ gates | ⬜ | |
| 4 | Simulated policy portfolio | ⬜ | |
| 5 | Features + GLM / LightGBM baseline | ⬜ | Includes the severity-index correlation task (§9 Measures): compute `severity_index_coarse` and `severity_index_fine` over 2020–2025 and report their correlation |
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
| 2026-08-13 | Safety net #2 threshold | Per-agency p99 (393/235/1,219 days for Highways/Rural Roads/Expressway), not a single global number, not the originally-drafted 90 | 90 was invented at design time (would reject 28.6% of rows); a single global 398 fixed that but conflated two different jobs (Gate A validity wants loose, discrimination wants tight) and under/over-flagged minority agencies (Rural Roads 0.10% vs. Expressway 10.87% under global 398) |
| 2026-08-13 | Data contracts | `src/rri/contracts/raw.py` asserts only proven facts (gates, twin columns, per-file date format, 77 provinces); no date-format shape assertion on `วันที่เกิดเหตุ`/`วันที่รายงาน` | 3 date formats coexist by design (P1); contract failures (coords, injury totals, 78 null report-dates) are the intended deliverable, not bugs to fix |
| 2026-08-14 | Severity Index / 2019 handling | Two measures — `severity_index_coarse` (all years, primary) and `severity_index_fine` (2020-2026 only) — keep 2019, don't drop it | `accident2019.csv` has no serious/minor breakdown (P7); 2019 is the only pre-COVID year, so dropping it contaminates every remaining baseline |
| 2026-08-14 | Step 3 transposed-date discriminator | Two-stage design: hard chronological filter (candidate must satisfy `accident_date ≤ report_date` for EVERY report date found anywhere in the duplicate group, not just its own row's) eliminates impossible candidates first; only among survivors does likelihood-vs-agency-median break a tie, and a 2-survivor tie stays `date_certainty=ambiguous` even though a likelihood winner is recorded | A single-stage likelihood-only version resolved 448/448 with 0 ambiguous — the owner caught that this was picking a winner by statistical preference alone, with nothing stopping it from selecting a logically-impossible post-report date if likelihood happened to favor it (concrete case: ACC_CODE 7543996). The group-wide (not row-local) report-date check matters because the report-date field can independently also be transposed |
