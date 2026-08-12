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

**Mandatory 3-layer safety net:**
1. Parsed year **must equal the existing `ปีที่เกิดเหตุ` column** — a single mismatch fails the pipeline
2. `accident_date ≤ report_date`, and the gap must be ≤ 90 days
3. Day-of-month histogram must be flat — a wrong parse shows a spike on days 1–12

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
random — different years used different column names. **Coalesce them.**

### 🟠 P4 — Inconsistent categories
`มูลเหตุสันนิษฐาน` has 120 distinct values, should collapse to ~20.
`ลักษณะการเกิดเหตุ` 152 → ~15. `บริเวณที่เกิดเหตุ` 89 → ~12.
Example: `ขับรถเร็วเกินอัตรากำหนด` (99,967) and `ขับรถเร็วเกินอัตราที่กำหนด` (14,636) are the
same thing — combined 114,603 (71% of all cases).
Use committed mapping tables. **Never rely on automatic fuzzy matching alone** — it will merge
"speeding" with "reckless driving", which are different causes.

### 🟠 P5 — Duplicates
- 82 fully identical rows → safe to drop
- 1,702 duplicate `ACC_CODE` → ⚠️ investigate first; across different years the code is reused,
  so the real key is `(ACC_CODE, year)`
- `ACC_CODE` is null in 17.3% of rows → **cannot be the primary key**; use a surrogate key

### 🟠 P6 — Province names don't match across sources
Accidents `ลพบุรี` · flood `จ.นนทบุรี` (prefix) · R35mm `Bangkok` (English) · registration `กระบี่`
Province counts: accidents 77 ✅ · rainfall 77 ✅ · registration 77 ✅ · **flood 75** ❌ · **TAT 78** ❌

### 🟡 P7 — Remaining
| Problem | Evidence | Fix |
|---|---|---|
| Coordinates outside Thailand | lat outside 5–21: 5 rows; lon outside 97–106: 10 rows | null + `is_geo_valid` flag |
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

**Reconciliation targets (full file, 2019–2026):** 161,307 accidents · 21,814 fatalities ·
134,313 total injuries · 248,825 vehicles involved

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