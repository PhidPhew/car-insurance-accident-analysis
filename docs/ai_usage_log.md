# AI Usage Log

A running record of cases where working with an AI assistant surfaced a
discrepancy, required a judgment call, or otherwise went beyond "type what
CLAUDE.md says" — kept as evidence for the assignment's AI-usage disclosure.

---

## 2026-08-12 — CLAUDE.md P7 out-of-bounds coordinate count was wrong (15 vs. 10)

**Phase / task:** Phase 1, Task 1 (fixture generation script)

**What CLAUDE.md said (before this entry):** §6 P7 listed "lat outside
5–21: 5 rows; lon outside 97–106: 10 rows" — read at face value, an
uncritical implementation would assume 5 + 10 = 15 distinct affected rows.

**What was found:** While writing `scripts/make_fixtures.py`, the fixture
generator needed to select *every* out-of-bounds coordinate row in the
accidents file (so the fixture — and any future `is_geo_valid` cleaning
rule test — actually exercises the defect). Computing the union of
lat-invalid and lon-invalid row indices against the real
`thailand_accidents_2019_2026_combined.csv` gave **10 rows, not 15**.

Verification:

```python
lat_bad = {i for i, r in rows if lat invalid}   # 5 rows
lon_bad = {i for i, r in rows if lon invalid}    # 10 rows
union = lat_bad | lon_bad                        # 10 rows
```

`lat_bad ⊆ lon_bad`: the same 5 rows are invalid on both axes, so the two
counts in CLAUDE.md were per-axis, not disjoint. Reported to the project
owner rather than silently using either 15 or 10.

**Resolution:** Project owner independently confirmed 10 is correct — the
original P7 entry stated two per-axis counts without specifying overlap,
which was an ambiguity in the spec, not an implementation error. CLAUDE.md
§6 P7 updated to: "lat outside 5–21 in 5 rows, lon outside 97–106 in 10
rows, overlapping in 5 rows → 10 distinct rows affected."

**Why this matters for the assignment:** N4 requires every data fix to log
before/after counts, and the Data Quality section (20% of the grade)
expects exactly this kind of "we verified the number, not just quoted it"
behavior. This is also the reason `scripts/make_fixtures.py` computes the
out-of-bounds row set dynamically from the source file at generation time
rather than hard-coding either 15 or 10 — the fixture stays correct even
if the source file changes.

---

## 2026-08-12 — Task 2 holiday backfill: a validation rule and a data source both needed a second look

**Phase / task:** Phase 1, Task 2 (`src/rri/io/holidays.py`, `scripts/backfill_holidays.py`)

**Finding 1 — the "Songkran must be 13-15 April" validation is wrong for 2020.**
Task 2's spec says to sanity-check that Songkran falls on 13-15 April every
year, "if not, the fetch is wrong." Fetching 2020 via the `holidays`
Python package showed April 13-15, 2020 were *not* public holidays that
year. This is real history, not a bug: the Thai government postponed
Songkran nationwide by Cabinet Resolution due to COVID-19 and rescheduled
substitute holidays to 27 Jul and 4/7 Sep 2020. Verified against the
package's own "(in lieu)" entries for those dates before accepting it.
Reported to the project owner rather than silently forcing fake April
dates into the data to satisfy the validation rule, or silently treating
the honest historical result as a "fetch failure." Resolution (owner
confirmed): keep the real 2020 data; the validation now special-cases
2020 as a documented exception (`KNOWN_SONGKRAN_EXCEPTIONS` in
`src/rri/io/holidays.py`) instead of a hard failure.

**Finding 2 — the `holidays` package's "public" category is broader than
what `public_holidays_all.json` actually represents.** Cross-checking
freshly-fetched 2025-2026 dates against the real DGA-sourced file (a
validation step Task 2 explicitly asks for) showed the package includes
National Children's Day and National Labor Day, which the real government
file does not — those are private-sector/observance days, not part of the
government-office holiday calendar the real file tracks. Left unfixed,
2019-2024 would have used a different, broader definition of "public
holiday" than 2025-2026, which would silently corrupt any year-over-year
comparison built on top of it later (e.g. BQ3's holiday-period case-count
analysis). Resolution (owner confirmed): exclude these two categories from
the backfill so all 8 years share one consistent scope.

**Why this matters for the assignment:** both findings came from actually
running the documented validation checks against real fetched data instead
of assuming the fetch was either "correct" or "wrong" at face value — the
first was a validation rule that needed an exception, the second was a
source that needed narrowing. Neither would have surfaced without the
cross-source and historical sanity checks Task 2 asked for.

---

## 2026-08-12 — Both Task 2 findings needed deeper handling than a log entry

**Phase / task:** Phase 1, Task 2 follow-up (`src/rri/io/holidays.py`)

The two findings logged above (2020 Songkran postponement, Labour Day
scope mismatch) were initially just logged as documented exceptions. The
project owner pushed back on both — correctly: logging a discrepancy
isn't the same as modeling its actual effect. Both were escalated from a
log entry to a schema change.

**Decision 1 — 2020 Songkran needed a `covid_disruption_level` dimension,
not just an exception note.** The initial fix treated 2020 as "Songkran on
different dates." The owner pointed out this understates what actually
happened: April 2020 was under a nationwide curfew with interprovincial
travel restricted, so April traffic that year isn't a shifted-holiday
signal, it's a suppressed-travel signal — averaging it into the normal
Songkran baseline would make BQ3 (holiday-period risk analysis) understate
the festival effect.

Added `covid_disruption_level` (none / partial / severe) to the daily
flags output, covering roughly Apr-Jun 2020. Every boundary is sourced
from a specific, citable government action rather than guessed — the
instruction was explicit that an uncitable date range should stop and
ask, not get invented:

- 2020-03-17 — Cabinet announces Songkran (13-15 Apr) postponed
- 2020-03-26 — Nationwide State of Emergency takes effect
- 2020-04-03 — Nationwide curfew begins (22:00-04:00) → **severe** starts
- 2020-05-03 — Phase 1 lockdown easing begins → **severe** ends, **partial** (phased reopening) begins
- 2020-06-15 — Nationwide curfew lifted → back to **none**

(Citations for each are inline as comments in `src/rri/io/holidays.py`
next to the constant definitions.) BQ3's analysis can now exclude the
severe-disruption window explicitly instead of silently blending it into
"normal" Songkran data.

**Decision 2 — Labour Day should be reclassified, not dropped.** The
initial fix excluded Labour Day from the backfill entirely, on the
reasoning that it's absent from the DGA government calendar
(`public_holidays_all.json`). The owner's correction: absence from that
one source doesn't mean absence of effect — most of the Thai workforce is
private-sector, so 1 May is a real non-working day that changes traffic
patterns, which is what the project actually measures risk against.
Dropping it silently would also have created a worse problem than
including it: `thai_bank_holidays_all.json` already has Labour Day for
2025-2026 (banks close for it), so the un-reclassified version had Labour
Day appearing for 2 of 8 years and silently vanishing for the other 6 — an
inconsistency across years that's more damaging than an extra column.

Fix: kept `is_public_holiday` scoped exactly to the DGA calendar (no
change there), and added a second, independent flag,
`is_private_sector_holiday`, sourced from the same `holidays` package data
including its in-lieu substitute dates for years Labour Day falls on a
weekend (verified: 2021-05-01 was a Saturday → substitute 2021-05-03;
2022-05-01 was a Sunday → substitute 2022-05-02). National Children's Day
stays excluded entirely — it's a symbolic observance with no mandated
non-working status, unlike Labour Day.

**Why this matters for the assignment:** the underlying lesson is that
"log the discrepancy" is necessary but not sufficient — the log entry has
to be checked against whether it changes what a downstream business
question (BQ3 here) can actually conclude. Both fixes turned a footnote
into a modeled dimension the dashboard/analysis can filter or split on.

---

## 2026-08-12 — Task 3 province master: two things needed real investigation, not a guess

**Phase / task:** Phase 1, Task 3 (`scripts/build_province_master.py`,
`scripts/build_tat_province_crosswalk.py`)

**Region classification has no single "official" answer in Thailand, so it
was researched rather than recalled from memory.** CLAUDE.md asks for a
"region (6 regions)" column, but Thailand has multiple co-existing regional
classifications that disagree on several provinces — the National
Geographic Committee's 1977 six-region system splits out a separate West
region, while the Meteorological Department's five-region system folds
those provinces into Central and North instead. An initial English-language
source (Wikipedia) only resolved 68 of 77 provinces and left Phetchaburi,
Prachuap Khiri Khan, and Tak ambiguous between Central/West/North. A second,
Thai-language search gave per-region province *counts* for the NGC system
(North 9, Central 21, Northeast 19-20, East 7, West 5, South 14) that pinned
West down to exactly 5 provinces (Tak, Kanchanaburi, Ratchaburi,
Phetchaburi, Prachuap Khiri Khan) — confirmed correct by checking the
resulting assignment reconciles to exactly 77 against this project's own
verified province codes, with zero left over and zero double-counted. Full
source citations are inline in `scripts/build_province_master.py`.

**TAT's 78th "province" is junk data, not a real place, and required
tracing 78 IDs back to source records to find that out.** TAT's
`ATT_PROVINCE_ID` uses its own internal numbering ("1001".."1077") with no
province name field anywhere in the file, so even the 77 legitimate values
needed a crosswalk before they could be used at all — not just the 78th.
Built one from the data itself: Thai postal code prefixes were verified to
line up with the project's standard province geocodes (postcode 81xxx
co-occurs with `ATT_PROVINCE_ID` 1002, and geocode 81 = Krabi; confirmed on
several more before trusting the method), so each ID was resolved to a
province via the mode of its records' postcode prefix. That resolved 75 of
77 automatically; the remaining 2 had no usable postcode at all and were
identified by web-searching their attraction names (Wat Nong Pling →
Kamphaeng Phet, Nam Tok Nang Non in Kudchum district → Yasothon) and
hand-confirmed rather than guessed from coordinates alone. The 78th distinct
value turned out to be the literal string `"1"`, attached to 13 records
whose names are literally `"test"`, `"Hello world!"` (×3), and similar —
an unset/default sentinel in TAT's own database, not a real 78th province.
It is excluded from the crosswalk and reported as such, not silently
dropped or miscounted as a real place.

**Why this matters for the assignment:** both findings needed the same
discipline as the P7 and Songkran cases above — treat a source number (77
provinces expected, 78 found) as something to trace to its root cause
rather than either dismissing it as noise or accepting it as a real 78th
province. The region classification could have been guessed from general
knowledge and looked plausible without ever being wrong in an obviously
checkable way; sourcing it and reconciling it against the real data (77 in,
77 out, no leftovers) is what makes it trustworthy rather than merely
confident-sounding.
