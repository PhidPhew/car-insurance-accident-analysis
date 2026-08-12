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

---

## 2026-08-12 — TAT crosswalk had a real bug; found by an independent check the owner asked for

**Phase / task:** Phase 1, Task 3 follow-up
(`scripts/build_tat_province_crosswalk.py`, `scripts/validate_tat_crosswalk.py`)

**What was asked:** the project owner asked for the TAT crosswalk to be
checked specifically against the Bangkok/Samut Prakan pair — Samut Prakan's
real postcodes (10270, 10280, 10290, 10540, 10560, 10570) fall inside
Bangkok's 10xxx postal block, which breaks any method assuming a postcode's
leading digits equal the province code — and then validated independently
(attraction-count plausibility, a total-record reconciliation, and 5 named
spot-checks), explicitly *not* with more postcode logic.

**What was found:** the original crosswalk (built from the leading 2 digits
of each `ATT_PROVINCE_ID`'s postcodes) had exactly the bug the owner
predicted, and it cascaded: `ATT_PROVINCE_ID` 1001 and 1059 both resolved to
Bangkok, while Saraburi ended up with no id mapped to it at all. Because
Samut Prakan has no postal block of its own, every province from Samut
Prakan through Saraburi (geocodes 11-19: Samut Prakan, Nonthaburi, Pathum
Thani, Ayutthaya, Ang Thong, Lopburi, Sing Buri, Chai Nat, Saraburi) had its
postcode prefix shifted one behind its actual geocode. This was found by
checking for duplicate `province_id` assignments across the crosswalk output
— a check with no postcode logic in it at all, just "does any province get
claimed twice."

**Fix:** rebuilt the crosswalk against a real postcode→province reference
(kongvut/thai-province-data, an open, actively-maintained dataset — spot
checked against known facts, e.g. Ayutthaya postcode 13000, Ang Thong
14000, before trusting it) instead of assuming a 2-digit prefix equals the
geocode. All 77 provinces now resolve to exactly one `ATT_PROVINCE_ID` with
zero duplicates and zero gaps.

**Independent validation (all passed after the fix):**
- Record count reconciles exactly: 1,718 matched + 13 junk = 1,731, the
  file's real total.
- 5 named landmarks (Yaowarat, Doi Suthep, Phi Phi Islands, an
  Ayutthaya-named attraction, Patong Beach) all resolve to the correct
  province. Notable miss during this check: Wat Phra Kaew / the Grand
  Palace isn't in the dataset under any name tried — logged as a coverage
  observation about the source, not a crosswalk defect.
- Top 10 / bottom 10 attraction counts by province are broadly plausible
  (Chiang Mai #1 at 256; small interior provinces at the bottom), with one
  numbers-worth-eyeballing flag: Prachuap Khiri Khan is #2 at 107, ahead of
  both Bangkok (47) and Phuket (50) — reported as-is rather than adjusted,
  since every id in the affected block was independently re-verified and
  the number is plausibly just uneven submission volume across TAT's
  regional offices, not a mis-mapping.

Added `tests/unit/test_tat_crosswalk.py` as a permanent regression guard
(no two `ATT_PROVINCE_ID`s may resolve to the same province; every one of
the 77 provinces must have exactly one).

**Also quantified per the owner's request:** 13 of 1,731 TAT records (0.75%)
carry the junk `"1"` province value, with record names literally including
`"test"` and `"Hello world!"` — now stated explicitly in
`reports/province_matching_report.html` as a concrete data-quality example.

**Why this matters for the assignment:** this is the clearest example yet of
why "validate independently, not by re-running the same method" matters —
the postcode-mode heuristic was internally confident (every affected id had
a >90% consistent postcode-prefix mode) while being systematically wrong.
Only a check with a different failure mode (duplicate/missing province
detection, then separately, spot-checks by name) could catch it. The bug
would have silently misattributed Samut Prakan's, Nonthaburi's, and seven
other provinces' entire attraction-density signal to their geocode
neighbor — directly relevant to BQ4 (tourism-dense province risk profile).

---

## 2026-08-12 — Task 4 profiling found P5's stated root cause is wrong

**Phase / task:** Phase 1, Task 4 (`src/rri/utils/profiling.py`,
`notebooks/01_data_profiling.ipynb`)

CLAUDE.md P5 said duplicate `ACC_CODE` values happen because the code is
reused **across different years**, and that the real key is therefore
`(ACC_CODE, year)`. Profiling this directly (grouping by `ACC_CODE` alone
vs. by `(ACC_CODE, year)`) showed **both produce the identical 1,641
duplicate groups** — meaning zero `ACC_CODE` values actually span more than
one year, and the composite key doesn't resolve the duplication at all.
Random-sampling 5 duplicate groups (seed 42) showed what's actually
happening: rows share `ACC_CODE`, year, province, road, KM, and date/time
exactly, but disagree on the free-text description columns
(`บริเวณที่เกิดเหตุ`, `ลักษณะการเกิดเหตุ`, `มูลเหตุสันนิษฐาน` — populated on
one row, null or differently worded on another). This is the same
real-world accident entered more than once with inconsistent
characterization, not a key-design problem. Updated CLAUDE.md P5 with the
correction rather than leaving the disproven explanation in place; the
"1,702 duplicate ACC_CODE" number itself was correct (it's the excess-row
count, 3,343 total rows in duplicate groups minus 1,641 groups), only the
proposed cause and fix were wrong.

Two smaller findings from the same profiling pass, also added to CLAUDE.md:
P3's 5th twin-column pair (`บริเวณที่เกิดเหตุ` / `บริเวณที่เกิดเหตุอื่น`) is
not cleanly mutually exclusive like the other 4 — both are populated
simultaneously on 7,606 rows (4.7%), so it needs a precedence rule, not a
plain coalesce; and a single row in `accident2020.csv` has an Excel-serial
date despite the file otherwise being pure MM/DD/YYYY text (immaterial at
1/161,307 rows, but a concrete example of why the 3-layer safety net is
row-level, not just file-level).

**Why this matters for the assignment:** this is the deepest correction so
far because it didn't just fix a number, it invalidated the *proposed fix*
for a documented problem. A cleaning rule built on "the key is
`(ACC_CODE, year)`" would have shipped, passed its own logic, and left the
real duplication (conflicting descriptions of the same accident) completely
unaddressed — the kind of bug that's invisible until someone asks why two
rows for the same crash tell different stories.

---

## 2026-08-12 — Follow-up: neither P5 nor P3-pair-5 was actually resolved, just diagnosed

**Phase / task:** Phase 1, Task 4 deep follow-up (no code changes — pure
investigation, `CLAUDE.md` §6/§8 updated with facts, no rule chosen)

The project owner's response to the P5/P3 findings above was, correctly,
that identifying a wrong explanation isn't the same as having a right one —
both needed to be resolved before Task 5, not just flagged.

**P5 (duplicate ACC_CODE), exhaustive checks across all 1,641 groups (not
sampled this time):**
- Agency: 100% same-agency. Rules out the "legitimate cross-agency
  double-report at a highway/rural-road junction" hypothesis the owner
  raised — every duplicate is one agency's own record twice.
- Casualty counts: 100% agree across all four columns, and province/road/
  KM/year are also 100% identical within every group. This matters beyond
  P5 itself: it means the fatality reconciliation number is exact and
  unambiguous under deduplication regardless of which row in a group is
  eventually kept — only the free-text tie-break is still open.
- Computed both reconciliation numbers as instructed rather than picking
  one: pre-dedup 21,814 fatalities (161,307 rows, the current §8 baseline)
  vs. post-dedup 21,497 (159,605 rows, exact, 317 fewer). Recorded both in
  CLAUDE.md §8 with an explicit note that the choice between them is still
  open — did not default to either.
- Investigated *why* free-text columns conflict instead of just noting
  that they do: of the 3 description columns, conflict rates vary from 4%
  (`ลักษณะการเกิดเหตุ` — mostly a null-vs-populated gap, "keep the more
  complete row" would work) to 74% (`บริเวณที่เกิดเหตุ` — mostly non-null
  vs. non-null disagreement, where "more complete" is meaningless). A
  single blanket tie-break rule would silently do the wrong thing for at
  least one of these columns.
- Followed up on *why* dates differ in 448 groups: every single one turned
  out to be a pure day/month digit transposition under the file's own
  established format (verified using the correct per-file DMY/MDY/serial
  mapping from P1 — my first pass at this used a naive try-both-formats
  parser and produced nonsense day-difference numbers before I caught it
  and redid it properly). This is a specific, nameable transcription-error
  pattern, not unexplained noise, and it supports rather than undermines
  the "same event entered twice" reading of the whole P5 problem.

**P3 pair 5, both proposed explanations checked directly against data and
both refuted:**
- "Old name phased out, new name replaces it" (the original CLAUDE.md
  claim) — false: the "old" column is populated in every year 2019–2026,
  including 90% of 2026 itself. It was never phased out.
- "New column is a free-text detail field used when the primary column is
  `อื่นๆ`" (the owner's alternative hypothesis) — also false: 0.0% of the
  7,606 rows where both are populated have the primary column set to
  `อื่นๆ`. Checked directly rather than assumed correct because it sounded
  plausible.
- What actually explains the data: the second column appears *only* in
  `accident2026.csv`, populated alongside (not instead of) the first, with
  paired values describing the same road-geometry concept under a
  different, overlapping taxonomy. Read as the source system introducing a
  second, more granular classification starting in the most recent
  (partial) file — recorded in CLAUDE.md as unresolved and flagged for
  Phase 2 to investigate further (possibly by asking the source agency),
  not coalesced and not routed into an "other" bucket, since neither
  action is justified by what the data actually shows.

**Also closed the ruff/mypy version-mismatch gap for good:** pinned exact
versions (not just matching floors) in both `pyproject.toml`
(`ruff==0.16.2`, `mypy==2.3.0`) and `.pre-commit-config.yaml`, with a
comment tying the two files together, since a `>=` floor in pyproject.toml
would let `uv sync` silently drift to a newer version than the still-pinned
pre-commit hook the next time dependencies are upgraded — the same failure
mode recurring later instead of actually being closed.

**Why this matters for the assignment:** the instruction to "report the
facts and I'll decide" rather than picking a dedup/tie-break rule is itself
worth preserving as a working pattern — investigation and decision are
different steps, and collapsing them (by quietly picking a plausible-
sounding rule and moving on) would have hidden a real business decision
(is the reconciliation target pre- or post-dedup?) inside an
implementation detail no one would think to review.

**A self-caught bug while wiring this into the notebook, worth recording
too:** `check_within_group_agreement(accidents_raw, ["ACC_CODE"], ...)`
initially reported 1 conflicting group out of 1,642 instead of 0 out of
1,641. Cause: polars' `group_by` treats every null key as one group, so
the 27,932 rows with a missing `ACC_CODE` were silently lumped into a
single giant "duplicate group" and (unsurprisingly) disagreed on agency
and fatality count. Caught by noticing the notebook's live output
(1,642 groups) didn't match the number already reported to the project
owner minutes earlier (1,641, computed via an explicit `drop_nulls` in an
ad-hoc check). Fixed at the function level, not the call site -- any
future caller grouping by a column with nulls would hit the same trap
otherwise. Added a regression test
(`test_null_group_keys_excluded_not_lumped_into_one_group`).
