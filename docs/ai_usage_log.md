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

---

## 2026-08-13 — The 448 date-transposition pairs nearly invalidated the P1 design

**Phase / task:** Phase 1, Task 4 second follow-up (no Task 5 code touched
until this closed)

The project owner pointed out that the previous round's P5 finding — 448
duplicate pairs with transposed day/month digits — had a load-bearing
implication that got buried as a side note: the entire P1 fix design rests
on "one file, one date format," detected per file and applied uniformly to
that file's ambiguous rows. If the 448 transposed pairs turned out to
straddle *different* files, that would confirm the assumption. If they sat
in the *same* file, the assumption could be broken and the design would
need to be redone before anything else proceeded.

**Checked directly: same file, 100% (all 448 in `accident2023.csv`
specifically).** By the owner's own stated decision criterion this is the
"redesign" branch. Investigated further before concluding either way,
since a mechanical rule shouldn't be applied without checking whether its
premise actually holds here:

- `accident2023.csv`'s MDY assignment rests on 14,578 unambiguous rows (a
  component > 12, decidable regardless of stated format). The 448
  transposed pairs (896 rows) are ~6% of that evidence base.
- Checked whether any of the 448 pairs contain an internally-unambiguous
  string that could independently confirm or contradict the file's format
  at the pair level: none do. This turns out to be structural, not a
  coincidence -- a digit transposition only produces two *plausible*
  dates when both components are ≤ 12 in both strings, so this defect
  class can only occur inside the already-ambiguous range. It can neither
  contradict the unambiguous-row majority vote nor contaminate it.
- The swap pattern itself (`a/b/y` ↔ `b/a/y`, exact, 100% of the 448)
  matches a re-keying transcription error, not a file that silently mixes
  formats.

**Reported this reasoning as evidence for the owner to rule on, rather
than unilaterally overriding their own stated either/or criterion** --
the criterion was theirs to set, and "same file" satisfying it literally
doesn't mean the underlying judgment call is mine to make instead of
theirs, even when the evidence points the same direction I'd argue for.
Proposed the constructive outcome either way: a 4th safety net using
known P5 duplicate pairs as post-parse ground truth (both rows of a known
pair must resolve to the identical calendar date) -- stronger than the
existing 3 layers because it validates at day resolution, not just year.

**Also checked, per the same message: is P5's 74% `บริเวณที่เกิดเหตุ`
conflict rate real conflict?** The owner connected two previously-separate
findings -- P3's rejected "twin column" pair (which turned out to be a
2026-only parallel taxonomy) and P5's high conflict rate on the same
column -- and asked whether they were the same phenomenon. Sampled 20
conflicting pairs directly: all 20 were synonym or elaboration-suffix
pairs (e.g. "no slope" vs. "flat/level" for the same concept). Quantified
across all 1,216 conflicts with a simple normalization: 85.1% collapse to
one value, and most of the remaining 14.9% are still recognizably the same
pattern under different phrasing. This changes the P5 remediation order:
taxonomy harmonization has to happen *before* deduplication, not after --
picking a tie-break rule first would discard rows that actually agree once
the vocabulary is unified.

**Reconciliation reframed as two permanent gates, not one chosen number:**
post-Extract (21,814, proves nothing lost on ingest) and post-Load
(21,497, proves dedup removed exactly what was intended). Recorded both in
CLAUDE.md §8 as separate, permanent checks rather than a single target to
pick between.

**Also reconciled two numbers from different rounds that looked
contradictory but weren't:** "1,702 duplicate ACC_CODE" (an earlier
profiling pass) vs. "1,641 groups" (this round). Confirmed these are
different units of the same fact (3,343 total rows across 1,641 groups,
minus 1,641 first-occurrences = 1,702 excess rows) and confirmed the
1,702 figure was never affected by the null-grouping bug logged above --
`profile_duplicates` already excluded null keys correctly from the start;
only the newer `check_within_group_agreement` (added this round) had the
gap.

**Why this matters for the assignment:** the load-bearing distinction
here is between "presenting evidence for a decision" and "making the
decision" -- the owner explicitly set a binary criterion or asked a direct
either/or question in every one of these four items, and each answer
(same file; still not fully resolved) technically triggers the more
serious branch of that criterion. Reporting the full reasoning and
recommendation without asserting the conclusion myself keeps the actual
decision where it belongs.

---

## 2026-08-13 — Safety net #4 was a genuine logic error, caught and fixed by the owner

The owner ruled on the previous entry's open question ("is the per-file MDY
assumption still safe given the 448 transposed pairs?") with **a sharper
argument than the one I offered**, then found a real bug in my own proposed
fix before it reached the codebase.

**The owner's reasoning was more decisive than mine.** I had argued "same
file in 100% of the 448 pairs, but structurally can't contradict the
majority vote" -- true, but indirect. The owner's version: if
`accident2023.csv` genuinely mixed DMY and MDY, the mixing would show up
as `dmy_evidence` (first component > 12) rows *spread across the whole
file*, not confined to duplicate pairs. Checking that directly is a
sharper, more falsifiable test than my structural argument. Verified: 0
`dmy_evidence` rows anywhere in the file's 23,841 dates, against 14,578
`mdy_evidence` rows -- not a majority vote, a zero-counter-evidence result.
Added `classify_slash_date_side_evidence` / `profile_slash_date_side_evidence`
to `src/rri/utils/profiling.py` to make this check reusable rather than a
one-off script.

**My proposed safety net #4 was wrong, and the owner caught it before it
shipped.** I had written it as "both rows of a known P5 duplicate pair
must resolve to the same calendar date after parsing." Once the file is
confirmed uniformly MDY, this is mathematically guaranteed to fail on
every one of the 448 transposed pairs: one row parses to `month=a day=b`,
the other to `month=b day=a`, by construction never equal. A rule built on
this premise would flag all 448 pairs as "parser broken" on every run --
a false-positive signal convincing enough that someone could act on it and
break a parser that was working correctly. This was not a subtle edge
case; the failure was guaranteed for the entire target set, which is what
should have been caught before proposing it, not after.

**The fix, exactly as the owner specified:** split into two non-competing
checks instead of one overloaded one.
- **Safety net 4a** (`check_duplicate_date_string_consistency`): restricted
  to duplicate groups where the raw date *string* is identical across every
  row (1,193 of the 1,641 `ACC_CODE` groups) -- this is the real parser
  determinism check, and it passed clean (0 mismatches).
- **`date_ambiguous_pairs` concept** (`find_transposed_date_string_pairs` +
  `discriminate_transposed_date_pairs`): the remaining 448 groups, resolved
  using safety net #2 (`accident_date <= report_date`, gap <= 90 days) as a
  discriminator, checked per-candidate against every row carrying that
  candidate's raw string (this generalizes correctly to the 23 three-row
  groups, not just the 425 two-row ones -- an earlier ad-hoc script missed
  this and silently dropped 23 groups from its totals before the reusable
  function was written). Result: 222 resolved uniquely, 196 genuinely
  undecidable (both candidates valid), 30 flagged (neither valid --
  spot-checked, these have their own report-date anomaly stacked on top of
  the transposition). Sums to exactly 448.

**Per the owner's instruction, no date was picked for the 226 undecidable
groups.** A `date_certainty` (`certain` / `ambiguous`) field on
`fact_accident` is recorded in CLAUDE.md P1 as a Phase 2 design note, not
implemented now -- 448/23,841 rows (~1.9%) is small, but a wrong date by
months would distort BQ3 (holiday-period risk), so it gets flagged rather
than guessed.

**Why this matters for the assignment:** this is a concrete instance of
"trust but verify" applied to my own output, not just to data -- the
owner did not accept a plausible-sounding safety-net design without
tracing what it would actually do against the real 448 pairs, and doing
so surfaced a rule that was unusable as written. Recording the failure
mode (guaranteed false positive on the entire target set) alongside the
fix so the same mistake -- checking "does this pass a plausible-sounding
review" instead of "what does this rule actually output against real
data" -- isn't repeated in Phase 2.

---

## 2026-08-13 — The 90-day threshold was invented, not measured; off by a wide margin

The owner flagged something I should have caught myself: the 90-day
reporting-lag limit written into CLAUDE.md's safety net #2 was picked at
design time, before any data existed to check it against. Asked for three
things: (1) record the 1,193 + 448 = 1,641 completeness check from the
previous round as an explicit proof the duplicate-group split is
exhaustive, (2) profile the actual `report_date - accident_date`
distribution before trusting 90 days, (3) re-run the 448-pair
discrimination with whatever threshold the data supports.

**The completeness check was real and worth recording as its own fact,
not just implied by the two counts summing correctly.** 1,193
(identical-raw-string groups, safety net 4a) + 448 (transposed groups,
`date_ambiguous_pairs`) = 1,641, exactly the total ACC_CODE duplicate-group
count from P5. Every group lands in exactly one branch. Added to CLAUDE.md
P1 as its own callout rather than leaving it as an arithmetic coincidence
a reader has to notice themselves.

**The 90-day number was not just imprecise -- it was wrong by an order of
magnitude relevant to the pipeline.** Profiled the full dataset (161,224
of 161,307 rows parseable under each file's known format) with new
functions in `src/rri/utils/profiling.py`
(`compute_report_lag_days`, `summarize_report_lag`,
`report_lag_quantiles_by_group`, plus `SOURCE_FILE_DATE_FORMAT` /
`parse_accident_date` / `parse_excel_serial_date` to parse dates under
each file's already-established format, including the Excel-serial years):
median lag is 45 days, but **28.6% of all rows (46,179 of 161,224) exceed
90 days**. A hard 90-day gate applied as a DQ Gate rule -- which was the
plan, since this same threshold gates every row, not just the 448 pairs --
would have silently rejected more than a quarter of genuinely valid data.
This is exactly the risk the owner named before I had the numbers: "a
threshold that's too tight will reject valid data at scale."

**The distribution also varies enough by agency that a single global
number is a real simplification, not a detail.** p99 by agency: Rural
Roads Dept. 235 days, Highways Dept. (92.6% of rows) 393 days, Expressway
Authority 1,219 days -- over three years. Recorded this as a documented
limitation in CLAUDE.md rather than either ignoring it or trying to solve
per-agency thresholds in the same pass; a per-agency threshold is a
reasonable Phase 2 refinement, not something to improvise here.

**Settled on p99 (398 days) as the data-derived global threshold**, per
the owner's suggestion that p99 was a reasonable starting point. This
deliberately accepts that ~1% of legitimate rows sit past the bound (that
is what a p99 threshold means) in exchange for not rejecting the 28.6%
that 90 days wrongly caught.

**Re-ran `discriminate_transposed_date_pairs` on the real 448 pairs with
398 days instead of 90:** resolved-unique dropped from 222 to 138, both-
valid (undecidable) rose from 196 to 310, and **neither-valid dropped from
30 to 0** -- confirming the owner's specific prediction that those 30 cases
were "rows with legitimately long reporting lags rather than corrupted
ones," not a second, independent bug. The Phase 2 `date_certainty` design
note (still not implemented, per standing instruction not to silently pick
a date) now covers 310 groups / 620 rows instead of 226 / 452.

**Why this matters for the assignment:** a threshold with no empirical
basis is exactly the kind of thing that looks fine in review (90 days
"sounds reasonable" for reporting a road accident) and is wrong at scale
in a way nobody notices until a downstream gate silently drops a quarter
of the data. The fix wasn't picking a better number from intuition either
-- it was building the reusable profiling capability to measure the real
distribution first, the same discipline already applied to every other P1
claim in CLAUDE.md.

---

## 2026-08-13 — Task 5 data contracts: the schema itself found a new bug

Given specific guidance before starting Task 5: contracts must encode
facts already proven during profiling, not re-introduce untested
assumptions the way the 90-day threshold did. Concretely: no date-format
shape assertion on the accident file (three formats coexist by design);
assert the things already proven (77 provinces / zero orphans, per-file
date format matching profiling, the 4 genuine twin-column pairs, both
reconciliation gates as code constants); and where the raw data is known
to diverge from a clean expectation, let the pandera check fail and report
it rather than loosening the schema to pass.

**Followed the letter of the date-column instruction and it immediately
found something no prior round of profiling had caught.** The schema
requires `วันที่เกิดเหตุ` and `วันที่รายงาน` to be present and non-null --
nothing about shape. Running it against the real file: `วันที่เกิดเหตุ`
(accident date) is never null, as expected, but `วันที่รายงาน` (report
date) is null in 78 rows (74 in `accident2026.csv`, the still-incomplete
year; 4 in `accident2023.csv`). This was never in CLAUDE.md P1-P7 --
every earlier round of profiling had reason to look at accident date
(the P1 format investigation) but nothing had specifically checked report
date for nulls. Did not loosen the schema to `nullable=True` to make it
pass; recorded the 78-row fact in P7 and the CLAUDE.md Task 5 summary
table instead, per the explicit "the failures are the deliverable"
instruction.

**The other checks came back exactly as predicted, which is itself the
validation that matters:** coordinate-bounds failures matched P7 exactly
(5 lat, 10 lon), the 4 twin-column pairs all showed zero overlap, all 8
files' per-file date-format check showed zero counter-evidence, province
normalization hit exactly 77 canonical names with zero orphans, and Gate 1
matched CLAUDE.md's 161,307 / 21,814 exactly. None of this was a surprise
-- it's the point: a contract that reproduces already-proven facts when
run against the same data it was built from is a contract that can be
trusted to catch a real regression in Phase 2, rather than one that just
happens to agree with itself by construction.

**One more precise number came out of building the injury-total check**:
9,872 of 161,307 rows (6.1%) have `serious + minor != total`. P7 had only
ever said "flag mismatches, don't silently fix" without a count -- the
contract needed an actual assertion to run, so building it produced the
number CLAUDE.md was missing.

**Why this matters for the assignment:** the date-column instruction could
have been treated as a formality (assert non-null, obviously it'll pass,
move on) -- instead running it against real data surfaced a genuine,
previously undocumented gap. This is the same lesson as the 90-day
threshold from the entry above, in the opposite direction: that entry was
about a check that was too strict where reality was fine; this one is a
check that revealed reality wasn't quite what everyone assumed, in a place
nobody had thought to look. Both come from the same discipline -- write
the assertion, run it against the real file, trust what comes back over
what seemed obvious going in.
