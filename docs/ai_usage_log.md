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
