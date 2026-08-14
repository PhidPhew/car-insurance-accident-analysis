# Simulated Policy Portfolio — Assumptions

> Companion to [CLAUDE.md](../CLAUDE.md) §7.3 and §8 (`fact_underwriting_monthly`).
> Constants referenced here live in `config/simulation_params.py` — this file is the
> sourcing write-up; the code file is the single source of truth for the numbers
> themselves.

## Why this exists

SiamShield Insurance is a fictional insurer (CLAUDE.md §1). We do not have — and could
not ethically obtain for a course assignment — a real Thai motor insurer's actual policy
book, premiums, or claims ledger. `fact_underwriting_monthly` exists to express the
project's **real** accident data in insurance-business terms (loss ratio, frequency,
severity, pure premium) so the ML and dashboard work reads as an underwriting exercise,
not just a traffic-accident analysis.

**Option B (decided):** every simulated constant is a stated, documented range from
published Thai motor-insurance market figures — not invented at random, and not
presented as a specific real insurer's numbers. Every number in this file has a
one-line sourcing note in `config/simulation_params.py` stating the range it sits in.

## What's real vs. simulated (CLAUDE.md §7.3, restated)

| Component | Status |
|---|---|
| Accidents / severity / location / time | 🟢 100% real — `fact_accident`, built in Phase 3 |
| Claim counts | 🟢 derived deterministically from real accident counts × `CLAIM_FREQUENCY` |
| Claim cost per case | 🟡 real severity (fatality/injury counts) × `MEAN_CLAIM_SEVERITY_THB` |
| Policy counts / premium / cover class / customer mix | 🔴 simulated, calibrated to `COVER_CLASS_MIX` |

**Non-negotiable (CLAUDE.md §7.3, repeated here because it's the rule most likely to be
violated by accident):** claim counts and costs are a deterministic function of real
`fact_accident` rows and the constants below — **never** an independent random
weather→claim relationship. If the simulator baked in its own weather effect, the ML
models in Phase 5-6 would just rediscover an assumption we invented, not a real signal.

## The constants

### Claim frequency — `CLAIM_FREQUENCY = 0.06`

Share of insured vehicle-years that file at least one claim in a year. Published Thai
motor-insurance market figures put this in a **5-8%** range depending on cover class and
vehicle type (comprehensive/Class 1 policies see claims more often than third-party-only
Class 3, since minor own-damage claims that a Class 3 holder would just pay out of pocket
get filed under Class 1). 6% is the mid-range figure used uniformly here — CLAUDE.md N9
determinism means every province/vehicle-group/month combination uses the same rate
rather than a class-specific one, a simplification stated as a limitation.

### Mean claim severity — `MEAN_CLAIM_SEVERITY_THB = 30_000`

Average payout per claim. Published ranges sit between **20,000-40,000 THB**, varying by
cover class, vehicle type, and region (Bangkok collision repairs cost more than upcountry
ones). 30,000 THB is the mid-range figure.

### Severity index fine weights — `40 : 8 : 1` (fatal : serious : minor)

Used by `severity_index_fine` (CLAUDE.md §9), computable 2020-2026 only (P7: 2019 never
captured the serious/minor breakdown). A mid-range figure reflecting the relative
cost/severity multiple between injury tiers in published Thai motor claim-cost analyses —
a fatality claim costs roughly 40× a minor-injury claim, a serious injury roughly 8×.

Distinct from `severity_index_coarse`'s `W_f`/`W_i` weights, which CLAUDE.md §9
explicitly defers to Phase 5 (calibrated there against OIC reference severity costs, not
invented in Phase 4) — do not add those here.

### Cover-class market mix

| Class | Share | What it covers |
|---|---|---|
| Class 1 (comprehensive) | 25% | Own damage (any cause) + theft + fire + third party |
| Class 2+ | 10% | Fire/theft + own damage (collision with a vehicle only) + third party |
| Class 3+ | 15% | Own damage (collision with a vehicle only) + third party, no fire/theft |
| Class 3 (third-party only) | 50% | Third party only, no own-damage cover |

Approximate mid-range figures from published OIC/TGIA Thai motor-market composition
summaries. Class 3 dominates by policy count on affordability; Class 1 skews toward
newer/financed vehicles. This is a national-average assumption, not province- or
vehicle-group-specific — stated as a limitation, same as claim frequency above.

## `fact_underwriting_monthly` grain and derivation

Grain: **one month × one province × one vehicle group × one cover class**
(~92,400 rows: ~86 months × 77 provinces × 5 vehicle groups × 4 cover classes, plus
Unknown Member rows).

1. **Claim counts** — aggregate real `fact_accident` rows to this grain (month × province
   × vehicle group), then `claim_count = accident_count × CLAIM_FREQUENCY`, split across
   the 4 cover classes by `COVER_CLASS_MIX`. Deterministic given the real accident count —
   no independent randomness.
2. **Claim cost** — `incurred_claim_amount = claim_count × MEAN_CLAIM_SEVERITY_THB`,
   scaled by each row's own `severity_index_coarse` relative to the portfolio average, so
   a province/month with a real elevated severity index shows elevated claim cost, not a
   flat rate. This is where real severity flows into the simulated book.
3. **Policy counts, premium, customer mix** — simulated from `COVER_CLASS_MIX` and
   published OIC market-share constants, seeded with `RANDOM_SEED = 42` (CLAUDE.md N9) so
   re-running the build produces byte-identical output.
4. **Exposure** — `exposure_vehicle_months` uses the Phase 1 back-cast (CLAUDE.md §7.1):
   per-province vehicle counts scaled by the national monthly registration growth curve
   from `car_registered_new_2565_2569_combined.csv`.

`loss_ratio`, `frequency`, `severity`, `pure_premium` are **never stored** — CLAUDE.md §8:
storing a ratio in a fact table breaks roll-up arithmetic. They're computed at query time
from the stored numerator/denominator measures.
