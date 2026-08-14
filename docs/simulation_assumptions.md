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
| Claim counts | 🟡 `policies_in_force × CLAIM_FREQUENCY` (policies sized from real exposure data) |
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

Grain: **one month × one province × one vehicle group × one cover class**, real dimension
rows only (77 provinces × 5 vehicle groups × 4 cover classes = 1,540 combos/month).

**Window: 2022-01 through 2025-12 (48 months, 73,920 rows)** — not CLAUDE.md §8's own
"~92,400 rows" estimate (which assumes 60 months). Deliberate: the exposure back-cast
below needs `car_registered_new`'s national growth curve, which only has data from
2022-01 onward. Stretching it back to 2019 to hit the estimated row count would mean
fabricating growth-rate data the source doesn't have, so this build uses the 48 months
the source actually supports instead.

1. **Exposure** — `exposure_vehicle_months` uses the Phase 1 back-cast (CLAUDE.md §7.1):
   per-province vehicle counts scaled by the national monthly registration growth curve
   from `car_registered_new_2565_2569_combined.csv`. Real for the 3 vehicle groups
   `stt_car_fuel` registers (motorcycle/passenger_car/pickup); **null** for
   `heavy_truck_bus`/`other` (§7.2 — no registration source covers them). Informational
   context only — see the next point for why it is *not* the basis for policy counts.
2. **Policy counts, premium** — simulated. `policies_in_force` is sized off
   `exposure_vehicle_count` (the 3 normalized groups) or a documented accident-based proxy
   (the 2 that aren't), times `SIAMSHIELD_MARKET_SHARE` (fictional — SiamShield has no real
   market share to source) × `COVER_CLASS_MIX`, seeded with `RANDOM_SEED = 42` (CLAUDE.md
   N9) so re-running the build produces byte-identical output.
3. **Claim counts** — **revised from this doc's original design** (kept here for the
   record; see `rri.warehouse.fact_underwriting_monthly`'s module docstring for the full
   account): originally `claim_count = accident_count × CLAIM_FREQUENCY` directly off the
   real per-cell accident count. That produced a loss ratio ~0.07% — three orders of
   magnitude under the Step 6 sanity band — because a single month's accident count in one
   (province, vehicle_group) cell is far too sparse, and is completely decoupled in scale
   from the exposure-driven policy count (real registered-fleet counts run into the
   millions for a large province). Corrected to the standard actuarial relationship
   instead: `claim_count = policies_in_force × (CLAIM_FREQUENCY / 12)`, solving frequency
   = claims/exposure for claims. Deterministic given the (real-data-derived)
   `policies_in_force` — still no independent weather/claim link.
4. **Claim cost** — `incurred_claim_amount = claim_count × MEAN_CLAIM_SEVERITY_THB`,
   scaled by a `relative_severity_multiplier` built from each cell's own real
   fatality/serious/minor injury counts (the fine-tier `SEVERITY_WEIGHT_*` weights,
   CLAUDE.md §9) relative to the window-wide accident-weighted average — a province/month
   with a real elevated severity mix shows elevated claim cost, not a flat rate. This is
   where real severity flows into the simulated book. (This window is fully post-2019, so
   every cell has `has_injury_breakdown = true` — no need for `severity_index_coarse`'s
   `W_f`/`W_i`, which CLAUDE.md §9 defers to Phase 5.)
5. **`paid_claim_amount`** — `incurred_claim_amount × PAID_TO_INCURRED_RATIO`.

`loss_ratio`, `frequency`, `severity`, `pure_premium` are **never stored** — CLAUDE.md §8:
storing a ratio in a fact table breaks roll-up arithmetic. They're computed at query time
from the stored numerator/denominator measures. Overall loss ratio at these constants:
**~65.3%**, inside the 55-75% Step 6 sanity band.
