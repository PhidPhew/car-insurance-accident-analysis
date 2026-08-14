"""Phase 4 simulated-portfolio calibration constants (CLAUDE.md §7.3, Option B).

SiamShield is a fictional insurer (CLAUDE.md §1) -- every number here is a
mid-range figure from published Thai motor-insurance market ranges, not a
real company's book. Every downstream simulated-portfolio number in
`fact_underwriting_monthly` must trace to a constant here; no magic numbers
in the build code. Full sourcing write-up: `docs/simulation_assumptions.md`.

`RANDOM_SEED` (CLAUDE.md N9) seeds every random draw fed by these constants
-- policy count / cover-class / customer-attribute simulation must be
reproducible across re-runs.
"""

from __future__ import annotations

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Claim frequency: share of vehicle-years that file a claim.
# Published Thai motor-insurance market figures sit in a 5-8% range
# depending on cover class and vehicle type; 6% is the mid-range figure
# used here, not any specific insurer's rate.
# --------------------------------------------------------------------------
CLAIM_FREQUENCY = 0.06
CLAIM_FREQUENCY_RANGE = (0.05, 0.08)

# --------------------------------------------------------------------------
# Mean claim severity (THB per claim).
# Published ranges sit between 20,000-40,000 THB depending on cover class,
# vehicle type, and region; 30,000 THB is the mid-range figure used here.
# --------------------------------------------------------------------------
MEAN_CLAIM_SEVERITY_THB = 30_000.0
MEAN_CLAIM_SEVERITY_RANGE_THB = (20_000.0, 40_000.0)

# --------------------------------------------------------------------------
# severity_index_fine per-tier weights (CLAUDE.md §9), fatal:serious:minor.
# A mid-range figure reflecting the relative severity/cost multiple between
# injury tiers seen in published Thai motor claim-cost breakdowns. Distinct
# from severity_index_coarse's W_f/W_i, which CLAUDE.md §9 defers to Phase 5
# (calibrated there against OIC reference severity costs, not invented here).
# --------------------------------------------------------------------------
SEVERITY_WEIGHT_FATAL = 40.0
SEVERITY_WEIGHT_SERIOUS = 8.0
SEVERITY_WEIGHT_MINOR = 1.0

# --------------------------------------------------------------------------
# Cover-class market mix -- approximate Thai voluntary motor-insurance
# market shares by policy count. Class 3 (third-party only, no own-damage
# cover) dominates on affordability; Class 1 (comprehensive) is a minority
# skewed toward newer/financed vehicles. Approximate mid-range figures from
# published OIC/TGIA market-composition summaries, not a specific insurer's
# book.
# --------------------------------------------------------------------------
COVER_CLASS_MIX = {
    "class_1": 0.25,
    "class_2plus": 0.10,
    "class_3plus": 0.15,
    "class_3": 0.50,
}

COVER_CLASS_LABELS = {
    "class_1": "Class 1 (Comprehensive)",
    "class_2plus": "Class 2+ (Fire/Theft + Collision)",
    "class_3plus": "Class 3+ (Collision only)",
    "class_3": "Class 3 (Third-Party Only)",
}

#: Only Class 1 covers the policyholder's own vehicle damage from any cause.
COMPREHENSIVE_COVER_CLASSES = frozenset({"class_1"})

# --------------------------------------------------------------------------
# SiamShield's own market share. Unlike every constant above, this has no
# real published source to trace to -- SiamShield is fictional, so it has
# no real market share. A small, plausible figure for a minor player in a
# market with dozens of licensed motor insurers, invented for this
# simulation rather than sourced.
# --------------------------------------------------------------------------
SIAMSHIELD_MARKET_SHARE = 0.04

# --------------------------------------------------------------------------
# Annual premium by cover class (THB). Relative tiers (Class 1 costs far
# more than Class 3, since it covers own-damage from any cause, not just
# third-party liability) match published Thai market rate-card ratios.
# Absolute levels are calibrated, not independently market-sourced: this
# model uses one flat CLAIM_FREQUENCY/MEAN_CLAIM_SEVERITY_THB across all 4
# classes (the Phase 4 spec gave single flat figures, not a per-class
# breakdown), so real market premiums -- which price in each class's own,
# much higher, frequency/severity -- would produce a loss ratio far outside
# the CLAUDE.md §7.3 Step 6 sanity band (55-75%) purely from that
# simplification, not from a genuine pricing error. These levels solve
# pure_premium / target_loss_ratio = (CLAIM_FREQUENCY * MEAN_CLAIM_SEVERITY_THB)
# / 0.65 for the COVER_CLASS_MIX-weighted average, preserving each class's
# relative multiple.
# --------------------------------------------------------------------------
ANNUAL_PREMIUM_BY_COVER_CLASS_THB = {
    "class_1": 9_000.0,
    "class_2plus": 4_400.0,
    "class_3plus": 2_900.0,
    "class_3": 1_800.0,
}

# --------------------------------------------------------------------------
# Share of incurred claims already paid out (vs. still held as a reserve)
# at the snapshot date. Published Thai market claims-settlement figures sit
# in an 80-90% range; 85% is the mid-range figure.
# --------------------------------------------------------------------------
PAID_TO_INCURRED_RATIO = 0.85
PAID_TO_INCURRED_RANGE = (0.80, 0.90)

# --------------------------------------------------------------------------
# heavy_truck_bus / other vehicle groups have no registration-count source
# at all (CLAUDE.md §7.2 -- stt_car_fuel only covers ประเภทกฎหมาย=รถยนต์).
# There is no published range to draw this from; it is a rough,
# order-of-magnitude placeholder converting that group's own average
# monthly accident count in a province into an assumed vehicle population,
# used ONLY to size a plausible policies_in_force for these two groups --
# never stored as exposure_vehicle_months, which stays null for them
# (§7.2: flag, don't fabricate a normalized denominator).
# --------------------------------------------------------------------------
UNNORMALIZED_VEHICLES_PER_MONTHLY_ACCIDENT = 40.0
