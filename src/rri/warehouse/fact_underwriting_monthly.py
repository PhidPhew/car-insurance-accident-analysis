"""`fact_underwriting_monthly` builder (CLAUDE.md §8, Phase 4 Step 5). Periodic snapshot.

Grain: one month x one province x one vehicle group x one cover class. Only real
(non-Unknown-Member) dimension rows are used -- 77 provinces x 5 vehicle groups x 4
cover classes = 1,540 combos per month.

**Window: 2022-01 through 2025-12 (48 months, 73,920 rows)**, not CLAUDE.md's own
`fact_accident` range (2019-2026, minus the incomplete 2026 holdout year, P7.11).
Deliberate choice, not an oversight: the exposure back-cast (§7.1) needs
`car_registered_new`'s national growth curve, which only has data from 2022-01 onward --
there is nothing to accumulate before that month, so extrapolating the curve backward to
cover 2019-2021 would mean fabricating growth-rate data the source doesn't have. CLAUDE.md
§8 estimates "~92,400 rows" for this fact, which assumes a 60-month window; this build
uses the 48 months the exposure source actually supports rather than stretch a data
source past what it covers. Flagged in the Phase 4 report for visibility.

**Real vs. simulated, restated from `docs/simulation_assumptions.md` (revised from that
doc's original wording -- see the note below on why):**
- `exposure_vehicle_months`: the §7.1 back-cast, real for the 3 vehicle groups
  `stt_car_fuel` registers (motorcycle/passenger_car/pickup). **Null** for
  `heavy_truck_bus`/`other` (§7.2 -- no registration source covers them; never fabricate a
  normalized denominator). Informational context only -- deliberately NOT the basis for
  `policies_in_force` below (see that note).
- `policies_in_force`, `earned_premium`: simulated. Basis is `exposure_vehicle_count` for
  the 3 normalized groups; for the 2 unnormalized groups, a documented rough proxy
  (`UNNORMALIZED_VEHICLES_PER_MONTHLY_ACCIDENT` x that cell's own average monthly
  accident count) since no registration source exists for them at all -- this proxy is
  used only to size a plausible policy count, never stored as if it were
  `exposure_vehicle_months`. Both scaled by `SIAMSHIELD_MARKET_SHARE` (fictional, no real
  source -- SiamShield doesn't exist) x `COVER_CLASS_MIX`, with seeded (`RANDOM_SEED = 42`)
  +/-5% noise for realism.
- `claim_count = policies_in_force * (CLAIM_FREQUENCY / 12)` -- the standard actuarial
  frequency = claims/exposure relationship, solved for claims. **Not** raw per-cell
  accident count x `CLAIM_FREQUENCY`, despite `docs/simulation_assumptions.md` originally
  describing it that way: a single month's accident count in one (province, vehicle_group)
  cell is too sparse (most cells see 0-1/month) and, worse, is completely decoupled in
  scale from `exposure_vehicle_count` (real registered-fleet counts run into the millions
  for a large province) -- during development that combination produced a loss ratio
  ~0.07%, three orders of magnitude under the CLAUDE.md §7.3 Step 6 sanity band (55-75%).
  Real accident data still drives claim cost, via `relative_severity_multiplier` next.
- `incurred_claim_amount`: `claim_count` x `MEAN_CLAIM_SEVERITY_THB`, scaled by a
  `relative_severity_multiplier` built from each cell's own real fatality/serious/minor
  injury counts (the `SEVERITY_WEIGHT_*` fine-tier weights, CLAUDE.md §9) relative to the
  window-wide accident-weighted average -- a cell with a real elevated severity mix shows
  elevated claim cost, not a flat rate. (This window is fully post-2019, so every cell has
  `has_injury_breakdown = true` -- no need for `severity_index_coarse`'s `W_f`/`W_i`,
  which CLAUDE.md §9 defers to Phase 5.)
- `paid_claim_amount = incurred_claim_amount * PAID_TO_INCURRED_RATIO`.

`loss_ratio`/`frequency`/`severity`/`pure_premium` are query-time ratios, never stored
(CLAUDE.md §8).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from config.simulation_params import (
    ANNUAL_PREMIUM_BY_COVER_CLASS_THB,
    CLAIM_FREQUENCY,
    COVER_CLASS_MIX,
    MEAN_CLAIM_SEVERITY_THB,
    PAID_TO_INCURRED_RATIO,
    RANDOM_SEED,
    SEVERITY_WEIGHT_FATAL,
    SEVERITY_WEIGHT_MINOR,
    SEVERITY_WEIGHT_SERIOUS,
    SIAMSHIELD_MARKET_SHARE,
    UNNORMALIZED_VEHICLES_PER_MONTHLY_ACCIDENT,
)
from rri.features.exposure import (
    backcast_monthly_exposure,
    national_cumulative_registrations,
    province_vehicle_group_snapshot_counts,
)
from rri.warehouse.keys import UNKNOWN_SK

WINDOW_START = (2022, 1)
WINDOW_END = (2025, 12)

#: Additivity contract for this fact's measures (CLAUDE.md §8) -- any query or
#: dashboard aggregating this table must consult this before choosing SUM.
#: `policies_in_force` is semi-additive: summing it across provinces/vehicle
#: groups/cover classes *within one month* is valid (a snapshot count at that
#: point in time), but summing it *across months* double-counts the same
#: policies and is meaningless -- use AVG or a specific month's value instead.
MEASURE_ADDITIVITY = {
    "policies_in_force": "semi-additive (never sum across months)",
    "exposure_vehicle_months": "additive",
    "earned_premium": "additive",
    "claim_count": "additive",
    "incurred_claim_amount": "additive",
    "paid_claim_amount": "additive",
}


def _window_months(
    start: tuple[int, int] = WINDOW_START, end: tuple[int, int] = WINDOW_END
) -> list[tuple[int, int]]:
    year, month = start
    months: list[tuple[int, int]] = []
    while (year, month) <= end:
        months.append((year, month))
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months


def _month_date_sk(dim_date: pl.DataFrame, year: int, month: int) -> int:
    row = dim_date.filter(
        (pl.col("year") == year) & (pl.col("month") == month) & (pl.col("day") == 1)
    )
    if row.height == 0:
        raise ValueError(f"dim_date has no row for {year}-{month:02d}-01")
    return int(row["date_sk"][0])


def _real_accident_aggregates(fact_accident: pl.DataFrame, dim_date: pl.DataFrame) -> pl.DataFrame:
    """Aggregate real fact_accident to (year, month, geography_sk, vehicle_sk).

    Excludes rows whose date/geography/vehicle FK is the Unknown Member -- an
    accident that can't be attributed to a real province/vehicle-group/month
    can't be assigned to a fact_underwriting_monthly cell either.
    """
    date_lookup = dim_date.filter(pl.col("date_sk") != UNKNOWN_SK).select(
        "date_sk", "year", "month"
    )
    joined = fact_accident.filter(
        (pl.col("date_sk") != UNKNOWN_SK)
        & (pl.col("geography_sk") != UNKNOWN_SK)
        & (pl.col("vehicle_sk") != UNKNOWN_SK)
    ).join(date_lookup, on="date_sk", how="inner")

    return joined.group_by(["year", "month", "geography_sk", "vehicle_sk"]).agg(
        pl.len().alias("accident_count"),
        pl.col("fatality_count").sum().alias("fatality_count"),
        pl.col("serious_injury_count").sum().alias("serious_injury_count"),
        pl.col("minor_injury_count").sum().alias("minor_injury_count"),
    )


def _with_relative_severity_multiplier(accident_agg: pl.DataFrame) -> pl.DataFrame:
    """Fine-tier severity score per cell, normalized to the accident-weighted window average."""
    scored = accident_agg.with_columns(
        (
            (
                pl.col("fatality_count") * SEVERITY_WEIGHT_FATAL
                + pl.col("serious_injury_count") * SEVERITY_WEIGHT_SERIOUS
                + pl.col("minor_injury_count") * SEVERITY_WEIGHT_MINOR
            )
            / pl.col("accident_count")
        ).alias("_fine_score")
    )
    portfolio_avg = scored.select(
        (pl.col("_fine_score") * pl.col("accident_count")).sum() / pl.col("accident_count").sum()
    ).item()
    return scored.with_columns(
        (pl.col("_fine_score") / portfolio_avg).alias("relative_severity_multiplier")
    ).drop("_fine_score")


def _unnormalized_group_avg_monthly_accidents(
    fact_accident: pl.DataFrame, dim_date: pl.DataFrame, dim_vehicle: pl.DataFrame
) -> pl.DataFrame:
    """Full-history (2019-2026) average monthly accident count per (province, vehicle_group),
    for the 2 vehicle groups with no registration source (`heavy_truck_bus`/`other`)."""
    unnormalized_sks = dim_vehicle.filter(
        (~pl.col("has_normalized_exposure")) & (pl.col("vehicle_sk") != UNKNOWN_SK)
    )["vehicle_sk"].to_list()

    date_lookup = dim_date.filter(pl.col("date_sk") != UNKNOWN_SK).select(
        "date_sk", "year", "month"
    )
    joined = fact_accident.filter(
        (pl.col("date_sk") != UNKNOWN_SK)
        & (pl.col("geography_sk") != UNKNOWN_SK)
        & (pl.col("vehicle_sk").is_in(unnormalized_sks))
    ).join(date_lookup, on="date_sk", how="inner")

    n_months = joined.select(pl.struct(["year", "month"]).n_unique()).item()
    if n_months == 0:
        return pl.DataFrame(
            schema={
                "geography_sk": pl.Int64,
                "vehicle_sk": pl.Int64,
                "avg_monthly_accidents": pl.Float64,
            }
        )

    return joined.group_by(["geography_sk", "vehicle_sk"]).agg(
        (pl.len() / n_months).alias("avg_monthly_accidents")
    )


def build_fact_underwriting_monthly(
    fact_accident: pl.DataFrame,
    dim_date: pl.DataFrame,
    dim_geography: pl.DataFrame,
    dim_vehicle: pl.DataFrame,
    dim_policy_type: pl.DataFrame,
    stt_car_fuel_df: pl.DataFrame,
    car_registered_new_df: pl.DataFrame,
    vehicle_mapping_path: Path,
) -> pl.DataFrame:
    """Build the finished `fact_underwriting_monthly` table (facts get no Unknown Member row)."""
    rng = np.random.default_rng(RANDOM_SEED)

    real_geography = dim_geography.filter(pl.col("geography_sk") != UNKNOWN_SK).select(
        "geography_sk", "province_th"
    )
    real_vehicle = dim_vehicle.filter(pl.col("vehicle_sk") != UNKNOWN_SK).select(
        "vehicle_sk", "vehicle_group", "has_normalized_exposure"
    )
    real_policy_type = dim_policy_type.filter(pl.col("policy_type_sk") != UNKNOWN_SK).select(
        "policy_type_sk", "cover_class"
    )

    accident_agg = _with_relative_severity_multiplier(
        _real_accident_aggregates(fact_accident, dim_date)
    )
    unnorm_avg = _unnormalized_group_avg_monthly_accidents(fact_accident, dim_date, dim_vehicle)

    snapshot_counts = province_vehicle_group_snapshot_counts(stt_car_fuel_df, vehicle_mapping_path)
    growth_curve = national_cumulative_registrations(car_registered_new_df)

    base_grid = real_geography.join(real_vehicle, how="cross").join(
        unnorm_avg, on=["geography_sk", "vehicle_sk"], how="left"
    )

    monthly_frames = []
    for year, month in _window_months():
        date_sk = _month_date_sk(dim_date, year, month)
        exposure = backcast_monthly_exposure(snapshot_counts, growth_curve, year, month)

        grid = base_grid.join(
            exposure.select("province_th", "vehicle_group", "exposure_vehicle_count"),
            on=["province_th", "vehicle_group"],
            how="left",
        )
        cell_acc = accident_agg.filter(
            (pl.col("year") == year) & (pl.col("month") == month)
        ).select("geography_sk", "vehicle_sk", "accident_count", "relative_severity_multiplier")
        grid = grid.join(cell_acc, on=["geography_sk", "vehicle_sk"], how="left").with_columns(
            pl.col("accident_count").fill_null(0),
            pl.col("relative_severity_multiplier").fill_null(1.0),
            pl.lit(date_sk).cast(pl.Int64).alias("date_sk"),
        )
        monthly_frames.append(grid)

    monthly = pl.concat(monthly_frames)

    cover_classes = real_policy_type.with_columns(
        pl.col("cover_class").replace_strict(COVER_CLASS_MIX).alias("cover_class_share")
    )
    full = monthly.join(cover_classes, how="cross")

    noise = rng.normal(loc=1.0, scale=0.05, size=full.height).clip(0.8, 1.2)
    full = full.with_columns(pl.Series("_noise", noise))

    # exposure_vehicle_months: informational, real-registration-based context.
    # Deliberately NOT the basis for policies_in_force/claim_count below --
    # a market-wide vehicle count (millions, for a big province) has no
    # relationship to SiamShield's own small book size, and tying claim_count
    # to it directly produced a loss ratio off by 3 orders of magnitude
    # during development (see this module's docstring / the Phase 4 report).
    full = full.with_columns(
        (pl.col("exposure_vehicle_count") * pl.col("cover_class_share"))
        .round(2)
        .alias("exposure_vehicle_months")
    )

    # policies_in_force: SiamShield's own simulated book, sized off real data
    # (registered-vehicle exposure for the 3 normalized groups; a documented
    # accident-based proxy for the 2 that have no registration source) times
    # its own fictional market share.
    full = full.with_columns(
        pl.when(pl.col("has_normalized_exposure"))
        .then(pl.col("exposure_vehicle_count").fill_null(0.0))
        .otherwise(
            pl.col("avg_monthly_accidents").fill_null(0.0)
            * UNNORMALIZED_VEHICLES_PER_MONTHLY_ACCIDENT
        )
        .alias("_policy_basis")
    )
    full = full.with_columns(
        (
            pl.col("_policy_basis")
            * SIAMSHIELD_MARKET_SHARE
            * pl.col("cover_class_share")
            * pl.col("_noise")
        )
        .round()
        .cast(pl.Int64)
        .alias("policies_in_force")
    )

    # claim_count: the standard actuarial relationship, frequency = claims /
    # exposure solved for claims (CLAIM_FREQUENCY is an annual rate, this
    # grain is monthly). Real accident data enters via
    # relative_severity_multiplier below, not the claim count itself --
    # deriving claim_count from a single month's raw accident count in one
    # (province, vehicle_group) cell is too sparse a signal (most cells see
    # 0-1 accidents/month) to size a stable monthly claim count against.
    full = full.with_columns(
        (pl.col("policies_in_force") * (CLAIM_FREQUENCY / 12.0))
        .round()
        .cast(pl.Int64)
        .alias("claim_count")
    )
    full = full.with_columns(
        (
            pl.col("claim_count") * MEAN_CLAIM_SEVERITY_THB * pl.col("relative_severity_multiplier")
        ).alias("incurred_claim_amount")
    )
    full = full.with_columns(
        (pl.col("incurred_claim_amount") * PAID_TO_INCURRED_RATIO).alias("paid_claim_amount")
    )
    full = full.with_columns(
        pl.col("cover_class")
        .replace_strict(ANNUAL_PREMIUM_BY_COVER_CLASS_THB)
        .alias("_annual_premium")
    )
    full = full.with_columns(
        (pl.col("policies_in_force") * pl.col("_annual_premium") / 12.0).alias("earned_premium")
    )

    full = full.with_row_index("underwriting_sk", offset=1).with_columns(
        pl.col("underwriting_sk").cast(pl.Int64)
    )

    return full.select(
        "underwriting_sk",
        "date_sk",
        "geography_sk",
        "vehicle_sk",
        "policy_type_sk",
        "policies_in_force",
        "exposure_vehicle_months",
        "earned_premium",
        "claim_count",
        "incurred_claim_amount",
        "paid_claim_amount",
    )
