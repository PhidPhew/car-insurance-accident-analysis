"""Severity-classification modeling frame (CLAUDE.md §10 L1/L2, Phase 5 Step 1).

Grain: **one accident** (`fact_accident.accident_sk`). This is the frame Phase 6's
MLP + entity embeddings uses, so it deliberately carries the raw, un-harmonized
high-cardinality categoricals (`cause_raw`, `crash_type_raw`, `road_geometry_raw`,
`road_code`, `province_th`) rather than Phase 2's collapsed `_group` taxonomy
columns -- CLAUDE.md §10 names exactly this ("120 causes, 152 crash types,
thousands of road codes") as the justification for entity embeddings over one-hot.

**Target: 4-tier `severity_class`** (`none` / `minor` / `serious` / `fatal`), the
worst injury tier present on the accident. Requires the fine fatal/serious/minor
split, so -- same restriction as `severity_index_fine`, CLAUDE.md §9 -- this frame
is built only from rows where `has_injury_breakdown` is true: **2020-2025**
(2019 dropped, no serious/minor split to classify with; 2026 dropped as a P7.11
incomplete holdout year, not a modeling year).

**COVID guard rail** (CLAUDE.md Step 1): rather than drop the ~81 severe/partial
-disruption days (2020-03-26 - 2020-06-14, entirely inside this frame's window),
`covid_disruption_level` is carried as an explicit categorical feature. Dropping
~81 days of a single pandemic quarter buys almost nothing (0.03% of rows) and
throws away real information a classifier can legitimately use; flagging it
explicitly lets the model (and SHAP) show whether disruption-era accidents were
systematically more/less severe instead of silently smoothing them into the
"normal" background rate.

**Weather is not leakage here**, unlike the frequency/severity frame: this frame
classifies the severity of an accident that has already happened (an explanatory/
retrospective task, e.g. "given this accident occurred, how severe was it and
why"), not a forward monthly forecast. The reported weather *at* the accident is
an attribute of the event itself, known the moment it's reported -- nothing like
the frequency frame's "would need next month's actual weather to know this"
problem.
"""

from __future__ import annotations

import polars as pl

from rri.warehouse.dim_accident_context import CONTEXT_COLUMNS
from rri.warehouse.keys import UNKNOWN_SK

YEAR_START = 2020
YEAR_END = 2025

#: `road_code`/`province_th` come from `fact_accident`/`dim_geography` instead
#: (already resolved there) -- these 3 are the only raw text columns not already
#: present anywhere in the fact/dims, per CLAUDE.md §10's "cause, crash type,
#: road geometry, road code, province" list.
_RAW_COLUMN_RENAME = {
    "มูลเหตุสันนิษฐาน": "cause_raw",
    "ลักษณะการเกิดเหตุ": "crash_type_raw",
    "บริเวณที่เกิดเหตุ": "road_geometry_raw",
}

#: Grouped (Phase 2 taxonomy) columns, kept alongside the raw ones as a
#: lower-cardinality fallback for models that want it.
_GROUP_COLUMN_RENAME = {c: c.replace("_group", "") + "_group" for c in CONTEXT_COLUMNS}


def _severity_class_expr() -> pl.Expr:
    return (
        pl.when(pl.col("fatality_count") > 0)
        .then(pl.lit("fatal"))
        .when(pl.col("serious_injury_count") > 0)
        .then(pl.lit("serious"))
        .when(pl.col("minor_injury_count") > 0)
        .then(pl.lit("minor"))
        .otherwise(pl.lit("none"))
    )


def build_severity_classification_frame(
    accidents_staging: pl.DataFrame,
    fact_accident: pl.DataFrame,
    dim_date: pl.DataFrame,
    dim_geography: pl.DataFrame,
    dim_vehicle: pl.DataFrame,
    dim_weather_band: pl.DataFrame,
) -> pl.DataFrame:
    """Build the finished record-level severity-classification frame.

    `accidents_staging` is `data/10_staging/accidents.parquet` -- `fact_accident`
    is built from it via `with_row_index("accident_sk", offset=1)` in the same row
    order (`rri.warehouse.fact_accident`), so re-deriving that same row index here
    is a safe, exact join key back to the raw high-cardinality text columns
    `fact_accident` itself doesn't carry.
    """
    raw_with_sk = accidents_staging.with_row_index("accident_sk", offset=1).with_columns(
        pl.col("accident_sk").cast(pl.Int64)
    )

    date_lookup = dim_date.filter(pl.col("date_sk") != UNKNOWN_SK).select(
        "date_sk",
        "year",
        "month",
        "season",
        "is_public_holiday",
        "is_weekend",
        "is_long_weekend",
        "is_songkran_7days",
        "is_newyear_7days",
        "days_to_next_holiday",
        "covid_disruption_level",
    )
    geography_lookup = dim_geography.select(
        "geography_sk",
        "province_th",
        "region",
        "flood_pct_high_risk_tambons",
        "ev_share",
        "tourist_attraction_count",
        "r35mm_avg_days_2014_2023",
    )
    vehicle_lookup = dim_vehicle.select("vehicle_sk", "vehicle_group")
    weather_lookup = dim_weather_band.select("weather_band_sk", "weather_band", "is_hazardous")

    frame = (
        fact_accident.join(
            raw_with_sk.select(
                "accident_sk", *_RAW_COLUMN_RENAME.keys(), *_GROUP_COLUMN_RENAME.keys()
            ),
            on="accident_sk",
            how="inner",
        )
        .rename(_RAW_COLUMN_RENAME)
        .rename(_GROUP_COLUMN_RENAME)
        .join(date_lookup, on="date_sk", how="left")
        .join(geography_lookup, on="geography_sk", how="left")
        .join(vehicle_lookup, on="vehicle_sk", how="left")
        .join(weather_lookup, on="weather_band_sk", how="left")
        .filter(
            pl.col("has_injury_breakdown"),
            pl.col("year").is_between(YEAR_START, YEAR_END),
        )
        .with_columns(_severity_class_expr().alias("severity_class"))
    )

    return frame.select(
        "accident_sk",
        "year",
        "month",
        "season",
        "province_th",
        "region",
        "vehicle_group",
        "cause_raw",
        "crash_type_raw",
        "road_geometry_raw",
        "road_code",
        "km_marker",
        "weather_band",
        "is_hazardous",
        "agency",
        "is_public_holiday",
        "is_weekend",
        "is_long_weekend",
        "is_songkran_7days",
        "is_newyear_7days",
        "days_to_next_holiday",
        "covid_disruption_level",
        "flood_pct_high_risk_tambons",
        "ev_share",
        "tourist_attraction_count",
        "r35mm_avg_days_2014_2023",
        "fatality_count",
        "serious_injury_count",
        "minor_injury_count",
        "total_injury_count",
        "severity_class",
    )
