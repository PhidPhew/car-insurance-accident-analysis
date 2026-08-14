"""Phase 3 warehouse build orchestrator (CLAUDE.md §8).

Load order matters: dims before facts, so every FK a fact table writes can
be validated against an already-loaded dimension. Callable from a notebook
(`notebooks/03_warehouse_build.ipynb`, narrate-only per N2) or directly:

    uv run python -m rri.warehouse.build
"""

from __future__ import annotations

import polars as pl

from config.settings import settings
from config.simulation_params import (
    COMPREHENSIVE_COVER_CLASSES,
    COVER_CLASS_LABELS,
    COVER_CLASS_MIX,
)
from rri.utils.logging import configure_logging, get_logger
from rri.warehouse.dim_accident_context import build_dim_accident_context
from rri.warehouse.dim_customer_segment import build_dim_customer_segment
from rri.warehouse.dim_date import build_dim_date
from rri.warehouse.dim_geography import build_dim_geography
from rri.warehouse.dim_policy_type import build_dim_policy_type
from rri.warehouse.dim_time_of_day import build_dim_time_of_day
from rri.warehouse.dim_vehicle import build_dim_vehicle, vehicle_group_mapping_path
from rri.warehouse.dim_weather_band import build_dim_weather_band
from rri.warehouse.duckdb_io import get_connection, write_table
from rri.warehouse.fact_accident import build_fact_accident
from rri.warehouse.fact_underwriting_monthly import build_fact_underwriting_monthly

log = get_logger(__name__)

#: dim_customer_segment members (Phase 4 Step 5) -- a simple reference cross
#: product, not weighted by a market-share constant, since
#: fact_underwriting_monthly's grain (CLAUDE.md §8) doesn't include customer
#: segment. Populated for future-phase/dashboard use, not joined here.
_AGE_BANDS = ("18-25", "26-35", "36-50", "51-65", "65+")
_GENDERS = ("male", "female")
_CUSTOMER_TYPES = ("individual", "corporate")


def load_staging_frames() -> dict[str, pl.DataFrame]:
    """Read every staging parquet the warehouse build needs."""
    return {
        "accidents": pl.read_parquet(settings.staging_dir / "accidents.parquet"),
        "flood_risk": pl.read_parquet(settings.staging_dir / "flood_risk.parquet"),
        "heavy_rain_r35mm": pl.read_parquet(settings.staging_dir / "heavy_rain_r35mm.parquet"),
        "stt_car_fuel": pl.read_parquet(settings.staging_dir / "stt_car_fuel.parquet"),
        "car_registered_new": pl.read_parquet(settings.staging_dir / "car_registered_new.parquet"),
    }


def _policy_type_members() -> list[dict[str, object]]:
    return [
        {
            "cover_class": cover_class,
            "cover_class_label": COVER_CLASS_LABELS[cover_class],
            "is_comprehensive": cover_class in COMPREHENSIVE_COVER_CLASSES,
        }
        for cover_class in COVER_CLASS_MIX
    ]


def _customer_segment_members() -> list[dict[str, object]]:
    return [
        {"age_band": age_band, "gender": gender, "customer_type": customer_type}
        for age_band in _AGE_BANDS
        for gender in _GENDERS
        for customer_type in _CUSTOMER_TYPES
    ]


def build_dimensions(staging: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Build all 8 dims (CLAUDE.md §8 Step 1; policy_type/customer_segment populated per Step 5)."""
    return {
        "dim_date": build_dim_date(),
        "dim_geography": build_dim_geography(
            staging["flood_risk"], staging["heavy_rain_r35mm"], staging["stt_car_fuel"]
        ),
        "dim_vehicle": build_dim_vehicle(),
        "dim_accident_context": build_dim_accident_context(staging["accidents"]),
        "dim_time_of_day": build_dim_time_of_day(),
        "dim_weather_band": build_dim_weather_band(),
        "dim_policy_type": build_dim_policy_type(members=_policy_type_members()),
        "dim_customer_segment": build_dim_customer_segment(members=_customer_segment_members()),
    }


def run_warehouse_build() -> None:
    """Phase 3 (dims + fact_accident) then Phase 4 (fact_underwriting_monthly)."""
    configure_logging()
    settings.ensure_writable_dirs()

    staging = load_staging_frames()
    dims = build_dimensions(staging)

    fact_accident = build_fact_accident(
        staging["accidents"],
        dims["dim_date"],
        dims["dim_geography"],
        dims["dim_vehicle"],
        dims["dim_accident_context"],
        dims["dim_time_of_day"],
        dims["dim_weather_band"],
        vehicle_group_mapping_path(),
    )

    fact_underwriting_monthly = build_fact_underwriting_monthly(
        fact_accident,
        dims["dim_date"],
        dims["dim_geography"],
        dims["dim_vehicle"],
        dims["dim_policy_type"],
        staging["stt_car_fuel"],
        staging["car_registered_new"],
        vehicle_group_mapping_path(),
    )

    con = get_connection()
    try:
        for table_name, df in dims.items():
            write_table(con, table_name, df)
        write_table(con, "fact_accident", fact_accident)
        write_table(con, "fact_underwriting_monthly", fact_underwriting_monthly)
    finally:
        con.close()

    log.info(
        "warehouse_build_complete",
        tables=[*dims.keys(), "fact_accident", "fact_underwriting_monthly"],
        fact_accident_rows=fact_accident.height,
        fact_underwriting_monthly_rows=fact_underwriting_monthly.height,
    )


if __name__ == "__main__":
    run_warehouse_build()
