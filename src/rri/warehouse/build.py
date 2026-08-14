"""Phase 3 warehouse build orchestrator (CLAUDE.md §8).

Load order matters: dims before facts, so every FK a fact table writes can
be validated against an already-loaded dimension. Callable from a notebook
(`notebooks/03_warehouse_build.ipynb`, narrate-only per N2) or directly:

    uv run python -m rri.warehouse.build
"""

from __future__ import annotations

import polars as pl

from config.settings import settings
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

log = get_logger(__name__)


def load_staging_frames() -> dict[str, pl.DataFrame]:
    """Read every staging parquet `dim_geography` needs (accidents included, for Step 2)."""
    return {
        "accidents": pl.read_parquet(settings.staging_dir / "accidents.parquet"),
        "flood_risk": pl.read_parquet(settings.staging_dir / "flood_risk.parquet"),
        "heavy_rain_r35mm": pl.read_parquet(settings.staging_dir / "heavy_rain_r35mm.parquet"),
        "stt_car_fuel": pl.read_parquet(settings.staging_dir / "stt_car_fuel.parquet"),
    }


def build_dimensions(staging: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Build all 8 dimension tables (CLAUDE.md §8 Step 1)."""
    return {
        "dim_date": build_dim_date(),
        "dim_geography": build_dim_geography(
            staging["flood_risk"], staging["heavy_rain_r35mm"], staging["stt_car_fuel"]
        ),
        "dim_vehicle": build_dim_vehicle(),
        "dim_accident_context": build_dim_accident_context(staging["accidents"]),
        "dim_time_of_day": build_dim_time_of_day(),
        "dim_weather_band": build_dim_weather_band(),
        "dim_policy_type": build_dim_policy_type(),
        "dim_customer_segment": build_dim_customer_segment(),
    }


def run_warehouse_build() -> None:
    """Phase 3: build all 8 dimensions (Step 1) then `fact_accident` (Step 2)."""
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

    con = get_connection()
    try:
        for table_name, df in dims.items():
            write_table(con, table_name, df)
        write_table(con, "fact_accident", fact_accident)
    finally:
        con.close()

    log.info(
        "warehouse_build_complete",
        tables=[*dims.keys(), "fact_accident"],
        fact_accident_rows=fact_accident.height,
    )


if __name__ == "__main__":
    run_warehouse_build()
