"""`dim_geography` builder (CLAUDE.md §8, Phase 3 Step 1). SCD Type 2, 77 provinces + Unknown.

Grain: one row per province per SCD version. Every source here is a single
snapshot (no history to version yet), so this build produces exactly one
current version per province (`valid_from` = today's build, `valid_to` =
null, `is_current` = true) -- the SCD2 columns exist so a future incremental
load has somewhere to close a version out, per the rigor checklist's
"incremental load via watermark" item.

Attributes, one helper per source, each returning `province_th` + its own
columns so they left-join cleanly onto the 77-row canonical spine
(`config/province_master.csv`):

- flood risk: `thailand_flood_risk_areas.csv`, aggregated tambon -> province
  (CLAUDE.md P6: 75 of 77 provinces present -- Phuket/Ranong get
  `flood_data_available = false` and null flood attributes, not sk = -1;
  they're still real provinces, just missing this one source).
- R35mm: `thailand_heavy_rain_R35mm.csv`, averaged over 2014-2023 only
  (CLAUDE.md P7 -- the years CLAUDE.md scoped this source to).
- EV share / motorcycle share: `stt_car_fuel_at_25690228.csv`, a single
  2026-02-28 snapshot (CLAUDE.md §7.1's exposure-snapshot limitation
  applies here too). P7.12: 697 rows have both count and fuel type null --
  filled 0 here (province registers none of that combination), not
  dropped.
- tourist attraction count: `tat_tourist_attractions.json`, joined via
  `config/mappings/tat_province_id_crosswalk.csv` (P7.7: `ATT_ID`/
  `ATT_PROVINCE_ID` read as strings to avoid the float/int overflow already
  found in Phase 1). Named `tourist_attraction_count`, not "density" --
  no source here has province land area or population to divide by, so a
  true density figure isn't computable from what we have. This raw count
  is the intentional stand-in.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import polars as pl

from config.settings import settings
from rri.cleaning.province import normalize_province
from rri.cleaning.taxonomy import apply_taxonomy, load_taxonomy_mapping
from rri.warehouse.keys import prepend_unknown_member

R35MM_YEAR_START = 2014
R35MM_YEAR_END = 2023

_HIGH_RISK_LABEL = "เสี่ยงสูง"


def _province_spine() -> pl.DataFrame:
    path = settings.config_dir / "province_master.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return pl.DataFrame(rows).rename({"province_id": "province_id", "province_th": "province_th"})


def _flood_attrs(flood_risk_df: pl.DataFrame) -> pl.DataFrame:
    df = flood_risk_df.with_columns(
        pl.col("PROV_T")
        .map_elements(normalize_province, return_dtype=pl.Utf8)
        .alias("province_th"),
        pl.col("Month").cast(pl.Int32),
    ).filter(pl.col("province_th").is_not_null())

    tambon_totals = df.group_by("province_th").agg(
        pl.col("TAMBON_T").n_unique().alias("_n_tambon_total")
    )
    high_risk = df.filter(pl.col("RISK") == _HIGH_RISK_LABEL)
    tambon_high_risk = high_risk.group_by("province_th").agg(
        pl.col("TAMBON_T").n_unique().alias("_n_tambon_high_risk")
    )
    peak_month = (
        high_risk.group_by(["province_th", "Month"])
        .agg(pl.len().alias("_n"))
        .sort(["province_th", "_n", "Month"], descending=[False, True, False])
        .group_by("province_th", maintain_order=True)
        .first()
        .select("province_th", pl.col("Month").alias("flood_peak_month"))
    )

    out = (
        tambon_totals.join(tambon_high_risk, on="province_th", how="left")
        .join(peak_month, on="province_th", how="left")
        .with_columns(
            pl.col("_n_tambon_high_risk").fill_null(0),
            (pl.col("_n_tambon_high_risk").fill_null(0) / pl.col("_n_tambon_total")).alias(
                "flood_pct_high_risk_tambons"
            ),
            pl.lit(True).alias("flood_data_available"),
        )
        .select(
            "province_th", "flood_pct_high_risk_tambons", "flood_peak_month", "flood_data_available"
        )
    )
    return out


def _r35mm_attrs(heavy_rain_df: pl.DataFrame) -> pl.DataFrame:
    df = heavy_rain_df.with_columns(
        pl.col("Province_Name")
        .map_elements(normalize_province, return_dtype=pl.Utf8)
        .alias("province_th"),
        pl.col("Year").cast(pl.Int32),
        pl.col("R35mm (Day)").cast(pl.Float64, strict=False).alias("r35mm_days"),
    ).filter(
        pl.col("province_th").is_not_null(),
        pl.col("Year").is_between(R35MM_YEAR_START, R35MM_YEAR_END),
    )
    return df.group_by("province_th").agg(
        pl.col("r35mm_days").mean().alias("r35mm_avg_days_2014_2023")
    )


def _fuel_share_attrs(stt_car_fuel_df: pl.DataFrame) -> pl.DataFrame:
    vehicle_mapping = load_taxonomy_mapping(settings.mappings_dir / "vehicle_groups.csv")
    df = stt_car_fuel_df.with_columns(
        pl.col("จำนวนรถ").cast(pl.Int64, strict=False).fill_null(0).alias("_count"),
        pl.col("จังหวัด").map_elements(normalize_province, return_dtype=pl.Utf8).alias("province_th"),
    ).filter(pl.col("province_th").is_not_null())

    df, _log = apply_taxonomy(df, "ประเภทรถ", vehicle_mapping, new_column="_vehicle_group")

    totals = df.group_by("province_th").agg(pl.col("_count").sum().alias("_total"))
    ev = (
        df.filter(pl.col("ประเภทเชื้อเพลิง").str.contains("ไฟฟ้า").fill_null(False))
        .group_by("province_th")
        .agg(pl.col("_count").sum().alias("_ev_count"))
    )
    motorcycle = (
        df.filter(pl.col("_vehicle_group") == "motorcycle")
        .group_by("province_th")
        .agg(pl.col("_count").sum().alias("_motorcycle_count"))
    )

    out = (
        totals.join(ev, on="province_th", how="left")
        .join(motorcycle, on="province_th", how="left")
        .with_columns(
            pl.col("_ev_count").fill_null(0),
            pl.col("_motorcycle_count").fill_null(0),
        )
        .with_columns(
            (pl.col("_ev_count") / pl.col("_total")).alias("ev_share"),
            (pl.col("_motorcycle_count") / pl.col("_total")).alias("motorcycle_share"),
        )
        .select("province_th", "ev_share", "motorcycle_share")
    )
    return out


def _tat_attraction_counts(tat_json_path: Path) -> pl.DataFrame:
    with tat_json_path.open(encoding="utf-8") as f:
        records = json.load(f)

    crosswalk_path = settings.mappings_dir / "tat_province_id_crosswalk.csv"
    with crosswalk_path.open(encoding="utf-8-sig", newline="") as f:
        crosswalk = {row["att_province_id"]: row["province_id"] for row in csv.DictReader(f)}

    province_by_id_path = settings.config_dir / "province_master.csv"
    with province_by_id_path.open(encoding="utf-8-sig", newline="") as f:
        province_th_by_id = {row["province_id"]: row["province_th"] for row in csv.DictReader(f)}

    att_province_ids = [
        str(r.get("ATT_PROVINCE_ID")) for r in records if r.get("ATT_PROVINCE_ID") is not None
    ]
    counts: dict[str, int] = {}
    for att_id in att_province_ids:
        province_id = crosswalk.get(att_id)
        if province_id is None:
            continue  # the documented junk att_province_id "1" (P6) -- not a real province
        province_th = province_th_by_id[province_id]
        counts[province_th] = counts.get(province_th, 0) + 1

    return pl.DataFrame(
        {
            "province_th": list(counts.keys()),
            "tourist_attraction_count": list(counts.values()),
        },
        schema={"province_th": pl.Utf8, "tourist_attraction_count": pl.Int64},
    )


def build_dim_geography(
    flood_risk_df: pl.DataFrame,
    heavy_rain_df: pl.DataFrame,
    stt_car_fuel_df: pl.DataFrame,
    tat_json_path: Path | None = None,
    build_date: str = "2026-01-01",
) -> pl.DataFrame:
    """Build the finished `dim_geography` table, including the Unknown Member row."""
    spine = _province_spine()
    tat_path = tat_json_path or (settings.raw_dir / "tat_tourist_attractions.json")

    df = (
        spine.join(_flood_attrs(flood_risk_df), on="province_th", how="left")
        .join(_r35mm_attrs(heavy_rain_df), on="province_th", how="left")
        .join(_fuel_share_attrs(stt_car_fuel_df), on="province_th", how="left")
        .join(_tat_attraction_counts(tat_path), on="province_th", how="left")
        .with_columns(
            pl.col("flood_data_available").fill_null(False),
            pl.col("tourist_attraction_count").fill_null(0),
        )
        .with_columns(
            pl.lit(build_date).str.to_date().alias("valid_from"),
            pl.lit(None, dtype=pl.Date).alias("valid_to"),
            pl.lit(True).alias("is_current"),
        )
        .sort("province_id")
    )

    df = df.with_row_index("geography_sk", offset=1).with_columns(
        pl.col("geography_sk").cast(pl.Int64)
    )

    return prepend_unknown_member(
        df,
        "geography_sk",
        overrides={
            "province_id": "-1",
            "province_th": "Unknown",
            "province_en": "Unknown",
            "region": "Unknown",
            "flood_data_available": False,
            "tourist_attraction_count": 0,
            "is_current": True,
        },
    )
