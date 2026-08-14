"""`dim_vehicle` builder (CLAUDE.md §8, Phase 3 Step 1). Type 1 -- 5 business groups.

Neither raw vehicle taxonomy (`accidents.รถคันที่1`'s 15 free-text values,
or the DLT `ประเภทรถ` รย.1-18 registration codes in `stt_car_fuel`/
`car_registered_new`) has a shared code between the two systems, so both
map onto the same 5 canonical groups via the single committed table
`config/mappings/vehicle_groups.csv` (built for this phase, hand-authored
the same way Phase 2's cause/crash-type/road-geometry tables were --
committed lookup, never fuzzy-matched, CLAUDE.md P4).

`has_normalized_exposure` records CLAUDE.md §7.2: `stt_car_fuel` only
covers `ประเภทกฎหมาย = รถยนต์`, so exposure-normalized rates are only
meaningful for the 3 groups it actually registers (pickup, passenger_car,
motorcycle -- 78% of accident cases). `heavy_truck_bus` and `other` must be
reported as raw counts and flagged, never divided by an exposure this
source doesn't have.
"""

from __future__ import annotations

import polars as pl

from config.settings import settings
from rri.warehouse.keys import prepend_unknown_member

_EXPOSURE_NORMALIZED_GROUPS = frozenset({"pickup", "passenger_car", "motorcycle"})

#: Business-group order + label -- the taxonomy CSV supplies the mapping,
#: this fixes the dimension's row order and display label deterministically.
_GROUP_LABELS: dict[str, str] = {
    "motorcycle": "Motorcycle",
    "passenger_car": "Passenger Car / Van",
    "pickup": "Pickup Truck",
    "heavy_truck_bus": "Heavy Truck / Bus",
    "other": "Other (pedestrian, bicycle, agricultural, three-wheeler)",
}


def build_dim_vehicle() -> pl.DataFrame:
    """Build the finished `dim_vehicle` table, including the Unknown Member row."""
    groups = list(_GROUP_LABELS.keys())
    df = pl.DataFrame({"vehicle_group": groups}).with_columns(
        pl.col("vehicle_group").replace_strict(_GROUP_LABELS).alias("vehicle_group_label"),
        pl.col("vehicle_group")
        .is_in(list(_EXPOSURE_NORMALIZED_GROUPS))
        .alias("has_normalized_exposure"),
    )
    df = df.with_row_index("vehicle_sk", offset=1).with_columns(pl.col("vehicle_sk").cast(pl.Int64))

    return prepend_unknown_member(
        df,
        "vehicle_sk",
        overrides={
            "vehicle_group": "unknown",
            "vehicle_group_label": "Unknown",
            "has_normalized_exposure": False,
        },
    )


def vehicle_group_mapping_path() -> str:
    """Path to the committed accident-side + DLT-side raw-value -> group CSV."""
    return str(settings.mappings_dir / "vehicle_groups.csv")
