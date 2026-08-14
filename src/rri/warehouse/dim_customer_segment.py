"""`dim_customer_segment` scaffold (CLAUDE.md §8, Phase 3 Step 1 / Phase 4 Step 5).

Same pattern as `dim_policy_type`: Phase 3 creates the table shape (schema +
Unknown Member row) only; Phase 4 Step 5 calls `build_dim_customer_segment
(members=...)` again with the real simulated segment rows and reloads the
table.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from rri.warehouse.keys import assign_surrogate_keys, prepend_unknown_member

_SCHEMA = {
    "age_band": pl.Utf8,
    "gender": pl.Utf8,
    "customer_type": pl.Utf8,
}


def build_dim_customer_segment(members: list[dict[str, Any]] | None = None) -> pl.DataFrame:
    """Build `dim_customer_segment`. Empty (Unknown Member only) until Phase 4 adds `members`."""
    df = pl.DataFrame(members or [], schema=_SCHEMA)
    df = assign_surrogate_keys(df, "customer_segment_sk")

    return prepend_unknown_member(
        df,
        "customer_segment_sk",
        overrides={
            "age_band": "Unknown",
            "gender": "Unknown",
            "customer_type": "Unknown",
        },
    )
