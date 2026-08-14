"""`dim_policy_type` scaffold (CLAUDE.md §8, Phase 3 Step 1 / Phase 4 Step 5).

Phase 3 only creates the table shape (schema + Unknown Member row) so the
warehouse has a stable target to FK against; Phase 4 Step 5 calls
`build_dim_policy_type(members=...)` again with the real cover-class rows
(sourced from `config/simulation_params.py`'s market-share constants) and
reloads the table -- same function, so the schema can't drift between the
two phases.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from rri.warehouse.keys import assign_surrogate_keys, prepend_unknown_member

_SCHEMA = {
    "cover_class": pl.Utf8,
    "cover_class_label": pl.Utf8,
    "is_comprehensive": pl.Boolean,
}


def build_dim_policy_type(members: list[dict[str, Any]] | None = None) -> pl.DataFrame:
    """Build `dim_policy_type`. Empty (Unknown Member only) until Phase 4 supplies `members`."""
    df = pl.DataFrame(members or [], schema=_SCHEMA)
    df = assign_surrogate_keys(df, "policy_type_sk")

    return prepend_unknown_member(
        df,
        "policy_type_sk",
        overrides={
            "cover_class": "unknown",
            "cover_class_label": "Unknown",
            "is_comprehensive": False,
        },
    )
