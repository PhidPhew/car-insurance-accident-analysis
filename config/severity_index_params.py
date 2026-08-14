"""Phase 5 `severity_index_coarse` calibration constants (CLAUDE.md §9).

`severity_index_fine`'s per-tier weights (`SEVERITY_WEIGHT_FATAL/SERIOUS/MINOR`,
`config/simulation_params.py`) were set in Phase 4 from published OIC-referenced
fatal:serious:minor cost multiples (40:8:1). `severity_index_coarse` only has a
fatal/total-injury split (no serious/minor breakdown in `accident2019.csv`, P7/§6),
so its injury weight `W_I_COARSE` blends `SEVERITY_WEIGHT_SERIOUS`/`SEVERITY_WEIGHT_MINOR`
by the real serious:minor mix observed where the breakdown *is* known
(`has_injury_breakdown = true`, 2020-2025 -- 2026 excluded as an incomplete holdout
year per P7.11):

    serious = 17,621, minor = 89,452  ->  serious_share = 0.1646, minor_share = 0.8354
    W_I_COARSE = 0.1646 * SEVERITY_WEIGHT_SERIOUS + 0.8354 * SEVERITY_WEIGHT_MINOR
               = 0.1646 * 8 + 0.8354 * 1 = 2.15

`W_F_COARSE` reuses `SEVERITY_WEIGHT_FATAL` directly -- a fatality is a fatality in
both formulas, no reason for two different fatal weights.
"""

from __future__ import annotations

from config.simulation_params import SEVERITY_WEIGHT_FATAL

W_F_COARSE = SEVERITY_WEIGHT_FATAL
W_I_COARSE = 2.15
