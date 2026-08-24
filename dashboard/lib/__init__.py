"""Shared dashboard infrastructure — data access, filters, theme, formatting.

Nothing here recomputes cleaning, features, or model training (CLAUDE.md
Phase 7 rule): this package only reads the built warehouse (`data/30_warehouse/`)
and calls the persisted models' `predict()` interface (`rri.models.persistence`).
"""

from __future__ import annotations
