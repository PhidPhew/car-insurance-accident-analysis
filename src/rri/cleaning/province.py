"""Normalize province names from every raw source to the canonical province_th.

Every accident/rainfall/flood/registration/TAT record refers to a province
under a different spelling convention (plain Thai, "จ."-prefixed Thai,
English, informal Bangkok abbreviations). `normalize_province` is the single
function all warehouse-building code should route through so a province
is only ever matched one way.

It does NOT use fuzzy string matching: CLAUDE.md Task 3 explicitly forbids
shipping fuzzy matches as the final normalization step, since fuzzy matching
can silently merge two different provinces that merely look similar. Instead
it does exact lookups against two committed, human-reviewed tables:
config/province_master.csv (the 77 canonical Thai names) and
config/mappings/province_aliases.csv (every known non-canonical spelling,
each traced to a real source file - see scripts/build_province_aliases.py).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass

from config.settings import settings

_BANGKOK_PREFIX_EXCEPTION = "กรุงเทพ"  # never gets a "จ." prefix stripped from it
_PROVINCE_PREFIXES = ("จังหวัด", "จ.")
_WHITESPACE_RE = re.compile(r"\s+")


def _load_canonical_names() -> frozenset[str]:
    path = settings.config_dir / "province_master.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return frozenset(row["province_th"].strip() for row in reader)


def load_canonical_province_names() -> frozenset[str]:
    """Public accessor for the 77 canonical `province_th` values (config/province_master.csv).

    Used by data contracts to assert the master list's size and membership
    rather than re-reading the CSV independently.
    """
    return _CANONICAL_NAMES


def _load_alias_map() -> dict[str, str]:
    path = settings.mappings_dir / "province_aliases.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {row["alias"].strip(): row["province_th"].strip() for row in reader}


_CANONICAL_NAMES = _load_canonical_names()
_ALIAS_MAP = _load_alias_map()


def _strip_prefix(name: str) -> str:
    if name.startswith(_BANGKOK_PREFIX_EXCEPTION):
        return name
    for prefix in _PROVINCE_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _normalize_whitespace(name: str) -> str:
    return _WHITESPACE_RE.sub(" ", name).strip()


def normalize_province(name: str | None) -> str | None:
    """Return the canonical province_th for a raw province-name string.

    Returns None when `name` is empty/whitespace-only/None, or when it
    doesn't match a known province or alias after normalization -- callers
    are expected to route unmatched values to the Unknown Member (sk = -1)
    per CLAUDE.md N7 rather than dropping the row, and to surface unmatched
    values for review (see reports/province_matching_report.html) rather
    than silently discarding them.
    """
    if name is None:
        return None

    cleaned = _normalize_whitespace(_strip_prefix(name.strip()))
    if not cleaned:
        return None

    if cleaned in _CANONICAL_NAMES:
        return cleaned

    if cleaned in _ALIAS_MAP:
        return _ALIAS_MAP[cleaned]

    return None


@dataclass(frozen=True)
class ProvinceMatchResult:
    """Match-rate summary for one source's raw province values against the master."""

    source_name: str
    total_distinct_values: int
    matched_count: int
    unmatched_values: tuple[str, ...]

    @property
    def match_rate(self) -> float:
        if self.total_distinct_values == 0:
            return 0.0
        return self.matched_count / self.total_distinct_values


def match_province_values(source_name: str, raw_values: list[str]) -> ProvinceMatchResult:
    """Summarize how many of a source's distinct raw province strings normalize successfully."""
    distinct = sorted({v for v in raw_values if v and v.strip()})
    unmatched = tuple(v for v in distinct if normalize_province(v) is None)
    return ProvinceMatchResult(
        source_name=source_name,
        total_distinct_values=len(distinct),
        matched_count=len(distinct) - len(unmatched),
        unmatched_values=unmatched,
    )
