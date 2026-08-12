"""Build config/mappings/province_aliases.csv, the hand-editable alias table.

Run: ``uv run python scripts/build_province_aliases.py``

Every alias here was derived from an exact, deterministic join or set-diff
against real source data -- never from fuzzy string matching (CLAUDE.md Task 3
explicitly forbids shipping fuzzy matches as final; a human must confirm each
pair). This script's role is to assemble the *candidates* that were already
verified this way; a human (recorded in the project's AI usage log) confirmed
them before this file was committed:

- English names: R35mm's Province_Name is joined to the Thai canonical name
  via ProvinceID, the same code both rainfall and R35mm were independently
  cross-validated to share (see build_province_master.py) -- not a guess.
- "อยุธยา" -> "พระนครศรีอยุธยา": found via an exact set-diff between the
  accidents file's 77 province names and the rainfall file's 77 canonical
  names (the only mismatch found).
- Bangkok's common informal short forms (กรุงเทพฯ, กทม., กรุงเทพ) are added
  even though no current source file uses them, per CLAUDE.md Task 3's
  explicit requirement to handle them -- future sources may.

Re-running this script overwrites the file; if a maintainer hand-edits
config/mappings/province_aliases.csv directly to fix a newly-found variant,
add it to MANUAL_ALIASES below too so it survives a regeneration.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from rri.utils.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)

# alias -> canonical province_th. Confirmed manually (see module docstring),
# not auto-generated from fuzzy matching.
MANUAL_ALIASES: dict[str, str] = {
    "อยุธยา": "พระนครศรีอยุธยา",
    "กรุงเทพฯ": "กรุงเทพมหานคร",
    "กทม.": "กรุงเทพมหานคร",
    "กรุงเทพ": "กรุงเทพมหานคร",
}


def load_master() -> list[dict[str, str]]:
    path = settings.config_dir / "province_master.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_alias_rows(master_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_aliases: set[str] = set()

    for row in master_rows:
        province_th = row["province_th"]
        english = row["province_en"]
        if english and english != province_th and english not in seen_aliases:
            rows.append({"alias": english, "province_th": province_th})
            seen_aliases.add(english)

    for alias, province_th in MANUAL_ALIASES.items():
        if alias in seen_aliases:
            continue
        rows.append({"alias": alias, "province_th": province_th})
        seen_aliases.add(alias)

    rows.sort(key=lambda r: r["alias"])
    return rows


def main() -> int:
    configure_logging()
    master_rows = load_master()
    alias_rows = build_alias_rows(master_rows)

    out_path = settings.mappings_dir / "province_aliases.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alias", "province_th"])
        writer.writeheader()
        writer.writerows(alias_rows)

    log.info("province_aliases_written", path=str(out_path), rows=len(alias_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
