"""Generate the committed AI insight cache (Phase 8 Step 4).

Runs the fixed scope set through generation + verification once and writes
`data/60_insights/insights_cache.json` -- the file the deployed dashboard
reads at serve time (no live Anthropic API call on Streamlit Community
Cloud, no key baked into the public repo).

Run: ``uv run python scripts/generate_insight_cache.py``
Requires ``ANTHROPIC_API_KEY`` in `.env` or the environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rri.insight.cache import build_insight_cache, save_cache  # noqa: E402
from rri.insight.facts import get_connection  # noqa: E402
from rri.insight.generate import api_key_available  # noqa: E402
from rri.utils.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def main() -> None:
    configure_logging()
    if not api_key_available():
        raise SystemExit(
            "ANTHROPIC_API_KEY not found in the environment or .env -- "
            "set it before running this script."
        )

    con = get_connection()
    cache = build_insight_cache(con)
    path = save_cache(cache)

    log.info("insight.cache.generated", path=str(path), n_entries=len(cache.entries))
    for entry in cache.entries:
        log.info(
            "insight.cache.summary",
            scope_key=entry.scope_key,
            hallucination_rate=round(entry.verification.hallucination_rate, 4),
            total_numbers=entry.verification.total_numbers,
            flagged=entry.verification.flagged,
        )


if __name__ == "__main__":
    main()
