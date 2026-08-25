"""AI Insight panel support (CLAUDE.md Phase 8, Step 4).

Default path: read the committed cache (`data/60_insights/insights_cache.json`)
-- no Anthropic API call, no key needed at serve time, works on Streamlit
Community Cloud exactly as deployed. A "Regenerate live" fallback is offered
only when a scope isn't in the fixed cache and an API key happens to be
available (local dev, or a key added to Streamlit secrets) -- CLAUDE.md
Phase 8: "the default path is the cache."

The verification badge is the point of this panel: every cached narrative
carries its own real VerificationRecord from Step 3, never a hardcoded
number.
"""

from __future__ import annotations

import os

import duckdb
import streamlit as st

from config.settings import settings
from rri.insight.cache import CachedInsight, InsightCache, load_cache
from rri.insight.facts import InsightScope, build_facts, get_connection
from rri.insight.generate import GeneratedInsight, api_key_available, generate_insight
from rri.insight.verify import VerificationRecord, verify_insight


@st.cache_resource(show_spinner=False)
def _facts_connection() -> duckdb.DuckDBPyConnection:
    return get_connection(str(settings.warehouse_db_path))


@st.cache_resource(show_spinner=False)
def get_insight_cache() -> InsightCache | None:
    try:
        return load_cache()
    except FileNotFoundError:
        return None


def _resolve_live_api_key() -> str | None:
    """Local dev: `.env` -> `ANTHROPIC_API_KEY` (via `generate.api_key_available()`,
    which loads `.env` into the process as a side effect). On Streamlit
    Community Cloud: an org/app owner can add the same key to `st.secrets`
    instead -- neither is required for the default cached path."""
    if api_key_available():
        return os.environ.get("ANTHROPIC_API_KEY")
    try:
        secret = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # noqa: BLE001 -- st.secrets raises if no secrets.toml exists at all
        secret = None
    return str(secret) if secret else None


def _render_verification_badge(verification: VerificationRecord) -> None:
    rate_pct = verification.hallucination_rate * 100
    if verification.flagged == 0:
        st.success(
            f"✓ Verified — {verification.matched}/{verification.total_numbers} numbers "
            f"checked against the warehouse, 0 flagged (hallucination rate: {rate_pct:.1f}%)."
        )
    else:
        st.error(
            f"⚠ {verification.flagged}/{verification.total_numbers} numbers flagged "
            f"(hallucination rate: {rate_pct:.1f}%) — see details below."
        )
        with st.expander("Flagged numbers"):
            for m in verification.matches:
                if m.status == "matched":
                    continue
                if m.status == "flagged_mismatch":
                    st.markdown(
                        f"- `{m.raw_text}` claimed for **{m.matched_fact_key}**, "
                        f"but the warehouse value is `{m.fact_value}` "
                        f"(relative error {m.relative_error:.1%})"
                    )
                else:
                    st.markdown(f"- `{m.raw_text}` — matches no fact in the warehouse")
    st.caption(
        f"Verifier: every number re-parsed from the narrative text itself and checked "
        f"against `fact_accident`/`fact_underwriting_monthly` within "
        f"{verification.tolerance:.1%} relative tolerance — never the model's own claim."
    )


def _render_entry(entry: CachedInsight, *, cached: bool) -> None:
    st.markdown(f"**{entry.scope_label}**")
    st.write(entry.insight.narrative)
    _render_verification_badge(entry.verification)
    source = "cached (committed at build time)" if cached else "generated live, just now"
    st.caption(f"Model: {entry.insight.model} · Source: {source}")


def render_insight_panel(
    *,
    title: str,
    scope_key: str | None,
    live_scope: InsightScope | None = None,
) -> None:
    """Render the AI Insight panel for one scope.

    `scope_key` looks up the committed cache first (the default, no-API-call
    path). If it's missing from the cache and `live_scope` is given, offer a
    "Regenerate live" button -- only enabled when an API key is resolvable.
    """
    st.markdown(f"#### {title}")

    cache = get_insight_cache()
    cached_entry = cache.by_scope_key().get(scope_key) if cache and scope_key else None

    if cached_entry is not None:
        _render_entry(cached_entry, cached=True)
        return

    if live_scope is None:
        st.info("No cached AI insight for this selection.")
        return

    st.info("No cached AI insight for this selection.")
    api_key = _resolve_live_api_key()
    if not api_key:
        st.caption(
            "Live generation requires an Anthropic API key (not configured in this "
            "deployment) — showing cached scopes only."
        )
        return

    if st.button("Regenerate live for this selection", key=f"regen_{live_scope.label}"):
        with st.spinner("Calling Claude and verifying the result..."):
            try:
                con = _facts_connection()
                facts = build_facts(con, live_scope)
                insight: GeneratedInsight = generate_insight(facts, api_key=api_key)
                verification = verify_insight(insight, facts)
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
                st.error(f"Live generation failed: {exc}")
                return
        st.write(insight.narrative)
        _render_verification_badge(verification)
        st.caption(f"Model: {insight.model} · Source: generated live, just now (not cached).")
