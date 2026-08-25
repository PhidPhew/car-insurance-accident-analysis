"""Constrained AI narrative generation (CLAUDE.md Phase 8, Step 2).

Turns a :class:`~rri.insight.facts.FactsPayload` into a short English prose
caption. The model is never allowed to compute a number -- it is handed the
closed set of facts from `facts.py` and instructed to cite only those. The
model's own self-reported citations are recorded but are *not* trusted as
verification -- `verify.py` re-parses the numbers straight out of the prose,
because the prose is what a human actually reads.

The Anthropic API key is never hardcoded: `Anthropic()` resolves it from the
`ANTHROPIC_API_KEY` environment variable (loaded from `.env` locally via
`python-dotenv`, or from Streamlit secrets on Cloud -- see `dashboard/`).
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from config.settings import settings
from rri.insight.facts import FactsPayload
from rri.utils.logging import get_logger

log = get_logger(__name__)

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    """Load `.env` into the process environment once, if present.

    A no-op when there is no `.env` file (e.g. on Streamlit Community Cloud,
    which supplies `ANTHROPIC_API_KEY` via `st.secrets` instead -- the
    dashboard layer is responsible for setting `os.environ` from secrets
    before calling into this module).
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv(Path(settings.base_dir) / ".env")
    _ENV_LOADED = True


class CitedFact(BaseModel):
    """One number the model claims to have used, as it self-reported it."""

    fact_key: str
    value_as_used: str


class GeneratedInsight(BaseModel):
    """The model's output: prose plus its own (untrusted) citation list."""

    narrative: str
    self_reported_citations: list[CitedFact] = Field(default_factory=list)
    model: str
    scope_label: str


_SYSTEM_PROMPT = """\
You are writing a short insight caption for an insurance analytics dashboard.

You will be given a JSON payload of facts: a closed set of numbers, each with
a stable key, a numeric value, a unit, and a description. This is the ONLY
source of numeric information you may use.

Rules, strictly enforced:
1. Use ONLY the numbers given in the facts payload. Do not compute, derive,
   round differently, convert units, or invent any number that is not
   present verbatim (or trivially reformatted, e.g. adding a % sign to a
   percent value) in the payload.
2. Every numeric claim in your narrative must correspond exactly to one of
   the provided facts' values.
3. Do not speculate about causation beyond what the facts support. This is a
   descriptive caption, not a causal claim.
4. Write in English, 2-4 sentences, suitable as a dashboard caption -- not an
   essay.
5. After writing the narrative, call the `submit_insight` tool with the
   narrative text and a list of every (fact_key, value_as_used) pair you
   cited, one entry per number you mentioned.
"""

_SUBMIT_INSIGHT_TOOL: anthropic.types.ToolParam = {
    "name": "submit_insight",
    "description": "Submit the finished insight narrative and its fact citations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "narrative": {
                "type": "string",
                "description": "The 2-4 sentence English insight caption.",
            },
            "cited_facts": {
                "type": "array",
                "description": "Every fact_key/value pair cited in the narrative.",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact_key": {"type": "string"},
                        "value_as_used": {
                            "type": "string",
                            "description": "The number as it appears in the narrative text.",
                        },
                    },
                    "required": ["fact_key", "value_as_used"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["narrative", "cited_facts"],
        "additionalProperties": False,
    },
}


def generate_insight(
    facts: FactsPayload,
    api_key: str | None = None,
    model: str | None = None,
) -> GeneratedInsight:
    """Call the Anthropic API to turn `facts` into a narrative caption.

    Raises whatever `anthropic` raises on API failure -- callers (the Step 4
    cache-generation script, the dashboard's "regenerate" button) decide how
    to handle that; this function does not swallow errors.
    """
    resolved_model = model or settings.insight_model
    if api_key is None:
        _ensure_env_loaded()
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    payload_json = facts.model_dump_json(indent=2)
    response = client.messages.create(
        model=resolved_model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        tools=[_SUBMIT_INSIGHT_TOOL],
        tool_choice={"type": "tool", "name": "submit_insight"},
        messages=[
            {
                "role": "user",
                "content": (f"Scope: {facts.scope_label}\n\nFacts payload:\n{payload_json}"),
            }
        ],
    )

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        raise RuntimeError(
            f"Expected a submit_insight tool call, got stop_reason={response.stop_reason!r}"
        )
    tool_input = tool_use.input
    assert isinstance(tool_input, dict)

    narrative = str(tool_input["narrative"])
    cited_facts_raw = tool_input.get("cited_facts") or []
    assert isinstance(cited_facts_raw, list)
    cited = [
        CitedFact(fact_key=str(c["fact_key"]), value_as_used=str(c["value_as_used"]))
        for c in cited_facts_raw
    ]

    log.info(
        "insight.generate.completed",
        scope=facts.scope_label,
        model=resolved_model,
        n_self_reported_citations=len(cited),
    )

    return GeneratedInsight(
        narrative=narrative,
        self_reported_citations=cited,
        model=resolved_model,
        scope_label=facts.scope_label,
    )


def api_key_available() -> bool:
    """Whether an Anthropic API key is resolvable right now (env or .env)."""
    _ensure_env_loaded()
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
