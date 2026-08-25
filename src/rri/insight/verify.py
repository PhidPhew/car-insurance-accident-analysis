"""The verifier -- code, not AI (CLAUDE.md Phase 8, Step 3).

This is the graded core of Phase 8: a deterministic check that the AI-written
prose says nothing it can't back up.

Design: every number is parsed straight out of the *narrative text itself*,
never trusted from the model's self-reported citation list -- the prose is
what a human reads, so the prose is what gets verified. The self-reported
citations are used only as a *hint* for which fact a prose number was meant
to represent (so a "matched fact but wrong value" flag can name the fact);
the actual pass/fail comparison always re-checks against the real fact value
from the Step 1 payload, never against the model's own claim.

A number is FLAGGED when it matches no fact in the payload within tolerance
(a fabricated/hallucinated figure), or when it matches a fact the model
cited but disagrees with that fact's true value beyond tolerance (a
misquoted figure).

One deliberate exclusion: a bare 4-digit integer in a plausible calendar-year
range (2000-2099) with no comma grouping, decimal point, or percent sign is
treated as scope framing ("in 2025, ...") rather than a measured fact, and is
excluded from parsing entirely -- CLAUDE.md's project years run 2019-2026,
and no real fact in this project is ever a bare unformatted 4-digit integer,
so this heuristic doesn't accidentally hide a real fabricated number.
"""

from __future__ import annotations

import csv
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from config.settings import settings
from rri.insight.facts import FactsPayload
from rri.insight.generate import GeneratedInsight
from rri.utils.logging import get_logger

log = get_logger(__name__)

#: Relative tolerance for a prose number to count as matching a fact's true
#: value (CLAUDE.md Phase 8 spec: "0.5% relative, configurable").
DEFAULT_TOLERANCE = 0.005

#: Bare 4-digit integers in this range are treated as calendar-year framing,
#: not measured facts -- see module docstring.
_YEAR_RANGE = range(2000, 2100)

_NUMBER_RE = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|-?\d+\.\d+%?|-?\d+%?)(?!\w)")

MatchStatus = Literal["matched", "flagged_mismatch", "flagged_unmatched"]


class ParsedNumber(BaseModel):
    """One numeric token found in the narrative prose."""

    raw_text: str
    value: float
    is_percent: bool


class NumberMatch(BaseModel):
    """The verification outcome for one parsed number."""

    raw_text: str
    parsed_value: float
    matched_fact_key: str | None
    fact_value: float | None
    relative_error: float | None
    status: MatchStatus


class VerificationRecord(BaseModel):
    """The per-insight verification summary logged to the audit trail."""

    scope_label: str
    model: str
    narrative: str
    tolerance: float
    total_numbers: int
    matched: int
    flagged: int
    hallucination_rate: float
    matches: list[NumberMatch]


def _is_calendar_year_token(raw_text: str) -> bool:
    if "%" in raw_text or "," in raw_text or "." in raw_text or raw_text.startswith("-"):
        return False
    return len(raw_text) == 4 and int(raw_text) in _YEAR_RANGE


def parse_numbers(text: str) -> list[ParsedNumber]:
    """Extract every numeric token from `text`, excluding bare calendar years."""
    numbers: list[ParsedNumber] = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0)
        if _is_calendar_year_token(raw):
            continue
        is_percent = raw.endswith("%")
        cleaned = raw.rstrip("%").replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        numbers.append(ParsedNumber(raw_text=raw, value=value, is_percent=is_percent))
    return numbers


def _relative_error(parsed_value: float, fact_value: float) -> float:
    denom = abs(fact_value) if fact_value != 0 else 1e-9
    return abs(abs(parsed_value) - abs(fact_value)) / denom


def _find_cited_fact_key(
    parsed: ParsedNumber, insight: GeneratedInsight, facts: FactsPayload
) -> str | None:
    """Which fact_key, if any, did the model's own citation list claim this
    number represents? Matched by parsing the model's self-reported
    `value_as_used` string the same way as prose text and comparing exactly
    (these are supposed to be verbatim copies of what's in the narrative)."""
    facts_by_key = facts.as_dict()
    for citation in insight.self_reported_citations:
        if citation.fact_key not in facts_by_key:
            continue
        cited_numbers = parse_numbers(citation.value_as_used)
        for cited in cited_numbers:
            if abs(cited.value) == abs(parsed.value) or (
                cited.value != 0 and _relative_error(parsed.value, cited.value) < 1e-6
            ):
                return citation.fact_key
    return None


def _find_best_value_match(
    parsed: ParsedNumber, facts: FactsPayload, tolerance: float
) -> str | None:
    """Fallback: find the fact whose true value is numerically closest to
    `parsed`, within tolerance, regardless of what (if anything) the model
    claimed to cite."""
    best_key: str | None = None
    best_err: float | None = None
    for fact in facts.facts:
        err = _relative_error(parsed.value, fact.value)
        if err <= tolerance and (best_err is None or err < best_err):
            best_key = fact.key
            best_err = err
    return best_key


def _verify_one_number(
    parsed: ParsedNumber,
    insight: GeneratedInsight,
    facts: FactsPayload,
    tolerance: float,
) -> NumberMatch:
    facts_by_key = facts.as_dict()

    cited_key = _find_cited_fact_key(parsed, insight, facts)
    if cited_key is not None:
        fact = facts_by_key[cited_key]
        err = _relative_error(parsed.value, fact.value)
        status: MatchStatus = "matched" if err <= tolerance else "flagged_mismatch"
        return NumberMatch(
            raw_text=parsed.raw_text,
            parsed_value=parsed.value,
            matched_fact_key=cited_key,
            fact_value=fact.value,
            relative_error=err,
            status=status,
        )

    value_key = _find_best_value_match(parsed, facts, tolerance)
    if value_key is not None:
        fact = facts_by_key[value_key]
        err = _relative_error(parsed.value, fact.value)
        return NumberMatch(
            raw_text=parsed.raw_text,
            parsed_value=parsed.value,
            matched_fact_key=value_key,
            fact_value=fact.value,
            relative_error=err,
            status="matched",
        )

    return NumberMatch(
        raw_text=parsed.raw_text,
        parsed_value=parsed.value,
        matched_fact_key=None,
        fact_value=None,
        relative_error=None,
        status="flagged_unmatched",
    )


def verify_insight(
    insight: GeneratedInsight,
    facts: FactsPayload,
    tolerance: float = DEFAULT_TOLERANCE,
) -> VerificationRecord:
    """Re-derive every number in `insight.narrative` from scratch and check
    it against `facts` -- the deterministic, non-AI audit."""
    parsed_numbers = parse_numbers(insight.narrative)
    matches = [_verify_one_number(p, insight, facts, tolerance) for p in parsed_numbers]

    total = len(matches)
    flagged = sum(1 for m in matches if m.status != "matched")
    matched = total - flagged
    hallucination_rate = (flagged / total) if total else 0.0

    record = VerificationRecord(
        scope_label=insight.scope_label,
        model=insight.model,
        narrative=insight.narrative,
        tolerance=tolerance,
        total_numbers=total,
        matched=matched,
        flagged=flagged,
        hallucination_rate=hallucination_rate,
        matches=matches,
    )
    log.info(
        "insight.verify.completed",
        scope=insight.scope_label,
        total_numbers=total,
        matched=matched,
        flagged=flagged,
        hallucination_rate=round(hallucination_rate, 4),
    )
    return record


_LOG_COLUMNS = [
    "timestamp",
    "scope_label",
    "model",
    "tolerance",
    "total_numbers",
    "matched",
    "flagged",
    "hallucination_rate",
    "flagged_details",
]


def log_verification(record: VerificationRecord, log_path: Path | None = None) -> Path:
    """Append `record` to the CSV audit trail (CLAUDE.md Phase 8 Step 3)."""
    resolved = log_path or (settings.insights_dir / "ai_verification_log.csv")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    write_header = not resolved.exists()

    flagged_details = "; ".join(
        f"{m.raw_text!r} (fact={m.matched_fact_key or 'none'})"
        for m in record.matches
        if m.status != "matched"
    )

    with resolved.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_LOG_COLUMNS)
        writer.writerow(
            [
                datetime.now(UTC).isoformat(),
                record.scope_label,
                record.model,
                record.tolerance,
                record.total_numbers,
                record.matched,
                record.flagged,
                round(record.hallucination_rate, 6),
                flagged_details,
            ]
        )
    return resolved
