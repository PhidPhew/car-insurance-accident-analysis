"""Unit tests for `rri.insight.verify` (CLAUDE.md Phase 8, Step 3 -- the
graded core: proving numbers get checked, not trusted)."""

from __future__ import annotations

from rri.insight.facts import Fact, FactsPayload
from rri.insight.generate import CitedFact, GeneratedInsight
from rri.insight.verify import (
    DEFAULT_TOLERANCE,
    parse_numbers,
    verify_insight,
)


def _facts() -> FactsPayload:
    return FactsPayload(
        scope_label="Test scope",
        facts=[
            Fact(key="loss_ratio", value=65.3, unit="percent", description="loss ratio"),
            Fact(key="accident_count", value=23715.0, unit="accidents", description="accidents"),
            Fact(
                key="severity_index_coarse_yoy_pct_change",
                value=-5.23,
                unit="percent",
                description="YoY change",
            ),
        ],
    )


def _insight(narrative: str, citations: list[CitedFact] | None = None) -> GeneratedInsight:
    return GeneratedInsight(
        narrative=narrative,
        self_reported_citations=citations or [],
        model="claude-sonnet-5",
        scope_label="Test scope",
    )


def test_parse_numbers_handles_thousands_commas_decimals_percent() -> None:
    numbers = parse_numbers("23,715 accidents, severity 6.46, down 5.23%.")
    values = {n.value for n in numbers}
    assert 23715.0 in values
    assert 6.46 in values
    assert 5.23 in values


def test_parse_numbers_excludes_bare_calendar_years() -> None:
    numbers = parse_numbers("In 2025, accidents totalled 23,715.")
    values = [n.raw_text for n in numbers]
    assert "2025" not in values
    assert "23,715" in values


def test_parse_numbers_keeps_four_digit_number_with_comma_or_percent() -> None:
    # A comma-grouped or percent-suffixed 4-digit-looking number is not a
    # bare year token and must still be parsed.
    numbers = parse_numbers("2,025 accidents and a rate of 2025%.")
    raw_texts = [n.raw_text for n in numbers]
    assert "2,025" in raw_texts
    assert "2025%" in raw_texts


def test_verify_insight_all_numbers_match_real_facts() -> None:
    insight = _insight(
        "Accidents totalled 23,715 with a loss ratio of 65.3%.",
        citations=[
            CitedFact(fact_key="accident_count", value_as_used="23,715"),
            CitedFact(fact_key="loss_ratio", value_as_used="65.3%"),
        ],
    )
    record = verify_insight(insight, _facts())
    assert record.total_numbers == 2
    assert record.matched == 2
    assert record.flagged == 0
    assert record.hallucination_rate == 0.0


def test_verify_insight_flags_a_planted_wrong_number() -> None:
    """The checkpoint proof: change loss ratio 65.3% -> 72.1% in the prose
    while still citing the real loss_ratio fact_key -- the verifier must
    catch the disagreement, not just rubber-stamp the citation."""
    insight = _insight(
        "Accidents totalled 23,715 with a loss ratio of 72.1%.",
        citations=[
            CitedFact(fact_key="accident_count", value_as_used="23,715"),
            CitedFact(fact_key="loss_ratio", value_as_used="72.1%"),
        ],
    )
    record = verify_insight(insight, _facts())
    assert record.flagged == 1
    assert record.hallucination_rate == 0.5

    flagged = [m for m in record.matches if m.status != "matched"]
    assert len(flagged) == 1
    assert flagged[0].raw_text == "72.1%"
    assert flagged[0].matched_fact_key == "loss_ratio"
    assert flagged[0].fact_value == 65.3
    assert flagged[0].status == "flagged_mismatch"
    assert flagged[0].relative_error is not None
    assert flagged[0].relative_error > DEFAULT_TOLERANCE


def test_verify_insight_flags_a_fabricated_number_matching_no_fact() -> None:
    insight = _insight("Fatalities reached an unprecedented 99,999 this year.")
    record = verify_insight(insight, _facts())
    flagged = [m for m in record.matches if m.status != "matched"]
    assert len(flagged) == 1
    assert flagged[0].status == "flagged_unmatched"
    assert flagged[0].matched_fact_key is None


def test_verify_insight_matches_directional_sign_flip_by_absolute_value() -> None:
    """severity_index_coarse_yoy_pct_change is stored as -5.23; prose commonly
    rephrases a negative delta directionally ("down 5.23%") without a minus
    sign -- that must match, not be flagged as a fabricated number."""
    insight = _insight("Severity dropped 5.23% year-over-year.")
    record = verify_insight(insight, _facts())
    assert record.flagged == 0
    assert record.matches[0].matched_fact_key == "severity_index_coarse_yoy_pct_change"


def test_verify_insight_empty_narrative_has_zero_hallucination_rate() -> None:
    insight = _insight("No numeric claims here.")
    record = verify_insight(insight, _facts())
    assert record.total_numbers == 0
    assert record.hallucination_rate == 0.0


def test_verify_insight_does_not_trust_self_reported_citation_value() -> None:
    """The self-reported citation claims a correct value_as_used, but the
    actual prose number disagrees -- verification must go by the prose, not
    the self-report (module docstring: 'the prose is what gets verified')."""
    insight = _insight(
        "Loss ratio came in at 72.1%.",
        citations=[CitedFact(fact_key="loss_ratio", value_as_used="65.3%")],
    )
    record = verify_insight(insight, _facts())
    assert record.flagged == 1
    assert record.matches[0].raw_text == "72.1%"
