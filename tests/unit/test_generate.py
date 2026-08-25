"""Unit tests for `rri.insight.generate` (CLAUDE.md Phase 8, Step 2).

Mocks the Anthropic client -- these tests never hit the live API. They
verify the request shape (constrained tool use) and the response parsing,
not model output quality.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rri.insight.facts import Fact, FactsPayload
from rri.insight.generate import GeneratedInsight, generate_insight


def _facts_payload() -> FactsPayload:
    return FactsPayload(
        scope_label="Test scope",
        facts=[
            Fact(key="accident_count", value=100.0, unit="accidents", description="test"),
        ],
    )


def _mock_tool_use_response(narrative: str, cited_facts: list[dict[str, str]]) -> SimpleNamespace:
    tool_block = SimpleNamespace(
        type="tool_use",
        name="submit_insight",
        input={"narrative": narrative, "cited_facts": cited_facts},
    )
    return SimpleNamespace(content=[tool_block], stop_reason="tool_use")


def test_generate_insight_parses_tool_use_response() -> None:
    mock_response = _mock_tool_use_response(
        "Accidents totalled 100 in the test scope.",
        [{"fact_key": "accident_count", "value_as_used": "100"}],
    )
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("rri.insight.generate.anthropic.Anthropic", return_value=mock_client):
        result = generate_insight(_facts_payload(), api_key="test-key")

    assert isinstance(result, GeneratedInsight)
    assert result.narrative == "Accidents totalled 100 in the test scope."
    assert len(result.self_reported_citations) == 1
    assert result.self_reported_citations[0].fact_key == "accident_count"
    assert result.scope_label == "Test scope"


def test_generate_insight_uses_constrained_tool_choice() -> None:
    mock_response = _mock_tool_use_response("Test narrative.", [])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("rri.insight.generate.anthropic.Anthropic", return_value=mock_client):
        generate_insight(_facts_payload(), api_key="test-key")

    _, kwargs = mock_client.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_insight"}
    assert kwargs["tools"][0]["name"] == "submit_insight"


def test_generate_insight_passes_facts_payload_in_message() -> None:
    mock_response = _mock_tool_use_response("Test narrative.", [])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("rri.insight.generate.anthropic.Anthropic", return_value=mock_client):
        generate_insight(_facts_payload(), api_key="test-key")

    _, kwargs = mock_client.messages.create.call_args
    user_content = kwargs["messages"][0]["content"]
    assert "accident_count" in user_content
    assert "100.0" in user_content


def test_generate_insight_raises_when_no_tool_use_block() -> None:
    mock_response = SimpleNamespace(content=[], stop_reason="end_turn")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("rri.insight.generate.anthropic.Anthropic", return_value=mock_client):
        try:
            generate_insight(_facts_payload(), api_key="test-key")
        except RuntimeError as exc:
            assert "end_turn" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")


def test_generate_insight_uses_configured_model_when_not_overridden() -> None:
    mock_response = _mock_tool_use_response("Test narrative.", [])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("rri.insight.generate.anthropic.Anthropic", return_value=mock_client):
        result = generate_insight(_facts_payload(), api_key="test-key")

    from config.settings import settings

    assert result.model == settings.insight_model
