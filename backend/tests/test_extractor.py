"""
test_extractor.py — Unit tests for Agent 1 (ExtractorAgent).

ALL tests use a fake GeminiClient — zero API calls, zero GCP cost.

Run: pytest backend/tests/test_extractor.py -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.extractor import ExtractorAgent, _sanitize_document_text
from models.schemas import ClauseType, ExtractorOutput


# ─────────────────────────────────────────────────────────────────────────────
# Fake GeminiClient — no credentials needed
# ─────────────────────────────────────────────────────────────────────────────


def _fake_client(response: Any) -> MagicMock:
    """Return a mock GeminiClient whose generate_json returns `response`."""
    client = MagicMock()
    client.generate_json = AsyncMock(return_value=response)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Canonical valid response from Gemini
# ─────────────────────────────────────────────────────────────────────────────

VALID_RESPONSE = {
    "document_type": "employment_agreement",
    "total_clauses": 2,
    "clauses": [
        {
            "clause_id": "clause_001",
            "clause_type": "non_compete",
            "original_text": "Employee agrees not to compete for 2 years within 50 miles.",
            "is_ambiguous": True,
            "ambiguity_note": "'Compete' is undefined.",
            "contradicts_clause_ids": [],
        },
        {
            "clause_id": "clause_002",
            "clause_type": "ip_transfer",
            "original_text": "All inventions belong to Employer.",
            "is_ambiguous": False,
            "ambiguity_note": None,
            "contradicts_clause_ids": [],
        },
    ],
}

SAMPLE_TEXT = (
    "Employee agrees not to compete for 2 years within 50 miles. "
    "All inventions belong to Employer. This agreement is governed by Delaware law."
)


# ─────────────────────────────────────────────────────────────────────────────
# Input sanitization tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSanitizeDocumentText:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _sanitize_document_text("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="too short"):
            _sanitize_document_text("   ")

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            _sanitize_document_text("hi")

    def test_normal_text_passes(self):
        result = _sanitize_document_text(SAMPLE_TEXT)
        assert isinstance(result, str)
        assert len(result) > 10

    def test_null_bytes_removed(self):
        text = "Valid contract text. " * 3 + "\x00extra"
        result = _sanitize_document_text(text)
        assert "\x00" not in result

    def test_control_chars_removed(self):
        text = "Valid contract text. " * 3 + "\x08\x0b\x1f"
        result = _sanitize_document_text(text)
        assert "\x08" not in result

    def test_truncates_to_max(self):
        from agents.extractor import MAX_DOC_CHARS
        long_text = "A" * (MAX_DOC_CHARS + 5000)
        result = _sanitize_document_text(long_text)
        assert len(result) == MAX_DOC_CHARS

    def test_legal_special_chars_preserved(self):
        text = "Section §12(a): Employee shall not disclose — subject to §7."
        result = _sanitize_document_text(text)
        assert "§" in result
        assert "—" in result


# ─────────────────────────────────────────────────────────────────────────────
# ExtractorAgent.run — success paths
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractorRun:
    def test_returns_extractor_output(self):
        agent = ExtractorAgent(_fake_client(VALID_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(agent.run(SAMPLE_TEXT))
        assert isinstance(result, ExtractorOutput)

    def test_correct_clause_count(self):
        agent = ExtractorAgent(_fake_client(VALID_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(agent.run(SAMPLE_TEXT))
        assert result.total_clauses == 2
        assert len(result.clauses) == 2

    def test_clause_types_correctly_labeled(self):
        agent = ExtractorAgent(_fake_client(VALID_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(agent.run(SAMPLE_TEXT))
        types = [c.clause_type for c in result.clauses]
        assert ClauseType.NON_COMPETE in types
        assert ClauseType.IP_TRANSFER in types

    def test_document_type_returned(self):
        agent = ExtractorAgent(_fake_client(VALID_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(agent.run(SAMPLE_TEXT))
        assert result.document_type == "employment_agreement"

    def test_ambiguous_flag_preserved(self):
        agent = ExtractorAgent(_fake_client(VALID_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(agent.run(SAMPLE_TEXT))
        non_compete = next(c for c in result.clauses if c.clause_type == ClauseType.NON_COMPETE)
        assert non_compete.is_ambiguous is True
        assert non_compete.ambiguity_note is not None

    def test_original_text_preserved_verbatim(self):
        agent = ExtractorAgent(_fake_client(VALID_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(agent.run(SAMPLE_TEXT))
        nc = next(c for c in result.clauses if c.clause_type == ClauseType.NON_COMPETE)
        assert "2 years" in nc.original_text

    def test_gemini_called_once(self):
        fake = _fake_client(VALID_RESPONSE)
        agent = ExtractorAgent(fake)
        asyncio.get_event_loop().run_until_complete(agent.run(SAMPLE_TEXT))
        fake.generate_json.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# ExtractorAgent._parse_response — edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestParseResponse:
    def _agent(self) -> ExtractorAgent:
        return ExtractorAgent(_fake_client(VALID_RESPONSE))

    def test_list_response_wrapped_correctly(self):
        agent = self._agent()
        response_as_list = VALID_RESPONSE["clauses"]
        result = agent._parse_response(response_as_list)
        assert len(result.clauses) == 2

    def test_missing_total_clauses_autocorrected(self):
        agent = self._agent()
        r = {**VALID_RESPONSE}
        del r["total_clauses"]
        result = agent._parse_response(r)
        assert result.total_clauses == 2

    def test_missing_document_type_defaults_to_unknown(self):
        agent = self._agent()
        r = {**VALID_RESPONSE}
        del r["document_type"]
        result = agent._parse_response(r)
        assert result.document_type == "unknown"

    def test_clause_type_with_space_coerced(self):
        agent = self._agent()
        r = {
            "document_type": "employment_agreement",
            "total_clauses": 1,
            "clauses": [{
                "clause_id": "clause_001",
                "clause_type": "non compete",  # space instead of underscore
                "original_text": "Employee shall not compete for 12 months.",
                "is_ambiguous": False,
                "ambiguity_note": None,
                "contradicts_clause_ids": [],
            }],
        }
        result = agent._parse_response(r)
        assert result.clauses[0].clause_type == ClauseType.NON_COMPETE

    def test_missing_clause_id_auto_assigned(self):
        agent = self._agent()
        r = {
            "document_type": "employment_agreement",
            "total_clauses": 1,
            "clauses": [{
                "clause_type": "termination",
                "original_text": "Company may terminate with 30 days notice.",
                "is_ambiguous": False,
                "ambiguity_note": None,
                "contradicts_clause_ids": [],
            }],
        }
        result = agent._parse_response(r)
        assert result.clauses[0].clause_id.startswith("clause_")

    def test_invalid_clause_type_raises_validation_error(self):
        agent = self._agent()
        r = {
            "document_type": "employment_agreement",
            "total_clauses": 1,
            "clauses": [{
                "clause_id": "clause_001",
                "clause_type": "totally_made_up_type",
                "original_text": "Some text.",
                "is_ambiguous": False,
                "ambiguity_note": None,
                "contradicts_clause_ids": [],
            }],
        }
        with pytest.raises(ValidationError):
            agent._parse_response(r)

    def test_empty_clauses_list_valid(self):
        agent = self._agent()
        r = {"document_type": "other", "total_clauses": 0, "clauses": []}
        result = agent._parse_response(r)
        assert result.total_clauses == 0
        assert result.clauses == []

    def test_non_dict_non_list_raises_value_error(self):
        agent = self._agent()
        with pytest.raises(ValueError, match="Expected dict"):
            agent._parse_response("not a dict or list")  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Error propagation
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractorErrors:
    def test_empty_document_raises_before_api_call(self):
        """Empty document should raise ValueError without calling Gemini."""
        fake = _fake_client(VALID_RESPONSE)
        agent = ExtractorAgent(fake)
        with pytest.raises(ValueError, match="empty"):
            asyncio.get_event_loop().run_until_complete(agent.run(""))
        fake.generate_json.assert_not_called()

    def test_gemini_call_error_propagates(self):
        from core.gemini_client import GeminiCallError
        fake = MagicMock()
        fake.generate_json = AsyncMock(side_effect=GeminiCallError("API failed"))
        agent = ExtractorAgent(fake)
        with pytest.raises(GeminiCallError):
            asyncio.get_event_loop().run_until_complete(agent.run(SAMPLE_TEXT))

    def test_gemini_parse_error_propagates(self):
        from core.gemini_client import GeminiParseError
        fake = MagicMock()
        fake.generate_json = AsyncMock(side_effect=GeminiParseError("bad JSON"))
        agent = ExtractorAgent(fake)
        with pytest.raises(GeminiParseError):
            asyncio.get_event_loop().run_until_complete(agent.run(SAMPLE_TEXT))
