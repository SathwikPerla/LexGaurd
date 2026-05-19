"""
test_risk_analyzer.py — Unit tests for Agent 2 (RiskAnalyzerAgent).

ALL tests use fake GeminiClient and no EmbeddingsStore — zero API calls.

Run: pytest backend/tests/test_risk_analyzer.py -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.risk_analyzer import RiskAnalyzerAgent
from models.schemas import (
    ExtractedClause,
    ExtractorOutput,
    RiskAnalyzerOutput,
    RiskLevel,
    RiskLabel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fake_client(response: Any) -> MagicMock:
    client = MagicMock()
    client.generate_json = AsyncMock(return_value=response)
    return client


def _make_extractor_output(
    clauses: list[dict] | None = None,
    document_type: str = "employment_agreement",
) -> ExtractorOutput:
    if clauses is None:
        clauses = [
            {
                "clause_id": "clause_001",
                "clause_type": "non_compete",
                "original_text": "Employee shall not compete for 2 years within 50 miles.",
                "is_ambiguous": True,
                "ambiguity_note": "'Compete' is undefined.",
                "contradicts_clause_ids": [],
            },
            {
                "clause_id": "clause_002",
                "clause_type": "ip_transfer",
                "original_text": "All inventions belong to Employer regardless of when created.",
                "is_ambiguous": False,
                "ambiguity_note": None,
                "contradicts_clause_ids": [],
            },
        ]
    return ExtractorOutput(
        clauses=[ExtractedClause(**c) for c in clauses],
        total_clauses=len(clauses),
        document_type=document_type,
    )


VALID_AGENT2_RESPONSE = {
    "document_type": "employment_agreement",
    "overall_score": 8.0,
    "red_count": 0,
    "yellow_count": 0,
    "green_count": 0,
    "clauses": [
        {
            "clause_id": "clause_001",
            "clause_type": "non_compete",
            "original_text": "Employee shall not compete for 2 years within 50 miles.",
            "is_ambiguous": True,
            "ambiguity_note": "'Compete' is undefined.",
            "contradicts_clause_ids": [],
            "severity_score": 8.5,
            "risk_level": "RED",
            "risk_label": "HIGH",
            "risk_category": "employment",
            "benchmark_comparison": "Standard non-competes are 6-12 months. 2 years is unusually long.",
            "is_predatory": True,
        },
        {
            "clause_id": "clause_002",
            "clause_type": "ip_transfer",
            "original_text": "All inventions belong to Employer regardless of when created.",
            "is_ambiguous": False,
            "ambiguity_note": None,
            "contradicts_clause_ids": [],
            "severity_score": 9.0,
            "risk_level": "RED",
            "risk_label": "HIGH",
            "risk_category": "intellectual_property",
            "benchmark_comparison": "Standard IP clauses apply only to work on company time.",
            "is_predatory": True,
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# run() — success paths
# ─────────────────────────────────────────────────────────────────────────────


class TestRiskAnalyzerRun:
    def test_returns_risk_analyzer_output(self):
        agent = RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE))
        input_data = _make_extractor_output()
        result = asyncio.get_event_loop().run_until_complete(agent.run(input_data))
        assert isinstance(result, RiskAnalyzerOutput)

    def test_clause_count_matches_input(self):
        agent = RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE))
        input_data = _make_extractor_output()
        result = asyncio.get_event_loop().run_until_complete(agent.run(input_data))
        assert len(result.clauses) == len(input_data.clauses)

    def test_severity_scores_in_range(self):
        agent = RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_extractor_output())
        )
        for clause in result.clauses:
            assert 1.0 <= clause.severity_score <= 10.0

    def test_risk_levels_are_valid(self):
        agent = RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_extractor_output())
        )
        for clause in result.clauses:
            assert clause.risk_level in (RiskLevel.RED, RiskLevel.YELLOW, RiskLevel.GREEN)

    def test_risk_labels_are_valid(self):
        agent = RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_extractor_output())
        )
        for clause in result.clauses:
            assert clause.risk_label in (RiskLabel.HIGH, RiskLabel.MEDIUM, RiskLabel.LOW)

    def test_risk_label_matches_score(self):
        """Pydantic must compute risk_label from score, not from Gemini's string."""
        agent = RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_extractor_output())
        )
        for clause in result.clauses:
            if clause.severity_score >= 7.0:
                assert clause.risk_label == RiskLabel.HIGH
            elif clause.severity_score >= 4.0:
                assert clause.risk_label == RiskLabel.MEDIUM
            else:
                assert clause.risk_label == RiskLabel.LOW

    def test_red_count_auto_computed(self):
        agent = RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_extractor_output())
        )
        expected_red = sum(1 for c in result.clauses if c.risk_level == RiskLevel.RED)
        assert result.red_count == expected_red

    def test_overall_score_in_range(self):
        agent = RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_extractor_output())
        )
        assert 1.0 <= result.overall_score <= 10.0

    def test_original_text_preserved_from_agent1(self):
        """Agent 2 must not modify original_text — must come from Agent 1 input."""
        agent = RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE))
        input_data = _make_extractor_output()
        result = asyncio.get_event_loop().run_until_complete(agent.run(input_data))

        input_texts = {c.clause_id: c.original_text for c in input_data.clauses}
        for clause in result.clauses:
            assert clause.original_text == input_texts[clause.clause_id]

    def test_benchmark_comparison_non_empty(self):
        agent = RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_extractor_output())
        )
        for clause in result.clauses:
            assert len(clause.benchmark_comparison) > 0

    def test_gemini_called_once(self):
        fake = _fake_client(VALID_AGENT2_RESPONSE)
        agent = RiskAnalyzerAgent(fake)
        asyncio.get_event_loop().run_until_complete(agent.run(_make_extractor_output()))
        fake.generate_json.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Empty input handling
# ─────────────────────────────────────────────────────────────────────────────


class TestEmptyInput:
    def test_empty_clause_list_returns_empty_output(self):
        agent = RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE))
        empty_input = ExtractorOutput(clauses=[], total_clauses=0, document_type="other")
        result = asyncio.get_event_loop().run_until_complete(agent.run(empty_input))
        assert isinstance(result, RiskAnalyzerOutput)
        assert len(result.clauses) == 0
        assert result.overall_score == 1.0

    def test_empty_input_does_not_call_gemini(self):
        fake = _fake_client(VALID_AGENT2_RESPONSE)
        agent = RiskAnalyzerAgent(fake)
        empty_input = ExtractorOutput(clauses=[], total_clauses=0, document_type="other")
        asyncio.get_event_loop().run_until_complete(agent.run(empty_input))
        fake.generate_json.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# _coerce_clause — field coercion tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCoerceClause:
    def _agent(self):
        return RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE))

    def test_score_above_10_clamped(self):
        agent = self._agent()
        raw = {
            "clause_id": "c1", "clause_type": "general", "original_text": "x",
            "severity_score": 15.0, "risk_level": "RED", "risk_category": "general",
            "benchmark_comparison": "n/a", "is_predatory": False,
        }
        result = agent._coerce_clause(raw, 0)
        assert result["severity_score"] == 10.0

    def test_score_below_1_clamped(self):
        agent = self._agent()
        raw = {
            "clause_id": "c1", "clause_type": "general", "original_text": "x",
            "severity_score": -3.0, "risk_level": "GREEN", "risk_category": "general",
            "benchmark_comparison": "n/a", "is_predatory": False,
        }
        result = agent._coerce_clause(raw, 0)
        assert result["severity_score"] == 1.0

    def test_invalid_risk_level_derived_from_score(self):
        agent = self._agent()
        raw = {
            "clause_id": "c1", "clause_type": "general", "original_text": "x",
            "severity_score": 8.0, "risk_level": "PURPLE",  # invalid
            "risk_category": "general", "benchmark_comparison": "n/a", "is_predatory": False,
        }
        result = agent._coerce_clause(raw, 0)
        assert result["risk_level"] == "RED"

    def test_invalid_risk_category_defaults_to_general(self):
        agent = self._agent()
        raw = {
            "clause_id": "c1", "clause_type": "general", "original_text": "x",
            "severity_score": 5.0, "risk_level": "YELLOW",
            "risk_category": "made_up_category",  # invalid
            "benchmark_comparison": "n/a", "is_predatory": False,
        }
        result = agent._coerce_clause(raw, 0)
        assert result["risk_category"] == "general"

    def test_missing_benchmark_comparison_gets_default(self):
        agent = self._agent()
        raw = {
            "clause_id": "c1", "clause_type": "general", "original_text": "x",
            "severity_score": 5.0, "risk_level": "YELLOW",
            "risk_category": "general", "is_predatory": False,
            # benchmark_comparison missing
        }
        result = agent._coerce_clause(raw, 0)
        assert len(result["benchmark_comparison"]) > 0

    def test_score_7_maps_to_red(self):
        agent = self._agent()
        raw = {
            "clause_id": "c1", "clause_type": "general", "original_text": "x",
            "severity_score": 7.0, "risk_level": "INVALID",
            "risk_category": "general", "benchmark_comparison": "x", "is_predatory": False,
        }
        result = agent._coerce_clause(raw, 0)
        assert result["risk_level"] == "RED"

    def test_score_4_maps_to_yellow(self):
        agent = self._agent()
        raw = {
            "clause_id": "c1", "clause_type": "general", "original_text": "x",
            "severity_score": 4.0, "risk_level": "INVALID",
            "risk_category": "general", "benchmark_comparison": "x", "is_predatory": False,
        }
        result = agent._coerce_clause(raw, 0)
        assert result["risk_level"] == "YELLOW"

    def test_score_3_maps_to_green(self):
        agent = self._agent()
        raw = {
            "clause_id": "c1", "clause_type": "general", "original_text": "x",
            "severity_score": 3.0, "risk_level": "INVALID",
            "risk_category": "general", "benchmark_comparison": "x", "is_predatory": False,
        }
        result = agent._coerce_clause(raw, 0)
        assert result["risk_level"] == "GREEN"


# ─────────────────────────────────────────────────────────────────────────────
# _parse_response — structural edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestParseResponse:
    def _agent(self):
        return RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE))

    def test_list_response_wrapped_correctly(self):
        agent = self._agent()
        input_data = _make_extractor_output()
        result = agent._parse_response(VALID_AGENT2_RESPONSE["clauses"], input_data)
        assert isinstance(result, RiskAnalyzerOutput)
        assert len(result.clauses) == 2

    def test_missing_clause_re_added_from_agent1(self):
        """If Gemini drops a clause, _parse_response adds it back."""
        agent = self._agent()
        input_data = _make_extractor_output()
        # Response missing clause_002
        response_missing_clause = {
            **VALID_AGENT2_RESPONSE,
            "clauses": [VALID_AGENT2_RESPONSE["clauses"][0]],  # only clause_001
        }
        result = agent._parse_response(response_missing_clause, input_data)
        ids = [c.clause_id for c in result.clauses]
        assert "clause_002" in ids

    def test_non_dict_non_list_raises(self):
        agent = self._agent()
        input_data = _make_extractor_output()
        with pytest.raises(ValueError, match="Expected dict"):
            agent._parse_response("not valid", input_data)

    def test_overall_score_out_of_range_clamped(self):
        agent = self._agent()
        input_data = _make_extractor_output()
        response = {**VALID_AGENT2_RESPONSE, "overall_score": 99.0}
        result = agent._parse_response(response, input_data)
        assert result.overall_score <= 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Error propagation
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorPropagation:
    def test_gemini_call_error_propagates(self):
        from core.gemini_client import GeminiCallError
        fake = MagicMock()
        fake.generate_json = AsyncMock(side_effect=GeminiCallError("API failed"))
        agent = RiskAnalyzerAgent(fake)
        with pytest.raises(GeminiCallError):
            asyncio.get_event_loop().run_until_complete(
                agent.run(_make_extractor_output())
            )

    def test_gemini_parse_error_propagates(self):
        from core.gemini_client import GeminiParseError
        fake = MagicMock()
        fake.generate_json = AsyncMock(side_effect=GeminiParseError("bad JSON"))
        agent = RiskAnalyzerAgent(fake)
        with pytest.raises(GeminiParseError):
            asyncio.get_event_loop().run_until_complete(
                agent.run(_make_extractor_output())
            )


# ─────────────────────────────────────────────────────────────────────────────
# Without EmbeddingsStore — graceful degradation
# ─────────────────────────────────────────────────────────────────────────────


class TestNoEmbeddings:
    def test_runs_without_embeddings(self):
        """embeddings=None should not raise — benchmark context just skipped."""
        agent = RiskAnalyzerAgent(_fake_client(VALID_AGENT2_RESPONSE), embeddings=None)
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_extractor_output())
        )
        assert isinstance(result, RiskAnalyzerOutput)
