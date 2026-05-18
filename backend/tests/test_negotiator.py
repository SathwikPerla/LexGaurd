"""
test_negotiator.py — Unit tests for Agent 4 (NegotiatorAgent).

Agent 4 now returns ONLY its unique fields (minimal schema), which graph.py
merges with Agent 3's full clause data. Tests verify this merge produces
a valid NegotiationAdvisorOutput.

Run: pytest backend/tests/test_negotiator.py -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.negotiator import NegotiatorAgent
from models.schemas import (
    ClauseType,
    LegalReasonerOutput,
    NegotiationAdvisorOutput,
    ReasonedClause,
    RiskCategory,
    RiskLevel,
    RiskLabel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fake_client(response) -> MagicMock:
    client = MagicMock()
    client.generate_json = AsyncMock(return_value=response)
    return client


def _make_reasoner_output(n_clauses: int = 2) -> LegalReasonerOutput:
    clauses = [
        ReasonedClause(
            clause_id=f"clause_{i:03d}",
            clause_type=ClauseType.NON_COMPETE if i == 1 else ClauseType.GOVERNING_LAW,
            original_text=f"Clause {i} text.",
            is_ambiguous=(i == 1),
            severity_score=8.5 if i == 1 else 2.0,
            risk_level=RiskLevel.RED if i == 1 else RiskLevel.GREEN,
            risk_label=RiskLabel.HIGH if i == 1 else RiskLabel.LOW,
            risk_category=RiskCategory.EMPLOYMENT if i == 1 else RiskCategory.COMPLIANCE,
            benchmark_comparison="Standard is 6 months." if i == 1 else "Standard.",
            is_predatory=(i == 1),
            plain_language_explanation=f"Plain English for clause {i}.",
            scenario_consequence=f"If you sign this and X, then Y for clause {i}.",
            key_implications=[f"Implication {i}"],
        )
        for i in range(1, n_clauses + 1)
    ]
    return LegalReasonerOutput(
        clauses=clauses,
        overall_score=7.5,
        red_count=0, yellow_count=0, green_count=0,
        document_type="employment_agreement",
        executive_summary="Do not sign without negotiating clause 1.",
    )


# Minimal response matching new Agent 4 schema
VALID_AGENT4_RESPONSE = {
    "executive_summary": "Do not sign without negotiating the non-compete.",
    "top_risks": ["Non-compete clause (8.5/10)", "Governing law (2.0/10)"],
    "overall_score": 7.5,
    "negotiation_advice": [
        {
            "clause_id": "clause_001",
            "recommended_action": "negotiate",
            "pushback_rationale": "2-year non-compete is above industry standard of 6-12 months.",
            "alternative_wording": "Employee agrees not to solicit Company clients for 6 months.",
            "negotiation_tips": ["Request 6-month limit", "Ask for garden leave"],
        },
        {
            "clause_id": "clause_002",
            "recommended_action": "accept",
            "pushback_rationale": None,
            "alternative_wording": None,
            "negotiation_tips": [],
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# run() — success paths
# ─────────────────────────────────────────────────────────────────────────────


class TestNegotiatorRun:
    def test_returns_negotiation_advisor_output(self):
        agent = NegotiatorAgent(_fake_client(VALID_AGENT4_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_reasoner_output())
        )
        assert isinstance(result, NegotiationAdvisorOutput)

    def test_clause_count_matches_agent3_input(self):
        agent = NegotiatorAgent(_fake_client(VALID_AGENT4_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_reasoner_output(2))
        )
        assert len(result.clauses) == 2

    def test_red_count_auto_computed(self):
        agent = NegotiatorAgent(_fake_client(VALID_AGENT4_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_reasoner_output())
        )
        expected = sum(1 for c in result.clauses if c.risk_level == RiskLevel.RED)
        assert result.red_count == expected

    def test_agent3_fields_preserved(self):
        """Agent 3 data (severity_score, original_text, etc.) must survive merge."""
        agent = NegotiatorAgent(_fake_client(VALID_AGENT4_RESPONSE))
        source = _make_reasoner_output()
        result = asyncio.get_event_loop().run_until_complete(agent.run(source))

        for out_clause in result.clauses:
            src = next(c for c in source.clauses if c.clause_id == out_clause.clause_id)
            assert out_clause.original_text == src.original_text
            assert out_clause.severity_score == src.severity_score
            assert out_clause.plain_language_explanation == src.plain_language_explanation
            assert out_clause.scenario_consequence == src.scenario_consequence

    def test_recommended_action_applied(self):
        agent = NegotiatorAgent(_fake_client(VALID_AGENT4_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_reasoner_output())
        )
        nc = next(c for c in result.clauses if c.clause_id == "clause_001")
        assert nc.recommended_action.value == "negotiate"

    def test_green_clause_gets_accept(self):
        agent = NegotiatorAgent(_fake_client(VALID_AGENT4_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_reasoner_output())
        )
        gl = next(c for c in result.clauses if c.clause_id == "clause_002")
        assert gl.recommended_action.value == "accept"

    def test_alternative_wording_on_red_clause(self):
        agent = NegotiatorAgent(_fake_client(VALID_AGENT4_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_reasoner_output())
        )
        nc = next(c for c in result.clauses if c.clause_id == "clause_001")
        assert nc.alternative_wording is not None

    def test_executive_summary_set(self):
        agent = NegotiatorAgent(_fake_client(VALID_AGENT4_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_reasoner_output())
        )
        assert len(result.executive_summary) > 10

    def test_top_risks_set(self):
        agent = NegotiatorAgent(_fake_client(VALID_AGENT4_RESPONSE))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_reasoner_output())
        )
        assert isinstance(result.top_risks, list) and len(result.top_risks) > 0

    def test_gemini_called_once(self):
        fake = _fake_client(VALID_AGENT4_RESPONSE)
        agent = NegotiatorAgent(fake)
        asyncio.get_event_loop().run_until_complete(agent.run(_make_reasoner_output()))
        fake.generate_json.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Missing clause fallback
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingClauseFallback:
    def test_dropped_clause_gets_default(self):
        """If Agent 4 omits a clause, it should get a default negotiate action."""
        response_missing_clause2 = {
            **VALID_AGENT4_RESPONSE,
            "negotiation_advice": [VALID_AGENT4_RESPONSE["negotiation_advice"][0]],
        }
        agent = NegotiatorAgent(_fake_client(response_missing_clause2))
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(_make_reasoner_output())
        )
        # Both clauses still present
        assert len(result.clauses) == 2
        ids = [c.clause_id for c in result.clauses]
        assert "clause_002" in ids


# ─────────────────────────────────────────────────────────────────────────────
# Empty input
# ─────────────────────────────────────────────────────────────────────────────


class TestEmptyInput:
    def test_empty_clause_list_returns_empty_output(self):
        agent = NegotiatorAgent(_fake_client(VALID_AGENT4_RESPONSE))
        empty_input = LegalReasonerOutput(
            clauses=[], overall_score=1.0,
            red_count=0, yellow_count=0, green_count=0,
            document_type="other",
            executive_summary="No clauses.",
        )
        result = asyncio.get_event_loop().run_until_complete(agent.run(empty_input))
        assert isinstance(result, NegotiationAdvisorOutput)
        assert len(result.clauses) == 0

    def test_empty_input_does_not_call_gemini(self):
        fake = _fake_client(VALID_AGENT4_RESPONSE)
        agent = NegotiatorAgent(fake)
        empty_input = LegalReasonerOutput(
            clauses=[], overall_score=1.0,
            red_count=0, yellow_count=0, green_count=0,
            document_type="other",
            executive_summary="No clauses.",
        )
        asyncio.get_event_loop().run_until_complete(agent.run(empty_input))
        fake.generate_json.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Error propagation
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorPropagation:
    def test_call_error_propagates(self):
        from core.gemini_client import LLMCallError
        fake = MagicMock()
        fake.generate_json = AsyncMock(side_effect=LLMCallError("API failed"))
        agent = NegotiatorAgent(fake)
        with pytest.raises(LLMCallError):
            asyncio.get_event_loop().run_until_complete(
                agent.run(_make_reasoner_output())
            )
