"""
test_legal_duty.py — clauses that only restate a legal duty must be ACCEPT, never "negotiate".

Regression for: "You must not alter evidence or influence witnesses." was shown as
MEDIUM with Recommendation: NEGOTIATE. Negotiating that away means asking to be
allowed to break the law, so it must be a low-risk ACCEPT.

ALL tests are mocked — zero API calls.

Run: pytest backend/tests/test_legal_duty.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.negotiator import NegotiatorAgent
from agents.risk_analyzer import RiskAnalyzerAgent
from core.legal_duty import is_legal_duty_clause
from models.schemas import (
    ClauseType,
    LegalReasonerOutput,
    ReasonedClause,
    RiskCategory,
    RiskLabel,
    RiskLevel,
)

DUTY = "You must not alter evidence or influence witnesses."


class TestDetector:
    @pytest.mark.parametrize(
        "text",
        [
            DUTY,
            "The Contractor shall not destroy, falsify or conceal evidence in any investigation.",
            "Employee agrees not to bribe any public official or offer kickbacks.",
            "Party A must not intimidate or threaten any witness in a dispute.",
            "Neither party may obstruct justice or an official investigation.",
        ],
    )
    def test_detects_legal_duty_clauses(self, text):
        assert is_legal_duty_clause(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Client may terminate this Agreement at any time without notice.",
            "Contractor's liability under this Agreement is unlimited.",
            # no prohibition: this one demands cooperation, it does not forbid anything
            "The Contractor will provide all evidence to Client on request.",
            # a risky burden hidden next to the boilerplate sentence (too long to auto-accept)
            "The Contractor shall not alter evidence. " + "Contractor shall pay Client a penalty of $50,000 per breach. " * 8,
            # short, but it carries an indemnity next to the evidence sentence
            "Contractor shall indemnify Client against claims, and must not alter evidence.",
            "Contractor shall pay a penalty if it destroys evidence.",
            "",
            None,
        ],
    )
    def test_does_not_excuse_real_risks(self, text):
        assert not is_legal_duty_clause(text)


class TestRiskAnalyzerOverride:
    def _agent(self):
        return RiskAnalyzerAgent(client=MagicMock(), embeddings=None)

    def test_medium_legal_duty_becomes_green_low(self):
        raw = {
            "clause_id": "clause_001",
            "original_text": DUTY,
            "severity_score": 5.5,
            "risk_level": "YELLOW",
            "is_predatory": True,
            "risk_category": "general",
            "benchmark_comparison": "x",
        }
        out = self._agent()._coerce_clause(raw, 0)
        assert out["risk_level"] == "GREEN"
        assert out["risk_label"] == "LOW"
        assert out["severity_score"] <= 2.0
        assert out["is_predatory"] is False

    def test_normal_clause_is_untouched(self):
        raw = {
            "clause_id": "clause_002",
            "original_text": "Client may terminate at any time without notice.",
            "severity_score": 5.5,
            "risk_level": "YELLOW",
            "risk_category": "termination",
            "benchmark_comparison": "x",
        }
        out = self._agent()._coerce_clause(raw, 0)
        assert out["risk_level"] == "YELLOW"
        assert out["severity_score"] == 5.5


class TestNegotiatorOverride:
    def _source(self, text: str) -> LegalReasonerOutput:
        clause = ReasonedClause(
            clause_id="clause_001",
            clause_type=ClauseType.GENERAL if hasattr(ClauseType, "GENERAL") else list(ClauseType)[0],
            original_text=text,
            is_ambiguous=False,
            severity_score=5.0,
            risk_level=RiskLevel.YELLOW,
            risk_label=RiskLabel.MEDIUM,
            risk_category=RiskCategory.GENERAL,
            benchmark_comparison="Standard.",
            is_predatory=False,
            plain_language_explanation="Explanation.",
            scenario_consequence="If you sign this, then x.",
            key_implications=["x"],
        )
        return LegalReasonerOutput(
            clauses=[clause],
            overall_score=5.0,
            red_count=0, yellow_count=0, green_count=0,
            document_type="contract",
            executive_summary="Summary.",
        )

    def _model_says_negotiate(self):
        return {
            "executive_summary": "s",
            "top_risks": [],
            "negotiation_advice": [
                {
                    "clause_id": "clause_001",
                    "recommended_action": "negotiate",
                    "pushback_rationale": "Ask to remove this.",
                    "alternative_wording": "Rewritten text.",
                    "negotiation_tips": ["a", "b"],
                }
            ],
        }

    def test_legal_duty_forced_to_accept_even_if_model_says_negotiate(self):
        agent = NegotiatorAgent(MagicMock())
        parsed = agent._parse_response(self._model_says_negotiate(), self._source(DUTY))
        advice = parsed["negotiation_advice"]["clause_001"]
        assert advice["recommended_action"] == "accept"
        assert advice["pushback_rationale"] is None
        assert advice["alternative_wording"] is None
        assert advice["negotiation_tips"] == []

    def test_normal_clause_keeps_model_recommendation(self):
        agent = NegotiatorAgent(MagicMock())
        source = self._source("Client may terminate at any time without notice.")
        parsed = agent._parse_response(self._model_says_negotiate(), source)
        assert parsed["negotiation_advice"]["clause_001"]["recommended_action"] == "negotiate"
