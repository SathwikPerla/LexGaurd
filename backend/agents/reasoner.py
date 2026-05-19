"""
reasoner.py — Agent 3: Legal Reasoner (minimal output schema).

Output: ONLY the new fields this agent adds, keyed by clause_id.
graph.py merges these with Agent 2's full clause data.

This keeps Agent 3's output ~100 tokens/clause instead of ~500,
preventing the 8192-token truncation that caused LLMParseError.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from pydantic import ValidationError

from core.gemini_client import LLMClient
from models.schemas import (
    LegalReasonerOutput,
    ReasonedClause,
    RiskAnalyzerOutput,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: str = """You are a legal communication expert. You receive risk-scored contract clauses and explain each one in plain language for a non-lawyer.

CRITICAL: Return ONLY valid JSON. No markdown. No explanation. Just raw JSON.

SCHEMA:
{
  "executive_summary": "<2 sentences: what are the 1-2 biggest risks in this contract? Be direct.>",
  "clause_explanations": [
    {
      "clause_id": "<same clause_id from input — do not change>",
      "plain_language_explanation": "<1-2 sentences: what does this clause actually mean for the person signing? No jargon.>",
      "scenario_consequence": "<Start with 'If you sign this and': one concrete realistic scenario and its consequence for the signer.>",
      "key_implications": ["<short implication 1>", "<short implication 2>"]
    }
  ]
}

RULES:
1. plain_language_explanation: Write for someone with zero legal background. What does this do TO THEM?
2. scenario_consequence: MUST start with "If you sign this and". One sentence. Make it concrete — mention money, job loss, lawsuits, specific consequences.
3. key_implications: 2-3 short bullet points. Each is a distinct practical consequence.
4. executive_summary: Lead with the worst risk. Be direct and actionable.
5. Include ALL clauses in clause_explanations — do not drop any."""


class ReasonerAgent:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def _build_user_prompt(self, risk_output: RiskAnalyzerOutput) -> str:
        lines = []
        for c in risk_output.clauses:
            lines.append(
                f"clause_id={c.clause_id}  [{c.clause_type.value}]  "
                f"{c.risk_level.value} score={c.severity_score}  predatory={c.is_predatory}\n"
                f"  Text: {c.original_text[:200]}"
            )
        return (
            f"Explain {len(risk_output.clauses)} clauses from a "
            f"'{risk_output.document_type}' contract. Overall risk: {risk_output.overall_score}/10.\n\n"
            + "\n\n".join(lines)
            + "\n\nReturn the JSON with plain_language_explanation, scenario_consequence, "
              "and key_implications for each clause."
        )

    def _parse_response(
        self, raw: Any, source: RiskAnalyzerOutput
    ) -> LegalReasonerOutput:
        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict from Agent 3, got {type(raw).__name__}")

        raw = copy.deepcopy(raw)

        executive_summary = raw.get("executive_summary") or source.document_type + " — review carefully."

        # Build lookup of explanations keyed by clause_id
        explanation_lookup: dict[str, dict] = {}
        for item in raw.get("clause_explanations", []):
            if not isinstance(item, dict):
                continue
            cid = item.get("clause_id", "")
            if cid:
                explanation_lookup[cid] = {
                    "plain_language_explanation": item.get("plain_language_explanation")
                        or f"This is a {item.get('clause_id', 'general')} clause — review carefully.",
                    "scenario_consequence": item.get("scenario_consequence")
                        or "If you sign this and a dispute arises, the terms as written apply.",
                    "key_implications": item.get("key_implications") or ["Review this clause with a professional."],
                }

        # Merge Agent 2 full data with Agent 3 explanations
        reasoned_clauses = []
        for clause in source.clauses:
            exp = explanation_lookup.get(clause.clause_id, {})
            reasoned_clauses.append(ReasonedClause(
                clause_id=clause.clause_id,
                clause_type=clause.clause_type,
                original_text=clause.original_text,
                is_ambiguous=clause.is_ambiguous,
                ambiguity_note=clause.ambiguity_note,
                contradicts_clause_ids=clause.contradicts_clause_ids,
                severity_score=clause.severity_score,
                risk_level=clause.risk_level,
                risk_label=clause.risk_label,
                risk_category=clause.risk_category,
                benchmark_comparison=clause.benchmark_comparison,
                is_predatory=clause.is_predatory,
                plain_language_explanation=exp.get(
                    "plain_language_explanation",
                    f"This clause has {clause.risk_level.value} risk implications. Review carefully."
                ),
                scenario_consequence=exp.get(
                    "scenario_consequence",
                    "If you sign this and a dispute arises, the terms as written apply."
                ),
                key_implications=exp.get("key_implications", ["Review this clause with a legal professional."]),
            ))

        try:
            return LegalReasonerOutput(
                clauses=reasoned_clauses,
                overall_score=source.overall_score,
                red_count=0, yellow_count=0, green_count=0,
                document_type=source.document_type,
                executive_summary=executive_summary,
            )
        except ValidationError as exc:
            logger.error("Agent 3 validation failed", extra={"errors": exc.errors()})
            raise

    async def run(self, risk_output: RiskAnalyzerOutput) -> LegalReasonerOutput:
        if not risk_output.clauses:
            return LegalReasonerOutput(
                clauses=[], overall_score=1.0,
                red_count=0, yellow_count=0, green_count=0,
                document_type=risk_output.document_type,
                executive_summary="No clauses found.",
            )

        logger.info("Agent 3 (Reasoner) starting", extra={"clause_count": len(risk_output.clauses)})

        raw = await self._client.generate_json(
            user_prompt=self._build_user_prompt(risk_output),
            system_prompt=_SYSTEM_PROMPT,
        )

        result = self._parse_response(raw, risk_output)
        logger.info("Agent 3 complete", extra={"clauses": len(result.clauses)})
        return result
