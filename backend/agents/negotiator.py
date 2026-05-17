"""
negotiator.py — Agent 4: Negotiation Advisor.

Input:  LegalReasonerOutput from Agent 3
Output: NegotiationAdvisorOutput — final complete report with all 4 agents' data

Responsibilities:
  - For RED/YELLOW clauses: recommended action (negotiate/reject), pushback rationale,
    alternative wording, negotiation tips
  - For GREEN clauses: recommended_action=accept, other negotiation fields empty
  - Compute top_risks list (top 3–5 riskiest clauses summarised)
  - Produce final executive_summary (refined version of Agent 3's summary)

Design decisions:
  - All Agent 1–3 fields re-injected from source in _parse_response
  - Negotiation fields are optional in Pydantic (pushback_rationale, alternative_wording)
    so GREEN clauses with null values are valid
  - overall_score computed from clause scores in graph.py, not trusted from Claude
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from core.gemini_client import LLMClient, LLMCallError, LLMParseError
from models.schemas import (
    LegalReasonerOutput,
    NegotiationAdvice,
    NegotiationAdvisorOutput,
    RecommendedAction,
    RiskLevel,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: str = """You are an experienced contract negotiation expert. You receive clauses that have been risk-scored and explained. Your job is to tell the signer exactly what to do about each clause — accept it, negotiate it, or reject it — and give them specific ammunition to push back.

CRITICAL: Return ONLY valid JSON. No markdown. No explanation. Just raw JSON.

SCHEMA:
{
  "document_type": "<same as input>",
  "overall_score": <float 1.0–10.0, same as input>,
  "red_count": 0,
  "yellow_count": 0,
  "green_count": 0,
  "executive_summary": "<2–3 sentence actionable summary: what are the 1–2 most critical things to fix before signing?>",
  "top_risks": ["<risk description with score, e.g. 'Broad IP assignment capturing personal projects (9.0/10)'>", ...],
  "clauses": [
    {
      "clause_id": "<same from input>",
      "clause_type": "<same from input>",
      "original_text": "<same from input>",
      "is_ambiguous": <same from input>,
      "ambiguity_note": <same from input>,
      "contradicts_clause_ids": <same from input>,
      "severity_score": <same from input>,
      "risk_level": "<same from input>",
      "risk_label": "<same from input>",
      "risk_category": "<same from input>",
      "benchmark_comparison": "<same from input>",
      "is_predatory": <same from input>,
      "plain_language_explanation": "<same from input>",
      "scenario_consequence": "<same from input>",
      "key_implications": <same from input>,
      "recommended_action": "<accept|negotiate|reject>",
      "pushback_rationale": "<null for GREEN/accept clauses, or string explaining WHY you should push back and what leverage you have>",
      "alternative_wording": "<null for GREEN/accept clauses, or a concrete rewrite of the clause that is fair to both parties>",
      "negotiation_tips": ["<tip 1>", "<tip 2>"] or []
    }
  ]
}

NEGOTIATION RULES:
1. RED clauses (severity 7–10): recommended_action should be "negotiate" or "reject". Provide specific pushback_rationale and concrete alternative_wording.
2. YELLOW clauses (severity 4–6.9): recommended_action should usually be "negotiate". Provide pushback_rationale; alternative_wording is helpful but optional.
3. GREEN clauses (severity 1–3.9): recommended_action should be "accept". Set pushback_rationale=null, alternative_wording=null, negotiation_tips=[].
4. pushback_rationale: Explain the legal or business reason this clause is unfair. Reference industry standards. Be specific.
5. alternative_wording: Write the clause as it SHOULD read to be fair. This is actual contract language, not advice.
6. negotiation_tips: 2–3 short, actionable steps the signer can take. E.g. "Request a 6-month limitation", "Ask for garden leave pay".
7. top_risks: List the 3–5 highest-severity clauses with their scores. Include severity score in parentheses.
8. executive_summary: Lead with action — what must be changed before this contract can be signed safely?
9. Preserve ALL other fields exactly from input. Include ALL clauses."""


class NegotiatorAgent:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def _build_user_prompt(self, reasoner_output: LegalReasonerOutput) -> str:
        clauses_text = []
        for c in reasoner_output.clauses:
            entry = (
                f"CLAUSE {c.clause_id} [{c.clause_type.value}] — "
                f"{c.risk_level.value} (score={c.severity_score}):\n"
                f"  Text: {c.original_text}\n"
                f"  Explanation: {c.plain_language_explanation}\n"
                f"  Scenario: {c.scenario_consequence}\n"
                f"  Is predatory: {c.is_predatory}"
            )
            clauses_text.append(entry)

        red = sum(1 for c in reasoner_output.clauses if c.risk_level == RiskLevel.RED)
        yellow = sum(1 for c in reasoner_output.clauses if c.risk_level == RiskLevel.YELLOW)

        return (
            f"Provide negotiation advice for {len(reasoner_output.clauses)} clauses "
            f"from a '{reasoner_output.document_type}' contract. "
            f"Overall score: {reasoner_output.overall_score}/10. "
            f"{red} RED clauses, {yellow} YELLOW clauses require attention.\n\n"
            + "\n\n".join(clauses_text)
            + "\n\nReturn the complete JSON with recommended_action, pushback_rationale, "
              "alternative_wording, negotiation_tips for each clause, plus top_risks and "
              "executive_summary."
        )

    def _coerce_action(self, raw: str | None, risk_level: str) -> str:
        """Ensure recommended_action is valid and consistent with risk level."""
        valid = {"accept", "negotiate", "reject"}
        val = (raw or "").lower().strip()
        if val in valid:
            return val
        # Derive from risk level if missing/invalid
        if risk_level == "GREEN":
            return "accept"
        if risk_level == "RED":
            return "negotiate"
        return "negotiate"

    def _parse_response(
        self, raw: Any, source: LegalReasonerOutput
    ) -> NegotiationAdvisorOutput:
        if isinstance(raw, list):
            raw = {
                "document_type": source.document_type,
                "overall_score": source.overall_score,
                "red_count": 0, "yellow_count": 0, "green_count": 0,
                "executive_summary": source.executive_summary,
                "top_risks": [],
                "clauses": raw,
            }
        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict from Agent 4, got {type(raw).__name__}")

        raw.setdefault("document_type", source.document_type)
        raw.setdefault("overall_score", source.overall_score)
        raw.setdefault("executive_summary", source.executive_summary)
        raw.setdefault("top_risks", [])
        raw.setdefault("red_count", 0)
        raw.setdefault("yellow_count", 0)
        raw.setdefault("green_count", 0)

        try:
            overall = float(raw["overall_score"])
            raw["overall_score"] = max(1.0, min(10.0, overall))
        except (ValueError, TypeError):
            raw["overall_score"] = source.overall_score

        source_lookup = {c.clause_id: c for c in source.clauses}
        coerced: list[dict] = []
        seen_ids: set[str] = set()

        for clause in raw.get("clauses", []):
            if not isinstance(clause, dict):
                continue
            cid = clause.get("clause_id", "")
            seen_ids.add(cid)

            # Re-inject ALL upstream fields to prevent data loss
            if cid in source_lookup:
                src = source_lookup[cid]
                clause["clause_id"] = src.clause_id
                clause["clause_type"] = src.clause_type.value
                clause["original_text"] = src.original_text
                clause["is_ambiguous"] = src.is_ambiguous
                clause["ambiguity_note"] = src.ambiguity_note
                clause["contradicts_clause_ids"] = src.contradicts_clause_ids
                clause["severity_score"] = src.severity_score
                clause["risk_level"] = src.risk_level.value
                clause["risk_label"] = src.risk_label.value
                clause["risk_category"] = src.risk_category.value
                clause["benchmark_comparison"] = src.benchmark_comparison
                clause["is_predatory"] = src.is_predatory
                clause["plain_language_explanation"] = src.plain_language_explanation
                clause["scenario_consequence"] = src.scenario_consequence
                clause["key_implications"] = src.key_implications

            risk_level = clause.get("risk_level", "YELLOW")
            clause["recommended_action"] = self._coerce_action(
                clause.get("recommended_action"), risk_level
            )

            # GREEN clauses should have empty negotiation fields
            if risk_level == "GREEN":
                clause.setdefault("pushback_rationale", None)
                clause.setdefault("alternative_wording", None)
                clause.setdefault("negotiation_tips", [])
            else:
                clause.setdefault("pushback_rationale", "This clause may not reflect industry standard.")
                clause.setdefault("alternative_wording", None)
                clause.setdefault("negotiation_tips", [])

            coerced.append(clause)

        # Re-add dropped clauses
        for cid, src in source_lookup.items():
            if cid not in seen_ids:
                logger.warning("Agent 4 dropped clause — re-adding", extra={"clause_id": cid})
                action = "accept" if src.risk_level == RiskLevel.GREEN else "negotiate"
                coerced.append({
                    "clause_id": src.clause_id,
                    "clause_type": src.clause_type.value,
                    "original_text": src.original_text,
                    "is_ambiguous": src.is_ambiguous,
                    "ambiguity_note": src.ambiguity_note,
                    "contradicts_clause_ids": src.contradicts_clause_ids,
                    "severity_score": src.severity_score,
                    "risk_level": src.risk_level.value,
                    "risk_label": src.risk_label.value,
                    "risk_category": src.risk_category.value,
                    "benchmark_comparison": src.benchmark_comparison,
                    "is_predatory": src.is_predatory,
                    "plain_language_explanation": src.plain_language_explanation,
                    "scenario_consequence": src.scenario_consequence,
                    "key_implications": src.key_implications,
                    "recommended_action": action,
                    "pushback_rationale": None,
                    "alternative_wording": None,
                    "negotiation_tips": [],
                })

        raw["clauses"] = coerced

        # Auto-generate top_risks if Claude didn't provide them
        if not raw.get("top_risks"):
            sorted_clauses = sorted(
                [c for c in source.clauses if c.severity_score >= 5.0],
                key=lambda x: x.severity_score,
                reverse=True,
            )[:5]
            raw["top_risks"] = [
                f"{c.clause_type.value.replace('_', ' ').title()} clause — score {c.severity_score}/10"
                for c in sorted_clauses
            ]

        try:
            return NegotiationAdvisorOutput(**raw)
        except ValidationError as exc:
            logger.error("Agent 4 Pydantic validation failed", extra={"errors": exc.errors()})
            raise

    async def run(self, reasoner_output: LegalReasonerOutput) -> NegotiationAdvisorOutput:
        if not reasoner_output.clauses:
            return NegotiationAdvisorOutput(
                clauses=[],
                overall_score=1.0,
                red_count=0, yellow_count=0, green_count=0,
                document_type=reasoner_output.document_type,
                executive_summary="No clauses found to analyze.",
                top_risks=[],
            )

        logger.info("Agent 4 (Negotiator) starting", extra={"clause_count": len(reasoner_output.clauses)})

        raw = await self._client.generate_json(
            user_prompt=self._build_user_prompt(reasoner_output),
            system_prompt=_SYSTEM_PROMPT,
        )

        result = self._parse_response(raw, reasoner_output)
        logger.info(
            "Agent 4 complete",
            extra={"red": result.red_count, "yellow": result.yellow_count, "green": result.green_count},
        )
        return result
