"""
negotiator.py — Agent 4: Negotiation Advisor.

Input:  LegalReasonerOutput from Agent 3
Output: NegotiationAdvisorOutput — Agent 3 clauses merged with negotiation advice

ARCHITECTURE: Agent 4 outputs ONLY the fields it uniquely adds:
  - executive_summary, top_risks, overall_score (document-level)
  - Per clause: recommended_action, pushback_rationale, alternative_wording,
                negotiation_tips (keyed by clause_id for merging)

This keeps Agent 4's output ~200 tokens per clause instead of ~1500 per clause
(the old approach re-stated all upstream fields), making the call safe for
documents with 10+ clauses without hitting the 8192-token output limit.

graph.py merges Agent 4's additions with Agent 3's full clause data to produce
the final NegotiationAdvisorOutput.
"""
from __future__ import annotations

import logging
from typing import Any


from core.gemini_client import LLMClient
from models.schemas import (
    LegalReasonerOutput,
    NegotiationAdvice,
    NegotiationAdvisorOutput,
    RiskLevel,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: str = """You are a contract negotiation expert. You receive clauses that have been risk-scored and explained in plain English. Your job is to advise the signer on what to do.

CRITICAL: Return ONLY valid JSON. No markdown. No explanation. Just raw JSON.

SCHEMA:
{
  "executive_summary": "<2 sentences: what must be fixed before signing? Be direct and specific.>",
  "top_risks": ["<clause type and score, e.g. 'IP transfer clause captures personal projects (9.5/10)'>"],
  "overall_score": <float 1.0–10.0, weighted average>,
  "negotiation_advice": [
    {
      "clause_id": "<same clause_id from input — do not change>",
      "recommended_action": "<accept|negotiate|reject>",
      "pushback_rationale": "<1–2 sentences: why push back, what leverage you have>",
      "alternative_wording": "<1–3 sentences: rewritten clause text that is fair to both parties>",
      "negotiation_tips": ["<tip 1 — concrete action>", "<tip 2 — concrete action>"]
    }
  ]
}

RULES:
1. RED (score 7–10): recommended_action = negotiate or reject. Always provide pushback_rationale and alternative_wording.
2. YELLOW (score 4–6.9): recommended_action = negotiate. Provide pushback_rationale; alternative_wording optional.
3. GREEN (score 1–3.9): recommended_action = accept. Set pushback_rationale=null, alternative_wording=null, negotiation_tips=[].
4. Keep pushback_rationale to 1–2 sentences — be direct about the problem and leverage.
5. Keep alternative_wording short (1–3 sentences of actual contract language, not advice).
6. top_risks: list 3–5 highest-severity clauses with their scores.
7. Include ALL clauses in negotiation_advice — do not drop any.
8. overall_score: weighted average (RED clauses weighted 2x, YELLOW 1x, GREEN 0.5x)."""


class NegotiatorAgent:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def _build_user_prompt(self, reasoner_output: LegalReasonerOutput) -> str:
        clauses_text = []
        for c in reasoner_output.clauses:
            entry = (
                f"clause_id={c.clause_id}  [{c.clause_type.value}]  "
                f"{c.risk_level.value} (score={c.severity_score})  "
                f"predatory={c.is_predatory}\n"
                f"  Text: {c.original_text[:150]}...\n" if len(c.original_text) > 150 else
                f"  Text: {c.original_text}\n"
            )
            entry += f"  Explanation: {c.plain_language_explanation[:120]}..."
            clauses_text.append(entry)

        red = sum(1 for c in reasoner_output.clauses if c.risk_level == RiskLevel.RED)
        yellow = sum(1 for c in reasoner_output.clauses if c.risk_level == RiskLevel.YELLOW)

        return (
            f"Advise on {len(reasoner_output.clauses)} clauses from a "
            f"'{reasoner_output.document_type}' contract. "
            f"{red} RED, {yellow} YELLOW clauses need attention.\n\n"
            + "\n\n".join(clauses_text)
            + "\n\nReturn ONLY the negotiation_advice JSON as specified."
        )

    def _coerce_action(self, raw: str | None, risk_level: str) -> str:
        valid = {"accept", "negotiate", "reject"}
        val = (raw or "").lower().strip()
        if val in valid:
            return val
        return "accept" if risk_level == "GREEN" else "negotiate"

    def _parse_response(
        self, raw: Any, source: LegalReasonerOutput
    ) -> dict[str, Any]:
        """
        Validate and clean Agent 4's minimal response.

        Returns a dict with keys: executive_summary, top_risks, overall_score,
        and negotiation_advice (dict keyed by clause_id for merging).
        """
        import copy
        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict from Agent 4, got {type(raw).__name__}")
        # Deep copy so we never mutate the caller's dict (important for test isolation)
        raw = copy.deepcopy(raw)

        raw.setdefault("executive_summary", source.executive_summary)
        raw.setdefault("top_risks", [])
        try:
            raw["overall_score"] = max(1.0, min(10.0, float(raw.get("overall_score", source.overall_score))))
        except (ValueError, TypeError):
            raw["overall_score"] = source.overall_score

        # Build advice lookup
        advice_lookup: dict[str, dict] = {}
        for item in raw.get("negotiation_advice", []):
            if not isinstance(item, dict):
                continue
            cid = item.get("clause_id", "")
            if not cid:
                continue
            risk_level = next(
                (c.risk_level.value for c in source.clauses if c.clause_id == cid),
                "YELLOW",
            )
            advice_lookup[cid] = {
                "recommended_action": self._coerce_action(item.get("recommended_action"), risk_level),
                "pushback_rationale": item.get("pushback_rationale") or None,
                "alternative_wording": item.get("alternative_wording") or None,
                "negotiation_tips": item.get("negotiation_tips") or [],
            }

        # Fill missing clause_ids with safe defaults
        for clause in source.clauses:
            if clause.clause_id not in advice_lookup:
                logger.warning("Agent 4 missing clause — using default", extra={"clause_id": clause.clause_id})
                action = "accept" if clause.risk_level == RiskLevel.GREEN else "negotiate"
                advice_lookup[clause.clause_id] = {
                    "recommended_action": action,
                    "pushback_rationale": None,
                    "alternative_wording": None,
                    "negotiation_tips": [],
                }

        # Auto-generate top_risks if not provided
        if not raw.get("top_risks"):
            sorted_clauses = sorted(
                [c for c in source.clauses if c.severity_score >= 5.0],
                key=lambda x: x.severity_score,
                reverse=True,
            )[:5]
            raw["top_risks"] = [
                f"{c.clause_type.value.replace('_', ' ').title()} clause — {c.severity_score}/10"
                for c in sorted_clauses
            ]

        raw["negotiation_advice"] = advice_lookup
        return raw

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

        parsed = self._parse_response(raw, reasoner_output)
        advice_lookup = parsed["negotiation_advice"]

        # Merge Agent 3 full clause data with Agent 4 additions
        final_clauses = []
        for clause in reasoner_output.clauses:
            adv = advice_lookup.get(clause.clause_id, {})
            final_clauses.append(NegotiationAdvice(
                clause_id=clause.clause_id,
                clause_type=clause.clause_type,
                original_text=clause.original_text,
                is_ambiguous=clause.is_ambiguous,
                ambiguity_note=clause.ambiguity_note,
                contradicts_clause_ids=clause.contradicts_clause_ids,
                severity_score=clause.severity_score,
                risk_level=clause.risk_level,
                risk_category=clause.risk_category,
                benchmark_comparison=clause.benchmark_comparison,
                is_predatory=clause.is_predatory,
                plain_language_explanation=clause.plain_language_explanation,
                scenario_consequence=clause.scenario_consequence,
                key_implications=clause.key_implications,
                recommended_action=adv.get("recommended_action", "accept"),
                pushback_rationale=adv.get("pushback_rationale"),
                alternative_wording=adv.get("alternative_wording"),
                negotiation_tips=adv.get("negotiation_tips", []),
            ))

        result = NegotiationAdvisorOutput(
            clauses=final_clauses,
            overall_score=parsed["overall_score"],
            red_count=0, yellow_count=0, green_count=0,
            document_type=reasoner_output.document_type,
            executive_summary=parsed["executive_summary"],
            top_risks=parsed["top_risks"],
        )

        logger.info(
            "Agent 4 complete",
            extra={"red": result.red_count, "yellow": result.yellow_count, "green": result.green_count},
        )
        return result
