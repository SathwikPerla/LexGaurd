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

_SYSTEM_PROMPT: str = """You are a contract negotiation expert. Advise the signer on what to do about each clause.

CRITICAL: Return ONLY valid JSON. No markdown. No explanation. Just raw JSON.

SCHEMA:
{
  "executive_summary": "<MAX 150 chars: what must be fixed before signing?>",
  "top_risks": ["<MAX 80 chars each, 3-5 items>"],
  "negotiation_advice": [
    {
      "clause_id": "<same clause_id from input>",
      "recommended_action": "<accept|negotiate|reject>",
      "pushback_rationale": "<MAX 120 chars: why push back. Null for GREEN clauses.>",
      "alternative_wording": "<MAX 150 chars: rewritten clause. Null for GREEN clauses.>",
      "negotiation_tips": ["<MAX 80 chars>", "<MAX 80 chars>"]
    }
  ]
}

STRICT LENGTH RULES — token budget is limited:
1. executive_summary: max 150 characters.
2. top_risks: 3-5 items, each max 80 characters.
3. pushback_rationale: max 120 characters. Null for accept clauses.
4. alternative_wording: max 150 characters of actual rewritten contract text. Null for accept.
5. negotiation_tips: exactly 2 items for negotiate/reject, empty array for accept. Each max 80 chars.
6. RED/YELLOW clauses: recommended_action = negotiate or reject, always provide pushback + wording.
7. GREEN clauses: recommended_action = accept, all other fields null or empty.
8. Include ALL clauses. BREVITY IS REQUIRED — every field must be as short as possible."""


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
        # Agent 2 is the source of truth for overall_score (passed through Agent 3).
        # Agent 4 must never generate or overwrite it — ignore any value Claude returns.
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
