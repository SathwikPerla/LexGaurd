"""
reasoner.py — Agent 3: Legal Reasoner.

Input:  RiskAnalyzerOutput from Agent 2
Output: LegalReasonerOutput — same clauses with plain-language explanations,
        scenario-based consequence simulation, and key implications added.

Responsibilities:
  - Write plain English explanation of each clause (accessible to non-lawyers)
  - Generate "if you sign this and X happens, then Y" scenario simulation
  - Extract 2–4 key implications per clause
  - Write a brief executive summary of the overall document

Design decisions:
  - All Agent 1+2 fields are re-injected from input in _parse_response — if
    Claude omits them, they come from the upstream output, never lost
  - Missing explanation fields get safe defaults rather than crashing
  - Same injection pattern as extractor.py / risk_analyzer.py for consistency
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from core.gemini_client import LLMClient, LLMCallError, LLMParseError
from models.schemas import (
    LegalReasonerOutput,
    ReasonedClause,
    RiskAnalyzerOutput,
    RiskLevel,
    score_to_label,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: str = """You are a legal communication expert. You receive a list of contractual clauses that have already been risk-scored. Your job is to explain each clause in plain language that anyone — including a non-lawyer — can immediately understand.

CRITICAL: Return ONLY valid JSON. No markdown. No explanation text. Just raw JSON.

SCHEMA:
{
  "document_type": "<same as input>",
  "overall_score": <float 1.0–10.0, same as input>,
  "red_count": 0,
  "yellow_count": 0,
  "green_count": 0,
  "executive_summary": "<2–3 sentence plain-English overview of the contract's biggest risks>",
  "clauses": [
    {
      "clause_id": "<same clause_id from input — do not change>",
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
      "plain_language_explanation": "<1–2 sentence plain English: what does this clause actually mean for the person signing?>",
      "scenario_consequence": "<'If you sign this and [realistic situation] happens, then [specific consequence for the signer] — be concrete and specific>",
      "key_implications": ["<implication 1>", "<implication 2>"]
    }
  ]
}

EXPLANATION RULES:
1. plain_language_explanation: Write as if explaining to someone with no legal background. Start with what the clause does to THEM specifically.
2. scenario_consequence: Always use the format "If you sign this and [scenario], then [consequence]." Make it concrete — mention money, job loss, lawsuits, rights lost, etc.
3. key_implications: 2–4 bullet points, each a distinct practical consequence. No legal jargon.
4. executive_summary: Focus on what the signer is most at risk of. Lead with the worst risk.
5. Preserve ALL other fields exactly from input — do not change clause_id, original_text, severity_score, etc.
6. Include EVERY clause from input — do not drop any."""


class ReasonerAgent:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def _build_user_prompt(self, risk_output: RiskAnalyzerOutput) -> str:
        clauses_text = []
        for c in risk_output.clauses:
            entry = (
                f"CLAUSE {c.clause_id} [{c.clause_type.value}] — "
                f"{c.risk_level.value} RISK (score={c.severity_score}):\n"
                f"  Text: {c.original_text}\n"
                f"  Category: {c.risk_category.value}, Predatory: {c.is_predatory}\n"
                f"  Benchmark: {c.benchmark_comparison}"
            )
            clauses_text.append(entry)

        return (
            f"Explain the following {len(risk_output.clauses)} clauses from a "
            f"'{risk_output.document_type}' contract in plain language. "
            f"Overall risk score: {risk_output.overall_score}/10.\n\n"
            + "\n\n".join(clauses_text)
            + "\n\nReturn the JSON object with plain_language_explanation, "
              "scenario_consequence, and key_implications added to each clause."
        )

    def _parse_response(
        self, raw: Any, source: RiskAnalyzerOutput
    ) -> LegalReasonerOutput:
        if isinstance(raw, list):
            raw = {
                "document_type": source.document_type,
                "overall_score": source.overall_score,
                "red_count": 0,
                "yellow_count": 0,
                "green_count": 0,
                "executive_summary": "Document analyzed.",
                "clauses": raw,
            }
        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict from Agent 3, got {type(raw).__name__}")

        raw.setdefault("document_type", source.document_type)
        raw.setdefault("overall_score", source.overall_score)
        raw.setdefault("executive_summary", "Document analysis complete.")
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

            # Re-inject upstream fields from Agent 2 to prevent data loss
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

            # Defaults for new fields
            clause.setdefault(
                "plain_language_explanation",
                f"This is a {clause.get('clause_type', 'general')} clause with "
                f"{clause.get('risk_level', 'UNKNOWN')} risk implications.",
            )
            clause.setdefault(
                "scenario_consequence",
                "If you sign this, you accept the terms as written with no modifications.",
            )
            clause.setdefault("key_implications", ["Review this clause carefully before signing."])

            coerced.append(clause)

        # Re-add any clauses Agent 3 dropped
        for cid, src in source_lookup.items():
            if cid not in seen_ids:
                logger.warning("Agent 3 dropped clause — re-adding", extra={"clause_id": cid})
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
                    "plain_language_explanation": "Review this clause with a legal professional.",
                    "scenario_consequence": "If you sign this, you accept this term as written.",
                    "key_implications": ["Consult a lawyer before agreeing."],
                })

        raw["clauses"] = coerced

        try:
            return LegalReasonerOutput(**raw)
        except ValidationError as exc:
            logger.error("Agent 3 Pydantic validation failed", extra={"errors": exc.errors()})
            raise

    async def run(self, risk_output: RiskAnalyzerOutput) -> LegalReasonerOutput:
        if not risk_output.clauses:
            return LegalReasonerOutput(
                clauses=[],
                overall_score=1.0,
                red_count=0, yellow_count=0, green_count=0,
                document_type=risk_output.document_type,
                executive_summary="No clauses found in this document.",
            )

        logger.info("Agent 3 (Reasoner) starting", extra={"clause_count": len(risk_output.clauses)})

        raw = await self._client.generate_json(
            user_prompt=self._build_user_prompt(risk_output),
            system_prompt=_SYSTEM_PROMPT,
        )

        result = self._parse_response(raw, risk_output)
        logger.info("Agent 3 complete", extra={"clauses": len(result.clauses)})
        return result
