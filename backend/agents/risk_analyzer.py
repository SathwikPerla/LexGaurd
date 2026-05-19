"""
risk_analyzer.py — Agent 2: Risk Analyzer.

Receives ExtractorOutput from Agent 1, returns RiskAnalyzerOutput.

Responsibilities:
  - Score each clause 1–10 on severity
  - Classify each clause as RED / YELLOW / GREEN
  - Identify risk category (financial, privacy, employment, IP, compliance, etc.)
  - Compare each clause against standard benchmarks using semantic similarity
  - Flag whether a clause is predatory vs industry-standard
  - Compute an overall contract risk score

Design decisions:
  - GeminiClient and EmbeddingsStore are injected — never instantiated inside this class
  - Benchmark comparison runs for every clause before calling Gemini, so the prompt
    includes concrete comparison context ("similar standard clause is X, yours is Y")
    This makes Gemini's scoring more grounded and defensible to judges
  - If EmbeddingsStore is None (e.g. no credentials yet), benchmark context is skipped
    gracefully — agent still works, just without semantic comparison
  - response_mime_type="application/json" enforced via generate_json
  - Pydantic validates all output — callers always get RiskAnalyzerOutput or an exception
  - risk_label is always computed from severity_score by Pydantic — never trusted from Gemini

Accessibility guarantee:
  - Every clause in output has risk_level + severity_score + risk_label together
  - Enforced by RiskScoredClause.compute_risk_label model_validator

Common failure points:
  - Gemini scores a clause outside 1–10 range → Pydantic rejects it (ge=1.0, le=10.0)
  - Gemini uses wrong risk_category string → coerced to "general" with a warning
  - Empty extractor output → returns empty RiskAnalyzerOutput gracefully
  - Benchmark query fails → logged and skipped, not fatal
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import ValidationError

from core.embeddings import EmbeddingsStore
from core.gemini_client import LLMCallError, LLMParseError, LLMClient
from models.schemas import (
    ExtractorOutput,
    RiskAnalyzerOutput,
    RiskCategory,
)

# Backward-compat aliases
GeminiClient = LLMClient
GeminiCallError = LLMCallError
GeminiParseError = LLMParseError

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — Agent 2 persona and output specification
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT: str = """You are a legal risk analysis expert. You receive a list of contractual clauses that have already been extracted and labeled. Your job is to score the risk of each clause for the person being asked to sign.

CRITICAL: Return ONLY valid JSON matching the schema below. No markdown, no code fences, no explanation. Just raw JSON.

SCHEMA:
{
  "document_type": "<string matching the input document_type>",
  "overall_score": <float 1.0–10.0, weighted average severity of all clauses>,
  "red_count": 0,
  "yellow_count": 0,
  "green_count": 0,
  "clauses": [
    {
      "clause_id": "<same clause_id from input — do not change>",
      "clause_type": "<same clause_type from input — do not change>",
      "original_text": "<same original_text from input — do not change>",
      "is_ambiguous": <true|false from input>,
      "ambiguity_note": <null or string from input>,
      "contradicts_clause_ids": <array from input>,
      "severity_score": <float 1.0–10.0>,
      "risk_level": "<RED|YELLOW|GREEN>",
      "risk_label": "<HIGH|MEDIUM|LOW>",
      "risk_category": "<financial|privacy|employment|intellectual_property|compliance|termination|arbitration|liability|data_collection|auto_renewal|general>",
      "benchmark_comparison": "<string: how this clause compares to industry standard>",
      "is_predatory": <true|false>
    }
  ]
}

SCORING RULES:
1. severity_score 7.0–10.0 → risk_level=RED, risk_label=HIGH (very harmful clauses)
2. severity_score 4.0–6.9 → risk_level=YELLOW, risk_label=MEDIUM (notable risk, common but worth negotiating)
3. severity_score 1.0–3.9 → risk_level=GREEN, risk_label=LOW (standard, acceptable clauses)
4. is_predatory=true when the clause goes significantly beyond industry standard to favor the drafter at the signer's expense
5. benchmark_comparison MUST reference specific industry norms (e.g. "Standard non-competes are 6-12 months; this is 24 months")
6. risk_category MUST be one of the exact strings in the schema
7. red_count, yellow_count, green_count in the output must be 0 — they will be auto-computed
8. overall_score is the weighted average of severity scores — weight RED clauses 2x, YELLOW 1x, GREEN 0.5x
9. Preserve clause_id, clause_type, original_text, is_ambiguous, ambiguity_note, contradicts_clause_ids exactly from input
10. Every clause from input MUST appear in output — do not drop any"""


# ─────────────────────────────────────────────────────────────────────────────
# Valid risk category strings — for coercion of Gemini output
# ─────────────────────────────────────────────────────────────────────────────

_VALID_RISK_CATEGORIES: frozenset[str] = frozenset(v.value for v in RiskCategory)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Risk Analyzer
# ─────────────────────────────────────────────────────────────────────────────


class RiskAnalyzerAgent:
    """
    Agent 2: Scores and classifies each clause for risk.

    Args:
        client:     GeminiClient (injected — tests pass a fake)
        embeddings: EmbeddingsStore for benchmark comparison (optional — if None,
                    benchmark context is skipped but agent still runs)
    """

    def __init__(
        self,
        client: GeminiClient,
        embeddings: Optional[EmbeddingsStore] = None,
    ) -> None:
        self._client = client
        self._embeddings = embeddings

    async def _get_benchmark_context(
        self, clause_text: str, clause_type: str
    ) -> str:
        """
        Query ChromaDB for similar benchmarks and format as context for Gemini.

        Returns empty string if embeddings unavailable or query fails.
        """
        if self._embeddings is None:
            return ""

        try:
            similar = await self._embeddings.find_similar(
                clause_text=clause_text,
                clause_type=clause_type,
                n_results=2,
            )
        except Exception as exc:
            logger.warning(
                "Benchmark query failed — proceeding without context",
                extra={"error": str(exc)},
            )
            return ""

        if not similar:
            return ""

        lines = []
        for b in similar:
            predatory = "PREDATORY" if b["is_predatory"] else "INDUSTRY STANDARD"
            lines.append(
                f"  - [{predatory}, {b['risk_level']}, score={b['severity_score']}] "
                f"{b['text'][:120]}... — {b['notes']}"
            )

        return "BENCHMARK CONTEXT (similar clauses from legal database):\n" + "\n".join(lines)

    def _build_user_prompt(
        self,
        extractor_output: ExtractorOutput,
        benchmark_contexts: dict[str, str],
    ) -> str:
        """Build the user prompt with all clauses and their benchmark contexts."""
        clauses_text = []
        for clause in extractor_output.clauses:
            ctx = benchmark_contexts.get(clause.clause_id, "")
            entry = (
                f"CLAUSE {clause.clause_id} [{clause.clause_type.value}]:\n"
                f"  Text: {clause.original_text}\n"
                f"  Ambiguous: {clause.is_ambiguous}"
            )
            if clause.ambiguity_note:
                entry += f" — {clause.ambiguity_note}"
            if ctx:
                entry += f"\n  {ctx}"
            clauses_text.append(entry)

        clauses_block = "\n\n".join(clauses_text)

        return (
            f"Analyze the following {len(extractor_output.clauses)} clauses from a "
            f"'{extractor_output.document_type}' document. "
            f"Score each for risk severity from the perspective of the person signing.\n\n"
            f"{clauses_block}\n\n"
            f"Return the complete JSON object with all clauses scored."
        )

    def _coerce_clause(self, raw: dict[str, Any], idx: int) -> dict[str, Any]:
        """
        Coerce Gemini output fields to valid values before Pydantic validation.

        Handles:
          - severity_score out of range → clamped to 1.0–10.0
          - risk_level not in {RED, YELLOW, GREEN} → derived from severity_score
          - risk_category not in valid set → defaults to 'general'
          - Missing fields → sensible defaults
        """
        # Clamp severity_score
        score = float(raw.get("severity_score", 5.0))
        score = max(1.0, min(10.0, score))
        raw["severity_score"] = score

        # Derive risk_level from score if missing or invalid
        valid_levels = {"RED", "YELLOW", "GREEN"}
        level = str(raw.get("risk_level", "")).upper()
        if level not in valid_levels:
            if score >= 7.0:
                level = "RED"
            elif score >= 4.0:
                level = "YELLOW"
            else:
                level = "GREEN"
            logger.warning(
                "Gemini returned invalid risk_level — derived from score",
                extra={"clause_id": raw.get("clause_id"), "score": score, "derived": level},
            )
        raw["risk_level"] = level

        # risk_label is always computed by Pydantic — but must be a valid string here
        label_map = {"RED": "HIGH", "YELLOW": "MEDIUM", "GREEN": "LOW"}
        raw["risk_label"] = label_map[level]

        # Coerce risk_category
        category = str(raw.get("risk_category", "general")).lower().replace(" ", "_").replace("-", "_")
        if category not in _VALID_RISK_CATEGORIES:
            logger.warning(
                "Gemini returned invalid risk_category — defaulting to 'general'",
                extra={"clause_id": raw.get("clause_id"), "received": category},
            )
            category = "general"
        raw["risk_category"] = category

        # Ensure benchmark_comparison is always a non-empty string
        if not raw.get("benchmark_comparison", "").strip():
            raw["benchmark_comparison"] = "No benchmark comparison available for this clause type."

        # Defaults for fields that must come from Agent 1 input
        raw.setdefault("is_ambiguous", False)
        raw.setdefault("ambiguity_note", None)
        raw.setdefault("contradicts_clause_ids", [])
        raw.setdefault("is_predatory", False)

        return raw

    def _parse_response(
        self,
        raw: Any,
        extractor_output: ExtractorOutput,
    ) -> RiskAnalyzerOutput:
        """
        Validate Gemini's response dict into a RiskAnalyzerOutput.

        Falls back to marking all clauses as YELLOW/5.0 if Gemini returns
        a structurally valid but incomplete response — ensures output is never
        missing clauses from the input.

        Raises:
            ValueError: If response is not a dict/list.
            ValidationError: If a clause fails Pydantic constraints after coercion.
        """
        if isinstance(raw, list):
            raw = {
                "document_type": extractor_output.document_type,
                "overall_score": 5.0,
                "red_count": 0,
                "yellow_count": 0,
                "green_count": 0,
                "clauses": raw,
            }

        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict from Gemini (Agent 2), got {type(raw).__name__}")

        raw.setdefault("document_type", extractor_output.document_type)
        raw.setdefault("overall_score", 5.0)
        raw.setdefault("red_count", 0)
        raw.setdefault("yellow_count", 0)
        raw.setdefault("green_count", 0)

        # Clamp overall_score
        try:
            overall = float(raw["overall_score"])
            raw["overall_score"] = max(1.0, min(10.0, overall))
        except (ValueError, TypeError):
            raw["overall_score"] = 5.0

        # Build a lookup of input clauses so we can fill in missing fields
        input_lookup: dict[str, Any] = {
            c.clause_id: c for c in extractor_output.clauses
        }

        coerced_clauses = []
        output_ids: set[str] = set()

        for idx, clause in enumerate(raw.get("clauses", [])):
            if not isinstance(clause, dict):
                logger.warning("Non-dict clause in Gemini output — skipping", extra={"idx": idx})
                continue

            clause_id = clause.get("clause_id", f"clause_{idx + 1:03d}")
            output_ids.add(clause_id)

            # Re-inject Agent 1 fields that must not change
            if clause_id in input_lookup:
                src = input_lookup[clause_id]
                clause["clause_id"] = src.clause_id
                clause["clause_type"] = src.clause_type.value
                clause["original_text"] = src.original_text
                clause["is_ambiguous"] = src.is_ambiguous
                clause["ambiguity_note"] = src.ambiguity_note
                clause["contradicts_clause_ids"] = src.contradicts_clause_ids

            coerced_clauses.append(self._coerce_clause(clause, idx))

        # Safety net: if Agent 2 dropped any clauses from Agent 1's output, add them back
        for cid, input_clause in input_lookup.items():
            if cid not in output_ids:
                logger.warning(
                    "Agent 2 dropped a clause — re-adding with default score",
                    extra={"clause_id": cid},
                )
                fallback = {
                    "clause_id": input_clause.clause_id,
                    "clause_type": input_clause.clause_type.value,
                    "original_text": input_clause.original_text,
                    "is_ambiguous": input_clause.is_ambiguous,
                    "ambiguity_note": input_clause.ambiguity_note,
                    "contradicts_clause_ids": input_clause.contradicts_clause_ids,
                    "severity_score": 5.0,
                    "risk_level": "YELLOW",
                    "risk_label": "MEDIUM",
                    "risk_category": "general",
                    "benchmark_comparison": "Unable to assess — clause not scored by agent.",
                    "is_predatory": False,
                }
                coerced_clauses.append(fallback)

        raw["clauses"] = coerced_clauses

        try:
            return RiskAnalyzerOutput(**raw)
        except ValidationError as exc:
            logger.error(
                "Pydantic validation failed for RiskAnalyzerOutput",
                extra={"errors": exc.errors()},
            )
            raise

    async def run(self, extractor_output: ExtractorOutput) -> RiskAnalyzerOutput:
        """
        Run Agent 2 on the output from Agent 1.

        Args:
            extractor_output: Validated output from ExtractorAgent.

        Returns:
            RiskAnalyzerOutput — every clause scored, classified, and benchmarked.

        Raises:
            GeminiCallError:  API failed after retries.
            GeminiParseError: Gemini returned non-JSON.
            ValidationError:  Output failed Pydantic constraints after coercion.
        """
        if not extractor_output.clauses:
            logger.warning("Agent 2 received empty clause list — returning empty output")
            return RiskAnalyzerOutput(
                clauses=[],
                overall_score=1.0,
                red_count=0,
                yellow_count=0,
                green_count=0,
                document_type=extractor_output.document_type,
            )

        logger.info(
            "Agent 2 (Risk Analyzer) starting",
            extra={"clause_count": len(extractor_output.clauses)},
        )

        # Gather benchmark context for all clauses in parallel-ish (sequential is safe)
        benchmark_contexts: dict[str, str] = {}
        for clause in extractor_output.clauses:
            ctx = await self._get_benchmark_context(
                clause_text=clause.original_text,
                clause_type=clause.clause_type.value,
            )
            benchmark_contexts[clause.clause_id] = ctx

        user_prompt = self._build_user_prompt(extractor_output, benchmark_contexts)

        raw_response = await self._client.generate_json(
            user_prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
        )

        logger.info(
            "Agent 2 Gemini response received",
            extra={
                "type": type(raw_response).__name__,
                "clause_count": len(raw_response.get("clauses", [])) if isinstance(raw_response, dict) else "?",
            },
        )

        result = self._parse_response(raw_response, extractor_output)

        logger.info(
            "Agent 2 complete",
            extra={
                "overall_score": result.overall_score,
                "red": result.red_count,
                "yellow": result.yellow_count,
                "green": result.green_count,
            },
        )

        return result
