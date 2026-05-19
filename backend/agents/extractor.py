"""
extractor.py — Agent 1: Clause Extractor.

Receives raw document text, returns a structured list of labeled clauses.

Responsibilities:
  - Split the document into individual clauses
  - Label each clause with a ClauseType (non_compete, ip_transfer, arbitration, etc.)
  - Flag ambiguous language with an explanation
  - Detect contradictions between clauses
  - Return ExtractorOutput validated by Pydantic

Design decisions:
  - GeminiClient is injected (not imported globally) — allows unit tests to pass a fake
  - response_mime_type="application/json" enforced via GeminiClient.generate_json()
  - System prompt is a module-level constant — no logic, easy to audit and update
  - Pydantic validation runs on the raw dict BEFORE returning — callers always get
    a valid ExtractorOutput or a raised exception, never a half-formed dict
  - User input sanitized: leading/trailing whitespace stripped, control chars removed,
    length capped at MAX_DOC_CHARS before passing to Gemini (prompt injection mitigation)
  - Document type detection is part of Agent 1's output — allows downstream agents to
    tailor their benchmarks (employment vs SaaS vs rental)

Common failure points:
  - Gemini returns valid JSON but wrong schema → Pydantic raises ValidationError
  - Very short documents (< 3 sentences) → Gemini returns empty clause list
  - Very long documents → truncated at MAX_DOC_CHARS before sending to Gemini
  - Ambiguity detection miss → is_ambiguous=False but text actually is vague — mitigated
    by explicit system prompt instruction
"""
from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import ValidationError

from core.gemini_client import LLMCallError, LLMParseError, LLMClient
from models.schemas import ExtractorOutput

# Backward-compat aliases (tests import these names)
GeminiClient = LLMClient
GeminiCallError = LLMCallError
GeminiParseError = LLMParseError

logger = logging.getLogger(__name__)

# Safety cap — prevents Gemini context overflow on very large contracts
MAX_DOC_CHARS: int = 80_000

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — Agent 1 persona and output specification
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT: str = """You are a legal clause extraction expert analyzing contracts from the perspective of the person who is being asked to sign.

CRITICAL: Return ONLY valid JSON. No markdown, no code fences, no explanation. Just the raw JSON object.

SCHEMA:
{
  "document_type": "<employment_agreement | saas_terms | rental_agreement | vendor_agreement | privacy_policy | freelance_contract | subscription_agreement | nda | other>",
  "total_clauses": <integer — count of clauses in the array below>,
  "clauses": [
    {
      "clause_id": "clause_001",
      "clause_type": "<termination | ip_transfer | arbitration | liability | privacy | non_compete | auto_renewal | data_collection | indemnification | governing_law | confidentiality | payment | general>",
      "original_text": "<verbatim clause text — keep under 300 characters, truncate with '...' if longer>",
      "is_ambiguous": <true|false>,
      "ambiguity_note": <null or one short sentence>,
      "contradicts_clause_ids": []
    }
  ]
}

EXTRACTION RULES:
1. Extract AT MOST 15 clauses — the 15 MOST legally significant ones for the person signing.
   Priority order: liability, ip_transfer, data_collection, privacy, non_compete, arbitration,
   auto_renewal, indemnification, termination, governing_law, confidentiality, payment.
   Skip routine boilerplate (definitions, notices, headings, integration clauses).
2. If the document has more than 15 significant clauses, pick the 15 most harmful or risky.
3. Keep original_text under 300 characters — truncate with '...' if the clause is longer.
4. Assign sequential IDs: clause_001, clause_002, etc.
5. total_clauses must equal the actual number of clauses in the array."""


# ─────────────────────────────────────────────────────────────────────────────
# Input sanitization
# ─────────────────────────────────────────────────────────────────────────────


def _sanitize_document_text(raw: str) -> str:
    """
    Sanitize user-supplied document text before inserting into a Gemini prompt.

    Mitigates prompt injection by:
      - Removing null bytes and unusual control characters
      - Stripping leading/trailing whitespace
      - Truncating to MAX_DOC_CHARS

    Does NOT alter legal content — preserves special characters that appear in
    legal text (§, ©, em-dashes, quotes, brackets).
    """
    if not raw:
        raise ValueError("Document text is empty.")

    text = raw.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.strip()

    if len(text) < 10:
        raise ValueError(
            "Document text is too short (under 10 characters after sanitization). "
            "Nothing to extract."
        )

    if len(text) > MAX_DOC_CHARS:
        logger.warning(
            "Document truncated for Agent 1",
            extra={"original_chars": len(text), "limit": MAX_DOC_CHARS},
        )
        text = text[:MAX_DOC_CHARS]

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — Extractor
# ─────────────────────────────────────────────────────────────────────────────


class ExtractorAgent:
    """
    Agent 1: Extracts and labels clauses from a raw contract document.

    Inject GeminiClient at construction time — do not instantiate GeminiClient
    inside this class so tests can pass a fake client without credentials.
    """

    def __init__(self, client: GeminiClient) -> None:
        self._client = client

    def _build_user_prompt(self, document_text: str) -> str:
        """Construct the user-facing part of the prompt with the sanitized document."""
        return (
            "Extract all legally significant clauses from the following contract document. "
            "Identify clause types, flag ambiguities, and detect contradictions.\n\n"
            "DOCUMENT TEXT:\n"
            "---\n"
            f"{document_text}\n"
            "---\n\n"
            "Return the JSON object as specified. Include every significant clause."
        )

    def _parse_response(self, raw: dict[str, Any]) -> ExtractorOutput:
        """
        Validate the Gemini response dict against ExtractorOutput schema.

        Handles edge cases:
          - Gemini returns clauses as top-level list instead of dict → wraps it
          - Missing total_clauses → Pydantic model_validator auto-corrects it
          - Invalid clause_type → Pydantic raises ValidationError with field name

        Raises:
            ValueError: If the response structure is unrecognisable.
            ValidationError: If clause data fails Pydantic constraints.
        """
        # Edge case: Gemini returns a list instead of a dict
        if isinstance(raw, list):
            raw = {"clauses": raw, "total_clauses": len(raw), "document_type": "unknown"}

        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict from Gemini, got {type(raw).__name__}")

        # Ensure clauses key exists
        if "clauses" not in raw:
            logger.warning("Gemini response missing 'clauses' key", extra={"keys": list(raw.keys())})
            raw["clauses"] = []

        if "total_clauses" not in raw:
            raw["total_clauses"] = len(raw["clauses"])

        if "document_type" not in raw:
            raw["document_type"] = "unknown"

        # Coerce clause_type values — Gemini sometimes adds spaces or wrong case
        for clause in raw.get("clauses", []):
            if "clause_type" in clause and isinstance(clause["clause_type"], str):
                clause["clause_type"] = clause["clause_type"].strip().lower().replace(" ", "_").replace("-", "_")
            # Ensure clause_id exists
            if "clause_id" not in clause or not clause["clause_id"]:
                idx = raw["clauses"].index(clause) + 1
                clause["clause_id"] = f"clause_{idx:03d}"
            # Ensure original_text exists
            if "original_text" not in clause or not clause["original_text"]:
                clause["original_text"] = "[Text not extracted]"
            # Default booleans
            clause.setdefault("is_ambiguous", False)
            clause.setdefault("ambiguity_note", None)
            clause.setdefault("contradicts_clause_ids", [])

        try:
            return ExtractorOutput(**raw)
        except ValidationError as exc:
            logger.error(
                "Pydantic validation failed for ExtractorOutput",
                extra={"errors": exc.errors(), "raw_keys": list(raw.keys())},
            )
            raise

    async def run(self, document_text: str) -> ExtractorOutput:
        """
        Run Agent 1 on the raw document text.

        Args:
            document_text: Raw text extracted from the uploaded contract.

        Returns:
            ExtractorOutput — validated, structured list of labeled clauses.

        Raises:
            ValueError:          Document is empty or too short.
            GeminiCallError:     Gemini API failed after retries.
            GeminiParseError:    Gemini returned non-JSON.
            ValidationError:     Gemini JSON failed Pydantic validation.
        """
        sanitized = _sanitize_document_text(document_text)

        logger.info(
            "Agent 1 (Extractor) starting",
            extra={"doc_chars": len(sanitized)},
        )

        user_prompt = self._build_user_prompt(sanitized)

        raw_response = await self._client.generate_json(
            user_prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
        )

        logger.info(
            "Agent 1 Gemini response received",
            extra={
                "type": type(raw_response).__name__,
                "clause_count": len(raw_response.get("clauses", [])) if isinstance(raw_response, dict) else "?",
            },
        )

        result = self._parse_response(raw_response)

        logger.info(
            "Agent 1 complete",
            extra={
                "document_type": result.document_type,
                "total_clauses": result.total_clauses,
                "clause_types": [c.clause_type.value for c in result.clauses],
            },
        )

        return result
