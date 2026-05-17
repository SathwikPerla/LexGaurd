"""
schemas.py — Pydantic v2 models for all LEXGUARD API inputs and outputs.

Design decisions:
  - Pydantic v2 syntax exclusively (model_validator, Field constraints, etc.)
  - All enums are str-based so JSON serialisation is automatic
  - No Optional[X] default=None without explicit reason — forces callers to be honest
  - severity_score is float constrained ge=1 le=10 at the model level (no runtime guard needed)
  - All list fields default to empty list, never None, to avoid downstream None checks
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums — str-based for automatic JSON serialisation
# ─────────────────────────────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class RiskLabel(str, Enum):
    """
    Text-only risk label for accessibility — never rely on color alone.
    Derived deterministically from severity_score:
      HIGH   → score 7.0–10.0  (RED)
      MEDIUM → score 4.0–6.9   (YELLOW)
      LOW    → score 1.0–3.9   (GREEN)
    Always shown alongside risk_level (color) and severity_score (number).
    """
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


def score_to_label(score: float) -> RiskLabel:
    """Deterministic mapping from severity score to text label."""
    if score >= 7.0:
        return RiskLabel.HIGH
    if score >= 4.0:
        return RiskLabel.MEDIUM
    return RiskLabel.LOW


class RiskCategory(str, Enum):
    FINANCIAL = "financial"
    PRIVACY = "privacy"
    EMPLOYMENT = "employment"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    COMPLIANCE = "compliance"
    TERMINATION = "termination"
    ARBITRATION = "arbitration"
    LIABILITY = "liability"
    DATA_COLLECTION = "data_collection"
    AUTO_RENEWAL = "auto_renewal"
    GENERAL = "general"


class RecommendedAction(str, Enum):
    ACCEPT = "accept"
    NEGOTIATE = "negotiate"
    REJECT = "reject"


class ClauseType(str, Enum):
    TERMINATION = "termination"
    IP_TRANSFER = "ip_transfer"
    ARBITRATION = "arbitration"
    LIABILITY = "liability"
    PRIVACY = "privacy"
    NON_COMPETE = "non_compete"
    AUTO_RENEWAL = "auto_renewal"
    DATA_COLLECTION = "data_collection"
    INDEMNIFICATION = "indemnification"
    GOVERNING_LAW = "governing_law"
    CONFIDENTIALITY = "confidentiality"
    PAYMENT = "payment"
    GENERAL = "general"


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — Extractor output
# ─────────────────────────────────────────────────────────────────────────────


class ExtractedClause(BaseModel):
    """
    One clause as labelled by Agent 1.
    clause_id is a stable string key (e.g. "clause_001") used to chain agents.
    """
    clause_id: str = Field(..., min_length=1, description="Unique identifier for this clause")
    clause_type: ClauseType
    original_text: str = Field(..., min_length=1, description="Verbatim clause text")
    is_ambiguous: bool = False
    ambiguity_note: Optional[str] = None
    contradicts_clause_ids: List[str] = Field(default_factory=list)

    # Accessibility: score present even on Agent 1 output so UI can sort/filter
    severity_score: float = Field(default=0.0, ge=0.0, le=10.0)


class ExtractorOutput(BaseModel):
    clauses: List[ExtractedClause] = Field(default_factory=list)
    total_clauses: int = Field(ge=0)
    document_type: str = "unknown"

    @model_validator(mode="after")
    def total_matches_clauses(self) -> "ExtractorOutput":
        """Ensure total_clauses is consistent with the actual list length."""
        if self.total_clauses != len(self.clauses):
            self.total_clauses = len(self.clauses)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Risk Analyzer output
# ─────────────────────────────────────────────────────────────────────────────


class RiskScoredClause(BaseModel):
    """Extends ExtractedClause with risk scoring from Agent 2."""
    clause_id: str
    clause_type: ClauseType
    original_text: str
    is_ambiguous: bool = False
    ambiguity_note: Optional[str] = None
    contradicts_clause_ids: List[str] = Field(default_factory=list)

    severity_score: float = Field(..., ge=1.0, le=10.0)
    risk_level: RiskLevel
    # Computed from severity_score — always present for accessibility
    risk_label: RiskLabel = Field(default=RiskLabel.LOW)
    risk_category: RiskCategory
    benchmark_comparison: str = Field(..., min_length=1)
    is_predatory: bool = False

    @model_validator(mode="after")
    def compute_risk_label(self) -> "RiskScoredClause":
        self.risk_label = score_to_label(self.severity_score)
        return self


class RiskAnalyzerOutput(BaseModel):
    clauses: List[RiskScoredClause] = Field(default_factory=list)
    overall_score: float = Field(..., ge=1.0, le=10.0)
    red_count: int = Field(ge=0)
    yellow_count: int = Field(ge=0)
    green_count: int = Field(ge=0)
    document_type: str

    @model_validator(mode="after")
    def counts_match_clauses(self) -> "RiskAnalyzerOutput":
        """Auto-compute risk counts from clause list to prevent drift."""
        self.red_count = sum(1 for c in self.clauses if c.risk_level == RiskLevel.RED)
        self.yellow_count = sum(1 for c in self.clauses if c.risk_level == RiskLevel.YELLOW)
        self.green_count = sum(1 for c in self.clauses if c.risk_level == RiskLevel.GREEN)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 — Legal Reasoner output
# ─────────────────────────────────────────────────────────────────────────────


class ReasonedClause(BaseModel):
    """Extends RiskScoredClause with plain-language explanation and scenarios."""
    clause_id: str
    clause_type: ClauseType
    original_text: str
    is_ambiguous: bool = False
    ambiguity_note: Optional[str] = None
    contradicts_clause_ids: List[str] = Field(default_factory=list)
    severity_score: float = Field(..., ge=1.0, le=10.0)
    risk_level: RiskLevel
    risk_label: RiskLabel = Field(default=RiskLabel.LOW)
    risk_category: RiskCategory
    benchmark_comparison: str
    is_predatory: bool = False

    @model_validator(mode="after")
    def compute_risk_label(self) -> "ReasonedClause":
        self.risk_label = score_to_label(self.severity_score)
        return self

    plain_language_explanation: str = Field(..., min_length=1)
    scenario_consequence: str = Field(..., min_length=1)
    key_implications: List[str] = Field(default_factory=list)


class LegalReasonerOutput(BaseModel):
    clauses: List[ReasonedClause] = Field(default_factory=list)
    overall_score: float = Field(..., ge=1.0, le=10.0)
    red_count: int = Field(ge=0)
    yellow_count: int = Field(ge=0)
    green_count: int = Field(ge=0)
    document_type: str
    executive_summary: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def counts_match_clauses(self) -> "LegalReasonerOutput":
        self.red_count = sum(1 for c in self.clauses if c.risk_level == RiskLevel.RED)
        self.yellow_count = sum(1 for c in self.clauses if c.risk_level == RiskLevel.YELLOW)
        self.green_count = sum(1 for c in self.clauses if c.risk_level == RiskLevel.GREEN)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 — Negotiation Advisor output (final report structure)
# ─────────────────────────────────────────────────────────────────────────────


class NegotiationAdvice(BaseModel):
    """
    Full per-clause data — aggregates all 4 agents' outputs.

    Accessibility guarantee: every instance carries all three risk indicators:
      risk_level  → color code  (RED / YELLOW / GREEN)
      severity_score → numeric  (1.0 – 10.0)
      risk_label  → text label  (HIGH / MEDIUM / LOW)
    The frontend must display all three. Never rely on color alone.
    """
    clause_id: str
    clause_type: ClauseType
    original_text: str
    is_ambiguous: bool = False
    ambiguity_note: Optional[str] = None
    contradicts_clause_ids: List[str] = Field(default_factory=list)
    severity_score: float = Field(..., ge=1.0, le=10.0)
    risk_level: RiskLevel
    risk_label: RiskLabel = Field(default=RiskLabel.LOW)
    risk_category: RiskCategory
    benchmark_comparison: str
    is_predatory: bool = False
    plain_language_explanation: str
    scenario_consequence: str
    key_implications: List[str] = Field(default_factory=list)

    # Negotiation-specific
    recommended_action: RecommendedAction
    pushback_rationale: Optional[str] = None
    alternative_wording: Optional[str] = None
    negotiation_tips: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def compute_and_validate(self) -> "NegotiationAdvice":
        # Always compute risk_label from score — never trust caller to set it
        self.risk_label = score_to_label(self.severity_score)
        return self


class NegotiationAdvisorOutput(BaseModel):
    clauses: List[NegotiationAdvice] = Field(default_factory=list)
    overall_score: float = Field(..., ge=1.0, le=10.0)
    red_count: int = Field(ge=0)
    yellow_count: int = Field(ge=0)
    green_count: int = Field(ge=0)
    document_type: str
    executive_summary: str = Field(..., min_length=1)
    top_risks: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def counts_match_clauses(self) -> "NegotiationAdvisorOutput":
        self.red_count = sum(1 for c in self.clauses if c.risk_level == RiskLevel.RED)
        self.yellow_count = sum(1 for c in self.clauses if c.risk_level == RiskLevel.YELLOW)
        self.green_count = sum(1 for c in self.clauses if c.risk_level == RiskLevel.GREEN)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# API — Request / Response types
# ─────────────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    service: str = "lexguard-backend"


class ParsedDocumentResponse(BaseModel):
    """Response from /parse — extracted text plus metadata."""
    filename: str
    extracted_text: str
    character_count: int = Field(ge=0)
    page_count: Optional[int] = None
    parse_method: str  # 'pdf' | 'docx' | 'ocr'

    @model_validator(mode="after")
    def character_count_matches_text(self) -> "ParsedDocumentResponse":
        self.character_count = len(self.extracted_text)
        return self


class AnalyzeResponse(BaseModel):
    """Full pipeline response — the complete risk intelligence report."""
    filename: str
    parse_method: str
    report: NegotiationAdvisorOutput
    agents_completed: List[str] = Field(default_factory=list)
