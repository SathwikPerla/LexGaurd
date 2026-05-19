"""
test_api.py — FastAPI endpoint integration tests.
Run: pytest backend/tests/test_api.py -v

The /analyze endpoint is tested with a mocked pipeline — no real API calls.
Real end-to-end tests require ANTHROPIC_API_KEY and are run manually.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.schemas import (
    AnalyzeResponse,
    ClauseType,
    NegotiationAdvice,
    NegotiationAdvisorOutput,
    RecommendedAction,
    RiskCategory,
    RiskLevel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fake pipeline response — matches full AnalyzeResponse schema
# ─────────────────────────────────────────────────────────────────────────────


def _make_fake_response(filename: str = "test.docx") -> AnalyzeResponse:
    clauses = [
        NegotiationAdvice(
            clause_id="clause_001",
            clause_type=ClauseType.NON_COMPETE,
            original_text="Employee shall not compete for 2 years.",
            is_ambiguous=True,
            ambiguity_note="'Compete' is undefined.",
            severity_score=8.5,
            risk_level=RiskLevel.RED,
            risk_category=RiskCategory.EMPLOYMENT,
            benchmark_comparison="Standard non-competes are 6-12 months.",
            is_predatory=True,
            plain_language_explanation="You cannot work for a competitor for 2 years.",
            scenario_consequence="If you leave and join a competitor, you could be sued.",
            key_implications=["Limits career options", "2 years is above standard"],
            recommended_action=RecommendedAction.NEGOTIATE,
            pushback_rationale="Courts often void non-competes beyond 12 months.",
            alternative_wording="Employee shall not solicit Company clients for 6 months.",
            negotiation_tips=["Request 6-month reduction", "Ask for garden leave pay"],
        ),
        NegotiationAdvice(
            clause_id="clause_002",
            clause_type=ClauseType.GOVERNING_LAW,
            original_text="This agreement is governed by Delaware law.",
            is_ambiguous=False,
            severity_score=2.0,
            risk_level=RiskLevel.GREEN,
            risk_category=RiskCategory.COMPLIANCE,
            benchmark_comparison="Delaware governing law is standard.",
            is_predatory=False,
            plain_language_explanation="Legal disputes use Delaware law.",
            scenario_consequence="If you have a dispute, Delaware law applies.",
            key_implications=["Standard corporate practice"],
            recommended_action=RecommendedAction.ACCEPT,
        ),
    ]
    report = NegotiationAdvisorOutput(
        clauses=clauses,
        overall_score=6.5,
        red_count=0, yellow_count=0, green_count=0,
        document_type="Employment Agreement",
        executive_summary="This contract has one significant red flag: the 2-year non-compete.",
        top_risks=["Non-compete clause (8.5/10)"],
    )
    return AnalyzeResponse(
        filename=filename,
        parse_method="docx",
        report=report,
        agents_completed=["extractor", "risk_analyzer", "reasoner", "negotiator"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# App client with pipeline and lifespan mocked
# ─────────────────────────────────────────────────────────────────────────────


def _make_client() -> TestClient:
    """TestClient with pipeline components patched — no API calls."""
    from main import app

    mock_llm = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.ingest_benchmarks.return_value = 0

    # Pre-set app state so lifespan sees it as ready
    app.state.llm_client = mock_llm
    app.state.embeddings_store = mock_embeddings
    app.state.pipeline_ready = True

    return TestClient(app)


client = _make_client()


def make_pdf(text: str = "Test legal clause.") -> bytes:
    pdf = (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length " + str(len(text) + 21).encode() + b" >>\nstream\nBT /F1 12 Tf 72 720 Td (" + text.encode() + b") Tj ET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000266 00000 n \n0000000360 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n441\n%%EOF"
    )
    return pdf


def make_docx(text: str = "Test legal clause.") -> bytes:
    from docx import Document
    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph(text)
    doc.save(buf)
    return buf.getvalue()


# ── /health ───────────────────────────────────────────────────────────────────


class TestHealth:
    def test_200(self):
        assert client.get("/health").status_code == 200

    def test_status_field_present(self):
        r = client.get("/health").json()
        assert "status" in r

    def test_service_name(self):
        assert client.get("/health").json()["service"] == "lexguard-backend"

    def test_request_id_header(self):
        r = client.get("/health")
        assert "x-request-id" in r.headers


# ── /parse ────────────────────────────────────────────────────────────────────


class TestParse:
    def test_pdf_200(self):
        r = client.post("/parse", files={"file": ("c.pdf", make_pdf(), "application/pdf")})
        assert r.status_code == 200

    def test_docx_200(self):
        mt = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        r = client.post("/parse", files={"file": ("c.docx", make_docx(), mt)})
        assert r.status_code == 200

    def test_docx_text_present(self):
        mt = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        r = client.post("/parse", files={"file": ("c.docx", make_docx("Confidentiality clause."), mt)})
        assert "Confidentiality" in r.json()["extracted_text"]

    def test_unsupported_type_415(self):
        r = client.post("/parse", files={"file": ("c.txt", b"data", "text/plain")})
        assert r.status_code == 415

    def test_jpg_rejected(self):
        r = client.post("/parse", files={"file": ("scan.jpg", b"\xff\xd8", "image/jpeg")})
        assert r.status_code == 415

    def test_empty_file_422(self):
        r = client.post("/parse", files={"file": ("empty.pdf", b"", "application/pdf")})
        assert r.status_code == 422

    def test_response_has_parse_method(self):
        mt = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        r = client.post("/parse", files={"file": ("c.docx", make_docx(), mt)})
        assert r.json()["parse_method"] in ("pdf", "docx", "ocr")

    def test_character_count_positive(self):
        mt = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        r = client.post("/parse", files={"file": ("c.docx", make_docx("Some contract text."), mt)})
        assert r.json()["character_count"] > 0

    def test_request_id_in_header(self):
        r = client.post("/parse", files={"file": ("c.pdf", make_pdf(), "application/pdf")})
        assert "x-request-id" in r.headers


# ── /analyze ──────────────────────────────────────────────────────────────────


class TestAnalyze:
    """All /analyze tests mock core.graph.run_pipeline — zero API calls."""

    def _post_docx(self, text: str = "Employment contract terms."):
        mt = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        with patch("core.graph.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
            mock_pipeline.return_value = _make_fake_response("c.docx")
            return client.post("/analyze", files={"file": ("c.docx", make_docx(text), mt)})

    def _post_pdf(self):
        with patch("core.graph.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
            mock_pipeline.return_value = _make_fake_response("c.pdf")
            return client.post("/analyze", files={"file": ("c.pdf", make_pdf(), "application/pdf")})

    def test_pdf_200(self):
        assert self._post_pdf().status_code == 200

    def test_docx_200(self):
        assert self._post_docx().status_code == 200

    def test_report_has_clauses(self):
        data = self._post_docx().json()
        assert len(data["report"]["clauses"]) > 0

    def test_overall_score_in_range(self):
        score = self._post_docx().json()["report"]["overall_score"]
        assert 1.0 <= score <= 10.0

    def test_risk_counts_present(self):
        report = self._post_docx().json()["report"]
        assert "red_count" in report
        assert "yellow_count" in report
        assert "green_count" in report

    def test_clause_has_required_fields(self):
        clauses = self._post_docx().json()["report"]["clauses"]
        required = {
            "clause_id", "severity_score", "risk_level",
            "plain_language_explanation", "recommended_action",
        }
        for clause in clauses:
            assert required.issubset(clause.keys())

    def test_risk_levels_valid(self):
        clauses = self._post_docx().json()["report"]["clauses"]
        for clause in clauses:
            assert clause["risk_level"] in ("RED", "YELLOW", "GREEN")

    def test_severity_scores_in_range(self):
        clauses = self._post_docx().json()["report"]["clauses"]
        for clause in clauses:
            assert 1.0 <= clause["severity_score"] <= 10.0

    def test_unsupported_type_415(self):
        r = client.post("/analyze", files={"file": ("d.xlsx", b"data", "application/vnd.ms-excel")})
        assert r.status_code == 415

    def test_empty_file_422(self):
        r = client.post("/analyze", files={"file": ("empty.pdf", b"", "application/pdf")})
        assert r.status_code == 422

    def test_agents_completed_has_four(self):
        data = self._post_docx().json()
        assert len(data["agents_completed"]) == 4

    def test_executive_summary_non_empty(self):
        summary = self._post_docx().json()["report"]["executive_summary"]
        assert len(summary) > 10

    def test_top_risks_present(self):
        top = self._post_docx().json()["report"]["top_risks"]
        assert isinstance(top, list) and len(top) > 0

    def test_pipeline_called_with_document_text(self):
        """Verify run_pipeline is actually called (not dummy JSON)."""
        mt = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        with patch("core.graph.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
            mock_pipeline.return_value = _make_fake_response()
            client.post("/analyze", files={"file": ("c.docx", make_docx("unique text"), mt)})
        mock_pipeline.assert_called_once()
        # Verify document_text was passed (not hardcoded)
        call_kwargs = mock_pipeline.call_args.kwargs
        assert "document_text" in call_kwargs
        assert len(call_kwargs["document_text"]) > 0

    def test_pipeline_receives_correct_filename(self):
        mt = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        with patch("core.graph.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
            mock_pipeline.return_value = _make_fake_response("mycontract.docx")
            client.post("/analyze", files={"file": ("mycontract.docx", make_docx(), mt)})
        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs["filename"] == "mycontract.docx"

    def test_pipeline_not_called_when_not_ready(self):
        """If pipeline_ready=False, 503 is returned without calling run_pipeline."""
        from main import app
        original = getattr(app.state, "pipeline_ready", True)
        app.state.pipeline_ready = False
        try:
            r = client.post("/analyze", files={
                "file": ("c.pdf", make_pdf(), "application/pdf")
            })
            assert r.status_code == 503
        finally:
            app.state.pipeline_ready = original
