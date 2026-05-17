"""
graph.py — LEXGUARD 4-agent sequential pipeline.

Runs all four agents in order: Extractor → Risk Analyzer → Reasoner → Negotiator.
Each agent's output feeds directly into the next.

LangGraph integration is planned for Step 3 (streaming support). This module
provides the same interface a LangGraph graph would expose, so main.py doesn't
need to change when streaming is added.

Env vars required (validated inside LLMClient / EmbeddingsStore):
  ANTHROPIC_API_KEY
"""
from __future__ import annotations

import logging
import time

from agents.extractor import ExtractorAgent
from agents.negotiator import NegotiatorAgent
from agents.reasoner import ReasonerAgent
from agents.risk_analyzer import RiskAnalyzerAgent
from core.embeddings import EmbeddingsStore
from core.gemini_client import LLMClient
from models.schemas import AnalyzeResponse

logger = logging.getLogger(__name__)


async def run_pipeline(
    document_text: str,
    filename: str,
    parse_method: str,
    llm_client: LLMClient,
    embeddings_store: EmbeddingsStore | None = None,
) -> AnalyzeResponse:
    """
    Run the full 4-agent pipeline on extracted document text.

    Args:
        document_text:   Clean text extracted from the uploaded document.
        filename:        Original filename for the response metadata.
        parse_method:    How the document was parsed ('pdf', 'docx', 'ocr').
        llm_client:      Shared LLMClient instance (initialised at app startup).
        embeddings_store: Shared EmbeddingsStore for benchmark comparison.
                          Pass None to skip benchmark enrichment.

    Returns:
        AnalyzeResponse — complete 4-agent risk intelligence report.

    Raises:
        Any exception from the agents propagates up — caught by the FastAPI
        exception handler in main.py which returns a clean 500 to the client.
    """
    start = time.monotonic()
    logger.info(
        "Pipeline starting",
        extra={"doc_filename": filename, "doc_chars": len(document_text)},
    )

    # Agent 1 — Extract and label clauses
    agent1 = ExtractorAgent(llm_client)
    extractor_output = await agent1.run(document_text)
    logger.info(
        "Agent 1 done",
        extra={"clauses": extractor_output.total_clauses, "doc_type": extractor_output.document_type},
    )

    # Agent 2 — Score and classify risk
    agent2 = RiskAnalyzerAgent(llm_client, embeddings_store)
    risk_output = await agent2.run(extractor_output)
    logger.info(
        "Agent 2 done",
        extra={"overall_score": risk_output.overall_score, "red": risk_output.red_count},
    )

    # Agent 3 — Plain-language explanations + scenario simulation
    agent3 = ReasonerAgent(llm_client)
    reasoner_output = await agent3.run(risk_output)
    logger.info("Agent 3 done", extra={"clauses": len(reasoner_output.clauses)})

    # Agent 4 — Negotiation advice + final report
    agent4 = NegotiatorAgent(llm_client)
    final_report = await agent4.run(reasoner_output)
    logger.info(
        "Agent 4 done",
        extra={"red": final_report.red_count, "yellow": final_report.yellow_count, "green": final_report.green_count},
    )

    elapsed = time.monotonic() - start
    logger.info(
        "Pipeline complete",
        extra={
            "doc_filename": filename,
            "elapsed_s": round(elapsed, 1),
            "overall_score": final_report.overall_score,
            "total_clauses": len(final_report.clauses),
        },
    )

    return AnalyzeResponse(
        filename=filename,
        parse_method=parse_method,
        report=final_report,
        agents_completed=["extractor", "risk_analyzer", "reasoner", "negotiator"],
    )
