"""
main.py — FastAPI application for LEXGUARD backend.

Constraints enforced:
  - All routes are async
  - Explicit status codes on every route
  - Structured logging with request_id on every request
  - File validation: 15 MB limit, .pdf/.docx only, non-empty
  - Env vars validated at startup via LLMClient / EmbeddingsStore
  - Global exception handler prevents stack traces leaking to clients
  - LLMClient and EmbeddingsStore initialised once at startup (not per-request)
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, List

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse

from core.history import (
    delete_analysis,
    get_analysis,
    get_chat_context,
    init_db,
    list_analyses,
    save_analysis,
)
from core.logging_config import setup_logging
from core.document_parser import parse_document
from core.ocr_handler import ocr_pdf_bytes
from models.schemas import (
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ParsedDocumentResponse,
)

# Load .env before any env var access
load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "15"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".doc"})


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle — pipeline components initialised once at startup
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("LEXGUARD backend starting", extra={"max_file_mb": MAX_FILE_SIZE_MB})

    # Initialise SQLite history database (creates file + table if needed)
    init_db()

    # Import here to avoid circular import issues
    from core.gemini_client import LLMClient, LLMConfigurationError
    from core.embeddings import EmbeddingsStore

    try:
        # LLMClient validates ANTHROPIC_API_KEY — raises LLMConfigurationError if missing
        app.state.llm_client = LLMClient()

        # Agent 2 (Risk Analyzer) runs on a stronger model — it produces the core
        # risk judgments (severity scores, RED/YELLOW/GREEN, predatory flag, and the
        # overall_score that flows through Agents 3 & 4 unchanged). Defaults to Opus;
        # override or align with the base model via RISK_ANALYZER_MODEL_NAME.
        risk_model = os.getenv("RISK_ANALYZER_MODEL_NAME", "claude-opus-4-8")
        app.state.risk_llm_client = LLMClient(model_name=risk_model)

        # EmbeddingsStore fits TF-IDF on 21 benchmark clauses (instant, in-memory)
        app.state.embeddings_store = EmbeddingsStore()
        app.state.embeddings_store.ingest_benchmarks()

        app.state.pipeline_ready = True
        logger.info(
            "Pipeline ready",
            extra={
                "model": app.state.llm_client._model_name,
                "risk_model": app.state.risk_llm_client._model_name,
            },
        )
    except LLMConfigurationError as exc:
        logger.error("Pipeline init failed — ANTHROPIC_API_KEY missing", extra={"error": str(exc)})
        app.state.pipeline_ready = False
        app.state.llm_client = None
        app.state.risk_llm_client = None
        app.state.embeddings_store = None
    except Exception as exc:
        logger.error("Pipeline init failed", extra={"error": str(exc)}, exc_info=True)
        app.state.pipeline_ready = False
        app.state.llm_client = None
        app.state.risk_llm_client = None
        app.state.embeddings_store = None

    yield
    logger.info("LEXGUARD backend shutting down")


app = FastAPI(
    title="LEXGUARD — AI Contract Intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Open CORS — this API is public, no auth cookies used.
    # allow_credentials=False is required when allow_origins=["*"] per CORS spec.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request-ID middleware
# ─────────────────────────────────────────────────────────────────────────────


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _validate_file_type(file: UploadFile) -> None:
    import pathlib
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{ext}' is not supported. Upload a .pdf or .docx file.",
        )


async def _read_and_validate(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {len(data) / 1_048_576:.1f} MB exceeds the "
                f"{MAX_FILE_SIZE_MB} MB limit."
            ),
        )
    return data


async def _extract_text(file_bytes: bytes, filename: str) -> dict:
    try:
        result = await asyncio.to_thread(parse_document, file_bytes, filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    if result["is_scanned_pdf"] and len(result["extracted_text"]) < 100:
        logger.info("Scanned PDF detected — attempting OCR", extra={"doc_filename": filename})
        try:
            ocr_text = await asyncio.to_thread(ocr_pdf_bytes, file_bytes)
            result["extracted_text"] = ocr_text
            result["parse_method"] = "ocr"
        except RuntimeError as exc:
            logger.warning("OCR unavailable", extra={"reason": str(exc)})
            result["extracted_text"] = (
                "[Note: This appears to be a scanned document. "
                "OCR is not configured — text extraction is incomplete.]"
            )

    if len(result["extracted_text"].strip()) < 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Could not extract readable text from this document. "
                "The file may be password-protected, corrupted, or image-only without OCR."
            ),
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Infrastructure"],
    summary="Service health check",
)
async def health_check(request: Request) -> HealthResponse:
    pipeline_ready = getattr(request.app.state, "pipeline_ready", False)
    return HealthResponse(
        status="ok" if pipeline_ready else "degraded",
        version="1.0.0",
        service="lexguard-backend",
    )


@app.post(
    "/parse",
    response_model=ParsedDocumentResponse,
    status_code=status.HTTP_200_OK,
    tags=["Document"],
    summary="Parse document and return extracted text (debug)",
)
async def parse_endpoint(
    request: Request,
    file: UploadFile = File(...),
) -> ParsedDocumentResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info("Parse request", extra={"request_id": request_id, "doc_filename": file.filename})

    _validate_file_type(file)
    file_bytes = await _read_and_validate(file)
    result = await _extract_text(file_bytes, file.filename or "document")

    return ParsedDocumentResponse(
        filename=file.filename or "document",
        extracted_text=result["extracted_text"],
        character_count=len(result["extracted_text"]),
        page_count=result.get("page_count"),
        parse_method=result["parse_method"],
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Full 4-agent contract risk analysis",
)
async def analyze_endpoint(
    request: Request,
    file: UploadFile = File(...),
) -> AnalyzeResponse:
    """
    Upload a PDF or DOCX contract and receive a complete risk intelligence report.

    Runs through 4 AI agents: Extractor → Risk Analyzer → Reasoner → Negotiator.
    Every document produces unique output based on its actual content.

    ⚠️ DISCLAIMER: This is not legal advice.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info("Analyze request", extra={"request_id": request_id, "doc_filename": file.filename})

    if not getattr(request.app.state, "pipeline_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "AI pipeline is not ready. ANTHROPIC_API_KEY may be missing from .env. "
                "Check server logs for details."
            ),
        )

    _validate_file_type(file)
    file_bytes = await _read_and_validate(file)
    result = await _extract_text(file_bytes, file.filename or "document")

    logger.info(
        "Document parsed — running 4-agent pipeline",
        extra={
            "request_id": request_id,
            "parse_method": result["parse_method"],
            "char_count": len(result["extracted_text"]),
        },
    )

    # Import here to allow tests to patch core.graph.run_pipeline cleanly
    from core.graph import run_pipeline
    from core.gemini_client import LLMCallError, LLMParseError

    try:
        response = await run_pipeline(
            document_text=result["extracted_text"],
            filename=file.filename or "document",
            parse_method=result["parse_method"],
            llm_client=request.app.state.llm_client,
            embeddings_store=request.app.state.embeddings_store,
            risk_llm_client=getattr(request.app.state, "risk_llm_client", None),
        )
    except LLMParseError as exc:
        logger.error("LLM returned invalid JSON", extra={"request_id": request_id, "error": str(exc)})
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"AI response parsing failed: {str(exc)[:300]}")
    except LLMCallError as exc:
        logger.error("LLM API call failed", extra={"request_id": request_id, "error": str(exc)})
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"AI API call failed: {str(exc)[:300]}")
    except asyncio.TimeoutError:
        logger.error("Pipeline timed out", extra={"request_id": request_id})
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                            detail="Analysis timed out after 3 minutes. Try a shorter document.")
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        logger.error("Pipeline error", extra={"request_id": request_id, "error": str(exc), "traceback": tb})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Pipeline error: {type(exc).__name__}: {str(exc)[:300]}")

    logger.info(
        "Analysis complete",
        extra={
            "request_id": request_id,
            "overall_score": response.report.overall_score,
            "red": response.report.red_count,
            "yellow": response.report.yellow_count,
            "green": response.report.green_count,
            "clauses": len(response.report.clauses),
        },
    )

    # Persist to history — fire-and-forget, never block the response.
    # We generate the record id up front and stamp it onto the response so the
    # client can chat about this analysis later; the document text is stored so
    # the chat endpoint can rebuild context server-side.
    client_id = request.headers.get("X-Client-ID", "anonymous")
    record_id = str(uuid.uuid4())
    response.analysis_id = record_id
    try:
        await asyncio.to_thread(
            save_analysis,
            client_id,
            response.model_dump(),
            result["extracted_text"],
            record_id,
        )
        logger.info("History saved", extra={"record_id": record_id})
    except Exception as exc:
        logger.warning("History save failed", extra={"error": str(exc)})

    return response


# ─────────────────────────────────────────────────────────────────────────────
# History endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get(
    "/history",
    status_code=status.HTTP_200_OK,
    tags=["History"],
    summary="List past analyses for this browser session",
)
async def list_history(request: Request) -> List[Any]:
    client_id = request.headers.get("X-Client-ID", "anonymous")
    return await asyncio.to_thread(list_analyses, client_id)


@app.get(
    "/history/{record_id}",
    status_code=status.HTTP_200_OK,
    tags=["History"],
    summary="Get a full past analysis by ID",
)
async def get_history_item(record_id: str, request: Request) -> Any:
    client_id = request.headers.get("X-Client-ID", "anonymous")
    data = await asyncio.to_thread(get_analysis, record_id, client_id)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found.")
    return data


@app.delete(
    "/history/{record_id}",
    status_code=status.HTTP_200_OK,
    tags=["History"],
    summary="Delete a past analysis",
)
async def delete_history_item(record_id: str, request: Request) -> Any:
    client_id = request.headers.get("X-Client-ID", "anonymous")
    deleted = await asyncio.to_thread(delete_analysis, record_id, client_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found.")
    return {"deleted": record_id}


# ─────────────────────────────────────────────────────────────────────────────
# Chat — context-aware assistant over an analyzed document
# ─────────────────────────────────────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are LEXGUARD's contract assistant. You are the USER'S advocate — never the counterparty's.

You are given, as context, the FULL text of a document the user is considering signing, plus LEXGUARD's clause-by-clause risk analysis of it. Answer the user's questions about this specific document.

CORE PRINCIPLES:
- Ground every answer in the actual document text and the analysis provided. Quote or cite the specific clause you are referring to. If something is not addressed in the document, say so plainly — never invent terms.
- When a clause is costly, weigh the cost across ALL dimensions, not just money: career mobility, personal time, work-life balance, autonomy/control, reputation, legal exposure, and mental/emotional load. Name the specific burden.
- When the user wants a solution or to push back, give concrete, actionable negotiation strategy: what to ask for, why it is reasonable (benchmark/standard practice), and suggested alternative wording they could propose.
- Be honest about severity. Distinguish "predatory / unusual" from "standard / acceptable." Do not alarm the user about boilerplate, and do not soften genuinely dangerous terms.
- Be concise and plain-spoken. Use short paragraphs or bullet points. Avoid legalese; explain any necessary term.

You are an informational tool, not a lawyer. If the user faces a high-stakes decision, remind them to consult a licensed attorney — but still give your best substantive analysis first."""

_MAX_CHAT_HISTORY_TURNS = 20  # cap forwarded turns to bound context growth


def _build_chat_context(document_text: str, report: dict[str, Any], filename: str) -> str:
    """Assemble the cached context prefix: full document + analysis report."""
    import json as _json
    return (
        f"=== DOCUMENT: {filename} ===\n"
        f"{document_text}\n\n"
        "=== LEXGUARD ANALYSIS REPORT (JSON) ===\n"
        f"{_json.dumps(report, ensure_ascii=False)}\n"
        "=== END CONTEXT ==="
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Chat with an AI assistant about an analyzed document",
)
async def chat_endpoint(request: Request, body: ChatRequest) -> ChatResponse:
    """
    Ask questions about a previously analyzed document. The assistant sees the
    full document text plus LEXGUARD's analysis and answers as the user's advocate.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    if not getattr(request.app.state, "pipeline_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI pipeline is not ready. ANTHROPIC_API_KEY may be missing from .env.",
        )

    message = (body.message or "").strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message cannot be empty.",
        )

    client_id = request.headers.get("X-Client-ID", "anonymous")
    ctx = await asyncio.to_thread(get_chat_context, body.analysis_id, client_id)
    if ctx is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found for this session.",
        )
    if not ctx["document_text"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This analysis predates chat support. Re-upload the document to chat about it.",
        )

    context_block = _build_chat_context(ctx["document_text"], ctx["report"], ctx["filename"])

    # Build the conversation: prior history (capped) + the new user message.
    messages: list[dict[str, str]] = [
        {"role": m.role, "content": m.content}
        for m in body.history[-_MAX_CHAT_HISTORY_TURNS:]
        if m.role in ("user", "assistant") and m.content.strip()
    ]
    messages.append({"role": "user", "content": message})

    from core.gemini_client import LLMCallError

    logger.info("Chat request", extra={"request_id": request_id, "analysis_id": body.analysis_id,
                                       "turns": len(messages)})
    try:
        reply = await request.app.state.llm_client.generate_chat(
            CHAT_SYSTEM_PROMPT, context_block, messages
        )
    except LLMCallError as exc:
        logger.error("Chat LLM call failed", extra={"request_id": request_id, "error": str(exc)})
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"AI chat call failed: {str(exc)[:300]}")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                            detail="Chat timed out. Try a shorter question.")

    return ChatResponse(reply=reply)


# ─────────────────────────────────────────────────────────────────────────────
# Global exception handler
# ─────────────────────────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "Unhandled exception",
        extra={"request_id": request_id, "error": str(exc)},
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred.", "request_id": request_id},
    )
