"""
ocr_handler.py — Google Cloud Vision OCR for scanned PDFs.

Design decisions:
  - Uses page-by-page rendering via PyMuPDF at 300 DPI (optimal for legal docs)
  - Timeout of 30s per page via google.api_core.retry — prevents hung requests
  - Async-safe: this is a sync function, call via asyncio.to_thread in routes
  - Returns empty string (not raises) when a single page fails — partial OCR
    is better than total failure

Common failure points:
  - GOOGLE_APPLICATION_CREDENTIALS not set → raises RuntimeError with clear message
  - Vision API quota exceeded → google.api_core.exceptions.ResourceExhausted
  - Single page OCR failure → logged and skipped, not fatal
  - Very large PDFs (>20 pages) → slow; consider async batch API for production

How to test:
  - Requires GOOGLE_APPLICATION_CREDENTIALS pointing to a service account key
  - pytest backend/tests/test_ocr.py -v (needs live API — skip in CI without creds)
"""
from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# Per-page OCR timeout in seconds — prevents hung Vision API calls
OCR_PAGE_TIMEOUT: int = 30
# DPI for page rendering — 300 DPI is optimal for legal text legibility
RENDER_DPI: int = 300


def ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from a scanned PDF using Google Cloud Vision API.

    Args:
        pdf_bytes: Raw bytes of the scanned PDF.

    Returns:
        Extracted text as a single string (pages joined by double newline).

    Raises:
        RuntimeError: If google-cloud-vision is not installed or credentials are missing.
        ValueError:   If pdf_bytes is empty.
    """
    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty — nothing to OCR.")

    # ── Validate credentials are configured before making API calls ──────────
    creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_env = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not creds_env and not project_env:
        raise RuntimeError(
            "Google Cloud credentials are not configured. "
            "Set GOOGLE_APPLICATION_CREDENTIALS (path to service account JSON) "
            "or GOOGLE_CLOUD_PROJECT (for Application Default Credentials)."
        )

    # ── Import guard — fails fast if package not installed ───────────────────
    try:
        from google.cloud import vision  # type: ignore[import-untyped]
        from google.api_core import retry as api_retry  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-vision is not installed. "
            "Add 'google-cloud-vision==3.7.3' to requirements.txt and reinstall."
        ) from exc

    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is not installed — required for PDF page rendering."
        ) from exc

    client = vision.ImageAnnotatorClient()
    page_texts: List[str] = []

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to open PDF for OCR: {exc}") from exc

    page_count = len(doc)
    logger.info("Starting OCR", extra={"page_count": page_count})

    for page_num in range(page_count):
        page = doc[page_num]

        try:
            # Render page to PNG image at 300 DPI
            scale = RENDER_DPI / 72.0
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
        except Exception as exc:
            logger.warning(
                "Failed to render PDF page for OCR — skipping",
                extra={"page": page_num + 1, "error": str(exc)},
            )
            continue

        try:
            image = vision.Image(content=img_bytes)
            # Use DOCUMENT_TEXT_DETECTION for structured document text
            response = client.document_text_detection(  # type: ignore[attr-defined]
                image=image,
                timeout=OCR_PAGE_TIMEOUT,
                retry=api_retry.Retry(deadline=OCR_PAGE_TIMEOUT),
            )
        except Exception as exc:
            logger.warning(
                "Vision API call failed for page — skipping",
                extra={"page": page_num + 1, "error": str(exc)},
            )
            continue

        if response.error.message:
            logger.warning(
                "Vision API returned error for page — skipping",
                extra={"page": page_num + 1, "error": response.error.message},
            )
            continue

        page_text = response.full_text_annotation.text
        if page_text:
            page_texts.append(page_text)
            logger.debug(
                "OCR page complete",
                extra={"page": page_num + 1, "chars": len(page_text)},
            )

    doc.close()

    result = "\n\n".join(page_texts)
    logger.info(
        "OCR complete",
        extra={"pages_processed": len(page_texts), "total_chars": len(result)},
    )
    return result
