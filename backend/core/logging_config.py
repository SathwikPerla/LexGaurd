"""
logging_config.py — Structured JSON logging for LEXGUARD backend.

Why structured JSON logging?
  - Cloud Run / GCP Cloud Logging parses JSON log lines automatically
  - Fields like request_id, filename, agent_name are queryable in Log Explorer
  - Replaces brittle string grep with structured filtering
  - One module so all loggers use the same format without repetition

Failure points:
  - Do NOT call setup_logging() more than once (causes duplicate handlers)
  - LEXGUARD_LOG_LEVEL env var must be a valid Python logging level name
"""
from __future__ import annotations

import logging
import os
import sys

from pythonjsonlogger import jsonlogger  # type: ignore[import-untyped]


def setup_logging() -> None:
    """
    Configure root logger to emit JSON-structured lines.
    Call this once at application startup in main.py.
    """
    log_level_name = os.getenv("LEXGUARD_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Suppress noisy library loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
