"""
conftest.py — shared test fixtures.

Keeps the optional Gemini fallback OFF for every test unless a test turns it on
explicitly. main.py calls load_dotenv(), so a real GEMINI_API_KEY in backend/.env
would otherwise leak into the test process and make failing-Claude tests reach
the real network.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_gemini_fallback_by_default(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_NAME", raising=False)
    monkeypatch.delenv("GEMINI_REASONING_EFFORT", raising=False)
    for name in ("FALLBACK_API_KEY", "FALLBACK_BASE_URL", "FALLBACK_MODEL_NAME", "GROQ_API_KEY", "GROQ_MODEL_NAME", "GROQ_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)
