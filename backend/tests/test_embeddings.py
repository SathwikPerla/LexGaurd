"""
test_embeddings.py — Unit tests for EmbeddingsStore (sentence-transformers + ChromaDB).

No credentials needed — sentence-transformers runs locally.
Model loading is mocked so tests run instantly without downloading 90MB.

Run: pytest backend/tests/test_embeddings.py -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.benchmarks import BENCHMARKS
from core.embeddings import EmbeddingsStore


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DIM = 8  # fake embedding dimension (real model uses 384)


def _fake_embedding(n: int = 1) -> list:
    return [[0.1] * DIM] * n


def _make_store(monkeypatch, tmp_path_factory) -> EmbeddingsStore:
    """Build an EmbeddingsStore with the model loading patched."""
    chroma_path = str(tmp_path_factory.mktemp("chroma"))
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1] * DIM] * max(len(BENCHMARKS), 1))
    with patch("core.embeddings.EmbeddingsStore._load_model", return_value=mock_model):
        store = EmbeddingsStore(chroma_db_path=chroma_path)
    return store


# ─────────────────────────────────────────────────────────────────────────────
# Init tests — no credentials needed (local model)
# ─────────────────────────────────────────────────────────────────────────────


class TestEmbeddingsStoreInit:
    def test_init_succeeds_without_credentials(self, tmp_path_factory):
        """No Google credentials required — model is local."""
        chroma = str(tmp_path_factory.mktemp("chroma"))
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * DIM])
        with patch("core.embeddings.EmbeddingsStore._load_model", return_value=mock_model):
            store = EmbeddingsStore(chroma_db_path=chroma)
        assert store is not None

    def test_chroma_path_from_env(self, monkeypatch, tmp_path_factory):
        chroma = str(tmp_path_factory.mktemp("chroma"))
        monkeypatch.setenv("CHROMA_DB_PATH", chroma)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * DIM])
        with patch("core.embeddings.EmbeddingsStore._load_model", return_value=mock_model):
            store = EmbeddingsStore()
        assert store._chroma_path == chroma

    def test_explicit_path_overrides_env(self, monkeypatch, tmp_path_factory):
        monkeypatch.setenv("CHROMA_DB_PATH", "/env/path")
        chroma = str(tmp_path_factory.mktemp("chroma"))
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * DIM])
        with patch("core.embeddings.EmbeddingsStore._load_model", return_value=mock_model):
            store = EmbeddingsStore(chroma_db_path=chroma)
        assert store._chroma_path == chroma


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark ingestion tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBenchmarkIngestion:
    def test_ingest_returns_count(self, monkeypatch, tmp_path_factory):
        store = _make_store(monkeypatch, tmp_path_factory)
        # Patch _embed_texts_sync to return correct size list
        with patch.object(
            store, "_embed_texts_sync",
            return_value=[[0.1] * DIM for _ in range(len(BENCHMARKS))],
        ):
            count = store.ingest_benchmarks()
        assert count == len(BENCHMARKS)

    def test_ingest_is_idempotent(self, monkeypatch, tmp_path_factory):
        store = _make_store(monkeypatch, tmp_path_factory)
        with patch.object(
            store, "_embed_texts_sync",
            return_value=[[0.1] * DIM for _ in range(len(BENCHMARKS))],
        ):
            first = store.ingest_benchmarks()
            second = store.ingest_benchmarks()
        assert first == len(BENCHMARKS)
        assert second == 0

    def test_collection_count_after_ingest(self, monkeypatch, tmp_path_factory):
        store = _make_store(monkeypatch, tmp_path_factory)
        with patch.object(
            store, "_embed_texts_sync",
            return_value=[[0.1] * DIM for _ in range(len(BENCHMARKS))],
        ):
            store.ingest_benchmarks()
        assert store.get_collection_count() == len(BENCHMARKS)


# ─────────────────────────────────────────────────────────────────────────────
# find_similar tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFindSimilar:
    def _populated_store(self, monkeypatch, tmp_path_factory) -> EmbeddingsStore:
        store = _make_store(monkeypatch, tmp_path_factory)
        with patch.object(
            store, "_embed_texts_sync",
            return_value=[[0.1] * DIM for _ in range(len(BENCHMARKS))],
        ):
            store.ingest_benchmarks()
        return store

    def test_returns_list(self, monkeypatch, tmp_path_factory):
        store = self._populated_store(monkeypatch, tmp_path_factory)
        with patch.object(store, "_embed_query_sync", return_value=[0.1] * DIM):
            results = asyncio.get_event_loop().run_until_complete(
                store.find_similar("non-compete clause for 2 years", n_results=3)
            )
        assert isinstance(results, list)

    def test_n_results_respected(self, monkeypatch, tmp_path_factory):
        store = self._populated_store(monkeypatch, tmp_path_factory)
        with patch.object(store, "_embed_query_sync", return_value=[0.1] * DIM):
            results = asyncio.get_event_loop().run_until_complete(
                store.find_similar("clause text", n_results=2)
            )
        assert len(results) <= 2

    def test_result_has_required_keys(self, monkeypatch, tmp_path_factory):
        store = self._populated_store(monkeypatch, tmp_path_factory)
        with patch.object(store, "_embed_query_sync", return_value=[0.1] * DIM):
            results = asyncio.get_event_loop().run_until_complete(
                store.find_similar("clause text", n_results=1)
            )
        if results:
            assert "text" in results[0]
            assert "is_predatory" in results[0]
            assert "risk_level" in results[0]
            assert "severity_score" in results[0]

    def test_empty_collection_returns_empty(self, monkeypatch, tmp_path_factory):
        store = _make_store(monkeypatch, tmp_path_factory)  # not ingested
        with patch.object(store, "_embed_query_sync", return_value=[0.1] * DIM):
            results = asyncio.get_event_loop().run_until_complete(
                store.find_similar("clause text", n_results=3)
            )
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────


class TestInputValidation:
    def test_empty_text_raises(self, monkeypatch, tmp_path_factory):
        store = _make_store(monkeypatch, tmp_path_factory)
        with pytest.raises(ValueError, match="empty"):
            store._embed_texts_sync([""])

    def test_whitespace_only_raises(self, monkeypatch, tmp_path_factory):
        store = _make_store(monkeypatch, tmp_path_factory)
        with pytest.raises(ValueError, match="empty"):
            store._embed_texts_sync(["   "])

    def test_empty_list_returns_empty(self, monkeypatch, tmp_path_factory):
        store = _make_store(monkeypatch, tmp_path_factory)
        result = store._embed_texts_sync([])
        assert result == []

    def test_empty_query_raises(self, monkeypatch, tmp_path_factory):
        store = _make_store(monkeypatch, tmp_path_factory)
        with pytest.raises(ValueError, match="empty"):
            store._embed_query_sync("")
