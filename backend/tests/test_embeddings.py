"""
test_embeddings.py — Unit tests for EmbeddingsStore (TF-IDF, in-memory).

No external deps, no mocking of models — TF-IDF runs locally in tests.

Run: pytest backend/tests/test_embeddings.py -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.benchmarks import BENCHMARKS
from core.embeddings import EmbeddingsStore


# ─────────────────────────────────────────────────────────────────────────────
# Init tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEmbeddingsStoreInit:
    def test_init_succeeds(self):
        store = EmbeddingsStore()
        assert store is not None

    def test_not_fitted_on_init(self):
        store = EmbeddingsStore()
        assert not store._fitted

    def test_benchmark_count_correct(self):
        store = EmbeddingsStore()
        assert store.get_collection_count() == len(BENCHMARKS)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark ingestion tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBenchmarkIngestion:
    def test_ingest_returns_count(self):
        store = EmbeddingsStore()
        count = store.ingest_benchmarks()
        assert count == len(BENCHMARKS)

    def test_ingest_is_idempotent(self):
        store = EmbeddingsStore()
        first = store.ingest_benchmarks()
        second = store.ingest_benchmarks()
        assert first == len(BENCHMARKS)
        assert second == 0  # already fitted

    def test_fitted_after_ingest(self):
        store = EmbeddingsStore()
        store.ingest_benchmarks()
        assert store._fitted
        assert store._vectorizer is not None
        assert store._benchmark_matrix is not None


# ─────────────────────────────────────────────────────────────────────────────
# find_similar tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFindSimilar:
    def _fitted_store(self) -> EmbeddingsStore:
        store = EmbeddingsStore()
        store.ingest_benchmarks()
        return store

    def test_returns_list(self):
        store = self._fitted_store()
        results = asyncio.get_event_loop().run_until_complete(
            store.find_similar("non-compete clause for 2 years", n_results=3)
        )
        assert isinstance(results, list)

    def test_n_results_respected(self):
        store = self._fitted_store()
        results = asyncio.get_event_loop().run_until_complete(
            store.find_similar("arbitration waives jury trial", n_results=2)
        )
        assert len(results) <= 2

    def test_result_has_required_keys(self):
        store = self._fitted_store()
        results = asyncio.get_event_loop().run_until_complete(
            store.find_similar("employee shall not compete", n_results=1)
        )
        assert len(results) >= 1
        keys = results[0].keys()
        assert "text" in keys
        assert "is_predatory" in keys
        assert "risk_level" in keys
        assert "severity_score" in keys
        assert "distance" in keys

    def test_distance_between_0_and_1(self):
        store = self._fitted_store()
        results = asyncio.get_event_loop().run_until_complete(
            store.find_similar("inventions belong to employer", n_results=3)
        )
        for r in results:
            assert 0.0 <= r["distance"] <= 1.0

    def test_not_fitted_returns_empty(self):
        store = EmbeddingsStore()  # not fitted
        results = asyncio.get_event_loop().run_until_complete(
            store.find_similar("some clause text", n_results=3)
        )
        assert results == []

    def test_non_compete_query_finds_non_compete_benchmark(self):
        """Semantic relevance test — non-compete query should surface non-compete benchmarks."""
        store = self._fitted_store()
        results = asyncio.get_event_loop().run_until_complete(
            store.find_similar(
                "Employee shall not work for any competing company for 18 months",
                clause_type="non_compete",
                n_results=2,
            )
        )
        types = [r["clause_type"] for r in results]
        assert "non_compete" in types

    def test_ip_query_finds_ip_benchmark(self):
        store = self._fitted_store()
        results = asyncio.get_event_loop().run_until_complete(
            store.find_similar(
                "All inventions and discoveries made during employment belong to company",
                clause_type="ip_transfer",
                n_results=2,
            )
        )
        types = [r["clause_type"] for r in results]
        assert "ip_transfer" in types


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────


class TestInputValidation:
    def test_empty_clause_text_raises(self):
        store = EmbeddingsStore()
        store.ingest_benchmarks()
        with pytest.raises(ValueError, match="empty"):
            store._find_similar_sync("")

    def test_whitespace_only_raises(self):
        store = EmbeddingsStore()
        store.ingest_benchmarks()
        with pytest.raises(ValueError, match="empty"):
            store._find_similar_sync("   ")
