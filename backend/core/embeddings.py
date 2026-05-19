"""
embeddings.py — In-memory TF-IDF benchmark similarity for LEXGUARD.

Replaces the previous sentence-transformers + ChromaDB approach.
Motivation: sentence-transformers requires torch (~500MB RAM), which exceeds
Railway's free-tier memory limit. TF-IDF via scikit-learn achieves the same
benchmark comparison goal with ~15MB RAM and zero network calls.

TF-IDF is adequate for comparing a clause against 21 hardcoded benchmark
clauses — the corpus is tiny and the differences between clause types are
large enough that TF-IDF similarity is a reliable signal.

No credentials, no file system, no model downloads required.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

from core.benchmarks import BENCHMARKS

logger = logging.getLogger(__name__)

DEFAULT_N_RESULTS: int = 3


class EmbeddingsStore:
    """
    In-memory benchmark similarity search using TF-IDF cosine similarity.

    Usage:
        store = EmbeddingsStore()
        store.ingest_benchmarks()               # fits vectorizer on corpus
        results = await store.find_similar(clause_text, n_results=3)
    """

    def __init__(self) -> None:
        self._benchmarks = list(BENCHMARKS)
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._benchmark_matrix = None  # sparse matrix after fit
        self._fitted = False

        logger.info(
            "EmbeddingsStore initialised",
            extra={"benchmark_count": len(self._benchmarks)},
        )

    def ingest_benchmarks(self) -> int:
        """
        Fit the TF-IDF vectorizer on the benchmark corpus.

        Returns:
            Number of benchmarks (always len(BENCHMARKS) on first call, 0 on repeat).
        """
        if self._fitted:
            logger.info("Benchmarks already fitted — skipping")
            return 0

        texts = [b["text"] for b in self._benchmarks]
        self._vectorizer = TfidfVectorizer(
            max_features=512,
            stop_words="english",
            ngram_range=(1, 2),  # unigrams + bigrams for better legal text matching
        )
        self._benchmark_matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True

        logger.info("Benchmarks fitted", extra={"count": len(texts)})
        return len(texts)

    def get_collection_count(self) -> int:
        return len(self._benchmarks)

    def _find_similar_sync(
        self,
        clause_text: str,
        clause_type: Optional[str] = None,
        n_results: int = DEFAULT_N_RESULTS,
    ) -> List[dict[str, Any]]:
        """
        Find the most similar benchmarks using TF-IDF cosine similarity.

        Args:
            clause_text:  The extracted clause text to compare.
            clause_type:  Optional filter — if set, only returns benchmarks of
                          this clause type. Falls back to cross-type search if
                          no matches found within the type.
            n_results:    Maximum number of results to return.

        Returns:
            List of dicts with benchmark metadata and distance score.
        """
        if not clause_text.strip():
            raise ValueError("Cannot search with empty clause text.")

        if not self._fitted or self._vectorizer is None:
            logger.warning("Vectorizer not fitted — call ingest_benchmarks() first")
            return []

        query_vec = self._vectorizer.transform([clause_text])
        similarities = cosine_similarity(query_vec, self._benchmark_matrix)[0]

        # Sort by descending similarity
        ranked_indices = similarities.argsort()[::-1]

        results: List[dict[str, Any]] = []
        for idx in ranked_indices:
            b = self._benchmarks[int(idx)]
            # If clause_type filter set, skip non-matching types (but don't starve)
            if clause_type and b["clause_type"] != clause_type and len(results) > 0:
                continue
            results.append({
                "benchmark_id": b["benchmark_id"],
                "text": b["text"],
                "clause_type": b["clause_type"],
                "is_predatory": b["is_predatory"],
                "risk_level": b["risk_level"],
                "severity_score": b["severity_score"],
                "notes": b["notes"],
                "distance": float(1.0 - similarities[int(idx)]),
            })
            if len(results) >= n_results:
                break

        return results

    async def find_similar(
        self,
        clause_text: str,
        clause_type: Optional[str] = None,
        n_results: int = DEFAULT_N_RESULTS,
    ) -> List[dict[str, Any]]:
        """Async wrapper — runs TF-IDF search in a thread to keep event loop free."""
        return await asyncio.to_thread(
            self._find_similar_sync, clause_text, clause_type, n_results
        )
