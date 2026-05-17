"""
embeddings.py — Local sentence-transformers embeddings + ChromaDB storage.

Design decisions:
  - SentenceTransformer('all-MiniLM-L6-v2') — 384-dim, ~90MB, fast CPU inference,
    no API key or network calls after first model download. Downloaded once to
    ~/.cache/huggingface/hub/ and cached permanently.
  - ChromaDB PersistentClient stores embeddings between restarts — benchmarks
    are ingested once, subsequent starts use the cached vectors.
  - No Google Cloud, Vertex AI, or any credentials required.
  - Model loaded lazily at first embed call — init is instant even without network.
  - Embedding calls are synchronous (sentence-transformers runs on CPU/MPS locally);
    wrapped in asyncio.to_thread where called from async routes.

Env vars:
  CHROMA_DB_PATH — directory for ChromaDB storage (default: ./chroma_db)

Common failure points:
  - First run downloads model (~90MB) — can fail on no-network environments.
    Subsequent runs use cached model regardless of network.
  - ChromaDB directory not writable → PermissionError at init.
  - Empty text passed to embed → raises ValueError before encoding.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, List

from core.benchmarks import BENCHMARKS

logger = logging.getLogger(__name__)

COLLECTION_NAME: str = "legal_benchmarks"
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
DEFAULT_N_RESULTS: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingsStore
# ─────────────────────────────────────────────────────────────────────────────


class EmbeddingsStore:
    """
    Local embedding store using sentence-transformers + ChromaDB.

    Usage:
        store = EmbeddingsStore()
        store.ingest_benchmarks()               # idempotent
        results = await store.find_similar(text, n_results=3)
    """

    def __init__(self, chroma_db_path: str | None = None) -> None:
        self._chroma_path = chroma_db_path or os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self._model = self._load_model()
        self._collection = self._init_chroma()

        logger.info(
            "EmbeddingsStore initialised",
            extra={"model": EMBEDDING_MODEL_NAME, "chroma_path": self._chroma_path},
        )

    def _load_model(self):  # type: ignore[return]
        """Load SentenceTransformer model — separated for easy mocking in tests."""
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        return SentenceTransformer(EMBEDDING_MODEL_NAME)

    def _init_chroma(self):  # type: ignore[return]
        """Create or open ChromaDB persistent collection."""
        import chromadb  # type: ignore[import-untyped]
        client = chromadb.PersistentClient(path=self._chroma_path)
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Embedding computation ─────────────────────────────────────────────────

    def _embed_texts_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using the local sentence-transformers model.

        Raises:
            ValueError: If any text is empty or whitespace-only.
        """
        if not texts:
            return []
        if any(not t.strip() for t in texts):
            raise ValueError("Cannot embed empty or whitespace-only text.")

        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def _embed_query_sync(self, text: str) -> List[float]:
        """Embed a single query string."""
        if not text.strip():
            raise ValueError("Cannot embed empty query text.")
        embedding = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return embedding[0].tolist()

    # ── Benchmark ingestion ───────────────────────────────────────────────────

    def ingest_benchmarks(self) -> int:
        """
        Load BENCHMARKS into ChromaDB. Idempotent — skips already-stored IDs.

        Returns:
            Number of benchmarks newly ingested (0 if all already present).
        """
        existing_ids: set[str] = set()
        try:
            existing = self._collection.get()
            existing_ids = set(existing.get("ids", []))
        except Exception as exc:
            logger.warning("Failed to fetch existing IDs", extra={"error": str(exc)})

        to_ingest = [b for b in BENCHMARKS if b["benchmark_id"] not in existing_ids]

        if not to_ingest:
            logger.info("All benchmarks already in ChromaDB — skipping")
            return 0

        texts = [b["text"] for b in to_ingest]
        embeddings = self._embed_texts_sync(texts)

        self._collection.add(
            ids=[b["benchmark_id"] for b in to_ingest],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "clause_type": b["clause_type"],
                    "is_predatory": str(b["is_predatory"]),
                    "risk_level": b["risk_level"],
                    "severity_score": str(b["severity_score"]),
                    "notes": b["notes"],
                }
                for b in to_ingest
            ],
        )

        logger.info("Benchmarks ingested", extra={"count": len(to_ingest)})
        return len(to_ingest)

    # ── Similarity search ─────────────────────────────────────────────────────

    def _find_similar_sync(
        self,
        clause_text: str,
        clause_type: str | None = None,
        n_results: int = DEFAULT_N_RESULTS,
    ) -> List[dict[str, Any]]:
        """Find most semantically similar benchmark clauses via ChromaDB."""
        query_embedding = self._embed_query_sync(clause_text)

        total = self._collection.count()
        if total == 0:
            logger.warning("ChromaDB collection is empty")
            return []

        clamped = min(n_results, total)
        where = {"clause_type": clause_type} if clause_type else None

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=clamped,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("ChromaDB query failed", extra={"error": str(exc)})
            return []

        output: List[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            output.append({
                "benchmark_id": ids[idx] if idx < len(ids) else "",
                "text": doc,
                "clause_type": meta.get("clause_type", ""),
                "is_predatory": meta.get("is_predatory", "False") == "True",
                "risk_level": meta.get("risk_level", "GREEN"),
                "severity_score": float(meta.get("severity_score", "2.0")),
                "notes": meta.get("notes", ""),
                "distance": dist,
            })

        return output

    async def find_similar(
        self,
        clause_text: str,
        clause_type: str | None = None,
        n_results: int = DEFAULT_N_RESULTS,
    ) -> List[dict[str, Any]]:
        """Async wrapper — runs embedding + ChromaDB query in a thread."""
        return await asyncio.to_thread(
            self._find_similar_sync, clause_text, clause_type, n_results
        )

    def get_collection_count(self) -> int:
        return self._collection.count()
