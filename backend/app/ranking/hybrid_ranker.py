"""
hybrid_ranker.py
----------------
Fuses semantic (FAISS cosine), keyword (BM25), and symbol-name match scores
into a single ranked list of SearchResult objects.

Fusion formula
~~~~~~~~~~~~~~
    fused = w_sem * sem_score
          + w_bm25 * bm25_score
          + w_sym  * symbol_score

All three component scores are in [0, 1] before fusion, so the fused score
is also in [0, 1] when the weights sum to 1.

Symbol-name scoring
~~~~~~~~~~~~~~~~~~~
Three tiers (highest-to-lowest):
  1. Exact match (case-insensitive)     → 1.0
  2. Query is a substring of the name   → 0.6
  3. Name is a substring of the query   → 0.4
  4. No match                           → 0.0

Public API
~~~~~~~~~~
    HybridRanker(semantic_weight, bm25_weight, symbol_weight)
    .rank(query, vector_store, keyword_index, top_k, language_filter,
          embedder) → SearchResponse
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from backend.app.models.schemas import (
    CodeChunk,
    Language,
    SearchResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)

# Default weights — overridden by config / caller.
_DEFAULT_SEM_W = 0.6
_DEFAULT_BM25_W = 0.3
_DEFAULT_SYM_W = 0.1


class HybridRanker:
    """
    Combines semantic, keyword, and symbol-name signals into one ranked list.

    Parameters
    ----------
    semantic_weight:
        Weight applied to the cosine similarity score (FAISS).
    bm25_weight:
        Weight applied to the normalised BM25 score.
    symbol_weight:
        Weight applied to the symbol-name match bonus.
    """

    def __init__(
        self,
        semantic_weight: float = _DEFAULT_SEM_W,
        bm25_weight: float = _DEFAULT_BM25_W,
        symbol_weight: float = _DEFAULT_SYM_W,
    ) -> None:
        total = semantic_weight + bm25_weight + symbol_weight
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                "Ranking weights sum to %.4f (expected 1.0) — scores will be "
                "outside [0, 1]. Consider normalising.",
                total,
            )
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.symbol_weight = symbol_weight

    # ── Main entry-point ──────────────────────────────────────────────────────

    def rank(
        self,
        query: str,
        vector_store,
        keyword_index,
        *,
        top_k: int = 10,
        language_filter: Optional[Language] = None,
        repo_filter: Optional[str] = None,
        embedder=None,
    ) -> SearchResponse:
        """
        Execute a hybrid search and return a SearchResponse.

        Parameters
        ----------
        query:
            Natural-language or symbol-name search string.
        vector_store:
            A ``VectorStore`` instance (already loaded / populated).
        keyword_index:
            A ``KeywordIndex`` instance (already loaded / populated).
        top_k:
            Number of results to return.
        language_filter:
            Restrict results to a specific programming language.
        repo_filter:
            When provided, only chunks from this repo_name are returned.
        embedder:
            ``Embedder`` instance used to encode the query.  If None, one is
            created via ``get_embedder()`` using the default model.

        Returns
        -------
        SearchResponse
        """
        # ── 1. Embed the query ────────────────────────────────────────────────
        if embedder is None:
            from backend.app.embeddings.embedder import get_embedder
            embedder = get_embedder()

        query_vector = embedder.encode_one(query)

        # ── 2. Retrieve candidates from both indexes ──────────────────────────
        candidate_k = min(top_k * 10, max(vector_store.size, keyword_index.size, 1))

        sem_hits: list[tuple[CodeChunk, float]] = vector_store.search(
            query_vector,
            top_k=candidate_k,
            language_filter=language_filter,
            repo_filter=repo_filter,
        )
        bm25_hits: list[tuple[CodeChunk, float]] = keyword_index.search(
            query,
            top_k=candidate_k,
            language_filter=language_filter,
            repo_filter=repo_filter,
        )

        # ── 3. Build per-chunk score maps (keyed by chunk_id) ─────────────────
        sem_scores: dict[str, float] = {
            str(c.chunk_id): s for c, s in sem_hits
        }
        bm25_scores: dict[str, float] = {
            str(c.chunk_id): s for c, s in bm25_hits
        }

        # Union of all candidate chunks (deduplicated by chunk_id)
        all_chunks: dict[str, CodeChunk] = {}
        for chunk, _ in sem_hits:
            all_chunks[str(chunk.chunk_id)] = chunk
        for chunk, _ in bm25_hits:
            all_chunks[str(chunk.chunk_id)] = chunk

        # ── 4. Compute fused scores ───────────────────────────────────────────
        results: list[SearchResult] = []
        for cid, chunk in all_chunks.items():
            sem_s = sem_scores.get(cid, 0.0)
            bm25_s = bm25_scores.get(cid, 0.0)
            sym_s = _symbol_score(query, chunk.symbol_name)

            fused = (
                self.semantic_weight * sem_s
                + self.bm25_weight * bm25_s
                + self.symbol_weight * sym_s
            )
            fused = float(min(max(fused, 0.0), 1.0))

            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    file_path=chunk.file_path,
                    language=chunk.language,
                    symbol_name=chunk.symbol_name,
                    chunk_type=chunk.chunk_type,
                    code=chunk.code,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    repo_name=chunk.repo_name,
                    score=fused,
                    semantic_score=sem_s,
                    bm25_score=bm25_s,
                    symbol_score=sym_s,
                )
            )

        # ── 5. Sort and truncate ──────────────────────────────────────────────
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_k]

        return SearchResponse(
            query=query,
            top_k=top_k,
            total_indexed=vector_store.size,
            results=results,
        )


# ── Symbol-name scoring ───────────────────────────────────────────────────────

def _symbol_score(query: str, symbol_name: Optional[str]) -> float:
    """
    Return a [0, 1] score representing how well ``query`` matches
    ``symbol_name``.

    Tiers
    -----
    1.0  — exact match (case-insensitive)
    0.6  — query is contained within symbol_name  (e.g. "parse" in "parseJSON")
    0.4  — symbol_name is contained within query   (e.g. "JSON" in "parse JSON response")
    0.0  — no match
    """
    if not symbol_name:
        return 0.0

    q = query.strip().lower()
    sym = symbol_name.strip().lower()

    if not q or not sym:
        return 0.0

    if q == sym:
        return 1.0
    if q in sym:
        return 0.6
    if sym in q:
        return 0.4
    return 0.0


# ── Module-level singleton ────────────────────────────────────────────────────

def get_ranker(
    semantic_weight: float = _DEFAULT_SEM_W,
    bm25_weight: float = _DEFAULT_BM25_W,
    symbol_weight: float = _DEFAULT_SYM_W,
) -> HybridRanker:
    """Return a HybridRanker configured with the given weights."""
    return HybridRanker(
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
        symbol_weight=symbol_weight,
    )
