"""
routes_search.py
----------------
GET /api/search

Query parameters
~~~~~~~~~~~~~~~~
    q          (str, required)   — search query (natural language or symbol name)
    top_k      (int, default 10) — number of results to return (1–100)
    language   (str, optional)   — restrict to a specific language enum value
    repo_name  (str, optional)   — restrict to a specific indexed repository
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, status

from backend.app.config import settings
from backend.app.models.schemas import Language, SearchResponse
from backend.app.ranking.hybrid_ranker import get_ranker

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_vector_store(request: Request):
    return request.app.state.vector_store


def _get_keyword_index(request: Request):
    return request.app.state.keyword_index


@router.get(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid code search",
    description=(
        "Search indexed code chunks using a hybrid of semantic (cosine similarity) "
        "and keyword (BM25) ranking, optionally boosted by symbol-name matching. "
        "Filter by language and/or repository. "
        "Returns up to `top_k` results sorted by fused relevance score."
    ),
)
async def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=512, description="Search query."),
    top_k: int = Query(
        default=None,
        ge=1,
        le=100,
        description="Number of results to return. Defaults to DEFAULT_TOP_K from config.",
    ),
    language: Optional[Language] = Query(
        default=None,
        description="Filter results to a specific programming language.",
    ),
    repo_name: Optional[str] = Query(
        default=None,
        description="Filter results to a specific indexed repository.",
    ),
) -> SearchResponse:
    effective_top_k = top_k if top_k is not None else settings.default_top_k

    vector_store  = _get_vector_store(request)
    keyword_index = _get_keyword_index(request)

    if vector_store.size == 0 and keyword_index.size == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No code has been indexed yet. POST to /api/index first.",
        )

    ranker = get_ranker(
        semantic_weight=settings.semantic_weight,
        bm25_weight=settings.bm25_weight,
        symbol_weight=settings.symbol_weight,
    )

    try:
        from backend.app.embeddings.embedder import get_embedder
        embedder = get_embedder(settings.embedding_model)

        response = ranker.rank(
            query=q,
            vector_store=vector_store,
            keyword_index=keyword_index,
            top_k=effective_top_k,
            language_filter=language,
            repo_filter=repo_name,
            embedder=embedder,
        )
    except Exception as exc:
        logger.exception("Search failed for query '%s': %s", q, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search error: {exc}",
        )

    logger.info(
        "Search '%s' (repo=%s) → %d results (top score %.3f).",
        q,
        repo_name or "all",
        len(response.results),
        response.results[0].score if response.results else 0.0,
    )

    return response
