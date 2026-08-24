"""
routes_index.py
---------------
POST /api/index

Accepts an IndexRequest, clones or resolves the repo, extracts AST chunks,
embeds them, and adds them to both the FAISS vector store and the BM25 index.

Flow
~~~~
    POST /api/index
        → repo_loader.load_repo()          resolve / clone
        → ast_extractor.extract_chunks()   parse each file
        → chunking.split_oversized()       guard against huge chunks
        → embedder.encode_chunks()         batch embed
        → vector_store.add()               insert into FAISS
        → keyword_index.add()              insert into BM25
        → IndexResponse                    return summary
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status

from backend.app.config import settings
from backend.app.ingestion.ast_extractor import extract_chunks
from backend.app.ingestion.repo_loader import load_repo
from backend.app.models.schemas import IndexRequest, IndexResponse, Language
from backend.app.utils.chunking import split_oversized_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Dependency helpers ────────────────────────────────────────────────────────

def _get_vector_store(request: Request):
    return request.app.state.vector_store


def _get_keyword_index(request: Request):
    return request.app.state.keyword_index


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/index",
    response_model=IndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Index a repository",
    description=(
        "Index a local directory or a remote git repository. "
        "Provide exactly one of `path` (local filesystem path) or "
        "`repo_url` (remote git URL). "
        "Existing chunks for the same repo are **not** deduplicated automatically; "
        "clear the index first if you want a clean re-index."
    ),
)
async def index_repository(body: IndexRequest, request: Request) -> IndexResponse:
    t_start = time.perf_counter()

    vector_store = _get_vector_store(request)
    keyword_index = _get_keyword_index(request)

    # ── 1. Resolve / clone the repo ───────────────────────────────────────────
    try:
        repo_result = load_repo(
            path=body.path,
            repo_url=body.repo_url,
            repos_dir=settings.repos_dir,
            language=body.language,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to clone repository: {exc}",
        )

    repo_name = body.repo_name or repo_result.repo_name
    logger.info(
        "Indexing repo '%s': %d files to process.",
        repo_name,
        repo_result.file_count,
    )

    # ── 2. Extract AST chunks from every source file ──────────────────────────
    all_chunks = []
    skipped_files = repo_result.skipped_files

    for file_path in repo_result.source_files:
        rel_path = str(file_path.relative_to(repo_result.repo_root))
        try:
            chunks = extract_chunks(
                file_path,
                repo_name=repo_name,
                language=body.language,
            )
            # Patch the file_path to a repo-relative form for cleaner display.
            for c in chunks:
                c.file_path = rel_path

            # Guard against runaway chunk sizes (e.g. minified JS files).
            chunks = split_oversized_chunks(chunks)
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.warning("Skipping %s — extraction error: %s", rel_path, exc)
            skipped_files += 1

    if not all_chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No code chunks could be extracted. "
                "Check that the repo contains supported source files "
                f"({', '.join(body.language.value if body.language else 'any language')}) "
                "and is not empty."
            ),
        )

    logger.info("Extracted %d chunks from %d files.", len(all_chunks), repo_result.file_count)

    # ── 3. Embed and index ────────────────────────────────────────────────────
    try:
        from backend.app.embeddings.embedder import get_embedder
        embedder = get_embedder(settings.embedding_model)
        embeddings = embedder.encode_chunks(all_chunks, show_progress=False)
    except Exception as exc:
        logger.exception("Embedding failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding error: {exc}",
        )

    try:
        vector_store.add(all_chunks, embeddings)
        keyword_index.add(all_chunks)
    except Exception as exc:
        logger.exception("Index insertion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index insertion error: {exc}",
        )

    # Persist after every successful index run.
    vector_store.save()
    keyword_index.save()

    duration = round(time.perf_counter() - t_start, 2)
    logger.info(
        "Indexing complete for '%s': %d chunks in %.2fs.",
        repo_name,
        len(all_chunks),
        duration,
    )

    return IndexResponse(
        repo_name=repo_name,
        chunks_indexed=len(all_chunks),
        files_processed=repo_result.file_count,
        skipped_files=skipped_files,
        duration_seconds=duration,
    )
