"""
routes_index.py
---------------
POST /api/index   — index a repo (auto-replaces existing chunks for same repo)
GET  /api/index/status — poll current indexing progress
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status

from backend.app.config import settings
from backend.app.ingestion.ast_extractor import extract_chunks
from backend.app.ingestion.repo_loader import load_repo
from backend.app.models.schemas import IndexRequest, IndexResponse, Language
from backend.app.utils.chunking import (
    deduplicate_chunks,
    filter_empty_chunks,
    split_oversized_chunks,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Progress state ────────────────────────────────────────────────────────────

@dataclass
class IndexProgress:
    """Mutable progress object stored on app.state.index_progress."""
    active: bool        = False
    repo_name: str      = ""
    stage: str          = ""          # e.g. "cloning", "extracting", "embedding", "done"
    files_total: int    = 0
    files_done: int     = 0
    chunks_total: int   = 0
    chunks_done: int    = 0
    message: str        = ""
    error: str          = ""

    def reset(self) -> None:
        self.active      = False
        self.repo_name   = ""
        self.stage       = ""
        self.files_total = 0
        self.files_done  = 0
        self.chunks_total = 0
        self.chunks_done  = 0
        self.message     = ""
        self.error       = ""

    def pct(self) -> int:
        """Overall progress 0–100 based on embedding stage."""
        if self.chunks_total == 0:
            return 0
        return min(int(self.chunks_done / self.chunks_total * 100), 99)


# ── Dependency helpers ────────────────────────────────────────────────────────
# Thin wrappers kept here so existing call-sites inside this file don't change.
# The shared implementations live in dependencies.py.

def _get_vector_store(request: Request):
    return request.app.state.vector_store

def _get_keyword_index(request: Request):
    return request.app.state.keyword_index

def _get_progress(request: Request) -> IndexProgress:
    if not hasattr(request.app.state, "index_progress"):
        request.app.state.index_progress = IndexProgress()
    return request.app.state.index_progress


# ── GET /api/index/status ─────────────────────────────────────────────────────

@router.get("/index/status", tags=["indexing"], summary="Poll indexing progress")
async def index_status(request: Request) -> dict:
    """
    Returns current indexing state. Poll this while POST /api/index is running.
    """
    p = _get_progress(request)
    return {
        "active":        p.active,
        "repo_name":     p.repo_name,
        "stage":         p.stage,
        "files_total":   p.files_total,
        "files_done":    p.files_done,
        "chunks_total":  p.chunks_total,
        "chunks_done":   p.chunks_done,
        "pct":           p.pct(),
        "message":       p.message,
        "error":         p.error,
    }


# ── POST /api/index ───────────────────────────────────────────────────────────

@router.post(
    "/index",
    response_model=IndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Index a repository",
    description=(
        "Index a local directory or a remote git repository. "
        "If the repo_name already exists in the index, its old chunks are "
        "automatically removed before the new ones are inserted (clean re-index). "
        "Poll GET /api/index/status for live progress."
    ),
)
async def index_repository(body: IndexRequest, request: Request) -> IndexResponse:
    t_start = time.perf_counter()

    vector_store  = _get_vector_store(request)
    keyword_index = _get_keyword_index(request)
    progress      = _get_progress(request)

    progress.reset()
    progress.active = True

    try:
        # ── 1. Resolve / clone the repo ───────────────────────────────────────
        progress.stage   = "cloning"
        progress.message = "Resolving repository…"

        try:
            repo_result = load_repo(
                path=body.path,
                repo_url=body.repo_url,
                repos_dir=settings.repos_dir,
                language=body.language,
            )
        except (FileNotFoundError, NotADirectoryError) as exc:
            progress.error = str(exc)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        except RuntimeError as exc:
            progress.error = str(exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to clone repository: {exc}",
            )

        repo_name = body.repo_name or repo_result.repo_name
        progress.repo_name   = repo_name
        progress.files_total = repo_result.file_count
        progress.message     = f"Found {repo_result.file_count} files"

        # ── FIX 1: auto-clear old chunks for this repo before re-indexing ────
        existing_repos = vector_store.list_repos()
        if repo_name in existing_repos:
            logger.info("Re-indexing '%s' — removing %d existing chunks first.", repo_name,
                        sum(1 for c in vector_store.get_all_chunks() if c.repo_name == repo_name))
            progress.stage   = "clearing"
            progress.message = f"Removing previous index for '{repo_name}'…"
            vector_store.delete_repo(repo_name)
            keyword_index.delete_repo(repo_name)

        # ── 2. Extract AST chunks from every source file ──────────────────────
        progress.stage   = "extracting"
        progress.message = "Extracting code chunks…"

        all_chunks    = []
        skipped_files = repo_result.skipped_files

        for file_path in repo_result.source_files:
            rel_path = str(file_path.relative_to(repo_result.repo_root))
            try:
                chunks = extract_chunks(file_path, repo_name=repo_name, language=body.language)
                for c in chunks:
                    c.file_path = rel_path
                chunks = split_oversized_chunks(chunks)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.warning("Skipping %s — extraction error: %s", rel_path, exc)
                skipped_files += 1

            progress.files_done += 1
            progress.message = f"Extracted {progress.files_done}/{progress.files_total} files"

        all_chunks = filter_empty_chunks(all_chunks)
        all_chunks = deduplicate_chunks(all_chunks)

        if not all_chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No code chunks could be extracted from this repository.",
            )

        progress.chunks_total = len(all_chunks)
        progress.message      = f"Extracted {len(all_chunks)} chunks — embedding…"

        # ── 3. Embed ──────────────────────────────────────────────────────────
        progress.stage = "embedding"

        try:
            from backend.app.embeddings.embedder import get_embedder
            import numpy as np

            embedder = get_embedder(settings.embedding_model)

            # Pre-allocate the full output array — avoids building a list of
            # intermediate arrays and calling np.concatenate at the end.
            _BATCH = 64
            embeddings = np.empty(
                (len(all_chunks), embedder.dimension), dtype=np.float32
            )

            for i in range(0, len(all_chunks), _BATCH):
                batch = all_chunks[i : i + _BATCH]
                # encode_chunks is the single source of truth for chunk→text
                # conversion (symbol_name + docstring + code).  No local
                # _chunk_text duplicate needed.
                embeddings[i : i + len(batch)] = embedder.encode_chunks(batch)
                progress.chunks_done = min(i + _BATCH, len(all_chunks))
                progress.message = (
                    f"Embedding {progress.chunks_done}/{len(all_chunks)} chunks…"
                )

        except Exception as exc:
            logger.exception("Embedding failed: %s", exc)
            progress.error = str(exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embedding error: {exc}",
            )

        # ── 4. Insert into indexes ────────────────────────────────────────────
        progress.stage   = "indexing"
        progress.message = "Inserting into index…"

        try:
            vector_store.add(all_chunks, embeddings)
            keyword_index.add(all_chunks)
        except Exception as exc:
            logger.exception("Index insertion failed: %s", exc)
            progress.error = str(exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Index insertion error: {exc}",
            )

        vector_store.save()
        keyword_index.save()

        duration = round(time.perf_counter() - t_start, 2)
        progress.stage        = "done"
        progress.chunks_done  = len(all_chunks)
        progress.message      = f"Done — {len(all_chunks)} chunks in {duration}s"
        progress.active       = False

        logger.info("Indexing complete for '%s': %d chunks in %.2fs.", repo_name, len(all_chunks), duration)

        return IndexResponse(
            repo_name=repo_name,
            chunks_indexed=len(all_chunks),
            files_processed=repo_result.file_count,
            skipped_files=skipped_files,
            duration_seconds=duration,
        )

    except HTTPException:
        progress.active = False
        raise
    except Exception as exc:
        progress.active = False
        progress.error  = str(exc)
        raise


