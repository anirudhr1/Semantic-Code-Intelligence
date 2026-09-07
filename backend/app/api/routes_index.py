"""
routes_index.py
---------------
POST /api/index   — index a repo (incremental: skips unchanged files)
GET  /api/index/status — poll current indexing progress
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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
from backend.app.utils.file_hash_cache import FileHashCache

logger = logging.getLogger(__name__)

router = APIRouter()

# Batch size used when embedding chunks. Exposed as a module constant so it
# is easy to tune without hunting through the request handler body.
_EMBEDDING_BATCH_SIZE = 64


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
        "Uses incremental indexing by default: only new and modified files "
        "are re-embedded, unchanged files are skipped. "
        "Set `force: true` to re-index everything from scratch. "
        "Poll GET /api/index/status for live progress."
    ),
)
async def index_repository(body: IndexRequest, request: Request) -> IndexResponse:
    t_start = time.perf_counter()

    vector_store  = _get_vector_store(request)
    keyword_index = _get_keyword_index(request)
    progress      = _get_progress(request)
    hash_cache    = FileHashCache(settings.index_dir)

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

        # ── 2. Incremental diff — decide which files to process ───────────────
        skipped_unchanged = 0
        files_to_process = repo_result.source_files
        new_hashes: Optional[dict] = None

        existing_repos = vector_store.list_repos()
        is_reindex = repo_name in existing_repos

        if is_reindex and not body.force:
            # Incremental mode: only process changed files
            progress.stage   = "diffing"
            progress.message = "Computing file diffs…"

            changed, new_hashes = hash_cache.get_changed_files(
                repo_name=repo_name,
                source_files=repo_result.source_files,
                repo_root=repo_result.repo_root,
            )

            skipped_unchanged = len(changed.unchanged)
            files_to_process = changed.files_to_process

            # Remove chunks for deleted files
            for deleted_rel in changed.deleted:
                vector_store.delete_by_file(repo_name, deleted_rel)
                keyword_index.delete_by_file(repo_name, deleted_rel)

            # Remove chunks for modified files (they'll be re-inserted below)
            for mod_file in changed.modified:
                rel_path = str(mod_file.relative_to(repo_result.repo_root))
                vector_store.delete_by_file(repo_name, rel_path)
                keyword_index.delete_by_file(repo_name, rel_path)

            if not changed.has_changes:
                # Nothing changed — skip entirely
                duration = round(time.perf_counter() - t_start, 2)
                progress.stage   = "done"
                progress.message = f"No changes — {skipped_unchanged} files unchanged"
                progress.active  = False

                hash_cache.save(repo_name, new_hashes)

                return IndexResponse(
                    repo_name=repo_name,
                    chunks_indexed=0,
                    files_processed=repo_result.file_count,
                    skipped_files=repo_result.skipped_files,
                    skipped_unchanged=skipped_unchanged,
                    duration_seconds=duration,
                    message=f"No changes detected — {skipped_unchanged} files unchanged.",
                )

            progress.files_total = len(files_to_process)
            progress.message = (
                f"Processing {len(files_to_process)} changed files "
                f"(skipping {skipped_unchanged} unchanged)"
            )

        elif is_reindex and body.force:
            # Force re-index: clear everything first
            progress.stage   = "clearing"
            progress.message = f"Removing previous index for '{repo_name}'…"
            vector_store.delete_repo(repo_name)
            keyword_index.delete_repo(repo_name)
            hash_cache.delete(repo_name)

        # ── 3. Extract AST chunks from files to process ───────────────────────
        progress.stage   = "extracting"
        progress.message = "Extracting code chunks…"

        all_chunks    = []
        skipped_files = repo_result.skipped_files

        for file_path in files_to_process:
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

        if not all_chunks and not is_reindex:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No code chunks could be extracted from this repository.",
            )

        if all_chunks:
            progress.chunks_total = len(all_chunks)
            progress.message      = f"Extracted {len(all_chunks)} chunks — embedding…"

            # ── 4. Embed ──────────────────────────────────────────────────────
            progress.stage = "embedding"

            try:
                from backend.app.embeddings.embedder import get_embedder
                import numpy as np

                embedder = get_embedder(settings.embedding_model)

                # Pre-allocate the full output array.
                embeddings = np.empty(
                    (len(all_chunks), embedder.dimension), dtype=np.float32
                )

                for i in range(0, len(all_chunks), _EMBEDDING_BATCH_SIZE):
                    batch = all_chunks[i : i + _EMBEDDING_BATCH_SIZE]
                    embeddings[i : i + len(batch)] = embedder.encode_chunks(batch)
                    progress.chunks_done = min(i + _EMBEDDING_BATCH_SIZE, len(all_chunks))
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

            # ── 5. Insert into indexes ────────────────────────────────────────
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

        # ── 6. Save file hash cache ───────────────────────────────────────────
        if new_hashes is None:
            # First index or force mode — compute hashes now
            new_hashes = {}
            for file_path in repo_result.source_files:
                rel_path = str(file_path.relative_to(repo_result.repo_root))
                new_hashes[rel_path] = FileHashCache.compute_hash(file_path)
        hash_cache.save(repo_name, new_hashes)

        duration = round(time.perf_counter() - t_start, 2)
        progress.stage        = "done"
        progress.chunks_done  = len(all_chunks)
        progress.message      = f"Done — {len(all_chunks)} chunks in {duration}s"
        progress.active       = False

        incremental_note = ""
        if skipped_unchanged > 0:
            incremental_note = f" ({skipped_unchanged} unchanged files skipped)"

        logger.info("Indexing complete for '%s': %d chunks in %.2fs.%s",
                     repo_name, len(all_chunks), duration, incremental_note)

        return IndexResponse(
            repo_name=repo_name,
            chunks_indexed=len(all_chunks),
            files_processed=repo_result.file_count,
            skipped_files=skipped_files,
            skipped_unchanged=skipped_unchanged,
            duration_seconds=duration,
            message=f"Indexed {len(all_chunks)} chunks{incremental_note}.",
        )

    except HTTPException:
        progress.active = False
        raise
    except Exception as exc:
        progress.active = False
        progress.error  = str(exc)
        raise
