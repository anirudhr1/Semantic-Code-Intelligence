"""
routes_repos.py
---------------
Repository management endpoints.

    GET    /api/repos                  — list all indexed repos with chunk counts
                                         and clone size on disk
    DELETE /api/repos/{repo_name}      — remove a repo from both indexes and
                                         (optionally) delete its cloned source files
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, status

from backend.app.config import settings
from backend.app.models.schemas import DeleteRepoResponse, RepoInfo, RepoListResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_vector_store(request: Request):
    return request.app.state.vector_store


def _get_keyword_index(request: Request):
    return request.app.state.keyword_index


def _clone_path(repo_name: str) -> Path:
    """Return the expected on-disk clone path for a repo."""
    return settings.repos_dir / repo_name


def _dir_size_mb(path: Path) -> float:
    """Return the total size of a directory tree in MB, or 0 if it doesn't exist."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)


# ── GET /api/repos ────────────────────────────────────────────────────────────

@router.get(
    "/repos",
    response_model=RepoListResponse,
    status_code=status.HTTP_200_OK,
    summary="List indexed repositories",
    description=(
        "Returns every distinct repository currently in the index, "
        "including chunk counts and whether the cloned source files "
        "are still present on disk."
    ),
)
async def list_repos(request: Request) -> RepoListResponse:
    vector_store  = _get_vector_store(request)
    keyword_index = _get_keyword_index(request)

    # Collect chunk counts per repo from the vector store (single source of truth).
    all_chunks = vector_store.get_all_chunks()
    counts: dict[str, int] = {}
    for chunk in all_chunks:
        name = chunk.repo_name or "(unknown)"
        counts[name] = counts.get(name, 0) + 1

    repos: list[RepoInfo] = []
    for repo_name, chunk_count in sorted(counts.items()):
        clone = _clone_path(repo_name)
        has_clone = clone.exists() and clone.is_dir()
        size_mb = _dir_size_mb(clone) if has_clone else 0.0
        repos.append(
            RepoInfo(
                repo_name=repo_name,
                chunk_count=chunk_count,
                has_local_clone=has_clone,
                clone_size_mb=size_mb,
            )
        )

    return RepoListResponse(
        repos=repos,
        total_chunks=vector_store.size,
    )


# ── DELETE /api/repos/{repo_name} ─────────────────────────────────────────────

@router.delete(
    "/repos/{repo_name}",
    response_model=DeleteRepoResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete an indexed repository",
    description=(
        "Removes all chunks for the given repository from the FAISS vector store "
        "and BM25 keyword index, then persists the updated indexes. "
        "Pass `delete_clone=true` to also delete the cloned source files from disk "
        "(frees the most space). The index rebuild happens in-memory — "
        "no re-embedding is needed."
    ),
)
async def delete_repo(
    repo_name: str,
    request: Request,
    delete_clone: bool = Query(
        default=True,
        description="Also delete the cloned source directory from data/repos/.",
    ),
) -> DeleteRepoResponse:
    vector_store  = _get_vector_store(request)
    keyword_index = _get_keyword_index(request)

    # Check repo exists in the index.
    known_repos = vector_store.list_repos()
    if repo_name not in known_repos:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Repository '{repo_name}' is not in the index. "
                f"Known repos: {known_repos or ['(none)']}"
            ),
        )

    # Measure clone size before deletion.
    clone = _clone_path(repo_name)
    clone_size_mb = _dir_size_mb(clone)

    # ── Remove from both indexes ──────────────────────────────────────────────
    vs_removed  = vector_store.delete_repo(repo_name)
    bm25_removed = keyword_index.delete_repo(repo_name)

    if vs_removed != bm25_removed:
        logger.warning(
            "Chunk count mismatch after delete: FAISS removed %d, BM25 removed %d "
            "for repo '%s'. Indexes may have been out of sync.",
            vs_removed, bm25_removed, repo_name,
        )

    # Persist the trimmed indexes to disk immediately.
    vector_store.save()
    keyword_index.save()

    # ── Optionally delete the cloned source tree ──────────────────────────────
    clone_deleted = False
    freed_mb = 0.0

    if delete_clone and clone.exists():
        try:
            shutil.rmtree(clone)
            clone_deleted = True
            freed_mb = clone_size_mb
            logger.info("Deleted clone directory: %s (%.2f MB freed)", clone, freed_mb)
        except OSError as exc:
            logger.error("Could not delete clone at %s: %s", clone, exc)
            # Don't raise — the index removal succeeded, clone deletion is best-effort.

    logger.info(
        "Deleted repo '%s': %d chunks removed, clone_deleted=%s.",
        repo_name, vs_removed, clone_deleted,
    )

    return DeleteRepoResponse(
        repo_name=repo_name,
        chunks_removed=vs_removed,
        clone_deleted=clone_deleted,
        freed_mb=freed_mb,
        message=(
            f"Removed {vs_removed} chunks for '{repo_name}' from the index"
            + (f" and deleted {freed_mb:.2f} MB of source files." if clone_deleted else ".")
        ),
    )
