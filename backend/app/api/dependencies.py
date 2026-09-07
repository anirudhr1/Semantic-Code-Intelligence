"""
dependencies.py
---------------
Shared FastAPI dependency functions used across all route modules.

Instead of every routes file re-defining the same two-line helpers, they
all import from here.  This is the single source of truth for accessing
shared application state.
"""

from __future__ import annotations

from fastapi import Request

from backend.app.api.routes_index import IndexProgress


def get_vector_store(request: Request):
    """Return the shared VectorStore instance from app.state."""
    return request.app.state.vector_store


def get_keyword_index(request: Request):
    """Return the shared KeywordIndex instance from app.state."""
    return request.app.state.keyword_index


def get_index_progress(request: Request) -> IndexProgress:
    """Return (or lazily create) the IndexProgress tracker from app.state."""
    if not hasattr(request.app.state, "index_progress"):
        request.app.state.index_progress = IndexProgress()
    return request.app.state.index_progress
