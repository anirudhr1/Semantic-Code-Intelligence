"""
FastAPI application entrypoint.

Startup sequence
----------------
1. Ensure storage directories exist.
2. Load (or create) the FAISS vector store.
3. Load (or create) the BM25 keyword index.
4. Register API routers.

Run locally:
    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.config import settings
from backend.app.api.routes_index import router as index_router
from backend.app.api.routes_search import router as search_router
from backend.app.api.routes_repos import router as repos_router
from backend.app.indexing.vector_store import VectorStore
from backend.app.indexing.keyword_index import KeywordIndex

logger = logging.getLogger(__name__)

# ── Shared state attached to app.state ───────────────────────────────────────
# Other modules retrieve these via `request.app.state.<attr>` or the
# dependency helpers defined in each route module.

_APP_START_TIME: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _APP_START_TIME
    _APP_START_TIME = time.time()

    logger.info("Starting Semantic Code Intelligence API …")
    settings.ensure_dirs()

    # Initialise shared indexes (loads from disk if persisted, else creates empty).
    app.state.vector_store = VectorStore(
        index_dir=settings.index_dir,
        model_name=settings.embedding_model,
    )
    app.state.keyword_index = KeywordIndex(index_dir=settings.index_dir)

    logger.info(
        "Indexes ready. Vector store: %d chunks, BM25: %d docs.",
        app.state.vector_store.size,
        app.state.keyword_index.size,
    )

    yield  # ← application runs here

    logger.info("Shutting down — persisting indexes …")
    app.state.vector_store.save()
    app.state.keyword_index.save()
    logger.info("Shutdown complete.")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    application = FastAPI(
        title="Semantic Code Intelligence",
        description=(
            "Hybrid semantic + keyword search over source code repositories. "
            "Index a repo via POST /api/index, then search with GET /api/search."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Allow the vanilla-JS frontend (served from /frontend or a dev server) to
    # call the API without CORS errors.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API routers ───────────────────────────────────────────────────────────
    application.include_router(index_router, prefix="/api", tags=["indexing"])
    application.include_router(search_router, prefix="/api", tags=["search"])
    application.include_router(repos_router, prefix="/api", tags=["repositories"])

    # ── Health check ──────────────────────────────────────────────────────────
    @application.get("/api/health", tags=["health"])
    async def health() -> dict:
        uptime = round(time.time() - _APP_START_TIME, 1)
        return {
            "status": "ok",
            "uptime_seconds": uptime,
            "embedding_model": settings.embedding_model,
            "vector_store_size": application.state.vector_store.size,
            "bm25_index_size": application.state.keyword_index.size,
        }

    # ── Serve frontend static files ───────────────────────────────────────────
    try:
        application.mount(
            "/frontend",
            StaticFiles(directory="frontend"),
            name="frontend",
        )

        @application.get("/", include_in_schema=False)
        async def serve_ui() -> FileResponse:
            return FileResponse("frontend/index.html")

    except RuntimeError:
        # frontend directory might not exist in some test environments
        logger.warning("frontend/ directory not found — UI will not be served.")

    return application


app: FastAPI = create_app()
