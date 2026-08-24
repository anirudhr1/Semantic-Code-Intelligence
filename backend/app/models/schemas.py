"""
Pydantic schemas shared across the application.

Hierarchy:
  CodeChunk        — a single extracted unit of source code (function, class, …)
  IndexRequest     — payload for POST /api/index
  IndexResponse    — response from POST /api/index
  SearchResult     — a single ranked hit returned by GET /api/search
  SearchResponse   — full response from GET /api/search
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class Language(str, Enum):
    python = "python"
    javascript = "javascript"
    typescript = "typescript"
    go = "go"
    rust = "rust"
    java = "java"
    unknown = "unknown"


class ChunkType(str, Enum):
    function = "function"
    method = "method"
    class_ = "class"
    module = "module"
    block = "block"


# ── Core domain object ────────────────────────────────────────────────────────

class CodeChunk(BaseModel):
    """A discrete, indexable unit of source code produced by the AST extractor."""

    chunk_id: UUID = Field(default_factory=uuid4, description="Stable unique identifier.")
    file_path: str = Field(..., description="Repo-relative file path, e.g. 'src/utils.py'.")
    language: Language = Field(default=Language.unknown)
    symbol_name: Optional[str] = Field(
        default=None,
        description="Function / class / method name if determinable.",
    )
    chunk_type: ChunkType = Field(default=ChunkType.block)
    code: str = Field(..., description="Raw source text of the chunk.")
    start_line: int = Field(..., ge=1, description="1-indexed start line in the source file.")
    end_line: int = Field(..., ge=1, description="1-indexed end line in the source file.")
    repo_name: Optional[str] = Field(default=None, description="Repository identifier.")
    docstring: Optional[str] = Field(default=None, description="Extracted docstring/comment.")

    @model_validator(mode="after")
    def end_after_start(self) -> "CodeChunk":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self

    class Config:
        use_enum_values = True


# ── Index API ─────────────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    """
    Exactly one of `path` or `repo_url` must be provided.
    - path     : absolute or workspace-relative path to an already-cloned repo.
    - repo_url : remote git URL; the backend will clone it into REPOS_DIR.
    """

    path: Optional[str] = Field(default=None, description="Local filesystem path to the repo.")
    repo_url: Optional[str] = Field(default=None, description="Remote git URL to clone.")
    language: Optional[Language] = Field(
        default=None,
        description="If given, only files matching this language are indexed.",
    )
    repo_name: Optional[str] = Field(
        default=None,
        description="Human-readable identifier stored on each chunk. Defaults to directory name.",
    )

    @model_validator(mode="after")
    def exactly_one_source(self) -> "IndexRequest":
        if not self.path and not self.repo_url:
            raise ValueError("Provide either 'path' or 'repo_url'.")
        if self.path and self.repo_url:
            raise ValueError("Provide only one of 'path' or 'repo_url', not both.")
        return self


class IndexResponse(BaseModel):
    """Summary returned after a successful indexing run."""

    repo_name: str
    chunks_indexed: int
    files_processed: int
    skipped_files: int
    duration_seconds: float
    message: str = "Indexing complete."


# ── Search API ────────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    """A single ranked code chunk returned by the search endpoint."""

    chunk_id: UUID
    file_path: str
    language: Language
    symbol_name: Optional[str]
    chunk_type: ChunkType
    code: str
    start_line: int
    end_line: int
    repo_name: Optional[str]
    score: float = Field(..., ge=0.0, le=1.0, description="Fused relevance score [0, 1].")
    semantic_score: float = Field(default=0.0)
    bm25_score: float = Field(default=0.0)
    symbol_score: float = Field(default=0.0)

    class Config:
        use_enum_values = True


class SearchResponse(BaseModel):
    """Full response envelope for GET /api/search."""

    query: str
    top_k: int
    total_indexed: int
    results: list[SearchResult]
