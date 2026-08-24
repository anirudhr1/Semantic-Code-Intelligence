"""
Application settings loaded from environment variables / .env file.
All other modules import `settings` from here — never read env vars directly.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Embedding model ───────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model identifier.",
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    index_dir: Path = Field(
        default=Path("data/index"),
        description="Directory where FAISS, BM25 and chunk metadata are persisted.",
    )
    repos_dir: Path = Field(
        default=Path("data/repos"),
        description="Directory where remote git repos are cloned.",
    )

    # ── Ranking weights ───────────────────────────────────────────────────────
    semantic_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    symbol_weight: float = Field(default=0.1, ge=0.0, le=1.0)

    # ── Search defaults ───────────────────────────────────────────────────────
    default_top_k: int = Field(default=10, ge=1, le=200)

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    @field_validator("semantic_weight", "bm25_weight", "symbol_weight", mode="before")
    @classmethod
    def _parse_float(cls, v: object) -> float:
        return float(v)

    def ensure_dirs(self) -> None:
        """Create storage directories if they don't exist yet."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.repos_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()


# Convenience alias used throughout the codebase.
settings: Settings = get_settings()
