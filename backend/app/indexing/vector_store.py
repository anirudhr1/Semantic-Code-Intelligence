"""
vector_store.py
---------------
FAISS-backed vector store that holds code-chunk embeddings and their metadata.

File layout on disk (inside index_dir)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    faiss.index       — serialised FAISS index
    chunks.pkl        — pickled list[CodeChunk]
    embeddings.npy    — raw float32 embedding matrix (N, dim)
                        kept so delete_repo() can rebuild FAISS without
                        re-running the embedding model.

Public API
~~~~~~~~~~
    VectorStore(index_dir, model_name)
    .add(chunks, embeddings)
    .search(query_vector, top_k, language_filter, repo_filter) → list[tuple[CodeChunk, float]]
    .delete_repo(repo_name) → int   (number of chunks removed)
    .list_repos() → list[str]
    .save() / .load() / .clear()
    .size  → int
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from backend.app.models.schemas import CodeChunk, Language

logger = logging.getLogger(__name__)

_FAISS_FILE      = "faiss.index"
_CHUNKS_FILE     = "chunks.pkl"
_EMBEDDINGS_FILE = "embeddings.npy"


class VectorStore:
    """Wraps a FAISS IndexFlatIP with chunk metadata and embedding storage."""

    def __init__(self, index_dir: Path, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._index_dir  = Path(index_dir)
        self._model_name = model_name
        self._index      = None                          # faiss.IndexFlatIP
        self._chunks: list[CodeChunk] = []
        self._embeddings: Optional[np.ndarray] = None   # shape (N, dim)
        self._dimension: int = 0

        if self._persisted_files_exist():
            try:
                self.load()
            except Exception as exc:
                logger.warning("Could not load persisted vector store: %s", exc)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def dimension(self) -> int:
        return self._dimension

    # ── Core operations ───────────────────────────────────────────────────────

    def add(self, chunks: list[CodeChunk], embeddings: np.ndarray) -> None:
        """Add pre-computed L2-normalised float32 embeddings and their CodeChunks."""
        if not chunks:
            return
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) "
                "must have the same length."
            )

        embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]

        if self._index is None:
            self._index     = _make_index(dim)
            self._dimension = dim

        elif dim != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: index has {self._dimension}, "
                f"new vectors have {dim}."
            )

        self._index.add(embeddings)
        self._chunks.extend(chunks)

        # Accumulate raw embeddings so delete_repo() can rebuild without re-encoding.
        if self._embeddings is None:
            self._embeddings = embeddings.copy()
        else:
            self._embeddings = np.concatenate([self._embeddings, embeddings], axis=0)

        logger.debug("Added %d vectors. Total: %d.", len(chunks), self.size)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        language_filter: Optional[Language] = None,
        repo_filter: Optional[str] = None,
    ) -> list[tuple[CodeChunk, float]]:
        """
        Find the top-k most similar chunks.

        Supports optional post-filters for language and repo_name.
        """
        if self._index is None or self.size == 0:
            return []

        query = query_vector.astype(np.float32).reshape(1, -1)

        # Fetch more when filtering so we still fill top_k after removals.
        any_filter = language_filter or repo_filter
        fetch_k = min(top_k * 5 if any_filter else top_k, self.size)

        scores_matrix, idx_matrix = self._index.search(query, fetch_k)
        scores: np.ndarray  = scores_matrix[0]
        indices: np.ndarray = idx_matrix[0]

        results: list[tuple[CodeChunk, float]] = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            chunk = self._chunks[int(idx)]
            if language_filter and chunk.language != language_filter.value:
                continue
            if repo_filter and chunk.repo_name != repo_filter:
                continue
            results.append((chunk, float(np.clip(score, 0.0, 1.0))))
            if len(results) >= top_k:
                break

        return results

    def delete_repo(self, repo_name: str) -> int:
        """
        Remove all chunks belonging to ``repo_name`` and rebuild the FAISS
        index from the remaining embeddings (no re-encoding needed).

        Returns the number of chunks removed.
        """
        if self._embeddings is None or not self._chunks:
            return 0

        keep_mask = np.array(
            [c.repo_name != repo_name for c in self._chunks], dtype=bool
        )
        removed = int((~keep_mask).sum())
        if removed == 0:
            return 0

        kept_chunks     = [c for c, keep in zip(self._chunks, keep_mask) if keep]
        kept_embeddings = self._embeddings[keep_mask]

        # Rebuild FAISS from surviving embeddings.
        self._chunks     = kept_chunks
        self._embeddings = kept_embeddings if len(kept_embeddings) > 0 else None

        if kept_chunks:
            self._index = _make_index(self._dimension)
            self._index.add(kept_embeddings.astype(np.float32))
        else:
            self._index     = None
            self._dimension = 0

        logger.info(
            "Deleted %d chunks for repo '%s'. Remaining: %d.",
            removed, repo_name, self.size,
        )
        return removed

    def list_repos(self) -> list[str]:
        """Return sorted list of distinct repo_name values in the index."""
        return sorted({c.repo_name for c in self._chunks if c.repo_name})

    def encode_and_add(self, chunks: list[CodeChunk], *, show_progress: bool = False) -> None:
        """Embed chunks using the configured model, then add."""
        if not chunks:
            return
        from backend.app.embeddings.embedder import get_embedder
        embedder = get_embedder(self._model_name)
        embeddings = embedder.encode_chunks(chunks, show_progress=show_progress)
        self.add(chunks, embeddings)

    def clear(self) -> None:
        """Remove all vectors and metadata from the store."""
        self._index      = None
        self._chunks     = []
        self._embeddings = None
        self._dimension  = 0
        logger.info("VectorStore cleared.")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist FAISS index, chunk metadata, and raw embeddings to disk."""
        if self._index is None:
            logger.debug("VectorStore.save() called on empty store — nothing written.")
            return

        import faiss

        self._index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_dir / _FAISS_FILE))

        with open(self._index_dir / _CHUNKS_FILE, "wb") as fh:
            pickle.dump(self._chunks, fh, protocol=pickle.HIGHEST_PROTOCOL)

        if self._embeddings is not None:
            np.save(str(self._index_dir / _EMBEDDINGS_FILE), self._embeddings)

        logger.info("VectorStore saved: %d vectors → %s", self.size, self._index_dir)

    def load(self) -> None:
        """Load a previously saved index from disk."""
        import faiss

        self._index = faiss.read_index(str(self._index_dir / _FAISS_FILE))
        with open(self._index_dir / _CHUNKS_FILE, "rb") as fh:
            self._chunks = pickle.load(fh)  # noqa: S301

        self._dimension = self._index.d

        emb_path = self._index_dir / _EMBEDDINGS_FILE
        if emb_path.exists():
            self._embeddings = np.load(str(emb_path))
        else:
            # Older index without embeddings.npy — delete_repo will re-embed if needed.
            self._embeddings = None

        logger.info("VectorStore loaded: %d vectors from %s", self.size, self._index_dir)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _persisted_files_exist(self) -> bool:
        return (
            (self._index_dir / _FAISS_FILE).exists()
            and (self._index_dir / _CHUNKS_FILE).exists()
        )

    def get_all_chunks(self) -> list[CodeChunk]:
        """Return a shallow copy of all stored CodeChunk objects."""
        return list(self._chunks)


# ── Factory helper ────────────────────────────────────────────────────────────

def _make_index(dimension: int):
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is not installed. Run: pip install faiss-cpu") from exc
    return faiss.IndexFlatIP(dimension)
