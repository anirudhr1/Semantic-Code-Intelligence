"""
vector_store.py
---------------
FAISS-backed vector store that holds code-chunk embeddings and their metadata.

Responsibilities
~~~~~~~~~~~~~~~~
* Add embeddings produced by Embedder to an IndexFlatIP index (inner-product
  == cosine similarity when vectors are L2-normalised).
* Store the corresponding CodeChunk objects in a parallel list so search
  results carry full metadata, not just IDs.
* Persist / load the index and metadata to / from disk.
* Expose a simple search(query_vector, top_k) → list[(chunk, score)] API.

File layout on disk (inside index_dir)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    faiss.index       — serialised FAISS index
    chunks.pkl        — pickled list[CodeChunk]

Public API
~~~~~~~~~~
    VectorStore(index_dir, model_name)
    .add(chunks, embeddings)
    .search(query_vector, top_k, language_filter) → list[tuple[CodeChunk, float]]
    .save()
    .load()
    .clear()
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

_FAISS_FILE = "faiss.index"
_CHUNKS_FILE = "chunks.pkl"


class VectorStore:
    """
    Wraps a FAISS IndexFlatIP with chunk metadata storage.

    Parameters
    ----------
    index_dir:
        Directory where ``faiss.index`` and ``chunks.pkl`` are persisted.
    model_name:
        sentence-transformers model name — used to instantiate the Embedder
        when encode_and_add() is called directly.  If you supply pre-computed
        embeddings via add() this parameter is unused.
    """

    def __init__(self, index_dir: Path, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._index_dir = Path(index_dir)
        self._model_name = model_name
        self._index = None          # faiss.IndexFlatIP, lazy-created
        self._chunks: list[CodeChunk] = []
        self._dimension: int = 0

        # Try to load a previously persisted index.
        if self._persisted_files_exist():
            try:
                self.load()
            except Exception as exc:
                logger.warning("Could not load persisted vector store: %s", exc)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return len(self._chunks)

    @property
    def dimension(self) -> int:
        return self._dimension

    # ── Core operations ───────────────────────────────────────────────────────

    def add(self, chunks: list[CodeChunk], embeddings: np.ndarray) -> None:
        """
        Add pre-computed embeddings (float32, L2-normalised) and their
        corresponding CodeChunk objects to the index.

        Parameters
        ----------
        chunks:
            List of CodeChunk objects (len must equal embeddings.shape[0]).
        embeddings:
            ndarray of shape (N, dim), dtype float32, L2-normalised.
        """
        if len(chunks) == 0:
            return
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) "
                "must have the same length."
            )

        embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]

        if self._index is None:
            self._index = _make_index(dim)
            self._dimension = dim
        elif dim != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: index has {self._dimension}, "
                f"new vectors have {dim}."
            )

        self._index.add(embeddings)
        self._chunks.extend(chunks)
        logger.debug("Added %d vectors. Total: %d.", len(chunks), self.size)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        language_filter: Optional[Language] = None,
    ) -> list[tuple[CodeChunk, float]]:
        """
        Find the top-k most similar chunks to ``query_vector``.

        Parameters
        ----------
        query_vector:
            1-D float32 array of length ``dimension``, L2-normalised.
        top_k:
            Maximum number of results to return.
        language_filter:
            When provided, only chunks with matching language are returned.
            This is applied as a post-filter; more candidates are fetched
            internally to compensate for filtered-out results.

        Returns
        -------
        List of (CodeChunk, cosine_score) tuples, highest score first.
        Scores are in [0, 1] (inner product of unit vectors).
        """
        if self._index is None or self.size == 0:
            return []

        query = query_vector.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Fetch extra candidates when filtering to avoid an empty result set.
        fetch_k = min(top_k * 5 if language_filter else top_k, self.size)

        scores_matrix, idx_matrix = self._index.search(query, fetch_k)
        scores: np.ndarray = scores_matrix[0]
        indices: np.ndarray = idx_matrix[0]

        results: list[tuple[CodeChunk, float]] = []
        for idx, score in zip(indices, scores):
            if idx < 0:          # FAISS returns -1 for unfilled slots
                continue
            chunk = self._chunks[int(idx)]
            if language_filter and chunk.language != language_filter.value:
                continue
            results.append((chunk, float(np.clip(score, 0.0, 1.0))))
            if len(results) >= top_k:
                break

        return results

    def encode_and_add(self, chunks: list[CodeChunk], *, show_progress: bool = False) -> None:
        """
        Convenience method: embed `chunks` using the configured model, then add.
        Imports Embedder lazily to avoid circular imports.
        """
        if not chunks:
            return
        from backend.app.embeddings.embedder import get_embedder
        embedder = get_embedder(self._model_name)
        embeddings = embedder.encode_chunks(chunks, show_progress=show_progress)
        self.add(chunks, embeddings)

    def clear(self) -> None:
        """Remove all vectors and metadata from the store."""
        self._index = None
        self._chunks = []
        self._dimension = 0
        logger.info("VectorStore cleared.")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the FAISS index and chunk metadata to ``index_dir``."""
        if self._index is None:
            logger.debug("VectorStore.save() called on empty store — nothing written.")
            return

        import faiss  # noqa: PLC0415

        self._index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_dir / _FAISS_FILE))

        with open(self._index_dir / _CHUNKS_FILE, "wb") as fh:
            pickle.dump(self._chunks, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "VectorStore saved: %d vectors → %s", self.size, self._index_dir
        )

    def load(self) -> None:
        """Load a previously saved index and metadata from ``index_dir``."""
        import faiss  # noqa: PLC0415

        index_path = self._index_dir / _FAISS_FILE
        chunks_path = self._index_dir / _CHUNKS_FILE

        self._index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as fh:
            self._chunks = pickle.load(fh)  # noqa: S301

        self._dimension = self._index.d
        logger.info(
            "VectorStore loaded: %d vectors from %s", self.size, self._index_dir
        )

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
    """Create a new FAISS IndexFlatIP for the given vector dimension."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is not installed. Run: pip install faiss-cpu"
        ) from exc
    return faiss.IndexFlatIP(dimension)
