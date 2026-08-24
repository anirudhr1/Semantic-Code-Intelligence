"""
embedder.py
-----------
Thin wrapper around sentence-transformers that produces fixed-size float32
embedding vectors for code chunks.

Design decisions
~~~~~~~~~~~~~~~~
* The model is loaded once and cached — loading is expensive (~1–5 s).
* Batched encoding is used for throughput; individual calls go through the
  same path for consistency.
* Embeddings are L2-normalised so that inner-product == cosine similarity,
  which is what FAISS IndexFlatIP expects.

Public API
~~~~~~~~~~
    Embedder(model_name)          — singleton-friendly class
    embedder.encode(texts)        → np.ndarray  shape (N, dim)
    embedder.encode_one(text)     → np.ndarray  shape (dim,)
    embedder.dimension            → int
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

# Default batch size for encode() calls.
_DEFAULT_BATCH_SIZE = 64


class Embedder:
    """
    Sentence-transformers wrapper with L2-normalised outputs.

    Parameters
    ----------
    model_name:
        Any model identifier accepted by sentence-transformers, e.g.
        ``"all-MiniLM-L6-v2"`` or ``"microsoft/graphcodebert-base"``.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        Defaults to ``"cpu"`` for portability; set to ``"cuda"`` in
        GPU environments for a significant throughput improvement.
    batch_size:
        Number of texts encoded in a single forward pass.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None  # lazy-loaded on first use
        self._dimension: int | None = None

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the sentence-transformers model (called once)."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model '%s' on device '%s' …", self._model_name, self._device)
        self._model = SentenceTransformer(self._model_name, device=self._device)
        # Warm-up: encode a dummy string to determine embedding dimension.
        dummy = self._model.encode(["hello"], convert_to_numpy=True, normalize_embeddings=True)
        self._dimension = int(dummy.shape[1])
        logger.info("Model loaded. Embedding dimension: %d.", self._dimension)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """Return the embedding vector size. Triggers model load if needed."""
        if self._dimension is None:
            self._load()
        return self._dimension  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of strings into L2-normalised float32 vectors.

        Parameters
        ----------
        texts:
            Input strings to embed. Empty strings are replaced with a single
            space to avoid model errors.
        show_progress:
            Show a tqdm progress bar (useful for large batch CLI indexing).

        Returns
        -------
        np.ndarray of shape ``(len(texts), dimension)`` and dtype float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._load()

        # Guard against empty strings which can cause NaN embeddings.
        cleaned = [t if t.strip() else " " for t in texts]

        embeddings: np.ndarray = self._model.encode(  # type: ignore[union-attr]
            cleaned,
            batch_size=self._batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalise → cosine via inner product
        )

        return embeddings.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """
        Encode a single string.

        Returns
        -------
        np.ndarray of shape ``(dimension,)`` and dtype float32.
        """
        return self.encode([text])[0]

    def encode_chunks(
        self,
        chunks,
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Convenience helper: encode a list of CodeChunk objects.

        The text passed to the model is constructed as::

            "<symbol_name>\n\n<code>"

        giving the model symbol-name context alongside the source text.

        Parameters
        ----------
        chunks:
            Iterable of ``CodeChunk`` instances.

        Returns
        -------
        np.ndarray of shape ``(len(chunks), dimension)``.
        """
        texts = [_chunk_to_text(c) for c in chunks]
        return self.encode(texts, show_progress=show_progress)


# ── Text preparation ──────────────────────────────────────────────────────────

def _chunk_to_text(chunk) -> str:
    """
    Build the string that will be embedded for a CodeChunk.

    Format (each section separated by a blank line):
        <symbol_name>           ← omitted when None
        <docstring>             ← omitted when None
        <code>
    """
    parts: list[str] = []
    if chunk.symbol_name:
        parts.append(chunk.symbol_name)
    if chunk.docstring:
        parts.append(chunk.docstring)
    parts.append(chunk.code)
    return "\n\n".join(parts)


# ── Module-level singleton ────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def get_embedder(model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> Embedder:
    """
    Return a cached Embedder instance for the given (model_name, device) pair.
    Calling this multiple times with the same arguments returns the same object.
    """
    return Embedder(model_name=model_name, device=device)
