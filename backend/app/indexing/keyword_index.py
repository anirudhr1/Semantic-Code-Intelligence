"""
keyword_index.py
----------------
BM25-based keyword index over code chunks.

BM25 is a proven bag-of-words ranking function that complements semantic
search well — it excels at exact symbol-name and API-name matches where
embedding models can be weak.

Design
~~~~~~
* Uses rank-bm25 (BM25Okapi) which operates on pre-tokenised token lists.
* Tokenisation is code-aware: splits on whitespace and common code
  delimiters (``_``, ``-``, camelCase, ``(``, ``.``, etc.) and lowercases.
* Scores are normalised to [0, 1] by dividing by the max score in a result
  set so they can be fused with cosine scores on the same scale.

File layout on disk (inside index_dir)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    bm25.pkl    — pickled (BM25Okapi, list[CodeChunk]) tuple

Public API
~~~~~~~~~~
    KeywordIndex(index_dir)
    .add(chunks)
    .search(query, top_k, language_filter) → list[(CodeChunk, float)]
    .save()
    .load()
    .clear()
    .size  → int
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

from backend.app.models.schemas import CodeChunk, Language

logger = logging.getLogger(__name__)

_BM25_FILE = "bm25.pkl"

# ── Tokeniser ─────────────────────────────────────────────────────────────────

_SPLIT_RE = re.compile(r"[_\-\s\.\(\)\[\]\{\},;:\"\'<>/\\|@#$%^&*+=~`!?]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenise(text: str) -> list[str]:
    """
    Convert source text to a lowercase token list suitable for BM25.

    Steps:
    1. Split camelCase → separate tokens.
    2. Split on common code delimiters.
    3. Lowercase and drop empty / single-char tokens.
    """
    # Split camelCase
    text = _CAMEL_RE.sub(" ", text)
    # Split on delimiters
    raw_tokens = _SPLIT_RE.split(text)
    # Normalise and filter
    return [t.lower() for t in raw_tokens if len(t) > 1]


def _chunk_to_token_list(chunk: CodeChunk) -> list[str]:
    """Build the token document for a chunk (symbol name + code)."""
    parts: list[str] = []
    if chunk.symbol_name:
        # Repeat the symbol name to up-weight it in BM25.
        parts.extend(tokenise(chunk.symbol_name) * 3)
    if chunk.docstring:
        parts.extend(tokenise(chunk.docstring))
    parts.extend(tokenise(chunk.code))
    return parts


# ── KeywordIndex ──────────────────────────────────────────────────────────────

class KeywordIndex:
    """
    BM25 index over CodeChunk objects.

    Parameters
    ----------
    index_dir:
        Directory where ``bm25.pkl`` is persisted.
    """

    def __init__(self, index_dir: Path) -> None:
        self._index_dir = Path(index_dir)
        self._bm25 = None          # rank_bm25.BM25Okapi, created on first add()
        self._chunks: list[CodeChunk] = []
        self._corpus: list[list[str]] = []   # parallel to _chunks

        if (self._index_dir / _BM25_FILE).exists():
            try:
                self.load()
            except Exception as exc:
                logger.warning("Could not load persisted BM25 index: %s", exc)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._chunks)

    # ── Core operations ───────────────────────────────────────────────────────

    def add(self, chunks: list[CodeChunk]) -> None:
        """
        Add chunks to the BM25 index.

        The internal BM25 model is rebuilt from scratch each time because
        rank-bm25 does not support incremental updates.  For large repos this
        is fast (< 1 s for tens of thousands of chunks).
        """
        if not chunks:
            return

        new_docs = [_chunk_to_token_list(c) for c in chunks]
        self._chunks.extend(chunks)
        self._corpus.extend(new_docs)
        self._rebuild()
        logger.debug("BM25 index rebuilt: %d documents.", self.size)

    def search(
        self,
        query: str,
        top_k: int = 10,
        language_filter: Optional[Language] = None,
    ) -> list[tuple[CodeChunk, float]]:
        """
        Retrieve the top-k chunks matching ``query`` using BM25.

        Returns
        -------
        List of (CodeChunk, normalised_bm25_score) sorted highest-first.
        Scores are normalised to [0, 1].
        """
        if self._bm25 is None or self.size == 0:
            return []

        query_tokens = tokenise(query)
        if not query_tokens:
            return []

        raw_scores: list[float] = self._bm25.get_scores(query_tokens).tolist()

        # Pair with chunks and filter
        paired: list[tuple[CodeChunk, float]] = []
        for chunk, score in zip(self._chunks, raw_scores):
            if language_filter and chunk.language != language_filter.value:
                continue
            paired.append((chunk, score))

        # Sort descending
        paired.sort(key=lambda x: x[1], reverse=True)
        top = paired[:top_k]

        # Normalise scores to [0, 1]
        if top:
            max_score = top[0][1]
            if max_score > 0:
                top = [(c, s / max_score) for c, s in top]
            else:
                top = [(c, 0.0) for c, _ in top]

        return top

    def clear(self) -> None:
        """Remove all documents from the index."""
        self._bm25 = None
        self._chunks = []
        self._corpus = []
        logger.info("KeywordIndex cleared.")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Pickle the BM25 model, chunks, and corpus to ``index_dir``."""
        if self._bm25 is None:
            logger.debug("KeywordIndex.save() called on empty index — nothing written.")
            return

        self._index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25": self._bm25,
            "chunks": self._chunks,
            "corpus": self._corpus,
        }
        with open(self._index_dir / _BM25_FILE, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("KeywordIndex saved: %d docs → %s", self.size, self._index_dir)

    def load(self) -> None:
        """Load a previously saved index from ``index_dir``."""
        path = self._index_dir / _BM25_FILE
        with open(path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301

        self._bm25 = payload["bm25"]
        self._chunks = payload["chunks"]
        self._corpus = payload["corpus"]
        logger.info("KeywordIndex loaded: %d docs from %s", self.size, self._index_dir)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        """Rebuild the BM25Okapi model from the current corpus."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "rank-bm25 is not installed. Run: pip install rank-bm25"
            ) from exc
        self._bm25 = BM25Okapi(self._corpus)
