"""
chunking.py
-----------
Utility helpers for post-processing CodeChunk lists produced by the AST
extractor before they are handed to the embedder and indexes.

Responsibilities
~~~~~~~~~~~~~~~~
* split_oversized_chunks  — break chunks that exceed a character/line budget
  into overlapping sub-chunks so the embedding model receives well-sized input.
* deduplicate_chunks      — drop exact-duplicate code blocks (same file + same
  source text) that can appear when a file is re-indexed.
* filter_empty_chunks     — remove chunks whose code is blank or whitespace-only.
* chunk_stats             — return a simple statistics dict for logging.

Constants
~~~~~~~~~
MAX_CHUNK_CHARS   — soft character limit per chunk (default 3 000).
                    sentence-transformers models typically truncate at 512
                    tokens (≈ 2 000–4 000 chars depending on language), so
                    staying under 3 000 chars keeps most chunks within the
                    model's context window.
OVERLAP_LINES     — number of lines of context carried over into the next
                    sub-chunk when a split occurs (default 5).
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from backend.app.models.schemas import ChunkType, CodeChunk

# ── Tuneable constants ────────────────────────────────────────────────────────

MAX_CHUNK_CHARS: int = 3_000
OVERLAP_LINES: int = 5


# ── Public helpers ────────────────────────────────────────────────────────────

def split_oversized_chunks(
    chunks: list[CodeChunk],
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_lines: int = OVERLAP_LINES,
) -> list[CodeChunk]:
    """
    Split any chunk whose ``code`` exceeds ``max_chars`` characters into
    multiple overlapping sub-chunks.

    Chunks within the limit are returned unchanged.  The original chunk is
    replaced in the output list by its sub-chunks; ordering is preserved.

    Parameters
    ----------
    chunks:
        Input list of CodeChunk objects (typically from ``ast_extractor``).
    max_chars:
        Maximum character length per output chunk.
    overlap_lines:
        Number of lines shared between consecutive sub-chunks to preserve
        context at the boundary.

    Returns
    -------
    New list of CodeChunk objects — never modifies inputs in place.
    """
    result: list[CodeChunk] = []
    for chunk in chunks:
        if len(chunk.code) <= max_chars:
            result.append(chunk)
        else:
            result.extend(_split_chunk(chunk, max_chars, overlap_lines))
    return result


def deduplicate_chunks(chunks: list[CodeChunk]) -> list[CodeChunk]:
    """
    Remove duplicate chunks that have the same ``file_path`` and ``code``.

    The first occurrence is kept; later duplicates are dropped.  This guards
    against double-indexing when a file appears under multiple paths or a
    repo is indexed twice without clearing.

    Returns a new list; input is not modified.
    """
    seen: set[str] = set()
    result: list[CodeChunk] = []
    for chunk in chunks:
        key = _content_key(chunk)
        if key not in seen:
            seen.add(key)
            result.append(chunk)
    return result


def filter_empty_chunks(chunks: list[CodeChunk]) -> list[CodeChunk]:
    """
    Drop chunks whose ``code`` is empty or contains only whitespace.

    Returns a new list; input is not modified.
    """
    return [c for c in chunks if c.code.strip()]


def chunk_stats(chunks: Sequence[CodeChunk]) -> dict:
    """
    Return a summary statistics dictionary for a list of chunks — useful for
    logging after indexing.

    Keys
    ----
    total          — total number of chunks
    by_language    — {language_value: count} breakdown
    by_type        — {chunk_type_value: count} breakdown
    avg_chars      — mean character length of code
    max_chars      — longest chunk in characters
    min_chars      — shortest chunk in characters
    """
    if not chunks:
        return {
            "total": 0,
            "by_language": {},
            "by_type": {},
            "avg_chars": 0,
            "max_chars": 0,
            "min_chars": 0,
        }

    by_language: dict[str, int] = {}
    by_type: dict[str, int] = {}
    char_lengths: list[int] = []

    for c in chunks:
        lang = c.language if isinstance(c.language, str) else c.language.value
        by_language[lang] = by_language.get(lang, 0) + 1

        ctype = c.chunk_type if isinstance(c.chunk_type, str) else c.chunk_type.value
        by_type[ctype] = by_type.get(ctype, 0) + 1

        char_lengths.append(len(c.code))

    return {
        "total": len(chunks),
        "by_language": by_language,
        "by_type": by_type,
        "avg_chars": round(sum(char_lengths) / len(char_lengths)),
        "max_chars": max(char_lengths),
        "min_chars": min(char_lengths),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_chunk(
    chunk: CodeChunk,
    max_chars: int,
    overlap_lines: int,
) -> list[CodeChunk]:
    """
    Split a single oversized chunk into multiple sub-chunks by line boundary.

    Strategy
    --------
    1.  Split the code into lines.
    2.  Greedily accumulate lines until adding the next line would exceed
        ``max_chars``.
    3.  Emit the current window as a sub-chunk.
    4.  Back up by ``overlap_lines`` and start the next window from there.
    5.  Repeat until all lines are consumed.
    """
    lines = chunk.code.splitlines(keepends=True)
    if not lines:
        return [chunk]

    sub_chunks: list[CodeChunk] = []
    window_start = 0          # index into `lines`
    part_index = 0

    while window_start < len(lines):
        window: list[str] = []
        char_count = 0
        i = window_start

        while i < len(lines):
            line = lines[i]
            if char_count + len(line) > max_chars and window:
                # Flush current window before adding this line.
                break
            window.append(line)
            char_count += len(line)
            i += 1

        if not window:
            # Single line longer than max_chars — include it as-is.
            window = [lines[window_start]]
            i = window_start + 1

        code_text = "".join(window)
        abs_start_line = chunk.start_line + window_start
        abs_end_line = chunk.start_line + i - 1

        sub_chunks.append(
            CodeChunk(
                file_path=chunk.file_path,
                language=chunk.language,
                symbol_name=(
                    f"{chunk.symbol_name}[{part_index}]"
                    if chunk.symbol_name
                    else None
                ),
                chunk_type=ChunkType.block,   # sub-chunks are always 'block'
                code=code_text,
                start_line=abs_start_line,
                end_line=abs_end_line,
                repo_name=chunk.repo_name,
                docstring=chunk.docstring if part_index == 0 else None,
            )
        )

        # Advance, keeping `overlap_lines` for context continuity.
        window_start = max(i - overlap_lines, window_start + 1)
        part_index += 1

    return sub_chunks if sub_chunks else [chunk]


def _content_key(chunk: CodeChunk) -> str:
    """Return a stable hash key for deduplication based on file path + code."""
    digest = hashlib.sha256(
        (chunk.file_path + chunk.code).encode("utf-8", errors="replace")
    ).hexdigest()
    return digest
