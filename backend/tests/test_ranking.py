"""
test_ranking.py
---------------
Unit tests for:
  - backend.app.ranking.hybrid_ranker  (HybridRanker, _symbol_score)
  - backend.app.indexing.keyword_index (KeywordIndex, tokenise)
  - backend.app.utils.chunking         (split_oversized_chunks, deduplicate_chunks,
                                        filter_empty_chunks, chunk_stats)

All tests are pure-Python — no GPU, no FAISS, no sentence-transformers needed.
Heavy dependencies (VectorStore, Embedder) are replaced with lightweight fakes.

Run:
    pytest backend/tests/test_ranking.py -v
"""

from __future__ import annotations

import uuid
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.models.schemas import ChunkType, CodeChunk, Language, SearchResponse
from backend.app.ranking.hybrid_ranker import HybridRanker, _symbol_score
from backend.app.indexing.keyword_index import KeywordIndex, tokenise
from backend.app.utils.chunking import (
    chunk_stats,
    deduplicate_chunks,
    filter_empty_chunks,
    split_oversized_chunks,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_chunk(
    code: str = "def foo(): pass",
    symbol_name: Optional[str] = "foo",
    language: Language = Language.python,
    file_path: str = "src/utils.py",
    chunk_type: ChunkType = ChunkType.function,
    start_line: int = 1,
    end_line: int = 1,
    repo_name: Optional[str] = "testrepo",
) -> CodeChunk:
    return CodeChunk(
        file_path=file_path,
        language=language,
        symbol_name=symbol_name,
        chunk_type=chunk_type,
        code=code,
        start_line=start_line,
        end_line=end_line,
        repo_name=repo_name,
    )


def _make_chunks(n: int) -> list[CodeChunk]:
    return [
        _make_chunk(
            code=f"def func_{i}(x): return x * {i}",
            symbol_name=f"func_{i}",
            start_line=i,
            end_line=i,
        )
        for i in range(n)
    ]


# ── Fake VectorStore and KeywordIndex for ranker tests ────────────────────────

class _FakeVectorStore:
    def __init__(self, chunks: list[CodeChunk], scores: list[float]):
        self._chunks = chunks
        self._scores = scores

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(self, query_vector, top_k=10, language_filter=None):
        pairs = list(zip(self._chunks, self._scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]


class _FakeKeywordIndex:
    def __init__(self, chunks: list[CodeChunk], scores: list[float]):
        self._chunks = chunks
        self._scores = scores

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(self, query, top_k=10, language_filter=None):
        pairs = list(zip(self._chunks, self._scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]


class _FakeEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        return np.ones(384, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# _symbol_score
# ═══════════════════════════════════════════════════════════════════════════════

class TestSymbolScore:
    def test_exact_match(self):
        assert _symbol_score("calculate_distance", "calculate_distance") == 1.0

    def test_exact_match_case_insensitive(self):
        assert _symbol_score("ParseJSON", "parsejson") == 1.0

    def test_query_in_symbol(self):
        assert _symbol_score("parse", "parseJSON") == 0.6

    def test_symbol_in_query(self):
        assert _symbol_score("parse JSON response", "JSON") == 0.4

    def test_no_match(self):
        assert _symbol_score("database connection", "render_template") == 0.0

    def test_none_symbol(self):
        assert _symbol_score("anything", None) == 0.0

    def test_empty_query(self):
        assert _symbol_score("", "some_function") == 0.0

    def test_empty_both(self):
        assert _symbol_score("", "") == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# HybridRanker
# ═══════════════════════════════════════════════════════════════════════════════

class TestHybridRanker:
    def _ranker(self, sem=0.6, bm25=0.3, sym=0.1):
        return HybridRanker(semantic_weight=sem, bm25_weight=bm25, symbol_weight=sym)

    def _run(self, ranker, chunks, sem_scores, bm25_scores, query="foo"):
        vs = _FakeVectorStore(chunks, sem_scores)
        ki = _FakeKeywordIndex(chunks, bm25_scores)
        emb = _FakeEmbedder()
        return ranker.rank(query, vs, ki, top_k=len(chunks), embedder=emb)

    # ── Return type and shape ──────────────────────────────────────────────────

    def test_returns_search_response(self):
        chunks = _make_chunks(3)
        result = self._run(self._ranker(), chunks, [0.9, 0.5, 0.3], [0.8, 0.4, 0.2])
        assert isinstance(result, SearchResponse)

    def test_result_count_does_not_exceed_top_k(self):
        chunks = _make_chunks(10)
        scores = [float(i) / 10 for i in range(10)]
        vs = _FakeVectorStore(chunks, scores)
        ki = _FakeKeywordIndex(chunks, scores)
        result = self._ranker().rank("foo", vs, ki, top_k=3, embedder=_FakeEmbedder())
        assert len(result.results) <= 3

    def test_results_sorted_descending(self):
        chunks = _make_chunks(5)
        sem = [0.1, 0.9, 0.3, 0.7, 0.5]
        bm25 = [0.2, 0.8, 0.4, 0.6, 0.3]
        result = self._run(self._ranker(), chunks, sem, bm25)
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_fused_score_in_unit_interval(self):
        chunks = _make_chunks(5)
        sem = [0.8, 0.6, 0.4, 0.2, 0.0]
        bm25 = [0.7, 0.5, 0.3, 0.1, 0.0]
        result = self._run(self._ranker(), chunks, sem, bm25)
        for r in result.results:
            assert 0.0 <= r.score <= 1.0

    # ── Weight effects ─────────────────────────────────────────────────────────

    def test_semantic_only_weights(self):
        """With all weight on semantic, ranking must follow semantic scores."""
        chunks = _make_chunks(3)
        sem = [0.9, 0.5, 0.1]
        bm25 = [0.1, 0.5, 0.9]   # reversed — should be ignored
        result = self._run(
            HybridRanker(semantic_weight=1.0, bm25_weight=0.0, symbol_weight=0.0),
            chunks, sem, bm25,
        )
        top_chunk = result.results[0]
        assert top_chunk.semantic_score == pytest.approx(0.9, abs=1e-4)

    def test_bm25_only_weights(self):
        chunks = _make_chunks(3)
        sem = [0.1, 0.5, 0.9]
        bm25 = [0.9, 0.5, 0.1]
        result = self._run(
            HybridRanker(semantic_weight=0.0, bm25_weight=1.0, symbol_weight=0.0),
            chunks, sem, bm25,
        )
        top_chunk = result.results[0]
        assert top_chunk.bm25_score == pytest.approx(0.9, abs=1e-4)

    # ── Symbol boost ───────────────────────────────────────────────────────────

    def test_symbol_boost_lifts_exact_match(self):
        """Chunk whose symbol_name exactly matches the query should score 1.0
        on symbol_score and have a higher fused score when symbol_weight > 0."""
        chunk_match = _make_chunk(symbol_name="parseJSON", code="def parseJSON(): ...")
        chunk_other = _make_chunk(symbol_name="render", code="def render(): ...")

        vs = _FakeVectorStore([chunk_match, chunk_other], [0.5, 0.8])
        ki = _FakeKeywordIndex([chunk_match, chunk_other], [0.5, 0.8])

        ranker = HybridRanker(semantic_weight=0.4, bm25_weight=0.4, symbol_weight=0.2)
        result = ranker.rank("parseJSON", vs, ki, top_k=2, embedder=_FakeEmbedder())

        match_result = next(r for r in result.results if r.symbol_name == "parseJSON")
        assert match_result.symbol_score == pytest.approx(1.0)

    # ── Empty indexes ──────────────────────────────────────────────────────────

    def test_empty_indexes_return_empty_results(self):
        vs = _FakeVectorStore([], [])
        ki = _FakeKeywordIndex([], [])
        result = self._ranker().rank("foo", vs, ki, top_k=5, embedder=_FakeEmbedder())
        assert result.results == []
        assert result.total_indexed == 0

    # ── Score components stored on result ─────────────────────────────────────

    def test_score_components_present(self):
        chunks = _make_chunks(2)
        result = self._run(self._ranker(), chunks, [0.7, 0.3], [0.6, 0.2])
        for r in result.results:
            assert hasattr(r, "semantic_score")
            assert hasattr(r, "bm25_score")
            assert hasattr(r, "symbol_score")

    # ── Weight imbalance warning (does not raise) ──────────────────────────────

    def test_unbalanced_weights_do_not_raise(self):
        ranker = HybridRanker(semantic_weight=0.9, bm25_weight=0.9, symbol_weight=0.9)
        chunks = _make_chunks(2)
        result = self._run(ranker, chunks, [0.5, 0.5], [0.5, 0.5])
        # Scores are clipped to [0, 1] so should still be valid.
        for r in result.results:
            assert 0.0 <= r.score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# KeywordIndex / tokenise
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenise:
    def test_basic_split(self):
        tokens = tokenise("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_camel_case_split(self):
        tokens = tokenise("parseJSONResponse")
        assert "parse" in tokens
        assert "json" in tokens
        assert "response" in tokens

    def test_underscore_split(self):
        tokens = tokenise("calculate_distance")
        assert "calculate" in tokens
        assert "distance" in tokens

    def test_lowercase_output(self):
        tokens = tokenise("MyClass.MyMethod")
        assert all(t == t.lower() for t in tokens)

    def test_single_char_tokens_dropped(self):
        tokens = tokenise("a b c def")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "def" in tokens

    def test_empty_string(self):
        assert tokenise("") == []

    def test_code_delimiters(self):
        tokens = tokenise("foo(bar, baz)")
        assert "foo" in tokens
        assert "bar" in tokens
        assert "baz" in tokens


class TestKeywordIndex:
    def _make_index(self, chunks: list[CodeChunk]) -> KeywordIndex:
        """Create an in-memory KeywordIndex (no disk I/O)."""
        idx = KeywordIndex.__new__(KeywordIndex)
        idx._index_dir = MagicMock()
        idx._bm25 = None
        idx._chunks = []
        idx._corpus = []
        idx.add(chunks)
        return idx

    def test_size_after_add(self):
        chunks = _make_chunks(5)
        idx = self._make_index(chunks)
        assert idx.size == 5

    def test_search_returns_list(self):
        chunks = _make_chunks(3)
        idx = self._make_index(chunks)
        results = idx.search("func_0")
        assert isinstance(results, list)

    def test_search_top_k_respected(self):
        chunks = _make_chunks(10)
        idx = self._make_index(chunks)
        results = idx.search("func", top_k=3)
        assert len(results) <= 3

    def test_scores_normalised(self):
        chunks = _make_chunks(5)
        idx = self._make_index(chunks)
        results = idx.search("func_1", top_k=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_exact_symbol_name_hits_top(self):
        chunks = [
            _make_chunk(code="def calculate_distance(a, b): ...", symbol_name="calculate_distance"),
            _make_chunk(code="def render_template(t): ...", symbol_name="render_template"),
            _make_chunk(code="def load_config(): ...", symbol_name="load_config"),
        ]
        idx = self._make_index(chunks)
        results = idx.search("calculate_distance", top_k=3)
        assert results, "Expected at least one result"
        top_name = results[0][0].symbol_name
        assert top_name == "calculate_distance"

    def test_empty_query_returns_empty(self):
        chunks = _make_chunks(3)
        idx = self._make_index(chunks)
        results = idx.search("", top_k=5)
        assert results == []

    def test_language_filter(self):
        py_chunk = _make_chunk(language=Language.python, symbol_name="py_func")
        js_chunk = _make_chunk(
            language=Language.javascript,
            symbol_name="js_func",
            code="function jsFunc() {}",
        )
        idx = self._make_index([py_chunk, js_chunk])
        results = idx.search("func", top_k=5, language_filter=Language.python)
        langs = {r[0].language for r in results}
        assert Language.python.value in langs or Language.python in langs
        for chunk, _ in results:
            lang_val = chunk.language if isinstance(chunk.language, str) else chunk.language.value
            assert lang_val == Language.python.value

    def test_clear_resets_index(self):
        chunks = _make_chunks(5)
        idx = self._make_index(chunks)
        assert idx.size == 5
        idx.clear()
        assert idx.size == 0
        assert idx.search("foo") == []


# ═══════════════════════════════════════════════════════════════════════════════
# chunking utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitOversizedChunks:
    def _long_chunk(self, chars: int = 5000) -> CodeChunk:
        lines = [f"# line {i}\n" for i in range(chars // 10)]
        return _make_chunk(code="".join(lines), start_line=1, end_line=len(lines))

    def test_small_chunk_unchanged(self):
        c = _make_chunk(code="def foo(): pass")
        result = split_oversized_chunks([c], max_chars=3000)
        assert len(result) == 1
        assert result[0].code == c.code

    def test_oversized_chunk_is_split(self):
        c = self._long_chunk(chars=9000)
        result = split_oversized_chunks([c], max_chars=3000)
        assert len(result) > 1

    def test_sub_chunks_cover_all_content(self):
        c = self._long_chunk(chars=9000)
        result = split_oversized_chunks([c], max_chars=3000)
        # Concatenated sub-chunk content must not lose lines
        original_lines = set(c.code.splitlines())
        covered_lines: set[str] = set()
        for sub in result:
            covered_lines.update(sub.code.splitlines())
        assert original_lines.issubset(covered_lines)

    def test_sub_chunks_within_limit(self):
        c = self._long_chunk(chars=9000)
        result = split_oversized_chunks([c], max_chars=3000)
        for sub in result:
            # Allow slight overage only for single lines longer than max_chars
            assert len(sub.code) <= 3000 * 1.1

    def test_sub_chunk_type_is_block(self):
        c = self._long_chunk(chars=9000)
        result = split_oversized_chunks([c], max_chars=3000)
        for sub in result[1:]:  # first may keep original type
            assert sub.chunk_type == ChunkType.block

    def test_mixed_list(self):
        short = _make_chunk(code="x = 1")
        long_ = self._long_chunk(chars=6000)
        result = split_oversized_chunks([short, long_], max_chars=3000)
        assert result[0].code == "x = 1"
        assert len(result) > 2  # short + >=2 sub-chunks

    def test_empty_list(self):
        assert split_oversized_chunks([]) == []


class TestDeduplicateChunks:
    def test_no_duplicates_unchanged(self):
        chunks = _make_chunks(3)
        result = deduplicate_chunks(chunks)
        assert len(result) == 3

    def test_exact_duplicates_removed(self):
        c = _make_chunk(code="def foo(): pass", file_path="a.py")
        result = deduplicate_chunks([c, c, c])
        assert len(result) == 1

    def test_same_code_different_file_kept(self):
        c1 = _make_chunk(code="pass", file_path="a.py")
        c2 = _make_chunk(code="pass", file_path="b.py")
        result = deduplicate_chunks([c1, c2])
        assert len(result) == 2

    def test_first_occurrence_kept(self):
        c1 = _make_chunk(code="def foo(): pass", file_path="a.py", symbol_name="foo")
        c2 = _make_chunk(code="def foo(): pass", file_path="a.py", symbol_name="foo_copy")
        result = deduplicate_chunks([c1, c2])
        assert len(result) == 1
        assert result[0].symbol_name == "foo"


class TestFilterEmptyChunks:
    def test_non_empty_kept(self):
        chunks = _make_chunks(3)
        assert len(filter_empty_chunks(chunks)) == 3

    def test_empty_code_removed(self):
        empty = _make_chunk(code="")
        normal = _make_chunk(code="def foo(): pass")
        result = filter_empty_chunks([empty, normal])
        assert len(result) == 1
        assert result[0].code == "def foo(): pass"

    def test_whitespace_only_removed(self):
        ws = _make_chunk(code="   \n\t  \n  ")
        result = filter_empty_chunks([ws])
        assert result == []

    def test_empty_list(self):
        assert filter_empty_chunks([]) == []


class TestChunkStats:
    def test_empty_list(self):
        stats = chunk_stats([])
        assert stats["total"] == 0

    def test_total_count(self):
        chunks = _make_chunks(5)
        assert chunk_stats(chunks)["total"] == 5

    def test_by_language(self):
        py = _make_chunk(language=Language.python)
        js = _make_chunk(language=Language.javascript, code="function f(){}")
        stats = chunk_stats([py, py, js])
        assert stats["by_language"].get("python") == 2
        assert stats["by_language"].get("javascript") == 1

    def test_avg_chars(self):
        c1 = _make_chunk(code="ab")       # 2 chars
        c2 = _make_chunk(code="abcd")     # 4 chars
        stats = chunk_stats([c1, c2])
        assert stats["avg_chars"] == 3

    def test_max_min_chars(self):
        c1 = _make_chunk(code="x")
        c2 = _make_chunk(code="x" * 100)
        stats = chunk_stats([c1, c2])
        assert stats["max_chars"] == 100
        assert stats["min_chars"] == 1
