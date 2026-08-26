"""
test_ast_extractor.py
---------------------
Unit tests for backend.app.ingestion.ast_extractor and the helper utilities
it depends on (repo_loader.detect_language, chunking helpers).

These tests do NOT require tree-sitter to be installed — the extractor falls
back to whole-file chunks when the parser is unavailable, and we test that
path explicitly.  Where tree-sitter IS available its output is also verified.

Run:
    pytest backend/tests/test_ast_extractor.py -v
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.app.ingestion.ast_extractor import (
    _fallback_whole_file,
    extract_chunks_from_source,
)
from backend.app.ingestion.repo_loader import detect_language
from backend.app.models.schemas import ChunkType, CodeChunk, Language


# ── Fixtures ──────────────────────────────────────────────────────────────────

PYTHON_SOURCE = textwrap.dedent("""\
    def add(a, b):
        \"\"\"Return the sum of a and b.\"\"\"
        return a + b


    class Calculator:
        def multiply(self, x, y):
            return x * y
""").encode()

JS_SOURCE = textwrap.dedent("""\
    function greet(name) {
        return `Hello, ${name}!`;
    }

    class Animal {
        constructor(name) {
            this.name = name;
        }
    }
""").encode()

EMPTY_SOURCE = b""
WHITESPACE_SOURCE = b"   \n\t\n  "


# ── detect_language ───────────────────────────────────────────────────────────

class TestDetectLanguage:
    def test_python_extension(self):
        assert detect_language(Path("foo/bar.py")) == Language.python

    def test_javascript_extension(self):
        assert detect_language(Path("src/app.js")) == Language.javascript

    def test_typescript_extension(self):
        assert detect_language(Path("src/main.ts")) == Language.typescript

    def test_go_extension(self):
        assert detect_language(Path("cmd/main.go")) == Language.go

    def test_rust_extension(self):
        assert detect_language(Path("src/lib.rs")) == Language.rust

    def test_java_extension(self):
        assert detect_language(Path("Main.java")) == Language.java

    def test_unknown_extension(self):
        assert detect_language(Path("README.md")) == Language.unknown

    def test_case_insensitive(self):
        assert detect_language(Path("script.PY")) == Language.python


# ── _fallback_whole_file ──────────────────────────────────────────────────────

class TestFallbackWholeFile:
    def test_returns_single_chunk(self):
        chunks = _fallback_whole_file(PYTHON_SOURCE, Language.python, "test.py", "myrepo")
        assert len(chunks) == 1

    def test_chunk_fields(self):
        chunks = _fallback_whole_file(PYTHON_SOURCE, Language.python, "src/utils.py", "repo")
        c = chunks[0]
        assert c.chunk_type == ChunkType.module
        assert c.language == Language.python
        assert c.file_path == "src/utils.py"
        assert c.repo_name == "repo"
        assert c.start_line == 1
        assert c.end_line >= 1

    def test_empty_source(self):
        chunks = _fallback_whole_file(EMPTY_SOURCE, Language.python, "empty.py", None)
        assert len(chunks) == 1
        assert chunks[0].end_line >= 1  # at least 1 even for empty file

    def test_code_matches_source(self):
        chunks = _fallback_whole_file(PYTHON_SOURCE, Language.python, "x.py", None)
        assert chunks[0].code == PYTHON_SOURCE.decode()


# ── extract_chunks_from_source (no tree-sitter) ───────────────────────────────

class TestExtractChunksNoParser:
    """When _get_parser returns None the extractor must fall back gracefully."""

    def _extract(self, source: bytes, language: Language) -> list[CodeChunk]:
        with patch(
            "backend.app.ingestion.ast_extractor._get_parser", return_value=None
        ):
            return extract_chunks_from_source(
                source, language, file_path="test.py", repo_name="repo"
            )

    def test_fallback_for_unknown_language(self):
        chunks = self._extract(PYTHON_SOURCE, Language.unknown)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.module

    def test_fallback_returns_full_code(self):
        chunks = self._extract(PYTHON_SOURCE, Language.python)
        assert chunks[0].code == PYTHON_SOURCE.decode()

    def test_fallback_for_empty_source(self):
        chunks = self._extract(EMPTY_SOURCE, Language.python)
        assert len(chunks) == 1

    def test_whitespace_only_source(self):
        chunks = self._extract(WHITESPACE_SOURCE, Language.python)
        assert len(chunks) == 1


# ── extract_chunks_from_source (with tree-sitter mock) ───────────────────────

class TestExtractChunksWithMockedParser:
    """
    Simulate what tree-sitter would return so we can test the AST-walk logic
    without installing any native extensions.
    """

    def _make_node(self, node_type: str, text: bytes, start: tuple, end: tuple, children=None):
        node = MagicMock()
        node.type = node_type
        node.text = text
        node.start_byte = 0
        node.end_byte = len(text)
        node.start_point = start
        node.end_point = end
        node.children = children or []
        return node

    def _make_identifier(self, name: str):
        node = MagicMock()
        node.type = "identifier"
        node.text = name.encode()
        node.children = []
        return node

    def test_function_node_produces_function_chunk(self):
        source = b"def add(a, b):\n    return a + b\n"
        identifier = self._make_identifier("add")
        func_node = self._make_node(
            "function_definition",
            source,
            (0, 0),
            (1, 18),
            children=[identifier],
        )
        root = self._make_node("module", source, (0, 0), (1, 18), children=[func_node])

        mock_tree = MagicMock()
        mock_tree.root_node = root

        mock_parser = MagicMock()
        mock_parser.parse.return_value = mock_tree

        with patch(
            "backend.app.ingestion.ast_extractor._get_parser",
            return_value=mock_parser,
        ):
            chunks = extract_chunks_from_source(
                source, Language.python, file_path="test.py", repo_name="r"
            )

        assert len(chunks) >= 1
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.function]
        assert func_chunks, "Expected at least one function chunk"
        assert func_chunks[0].symbol_name == "add"
        assert func_chunks[0].start_line == 1   # tree-sitter 0-indexed → 1-indexed

    def test_class_node_produces_class_chunk(self):
        source = b"class Foo:\n    pass\n"
        identifier = self._make_identifier("Foo")
        class_node = self._make_node(
            "class_definition",
            source,
            (0, 0),
            (1, 8),
            children=[identifier],
        )
        root = self._make_node("module", source, (0, 0), (1, 8), children=[class_node])

        mock_tree = MagicMock()
        mock_tree.root_node = root

        mock_parser = MagicMock()
        mock_parser.parse.return_value = mock_tree

        with patch(
            "backend.app.ingestion.ast_extractor._get_parser",
            return_value=mock_parser,
        ):
            chunks = extract_chunks_from_source(
                source, Language.python, file_path="test.py", repo_name="r"
            )

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.class_]
        assert class_chunks
        assert class_chunks[0].symbol_name == "Foo"

    def test_parse_error_falls_back_to_whole_file(self):
        mock_parser = MagicMock()
        mock_parser.parse.side_effect = RuntimeError("parse error")

        with patch(
            "backend.app.ingestion.ast_extractor._get_parser",
            return_value=mock_parser,
        ):
            chunks = extract_chunks_from_source(
                PYTHON_SOURCE, Language.python, file_path="bad.py"
            )

        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.module


# ── CodeChunk model validation ────────────────────────────────────────────────

class TestCodeChunkValidation:
    def test_valid_chunk(self):
        c = CodeChunk(
            file_path="src/main.py",
            language=Language.python,
            chunk_type=ChunkType.function,
            code="def foo(): pass",
            start_line=1,
            end_line=1,
        )
        assert c.start_line == 1

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="end_line"):
            CodeChunk(
                file_path="x.py",
                language=Language.python,
                chunk_type=ChunkType.function,
                code="...",
                start_line=10,
                end_line=5,
            )

    def test_chunk_id_auto_assigned(self):
        c = CodeChunk(
            file_path="x.py",
            language=Language.python,
            chunk_type=ChunkType.function,
            code="def f(): pass",
            start_line=1,
            end_line=1,
        )
        assert c.chunk_id is not None

    def test_two_chunks_have_different_ids(self):
        kwargs = dict(
            file_path="x.py",
            language=Language.python,
            chunk_type=ChunkType.function,
            code="def f(): pass",
            start_line=1,
            end_line=1,
        )
        c1 = CodeChunk(**kwargs)
        c2 = CodeChunk(**kwargs)
        assert c1.chunk_id != c2.chunk_id
