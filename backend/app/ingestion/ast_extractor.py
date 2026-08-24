"""
ast_extractor.py
----------------
Uses tree-sitter to parse source files into CodeChunk objects.

Each CodeChunk corresponds to a top-level or class-level definition
(function, method, class) or, when no definitions are found, the entire file
is returned as a single module-level chunk.

Public API
~~~~~~~~~~
    extract_chunks(file_path, repo_name, language) -> list[CodeChunk]
    extract_chunks_from_source(source, language, file_path, repo_name) -> list[CodeChunk]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from backend.app.models.schemas import ChunkType, CodeChunk, Language
from backend.app.ingestion.repo_loader import detect_language

logger = logging.getLogger(__name__)

# ── tree-sitter lazy imports ──────────────────────────────────────────────────
# We import lazily so that the module can be imported even in environments
# where tree-sitter language bindings are not installed (tests can mock them).

def _get_parser(language: Language):  # type: ignore[return]
    """Return a tree-sitter Parser configured for the given language."""
    try:
        from tree_sitter import Language as TSLanguage, Parser

        lang_module_map = {
            Language.python: ("tree_sitter_python", "python"),
            Language.javascript: ("tree_sitter_javascript", "javascript"),
            Language.typescript: ("tree_sitter_typescript", "typescript"),
            Language.go: ("tree_sitter_go", "go"),
            Language.rust: ("tree_sitter_rust", "rust"),
            Language.java: ("tree_sitter_java", "java"),
        }
        entry = lang_module_map.get(language)
        if entry is None:
            return None

        module_name, lang_name = entry
        import importlib
        lang_module = importlib.import_module(module_name)
        ts_language = TSLanguage(lang_module.language(), lang_name)
        parser = Parser()
        parser.set_language(ts_language)
        return parser
    except Exception as exc:  # pragma: no cover
        logger.warning("tree-sitter not available for %s: %s", language, exc)
        return None


# ── Node-type → ChunkType mapping per language ────────────────────────────────

_NODE_TYPE_MAP: dict[Language, dict[str, ChunkType]] = {
    Language.python: {
        "function_definition": ChunkType.function,
        "async_function_definition": ChunkType.function,
        "class_definition": ChunkType.class_,
    },
    Language.javascript: {
        "function_declaration": ChunkType.function,
        "arrow_function": ChunkType.function,
        "method_definition": ChunkType.method,
        "class_declaration": ChunkType.class_,
    },
    Language.typescript: {
        "function_declaration": ChunkType.function,
        "arrow_function": ChunkType.function,
        "method_definition": ChunkType.method,
        "class_declaration": ChunkType.class_,
    },
    Language.go: {
        "function_declaration": ChunkType.function,
        "method_declaration": ChunkType.method,
        "type_declaration": ChunkType.class_,
    },
    Language.rust: {
        "function_item": ChunkType.function,
        "impl_item": ChunkType.class_,
        "struct_item": ChunkType.class_,
        "enum_item": ChunkType.class_,
    },
    Language.java: {
        "method_declaration": ChunkType.method,
        "class_declaration": ChunkType.class_,
        "interface_declaration": ChunkType.class_,
    },
}

# ── Symbol-name extraction per language ───────────────────────────────────────

def _extract_symbol_name(node, language: Language) -> Optional[str]:
    """
    Heuristically extract the symbol name from a tree-sitter node.
    Looks for a child node whose type is 'identifier' or 'name'.
    """
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return child.text.decode("utf-8", errors="replace")
    return None


def _extract_docstring(node, source_bytes: bytes, language: Language) -> Optional[str]:
    """
    Extract the first docstring or leading block comment from a definition node.
    Returns plain text (whitespace-normalised) or None.
    """
    if language == Language.python:
        # First child of the body block that is an expression_statement
        # containing a string.
        for child in node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for sub in stmt.children:
                            if sub.type == "string":
                                raw = source_bytes[sub.start_byte:sub.end_byte]
                                text = raw.decode("utf-8", errors="replace").strip("'\"")
                                return re.sub(r"\s+", " ", text).strip()
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def extract_chunks(
    file_path: Path,
    *,
    repo_name: Optional[str] = None,
    language: Optional[Language] = None,
) -> list[CodeChunk]:
    """
    Parse a source file and return a list of CodeChunk objects.

    Parameters
    ----------
    file_path:  Absolute path to the source file.
    repo_name:  Optional repo identifier stored on each chunk.
    language:   Override language detection (defaults to extension-based detection).
    """
    try:
        source = file_path.read_bytes()
    except OSError as exc:
        logger.warning("Cannot read %s: %s", file_path, exc)
        return []

    detected = language or detect_language(file_path)
    return extract_chunks_from_source(
        source=source,
        language=detected,
        file_path=str(file_path),
        repo_name=repo_name,
    )


def extract_chunks_from_source(
    source: bytes,
    language: Language,
    *,
    file_path: str = "<unknown>",
    repo_name: Optional[str] = None,
) -> list[CodeChunk]:
    """
    Parse raw source bytes and return a list of CodeChunk objects.
    Falls back to a single whole-file chunk if tree-sitter is unavailable
    or produces no recognised nodes.
    """
    parser = _get_parser(language)
    if parser is None:
        return _fallback_whole_file(source, language, file_path, repo_name)

    try:
        tree = parser.parse(source)
    except Exception as exc:
        logger.warning("tree-sitter parse error for %s: %s", file_path, exc)
        return _fallback_whole_file(source, language, file_path, repo_name)

    node_map = _NODE_TYPE_MAP.get(language, {})
    chunks: list[CodeChunk] = []

    _walk_tree(
        node=tree.root_node,
        source_bytes=source,
        node_map=node_map,
        language=language,
        file_path=file_path,
        repo_name=repo_name,
        chunks=chunks,
        depth=0,
    )

    if not chunks:
        return _fallback_whole_file(source, language, file_path, repo_name)

    return chunks


# ── Tree walker ───────────────────────────────────────────────────────────────

_MAX_DEPTH = 6  # Avoid runaway recursion on deeply nested code


def _walk_tree(
    node,
    source_bytes: bytes,
    node_map: dict[str, ChunkType],
    language: Language,
    file_path: str,
    repo_name: Optional[str],
    chunks: list[CodeChunk],
    depth: int,
) -> None:
    if depth > _MAX_DEPTH:
        return

    chunk_type = node_map.get(node.type)
    if chunk_type is not None:
        code_text = source_bytes[node.start_byte:node.end_byte].decode(
            "utf-8", errors="replace"
        )
        symbol_name = _extract_symbol_name(node, language)
        docstring = _extract_docstring(node, source_bytes, language)

        # line numbers are 0-indexed in tree-sitter; convert to 1-indexed
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        chunks.append(
            CodeChunk(
                file_path=file_path,
                language=language,
                symbol_name=symbol_name,
                chunk_type=chunk_type,
                code=code_text,
                start_line=start_line,
                end_line=end_line,
                repo_name=repo_name,
                docstring=docstring,
            )
        )
        # Still recurse to catch nested classes / inner functions
        depth += 1

    for child in node.children:
        _walk_tree(
            node=child,
            source_bytes=source_bytes,
            node_map=node_map,
            language=language,
            file_path=file_path,
            repo_name=repo_name,
            chunks=chunks,
            depth=depth,
        )


# ── Fallback ──────────────────────────────────────────────────────────────────

def _fallback_whole_file(
    source: bytes,
    language: Language,
    file_path: str,
    repo_name: Optional[str],
) -> list[CodeChunk]:
    """Return the entire file as a single module-level CodeChunk."""
    text = source.decode("utf-8", errors="replace")
    line_count = max(text.count("\n"), 1)
    return [
        CodeChunk(
            file_path=file_path,
            language=language,
            symbol_name=None,
            chunk_type=ChunkType.module,
            code=text,
            start_line=1,
            end_line=line_count,
            repo_name=repo_name,
        )
    ]
