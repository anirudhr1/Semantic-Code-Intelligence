"""
build_index.py
--------------
CLI tool for indexing a repository without starting the API server.

Useful for pre-building the index in CI/CD pipelines, Docker build steps,
or one-off indexing jobs before deploying the API.

Usage examples
~~~~~~~~~~~~~~
Index a local directory:
    python -m backend.scripts.build_index --path ./data/repos/my-project

Index a remote git repository:
    python -m backend.scripts.build_index --repo-url https://github.com/owner/repo

Restrict to a single language:
    python -m backend.scripts.build_index --path ./data/repos/my-project --language python

Use a custom embedding model:
    python -m backend.scripts.build_index --path ./my-repo --model microsoft/graphcodebert-base

Clear existing index before building:
    python -m backend.scripts.build_index --path ./my-repo --clear

Run from the workspace root so that relative paths in .env are resolved correctly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Make sure the workspace root is on sys.path when invoked directly ─────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.app.config import settings
from backend.app.embeddings.embedder import get_embedder
from backend.app.indexing.keyword_index import KeywordIndex
from backend.app.indexing.vector_store import VectorStore
from backend.app.ingestion.ast_extractor import extract_chunks
from backend.app.ingestion.repo_loader import load_repo
from backend.app.models.schemas import Language
from backend.app.utils.chunking import (
    chunk_stats,
    deduplicate_chunks,
    filter_empty_chunks,
    split_oversized_chunks,
)

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_index",
        description="Index a source-code repository into FAISS + BM25.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--path",
        metavar="DIR",
        help="Local filesystem path to the repository to index.",
    )
    source.add_argument(
        "--repo-url",
        metavar="URL",
        help="Remote git URL; the repo will be cloned into REPOS_DIR.",
    )

    parser.add_argument(
        "--language",
        metavar="LANG",
        choices=[lang.value for lang in Language if lang != Language.unknown],
        default=None,
        help="Only index files of this language (default: all supported languages).",
    )
    parser.add_argument(
        "--repo-name",
        metavar="NAME",
        default=None,
        help="Human-readable name stored on each chunk. Defaults to directory name.",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default=settings.embedding_model,
        help=f"sentence-transformers model (default: {settings.embedding_model}).",
    )
    parser.add_argument(
        "--index-dir",
        metavar="DIR",
        default=str(settings.index_dir),
        help=f"Where to persist the index (default: {settings.index_dir}).",
    )
    parser.add_argument(
        "--repos-dir",
        metavar="DIR",
        default=str(settings.repos_dir),
        help=f"Where remote repos are cloned (default: {settings.repos_dir}).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        default=False,
        help="Clear the existing index before building.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        default=False,
        help="Skip chunk deduplication (faster, but may index duplicate code).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    return parser


# ── Main logic ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    t_start = time.perf_counter()

    index_dir = Path(args.index_dir)
    repos_dir = Path(args.repos_dir)
    language = Language(args.language) if args.language else None

    # ── 1. Initialise stores ──────────────────────────────────────────────────
    logger.info("Initialising indexes at %s …", index_dir)
    vector_store = VectorStore(index_dir=index_dir, model_name=args.model)
    keyword_index = KeywordIndex(index_dir=index_dir)

    if args.clear:
        logger.info("--clear flag set: wiping existing index.")
        vector_store.clear()
        keyword_index.clear()

    # ── 2. Load repo ──────────────────────────────────────────────────────────
    logger.info("Resolving repository …")
    try:
        repo_result = load_repo(
            path=args.path,
            repo_url=args.repo_url,
            repos_dir=repos_dir,
            language=language,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        logger.error("Failed to load repository: %s", exc)
        return 1

    repo_name = args.repo_name or repo_result.repo_name
    logger.info(
        "Repo '%s': %d source files (skipped %d non-matching).",
        repo_name,
        repo_result.file_count,
        repo_result.skipped_files,
    )

    if repo_result.file_count == 0:
        logger.error("No source files found. Check --path / --language.")
        return 1

    # ── 3. Extract chunks ─────────────────────────────────────────────────────
    logger.info("Extracting AST chunks …")
    all_chunks = []
    extraction_errors = 0

    for file_path in repo_result.source_files:
        rel_path = str(file_path.relative_to(repo_result.repo_root))
        try:
            chunks = extract_chunks(file_path, repo_name=repo_name, language=language)
            for c in chunks:
                c.file_path = rel_path
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.debug("Skipping %s: %s", rel_path, exc)
            extraction_errors += 1

    logger.info(
        "Raw extraction: %d chunks from %d files (%d files skipped due to errors).",
        len(all_chunks),
        repo_result.file_count,
        extraction_errors,
    )

    # ── 4. Post-process chunks ────────────────────────────────────────────────
    all_chunks = filter_empty_chunks(all_chunks)
    all_chunks = split_oversized_chunks(all_chunks)

    if not args.no_dedup:
        before = len(all_chunks)
        all_chunks = deduplicate_chunks(all_chunks)
        dropped = before - len(all_chunks)
        if dropped:
            logger.info("Deduplication removed %d duplicate chunks.", dropped)

    if not all_chunks:
        logger.error("No chunks remain after post-processing. Aborting.")
        return 1

    stats = chunk_stats(all_chunks)
    logger.info(
        "Chunks ready: %d total | avg %d chars | by language: %s | by type: %s",
        stats["total"],
        stats["avg_chars"],
        stats["by_language"],
        stats["by_type"],
    )

    # ── 5. Embed ──────────────────────────────────────────────────────────────
    logger.info("Loading embedding model '%s' …", args.model)
    try:
        embedder = get_embedder(args.model)
        logger.info("Embedding %d chunks (this may take a while) …", len(all_chunks))
        embeddings = embedder.encode_chunks(all_chunks, show_progress=True)
    except Exception as exc:
        logger.exception("Embedding failed: %s", exc)
        return 1

    # ── 6. Insert into indexes ────────────────────────────────────────────────
    logger.info("Inserting into FAISS vector store …")
    try:
        vector_store.add(all_chunks, embeddings)
    except Exception as exc:
        logger.exception("FAISS insertion failed: %s", exc)
        return 1

    logger.info("Inserting into BM25 keyword index …")
    try:
        keyword_index.add(all_chunks)
    except Exception as exc:
        logger.exception("BM25 insertion failed: %s", exc)
        return 1

    # ── 7. Persist ────────────────────────────────────────────────────────────
    logger.info("Persisting indexes to %s …", index_dir)
    vector_store.save()
    keyword_index.save()

    elapsed = round(time.perf_counter() - t_start, 1)
    logger.info(
        "Done. Indexed %d chunks from '%s' in %.1fs.",
        len(all_chunks),
        repo_name,
        elapsed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
