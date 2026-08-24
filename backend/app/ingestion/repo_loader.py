"""
repo_loader.py
--------------
Resolves a source location (local path or remote git URL) to a flat list of
source-file paths that the AST extractor can process.

Public API
~~~~~~~~~~
    load_repo(path, repo_url, repos_dir, language) -> RepoLoadResult
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import git  # gitpython

from backend.app.models.schemas import Language

logger = logging.getLogger(__name__)

# ── Language ↔ file extension mapping ────────────────────────────────────────

LANGUAGE_EXTENSIONS: dict[Language, list[str]] = {
    Language.python: [".py"],
    Language.javascript: [".js", ".mjs", ".cjs"],
    Language.typescript: [".ts", ".tsx"],
    Language.go: [".go"],
    Language.rust: [".rs"],
    Language.java: [".java"],
}

EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    ext: lang
    for lang, exts in LANGUAGE_EXTENSIONS.items()
    for ext in exts
}

# Directories that are never worth indexing
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".tox",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "target",        # Rust / Java
        "vendor",        # Go
    }
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RepoLoadResult:
    """Outcome of a load_repo call."""

    repo_name: str
    repo_root: Path
    source_files: list[Path] = field(default_factory=list)
    skipped_files: int = 0
    cloned: bool = False

    @property
    def file_count(self) -> int:
        return len(self.source_files)


# ── Public entry-point ────────────────────────────────────────────────────────

def load_repo(
    *,
    path: Optional[str] = None,
    repo_url: Optional[str] = None,
    repos_dir: Path,
    language: Optional[Language] = None,
) -> RepoLoadResult:
    """
    Resolve a repo source and return a RepoLoadResult with all matching files.

    Parameters
    ----------
    path:       Local filesystem path to an already-cloned repo.
    repo_url:   Remote git URL; will be cloned into `repos_dir` if not already present.
    repos_dir:  Base directory where clones are stored.
    language:   When provided, only files for that language are returned.
                When None, all recognised source files are returned.
    """
    if not path and not repo_url:
        raise ValueError("Provide either 'path' or 'repo_url'.")
    if path and repo_url:
        raise ValueError("Provide only one of 'path' or 'repo_url'.")

    cloned = False

    if repo_url:
        repo_root, cloned = _ensure_cloned(repo_url, repos_dir)
    else:
        repo_root = Path(path).expanduser().resolve()  # type: ignore[arg-type]
        if not repo_root.exists():
            raise FileNotFoundError(f"Path does not exist: {repo_root}")
        if not repo_root.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {repo_root}")

    repo_name = _derive_repo_name(repo_url or str(repo_root))
    source_files, skipped = _collect_files(repo_root, language)

    logger.info(
        "Repo '%s': %d source files found, %d skipped.",
        repo_name,
        len(source_files),
        skipped,
    )

    return RepoLoadResult(
        repo_name=repo_name,
        repo_root=repo_root,
        source_files=source_files,
        skipped_files=skipped,
        cloned=cloned,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_cloned(repo_url: str, repos_dir: Path) -> tuple[Path, bool]:
    """
    Clone `repo_url` into `repos_dir/<repo_name>` if the directory does not
    already exist.  Returns (clone_path, was_freshly_cloned).
    """
    repos_dir.mkdir(parents=True, exist_ok=True)
    repo_name = _derive_repo_name(repo_url)
    clone_path = repos_dir / repo_name

    if clone_path.exists():
        logger.info("Repo '%s' already cloned at %s — skipping clone.", repo_name, clone_path)
        return clone_path, False

    logger.info("Cloning %s → %s …", repo_url, clone_path)
    try:
        git.Repo.clone_from(repo_url, str(clone_path), depth=1)
    except git.GitCommandError as exc:
        # Clean up a partial clone so future retries start fresh.
        if clone_path.exists():
            shutil.rmtree(clone_path, ignore_errors=True)
        raise RuntimeError(f"git clone failed: {exc}") from exc

    return clone_path, True


def _derive_repo_name(source: str) -> str:
    """
    Turn a URL or filesystem path into a short, filesystem-safe repo name.

    Examples
    --------
    https://github.com/owner/my-repo.git  →  my-repo
    /home/user/projects/my-project        →  my-project
    """
    # Try URL path last segment
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https", "git", "ssh"):
        name = parsed.path.rstrip("/").split("/")[-1]
        name = re.sub(r"\.git$", "", name)
    else:
        name = Path(source).name

    # Sanitise to alphanumeric + hyphens/underscores
    name = re.sub(r"[^\w\-]", "_", name)
    return name or "repo"


def _collect_files(
    root: Path,
    language: Optional[Language],
) -> tuple[list[Path], int]:
    """
    Walk the directory tree and return (matching_files, skipped_count).

    A file is 'skipped' if it has an unrecognised extension OR belongs to a
    directory in _SKIP_DIRS.
    """
    wanted_extensions: Optional[frozenset[str]] = None
    if language is not None:
        exts = LANGUAGE_EXTENSIONS.get(language)
        if exts:
            wanted_extensions = frozenset(exts)

    matched: list[Path] = []
    skipped = 0

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Prune ignored directories in-place (modifies os.walk's traversal).
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

        for filename in filenames:
            full_path = Path(dirpath) / filename
            suffix = full_path.suffix.lower()

            if wanted_extensions is not None:
                if suffix not in wanted_extensions:
                    skipped += 1
                    continue
            else:
                if suffix not in EXTENSION_TO_LANGUAGE:
                    skipped += 1
                    continue

            matched.append(full_path)

    # Stable ordering for reproducible indexing
    matched.sort()
    return matched, skipped


def detect_language(file_path: Path) -> Language:
    """Return the Language enum value for a given file path, or Language.unknown."""
    return EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower(), Language.unknown)
