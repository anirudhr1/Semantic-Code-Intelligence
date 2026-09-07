"""
file_hash_cache.py
------------------
Tracks SHA-256 content hashes per file so that incremental re-indexing can
skip unchanged files.

File layout on disk (inside index_dir)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    file_hashes_{repo_name}.json   — JSON mapping of relative_path → sha256

Public API
~~~~~~~~~~
    FileHashCache(index_dir)
    .compute_hash(file_path)            → str
    .get_changed_files(repo_name, ...)  → ChangedFiles
    .save(repo_name, hashes)
    .delete(repo_name)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ChangedFiles:
    """Result of diffing current files against the cached hashes."""

    new: list[Path] = field(default_factory=list)
    modified: list[Path] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)  # relative paths of files no longer present
    unchanged: list[Path] = field(default_factory=list)

    @property
    def files_to_process(self) -> list[Path]:
        """Files that need extraction + embedding (new + modified)."""
        return self.new + self.modified

    @property
    def has_changes(self) -> bool:
        return bool(self.new or self.modified or self.deleted)


class FileHashCache:
    """
    Manages per-file SHA-256 hashes for a repository, stored as a JSON file
    inside the index directory.

    Parameters
    ----------
    index_dir:
        Directory where hash cache JSON files are stored (same as FAISS/BM25).
    """

    def __init__(self, index_dir: Path) -> None:
        self._index_dir = Path(index_dir)

    def _cache_path(self, repo_name: str) -> Path:
        """Return the path to the hash cache file for a given repo."""
        safe_name = repo_name.replace("/", "_").replace("\\", "_")
        return self._index_dir / f"file_hashes_{safe_name}.json"

    @staticmethod
    def compute_hash(file_path: Path) -> str:
        """Compute the SHA-256 hex digest of a file's contents."""
        sha = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
        except OSError as exc:
            logger.warning("Cannot hash %s: %s", file_path, exc)
            return ""
        return sha.hexdigest()

    def load(self, repo_name: str) -> dict[str, str]:
        """Load cached hashes for a repo. Returns empty dict if no cache."""
        path = self._cache_path(repo_name)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load hash cache for '%s': %s", repo_name, exc)
            return {}

    def save(self, repo_name: str, hashes: dict[str, str]) -> None:
        """Persist the file hash mapping to disk."""
        self._index_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(repo_name)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(hashes, f, indent=2)
            logger.debug("Hash cache saved for '%s': %d files.", repo_name, len(hashes))
        except OSError as exc:
            logger.warning("Could not save hash cache for '%s': %s", repo_name, exc)

    def delete(self, repo_name: str) -> None:
        """Remove the hash cache file for a repo."""
        path = self._cache_path(repo_name)
        if path.exists():
            try:
                path.unlink()
                logger.debug("Hash cache deleted for '%s'.", repo_name)
            except OSError as exc:
                logger.warning("Could not delete hash cache for '%s': %s", repo_name, exc)

    def get_changed_files(
        self,
        repo_name: str,
        source_files: list[Path],
        repo_root: Path,
    ) -> tuple[ChangedFiles, dict[str, str]]:
        """
        Diff current source files against the cached hashes.

        Parameters
        ----------
        repo_name:
            Repository identifier.
        source_files:
            List of absolute paths to current source files.
        repo_root:
            Root directory of the repo (used to compute relative paths).

        Returns
        -------
        (ChangedFiles, new_hashes)
            The diff result and the complete new hash mapping (to be saved
            after successful indexing).
        """
        old_hashes = self.load(repo_name)
        new_hashes: dict[str, str] = {}
        result = ChangedFiles()

        # Compute hashes for all current files
        for file_path in source_files:
            rel_path = str(file_path.relative_to(repo_root))
            current_hash = self.compute_hash(file_path)
            new_hashes[rel_path] = current_hash

            if rel_path not in old_hashes:
                result.new.append(file_path)
            elif old_hashes[rel_path] != current_hash:
                result.modified.append(file_path)
            else:
                result.unchanged.append(file_path)

        # Find deleted files (in old cache but not in current files)
        current_rel_paths = set(new_hashes.keys())
        for old_rel_path in old_hashes:
            if old_rel_path not in current_rel_paths:
                result.deleted.append(old_rel_path)

        logger.info(
            "Incremental diff for '%s': %d new, %d modified, %d deleted, %d unchanged.",
            repo_name,
            len(result.new),
            len(result.modified),
            len(result.deleted),
            len(result.unchanged),
        )

        return result, new_hashes
