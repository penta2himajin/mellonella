"""Shared helpers for dataset downloaders."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def default_data_dir() -> Path:
    """Return the directory where datasets are cached.

    Honors ``$MELLONELLA_DATA_DIR``; defaults to ``./data`` relative to the
    repository root.
    """
    raw = os.environ.get("MELLONELLA_DATA_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd() / "data"


@dataclass(frozen=True)
class DatasetSpec:
    """Minimal description of a downloadable dataset."""

    name: str
    url: str
    archive_sha256: str
    license: str
    notes: str = ""


def sha256_of(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream-hash ``path`` and return the hex digest."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, expected_sha256: str) -> None:
    """Raise :class:`RuntimeError` when ``path`` doesn't match the expected hash."""
    actual = sha256_of(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"sha256 mismatch for {path}\n  expected: {expected_sha256}\n  actual:   {actual}"
        )


def ensure_clean_dir(path: Path) -> None:
    """Create ``path`` empty, removing it first if it already exists."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
