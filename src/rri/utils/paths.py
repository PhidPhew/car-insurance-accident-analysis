"""Generic, environment-agnostic path helpers.

This module has no knowledge of the project's specific data layout — that
lives in ``config/settings.py``. It only knows how to locate the repo root
and create directories on demand, so it has no dependency on ``Settings``
and can be imported from anywhere without a circular import.
"""

from __future__ import annotations

from pathlib import Path

_ROOT_MARKERS = ("pyproject.toml", ".git")


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` (default: this file) until a root marker is found.

    A root marker is any file/directory in ``_ROOT_MARKERS``. Raises
    ``FileNotFoundError`` if no marker is found before hitting the filesystem root,
    since that indicates the project layout is not what the codebase expects.
    """
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    raise FileNotFoundError(
        f"Could not find project root (looked for {_ROOT_MARKERS}) above {current}"
    )


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing, and return it unchanged."""
    path.mkdir(parents=True, exist_ok=True)
    return path
