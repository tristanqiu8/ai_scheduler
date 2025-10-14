#!/usr/bin/env python3
"""Artifact path utilities ensuring outputs stay under a common directory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

ARTIFACTS_ENV_VAR = "AI_SCHEDULER_ARTIFACTS_DIR"
DEFAULT_ARTIFACTS_DIR = Path("artifacts_sim")

_ARTIFACTS_ROOT: Optional[Path] = None


def _canonicalize_root(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def get_artifacts_root() -> Path:
    """Return the absolute artifacts root, creating it on first use."""

    global _ARTIFACTS_ROOT
    if _ARTIFACTS_ROOT is None:
        configured = Path(os.getenv(ARTIFACTS_ENV_VAR, str(DEFAULT_ARTIFACTS_DIR)))
        _ARTIFACTS_ROOT = _canonicalize_root(configured)
        _ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    return _ARTIFACTS_ROOT


def _normalize_subpath(path: Path, root: Path) -> Path:
    parts = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":  # prevent escaping the root
            if parts:
                parts.pop()
            continue
        if not parts and part == root.name:
            # Treat a leading root name as referring to the root itself
            continue
        parts.append(part)
    return Path(*parts)


def resolve_artifact_path(*subpaths: Union[str, os.PathLike], ensure_parent: bool = True) -> Path:
    """Resolve a path under the artifacts root, avoiding nested root prefixes."""

    root = get_artifacts_root()

    if not subpaths:
        path = root
    else:
        combined = Path()
        for sub in subpaths:
            combined /= Path(sub)

        if combined.is_absolute():
            path = combined.expanduser()
        else:
            normalized = _normalize_subpath(combined, root)
            path = root.joinpath(normalized)

    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


def ensure_artifact_path(path: Union[str, os.PathLike], ensure_parent: bool = True) -> Path:
    """Return a path guaranteed to sit beneath the artifacts root."""

    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        target = path_obj
    else:
        target = resolve_artifact_path(path_obj, ensure_parent=False)

    if ensure_parent:
        target.parent.mkdir(parents=True, exist_ok=True)

    return target
