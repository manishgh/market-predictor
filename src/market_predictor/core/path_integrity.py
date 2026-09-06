"""Cross-platform path-containment and reparse-point checks."""

from __future__ import annotations

import os
from pathlib import Path

from market_predictor.core.errors import DataReadinessError

_WINDOWS_REPARSE_POINT = 0x400


def is_reparse_point(path: Path) -> bool:
    """Return whether a path is a symlink or Windows reparse point."""

    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(os.lstat(path), "st_file_attributes", 0))
    except OSError:
        return False
    flag = int(
        getattr(os.stat, "FILE_ATTRIBUTE_REPARSE_POINT", _WINDOWS_REPARSE_POINT)
    )
    return bool(attributes & flag)


def verify_no_reparse_ancestry(path: Path, *, label: str) -> Path:
    """Return a lexical absolute path after checking every existing component."""

    absolute = Path(os.path.abspath(path))
    components = (*reversed(absolute.parents), absolute)
    for component in components:
        if is_reparse_point(component):
            raise DataReadinessError(
                f"{label} path contains a symlink or reparse point"
            )
        if not component.exists():
            break
    return absolute


def verify_tree_containment(root: Path, *, label: str) -> Path:
    """Verify an existing directory tree contains no link-like escape points."""

    root = verify_no_reparse_ancestry(root, label=label)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise DataReadinessError(f"{label} is unavailable") from exc
    if not resolved_root.is_dir():
        raise DataReadinessError(f"{label} is not a directory")
    try:
        descendants = tuple(root.rglob("*"))
        for path in descendants:
            if is_reparse_point(path):
                raise DataReadinessError(
                    f"{label} contains a symlink or reparse point"
                )
            if not path.resolve(strict=True).is_relative_to(resolved_root):
                raise DataReadinessError(f"{label} escapes its authority root")
    except OSError as exc:
        raise DataReadinessError(f"{label} inventory is unreadable") from exc
    return resolved_root


def resolve_existing_file_inside(root: Path, relative: str, *, label: str) -> Path:
    """Resolve one existing regular file without crossing a reparse point."""

    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or "." in relative_path.parts
        or ".." in relative_path.parts
    ):
        raise DataReadinessError(f"{label} path is unsafe")
    root = verify_no_reparse_ancestry(root, label=label)
    resolved_root = root.resolve(strict=True)
    current = root
    for part in relative_path.parts:
        current = current / part
        if is_reparse_point(current):
            raise DataReadinessError(
                f"{label} path contains a symlink or reparse point"
            )
    try:
        candidate = current.resolve(strict=True)
    except OSError as exc:
        raise DataReadinessError(f"{label} is missing") from exc
    if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
        raise DataReadinessError(f"{label} escapes its authority root")
    return candidate


def resolve_write_path_inside(
    root: Path,
    relative: Path,
    *,
    label: str,
) -> Path:
    """Resolve a prospective child path without crossing existing reparse points."""

    if (
        relative.is_absolute()
        or not relative.parts
        or "." in relative.parts
        or ".." in relative.parts
    ):
        raise DataReadinessError(f"{label} path is unsafe")
    root = verify_no_reparse_ancestry(root, label=label)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise DataReadinessError(f"{label} root is unavailable") from exc
    if not resolved_root.is_dir():
        raise DataReadinessError(f"{label} root is not a directory")
    current = root
    for part in relative.parts:
        current = current / part
        if is_reparse_point(current):
            raise DataReadinessError(
                f"{label} path contains a symlink or reparse point"
            )
    candidate = current.resolve(strict=False)
    if not candidate.is_relative_to(resolved_root):
        raise DataReadinessError(f"{label} path escapes its authority root")
    return candidate
