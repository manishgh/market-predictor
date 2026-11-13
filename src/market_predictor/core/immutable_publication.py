"""Publish staged files atomically without replacing an existing artifact."""
from __future__ import annotations

import os
from pathlib import Path


def publish_new_file(staged: Path, destination: Path) -> None:
    """Publish a sibling staging file, rejecting a destination that already exists."""
    if staged.parent != destination.parent:
        raise ValueError("immutable publication requires sibling staging")
    if os.name == "nt":
        # Windows rename is no-replace and works on exFAT, which lacks hard links.
        os.rename(staged, destination)
    else:
        os.link(staged, destination)
