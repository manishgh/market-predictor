"""Dependency-free, portable advisory file lock for atomic publish paths."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class LockTimeout(RuntimeError):
    """Raised when a file lock cannot be acquired within the timeout."""


@contextmanager
def file_lock(target: Path, *, timeout: float = 30.0, poll_seconds: float = 0.05) -> Iterator[None]:
    """Serialize writers to ``target`` via an exclusive lock file.

    Uses an atomic ``O_CREAT | O_EXCL`` create so only one holder exists at a
    time across processes on a single host (portable on Windows and POSIX). This
    is advisory: it blocks other cooperating callers of ``file_lock`` on the same
    target, not arbitrary readers or writers.
    """

    lock_path = target.with_name(f"{target.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise LockTimeout(f"could not acquire lock within {timeout}s: {lock_path}") from None
            time.sleep(poll_seconds)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
