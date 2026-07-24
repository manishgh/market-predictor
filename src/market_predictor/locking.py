"""Dependency-free, OS-released advisory file lock for atomic publish paths."""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class LockTimeout(RuntimeError):
    """Raised when a file lock cannot be acquired within the timeout."""


@contextmanager
def file_lock(target: Path, *, timeout: float = 30.0, poll_seconds: float = 0.05) -> Iterator[None]:
    """Serialize cooperating writers without leaving a stale lock after process death."""

    lock_path = target.with_name(f"{target.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    descriptor = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    if os.fstat(descriptor).st_size == 0:
        os.write(descriptor, b"\0")
        os.fsync(descriptor)
    acquired = False
    while not acquired:
        try:
            _try_lock(descriptor)
            acquired = True
        except OSError:
            if time.monotonic() >= deadline:
                os.close(descriptor)
                raise LockTimeout(f"could not acquire lock within {timeout}s: {lock_path}") from None
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        _unlock(descriptor)
        os.close(descriptor)


def _try_lock(descriptor: int) -> None:
    if os.name == "nt":
        msvcrt: Any = importlib.import_module("msvcrt")
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        return
    fcntl: Any = importlib.import_module("fcntl")
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(descriptor: int) -> None:
    if os.name == "nt":
        msvcrt: Any = importlib.import_module("msvcrt")
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    fcntl: Any = importlib.import_module("fcntl")
    fcntl.flock(descriptor, fcntl.LOCK_UN)
