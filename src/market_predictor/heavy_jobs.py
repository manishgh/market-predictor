"""Cross-process exclusion for memory-heavy dataset and model commands."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ParamSpec, TypeVar
from uuid import uuid4

import typer

from market_predictor.locking import LockTimeout, file_lock

P = ParamSpec("P")
R = TypeVar("R")
HEAVY_JOB_BUSY_EXIT_CODE = 75


class HeavyJobBusyError(RuntimeError):
    """Raised before input loading when another heavy process owns the lease."""


def heavy_job_runtime_dir() -> Path:
    configured = os.environ.get("MARKET_PREDICTOR_RUNTIME_DIR", "").strip()
    return Path(configured) if configured else Path("data/runtime")


@contextmanager
def heavy_job_lease(
    command: str,
    *,
    runtime_dir: Path | None = None,
    config_path: Path | None = None,
) -> Iterator[dict[str, object]]:
    """Acquire the one non-queueing heavy-job lease for this workspace."""

    root = runtime_dir or heavy_job_runtime_dir()
    lock_target = root / "heavy-job"
    owner_path = root / "heavy-job.owner.json"
    try:
        with file_lock(lock_target, timeout=0.0):
            owner = {
                "schema": "market_predictor.heavy_job_owner.v1",
                "run_id": uuid4().hex,
                "command": command,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_at_utc": datetime.now(UTC).isoformat(),
                "config_sha256": _optional_file_sha256(config_path),
            }
            _write_owner(owner_path, owner)
            try:
                yield owner
            finally:
                owner_path.unlink(missing_ok=True)
    except LockTimeout as exc:
        current = _read_owner(owner_path)
        detail = json.dumps(current, sort_keys=True) if current else "owner metadata unavailable"
        raise HeavyJobBusyError(
            f"another heavy market-predictor job owns the workspace lease: {detail}"
        ) from exc


def serialized_heavy_job(
    command: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Typer decorator that exits with EX_TEMPFAIL before a competing load."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            raw_config = kwargs.get("config_path")
            config_path = raw_config if isinstance(raw_config, Path) else None
            try:
                with heavy_job_lease(
                    command,
                    config_path=config_path,
                ):
                    return function(*args, **kwargs)
            except HeavyJobBusyError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from exc

        return wrapped

    return decorate


def _write_owner(path: Path, owner: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(owner, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_owner(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _optional_file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
