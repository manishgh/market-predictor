"""The lease shared by the nightly monitoring steps.

Registration, outcome collection, maturation and reporting run one after another each night.
They share this lease rather than the heavy-job lease, so a multi-day research job never blocks
them, and a step waits a bounded time for the step before it instead of failing.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from market_predictor.heavy_jobs import heavy_job_runtime_dir
from market_predictor.locking import LockTimeout, file_lock


class MonitoringBusyError(RuntimeError):
    """Raised when another monitoring step still holds the lease after the wait."""


@contextmanager
def monitoring_lease(
    command: str,
    *,
    runtime_dir: Path | None = None,
    wait_seconds: float = 900.0,
) -> Iterator[dict[str, object]]:
    """Hold the monitoring lease for `command`, waiting up to `wait_seconds` for it."""
    root = runtime_dir or heavy_job_runtime_dir()
    owner_path = root / "monitoring.owner.json"
    with ExitStack() as stack:
        try:
            stack.enter_context(file_lock(root / "monitoring", timeout=wait_seconds))
        except LockTimeout as exc:
            raise MonitoringBusyError(f"another monitoring step holds the lease: {_read_owner(owner_path)}") from exc
        owner: dict[str, object] = {
            "schema": "market_predictor.monitoring_lease_owner",
            "run_id": uuid4().hex,
            "command": command,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at_utc": datetime.now(UTC).isoformat(),
        }
        temporary = owner_path.with_name(f".{owner_path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(owner, sort_keys=True), encoding="utf-8")
            os.replace(temporary, owner_path)
        finally:
            temporary.unlink(missing_ok=True)
        try:
            yield owner
        finally:
            owner_path.unlink(missing_ok=True)


def _read_owner(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return "owner metadata unavailable"
