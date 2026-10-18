"""Runtime resource policy for new news work; completed producers stay unchanged.

System memory must be strictly below 91 percent. There is no absolute free-memory
floor. The existing 5 GiB process budget and 0.75 GiB headroom remain unchanged.
These bounded I/O helpers preserve byte hashes and source checks; they do not
reinterpret old producer policy or grant data/model admission.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.system_memory import system_memory_snapshot
from market_predictor.evidence.io import inside
from market_predictor.resources import assert_memory_budget

MAXIMUM_SYSTEM_USED_PERCENT = 91.0
MAXIMUM_SYSTEM_PAUSE_SECONDS = 120.0
SYSTEM_MEMORY_POLL_SECONDS = 5.0


def _used_percent(stage: str) -> float:
    assert_memory_budget(stage=stage, hard_budget_gib=5.0, headroom_gib=0.75)
    snapshot = system_memory_snapshot()
    if (snapshot is None
            or type(snapshot.total_bytes) is not int
            or type(snapshot.available_bytes) is not int
            or snapshot.total_bytes <= 0
            or not 0 <= snapshot.available_bytes <= snapshot.total_bytes):
        raise MemoryBudgetError(f"memory guard stopped {stage}: system memory measurement unavailable")
    used = snapshot.used_percent
    if not math.isfinite(used) or not 0 <= used <= 100:
        raise MemoryBudgetError(f"memory guard stopped {stage}: system memory measurement unavailable")
    return used


def _event(event: str, stage: str, used: float, elapsed: float) -> None:
    print(json.dumps({
        "event": event, "timestamp_utc": datetime.now(UTC).isoformat(), "stage": stage,
        "used_percent": used, "ceiling_percent": MAXIMUM_SYSTEM_USED_PERCENT,
        "elapsed_seconds": elapsed, "maximum_wait_seconds": MAXIMUM_SYSTEM_PAUSE_SECONDS,
    }, allow_nan=False, sort_keys=True), file=sys.stderr, flush=True)


def guard(stage: str = "news source links") -> None:
    """Pause only valid system pressure; callers retain their lease and transaction.

    No caller data work runs during the bounded wait. Invalid measurements and
    process-budget failures propagate immediately, including during a pause.
    """
    used = _used_percent(stage)
    if used < MAXIMUM_SYSTEM_USED_PERCENT:
        return
    started = time.monotonic()
    deadline = started + MAXIMUM_SYSTEM_PAUSE_SECONDS
    _event("pause", stage, used, 0.0)
    while True:
        now = time.monotonic()
        if now >= deadline:
            _event("timeout", stage, used, now - started)
            raise MemoryBudgetError(
                f"memory guard stopped {stage}: system memory {used:.2f}% used; "
                f"requires strictly below {MAXIMUM_SYSTEM_USED_PERCENT:g}% "
                f"within {MAXIMUM_SYSTEM_PAUSE_SECONDS:g} seconds"
            )
        time.sleep(min(SYSTEM_MEMORY_POLL_SECONDS, deadline - now))
        used = _used_percent(stage)
        now = time.monotonic()
        if used < MAXIMUM_SYSTEM_USED_PERCENT and now <= deadline:
            _event("resume", stage, used, now - started)
            return


def file_sha256(path: Path) -> str:
    guard()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for number, block in enumerate(iter(lambda: stream.read(1024**2), b""), 1):
            if number % 64 == 0:
                guard()
            digest.update(block)
    return digest.hexdigest()


def pin_file(root: Path, name: str, digest: str, files: dict[str, str]) -> Path:
    path = inside(root, name)
    key = path.relative_to(root).as_posix()
    if key in files and files[key] != digest:
        raise DataReadinessError("conflicting news source pin")
    if file_sha256(path) != digest:
        raise DataReadinessError(f"news input changed: {key}")
    files[key] = digest
    return path


def capture_file(root: Path, path: Path, files: dict[str, str]) -> str:
    path = inside(root, path)
    digest = file_sha256(path)
    key = path.relative_to(root).as_posix()
    if key in files and files[key] != digest:
        raise DataReadinessError("conflicting news source capture")
    files[key] = digest
    return digest


def recheck_files(root: Path, files: Mapping[str, str]) -> None:
    for name, digest in files.items():
        if file_sha256(inside(root, name)) != digest:
            raise DataReadinessError(f"news bound bytes changed: {name}")


def parquet_rows(path: Path, columns: Sequence[str]) -> Iterator[dict[str, Any]]:
    guard()
    for batch in cast(Any, pq).ParquetFile(path).iter_batches(batch_size=256, columns=list(columns)):
        guard()
        yield from batch.to_pylist()
