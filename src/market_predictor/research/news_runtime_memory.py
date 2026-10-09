"""Runtime resource policy for new news work; completed producers stay unchanged.

System memory must be strictly below 90 percent. There is no absolute free-memory
floor. The existing 5 GiB process budget and 0.75 GiB headroom remain unchanged.
These bounded I/O helpers preserve byte hashes and source checks; they do not
reinterpret old producer policy or grant data/model admission.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.system_memory import system_memory_snapshot
from market_predictor.evidence.io import inside
from market_predictor.resources import assert_memory_budget


def guard(stage: str = "news source links") -> None:
    assert_memory_budget(stage=stage, hard_budget_gib=5.0, headroom_gib=0.75)
    snapshot = system_memory_snapshot()
    if snapshot is None or snapshot.total_bytes <= 0 or not 0 <= snapshot.available_bytes <= snapshot.total_bytes:
        raise MemoryBudgetError(f"memory guard stopped {stage}: system memory measurement unavailable")
    used = snapshot.used_percent
    if not math.isfinite(used) or used >= 90.0:
        raise MemoryBudgetError(f"memory guard stopped {stage}: system memory {used:.2f}% used; requires strictly below 90%")


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
