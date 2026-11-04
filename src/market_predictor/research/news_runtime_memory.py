"""Bounded research evidence I/O using the shared runtime memory policy."""
from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from market_predictor.core import runtime_memory as _memory
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import inside


def file_sha256(path: Path) -> str:
    _memory.guard()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for number, block in enumerate(iter(lambda: stream.read(1024**2), b""), 1):
            if number % 64 == 0:
                _memory.guard()
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
    _memory.guard()
    for batch in cast(Any, pq).ParquetFile(path).iter_batches(batch_size=256, columns=list(columns)):
        _memory.guard()
        yield from batch.to_pylist()
