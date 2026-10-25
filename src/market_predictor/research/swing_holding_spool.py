"""Bounded, temporary storage of unchanged canonical holding specifications."""
from __future__ import annotations

import gzip
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TextIO

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.contracts.holding_accounting import HoldingSpecification


class HoldingSpool:
    """Read the identical sealed stream for base and stress under the caller's lease."""

    def __init__(self, path: Path, memory_guard: Callable[[], None]) -> None:
        self.path = path
        self._guard = memory_guard
        self._writer: TextIO | None = gzip.open(path, "xt", encoding="utf-8")
        self._sha256: str | None = None
        self._count = 0

    def append(self, specification: HoldingSpecification) -> None:
        self._guard()
        if self._writer is None:
            raise DataReadinessError("holding spool is already sealed")
        self._writer.write(specification.model_dump_json() + "\n")
        self._count += 1

    def seal(self) -> None:
        self._guard()
        if self._writer is None:
            raise DataReadinessError("holding spool cannot be sealed twice")
        self.close()
        self._sha256 = file_sha256(self.path)

    def read(self) -> Iterator[HoldingSpecification]:
        self._guard()
        if self._sha256 is None or self._writer is not None:
            raise DataReadinessError("holding spool must be sealed before replay")
        if file_sha256(self.path) != self._sha256:
            raise DataReadinessError("sealed holding spool changed")
        count = 0
        with gzip.open(self.path, "rt", encoding="utf-8") as stream:
            for line in stream:
                self._guard()
                specification = HoldingSpecification.model_validate_json(line)
                count += 1
                yield specification
        self._guard()
        if count != self._count or file_sha256(self.path) != self._sha256:
            raise DataReadinessError("holding spool count or bytes changed during replay")

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


@contextmanager
def temporary_holding_spool(parent: Path, *, memory_guard: Callable[[], None]) -> Iterator[HoldingSpool]:
    """Remove only this unpublished derived stream on success or failure."""
    with TemporaryDirectory(prefix=".holding-spool-", dir=parent) as directory:
        spool = HoldingSpool(Path(directory) / "holdings.jsonl.gz", memory_guard)
        try:
            yield spool
        finally:
            spool.close()
