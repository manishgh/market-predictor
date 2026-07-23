"""Canonical symbol master: point-in-time symbol -> security identity.

Resolves a ticker to a stable canonical security id across renames (aliases),
ticker reuse, and delistings, so historical rows bind to the security that
actually held a symbol at a point in time rather than to today's survivor. The
master is content-addressed so its identity can be bound to datasets and models.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from market_predictor.symbols import canonical_symbol

ACTIVE = "active"
DELISTED = "delisted"
_STATUSES = frozenset({ACTIVE, DELISTED})


@dataclass(frozen=True)
class SymbolRecord:
    """One symbol interval for a security (half-open ``[effective_from, effective_to)``)."""

    canonical_id: str
    symbol: str
    effective_from: date
    effective_to: date | None = None
    status: str = ACTIVE
    source: str = ""

    def to_record(self) -> dict[str, object]:
        return {
            "canonical_id": self.canonical_id,
            "symbol": canonical_symbol(self.symbol),
            "effective_from": self.effective_from.isoformat(),
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "status": self.status,
            "source": self.source,
        }


class SymbolMaster:
    def __init__(self, records: list[SymbolRecord]) -> None:
        self._records = [
            SymbolRecord(
                canonical_id=record.canonical_id,
                symbol=canonical_symbol(record.symbol),
                effective_from=record.effective_from,
                effective_to=record.effective_to,
                status=record.status,
                source=record.source,
            )
            for record in records
        ]
        self._validate()

    def _validate(self) -> None:
        for record in self._records:
            if record.status not in _STATUSES:
                raise ValueError(f"unknown symbol status: {record.status}")
            if record.effective_to is not None and record.effective_to <= record.effective_from:
                raise ValueError(f"symbol interval end must be after start: {record.symbol}")
        # A symbol maps to at most one security at a time; a security carries at most one symbol at a time.
        self._assert_non_overlapping(key=lambda record: record.symbol, label="symbol")
        self._assert_non_overlapping(key=lambda record: record.canonical_id, label="canonical_id")

    def _assert_non_overlapping(self, *, key: Callable[[SymbolRecord], str], label: str) -> None:
        groups: dict[str, list[SymbolRecord]] = {}
        for record in self._records:
            groups.setdefault(key(record), []).append(record)
        for group_key, items in groups.items():
            ordered = sorted(items, key=lambda record: record.effective_from)
            for earlier, later in zip(ordered, ordered[1:], strict=False):
                if earlier.effective_to is None or earlier.effective_to > later.effective_from:
                    raise ValueError(f"overlapping {label} intervals for {group_key}")

    def resolve(self, symbol: str, as_of: date) -> str | None:
        """Return the canonical security id that held ``symbol`` on ``as_of``, or None."""

        target = canonical_symbol(symbol)
        for record in self._records:
            if (
                record.symbol == target
                and record.effective_from <= as_of
                and (record.effective_to is None or as_of < record.effective_to)
            ):
                return record.canonical_id
        return None

    def is_active(self, symbol: str, as_of: date) -> bool:
        return self.resolve(symbol, as_of) is not None

    def sha256(self) -> str:
        payload = sorted(
            (record.to_record() for record in self._records),
            key=lambda item: (str(item["canonical_id"]), str(item["symbol"]), str(item["effective_from"])),
        )
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
