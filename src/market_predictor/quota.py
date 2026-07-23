from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_predictor.locking import file_lock


@dataclass(frozen=True)
class QuotaStatus:
    month: str
    used: int
    limit: int
    remaining: int
    last_headers: dict[str, str]


class MonthlyQuotaTracker:
    def __init__(self, path: str | Path, source: str, monthly_limit: int) -> None:
        self.path = Path(path)
        self.source = source
        self.monthly_limit = monthly_limit

    def status(self) -> QuotaStatus:
        return self._status_from(self._load(), self._month_key())

    def assert_available(self) -> None:
        status = self.status()
        if status.used >= status.limit:
            raise RuntimeError(
                f"{self.source} monthly request limit reached: "
                f"{status.used}/{status.limit} for {status.month}."
            )

    def reserve(self, headers: dict[str, str] | None = None) -> QuotaStatus:
        """Atomically check availability and record one call under a file lock.

        This is the concurrency-safe primitive: the limit check and the increment
        happen inside one lock, so two processes cannot both pass the check on the
        last remaining request.
        """

        with file_lock(self.path):
            data = self._load()
            month = self._month_key()
            record = self._record(data, month)
            used = int(record.get("used", 0))
            if used >= self.monthly_limit:
                raise RuntimeError(f"{self.source} monthly request limit reached: {used}/{self.monthly_limit} for {month}.")
            self._apply_call(record, headers)
            self._save(data)
            return self._status_from(data, month)

    def record_call(self, headers: dict[str, str] | None = None) -> QuotaStatus:
        with file_lock(self.path):
            data = self._load()
            month = self._month_key()
            self._apply_call(self._record(data, month), headers)
            self._save(data)
            return self._status_from(data, month)

    def record_headers(self, headers: dict[str, str] | None) -> None:
        """Update the last observed rate-limit headers without consuming quota."""

        if not headers:
            return
        with file_lock(self.path):
            data = self._load()
            record = self._record(data, self._month_key())
            filtered = {
                key: value
                for key, value in headers.items()
                if key.lower().startswith("x-ratelimit") or key.lower().startswith("x-rapidapi")
            }
            if filtered:
                record["last_headers"] = filtered
                self._save(data)

    def _record(self, data: dict[str, Any], month: str) -> dict[str, Any]:
        source_record: dict[str, Any] = data.setdefault(self.source, {})
        record: dict[str, Any] = source_record.setdefault(month, {"used": 0, "calls": [], "last_headers": {}})
        return record

    def _apply_call(self, record: dict[str, Any], headers: dict[str, str] | None) -> None:
        record["used"] = int(record.get("used", 0)) + 1
        record.setdefault("calls", []).append(datetime.now(UTC).isoformat())
        if headers:
            record["last_headers"] = {
                key: value
                for key, value in headers.items()
                if key.lower().startswith("x-ratelimit") or key.lower().startswith("x-rapidapi")
            }

    def _status_from(self, data: dict[str, Any], month: str) -> QuotaStatus:
        record = data.get(self.source, {}).get(month, {})
        used = int(record.get("used", 0))
        return QuotaStatus(
            month=month,
            used=used,
            limit=self.monthly_limit,
            remaining=max(self.monthly_limit - used, 0),
            last_headers=dict(record.get("last_headers", {})),
        )

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"quota state must contain a JSON object: {self.path}")
        return {str(key): value for key, value in loaded.items()}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _month_key() -> str:
        return datetime.now(UTC).strftime("%Y-%m")
