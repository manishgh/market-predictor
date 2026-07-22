from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
        data = self._load()
        month = self._month_key()
        record = data.get(self.source, {}).get(month, {})
        used = int(record.get("used", 0))
        return QuotaStatus(
            month=month,
            used=used,
            limit=self.monthly_limit,
            remaining=max(self.monthly_limit - used, 0),
            last_headers=dict(record.get("last_headers", {})),
        )

    def assert_available(self) -> None:
        status = self.status()
        if status.used >= status.limit:
            raise RuntimeError(
                f"{self.source} monthly request limit reached: "
                f"{status.used}/{status.limit} for {status.month}."
            )

    def record_call(self, headers: dict[str, str] | None = None) -> QuotaStatus:
        data = self._load()
        month = self._month_key()
        source_record = data.setdefault(self.source, {})
        record = source_record.setdefault(month, {"used": 0, "calls": [], "last_headers": {}})
        record["used"] = int(record.get("used", 0)) + 1
        record.setdefault("calls", []).append(datetime.now(UTC).isoformat())
        if headers:
            record["last_headers"] = {
                key: value
                for key, value in headers.items()
                if key.lower().startswith("x-ratelimit") or key.lower().startswith("x-rapidapi")
            }
        self._save(data)
        return self.status()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"quota state must contain a JSON object: {self.path}")
        return {str(key): value for key, value in loaded.items()}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _month_key() -> str:
        return datetime.now(UTC).strftime("%Y-%m")
