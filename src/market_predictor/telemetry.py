from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from market_predictor.prediction_contracts import InvestmentReplayResponse, PredictionResponse
from market_predictor.resources import memory_audit

LOGGER = logging.getLogger("market_predictor.telemetry")


class RuntimeTelemetry:
    """Bounded in-process operational counters; emits structured events to stdout."""

    def __init__(self, *, memory_budget_gib: float = 4.0, memory_headroom_gib: float = 0.25) -> None:
        if memory_budget_gib <= 0 or not 0 < memory_headroom_gib < memory_budget_gib:
            raise ValueError("runtime memory budget and headroom are invalid")
        self.started_at = datetime.now(UTC)
        self.memory_budget_gib = memory_budget_gib
        self.memory_headroom_gib = memory_headroom_gib
        self._lock = threading.Lock()
        self._requests: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"count": 0, "errors": 0, "latency_ms_sum": 0.0, "latency_ms_max": 0.0}
        )
        self._predictions: dict[str, dict[str, int]] = defaultdict(
            lambda: {"requests": 0, "rows": 0, "valid": 0, "warn": 0, "invalid": 0, "errors": 0}
        )
        self._replays: dict[str, int] = defaultdict(int)
        self._last_health: dict[str, object] | None = None
        self._last_models: dict[str, str | None] = {}

    def record_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        elapsed_ms: float,
        principal_id: str | None = None,
        correlation_id: str | None = None,
        required_scope: str | None = None,
    ) -> None:
        key = f"{method.upper()} {path}"
        with self._lock:
            record = self._requests[key]
            record["count"] = int(record["count"]) + 1
            record["errors"] = int(record["errors"]) + int(status_code >= 400)
            record["latency_ms_sum"] = float(record["latency_ms_sum"]) + elapsed_ms
            record["latency_ms_max"] = max(float(record["latency_ms_max"]), elapsed_ms)
        self.emit(
            "http_request",
            route=key,
            status_code=status_code,
            elapsed_ms=round(elapsed_ms, 3),
            principal_id=principal_id,
            correlation_id=correlation_id,
            required_scope=required_scope,
        )

    def record_prediction(
        self,
        response: PredictionResponse,
        *,
        principal_id: str | None = None,
        correlation_id: str | None = None,
        admission: Mapping[str, object] | None = None,
    ) -> None:
        counts = {"valid": 0, "warn": 0, "invalid": 0}
        for row in response.predictions:
            counts[row.readiness_status] += 1
        with self._lock:
            record = self._predictions[response.mode]
            record["requests"] += 1
            record["rows"] += len(response.predictions)
            record["errors"] += len(response.errors)
            for status, count in counts.items():
                record[status] += count
            self._last_models = {
                mode: model.artifact_sha256 for mode, model in response.models.items()
            }
        self.emit(
            "prediction",
            request_id=response.request_id,
            mode=response.mode,
            rows=len(response.predictions),
            readiness=counts,
            model_hashes=self._last_models,
            error_count=len(response.errors),
            principal_id=principal_id,
            correlation_id=correlation_id,
            model_release_ids=(
                response.evidence.model_release_ids
                if response.evidence is not None
                else {}
            ),
            admission=dict(admission) if admission is not None else None,
        )

    def record_replay(self, response: InvestmentReplayResponse) -> None:
        with self._lock:
            self._replays[response.status] += 1
        self.emit(
            "prediction_outcome",
            replay_id=response.replay_id,
            snapshot_id=response.snapshot_id,
            ticker=response.ticker,
            model_view=response.model_view,
            status=response.status,
            stock_return=(response.stock.return_pct if response.stock is not None else None),
            excess_return_vs_spy=response.excess_return_vs_spy,
            excess_return_vs_qqq=response.excess_return_vs_qqq,
        )

    def record_health(self, result: dict[str, object]) -> None:
        with self._lock:
            self._last_health = result
        self.emit("readiness", status=result.get("status"), components=result.get("components"))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            requests = {
                route: {
                    **record,
                    "latency_ms_average": (
                        float(record["latency_ms_sum"]) / int(record["count"])
                        if int(record["count"])
                        else 0.0
                    ),
                }
                for route, record in self._requests.items()
            }
            predictions = {mode: dict(record) for mode, record in self._predictions.items()}
            replays = dict(self._replays)
            last_health = self._last_health
            last_models = dict(self._last_models)
        return {
            "schema": "market_predictor.runtime_metrics.v1",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "started_at_utc": self.started_at.isoformat(),
            "requests": requests,
            "predictions": predictions,
            "prediction_outcomes": replays,
            "last_models": last_models,
            "last_health": last_health,
            "memory": memory_audit(
                hard_budget_gib=self.memory_budget_gib,
                headroom_gib=self.memory_headroom_gib,
            ).to_record(),
        }

    @staticmethod
    def emit(event: str, **fields: Any) -> None:
        payload = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        LOGGER.info(json.dumps(payload, sort_keys=True, default=str))
