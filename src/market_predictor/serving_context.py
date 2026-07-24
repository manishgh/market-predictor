from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib

from market_predictor.intraday.contracts import (
    INTRADAY_MODEL_SCHEMA_VERSION,
    INTRADAY_MODEL_TYPE,
)
from market_predictor.registry import MODEL_STATUS_PROMOTED, verify_model_artifact
from market_predictor.release import (
    load_active_local_pointer,
    load_active_local_release,
)
from market_predictor.resources import (
    assert_memory_budget,
    process_memory_snapshot,
    release_process_memory,
)
from market_predictor.swing.contracts import SWING_MODEL_SCHEMA_VERSION, SWING_MODEL_TYPE
from market_predictor.v3.errors import DataReadinessError


@dataclass(frozen=True, slots=True)
class ActiveReleaseRoute:
    repository: Path
    attestation_trust_store: Path
    bar_timeframe: str = "unknown"
    curated_dataset: Path | None = None
    estimated_resident_gib: float = 0.5


@dataclass(frozen=True, slots=True)
class ActiveModelContext:
    mode: str
    horizon: str
    release_id: str
    pointer_sha256: str
    model_path: Path
    manifest: Mapping[str, Any]
    payload: Mapping[str, Any]


class ModelContextProvider(Protocol):
    def get(
        self,
        mode: str,
        horizon: str,
        route: ActiveReleaseRoute,
    ) -> ActiveModelContext: ...

    def snapshot(self) -> dict[str, object]: ...

    def cached(self, mode: str, horizon: str) -> ActiveModelContext | None: ...

    def is_current(
        self,
        mode: str,
        horizon: str,
        route: ActiveReleaseRoute,
    ) -> bool: ...


class ActiveModelContextCache:
    """Keep exactly one verified, deserialized model per serving route."""

    def __init__(
        self,
        root: Path,
        *,
        memory_budget_gib: float,
        memory_headroom_gib: float,
        max_contexts: int,
    ) -> None:
        self._root = root.resolve()
        self._memory_budget_gib = memory_budget_gib
        self._memory_headroom_gib = memory_headroom_gib
        if max_contexts < 1:
            raise ValueError("active model context limit must be positive")
        self._max_contexts = max_contexts
        self._contexts: dict[tuple[str, str], ActiveModelContext] = {}
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        self._index_lock = threading.Lock()
        self._load_lock = threading.Lock()

    def get(
        self,
        mode: str,
        horizon: str,
        route: ActiveReleaseRoute,
    ) -> ActiveModelContext:
        key = (mode, horizon)
        with self._index_lock:
            route_lock = self._locks.setdefault(key, threading.Lock())
        with route_lock:
            repository = self._resolve(route.repository)
            pointer = load_active_local_pointer(repository)
            cached = self._contexts.get(key)
            if (
                cached is not None
                and cached.release_id == pointer["release_id"]
                and cached.pointer_sha256 == pointer["pointer_sha256"]
            ):
                return cached

            with self._load_lock:
                self._assert_memory("before active model verification")
                active = load_active_local_release(
                    repository,
                    attestation_trust_store_path=self._resolve(
                        route.attestation_trust_store
                    ),
                )
                verified_pointer = active["pointer"]
                release = active["release"]
                release_id = str(verified_pointer["release_id"])
                model_path = (
                    repository
                    / "releases"
                    / release_id
                    / str(release["model_path"])
                ).resolve()
                expected_type, expected_schema = _model_contract(mode)
                manifest = _verify_manifest_contract(
                    model_path,
                    horizon=horizon,
                    expected_model_type=expected_type,
                    expected_schema_version=expected_schema,
                    attestation_trust_store_path=self._resolve(
                        route.attestation_trust_store
                    ),
                )
                if cached is None and len(self._contexts) >= self._max_contexts:
                    raise DataReadinessError("active model context cache is full")
                if cached is not None:
                    self._contexts.pop(key, None)
                    cached = None
                    release_process_memory()
                self._assert_projected_memory(route.estimated_resident_gib)
                payload = joblib.load(model_path)
                if not isinstance(payload, dict):
                    raise DataReadinessError("active model payload is not a mapping")
                if payload.get("model_type") != expected_type:
                    raise DataReadinessError(
                        "active model payload type is incompatible with its route"
                    )
                _validate_payload(payload, mode=mode)
                self._assert_memory("after active model load")
                context = ActiveModelContext(
                    mode=mode,
                    horizon=horizon,
                    release_id=release_id,
                    pointer_sha256=str(verified_pointer["pointer_sha256"]),
                    model_path=model_path,
                    manifest=dict(manifest),
                    payload=payload,
                )
                self._contexts[key] = context
                return context

    def snapshot(self) -> dict[str, object]:
        with self._index_lock:
            contexts = list(self._contexts.values())
        return {
            "loaded_contexts": len(contexts),
            "contexts": [
                {
                    "mode": context.mode,
                    "horizon": context.horizon,
                    "release_id": context.release_id,
                    "artifact_sha256": context.manifest.get("artifact_sha256"),
                }
                for context in sorted(contexts, key=lambda value: (value.mode, value.horizon))
            ],
        }

    def cached(self, mode: str, horizon: str) -> ActiveModelContext | None:
        return self._contexts.get((mode, horizon))

    def is_current(
        self,
        mode: str,
        horizon: str,
        route: ActiveReleaseRoute,
    ) -> bool:
        context = self.cached(mode, horizon)
        if context is None:
            return False
        pointer = load_active_local_pointer(self._resolve(route.repository))
        return (
            context.release_id == str(pointer["release_id"])
            and context.pointer_sha256 == str(pointer["pointer_sha256"])
        )

    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self._root / path

    def _assert_memory(self, stage: str) -> None:
        assert_memory_budget(
            hard_budget_gib=self._memory_budget_gib,
            headroom_gib=self._memory_headroom_gib,
            stage=stage,
        )

    def _assert_projected_memory(self, estimated_resident_gib: float) -> None:
        if estimated_resident_gib <= 0:
            raise DataReadinessError("active route resident-memory estimate is invalid")
        snapshot = process_memory_snapshot()
        if snapshot is None:
            return
        current_gib = snapshot[0] / 1024**3
        threshold = self._memory_budget_gib - self._memory_headroom_gib
        if current_gib + estimated_resident_gib > threshold:
            raise DataReadinessError(
                "active model load would exceed the configured memory safety threshold"
            )


def verify_serving_model_artifact(
    model_path: Path,
    *,
    resolved_horizon: str,
    expected_model_type: str,
    expected_schema_version: str,
    attestation_trust_store_path: Path | None = None,
) -> dict[str, Any]:
    """Verify registry integrity and the route-specific production contract."""

    manifest = verify_model_artifact(
        model_path,
        allowed_statuses={MODEL_STATUS_PROMOTED},
        attestation_trust_store_path=attestation_trust_store_path,
    )
    if manifest.get("model_type") != expected_model_type:
        raise ValueError(
            f"model type {manifest.get('model_type', 'unknown')} is incompatible "
            f"with {expected_model_type} serving"
        )
    if manifest.get("schema_version") != expected_schema_version:
        raise ValueError(
            f"model schema {manifest.get('schema_version', 'unknown')} is incompatible "
            f"with {expected_schema_version} serving"
        )
    target_horizon = _target_horizon(_optional_str(manifest.get("target_col")))
    if target_horizon != resolved_horizon:
        raise ValueError(
            f"requested model horizon {resolved_horizon} is incompatible with model "
            f"target horizon {target_horizon or 'unknown'}"
        )
    return manifest


def _model_contract(mode: str) -> tuple[str, str]:
    if mode == "swing":
        return SWING_MODEL_TYPE, SWING_MODEL_SCHEMA_VERSION
    if mode == "intraday":
        return INTRADAY_MODEL_TYPE, INTRADAY_MODEL_SCHEMA_VERSION
    raise DataReadinessError(f"unsupported active model mode: {mode}")


def _validate_payload(payload: dict[str, Any], *, mode: str) -> None:
    features = payload.get("features")
    if not isinstance(features, list) or not features or not all(
        isinstance(feature, str) and feature for feature in features
    ):
        raise DataReadinessError("active model payload has no valid feature set")
    if mode == "swing":
        if not callable(getattr(payload.get("model"), "predict_proba", None)):
            raise DataReadinessError("active swing payload has no probability estimator")
        if not str(payload.get("target_col") or ""):
            raise DataReadinessError("active swing payload has no target identity")
        return
    models = payload.get("models")
    calibrators = payload.get("calibrators")
    opportunity_target = str(payload.get("opportunity_target_col") or "")
    downside_target = str(payload.get("downside_target_col") or "")
    if not isinstance(models, dict) or not isinstance(calibrators, dict):
        raise DataReadinessError("active intraday payload has no model/calibrator mappings")
    if not opportunity_target or not downside_target:
        raise DataReadinessError("active intraday payload has incomplete target identity")
    if any(
        not callable(getattr(models.get(target), "predict_proba", None))
        for target in (opportunity_target, downside_target)
    ):
        raise DataReadinessError("active intraday payload has incomplete probability estimators")


def _verify_manifest_contract(
    model_path: Path,
    *,
    horizon: str,
    expected_model_type: str,
    expected_schema_version: str,
    attestation_trust_store_path: Path | None = None,
) -> dict[str, Any]:
    manifest = verify_serving_model_artifact(
        model_path,
        resolved_horizon=horizon,
        expected_model_type=expected_model_type,
        expected_schema_version=expected_schema_version,
        attestation_trust_store_path=attestation_trust_store_path,
    )
    if manifest.get("status") != MODEL_STATUS_PROMOTED:
        raise DataReadinessError("active release model is not promoted")
    return manifest


def _target_horizon(target_col: str | None) -> str | None:
    normalized = (target_col or "").strip().lower()
    if "next_week" in normalized:
        return "5d"
    if "next_day" in normalized:
        return "1d"
    matches = re.findall(r"(?:^|_)(\d+)([dbm])(?:_|$)", normalized)
    if not matches:
        return None
    amount, unit = matches[-1]
    return f"{int(amount)}{unit}"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
