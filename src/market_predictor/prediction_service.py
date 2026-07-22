from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.cutoffs import SWING_NIGHTLY_CUTOFF
from market_predictor.catalyst_overlay import (
    CatalystAssessment,
    assess_catalyst_overlay,
)
from market_predictor.drift import audit_feature_drift
from market_predictor.feature_store import LiveFeatureStore
from market_predictor.intraday.contracts import (
    INTRADAY_MODEL_SCHEMA_VERSION,
    INTRADAY_MODEL_TYPE,
)
from market_predictor.intraday.model import score_intraday_frame
from market_predictor.prediction_contracts import (
    CatalystConfirmationInfo,
    FeatureArtifactIdentityV1,
    GlobalContextInfo,
    IntradayPrediction,
    ModelInfo,
    PredictionDataSource,
    PredictionDependencyError,
    PredictionEvidenceV1,
    PredictionReadinessError,
    PredictionRequest,
    PredictionResponse,
    PredictionRowEvidenceV1,
    PredictionServiceError,
    PredictionValidationError,
    ReadinessInfo,
    SwingPrediction,
    UnifiedTickerPrediction,
)
from market_predictor.prediction_snapshot import PredictionSnapshotStore
from market_predictor.readiness import (
    INVALID,
    VALID,
    WARN,
    assess_daily_readiness,
    assess_intraday_readiness,
)
from market_predictor.registry import MODEL_STATUS_PROMOTED, file_sha256, verify_model_artifact
from market_predictor.resources import memory_audit
from market_predictor.swing.contracts import SWING_MODEL_SCHEMA_VERSION, SWING_MODEL_TYPE
from market_predictor.swing.model import score_swing_frame

DEFAULT_MODE_HORIZONS = {"swing": "5d", "intraday": "60m"}
SERVING_POLICY_ID = "market_predictor.serving_policy.r1_a.v1"
_SWING_STRONG = 0.65
_SWING_WATCH = 0.55
_SWING_LOW = 0.40
_INTRADAY_DOWNSIDE_VETO = 0.55
_INTRADAY_ENTRY = 0.70
_INTRADAY_ENTRY_MAX_DOWNSIDE = 0.35
_INTRADAY_WATCH = 0.55
_INTRADAY_WATCH_MAX_DOWNSIDE = 0.45
_INTRADAY_LOW = 0.40
_INTRADAY_AVOID_DOWNSIDE = 0.50
_SERVING_POLICY = {
    "actionable_readiness": "valid",
    "intraday_rank": "opportunity_probability * (1 - downside_probability)",
    "intraday_signal": {
        "avoid_downside_at_or_above": _INTRADAY_DOWNSIDE_VETO,
        "entry_opportunity_at_or_above": _INTRADAY_ENTRY,
        "entry_max_downside": _INTRADAY_ENTRY_MAX_DOWNSIDE,
        "watch_opportunity_at_or_above": _INTRADAY_WATCH,
        "watch_max_downside": _INTRADAY_WATCH_MAX_DOWNSIDE,
        "avoid_opportunity_at_or_below": _INTRADAY_LOW,
        "avoid_downside_above": _INTRADAY_AVOID_DOWNSIDE,
    },
    "swing_rank": "model_probability",
    "swing_signal": {
        "strong_at_or_above": _SWING_STRONG,
        "watch_at_or_above": _SWING_WATCH,
        "low_at_or_below": _SWING_LOW,
    },
    "unified_signal": {
        "high_conviction_swing_at_or_above": _SWING_STRONG,
        "watch_swing_at_or_above": _SWING_WATCH,
        "intraday_support_opportunity_at_or_above": _INTRADAY_WATCH,
        "intraday_support_max_downside": _INTRADAY_WATCH_MAX_DOWNSIDE,
        "intraday_only_at_or_above": _SWING_STRONG,
        "intraday_only_max_downside": _SWING_LOW,
        "swing_wait_intraday_below": 0.50,
    },
    "catalyst_role": "explanation_only",
}
SERVING_POLICY_SHA256 = hashlib.sha256(
    json.dumps(_SERVING_POLICY, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class ServingRoute:
    """Server-owned route from a prediction horizon to registered artifacts."""

    model: Path
    curated_dataset: Path | None = None
    bar_timeframe: str = "unknown"


@dataclass(frozen=True)
class _FeatureSource:
    frame: pd.DataFrame
    artifact_sha256: str | None
    source_artifact_sha256: str | None = None
    source_artifact_type: str | None = None
    feature_schema_version: str | None = None
    source_watermarks: dict[str, str] | None = None
    release_id: str | None = None


def serving_routes_from_config(config: Mapping[str, Any]) -> dict[str, dict[str, ServingRoute]]:
    """Parse and validate server-owned serving routes from application config."""

    serving = config.get("prediction_serving")
    route_config = serving.get("routes") if isinstance(serving, dict) else None
    if not isinstance(route_config, dict):
        raise ValueError("prediction_serving.routes must be configured")
    routes: dict[str, dict[str, ServingRoute]] = {}
    for mode, raw_mode_routes in route_config.items():
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in DEFAULT_MODE_HORIZONS:
            raise ValueError(f"unsupported configured prediction mode: {mode}")
        if not isinstance(raw_mode_routes, dict):
            raise ValueError(f"prediction_serving.routes.{mode} must be a table")
        parsed: dict[str, ServingRoute] = {}
        for horizon, raw_route in raw_mode_routes.items():
            if not isinstance(raw_route, dict):
                raise ValueError(f"prediction serving route {mode}.{horizon} must be a table")
            model = str(raw_route.get("model", "")).strip()
            if not model:
                raise ValueError(f"prediction serving route {mode}.{horizon} is missing model")
            canonical_horizon = _canonical_horizon(str(horizon))
            if canonical_horizon in parsed:
                raise ValueError(f"duplicate prediction serving route after horizon normalization: {mode}.{canonical_horizon}")
            parsed[canonical_horizon] = ServingRoute(
                model=Path(model),
                bar_timeframe=str(raw_route.get("bar_timeframe", "unknown")).strip() or "unknown",
            )
        if parsed:
            routes[normalized_mode] = parsed
    if not routes:
        raise ValueError("at least one production prediction serving route is required")
    return routes


def verify_serving_model_artifact(
    model_path: Path,
    *,
    resolved_horizon: str,
    expected_model_type: str,
    expected_schema_version: str,
) -> dict[str, Any]:
    """Verify registry integrity and the route-specific production model contract."""

    manifest = verify_model_artifact(
        model_path,
        allowed_statuses={MODEL_STATUS_PROMOTED},
    )
    if manifest.get("model_type") != expected_model_type:
        raise ValueError(f"model type {manifest.get('model_type', 'unknown')} is incompatible with {expected_model_type} serving")
    if manifest.get("schema_version") != expected_schema_version:
        raise ValueError(
            f"model schema {manifest.get('schema_version', 'unknown')} is incompatible with {expected_schema_version} serving"
        )
    target_horizon = _target_horizon(_optional_str(manifest.get("target_col")))
    if target_horizon != resolved_horizon:
        raise ValueError(
            f"requested model horizon {resolved_horizon} is incompatible with model target horizon {target_horizon or 'unknown'}"
        )
    return manifest


class PredictionService:
    """Production serving boundary for promoted market prediction models."""

    def __init__(
        self,
        root: Path | str = Path("."),
        *,
        snapshot_store: PredictionSnapshotStore | None = None,
        live_feature_store: LiveFeatureStore | None = None,
        persist_snapshots: bool = True,
        routes: Mapping[str, Mapping[str, ServingRoute]],
        data_source: PredictionDataSource = "live",
        memory_budget_gib: float = 4.0,
        memory_headroom_gib: float = 0.25,
    ) -> None:
        self.root = Path(root)
        self.snapshot_store = snapshot_store or PredictionSnapshotStore(self.root / "data/predictions/snapshots")
        self.live_feature_store = live_feature_store or LiveFeatureStore(self.root)
        self.persist_snapshots = persist_snapshots
        if not routes:
            raise ValueError("at least one prediction serving route is required")
        self.routes = {mode: dict(mode_routes) for mode, mode_routes in routes.items()}
        self.data_source = data_source
        if memory_budget_gib <= 0 or not 0 < memory_headroom_gib < memory_budget_gib:
            raise ValueError("runtime memory budget and headroom are invalid")
        self.memory_budget_gib = memory_budget_gib
        self.memory_headroom_gib = memory_headroom_gib

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        try:
            if request.mode == "swing":
                response = self.predict_swing(request)
            elif request.mode == "intraday":
                response = self.predict_intraday(request)
            else:
                response = self.predict_unified(request)
            if not self.persist_snapshots:
                return response
            return self.snapshot_store.record(request, response)
        except PredictionServiceError:
            raise
        except OSError as exc:
            raise PredictionDependencyError from exc

    def predict_swing(self, request: PredictionRequest) -> PredictionResponse:
        try:
            route, model_path, resolved_horizon = self._serving_route("swing", request)
            model = self._model_info(
                model_path,
                resolved_horizon=resolved_horizon,
                bar_timeframe=route.bar_timeframe,
                expected_model_type=SWING_MODEL_TYPE,
                expected_schema_version=SWING_MODEL_SCHEMA_VERSION,
            )
            source = self._load_feature_source("swing", route, request)
            frame = self._feature_frame(
                source.frame,
                request=request,
                timeframe="daily",
            )
            scored = self._score_swing_frame(
                frame=frame,
                model_path=model_path,
            )
            predictions = self._swing_predictions(scored, frame, model.status)
            return self._response(
                request,
                models={"swing": model},
                feature_sources={"swing": source},
                feature_frames={"swing": frame},
                swing_predictions=predictions,
            )
        except PredictionServiceError:
            raise
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            raise PredictionReadinessError from exc

    def predict_intraday(self, request: PredictionRequest) -> PredictionResponse:
        try:
            route, model_path, resolved_horizon = self._serving_route("intraday", request)
            model = self._model_info(
                model_path,
                resolved_horizon=resolved_horizon,
                bar_timeframe=route.bar_timeframe,
                expected_model_type=INTRADAY_MODEL_TYPE,
                expected_schema_version=INTRADAY_MODEL_SCHEMA_VERSION,
            )
            source = self._load_feature_source("intraday", route, request)
            frame = self._feature_frame(
                source.frame,
                request=request,
                timeframe="intraday",
            )
            scored = self._score_intraday_frame(frame=frame, model_path=model_path)
            predictions = self._intraday_predictions(scored, frame, model.status)
            return self._response(
                request,
                models={"intraday": model},
                feature_sources={"intraday": source},
                feature_frames={"intraday": frame},
                intraday_predictions=predictions,
            )
        except PredictionServiceError:
            raise
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            raise PredictionReadinessError from exc

    def predict_unified(self, request: PredictionRequest) -> PredictionResponse:
        errors: list[str] = []
        models: dict[str, ModelInfo] = {}
        swing: dict[str, SwingPrediction] = {}
        intraday: dict[str, IntradayPrediction] = {}
        resolved_horizons: dict[str, str] = {}
        evidence_parts: list[PredictionEvidenceV1] = []

        try:
            swing_response = self.predict_swing(request.model_copy(update={"mode": "swing"}))
            models.update(swing_response.models)
            resolved_horizons.update(swing_response.resolved_horizons)
            swing = {row.ticker: row.swing for row in swing_response.predictions if row.swing is not None}
            if swing_response.evidence is not None:
                evidence_parts.append(swing_response.evidence)
            errors.extend(swing_response.errors)
        except PredictionServiceError as exc:
            errors.append(f"swing prediction failed: {exc.public_message}")

        try:
            intraday_response = self.predict_intraday(request.model_copy(update={"mode": "intraday"}))
            models.update(intraday_response.models)
            resolved_horizons.update(intraday_response.resolved_horizons)
            intraday = {row.ticker: row.intraday for row in intraday_response.predictions if row.intraday is not None}
            if intraday_response.evidence is not None:
                evidence_parts.append(intraday_response.evidence)
            errors.extend(intraday_response.errors)
        except PredictionServiceError as exc:
            errors.append(f"intraday prediction failed: {exc.public_message}")

        if not swing and not intraday:
            raise PredictionReadinessError

        rows: list[UnifiedTickerPrediction] = []
        for ticker in request.tickers:
            swing_row = swing.get(ticker)
            intraday_row = intraday.get(ticker)
            row_errors = []
            if swing_row is None:
                row_errors.append("missing swing prediction")
            if intraday_row is None:
                row_errors.append("missing intraday prediction")
            rows.append(
                UnifiedTickerPrediction(
                    ticker=ticker,
                    swing=swing_row,
                    intraday=intraday_row,
                    final_signal=_final_signal(swing_row, intraday_row),
                    readiness_status=_combined_readiness(swing_row, intraday_row),
                    errors=row_errors,
                )
            )
        request_id = str(uuid4())
        evidence = _combine_evidence(request, request_id=request_id, evidence_parts=evidence_parts, data_source=self.data_source)
        return PredictionResponse(
            request_id=request_id,
            mode="unified",
            data_source=self.data_source,
            horizon=request.horizon,
            resolved_horizons=resolved_horizons,
            models=models,
            predictions=rows,
            errors=errors,
            evidence=evidence,
        )

    def health(self, *, as_of: datetime | None = None) -> dict[str, object]:
        """Return deployment readiness without deserializing model artifacts."""

        checked_at = as_of or datetime.now(UTC)
        components: dict[str, dict[str, object]] = {}
        model_manifests: dict[tuple[str, str], dict[str, Any]] = {}
        ready = True
        for mode, mode_routes in self.routes.items():
            for horizon, route in mode_routes.items():
                name = f"model:{mode}:{horizon}"
                try:
                    model_path = self._resolve(route.model)
                    expected_model_type = SWING_MODEL_TYPE if mode == "swing" else INTRADAY_MODEL_TYPE
                    expected_schema_version = SWING_MODEL_SCHEMA_VERSION if mode == "swing" else INTRADAY_MODEL_SCHEMA_VERSION
                    manifest = verify_serving_model_artifact(
                        model_path,
                        resolved_horizon=horizon,
                        expected_model_type=expected_model_type,
                        expected_schema_version=expected_schema_version,
                    )
                    info = self._model_info_from_manifest(
                        model_path,
                        manifest=manifest,
                        resolved_horizon=horizon,
                        bar_timeframe=route.bar_timeframe,
                    )
                    components[name] = {
                        "status": "ready",
                        "model_status": info.status,
                        "artifact_sha256": info.artifact_sha256,
                    }
                    model_manifests[(mode, horizon)] = manifest
                except Exception as exc:
                    ready = False
                    components[name] = {"status": "not_ready", "reason": str(exc)}

        for mode, mode_routes in self.routes.items():
            if not mode_routes:
                continue
            name = f"features:{mode}"
            try:
                if self.data_source == "live":
                    manifest = self.live_feature_store.validate(mode, as_of=checked_at)  # type: ignore[arg-type]
                    components[name] = {
                        "status": "ready",
                        "generated_at_utc": manifest.get("generated_at_utc"),
                        "last_feature_time": manifest.get("last_feature_time"),
                        "price_feed": manifest.get("price_feed"),
                        "source_artifact_sha256": manifest.get("source_artifact_sha256"),
                        "feature_schema_version": manifest.get("feature_schema_version"),
                    }
                    feature_frame = self.live_feature_store.load(mode, as_of=checked_at)  # type: ignore[arg-type]
                    for horizon in mode_routes:
                        model_manifest = model_manifests.get((mode, horizon), {})
                        metrics = model_manifest.get("metrics")
                        reference = (
                            metrics.get("feature_reference_profile")
                            if isinstance(metrics, dict)
                            else None
                        )
                        drift = audit_feature_drift(
                            feature_frame,
                            reference if isinstance(reference, dict) else None,
                        )
                        components[f"drift:{mode}:{horizon}"] = drift
                else:
                    missing = []
                    for route in mode_routes.values():
                        if route.curated_dataset is None:
                            missing.append("<not configured>")
                            continue
                        dataset_path = self._resolve(route.curated_dataset)
                        if not dataset_path.exists():
                            missing.append(str(dataset_path))
                    if missing:
                        raise FileNotFoundError(f"configured curated {mode} feature datasets are unavailable: {missing}")
                    components[name] = {"status": "ready", "source": "curated"}
            except Exception as exc:
                ready = False
                components[name] = {"status": "not_ready", "reason": str(exc)}

        process_memory = memory_audit(
            hard_budget_gib=self.memory_budget_gib,
            headroom_gib=self.memory_headroom_gib,
        ).to_record()
        current_memory = process_memory.get("current_working_set_gib")
        threshold = float(process_memory["safety_threshold_gib"] or 0.0)
        memory_ready = current_memory is None or float(current_memory) <= threshold
        components["process_memory"] = {
            "status": "ready" if memory_ready else "not_ready",
            **process_memory,
        }
        ready &= memory_ready

        return {
            "status": "ready" if ready else "not_ready",
            "checked_at_utc": checked_at.astimezone(UTC).isoformat(),
            "data_source": self.data_source,
            "components": components,
        }

    def _score_swing_frame(
        self,
        *,
        frame: pd.DataFrame,
        model_path: Path,
    ) -> pd.DataFrame:
        latest = self._latest_rows(frame)
        return score_swing_frame(latest, model_path, require_promoted=True)

    def _score_intraday_frame(
        self,
        *,
        frame: pd.DataFrame,
        model_path: Path,
    ) -> pd.DataFrame:
        latest = self._latest_rows(frame)
        return score_intraday_frame(latest, model_path, require_promoted=True)

    def _swing_predictions(
        self,
        scored: pd.DataFrame,
        source_frame: pd.DataFrame,
        model_status: str,
    ) -> list[SwingPrediction]:
        rows = scored.copy()
        rows["_catalyst_assessment"] = rows.apply(
            lambda row: assess_catalyst_overlay(
                row,
                model_probability=_float_or_none(row.get("swing_model_probability")),
            ),
            axis=1,
        )
        rows["_decision_score"] = rows["swing_model_probability"].map(_float_or_none)
        rows = rows.sort_values("_decision_score", ascending=False, na_position="last").reset_index(drop=True)
        daily_counts = pd.Series(dtype="int64")
        if self.data_source == "curated":
            daily_counts = (
                source_frame.assign(
                    ticker=source_frame["ticker"].astype(str).str.upper(),
                    _trading_date=pd.to_datetime(source_frame["date"], errors="coerce").dt.date,
                )
                .groupby("ticker")["_trading_date"]
                .nunique()
            )
        predictions: list[SwingPrediction] = []
        ready_rank = 0
        for _, row in rows.iterrows():
            ticker = str(row["ticker"]).upper()
            catalyst = row["_catalyst_assessment"]
            audited_count = _int_or_none(row.get("daily_bar_count"))
            research_fallback = self.data_source == "curated" and audited_count is None
            daily_bar_count = int(daily_counts.get(ticker, 0)) if research_fallback else audited_count
            readiness = self._daily_readiness(
                row,
                daily_bar_count,
                model_status,
                missing_audited_count=audited_count is None and self.data_source == "live",
            )
            is_ready = readiness.status == VALID
            if is_ready:
                ready_rank += 1
            predictions.append(
                SwingPrediction(
                    ticker=ticker,
                    date=_string_or_none(row.get("date")),
                    probability=_float_or_none(row.get("swing_model_probability")),
                    decision_score=(_float_or_none(row.get("_decision_score")) if is_ready else None),
                    model_prediction=(_int_or_none(row.get("swing_model_prediction")) if is_ready else None),
                    signal=(_swing_signal(row.get("swing_model_probability")) if is_ready else "not_ready"),
                    rank=ready_rank if is_ready else None,
                    close=_float_or_none(row.get("close")),
                    return_1d=_float_or_none(row.get("return_1d")),
                    volume_z20=_float_or_none(row.get("volume_z20")),
                    news_count=_float_or_none(row.get("event_count_3d")),
                    event_count=_float_or_none(row.get("event_count_3d")),
                    sentiment_mean=_float_or_none(row.get("sentiment_mean_3d")),
                    monitor_theme=_string_or_none(row.get("monitor_theme")),
                    global_context=GlobalContextInfo(
                        net_impact=float(row.get("global_net_impact", 0.0) or 0.0),
                        positive_impact=float(row.get("global_positive_impact", 0.0) or 0.0),
                        negative_impact=float(row.get("global_negative_impact", 0.0) or 0.0),
                    ),
                    catalyst=_catalyst_info(catalyst),
                    readiness=readiness,
                    drivers=_drivers(
                        row,
                        [
                            "volume_z20",
                            "event_count_3d",
                            "sentiment_mean_3d",
                            "event_relevance_mean_3d",
                            "return_1d",
                            "sector_return_1d",
                            "rel_return_1d_vs_sector",
                            "global_net_impact",
                        ],
                    ),
                )
            )
        return predictions

    def _intraday_predictions(
        self,
        scored: pd.DataFrame,
        source_frame: pd.DataFrame,
        model_status: str,
    ) -> list[IntradayPrediction]:
        opportunity_col = "intraday_opportunity_probability"
        downside_col = "intraday_downside_probability"
        if opportunity_col not in scored or downside_col not in scored:
            raise ValueError("canonical intraday scorer did not produce both probabilities")
        rows = scored.copy()
        rows["_catalyst_assessment"] = rows.apply(
            lambda row: assess_catalyst_overlay(
                row,
                model_probability=_float_or_none(row.get(opportunity_col)),
            ),
            axis=1,
        )
        rows["_decision_score"] = rows.apply(
            lambda row: _risk_adjusted_intraday_score(row, opportunity_col, downside_col),
            axis=1,
        )
        rows = rows.sort_values("_decision_score", ascending=False).reset_index(drop=True)
        intraday_counts = source_frame.assign(ticker=source_frame["ticker"].astype(str).str.upper()).groupby("ticker").size()
        predictions: list[IntradayPrediction] = []
        ready_rank = 0
        for _, row in rows.iterrows():
            ticker = str(row["ticker"]).upper()
            catalyst = row["_catalyst_assessment"]
            warm_count = _int_or_none(row.get("five_minute_bar_count"))
            readiness = self._intraday_readiness(
                row,
                warm_count if warm_count is not None else intraday_counts.get(ticker, 0),
                model_status,
            )
            is_ready = readiness.status == VALID
            if is_ready:
                ready_rank += 1
            predictions.append(
                IntradayPrediction(
                    ticker=ticker,
                    date=_string_or_none(row.get("date")),
                    opportunity_probability=_float_or_none(row.get(opportunity_col)),
                    downside_probability=_float_or_none(row.get(downside_col)),
                    decision_score=(_float_or_none(row.get("_decision_score")) if is_ready else None),
                    opportunity_prediction=(_int_or_none(row.get("intraday_opportunity_prediction")) if is_ready else None),
                    downside_prediction=(_int_or_none(row.get("intraday_downside_prediction")) if is_ready else None),
                    signal=(
                        _intraday_signal(
                            row.get(opportunity_col),
                            row.get(downside_col),
                        )
                        if is_ready
                        else "not_ready"
                    ),
                    rank=ready_rank if is_ready else None,
                    close=_float_or_none(row.get("close")),
                    return_15m=_float_or_none(row.get("return_3bar_5m")),
                    relative_volume=_float_or_none(row.get("relative_volume_same_slot_20d_5m")),
                    rsi_14=_float_or_none(row.get("rsi_14_5m")),
                    macd_signal_diff=_float_or_none(row.get("macd_signal_diff_pct_5m")),
                    entry_stop_pct=_float_or_none(row.get("entry_stop_pct")),
                    entry_target_pct=_float_or_none(row.get("entry_target_pct")),
                    catalyst=_catalyst_info(catalyst),
                    readiness=readiness,
                    drivers=_drivers(
                        row,
                        [
                            "return_3bar_5m",
                            "relative_volume_same_slot_20d_5m",
                            "rsi_14_5m",
                            "macd_signal_diff_pct_5m",
                            "dist_session_vwap_5m",
                            "rel_return_3bar_vs_qqq_5m",
                            "entry_stop_pct",
                            "entry_target_pct",
                            "event_count_2h",
                            "sentiment_mean_2h",
                        ],
                    ),
                )
            )
        return predictions

    def _daily_readiness(
        self,
        row: pd.Series,
        daily_bar_count: int | None,
        model_status: str,
        *,
        missing_audited_count: bool = False,
    ) -> ReadinessInfo:
        benchmark_present = _has_any_value(
            row,
            ["sector_return_1d", "rel_return_1d_vs_sector", "spy_return_1d"],
        )
        market_context_present = _has_any_value(
            row,
            [
                "global_event_count_1d",
                "global_event_count_3d",
                "global_sentiment_mean_1d",
                "global_net_impact",
            ],
        )
        price_feed = str(row.get("price_feed", "unknown") or "unknown")
        assessed = assess_daily_readiness(
            daily_bar_count=int(daily_bar_count or 0),
            latest_price_date=_string_or_none(row.get("date")),
            price_feed=price_feed,
            benchmark_present=benchmark_present,
            market_context_present=market_context_present,
            model_status=model_status,
            news_candle_mismatch_count=int(row.get("news_candle_mismatch_count", 0) or 0),
            stale_cache=bool(row.get("stale_cache", False)),
        )
        reasons = list(assessed.reasons)
        if missing_audited_count:
            reasons.insert(0, "live feature row is missing audited daily_bar_count")
        return ReadinessInfo(
            status=assessed.status,
            reasons=reasons,
            timeframe="daily",
            daily_bar_count=assessed.daily_bar_count,
            intraday_bar_count=assessed.intraday_bar_count,
            required_bar_count=assessed.required_bar_count,
            latest_price_date=assessed.latest_price_date,
            price_feed=assessed.price_feed,
            benchmark_status=assessed.benchmark_status,
            market_context_status=assessed.market_context_status,
            model_status=assessed.model_status,
            source_status=assessed.source_status,
        )

    def _intraday_readiness(
        self,
        row: pd.Series,
        intraday_bar_count: int,
        model_status: str,
    ) -> ReadinessInfo:
        benchmark_present = _has_any_value(
            row,
            [
                "qqq_return_1bar_5m",
                "qqq_return_3bar_5m",
                "qqq_return_6bar_5m",
                "spy_return_1bar_5m",
                "spy_return_3bar_5m",
                "spy_return_6bar_5m",
            ],
        )
        market_context_present = _has_any_value(
            row,
            ["global_event_count_2h", "global_sentiment_mean_2h", "global_net_impact"],
        )
        price_feed = str(row.get("price_feed", "unknown") or "unknown")
        assessed = assess_intraday_readiness(
            intraday_bar_count=int(intraday_bar_count),
            latest_price_timestamp=_string_or_none(row.get("date")),
            price_feed=price_feed,
            benchmark_present=benchmark_present,
            market_context_present=market_context_present,
            model_status=model_status,
            news_candle_mismatch_count=int(row.get("news_candle_mismatch_count", 0) or 0),
            stale_cache=bool(row.get("stale_cache", False)),
        )
        return ReadinessInfo(
            status=assessed.status,
            reasons=assessed.reasons,
            timeframe="intraday",
            daily_bar_count=assessed.daily_bar_count,
            intraday_bar_count=assessed.intraday_bar_count,
            required_bar_count=assessed.required_bar_count,
            latest_price_date=assessed.latest_price_date,
            price_feed=assessed.price_feed,
            benchmark_status=assessed.benchmark_status,
            market_context_status=assessed.market_context_status,
            model_status=assessed.model_status,
            source_status=assessed.source_status,
        )

    def _serving_route(
        self,
        mode: str,
        request: PredictionRequest,
    ) -> tuple[ServingRoute, Path, str]:
        if mode not in DEFAULT_MODE_HORIZONS:
            raise PredictionValidationError
        routes = self.routes.get(mode, {})
        resolved_horizon = DEFAULT_MODE_HORIZONS[mode] if request.horizon == "auto" else _canonical_horizon(request.horizon)
        if resolved_horizon not in routes:
            raise PredictionValidationError
        route = routes[resolved_horizon]
        return route, self._resolve(route.model), resolved_horizon

    def _feature_frame(
        self,
        frame: pd.DataFrame,
        *,
        request: PredictionRequest,
        timeframe: str,
    ) -> pd.DataFrame:
        if "ticker" not in frame.columns:
            raise ValueError("feature dataset must contain ticker")
        symbols = set(request.tickers)
        working = frame.copy()
        if "date" not in working.columns:
            if timeframe == "daily" and "session_date_et" in working.columns:
                working["date"] = working["session_date_et"]
            elif timeframe == "intraday" and "bar_start_utc" in working.columns:
                working["date"] = working["bar_start_utc"]
            else:
                raise ValueError("feature dataset has no canonical decision date")
        working["ticker"] = working["ticker"].astype(str).str.upper().str.strip()
        working = working[working["ticker"].isin(symbols)].copy()
        if working.empty:
            raise ValueError(f"no {timeframe} feature rows found for requested tickers")
        if "feature_available_at_utc" in working.columns:
            availability = pd.to_datetime(working["feature_available_at_utc"], errors="coerce", utc=True)
        elif timeframe == "daily":
            raise ValueError("daily feature dataset must contain feature_available_at_utc")
        elif timeframe == "intraday":
            timestamps = pd.to_datetime(working["date"], errors="coerce", utc=True)
            bar_duration = _infer_intraday_bar_duration(timestamps, working["ticker"])
            availability = timestamps + bar_duration
        else:
            raise ValueError(f"unsupported feature timeframe: {timeframe}")
        if availability.isna().any():
            raise ValueError(f"{timeframe} feature dataset contains invalid timestamps")
        working["_feature_available_at_utc"] = availability
        if timeframe == "daily" and self.data_source == "live":
            cutoff_columns = {"bar_available_at_utc", "decision_time_utc", "prediction_cutoff_policy_id"}
            missing_cutoff = sorted(cutoff_columns.difference(working.columns))
            if missing_cutoff:
                raise ValueError(f"live swing feature identity is incomplete: {', '.join(missing_cutoff)}")
            decision = _strict_utc_series(working["decision_time_utc"])
            bar_available = _strict_utc_series(working["bar_available_at_utc"])
            if bool(decision.isna().any() | bar_available.isna().any() | bar_available.gt(decision).any()):
                raise ValueError("live swing cutoff timestamps are invalid")
            if bool(
                working["prediction_cutoff_policy_id"]
                .astype(str)
                .ne(SWING_NIGHTLY_CUTOFF.policy_id)
                .any()
            ):
                raise ValueError("live swing cutoff policy identity is invalid")
        if request.as_of is None:
            return working

        cutoff = pd.Timestamp(request.as_of).tz_convert("UTC")
        eligible = working["_feature_available_at_utc"] <= cutoff
        if "decision_time_utc" in working.columns:
            decision_times = pd.to_datetime(working["decision_time_utc"], errors="coerce", utc=True)
            if decision_times.isna().any():
                raise ValueError(f"{timeframe} feature dataset contains invalid decision timestamps")
            eligible &= decision_times <= cutoff
        working = working[eligible].copy()
        if working.empty:
            raise ValueError(f"no {timeframe} feature rows are available at or before {request.as_of.isoformat()}")
        return working

    def _model_info(
        self,
        model_path: Path,
        *,
        resolved_horizon: str,
        bar_timeframe: str,
        expected_model_type: str,
        expected_schema_version: str,
    ) -> ModelInfo:
        manifest = verify_serving_model_artifact(
            model_path,
            resolved_horizon=resolved_horizon,
            expected_model_type=expected_model_type,
            expected_schema_version=expected_schema_version,
        )
        return self._model_info_from_manifest(
            model_path,
            manifest=manifest,
            resolved_horizon=resolved_horizon,
            bar_timeframe=bar_timeframe,
        )

    def _model_info_from_manifest(
        self,
        model_path: Path,
        *,
        manifest: Mapping[str, Any],
        resolved_horizon: str,
        bar_timeframe: str,
    ) -> ModelInfo:
        status = str(manifest.get("status", "unknown"))
        target = _optional_str(manifest.get("target_col"))
        dataset_value = manifest.get("dataset")
        dataset = dataset_value if isinstance(dataset_value, dict) else {}
        return ModelInfo(
            path=str(model_path),
            status=status,
            model_type=_optional_str(manifest.get("model_type")),
            schema_version=_optional_str(manifest.get("schema_version")),
            target=target,
            validation_split=_optional_str(manifest.get("validation_split")),
            artifact_sha256=_optional_str(manifest.get("artifact_sha256")),
            resolved_horizon=resolved_horizon,
            bar_timeframe=bar_timeframe,
            created_at_utc=_optional_str(manifest.get("created_at_utc")),
            training_data_start=_optional_str(dataset.get("first_date")),
            training_data_end=_optional_str(dataset.get("last_date")),
        )

    def _response(
        self,
        request: PredictionRequest,
        *,
        models: dict[str, ModelInfo],
        feature_sources: dict[str, _FeatureSource],
        feature_frames: dict[str, pd.DataFrame],
        swing_predictions: list[SwingPrediction] | None = None,
        intraday_predictions: list[IntradayPrediction] | None = None,
    ) -> PredictionResponse:
        request_id = str(uuid4())
        evidence = self._prediction_evidence(
            request,
            request_id=request_id,
            models=models,
            feature_sources=feature_sources,
            feature_frames=feature_frames,
        )
        swing_rows = swing_predictions or []
        intraday_rows = intraday_predictions or []
        if self.data_source == "live" and evidence.identity_status != "complete":
            reason = "live prediction identity is incomplete"
            swing_rows = [_suppress_swing_prediction(row, reason) for row in swing_rows]
            intraday_rows = [_suppress_intraday_prediction(row, reason) for row in intraday_rows]
        swing_by_ticker = {row.ticker: row for row in swing_rows}
        intraday_by_ticker = {row.ticker: row for row in intraday_rows}
        rows = []
        for ticker in request.tickers:
            swing_row = swing_by_ticker.get(ticker)
            intraday_row = intraday_by_ticker.get(ticker)
            rows.append(
                UnifiedTickerPrediction(
                    ticker=ticker,
                    swing=swing_row,
                    intraday=intraday_row,
                    final_signal=_final_signal(swing_row, intraday_row),
                    readiness_status=_combined_readiness(swing_row, intraday_row),
                    errors=[],
                )
            )
        return PredictionResponse(
            request_id=request_id,
            mode=request.mode,
            data_source=self.data_source,
            horizon=_response_horizon(request, models),
            resolved_horizons={name: info.resolved_horizon for name, info in models.items() if info.resolved_horizon is not None},
            models=models,
            predictions=rows,
            evidence=evidence,
        )

    def _prediction_evidence(
        self,
        request: PredictionRequest,
        *,
        request_id: str,
        models: dict[str, ModelInfo],
        feature_sources: dict[str, _FeatureSource],
        feature_frames: dict[str, pd.DataFrame],
    ) -> PredictionEvidenceV1:
        rows: list[PredictionRowEvidenceV1] = []
        gaps: list[str] = []
        cutoffs: list[datetime] = []
        feature_artifacts: dict[str, FeatureArtifactIdentityV1] = {}
        source_watermarks: dict[str, dict[str, str]] = {}
        release_ids: set[str] = set()

        for mode, frame in feature_frames.items():
            latest = self._latest_rows(frame)
            for _, row in latest.iterrows():
                decision = _aware_datetime_or_none(row.get("decision_time_utc"))
                availability = _aware_datetime_or_none(
                    row.get("_feature_available_at_utc", row.get("feature_available_at_utc"))
                )
                ticker = str(row.get("ticker", "")).upper()
                if decision is None or availability is None or not ticker:
                    gaps.append(f"{mode} row availability identity is missing")
                    continue
                if availability > decision:
                    gaps.append(f"{mode} feature availability exceeds prediction cutoff")
                    continue
                rows.append(
                    PredictionRowEvidenceV1(
                        ticker=ticker,
                        view=mode,
                        decision_time_utc=decision,
                        feature_available_at_utc=availability,
                    )
                )
                cutoffs.append(decision)

            source = feature_sources[mode]
            if _is_sha256(source.artifact_sha256):
                feature_artifacts[mode] = FeatureArtifactIdentityV1(
                    mode=mode,
                    artifact_sha256=str(source.artifact_sha256),
                    source_artifact_sha256=(
                        str(source.source_artifact_sha256) if _is_sha256(source.source_artifact_sha256) else None
                    ),
                    source_artifact_type=source.source_artifact_type,
                    feature_schema_version=source.feature_schema_version,
                )
            else:
                gaps.append(f"{mode} feature artifact identity is missing")
            source_watermarks[mode] = dict(source.source_watermarks or {})
            if self.data_source == "live" and not source_watermarks[mode]:
                gaps.append(f"{mode} source coverage watermarks are missing")
            if source.release_id is not None:
                if _is_sha256(source.release_id):
                    release_ids.add(source.release_id)
                else:
                    gaps.append(f"{mode} release identity is invalid")

        model_hashes: dict[str, str] = {}
        for mode, model in models.items():
            if _is_sha256(model.artifact_sha256):
                model_hashes[mode] = str(model.artifact_sha256)
            else:
                gaps.append(f"{mode} model artifact identity is missing")

        if not cutoffs:
            raise PredictionReadinessError
        if len(release_ids) > 1:
            gaps.append("feature sources belong to different serving releases")
        release_id = next(iter(release_ids)) if len(release_ids) == 1 else None
        identity_status = "research_only" if self.data_source == "curated" else ("incomplete" if gaps else "complete")
        return PredictionEvidenceV1(
            request_id=request_id,
            correlation_id=request.correlation_id or request_id,
            prediction_cutoff_utc=max(cutoffs),
            row_feature_availability=rows,
            feature_artifacts=feature_artifacts,
            release_id=release_id,
            model_artifact_sha256=model_hashes,
            source_watermarks=source_watermarks,
            resolved_horizons={
                name: info.resolved_horizon for name, info in models.items() if info.resolved_horizon is not None
            },
            view_prediction_cutoffs_utc={
                mode: max(
                    row.decision_time_utc
                    for row in rows
                    if row.view == mode
                )
                for mode in feature_frames
                if any(row.view == mode for row in rows)
            },
            serving_policy_id=SERVING_POLICY_ID,
            serving_policy_sha256=SERVING_POLICY_SHA256,
            identity_status=identity_status,
            identity_gaps=sorted(set(gaps)),
        )

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"missing feature dataset: {path}")
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"unsupported dataset format: {path}")

    def _load_feature_source(
        self,
        mode: str,
        route: ServingRoute,
        request: PredictionRequest,
    ) -> _FeatureSource:
        if self.data_source == "curated":
            if route.curated_dataset is None:
                raise PredictionReadinessError
            path = self._resolve(route.curated_dataset)
            return _FeatureSource(
                frame=self._read_frame(path),
                artifact_sha256=file_sha256(path),
                source_artifact_type="curated_feature_dataset",
            )
        manifest = self.live_feature_store.validate(mode, as_of=request.as_of)  # type: ignore[arg-type]
        watermarks_raw = manifest.get("source_watermarks")
        watermarks = (
            {str(key): str(value) for key, value in watermarks_raw.items()}
            if isinstance(watermarks_raw, dict)
            else {}
        )
        return _FeatureSource(
            frame=self.live_feature_store.load(mode, as_of=request.as_of),  # type: ignore[arg-type]
            artifact_sha256=_optional_str(manifest.get("artifact_sha256")),
            source_artifact_sha256=_optional_str(manifest.get("source_artifact_sha256")),
            source_artifact_type=_optional_str(manifest.get("source_artifact_type")),
            feature_schema_version=_optional_str(manifest.get("feature_schema_version")),
            source_watermarks=watermarks,
            release_id=self._active_release_id(),
        )

    def _active_release_id(self) -> str | None:
        marker = self.root / "data/live/.active_release.json"
        if not marker.exists():
            return None
        loaded = json.loads(marker.read_text(encoding="utf-8"))
        release_id = loaded.get("release_id") if isinstance(loaded, dict) else None
        if not _is_sha256(release_id):
            raise ValueError("active release identity is invalid")
        return str(release_id)

    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _latest_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if "ticker" not in frame.columns:
            raise ValueError("feature dataset must contain ticker")
        if "date" not in frame.columns:
            raise ValueError("feature dataset must contain date")
        working = frame.copy()
        working["ticker"] = working["ticker"].astype(str).str.upper()
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        return working.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)


def _combined_readiness(
    swing: SwingPrediction | None,
    intraday: IntradayPrediction | None,
) -> str:
    statuses = [row.readiness.status for row in [swing, intraday] if row is not None]
    if not statuses:
        return INVALID
    if INVALID in statuses:
        return INVALID
    if WARN in statuses:
        return WARN
    return VALID


def _final_signal(swing: SwingPrediction | None, intraday: IntradayPrediction | None) -> str:
    if swing is None and intraday is None:
        return "not_ready"
    if swing is not None and swing.readiness.status != VALID:
        return "not_ready"
    if intraday is not None and intraday.readiness.status != VALID:
        return "not_ready"
    swing_prob = swing.probability if swing else None
    intra_prob = intraday.opportunity_probability if intraday else None
    intra_downside = intraday.downside_probability if intraday else None
    intraday_supports_entry = intra_prob is None or (
        intra_prob >= _INTRADAY_WATCH
        and (intra_downside is None or intra_downside <= _INTRADAY_WATCH_MAX_DOWNSIDE)
    )
    if swing_prob is not None and swing_prob >= _SWING_STRONG and intraday_supports_entry:
        return "high_conviction_watch"
    if swing_prob is not None and swing_prob >= _SWING_WATCH and intraday_supports_entry:
        return "watch_for_entry"
    if (
        intra_prob is not None
        and intra_prob >= _SWING_STRONG
        and (intra_downside is None or intra_downside <= _SWING_LOW)
        and swing_prob is None
    ):
        return "intraday_watch"
    if swing_prob is not None and swing_prob >= _SWING_WATCH and intra_prob is not None and intra_prob < 0.50:
        return "swing_positive_wait_for_intraday"
    return "neutral"


def _swing_signal(probability: Any) -> str:
    value = _float_or_none(probability)
    if value is None:
        return "not_scored"
    if value >= _SWING_STRONG:
        return "strong_bullish_watch"
    if value >= _SWING_WATCH:
        return "bullish_watch"
    if value <= _SWING_LOW:
        return "low_probability"
    return "neutral"


def _intraday_signal(
    opportunity_probability: Any,
    downside_probability: Any,
) -> str:
    opportunity = _float_or_none(opportunity_probability)
    downside = _float_or_none(downside_probability)
    if opportunity is None or downside is None:
        return "not_scored"
    if downside >= _INTRADAY_DOWNSIDE_VETO:
        return "avoid_entry_downside_risk"
    if opportunity >= _INTRADAY_ENTRY and downside <= _INTRADAY_ENTRY_MAX_DOWNSIDE:
        return "entry_candidate"
    if opportunity >= _INTRADAY_WATCH and downside <= _INTRADAY_WATCH_MAX_DOWNSIDE:
        return "watch_for_confirmation"
    if opportunity <= _INTRADAY_LOW or downside > _INTRADAY_AVOID_DOWNSIDE:
        return "avoid_entry"
    return "neutral"


def _risk_adjusted_intraday_score(
    row: pd.Series,
    opportunity_column: str,
    downside_column: str,
) -> float:
    opportunity = _float_or_none(row.get(opportunity_column))
    downside = _float_or_none(row.get(downside_column))
    if opportunity is None or downside is None:
        return float("-inf")
    return opportunity * (1.0 - downside)


def _suppress_swing_prediction(row: SwingPrediction, reason: str) -> SwingPrediction:
    readiness = row.readiness.model_copy(
        update={
            "status": INVALID,
            "reasons": list(dict.fromkeys([*row.readiness.reasons, reason])),
        }
    )
    return row.model_copy(
        update={
            "decision_score": None,
            "model_prediction": None,
            "signal": "not_ready",
            "rank": None,
            "readiness": readiness,
        }
    )


def _suppress_intraday_prediction(row: IntradayPrediction, reason: str) -> IntradayPrediction:
    readiness = row.readiness.model_copy(
        update={
            "status": INVALID,
            "reasons": list(dict.fromkeys([*row.readiness.reasons, reason])),
        }
    )
    return row.model_copy(
        update={
            "decision_score": None,
            "opportunity_prediction": None,
            "downside_prediction": None,
            "signal": "not_ready",
            "rank": None,
            "readiness": readiness,
        }
    )


def _combine_evidence(
    request: PredictionRequest,
    *,
    request_id: str,
    evidence_parts: list[PredictionEvidenceV1],
    data_source: PredictionDataSource,
) -> PredictionEvidenceV1:
    if not evidence_parts:
        raise PredictionReadinessError
    rows = [row for evidence in evidence_parts for row in evidence.row_feature_availability]
    artifacts = {
        mode: artifact
        for evidence in evidence_parts
        for mode, artifact in evidence.feature_artifacts.items()
    }
    model_hashes = {
        mode: digest
        for evidence in evidence_parts
        for mode, digest in evidence.model_artifact_sha256.items()
    }
    watermarks = {
        mode: values
        for evidence in evidence_parts
        for mode, values in evidence.source_watermarks.items()
    }
    horizons = {
        mode: horizon
        for evidence in evidence_parts
        for mode, horizon in evidence.resolved_horizons.items()
    }
    gaps = [gap for evidence in evidence_parts for gap in evidence.identity_gaps]
    policy_hashes = {evidence.serving_policy_sha256 for evidence in evidence_parts}
    if policy_hashes != {SERVING_POLICY_SHA256}:
        gaps.append("serving policy identities do not match")
    release_ids = {evidence.release_id for evidence in evidence_parts if evidence.release_id is not None}
    if len(release_ids) > 1:
        gaps.append("prediction views belong to different serving releases")
    if data_source == "curated":
        identity_status = "research_only"
    elif gaps or any(evidence.identity_status != "complete" for evidence in evidence_parts):
        identity_status = "incomplete"
    else:
        identity_status = "complete"
    return PredictionEvidenceV1(
        request_id=request_id,
        correlation_id=request.correlation_id or request_id,
        prediction_cutoff_utc=max(evidence.prediction_cutoff_utc for evidence in evidence_parts),
        row_feature_availability=rows,
        feature_artifacts=artifacts,
        release_id=next(iter(release_ids)) if len(release_ids) == 1 else None,
        model_artifact_sha256=model_hashes,
        source_watermarks=watermarks,
        resolved_horizons=horizons,
        view_prediction_cutoffs_utc={
            mode: cutoff
            for evidence in evidence_parts
            for mode, cutoff in evidence.view_prediction_cutoffs_utc.items()
        },
        serving_policy_id=SERVING_POLICY_ID,
        serving_policy_sha256=SERVING_POLICY_SHA256,
        identity_status=identity_status,
        identity_gaps=sorted(set(gaps)),
    )


def _response_horizon(request: PredictionRequest, models: dict[str, ModelInfo]) -> str:
    resolved = {model.resolved_horizon for model in models.values() if model.resolved_horizon is not None}
    if len(resolved) == 1:
        return next(iter(resolved))
    return request.horizon


def _canonical_horizon(value: str) -> str:
    normalized = value.strip().lower()
    return "60m" if normalized == "1h" else normalized


def _aware_datetime_or_none(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        return None
    return cast(datetime, timestamp.tz_convert("UTC").to_pydatetime())


def _strict_utc_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.map(_aware_datetime_or_none), utc=True)


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _drivers(row: pd.Series, columns: list[str]) -> dict[str, float | int | str | None]:
    output: dict[str, float | int | str | None] = {}
    for column in columns:
        if column in row.index:
            value = row.get(column)
            output[column] = _json_value(value)
    return output


def _catalyst_info(assessment: CatalystAssessment) -> CatalystConfirmationInfo:
    return CatalystConfirmationInfo.model_validate(assessment.as_record())


def _infer_intraday_bar_duration(timestamps: pd.Series, tickers: pd.Series) -> pd.Timedelta:
    ordered = pd.DataFrame({"timestamp": timestamps, "ticker": tickers}).sort_values(["ticker", "timestamp"])
    differences = ordered.groupby("ticker")["timestamp"].diff()
    usable = differences[(differences > pd.Timedelta(0)) & (differences <= pd.Timedelta(hours=6))]
    if usable.empty:
        raise ValueError("cannot infer intraday bar duration for point-in-time filtering")
    duration = usable.median()
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError("invalid inferred intraday bar duration")
    return duration


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


def _has_any_value(row: pd.Series, columns: list[str]) -> bool:
    return any(column in row.index and not pd.isna(row.get(column)) for column in columns)


def _json_value(value: Any) -> float | int | str | None:
    numeric = _float_or_none(value)
    if numeric is not None:
        return numeric
    text = _optional_str(value)
    return text


def _float_or_none(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(converted):
        return None
    return converted


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return int(number)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    text = str(value)
    return text if text else None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
