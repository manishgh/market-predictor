from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.catalyst_overlay import (
    CatalystAssessment,
    assess_catalyst_overlay,
    overlay_decision_score,
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
    GlobalContextInfo,
    IntradayPrediction,
    ModelInfo,
    PredictionDataSource,
    PredictionRequest,
    PredictionResponse,
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
from market_predictor.registry import MODEL_STATUS_PROMOTED, verify_model_artifact
from market_predictor.resources import memory_audit
from market_predictor.swing.contracts import SWING_MODEL_SCHEMA_VERSION, SWING_MODEL_TYPE
from market_predictor.swing.model import score_swing_frame

DEFAULT_MODE_HORIZONS = {"swing": "5d", "intraday": "60m"}


@dataclass(frozen=True)
class ServingRoute:
    """Server-owned route from a prediction horizon to registered artifacts."""

    model: Path
    curated_dataset: Path | None = None
    bar_timeframe: str = "unknown"


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
            parsed[str(horizon).strip().lower()] = ServingRoute(
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
        if request.mode == "swing":
            response = self.predict_swing(request)
        elif request.mode == "intraday":
            response = self.predict_intraday(request)
        else:
            response = self.predict_unified(request)
        if not self.persist_snapshots:
            return response
        return self.snapshot_store.record(request, response)

    def predict_swing(self, request: PredictionRequest) -> PredictionResponse:
        route, model_path, resolved_horizon = self._serving_route("swing", request)
        model = self._model_info(
            model_path,
            resolved_horizon=resolved_horizon,
            bar_timeframe=route.bar_timeframe,
            expected_model_type=SWING_MODEL_TYPE,
            expected_schema_version=SWING_MODEL_SCHEMA_VERSION,
        )
        frame = self._feature_frame(
            self._load_feature_source("swing", route, request),
            request=request,
            timeframe="daily",
        )
        scored = self._score_swing_frame(
            frame=frame,
            model_path=model_path,
        )
        predictions = self._swing_predictions(scored, frame, model.status, request)
        return self._response(request, models={"swing": model}, swing_predictions=predictions)

    def predict_intraday(self, request: PredictionRequest) -> PredictionResponse:
        route, model_path, resolved_horizon = self._serving_route("intraday", request)
        model = self._model_info(
            model_path,
            resolved_horizon=resolved_horizon,
            bar_timeframe=route.bar_timeframe,
            expected_model_type=INTRADAY_MODEL_TYPE,
            expected_schema_version=INTRADAY_MODEL_SCHEMA_VERSION,
        )
        frame = self._feature_frame(
            self._load_feature_source("intraday", route, request),
            request=request,
            timeframe="intraday",
        )
        scored = self._score_intraday_frame(frame=frame, model_path=model_path)
        predictions = self._intraday_predictions(scored, frame, model.status)
        return self._response(request, models={"intraday": model}, intraday_predictions=predictions)

    def predict_unified(self, request: PredictionRequest) -> PredictionResponse:
        errors: list[str] = []
        models: dict[str, ModelInfo] = {}
        swing: dict[str, SwingPrediction] = {}
        intraday: dict[str, IntradayPrediction] = {}
        resolved_horizons: dict[str, str] = {}

        try:
            swing_response = self.predict_swing(request.model_copy(update={"mode": "swing"}))
            models.update(swing_response.models)
            resolved_horizons.update(swing_response.resolved_horizons)
            swing = {row.ticker: row.swing for row in swing_response.predictions if row.swing is not None}
            errors.extend(swing_response.errors)
        except Exception as exc:
            errors.append(f"swing prediction failed: {exc}")

        try:
            intraday_response = self.predict_intraday(request.model_copy(update={"mode": "intraday"}))
            models.update(intraday_response.models)
            resolved_horizons.update(intraday_response.resolved_horizons)
            intraday = {row.ticker: row.intraday for row in intraday_response.predictions if row.intraday is not None}
            errors.extend(intraday_response.errors)
        except Exception as exc:
            errors.append(f"intraday prediction failed: {exc}")

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
        return PredictionResponse(
            mode="unified",
            data_source=self.data_source,
            horizon=request.horizon,
            resolved_horizons=resolved_horizons,
            models=models,
            predictions=rows,
            errors=errors,
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
        request: PredictionRequest,
    ) -> list[SwingPrediction]:
        rows = scored.copy()
        rows["_catalyst_assessment"] = rows.apply(
            lambda row: assess_catalyst_overlay(
                row,
                model_probability=_float_or_none(row.get("swing_model_probability")),
            ),
            axis=1,
        )
        rows["_decision_score"] = rows.apply(
            lambda row: overlay_decision_score(
                _float_or_none(row.get("swing_model_probability")),
                row["_catalyst_assessment"],
            ),
            axis=1,
        )
        rows = rows.sort_values("_decision_score", ascending=False).reset_index(drop=True)
        daily_counts = (
            source_frame.assign(
                ticker=source_frame["ticker"].astype(str).str.upper(),
                _trading_date=pd.to_datetime(source_frame["date"], errors="coerce").dt.date,
            )
            .groupby("ticker")["_trading_date"]
            .nunique()
        )
        predictions: list[SwingPrediction] = []
        for idx, row in rows.iterrows():
            ticker = str(row["ticker"]).upper()
            catalyst = row["_catalyst_assessment"]
            readiness = self._daily_readiness(row, daily_counts.get(ticker, 0), model_status)
            is_ready = readiness.status == VALID
            predictions.append(
                SwingPrediction(
                    ticker=ticker,
                    date=_string_or_none(row.get("date")),
                    probability=_float_or_none(row.get("swing_model_probability")),
                    decision_score=(_float_or_none(row.get("_decision_score")) if is_ready else None),
                    model_prediction=(_int_or_none(row.get("swing_model_prediction")) if is_ready else None),
                    signal=(_swing_signal(row.get("swing_model_probability"), catalyst) if is_ready else "not_ready"),
                    rank=idx + 1 if is_ready else None,
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
        for idx, row in rows.iterrows():
            ticker = str(row["ticker"]).upper()
            catalyst = row["_catalyst_assessment"]
            warm_count = _int_or_none(row.get("five_minute_bar_count"))
            readiness = self._intraday_readiness(
                row,
                warm_count if warm_count is not None else intraday_counts.get(ticker, 0),
                model_status,
            )
            is_ready = readiness.status == VALID
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
                            catalyst,
                        )
                        if is_ready
                        else "not_ready"
                    ),
                    rank=idx + 1 if is_ready else None,
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

    def _daily_readiness(self, row: pd.Series, daily_bar_count: int, model_status: str) -> ReadinessInfo:
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
            daily_bar_count=int(daily_bar_count),
            latest_price_date=_string_or_none(row.get("date")),
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
            raise ValueError(f"unsupported prediction mode: {mode}")
        routes = self.routes.get(mode, {})
        resolved_horizon = DEFAULT_MODE_HORIZONS[mode] if request.horizon == "auto" else request.horizon
        if resolved_horizon not in routes:
            supported = ", ".join(sorted(routes))
            raise ValueError(f"unsupported {mode} horizon {resolved_horizon}; supported horizons: {supported}")
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
        if request.as_of is None:
            return working

        cutoff = pd.Timestamp(request.as_of).tz_convert("UTC")
        working = working[working["_feature_available_at_utc"] <= cutoff].copy()
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
        swing_predictions: list[SwingPrediction] | None = None,
        intraday_predictions: list[IntradayPrediction] | None = None,
    ) -> PredictionResponse:
        swing_by_ticker = {row.ticker: row for row in swing_predictions or []}
        intraday_by_ticker = {row.ticker: row for row in intraday_predictions or []}
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
            mode=request.mode,
            data_source=self.data_source,
            horizon=request.horizon,
            resolved_horizons={name: info.resolved_horizon for name, info in models.items() if info.resolved_horizon is not None},
            models=models,
            predictions=rows,
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
    ) -> pd.DataFrame:
        if self.data_source == "curated":
            if route.curated_dataset is None:
                raise ValueError(f"no curated {mode} dataset is configured for this serving route")
            return self._read_frame(self._resolve(route.curated_dataset))
        return self.live_feature_store.load(mode, as_of=request.as_of)  # type: ignore[arg-type]

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
    if intraday is not None and intraday.catalyst.status == "veto":
        return "avoid_entry_catalyst_veto"
    if intraday is not None and intraday.catalyst.status == "conflicting" and intra_prob is not None and intra_prob >= 0.55:
        return "wait_catalyst_conflict"
    intraday_supports_entry = intra_prob is None or (intra_prob >= 0.55 and (intra_downside is None or intra_downside <= 0.45))
    if swing_prob is not None and swing_prob >= 0.65 and intraday_supports_entry:
        if intraday is not None and intraday.catalyst.status == "confirmed":
            return "high_conviction_watch_confirmed"
        return "high_conviction_watch"
    if swing_prob is not None and swing_prob >= 0.55 and intraday_supports_entry:
        return "watch_for_entry"
    if intra_prob is not None and intra_prob >= 0.65 and (intra_downside is None or intra_downside <= 0.40) and swing_prob is None:
        return "intraday_watch"
    if swing_prob is not None and swing_prob >= 0.55 and intra_prob is not None and intra_prob < 0.50:
        return "swing_positive_wait_for_intraday"
    return "neutral"


def _swing_signal(probability: Any, catalyst: CatalystAssessment | None = None) -> str:
    value = _float_or_none(probability)
    if value is None:
        return "not_scored"
    if catalyst is not None and catalyst.status == "veto" and value >= 0.55:
        return "bullish_model_catalyst_veto"
    if catalyst is not None and catalyst.status == "conflicting" and value >= 0.55:
        return "bullish_model_catalyst_conflict"
    if value >= 0.65:
        if catalyst is not None and catalyst.status == "confirmed":
            return "strong_bullish_watch_confirmed"
        return "strong_bullish_watch"
    if value >= 0.55:
        if catalyst is not None and catalyst.status == "confirmed":
            return "bullish_watch_confirmed"
        return "bullish_watch"
    if value <= 0.40:
        return "low_probability"
    return "neutral"


def _intraday_signal(
    opportunity_probability: Any,
    downside_probability: Any,
    catalyst: CatalystAssessment | None = None,
) -> str:
    opportunity = _float_or_none(opportunity_probability)
    downside = _float_or_none(downside_probability)
    if opportunity is None or downside is None:
        return "not_scored"
    if downside >= 0.55:
        return "avoid_entry_downside_risk"
    if catalyst is not None and catalyst.status == "veto" and opportunity >= 0.55:
        return "avoid_entry_catalyst_veto"
    if catalyst is not None and catalyst.status == "conflicting" and opportunity >= 0.55:
        return "wait_catalyst_conflict"
    if opportunity >= 0.70 and downside <= 0.35:
        if catalyst is not None and catalyst.status == "confirmed":
            return "entry_candidate_confirmed"
        return "entry_candidate"
    if opportunity >= 0.55 and downside <= 0.45:
        if catalyst is not None and catalyst.status == "confirmed":
            return "watch_for_entry_confirmed"
        return "watch_for_confirmation"
    if opportunity <= 0.40 or downside > 0.50:
        return "avoid_entry"
    return "neutral"


def _risk_adjusted_intraday_score(
    row: pd.Series,
    opportunity_column: str,
    downside_column: str,
) -> float:
    opportunity = _float_or_none(row.get(opportunity_column))
    downside = _float_or_none(row.get(downside_column))
    overlaid = overlay_decision_score(opportunity, row["_catalyst_assessment"])
    if overlaid is None or downside is None:
        return float("-inf")
    return max(0.0, min(1.0, overlaid * (1.0 - downside)))


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
