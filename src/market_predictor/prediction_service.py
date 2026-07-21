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
from market_predictor.entry_exit import score_entry_exit_frame
from market_predictor.feature_store import LiveFeatureStore
from market_predictor.global_context import build_sector_theme_monitor
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
from market_predictor.volatile import score_volatile_frame

DEFAULT_MODE_HORIZONS = {"swing": "5d", "intraday": "12b"}


@dataclass(frozen=True)
class ServingRoute:
    """Server-owned route from a prediction horizon to registered artifacts."""

    model: Path
    curated_dataset: Path | None = None
    universe: Path | None = None
    flashpoints: Path | None = None
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
                universe=_optional_path(raw_route.get("universe")),
                flashpoints=_optional_path(raw_route.get("flashpoints")),
                bar_timeframe=str(raw_route.get("bar_timeframe", "unknown")).strip()
                or "unknown",
            )
        if parsed:
            routes[normalized_mode] = parsed
    if not routes:
        raise ValueError("at least one production prediction serving route is required")
    return routes


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
    ) -> None:
        self.root = Path(root)
        self.snapshot_store = snapshot_store or PredictionSnapshotStore(
            self.root / "data/predictions/snapshots"
        )
        self.live_feature_store = live_feature_store or LiveFeatureStore(self.root)
        self.persist_snapshots = persist_snapshots
        if not routes:
            raise ValueError("at least one prediction serving route is required")
        self.routes = {mode: dict(mode_routes) for mode, mode_routes in routes.items()}
        self.data_source = data_source

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
        )
        frame = self._feature_frame(
            self._load_feature_source("swing", route, request),
            request=request,
            timeframe="daily",
        )
        scored = self._score_swing_frame(
            frame=frame,
            model_path=model_path,
            route=route,
        )
        predictions = self._swing_predictions(scored, frame, model.status, request)
        return self._response(request, models={"swing": model}, swing_predictions=predictions)

    def predict_intraday(self, request: PredictionRequest) -> PredictionResponse:
        route, model_path, resolved_horizon = self._serving_route("intraday", request)
        model = self._model_info(
            model_path,
            resolved_horizon=resolved_horizon,
            bar_timeframe=route.bar_timeframe,
        )
        frame = self._feature_frame(
            self._load_feature_source("intraday", route, request),
            request=request,
            timeframe="intraday",
        )
        scored = self._score_intraday_frame(frame=frame, model_path=model_path, request=request)
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
            intraday = {
                row.ticker: row.intraday for row in intraday_response.predictions if row.intraday is not None
            }
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
        ready = True
        for mode, mode_routes in self.routes.items():
            for horizon, route in mode_routes.items():
                name = f"model:{mode}:{horizon}"
                try:
                    info = self._model_info(
                        self._resolve(route.model),
                        resolved_horizon=horizon,
                        bar_timeframe=route.bar_timeframe,
                    )
                    components[name] = {
                        "status": "ready",
                        "model_status": info.status,
                        "artifact_sha256": info.artifact_sha256,
                    }
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
                    }
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
                        raise FileNotFoundError(
                            f"configured curated {mode} feature datasets are unavailable: {missing}"
                        )
                    components[name] = {"status": "ready", "source": "curated"}
            except Exception as exc:
                ready = False
                components[name] = {"status": "not_ready", "reason": str(exc)}

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
        route: ServingRoute,
    ) -> pd.DataFrame:
        feature_frame = frame.copy()
        universe_path = self._resolve(route.universe) if route.universe is not None else None
        if universe_path is not None and universe_path.exists():
            flashpoints = self._optional_frame(route.flashpoints)
            universe = pd.read_csv(universe_path)
            _, ticker_report = build_sector_theme_monitor(
                dataset=feature_frame,
                universe=universe,
                model_path=model_path,
                flashpoints=flashpoints,
                require_promoted=True,
            )
            return ticker_report
        latest = self._latest_rows(feature_frame)
        scored = score_volatile_frame(latest, model_path)
        scored["monitor_signal"] = scored["volatile_model_probability"].map(_swing_signal)
        scored["monitor_score"] = scored["volatile_model_probability"]
        return scored

    def _score_intraday_frame(
        self,
        *,
        frame: pd.DataFrame,
        model_path: Path,
        request: PredictionRequest,
    ) -> pd.DataFrame:
        latest = self._latest_rows(frame)
        return score_entry_exit_frame(latest, model_path)

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
                model_probability=_float_or_none(row.get("volatile_model_probability")),
            ),
            axis=1,
        )
        rows["_decision_score"] = rows.apply(
            lambda row: overlay_decision_score(
                _float_or_none(row.get("volatile_model_probability")),
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
                    probability=_float_or_none(row.get("volatile_model_probability")),
                    decision_score=(
                        _float_or_none(row.get("_decision_score")) if is_ready else None
                    ),
                    model_prediction=(
                        _int_or_none(row.get("volatile_model_prediction")) if is_ready else None
                    ),
                    signal=(
                        _swing_signal(row.get("volatile_model_probability"), catalyst)
                        if is_ready
                        else "not_ready"
                    ),
                    rank=idx + 1 if is_ready else None,
                    close=_float_or_none(row.get("close")),
                    return_1d=_float_or_none(row.get("return_1d")),
                    volume_z20=_float_or_none(row.get("volume_z20")),
                    news_count=_float_or_none(row.get("news_count")),
                    event_count=_float_or_none(row.get("event_count")),
                    sentiment_mean=_float_or_none(row.get("sentiment_mean")),
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
                            "volatile_setup_score",
                            "volume_z20",
                            "news_count",
                            "news_count_z30",
                            "event_count",
                            "sentiment_mean",
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
        probability_cols = [col for col in scored.columns if col.endswith("_probability")]
        probability_col = probability_cols[-1] if probability_cols else None
        if probability_col is None:
            raise ValueError("entry/exit scorer did not produce a probability column")
        prediction_col = probability_col.replace("probability", "prediction")
        rows = scored.copy()
        rows["_catalyst_assessment"] = rows.apply(
            lambda row: assess_catalyst_overlay(
                row,
                model_probability=_float_or_none(row.get(probability_col)),
            ),
            axis=1,
        )
        rows["_decision_score"] = rows.apply(
            lambda row: overlay_decision_score(
                _float_or_none(row.get(probability_col)),
                row["_catalyst_assessment"],
            ),
            axis=1,
        )
        rows = rows.sort_values("_decision_score", ascending=False).reset_index(drop=True)
        intraday_counts = (
            source_frame.assign(ticker=source_frame["ticker"].astype(str).str.upper())
            .groupby("ticker")
            .size()
        )
        predictions: list[IntradayPrediction] = []
        for idx, row in rows.iterrows():
            ticker = str(row["ticker"]).upper()
            catalyst = row["_catalyst_assessment"]
            readiness = self._intraday_readiness(row, intraday_counts.get(ticker, 0), model_status)
            is_ready = readiness.status == VALID
            predictions.append(
                IntradayPrediction(
                    ticker=ticker,
                    date=_string_or_none(row.get("date")),
                    probability=_float_or_none(row.get(probability_col)),
                    decision_score=(
                        _float_or_none(row.get("_decision_score")) if is_ready else None
                    ),
                    model_prediction=_int_or_none(row.get(prediction_col)) if is_ready else None,
                    probability_field=probability_col,
                    signal=(
                        _intraday_signal(row.get(probability_col), catalyst)
                        if is_ready
                        else "not_ready"
                    ),
                    rank=idx + 1 if is_ready else None,
                    close=_float_or_none(row.get("close")),
                    return_1d=_float_or_none(row.get("return_1d")),
                    volume_z20=_float_or_none(row.get("volume_z20")),
                    rsi_14=_float_or_none(row.get("rsi_14")),
                    macd_signal_diff=_float_or_none(row.get("macd_signal_diff")),
                    entry_stop_pct=_float_or_none(row.get("entry_stop_pct")),
                    entry_target_pct=_float_or_none(row.get("entry_target_pct")),
                    catalyst=_catalyst_info(catalyst),
                    readiness=readiness,
                    drivers=_drivers(
                        row,
                        [
                            "volume_z20",
                            "return_1d",
                            "rsi_14",
                            "macd_signal_diff",
                            "entry_stop_pct",
                            "entry_target_pct",
                            "news_count",
                            "event_count",
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
            ["global_net_impact", "market_context_score", "market_context_event_count"],
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
                "qqq_return_1bar",
                "qqq_return_3bar",
                "qqq_return_6bar",
                "spy_return_1bar",
                "spy_return_3bar",
                "spy_return_6bar",
            ],
        )
        market_context_present = _has_any_value(
            row,
            ["market_context_intraday_shock_score_2h", "market_context_event_count_2h", "global_net_impact"],
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
        resolved_horizon = (
            DEFAULT_MODE_HORIZONS[mode] if request.horizon == "auto" else request.horizon
        )
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
        if "ticker" not in frame.columns or "date" not in frame.columns:
            raise ValueError("feature dataset must contain ticker and date")
        symbols = set(request.tickers)
        working = frame.copy()
        working["ticker"] = working["ticker"].astype(str).str.upper().str.strip()
        working = working[working["ticker"].isin(symbols)].copy()
        if working.empty:
            raise ValueError(f"no {timeframe} feature rows found for requested tickers")
        if request.as_of is None:
            return working

        cutoff = pd.Timestamp(request.as_of).tz_convert("UTC")
        if timeframe == "daily":
            availability = _daily_availability_utc(working["date"])
        elif timeframe == "intraday":
            timestamps = pd.to_datetime(working["date"], errors="coerce", utc=True)
            bar_duration = _infer_intraday_bar_duration(timestamps, working["ticker"])
            availability = timestamps + bar_duration
        else:
            raise ValueError(f"unsupported feature timeframe: {timeframe}")
        if availability.isna().any():
            raise ValueError(f"{timeframe} feature dataset contains invalid timestamps")
        working["_feature_available_at_utc"] = availability
        working = working[working["_feature_available_at_utc"] <= cutoff].copy()
        if working.empty:
            raise ValueError(
                f"no {timeframe} feature rows are available at or before {request.as_of.isoformat()}"
            )
        return working

    def _model_info(
        self,
        model_path: Path,
        *,
        resolved_horizon: str,
        bar_timeframe: str,
    ) -> ModelInfo:
        manifest = verify_model_artifact(
            model_path,
            allowed_statuses={MODEL_STATUS_PROMOTED},
        )
        status = str(manifest.get("status", "unknown"))
        target = _optional_str(manifest.get("target_col"))
        target_horizon = _target_horizon(target)
        if target_horizon != resolved_horizon:
            raise ValueError(
                f"requested model horizon {resolved_horizon} is incompatible with model target horizon "
                f"{target_horizon or 'unknown'}"
            )
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
            resolved_horizons={
                name: info.resolved_horizon
                for name, info in models.items()
                if info.resolved_horizon is not None
            },
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

    def _optional_frame(self, path: Path | None) -> pd.DataFrame | None:
        if path is None:
            return None
        resolved = self._resolve(path)
        if not resolved.exists():
            return None
        return self._read_frame(resolved)

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
    intra_prob = intraday.probability if intraday else None
    if intraday is not None and intraday.catalyst.status == "veto":
        return "avoid_entry_catalyst_veto"
    if intraday is not None and intraday.catalyst.status == "conflicting" and intra_prob is not None and intra_prob >= 0.55:
        return "wait_catalyst_conflict"
    if swing_prob is not None and swing_prob >= 0.30 and (intra_prob is None or intra_prob >= 0.55):
        if intraday is not None and intraday.catalyst.status == "confirmed":
            return "high_conviction_watch_confirmed"
        return "high_conviction_watch"
    if swing_prob is not None and swing_prob >= 0.18 and (intra_prob is None or intra_prob >= 0.50):
        return "watch_for_entry"
    if intra_prob is not None and intra_prob >= 0.65 and swing_prob is None:
        return "intraday_watch"
    if swing_prob is not None and swing_prob >= 0.18 and intra_prob is not None and intra_prob < 0.50:
        return "swing_positive_wait_for_intraday"
    return "neutral"


def _swing_signal(probability: Any, catalyst: CatalystAssessment | None = None) -> str:
    value = _float_or_none(probability)
    if value is None:
        return "not_scored"
    if catalyst is not None and catalyst.status == "veto" and value >= 0.18:
        return "bullish_model_catalyst_veto"
    if catalyst is not None and catalyst.status == "conflicting" and value >= 0.18:
        return "bullish_model_catalyst_conflict"
    if value >= 0.30:
        if catalyst is not None and catalyst.status == "confirmed":
            return "strong_bullish_watch_confirmed"
        return "strong_bullish_watch"
    if value >= 0.18:
        if catalyst is not None and catalyst.status == "confirmed":
            return "bullish_watch_confirmed"
        return "bullish_watch"
    if value <= 0.05:
        return "low_probability"
    return "neutral"


def _intraday_signal(probability: Any, catalyst: CatalystAssessment | None = None) -> str:
    value = _float_or_none(probability)
    if value is None:
        return "not_scored"
    if catalyst is not None and catalyst.status == "veto" and value >= 0.55:
        return "avoid_entry_catalyst_veto"
    if catalyst is not None and catalyst.status == "conflicting" and value >= 0.55:
        return "wait_catalyst_conflict"
    if value >= 0.70:
        if catalyst is not None and catalyst.status == "confirmed":
            return "entry_candidate_confirmed"
        return "entry_candidate"
    if value >= 0.55:
        if catalyst is not None and catalyst.status == "confirmed":
            return "watch_for_entry_confirmed"
        return "watch_for_confirmation"
    if value <= 0.40:
        return "avoid_entry"
    return "neutral"


def _drivers(row: pd.Series, columns: list[str]) -> dict[str, float | int | str | None]:
    output: dict[str, float | int | str | None] = {}
    for column in columns:
        if column in row.index:
            value = row.get(column)
            output[column] = _json_value(value)
    return output


def _catalyst_info(assessment: CatalystAssessment) -> CatalystConfirmationInfo:
    return CatalystConfirmationInfo.model_validate(assessment.as_record())


def _daily_availability_utc(values: pd.Series) -> pd.Series:
    """Daily close-derived features become available at the regular-session close."""
    date_text = values.astype(str).str.slice(0, 10)
    local_dates = pd.to_datetime(date_text, errors="coerce")
    local_close = local_dates.dt.tz_localize(
        "America/New_York",
        ambiguous="NaT",
        nonexistent="shift_forward",
    ) + pd.Timedelta(hours=16)
    return local_close.dt.tz_convert("UTC")


def _infer_intraday_bar_duration(timestamps: pd.Series, tickers: pd.Series) -> pd.Timedelta:
    ordered = pd.DataFrame({"timestamp": timestamps, "ticker": tickers}).sort_values(
        ["ticker", "timestamp"]
    )
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
    matches = re.findall(r"(?:^|_)(\d+)([db])(?:_|$)", normalized)
    if not matches:
        return None
    amount, unit = matches[-1]
    return f"{int(amount)}{unit}"


def _has_any_value(row: pd.Series, columns: list[str]) -> bool:
    return any(column in row.index and not pd.isna(row.get(column)) for column in columns)


def _optional_path(value: Any) -> Path | None:
    text = _optional_str(value)
    return Path(text) if text else None


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
