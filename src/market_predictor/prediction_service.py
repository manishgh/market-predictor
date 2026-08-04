from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pandas as pd

from market_predictor.admission import InferenceAdmissionController
from market_predictor.canonical.cutoffs import SWING_NIGHTLY_CUTOFF
from market_predictor.catalyst_overlay import (
    CatalystAssessment,
    assess_catalyst_overlay,
)
from market_predictor.drift_policy import DriftAssessmentV2, DriftStateStore
from market_predictor.edge_rebuild.serving import (
    LoadedSwingModelGeneration,
    PromotedSwingBundle,
    SwingModelGenerationCache,
    SwingInferenceEngine,
    score_promoted_swing_model,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract, load_strategy_contract
from market_predictor.edge_rebuild.swing_live import (
    FileSwingLiveInputProvider,
    SwingLiveInputProvider,
    build_live_swing_features,
)
from market_predictor.edge_rebuild.swing_selection import (
    select_constrained_swing_portfolio,
)
from market_predictor.feature_store import LiveFeatureStore
from market_predictor.intraday.model import score_intraday_payload
from market_predictor.prediction_contracts import (
    CatalystConfirmationInfo,
    FeatureArtifactIdentityV1,
    IntradayPrediction,
    ModelInfo,
    PredictionConflictError,
    PredictionDataSource,
    PredictionDependencyError,
    PredictionDriftBlockedError,
    PredictionEvidenceV3,
    PredictionModelUnavailableError,
    PredictionReadinessError,
    PredictionRequest,
    PredictionResponse,
    PredictionRowEvidenceV1,
    PredictionServiceError,
    PredictionValidationError,
    ReadinessInfo,
    SwingBenchmarkContext,
    SwingManagedRiskContext,
    SwingPrediction,
    UnifiedTickerPrediction,
)
from market_predictor.prediction_policy import (
    PredictionSelectionPolicy,
    intraday_decision_score,
    parse_prediction_policy,
    select_intraday_candidates,
)
from market_predictor.edge_rebuild.policy import (
    combined_readiness,
    determine_final_signal,
    determine_intraday_signal,
)
from market_predictor.prediction_snapshot import PredictionSnapshotStore
from market_predictor.readiness import (
    INVALID,
    VALID,
    WARN,
    assess_intraday_readiness,
)
from market_predictor.registry import file_sha256
from market_predictor.resources import assert_memory_budget, memory_audit
from market_predictor.serving_context import (
    ActiveModelContext,
    ActiveModelContextCache,
    ActiveReleaseRoute,
    ModelContextProvider,
)
from market_predictor.v3.errors import DataReadinessError, MarketPredictorError

DEFAULT_MODE_HORIZONS = {"swing": "10b", "intraday": "60m"}
SERVING_POLICY_ID = "market_predictor.serving_policy_bundle.v2"
# Serving thresholds are sourced from the canonical prediction policy so the
# served signal semantics and the promotion-evaluated policy share one definition.
ServingRoute = ActiveReleaseRoute


@dataclass(frozen=True)
class _FeatureSource:
    frame: pd.DataFrame
    artifact_sha256: str | None
    source_artifact_sha256: str | None = None
    source_artifact_type: str | None = None
    feature_schema_version: str | None = None
    source_watermarks: dict[str, str] | None = None
    release_id: str | None = None
    serving_bundle_id: str | None = None


def serving_routes_from_config(config: Mapping[str, Any]) -> dict[str, dict[str, ServingRoute]]:
    """Parse and validate server-owned serving routes from application config."""

    serving = config.get("prediction_serving")
    route_config = serving.get("routes") if isinstance(serving, dict) else None
    trust_store = (
        str(serving.get("attestation_trust_store", "")).strip()
        if isinstance(serving, dict)
        else ""
    )
    promotion_gate_policy_sha256 = (
        str(serving.get("promotion_gate_policy_sha256", "")).strip().lower()
        if isinstance(serving, dict)
        else ""
    )
    if not trust_store:
        raise ValueError("prediction_serving.attestation_trust_store must be configured")
    if len(promotion_gate_policy_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in promotion_gate_policy_sha256
    ):
        raise ValueError(
            "prediction_serving.promotion_gate_policy_sha256 must be configured"
        )
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
            repository = str(raw_route.get("release_repository", "")).strip()
            if not repository:
                raise ValueError(
                    f"prediction serving route {mode}.{horizon} is missing release_repository"
                )
            if "model" in raw_route:
                raise ValueError(
                    f"prediction serving route {mode}.{horizon} cannot use a direct model path"
                )
            canonical_horizon = _canonical_horizon(str(horizon))
            if normalized_mode == "swing" and canonical_horizon != "10b":
                raise ValueError(
                    "public swing serving accepts only the ten-session 10b route"
                )
            if canonical_horizon in parsed:
                raise ValueError(f"duplicate prediction serving route after horizon normalization: {mode}.{canonical_horizon}")
            estimated_resident_gib = float(
                raw_route.get("estimated_resident_gib", 0.5)
            )
            if estimated_resident_gib <= 0:
                raise ValueError(
                    f"prediction serving route {mode}.{horizon} has an invalid "
                    "estimated_resident_gib"
                )
            max_model_bytes = int(
                raw_route.get("max_model_bytes", 512 * 1024 * 1024)
            )
            max_feature_bytes = int(
                raw_route.get("max_feature_bytes", 512 * 1024 * 1024)
            )
            max_feature_rows = int(raw_route.get("max_feature_rows", 250_000))
            if min(max_model_bytes, max_feature_bytes, max_feature_rows) < 1:
                raise ValueError(
                    f"prediction serving route {mode}.{horizon} has invalid "
                    "model/feature artifact limits"
                )
            parsed[canonical_horizon] = ServingRoute(
                repository=Path(repository),
                attestation_trust_store=Path(trust_store),
                promotion_gate_policy_sha256=promotion_gate_policy_sha256,
                bar_timeframe=str(raw_route.get("bar_timeframe", "unknown")).strip() or "unknown",
                estimated_resident_gib=estimated_resident_gib,
                max_model_bytes=max_model_bytes,
                max_feature_bytes=max_feature_bytes,
                max_feature_rows=max_feature_rows,
            )
        if parsed:
            routes[normalized_mode] = parsed
    if not routes:
        raise ValueError("at least one production prediction serving route is required")
    return routes


def swing_live_input_provider_from_config(
    config: Mapping[str, Any],
    *,
    root: Path = Path("."),
    memory_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.5,
) -> FileSwingLiveInputProvider:
    serving = config.get("prediction_serving")
    live = serving.get("swing_live") if isinstance(serving, Mapping) else None
    configured = live.get("input_directory") if isinstance(live, Mapping) else None
    path = Path(str(configured or "data/live/edge_rebuild/swing"))
    return FileSwingLiveInputProvider(
        path if path.is_absolute() else root / path,
        memory_budget_gib=memory_budget_gib,
        memory_headroom_gib=memory_headroom_gib,
    )


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
        max_concurrent_inference: int = 1,
        max_tickers_per_request: int = 100,
        inference_memory_reservation_gib: float = 0.5,
        reject_unknown_memory: bool = False,
        model_context_cache: ModelContextProvider | None = None,
        drift_state_store: DriftStateStore | None = None,
        enforce_drift: bool = True,
        maximum_drift_assessment_age_minutes: int = 1_440,
        swing_live_input_provider: SwingLiveInputProvider | None = None,
        swing_model_generation_cache: SwingModelGenerationCache | None = None,
    ) -> None:
        self.root = Path(root)
        self.snapshot_store = snapshot_store or PredictionSnapshotStore(self.root / "data/predictions/snapshots")
        self.live_feature_store = live_feature_store or LiveFeatureStore(self.root)
        self.swing_live_input_provider = swing_live_input_provider
        self.persist_snapshots = persist_snapshots
        if not routes:
            raise ValueError("at least one prediction serving route is required")
        self.routes = {mode: dict(mode_routes) for mode, mode_routes in routes.items()}
        self.data_source = data_source
        if memory_budget_gib <= 0 or not 0 < memory_headroom_gib < memory_budget_gib:
            raise ValueError("runtime memory budget and headroom are invalid")
        self.memory_budget_gib = memory_budget_gib
        self.memory_headroom_gib = memory_headroom_gib
        if self.swing_live_input_provider is None:
            self.swing_live_input_provider = FileSwingLiveInputProvider(
                self.root / "data/live/edge_rebuild/swing",
                memory_budget_gib=memory_budget_gib,
                memory_headroom_gib=memory_headroom_gib,
            )
        self.swing_model_generation_cache = (
            swing_model_generation_cache
            or SwingModelGenerationCache(
                memory_budget_gib=memory_budget_gib,
                memory_headroom_gib=memory_headroom_gib,
            )
        )
        maximum_artifact_bytes = int(
            (memory_budget_gib - memory_headroom_gib) * 1024**3
        )
        for mode_routes in self.routes.values():
            for route in mode_routes.values():
                if (
                    route.max_model_bytes + route.max_feature_bytes
                    > maximum_artifact_bytes
                ):
                    raise ValueError(
                        "combined route artifact byte limits exceed the memory "
                        "safety threshold"
                    )
        if max_concurrent_inference != 1 or max_tickers_per_request < 1:
            raise ValueError(
                "inference concurrency must be one and the ticker limit must be positive"
            )
        self.max_concurrent_inference = max_concurrent_inference
        self.max_tickers_per_request = max_tickers_per_request
        if inference_memory_reservation_gib <= 0:
            raise ValueError("inference memory reservation must be positive")
        self.inference_memory_reservation_gib = inference_memory_reservation_gib
        self.admission = InferenceAdmissionController(
            max_concurrent_requests=max_concurrent_inference,
            memory_budget_gib=memory_budget_gib,
            memory_headroom_gib=memory_headroom_gib,
            reject_unknown_memory=reject_unknown_memory,
        )
        self.model_context_cache = model_context_cache or ActiveModelContextCache(
            self.root,
            memory_budget_gib=memory_budget_gib,
            memory_headroom_gib=memory_headroom_gib,
            max_contexts=sum(len(mode_routes) for mode_routes in self.routes.values()),
        )
        if maximum_drift_assessment_age_minutes < 1:
            raise ValueError("maximum drift assessment age must be positive")
        self.drift_state_store = drift_state_store or DriftStateStore(
            self.root / "data/monitoring/drift"
        )
        self.enforce_drift = enforce_drift
        self.maximum_drift_assessment_age = timedelta(
            minutes=maximum_drift_assessment_age_minutes
        )

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        if len(request.tickers) > self.max_tickers_per_request:
            raise PredictionValidationError
        try:
            with self.admission.lease(
                estimated_incremental_gib=self.inference_memory_reservation_gib
            ):
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
            route, resolved_horizon = self._serving_route("swing", request)
            try:
                contract = load_strategy_contract(
                    self._resolve(
                        Path("configs/edge_rebuild_strategy_contract.toml")
                    )
                )
            except DataReadinessError as exc:
                repository = self._resolve(route.repository)
                if not (repository / "active_generation.json").is_file():
                    raise PredictionModelUnavailableError from exc
                raise
            generation = self._edge_swing_generation(route, contract=contract)
            bundle = generation.bundle
            as_of = request.as_of or datetime.now(UTC)
            if bundle.promoted_at_utc > as_of.astimezone(UTC):
                raise DataReadinessError(
                    "promoted swing bundle was unavailable at the requested as_of"
                )
            if self.swing_live_input_provider is None:
                raise DataReadinessError("swing live-input provider is unavailable")
            inputs = self.swing_live_input_provider.load(
                as_of_utc=as_of,
                maximum_bytes=route.max_feature_bytes,
                maximum_rows=route.max_feature_rows,
            )
            live = build_live_swing_features(
                inputs.stock_daily_bars,
                inputs.benchmark_daily_bars,
                inputs.point_in_time_memberships,
                contract=contract,
                catalyst_authority_directory=inputs.catalyst_authority_directory,
                expected_catalyst_authority_sha256=inputs.catalyst_authority_sha256,
                live_manifest_path=inputs.manifest_path,
                expected_live_manifest_sha256=inputs.manifest_sha256,
                as_of_utc=as_of,
                memory_budget_gib=self.memory_budget_gib,
                memory_headroom_gib=self.memory_headroom_gib,
            )
            engine = SwingInferenceEngine(generation)
            raw_scores = engine.predict(
                feature_frame=live.catalyst_full,
                requested_models=request.requested_models,
            )
            assert_memory_budget(
                hard_budget_gib=self.memory_budget_gib,
                headroom_gib=self.memory_headroom_gib,
                stage="after promoted swing scoring",
            )
            scored_context = live.context.reset_index(drop=True).copy()
            scored_context["__probability"] = raw_scores.get("classifier", tuple())
            if "classifier" in raw_scores:
                scored_context["__classifier_probability"] = raw_scores["classifier"]
            if "xgboost_regressor" in raw_scores:
                scored_context["__regressor_probability"] = raw_scores["xgboost_regressor"]
            if "dualhurdle" in raw_scores:
                scored_context["__unified_probability"] = raw_scores["dualhurdle"]
            selected_ids = _selected_edge_swing_security_ids(
                scored_context,
                probability_threshold=engine.threshold,
                maximum_trades=contract.swing.maximum_trades_per_decision,
                target_maximum_sector_weight=(
                    contract.swing.target_maximum_sector_weight
                ),
                hard_maximum_sector_weight=(
                    contract.swing.hard_maximum_sector_weight
                ),
                minimum_distinct_sectors=(
                    contract.swing.minimum_distinct_sectors_for_selection
                ),
            )
            predictions = _edge_swing_predictions(
                request=request,
                context=scored_context,
                bundle=bundle,
                bundle_sha256=bundle.sha256(),
                threshold=engine.threshold,
                selected_security_ids=selected_ids,
                contract=contract,
                model_as_of_utc=bundle.promoted_at_utc,
                data_as_of_utc=inputs.generated_at_utc,
                live_input_manifest_sha256=inputs.manifest_sha256,
                catalyst_authority_sha256=inputs.catalyst_authority_sha256,
            )
            model = _edge_swing_model_info(
                generation,
                bundle_root=self._resolve(route.repository),
                resolved_horizon=resolved_horizon,
            )
            response = _edge_swing_response(
                request=request,
                model=model,
                predictions=predictions,
                context=scored_context,
                bundle=bundle,
                live_input_manifest_sha256=inputs.manifest_sha256,
                catalyst_authority_sha256=inputs.catalyst_authority_sha256,
                source_watermarks=dict(inputs.source_watermarks),
            )
            assert_memory_budget(
                hard_budget_gib=self.memory_budget_gib,
                headroom_gib=self.memory_headroom_gib,
                stage="after swing response construction",
            )
            return response
        except PredictionServiceError:
            raise
        except (
            DataReadinessError,
            FileNotFoundError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise PredictionReadinessError from exc

    def preload(self) -> None:
        """Verify and deserialize every configured active route before readiness."""

        for mode, mode_routes in sorted(self.routes.items()):
            for horizon, route in sorted(mode_routes.items()):
                if mode == "swing":
                    try:
                        contract = load_strategy_contract(
                            self._resolve(Path("configs/edge_rebuild_strategy_contract.toml"))
                        )
                        self._edge_swing_generation(route, contract=contract)
                    except (DataReadinessError, PredictionModelUnavailableError):
                        # Absence is an expected fail-closed deployment state;
                        # the API remains available and returns a typed 503.
                        continue
                else:
                    self.model_context_cache.get(mode, horizon, route)

    def predict_intraday(self, request: PredictionRequest) -> PredictionResponse:
        try:
            route, resolved_horizon = self._serving_route("intraday", request)
            context = self.model_context_cache.get("intraday", resolved_horizon, route)
            model = self._model_info_from_context(
                context,
                bar_timeframe=route.bar_timeframe,
            )
            prediction_policy = _prediction_policy_for_model(model)
            source = self._load_feature_source(
                "intraday",
                route,
                request,
                context=context,
            )
            self._require_actionable_drift(
                mode="intraday",
                horizon=resolved_horizon,
                model=model,
            )
            frame = self._feature_frame(
                source.frame,
                request=request,
                timeframe="intraday",
            )
            scored = self._score_intraday_frame(frame=frame, context=context)
            predictions = self._intraday_predictions(
                scored,
                frame,
                model.status,
                prediction_policy,
            )
            return self._response(
                request,
                models={"intraday": model},
                feature_sources={"intraday": source},
                feature_frames={"intraday": frame},
                intraday_predictions=predictions,
            )
        except PredictionServiceError:
            raise
        except (
            DataReadinessError,
            FileNotFoundError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise PredictionReadinessError from exc

    def predict_unified(self, request: PredictionRequest) -> PredictionResponse:
        errors: list[str] = []
        models: dict[str, ModelInfo] = {}
        swing: dict[str, SwingPrediction] = {}
        intraday: dict[str, IntradayPrediction] = {}
        resolved_horizons: dict[str, str] = {}
        evidence_parts: list[PredictionEvidenceV3] = []

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
                    final_signal=(
                        "not_ready"
                        if row_errors
                        else determine_final_signal(swing_row, intraday_row)
                    ),
                    readiness_status=(
                        INVALID
                        if row_errors
                        else combined_readiness(swing_row, intraday_row)
                    ),
                    errors=row_errors,
                )
            )
        request_id = str(uuid4())
        evidence = _combine_evidence(request, request_id=request_id, evidence_parts=evidence_parts, data_source=self.data_source)
        if self.data_source == "live" and evidence.identity_status != "complete":
            reason = "unified prediction identity is incomplete"
            rows = [
                row.model_copy(
                    update={
                        "swing": (
                            _suppress_swing_prediction(row.swing, reason)
                            if row.swing is not None
                            else None
                        ),
                        "intraday": (
                            _suppress_intraday_prediction(row.intraday, reason)
                            if row.intraday is not None
                            else None
                        ),
                        "final_signal": "not_ready",
                        "readiness_status": INVALID,
                        "errors": list(dict.fromkeys([*row.errors, reason])),
                    }
                )
                for row in rows
            ]
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
        """Return deployment readiness from the verified cached generations."""

        checked_at = as_of or datetime.now(UTC)
        components: dict[str, dict[str, object]] = {}
        ready = True
        for mode, mode_routes in self.routes.items():
            for horizon, route in mode_routes.items():
                name = f"model:{mode}:{horizon}"
                try:
                    if mode == "swing":
                        contract = load_strategy_contract(
                            self._resolve(Path("configs/edge_rebuild_strategy_contract.toml"))
                        )
                        generation = self._edge_swing_generation(
                            route,
                            contract=contract,
                        )
                        bundle = generation.bundle
                        components[name] = {
                            "status": "ready",
                            "model_status": bundle.model_status,
                            "artifact_sha256": bundle.model_artifact_sha256,
                            "serving_bundle_sha256": bundle.sha256(),
                            "horizon_sessions": bundle.horizon_sessions,
                        }
                        if self.data_source == "live":
                            if self.swing_live_input_provider is None:
                                raise DataReadinessError(
                                    "swing live-input provider is unavailable"
                                )
                            inputs = self.swing_live_input_provider.load(
                                as_of_utc=checked_at,
                                maximum_bytes=route.max_feature_bytes,
                                maximum_rows=route.max_feature_rows,
                            )
                            components[f"features:{mode}:{horizon}"] = {
                                "status": "ready",
                                "manifest_sha256": inputs.manifest_sha256,
                                "generated_at_utc": inputs.generated_at_utc.isoformat(),
                                "catalyst_authority_sha256": inputs.catalyst_authority_sha256,
                                "price_feed": "sip",
                                "adjustment": "all",
                            }
                        continue
                    if not self.model_context_cache.is_current(mode, horizon, route):
                        raise DataReadinessError(
                            "active model context is missing or its pointer changed"
                        )
                    context = self.model_context_cache.cached(mode, horizon)
                    if context is None:
                        raise DataReadinessError("active model context is not preloaded")
                    info = self._model_info_from_context(
                        context,
                        bar_timeframe=route.bar_timeframe,
                    )
                    components[name] = {
                        "status": "ready",
                        "model_status": info.status,
                        "artifact_sha256": info.artifact_sha256,
                        "model_release_id": context.release_id,
                        "serving_bundle_id": context.serving_bundle_id,
                    }
                    if self.data_source == "live":
                        _require_bundle_available_at(context, checked_at)
                        feature_manifest = {
                            str(key): value
                            for key, value in context.feature_manifest.items()
                        }
                        self.live_feature_store.validate_bound_manifest(
                            cast(Any, mode),
                            feature_manifest,
                            as_of=checked_at,
                        )
                        components[f"features:{mode}:{horizon}"] = {
                            "status": "ready",
                            "serving_bundle_id": context.serving_bundle_id,
                            "generated_at_utc": feature_manifest.get(
                                "generated_at_utc"
                            ),
                            "last_feature_time": feature_manifest.get(
                                "last_feature_time"
                            ),
                            "price_feed": feature_manifest.get("price_feed"),
                            "source_artifact_sha256": feature_manifest.get(
                                "source_artifact_sha256"
                            ),
                            "feature_schema_version": feature_manifest.get(
                                "feature_schema_version"
                            ),
                        }
                except Exception as exc:
                    ready = False
                    components[name] = {"status": "not_ready", "reason": str(exc)}
                    continue
                drift_name = f"drift:{mode}:{horizon}"
                if not self.enforce_drift:
                    components[drift_name] = {
                        "status": "disabled",
                        "reason": "drift enforcement is disabled",
                    }
                    continue
                try:
                    assessment = self._load_drift_assessment(
                        mode=mode,
                        horizon=horizon,
                        model=info,
                        checked_at=checked_at,
                    )
                    components[drift_name] = {
                        "status": (
                            "ready"
                            if assessment.actionability == "actionable"
                            else "not_ready"
                        ),
                        **assessment.model_dump(mode="json"),
                    }
                    if assessment.actionability != "actionable":
                        ready = False
                except Exception as exc:
                    ready = False
                    components[drift_name] = {
                        "status": "not_ready",
                        "reason": str(exc),
                    }

        for mode, mode_routes in self.routes.items():
            if not mode_routes:
                continue
            if self.data_source == "live":
                continue
            name = f"features:{mode}"
            try:
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
        components["model_context_cache"] = {
            "status": "ready",
            **self.model_context_cache.snapshot(),
        }
        components["inference_admission"] = {
            "status": "ready",
            **self.admission.snapshot().to_record(),
        }

        return {
            "status": "ready" if ready else "not_ready",
            "checked_at_utc": checked_at.astimezone(UTC).isoformat(),
            "data_source": self.data_source,
            "components": components,
        }

    def _require_actionable_drift(
        self,
        *,
        mode: str,
        horizon: str,
        model: ModelInfo,
    ) -> DriftAssessmentV2 | None:
        if not self.enforce_drift:
            return None
        try:
            assessment = self._load_drift_assessment(
                mode=mode,
                horizon=horizon,
                model=model,
                checked_at=datetime.now(UTC),
            )
        except (DataReadinessError, PredictionConflictError, ValueError) as exc:
            raise PredictionDriftBlockedError from exc
        if assessment.actionability != "actionable":
            raise PredictionDriftBlockedError
        return assessment

    def _load_drift_assessment(
        self,
        *,
        mode: str,
        horizon: str,
        model: ModelInfo,
        checked_at: datetime,
    ) -> DriftAssessmentV2:
        if not self.enforce_drift:
            raise DataReadinessError("drift enforcement is disabled")
        route_identity = _model_drift_identity(model)
        assessment = self.drift_state_store.load(
            mode,
            horizon,
            route_identity["model_release_id"],
        )
        if any(
            getattr(assessment, field) != value
            for field, value in route_identity.items()
        ):
            raise DataReadinessError(
                "route drift assessment model or policy identity mismatch"
            )
        evaluated_at = assessment.evaluated_at_utc.astimezone(UTC)
        if checked_at.astimezone(UTC) - evaluated_at > self.maximum_drift_assessment_age:
            raise DataReadinessError("route drift assessment is stale")
        if evaluated_at > checked_at.astimezone(UTC) + timedelta(minutes=5):
            raise DataReadinessError("route drift assessment is from the future")
        return assessment

    def _edge_swing_generation(
        self,
        route: ServingRoute,
        *,
        contract: StrategyContract,
    ) -> LoadedSwingModelGeneration:
        try:
            return self.swing_model_generation_cache.get(
                self._resolve(route.repository),
                strategy_contract=contract,
                attestation_trust_store_path=self._resolve(
                    route.attestation_trust_store
                ),
                promotion_gate_policy_sha256=route.promotion_gate_policy_sha256,
                maximum_model_bytes=route.max_model_bytes,
                estimated_resident_gib=route.estimated_resident_gib,
            )
        except (MarketPredictorError, OSError, TypeError, ValueError) as exc:
            raise PredictionModelUnavailableError from exc

    def _score_intraday_frame(
        self,
        *,
        frame: pd.DataFrame,
        context: ActiveModelContext,
    ) -> pd.DataFrame:
        latest = self._latest_rows(frame)
        return score_intraday_payload(latest, context.payload)

    def _intraday_predictions(
        self,
        scored: pd.DataFrame,
        source_frame: pd.DataFrame,
        model_status: str,
        prediction_policy: PredictionSelectionPolicy,
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
        readiness_by_index: dict[int, ReadinessInfo] = {}
        for row_index, row in rows.iterrows():
            ticker = str(row["ticker"]).upper()
            warm_count = _int_or_none(row.get("five_minute_bar_count"))
            readiness_by_index[int(row_index)] = self._intraday_readiness(
                row,
                (
                    warm_count
                    if warm_count is not None
                    else int(intraday_counts.get(ticker, 0))
                ),
                model_status,
            )
        ready_rows = rows.loc[
            [
                index
                for index, readiness in readiness_by_index.items()
                if readiness.status == VALID
            ]
        ]
        selected_indexes = set(
            select_intraday_candidates(
                ready_rows,
                policy=prediction_policy,
                opportunity_column=opportunity_col,
                downside_column=downside_col,
            ).index
        )
        predictions: list[IntradayPrediction] = []
        ready_rank = 0
        for row_index, row in rows.iterrows():
            ticker = str(row["ticker"]).upper()
            catalyst = row["_catalyst_assessment"]
            readiness = readiness_by_index[int(row_index)]
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
                        determine_intraday_signal(
                            row.get(opportunity_col),
                            row.get(downside_col),
                        )
                        if is_ready
                        else "not_ready"
                    ),
                    rank=ready_rank if is_ready else None,
                    selection_eligible=(
                        is_ready
                        and (
                            _float_or_none(row.get(downside_col)) is not None
                            and float(row[downside_col])
                            <= prediction_policy.intraday_downside_ceiling
                        )
                    ),
                    selected_for_policy=(
                        is_ready and row_index in selected_indexes
                    ),
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
    ) -> tuple[ServingRoute, str]:
        if mode not in DEFAULT_MODE_HORIZONS:
            raise PredictionValidationError
        routes = self.routes.get(mode, {})
        resolved_horizon = DEFAULT_MODE_HORIZONS[mode] if request.horizon == "auto" else _canonical_horizon(request.horizon)
        if resolved_horizon not in routes:
            raise PredictionValidationError
        route = routes[resolved_horizon]
        return route, resolved_horizon

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
        if working.empty:
            raise ValueError(f"{timeframe} feature dataset is empty")
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
            _require_requested_tickers(working, symbols, timeframe=timeframe)
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
        _require_requested_tickers(working, symbols, timeframe=timeframe)
        return working

    def _model_info_from_context(
        self,
        context: ActiveModelContext,
        *,
        bar_timeframe: str,
    ) -> ModelInfo:
        return self._model_info_from_manifest(
            context.model_path,
            manifest=context.manifest,
            resolved_horizon=context.horizon,
            bar_timeframe=bar_timeframe,
            release_id=context.release_id,
            serving_bundle_id=context.serving_bundle_id,
        )

    def _model_info_from_manifest(
        self,
        model_path: Path,
        *,
        manifest: Mapping[str, Any],
        resolved_horizon: str,
        bar_timeframe: str,
        release_id: str | None = None,
        serving_bundle_id: str | None = None,
    ) -> ModelInfo:
        status = str(manifest.get("status", "unknown"))
        target = _optional_str(manifest.get("target_col"))
        dataset_value = manifest.get("dataset")
        dataset = dataset_value if isinstance(dataset_value, dict) else {}
        extra_value = manifest.get("extra")
        extra = extra_value if isinstance(extra_value, dict) else {}
        metrics_value = manifest.get("metrics")
        metrics = metrics_value if isinstance(metrics_value, dict) else {}
        return ModelInfo(
            path=str(model_path),
            status=status,
            release_id=release_id,
            serving_bundle_id=serving_bundle_id,
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
            label_policy_sha256=_optional_str(
                manifest.get("dataset_label_config_sha256")
                or metrics.get("dataset_label_config_sha256")
            ),
            label_policy=(
                dict(extra["label_policy"])
                if isinstance(extra.get("label_policy"), dict)
                else None
            ),
            execution_policy_sha256=_optional_str(
                manifest.get("execution_policy_sha256")
                or metrics.get("execution_policy_sha256")
            ),
            prediction_policy_sha256=_optional_str(
                manifest.get("prediction_policy_sha256")
                or metrics.get("prediction_policy_sha256")
            ),
            prediction_policy=(
                dict(extra["prediction_policy"])
                if isinstance(extra.get("prediction_policy"), dict)
                else (
                    dict(metrics["prediction_policy"])
                    if isinstance(metrics.get("prediction_policy"), dict)
                    else None
                )
            ),
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
                    final_signal=determine_final_signal(swing_row, intraday_row),
                    readiness_status=combined_readiness(swing_row, intraday_row),
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
    ) -> PredictionEvidenceV3:
        rows: list[PredictionRowEvidenceV1] = []
        gaps: list[str] = []
        cutoffs: list[datetime] = []
        feature_artifacts: dict[str, FeatureArtifactIdentityV1] = {}
        source_watermarks: dict[str, dict[str, str]] = {}
        feature_release_ids: dict[str, str] = {}
        feature_bundle_ids: dict[str, str] = {}

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
                        canonical_security_id=_optional_str(
                            row.get("canonical_security_id", row.get("canonical_id"))
                        ),
                        decision_group_id=_optional_str(row.get("decision_group_id")),
                        session_date_et=_optional_str(row.get("session_date_et")),
                        primary_benchmark=_optional_str(row.get("primary_benchmark")),
                        market_regime=_optional_str(row.get("market_regime")),
                        sector=_optional_str(row.get("sector")),
                        market_cap_bucket=_optional_str(row.get("market_cap_bucket")),
                        liquidity_bucket=_optional_str(row.get("liquidity_bucket")),
                        price_feed=_optional_str(row.get("price_feed")),
                        decision_atr=_float_or_none(row.get("atr_14_price_5m")),
                    )
                )
                if self.data_source == "live":
                    required_row_identity = {
                        "canonical_security_id": row.get(
                            "canonical_security_id",
                            row.get("canonical_id"),
                        ),
                        "decision_group_id": row.get("decision_group_id"),
                        "primary_benchmark": row.get("primary_benchmark"),
                        "market_regime": row.get("market_regime"),
                        "sector": row.get("sector"),
                        "market_cap_bucket": row.get("market_cap_bucket"),
                        "liquidity_bucket": row.get("liquidity_bucket"),
                        "price_feed": row.get("price_feed"),
                    }
                    missing_row_identity = sorted(
                        name
                        for name, value in required_row_identity.items()
                        if _optional_str(value) is None
                    )
                    if mode == "intraday" and _float_or_none(
                        row.get("atr_14_price_5m")
                    ) is None:
                        missing_row_identity.append("decision_atr")
                    if missing_row_identity:
                        gaps.append(
                            f"{mode} maturation row identity is missing: "
                            f"{', '.join(missing_row_identity)}"
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
                    feature_release_ids[mode] = source.release_id
                else:
                    gaps.append(f"{mode} release identity is invalid")
            if source.serving_bundle_id is not None:
                if _is_sha256(source.serving_bundle_id):
                    feature_bundle_ids[mode] = source.serving_bundle_id
                else:
                    gaps.append(f"{mode} serving bundle identity is invalid")

        model_hashes: dict[str, str] = {}
        model_release_ids: dict[str, str] = {}
        model_bundle_ids: dict[str, str] = {}
        prediction_policy_hashes: dict[str, str] = {}
        for mode, model in models.items():
            if _is_sha256(model.artifact_sha256):
                model_hashes[mode] = str(model.artifact_sha256)
            else:
                gaps.append(f"{mode} model artifact identity is missing")
            if _is_sha256(model.release_id):
                model_release_ids[mode] = str(model.release_id)
            elif self.data_source == "live":
                gaps.append(f"{mode} model release identity is missing")
            if _is_sha256(model.serving_bundle_id):
                model_bundle_ids[mode] = str(model.serving_bundle_id)
            elif self.data_source == "live":
                gaps.append(f"{mode} model serving bundle identity is missing")
            if model.label_policy is None or not _is_sha256(
                model.label_policy_sha256
            ):
                gaps.append(f"{mode} model label policy identity is missing")
            if not _is_sha256(model.execution_policy_sha256):
                gaps.append(f"{mode} execution policy identity is missing")
            if model.prediction_policy is None or not _is_sha256(
                model.prediction_policy_sha256
            ):
                gaps.append(f"{mode} prediction policy identity is missing")
            else:
                try:
                    parse_prediction_policy(
                        model.prediction_policy,
                        expected_sha256=model.prediction_policy_sha256,
                    )
                except (TypeError, ValueError):
                    gaps.append(f"{mode} prediction policy identity is invalid")
                else:
                    prediction_policy_hashes[mode] = str(
                        model.prediction_policy_sha256
                    )

        if not cutoffs:
            raise PredictionReadinessError
        if self.data_source == "live":
            for mode in models:
                if feature_release_ids.get(mode) != model_release_ids.get(mode):
                    gaps.append(
                        f"{mode} model and feature release identities conflict"
                    )
                if feature_bundle_ids.get(mode) != model_bundle_ids.get(mode):
                    gaps.append(
                        f"{mode} model and feature serving bundle identities conflict"
                    )
        release_id = (
            next(iter(model_release_ids.values()))
            if len(set(model_release_ids.values())) == 1
            else None
        )
        identity_status = "research_only" if self.data_source == "curated" else ("incomplete" if gaps else "complete")
        return PredictionEvidenceV3(
            request_id=request_id,
            correlation_id=request.correlation_id or request_id,
            prediction_cutoff_utc=max(cutoffs),
            row_feature_availability=rows,
            feature_artifacts=feature_artifacts,
            release_id=release_id,
            model_release_ids=model_release_ids,
            view_serving_bundle_ids=model_bundle_ids,
            serving_bundle_sha256=(
                _serving_bundle_set_sha256(model_bundle_ids)
                if model_bundle_ids
                else None
            ),
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
            view_prediction_policy_sha256=prediction_policy_hashes,
            serving_policy_id=SERVING_POLICY_ID,
            serving_policy_sha256=_serving_policy_bundle_sha256(
                prediction_policy_hashes
            ),
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
        *,
        context: ActiveModelContext,
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
        if context.feature_frame is None or context.serving_bundle_id is None:
            raise ValueError("live serving requires an atomic model/feature bundle")
        _require_bundle_available_at(
            context,
            request.as_of or datetime.now(UTC),
        )
        manifest = {
            str(key): value for key, value in context.feature_manifest.items()
        }
        self.live_feature_store.validate_bound_manifest(
            cast(Any, mode),
            manifest,
            as_of=request.as_of,
        )
        watermarks_raw = manifest.get("source_watermarks")
        watermarks = (
            {str(key): str(value) for key, value in watermarks_raw.items()}
            if isinstance(watermarks_raw, dict)
            else {}
        )
        return _FeatureSource(
            frame=context.feature_frame,
            artifact_sha256=_optional_str(manifest.get("artifact_sha256")),
            source_artifact_sha256=_optional_str(manifest.get("source_artifact_sha256")),
            source_artifact_type=_optional_str(manifest.get("source_artifact_type")),
            feature_schema_version=_optional_str(manifest.get("feature_schema_version")),
            source_watermarks=watermarks,
            release_id=context.release_id,
            serving_bundle_id=context.serving_bundle_id,
        )

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


def _edge_swing_model_info(
    generation: LoadedSwingModelGeneration,
    *,
    bundle_root: Path,
    resolved_horizon: str,
) -> ModelInfo:
    bundle = generation.bundle
    return ModelInfo(
        path=str(
            bundle_root
            / "generations"
            / generation.generation_id
            / bundle.model_artifact_path
        ),
        status=bundle.model_status,
        release_id=bundle.sha256(),
        serving_bundle_id=bundle.sha256(),
        model_type="ten_session_sector_relative_swing_classifier",
        schema_version=bundle.feature_schema_version,
        target="top_sector_relative_quantile_of_managed_barrier_net_return",
        validation_split="purged_walk_forward_with_locked_final_test",
        artifact_sha256=bundle.model_artifact_sha256,
        resolved_horizon=resolved_horizon,
        bar_timeframe="1Day",
        created_at_utc=bundle.promoted_at_utc.isoformat(),
        label_policy_sha256=bundle.strategy_contract_sha256,
        label_policy={
            "horizon_sessions": bundle.horizon_sessions,
            "strategy_contract_sha256": bundle.strategy_contract_sha256,
        },
        execution_policy_sha256=bundle.strategy_contract_sha256,
    )


def _selected_edge_swing_security_ids(
    frame: pd.DataFrame,
    *,
    probability_threshold: float,
    maximum_trades: int,
    target_maximum_sector_weight: float,
    hard_maximum_sector_weight: float,
    minimum_distinct_sectors: int,
) -> set[str]:
    eligible = frame.loc[
        pd.to_numeric(frame["__probability"], errors="coerce").ge(
            probability_threshold
        )
    ]
    selected = select_constrained_swing_portfolio(
        eligible,
        maximum_trades=maximum_trades,
        target_maximum_sector_weight=target_maximum_sector_weight,
        hard_maximum_sector_weight=hard_maximum_sector_weight,
        minimum_distinct_sectors=minimum_distinct_sectors,
    )
    return set(selected["security_id"].astype(str))


def _edge_swing_predictions(
    *,
    request: PredictionRequest,
    context: pd.DataFrame,
    bundle: PromotedSwingBundle,
    bundle_sha256: str,
    threshold: float,
    selected_security_ids: set[str],
    contract: StrategyContract,
    model_as_of_utc: datetime,
    data_as_of_utc: datetime,
    live_input_manifest_sha256: str,
    catalyst_authority_sha256: str,
) -> list[SwingPrediction]:
    by_ticker = {
        str(row["ticker"]).upper(): row
        for _, row in context.iterrows()
    }
    ranked = context.sort_values(
        ["__probability", "security_id"],
        ascending=[False, True],
        kind="stable",
    )
    ranks = {
        str(row["security_id"]): rank
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1)
    }
    predictions: list[SwingPrediction] = []
    for ticker in request.tickers:
        row = by_ticker.get(ticker)
        if row is None:
            predictions.append(
                SwingPrediction(
                    ticker=ticker,
                    signal="abstain",
                    action="abstain",
                    abstention_reasons=["out_of_universe"],
                    model_id=bundle.model_id,
                    serving_bundle_sha256=bundle_sha256,
                    model_as_of_utc=model_as_of_utc,
                    data_as_of_utc=data_as_of_utc,
                    feature_schema_version=bundle.feature_schema_version,
                    classifier_score=None,
                    regressor_score=None,
                    unified_score=None,
                    readiness=ReadinessInfo(
                        status=INVALID,
                        reasons=["Ticker is absent from the verified live reference universe."],
                        daily_bar_count=0,
                        required_bar_count=contract.swing.minimum_warmup_sessions,
                        price_feed="sip",
                        model_status="promoted",
                        source_status="unavailable",
                    ),
                    lineage={
                        "model_artifact_sha256": bundle.model_artifact_sha256,
                        "live_input_manifest_sha256": live_input_manifest_sha256,
                        "catalyst_authority_sha256": catalyst_authority_sha256,
                    },
                )
            )
            continue
        probability = _required_edge_float(row, "__probability")
        security_id = str(row["security_id"])
        selected_for_policy = security_id in selected_security_ids
        action = (
            "watch_for_entry"
            if selected_for_policy
            else "observe_ranked_candidate"
            if probability >= threshold
            else "avoid"
            if probability <= 0.40
            else "hold_off"
        )
        signal = (
            "positive_setup"
            if action == "watch_for_entry"
            else "ranked_candidate"
            if action == "observe_ranked_candidate"
            else "low_probability"
            if action == "avoid"
            else "neutral"
        )
        catalyst = _edge_catalyst_confirmation(row, positive_setup=probability >= threshold)
        close = _required_edge_float(row, "close")
        atr_pct = _required_edge_float(row, "atr_pct_14")
        decision_time = _required_edge_datetime(row, "decision_time_utc")
        predictions.append(
            SwingPrediction(
                ticker=ticker,
                date=str(row["session_date_et"]),
                probability=probability,
                decision_score=probability,
                classifier_score=_float_or_none(row.get("__classifier_probability")),
                regressor_score=_float_or_none(row.get("__regressor_probability")),
                unified_score=_float_or_none(row.get("__unified_probability")),
                model_prediction=int(probability >= threshold),
                signal=signal,
                action=action,
                rank=ranks[security_id],
                selection_eligible=probability >= threshold,
                selected_for_policy=selected_for_policy,
                close=close,
                return_1d=_required_edge_float(row, "return_1d"),
                volume_z20=_required_edge_float(row, "volume_z20"),
                news_count=_required_edge_float(row, "event_count_3d"),
                event_count=_required_edge_float(row, "event_count_3d"),
                sentiment_mean=_required_edge_float(row, "sentiment_mean_3d"),
                catalyst=catalyst,
                benchmark_context=_edge_benchmark_context(row),
                managed_risk=SwingManagedRiskContext(
                    entry_reference="next_session_open",
                    atr_fraction_of_latest_close=atr_pct,
                    target_distance_fraction=(
                        contract.swing.target_atr_multiple * atr_pct
                    ),
                    stop_distance_fraction=(
                        contract.swing.stop_atr_multiple * atr_pct
                    ),
                    target_atr_multiple=contract.swing.target_atr_multiple,
                    stop_atr_multiple=contract.swing.stop_atr_multiple,
                    maximum_holding_sessions=10,
                    exit_rule=contract.swing.exit_rule,
                    round_trip_cost_bps=contract.swing.round_trip_cost_bps,
                ),
                model_as_of_utc=model_as_of_utc,
                data_as_of_utc=data_as_of_utc,
                feature_schema_version=bundle.feature_schema_version,
                model_id=bundle.model_id,
                serving_bundle_sha256=bundle_sha256,
                readiness=ReadinessInfo(
                    status=VALID,
                    timeframe="daily",
                    daily_bar_count=int(_required_edge_float(row, "daily_bar_count")),
                    required_bar_count=contract.swing.minimum_warmup_sessions,
                    latest_price_date=str(row["session_date_et"]),
                    price_feed=str(row["price_feed"]),
                    benchmark_status="SPY, QQQ, and sector context available",
                    market_context_status="separate overlay; not used by estimator",
                    model_status="promoted",
                    source_status="Alpaca SIP/all and Alpaca catalyst coverage verified",
                ),
                drivers={
                    "model_probability": probability,
                    "promoted_probability_threshold": threshold,
                    "atr_pct_14": atr_pct,
                    "return_20d": _required_edge_float(row, "return_20d"),
                    "relative_return_20d_vs_spy": _required_edge_float(
                        row, "rel_return_20d_vs_spy"
                    ),
                    "decision_time_utc": decision_time.isoformat(),
                    "sector": str(row["sector"]),
                    "primary_benchmark": str(row["primary_benchmark"]),
                },
                lineage={
                    "serving_bundle_sha256": bundle_sha256,
                    "model_artifact_sha256": bundle.model_artifact_sha256,
                    "strategy_contract_sha256": bundle.strategy_contract_sha256,
                    "live_input_manifest_sha256": live_input_manifest_sha256,
                    "catalyst_authority_sha256": catalyst_authority_sha256,
                },
            )
        )
    return predictions


def _edge_benchmark_context(row: pd.Series) -> list[SwingBenchmarkContext]:
    stock_5d = _required_edge_float(row, "return_5d")
    stock_20d = _required_edge_float(row, "return_20d")
    specifications = (
        ("SPY", "broad_market", "spy"),
        ("QQQ", "growth_market", "qqq"),
        (str(row["primary_benchmark"]), "sector", "sector"),
    )
    output: list[SwingBenchmarkContext] = []
    for symbol, role, prefix in specifications:
        benchmark_5d = _required_edge_float(row, f"{prefix}_return_5d")
        benchmark_20d = _required_edge_float(row, f"{prefix}_return_20d")
        output.append(
            SwingBenchmarkContext(
                symbol=symbol,
                role=cast(Any, role),
                stock_return_5d=stock_5d,
                benchmark_return_5d=benchmark_5d,
                excess_return_5d=stock_5d - benchmark_5d,
                stock_return_20d=stock_20d,
                benchmark_return_20d=benchmark_20d,
                excess_return_20d=stock_20d - benchmark_20d,
            )
        )
    return output


def _edge_catalyst_confirmation(
    row: pd.Series,
    *,
    positive_setup: bool,
) -> CatalystConfirmationInfo:
    count = int(_required_edge_float(row, "event_count_3d"))
    sentiment = _required_edge_float(row, "sentiment_mean_3d")
    relevance = _required_edge_float(row, "event_relevance_mean_3d")
    latest = _optional_edge_datetime(row.get("latest_event_feature_available_at_utc"))
    decision = _required_edge_datetime(row, "decision_time_utc")
    minutes = (
        max(0.0, (decision - latest).total_seconds() / 60.0)
        if latest is not None
        else None
    )
    if count == 0:
        status, direction = "absent", "none"
    elif sentiment > 0.10:
        status, direction = ("confirmed" if positive_setup else "mixed"), "positive"
    elif sentiment < -0.10:
        status, direction = ("conflicting" if positive_setup else "confirmed"), "negative"
    else:
        status, direction = "mixed", "mixed"
    return CatalystConfirmationInfo(
        status=cast(Any, status),
        direction=cast(Any, direction),
        score=max(-1.0, min(1.0, sentiment * relevance)),
        event_count=count,
        source_diversity=1 if count else 0,
        sentiment=sentiment,
        relevance=relevance,
        minutes_since_latest=minutes,
        material_event_count=count,
        reasons=["Alpaca ticker news is incorporated in the promoted estimator."],
    )


def _edge_swing_response(
    *,
    request: PredictionRequest,
    model: ModelInfo,
    predictions: list[SwingPrediction],
    context: pd.DataFrame,
    bundle: PromotedSwingBundle,
    live_input_manifest_sha256: str,
    catalyst_authority_sha256: str,
    source_watermarks: dict[str, str],
) -> PredictionResponse:
    request_id = str(uuid4())
    latest = context.sort_values("decision_time_utc", kind="stable").groupby(
        "ticker", as_index=False
    ).tail(1)
    requested = latest.loc[latest["ticker"].astype(str).str.upper().isin(request.tickers)]
    row_evidence = [
        PredictionRowEvidenceV1(
            ticker=str(row["ticker"]).upper(),
            view="swing",
            decision_time_utc=_required_edge_datetime(row, "decision_time_utc"),
            feature_available_at_utc=_required_edge_datetime(row, "feature_available_at_utc"),
            canonical_security_id=str(row["security_id"]),
            decision_group_id=str(row["decision_group_id"]),
            session_date_et=str(row["session_date_et"]),
            primary_benchmark=str(row["primary_benchmark"]),
            market_regime=str(row["market_regime"]),
            sector=str(row["sector"]),
            price_feed=str(row["price_feed"]),
        )
        for _, row in requested.iterrows()
    ]
    cutoff = max((row.decision_time_utc for row in row_evidence), default=request.as_of or datetime.now(UTC))
    bundle_sha256 = bundle.sha256()
    policy_sha256 = hashlib.sha256(
        json.dumps(
            {
                "bundle_sha256": bundle_sha256,
                "horizon_sessions": 10,
                "role": "prediction_intelligence_only_no_alerts_or_execution",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    evidence = PredictionEvidenceV3(
        request_id=request_id,
        correlation_id=request.correlation_id or request_id,
        prediction_cutoff_utc=cutoff,
        row_feature_availability=row_evidence,
        feature_artifacts={
            "swing": FeatureArtifactIdentityV1(
                mode="swing",
                artifact_sha256=live_input_manifest_sha256,
                source_artifact_sha256=catalyst_authority_sha256,
                source_artifact_type="edge_rebuild_live_swing_inputs",
                feature_schema_version=bundle.feature_schema_version,
            )
        },
        release_id=bundle_sha256,
        model_release_ids={"swing": bundle_sha256},
        view_serving_bundle_ids={"swing": bundle_sha256},
        serving_bundle_sha256=_serving_bundle_set_sha256({"swing": bundle_sha256}),
        model_artifact_sha256={"swing": bundle.model_artifact_sha256},
        source_watermarks={"swing": source_watermarks},
        resolved_horizons={"swing": "10b"},
        view_prediction_cutoffs_utc={"swing": cutoff},
        view_prediction_policy_sha256={"swing": policy_sha256},
        serving_policy_id="edge_rebuild.swing_prediction_intelligence.v1",
        serving_policy_sha256=policy_sha256,
        identity_status="complete",
    )
    unified = [
        UnifiedTickerPrediction(
            ticker=row.ticker,
            swing=row,
            final_signal=row.signal,
            readiness_status=row.readiness.status,
            errors=list(row.abstention_reasons),
        )
        for row in predictions
    ]
    return PredictionResponse(
        request_id=request_id,
        mode="swing",
        data_source="live",
        horizon="10b",
        resolved_horizons={"swing": "10b"},
        models={"swing": model},
        predictions=unified,
        evidence=evidence,
    )


def _required_edge_float(row: pd.Series, column: str) -> float:
    value = _float_or_none(row.get(column))
    if value is None:
        raise DataReadinessError(f"live swing context is missing finite {column}")
    return value


def _required_edge_datetime(row: pd.Series, column: str) -> datetime:
    value = _aware_datetime_or_none(row.get(column))
    if value is None:
        raise DataReadinessError(f"live swing context is missing {column}")
    return value


def _optional_edge_datetime(value: object) -> datetime | None:
    return _aware_datetime_or_none(value)


def _risk_adjusted_intraday_score(
    row: pd.Series,
    opportunity_column: str,
    downside_column: str,
) -> float:
    return intraday_decision_score(
        _float_or_none(row.get(opportunity_column)),
        _float_or_none(row.get(downside_column)),
    )


def _prediction_policy_for_model(model: ModelInfo) -> PredictionSelectionPolicy:
    if model.prediction_policy is None or model.prediction_policy_sha256 is None:
        raise ValueError("model prediction policy identity is missing")
    return parse_prediction_policy(
        model.prediction_policy,
        expected_sha256=model.prediction_policy_sha256,
    )


def _serving_policy_bundle_sha256(
    view_policy_hashes: Mapping[str, str],
) -> str:
    payload = {
        "contract_version": SERVING_POLICY_ID,
        "view_prediction_policy_sha256": dict(sorted(view_policy_hashes.items())),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _serving_bundle_set_sha256(view_bundle_ids: Mapping[str, str]) -> str:
    payload = {
        "contract_version": "market_predictor.serving_bundle_set.v1",
        "view_serving_bundle_ids": dict(sorted(view_bundle_ids.items())),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_requested_tickers(
    frame: pd.DataFrame,
    requested: set[str],
    *,
    timeframe: str,
) -> None:
    available = set(frame["ticker"].astype(str))
    missing = sorted(requested.difference(available))
    if missing:
        raise ValueError(
            f"no {timeframe} feature rows found for requested tickers: "
            f"{', '.join(missing)}"
        )


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
            "action": "abstain",
            "abstention_reasons": list(
                dict.fromkeys([*row.abstention_reasons, reason])
            ),
            "rank": None,
            "selection_eligible": False,
            "selected_for_policy": False,
            "classifier_score": None,
            "regressor_score": None,
            "unified_score": None,
            "managed_risk": None,
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
            "selection_eligible": False,
            "selected_for_policy": False,
            "readiness": readiness,
        }
    )


def _combine_evidence(
    request: PredictionRequest,
    *,
    request_id: str,
    evidence_parts: list[PredictionEvidenceV3],
    data_source: PredictionDataSource,
) -> PredictionEvidenceV3:
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
    model_release_ids = {
        mode: release_id
        for evidence in evidence_parts
        for mode, release_id in evidence.model_release_ids.items()
    }
    serving_bundle_ids = {
        mode: bundle_id
        for evidence in evidence_parts
        for mode, bundle_id in evidence.view_serving_bundle_ids.items()
    }
    gaps = [gap for evidence in evidence_parts for gap in evidence.identity_gaps]
    prediction_policy_hashes = {
        mode: digest
        for evidence in evidence_parts
        for mode, digest in evidence.view_prediction_policy_sha256.items()
    }
    release_ids = {evidence.release_id for evidence in evidence_parts if evidence.release_id is not None}
    expected_views = {
        mode
        for evidence in evidence_parts
        for mode in evidence.model_release_ids
    }
    if data_source == "live" and set(serving_bundle_ids) != expected_views:
        gaps.append("prediction views do not have complete serving bundle identities")
    if data_source == "curated":
        identity_status = "research_only"
    elif gaps or any(evidence.identity_status != "complete" for evidence in evidence_parts):
        identity_status = "incomplete"
    else:
        identity_status = "complete"
    return PredictionEvidenceV3(
        request_id=request_id,
        correlation_id=request.correlation_id or request_id,
        prediction_cutoff_utc=max(evidence.prediction_cutoff_utc for evidence in evidence_parts),
        row_feature_availability=rows,
        feature_artifacts=artifacts,
        release_id=next(iter(release_ids)) if len(release_ids) == 1 else None,
        model_release_ids=model_release_ids,
        view_serving_bundle_ids=serving_bundle_ids,
        serving_bundle_sha256=(
            _serving_bundle_set_sha256(serving_bundle_ids)
            if serving_bundle_ids
            else None
        ),
        model_artifact_sha256=model_hashes,
        source_watermarks=watermarks,
        resolved_horizons=horizons,
        view_prediction_cutoffs_utc={
            mode: cutoff
            for evidence in evidence_parts
            for mode, cutoff in evidence.view_prediction_cutoffs_utc.items()
        },
        view_prediction_policy_sha256=prediction_policy_hashes,
        serving_policy_id=SERVING_POLICY_ID,
        serving_policy_sha256=_serving_policy_bundle_sha256(
            prediction_policy_hashes
        ),
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


def _require_bundle_available_at(
    context: ActiveModelContext,
    as_of: datetime,
) -> None:
    generated = _aware_datetime_or_none(context.serving_bundle_generated_at_utc)
    cutoff = _aware_datetime_or_none(as_of)
    if generated is None:
        raise ValueError("serving bundle generation timestamp is missing")
    if cutoff is None:
        raise ValueError("serving bundle availability cutoff is invalid")
    if generated > cutoff + timedelta(minutes=1):
        raise ValueError("serving bundle was generated after the requested as_of")


def _strict_utc_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.map(_aware_datetime_or_none), utc=True)


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _model_drift_identity(model: ModelInfo) -> dict[str, str]:
    values = {
        "model_release_id": model.release_id,
        "model_artifact_sha256": model.artifact_sha256,
        "prediction_policy_sha256": model.prediction_policy_sha256,
        "label_policy_sha256": model.label_policy_sha256,
        "execution_policy_sha256": model.execution_policy_sha256,
    }
    if any(not _is_sha256(value) for value in values.values()):
        raise DataReadinessError(
            "active model identity is incomplete for drift enforcement"
        )
    return {
        field: str(value)
        for field, value in values.items()
    }


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
