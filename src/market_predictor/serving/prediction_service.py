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
from market_predictor.core.errors import DataReadinessError, MarketPredictorError
from market_predictor.core.prediction_contracts import (
    CatalystConfirmationInfo,
    FeatureArtifactIdentityV1,
    ModelInfo,
    PredictionConflictError,
    PredictionDependencyError,
    PredictionDriftBlockedError,
    PredictionEvidenceV4,
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
    TickerPrediction,
)
from market_predictor.governance.drift.policy import DriftAssessmentV3, DriftStateStore
from market_predictor.governance.promotion.bundle_contracts import PromotedSwingBundle
from market_predictor.modeling.feature_reference import (
    feature_reference_names_sha256,
    feature_reference_profile_sha256,
)
from market_predictor.modeling.prediction_selection import SwingPredictionPolicy
from market_predictor.modeling.strategy_contract import StrategyContract, load_strategy_contract
from market_predictor.readiness import INVALID, VALID
from market_predictor.resources import assert_memory_budget, memory_audit
from market_predictor.serving.routes import ServingRoute
from market_predictor.serving.snapshot_store import PredictionSnapshotStore
from market_predictor.serving.swing_features import (
    FileSwingLiveInputProvider,
    SwingLiveInputProvider,
    build_live_swing_features,
)
from market_predictor.serving.swing_inference import (
    LoadedSwingModelGeneration,
    SwingInferenceEngine,
    SwingModelGenerationCache,
)
from market_predictor.swing.contracts.outcome_policy import (
    swing_outcome_policy,
    swing_outcome_policy_sha256,
)
from market_predictor.swing.selection import (
    select_constrained_swing_portfolio,
)

DEFAULT_MODE_HORIZONS = {"swing": "10b"}
SERVING_POLICY_ID = "market_predictor.serving_policy_bundle.v2"
# Serving thresholds are sourced from the canonical prediction policy so the
# served signal semantics and the promotion-evaluated policy share one definition.


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
    trust_store = str(serving.get("attestation_trust_store", "")).strip() if isinstance(serving, dict) else ""
    promotion_gate_policy_sha256 = str(serving.get("promotion_gate_policy_sha256", "")).strip().lower() if isinstance(serving, dict) else ""
    drift_policy_sha256 = str(serving.get("drift_policy_sha256", "")).strip().lower() if isinstance(serving, dict) else ""
    if not trust_store:
        raise ValueError("prediction_serving.attestation_trust_store must be configured")
    if len(promotion_gate_policy_sha256) != 64 or any(character not in "0123456789abcdef" for character in promotion_gate_policy_sha256):
        raise ValueError("prediction_serving.promotion_gate_policy_sha256 must be configured")
    if len(drift_policy_sha256) != 64 or any(character not in "0123456789abcdef" for character in drift_policy_sha256):
        raise ValueError("prediction_serving.drift_policy_sha256 must be configured")
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
                raise ValueError(f"prediction serving route {mode}.{horizon} is missing release_repository")
            if "model" in raw_route:
                raise ValueError(f"prediction serving route {mode}.{horizon} cannot use a direct model path")
            canonical_horizon = _canonical_horizon(str(horizon))
            if normalized_mode == "swing" and canonical_horizon != "10b":
                raise ValueError("public swing serving accepts only the ten-session 10b route")
            if canonical_horizon in parsed:
                raise ValueError(f"duplicate prediction serving route after horizon normalization: {mode}.{canonical_horizon}")
            estimated_resident_gib = float(raw_route.get("estimated_resident_gib", 0.5))
            if estimated_resident_gib <= 0:
                raise ValueError(f"prediction serving route {mode}.{horizon} has an invalid estimated_resident_gib")
            max_model_bytes = int(raw_route.get("max_model_bytes", 512 * 1024 * 1024))
            max_feature_bytes = int(raw_route.get("max_feature_bytes", 512 * 1024 * 1024))
            max_feature_rows = int(raw_route.get("max_feature_rows", 250_000))
            if min(max_model_bytes, max_feature_bytes, max_feature_rows) < 1:
                raise ValueError(f"prediction serving route {mode}.{horizon} has invalid model/feature artifact limits")
            parsed[canonical_horizon] = ServingRoute(
                repository=Path(repository),
                attestation_trust_store=Path(trust_store),
                promotion_gate_policy_sha256=promotion_gate_policy_sha256,
                drift_policy_sha256=drift_policy_sha256,
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
        persist_snapshots: bool = True,
        routes: Mapping[str, Mapping[str, ServingRoute]],
        memory_budget_gib: float = 4.0,
        memory_headroom_gib: float = 0.25,
        max_concurrent_inference: int = 1,
        max_tickers_per_request: int = 100,
        inference_memory_reservation_gib: float = 0.5,
        reject_unknown_memory: bool = False,
        drift_state_store: DriftStateStore | None = None,
        enforce_drift: bool = True,
        maximum_drift_assessment_age_minutes: int = 1_440,
        swing_live_input_provider: SwingLiveInputProvider | None = None,
        swing_model_generation_cache: SwingModelGenerationCache | None = None,
    ) -> None:
        self.root = Path(root)
        self.snapshot_store = snapshot_store or PredictionSnapshotStore(self.root / "data/predictions/snapshots")
        self.swing_live_input_provider = swing_live_input_provider
        self.persist_snapshots = persist_snapshots
        if not routes:
            raise ValueError("at least one prediction serving route is required")
        if set(routes) != {"swing"}:
            raise ValueError("prediction serving routes must be swing routes only")
        self.routes = {mode: dict(mode_routes) for mode, mode_routes in routes.items()}
        if memory_budget_gib <= 0 or not 0 < memory_headroom_gib < memory_budget_gib:
            raise ValueError("runtime memory budget and headroom are invalid")
        self.memory_budget_gib = memory_budget_gib
        self.memory_headroom_gib = memory_headroom_gib
        if self.swing_live_input_provider is None:
            self.swing_live_input_provider = FileSwingLiveInputProvider(
                self.root / "data/live/swing",
                memory_budget_gib=memory_budget_gib,
                memory_headroom_gib=memory_headroom_gib,
            )
        self.swing_model_generation_cache = swing_model_generation_cache or SwingModelGenerationCache(
            memory_budget_gib=memory_budget_gib,
            memory_headroom_gib=memory_headroom_gib,
        )
        maximum_artifact_bytes = int((memory_budget_gib - memory_headroom_gib) * 1024**3)
        for mode_routes in self.routes.values():
            for route in mode_routes.values():
                if route.max_model_bytes + route.max_feature_bytes > maximum_artifact_bytes:
                    raise ValueError("combined route artifact byte limits exceed the memory safety threshold")
                if enforce_drift and not _is_sha256(route.drift_policy_sha256):
                    raise ValueError("serving route has no valid drift policy identity")
        if max_concurrent_inference != 1 or max_tickers_per_request < 1:
            raise ValueError("inference concurrency must be one and the ticker limit must be positive")
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
        if maximum_drift_assessment_age_minutes < 1:
            raise ValueError("maximum drift assessment age must be positive")
        self.drift_state_store = drift_state_store or DriftStateStore(self.root / "data/monitoring/drift")
        self.enforce_drift = enforce_drift
        self.maximum_drift_assessment_age = timedelta(minutes=maximum_drift_assessment_age_minutes)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        if len(request.tickers) > self.max_tickers_per_request:
            raise PredictionValidationError
        try:
            with self.admission.lease(estimated_incremental_gib=self.inference_memory_reservation_gib):
                response = self.predict_swing(request)
                if not self.persist_snapshots:
                    return response
                return self.snapshot_store.record(request, response)
        except PredictionServiceError:
            raise
        except OSError as exc:
            raise PredictionDependencyError from exc

    def predict_swing(self, request: PredictionRequest) -> PredictionResponse:
        try:
            _validate_swing_requested_models(request.requested_models)
            route, resolved_horizon = self._serving_route("swing", request)
            try:
                contract = load_strategy_contract(self._resolve(Path("configs/edge_rebuild_strategy_contract.toml")))
            except DataReadinessError as exc:
                repository = self._resolve(route.repository)
                if not (repository / "active_generation.json").is_file():
                    raise PredictionModelUnavailableError from exc
                raise
            generation = self._edge_swing_generation(route, contract=contract)
            bundle = generation.bundle
            engine = SwingInferenceEngine(generation)
            prediction_policy, prediction_policy_sha256 = _swing_prediction_policy(
                probability_threshold=engine.threshold,
                contract=contract,
            )
            model = _edge_swing_model_info(
                generation,
                bundle_root=self._resolve(route.repository),
                resolved_horizon=resolved_horizon,
                contract=contract,
                prediction_policy=prediction_policy,
                prediction_policy_sha256=prediction_policy_sha256,
            )
            as_of = request.as_of or datetime.now(UTC)
            self._require_actionable_drift(
                mode="swing",
                horizon=resolved_horizon,
                model=model,
                checked_at=as_of,
            )
            if bundle.promoted_at_utc > as_of.astimezone(UTC):
                raise DataReadinessError("promoted swing bundle was unavailable at the requested as_of")
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
            model_features = live.technical_market if generation.bundle.feature_profile == "technical_market" else live.catalyst_full
            raw_scores = engine.predict(
                feature_frame=model_features,
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
            selected_ids = _selected_edge_swing_security_ids(
                scored_context,
                probability_threshold=engine.threshold,
                maximum_trades=contract.swing.maximum_trades_per_decision,
                target_maximum_sector_weight=(contract.swing.target_maximum_sector_weight),
                hard_maximum_sector_weight=(contract.swing.hard_maximum_sector_weight),
                minimum_distinct_sectors=(contract.swing.minimum_distinct_sectors_for_selection),
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
            if not self.swing_model_generation_cache.is_current(
                self._resolve(route.repository),
                generation,
            ):
                raise DataReadinessError("active swing model generation changed during inference")
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

        for _horizon, route in sorted(self.routes["swing"].items()):
            try:
                contract = load_strategy_contract(self._resolve(Path("configs/edge_rebuild_strategy_contract.toml")))
                self._edge_swing_generation(route, contract=contract)
            except (DataReadinessError, PredictionModelUnavailableError):
                # Absence is an expected fail-closed deployment state;
                # the API remains available and returns a typed 503.
                continue

    def health(self, *, as_of: datetime | None = None) -> dict[str, object]:
        """Return deployment readiness from the verified cached generations."""

        checked_at = as_of or datetime.now(UTC)
        components: dict[str, dict[str, object]] = {}
        ready = True
        for mode, mode_routes in self.routes.items():
            for horizon, route in mode_routes.items():
                name = f"model:{mode}:{horizon}"
                try:
                    contract = load_strategy_contract(self._resolve(Path("configs/edge_rebuild_strategy_contract.toml")))
                    generation = self._edge_swing_generation(
                        route,
                        contract=contract,
                    )
                    bundle = generation.bundle
                    engine = SwingInferenceEngine(generation)
                    prediction_policy, prediction_policy_sha256 = _swing_prediction_policy(
                        probability_threshold=engine.threshold,
                        contract=contract,
                    )
                    info = _edge_swing_model_info(
                        generation,
                        bundle_root=self._resolve(route.repository),
                        resolved_horizon=horizon,
                        contract=contract,
                        prediction_policy=prediction_policy,
                        prediction_policy_sha256=prediction_policy_sha256,
                    )
                    components[name] = {
                        "status": "ready",
                        "model_status": bundle.model_status,
                        "artifact_sha256": bundle.model_artifact_sha256,
                        "serving_bundle_sha256": bundle.sha256(),
                        "horizon_sessions": bundle.horizon_sessions,
                    }
                    if self.swing_live_input_provider is None:
                        raise DataReadinessError("swing live-input provider is unavailable")
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
                    drift_name = f"drift:{mode}:{horizon}"
                    if not self.enforce_drift:
                        components[drift_name] = {
                            "status": "disabled",
                            "reason": "drift enforcement is disabled",
                        }
                    else:
                        assessment = self._load_drift_assessment(
                            mode=mode,
                            horizon=horizon,
                            model=info,
                            checked_at=checked_at,
                        )
                        components[drift_name] = {
                            "status": ("ready" if assessment.actionability == "actionable" else "not_ready"),
                            **assessment.model_dump(mode="json"),
                        }
                        if assessment.actionability != "actionable":
                            ready = False
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
        components["inference_admission"] = {
            "status": "ready",
            **self.admission.snapshot().to_record(),
        }

        return {
            "status": "ready" if ready else "not_ready",
            "checked_at_utc": checked_at.astimezone(UTC).isoformat(),
            "data_source": "live",
            "components": components,
        }

    def _require_actionable_drift(
        self,
        *,
        mode: str,
        horizon: str,
        model: ModelInfo,
        checked_at: datetime | None = None,
    ) -> DriftAssessmentV3 | None:
        if not self.enforce_drift:
            return None
        try:
            assessment = self._load_drift_assessment(
                mode=mode,
                horizon=horizon,
                model=model,
                checked_at=checked_at or datetime.now(UTC),
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
    ) -> DriftAssessmentV3:
        if not self.enforce_drift:
            raise DataReadinessError("drift enforcement is disabled")
        route_identity = _model_drift_identity(model)
        assessment = self.drift_state_store.load(
            mode,
            horizon,
            route_identity["model_release_id"],
        )
        if any(getattr(assessment, field) != value for field, value in route_identity.items()):
            raise DataReadinessError("route drift assessment model or policy identity mismatch")
        expected_policy_sha256 = self.routes[mode][horizon].drift_policy_sha256
        if assessment.policy_sha256 != expected_policy_sha256:
            raise DataReadinessError("route drift assessment policy identity mismatch")
        evaluated_at = assessment.evaluated_at_utc.astimezone(UTC)
        if checked_at.astimezone(UTC) - evaluated_at > self.maximum_drift_assessment_age:
            raise DataReadinessError("route drift assessment is stale")
        if evaluated_at > checked_at.astimezone(UTC):
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
                attestation_trust_store_path=self._resolve(route.attestation_trust_store),
                promotion_gate_policy_sha256=route.promotion_gate_policy_sha256,
                maximum_model_bytes=route.max_model_bytes,
                estimated_resident_gib=route.estimated_resident_gib,
            )
        except (MarketPredictorError, OSError, TypeError, ValueError) as exc:
            raise PredictionModelUnavailableError from exc

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

    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

def _edge_swing_model_info(
    generation: LoadedSwingModelGeneration,
    *,
    bundle_root: Path,
    resolved_horizon: str,
    contract: StrategyContract,
    prediction_policy: dict[str, object],
    prediction_policy_sha256: str,
) -> ModelInfo:
    bundle = generation.bundle
    feature_reference = generation.model_payload.get("feature_reference_profile")
    if not isinstance(feature_reference, dict):
        raise DataReadinessError("promoted swing model has no feature reference")
    feature_reference_sha256 = feature_reference_profile_sha256(feature_reference)
    feature_names_sha256 = feature_reference_names_sha256(feature_reference)
    if (
        generation.model_payload.get("feature_reference_profile_sha256")
        != feature_reference_sha256
    ):
        raise DataReadinessError("promoted swing feature reference identity is invalid")
    return ModelInfo(
        path=str(bundle_root / "generations" / generation.generation_id / bundle.model_artifact_path),
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
        label_policy_sha256=swing_outcome_policy_sha256(contract.swing),
        label_policy=swing_outcome_policy(contract.swing),
        execution_policy_sha256=bundle.execution_policy_sha256,
        prediction_policy_sha256=prediction_policy_sha256,
        prediction_policy=prediction_policy,
        feature_reference_profile_sha256=feature_reference_sha256,
        feature_reference_names_sha256=feature_names_sha256,
    )


def _validate_swing_requested_models(requested_models: list[str] | None) -> None:
    if requested_models is None:
        return
    requested = tuple(requested_models)
    if not requested or len(requested) != len(set(requested)):
        raise PredictionValidationError
    allowed = {"all", "classifier", "xgboost_regressor"}
    if not set(requested).issubset(allowed):
        raise PredictionValidationError
    if "all" in requested:
        if len(requested) != 1:
            raise PredictionValidationError
        return
    if "classifier" not in requested:
        raise PredictionValidationError


def _swing_prediction_policy(
    *,
    probability_threshold: float,
    contract: StrategyContract,
) -> tuple[dict[str, object], str]:
    policy = SwingPredictionPolicy(
        horizon_sessions=contract.swing.horizon_sessions,
        minimum_probability=probability_threshold,
        maximum_predictions_per_decision=contract.swing.maximum_trades_per_decision,
        target_maximum_sector_weight=contract.swing.target_maximum_sector_weight,
        hard_maximum_sector_weight=contract.swing.hard_maximum_sector_weight,
        minimum_distinct_sectors=contract.swing.minimum_distinct_sectors_for_selection,
    )
    return policy.specification(), policy.sha256()


def _selected_edge_swing_security_ids(
    frame: pd.DataFrame,
    *,
    probability_threshold: float,
    maximum_trades: int,
    target_maximum_sector_weight: float,
    hard_maximum_sector_weight: float,
    minimum_distinct_sectors: int,
) -> set[str]:
    eligible = frame.loc[pd.to_numeric(frame["__probability"], errors="coerce").ge(probability_threshold)]
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
    by_ticker = {str(row["ticker"]).upper(): row for _, row in context.iterrows()}
    ranked = context.sort_values(
        ["__probability", "security_id"],
        ascending=[False, True],
        kind="stable",
    )
    ranks = {str(row["security_id"]): rank for rank, (_, row) in enumerate(ranked.iterrows(), start=1)}
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
                    target_distance_fraction=(contract.swing.target_atr_multiple * atr_pct),
                    stop_distance_fraction=(contract.swing.stop_atr_multiple * atr_pct),
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
                    "relative_return_20d_vs_spy": _required_edge_float(row, "rel_return_20d_vs_spy"),
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
    minutes = max(0.0, (decision - latest).total_seconds() / 60.0) if latest is not None else None
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
    latest = context.sort_values("decision_time_utc", kind="stable").groupby("ticker", as_index=False).tail(1)
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
            market_cap_bucket=str(row["market_cap_bucket"]),
            liquidity_bucket=str(row["liquidity_bucket"]),
            price_feed=str(row["price_feed"]),
            decision_atr=(
                _required_edge_float(row, "atr_pct_14")
                * _required_edge_float(row, "close")
            ),
        )
        for _, row in requested.iterrows()
    ]
    cutoff = max((row.decision_time_utc for row in row_evidence), default=request.as_of or datetime.now(UTC))
    bundle_sha256 = bundle.sha256()
    policy_sha256 = model.prediction_policy_sha256
    if policy_sha256 is None or model.prediction_policy is None:
        raise DataReadinessError("swing prediction policy identity is incomplete")
    evidence = PredictionEvidenceV4(
        request_id=request_id,
        correlation_id=request.correlation_id or request_id,
        prediction_cutoff_utc=cutoff,
        row_feature_availability=row_evidence,
        feature_artifacts={
            "swing": FeatureArtifactIdentityV1(
                mode="swing",
                artifact_sha256=live_input_manifest_sha256,
                source_artifact_sha256=catalyst_authority_sha256,
                source_artifact_type="swing_live_inputs",
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
        serving_policy_id="market_predictor.swing_prediction_policy.v1",
        serving_policy_sha256=policy_sha256,
        identity_status="complete",
    )
    rows = [
        TickerPrediction(
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
        predictions=rows,
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


def _serving_bundle_set_sha256(view_bundle_ids: Mapping[str, str]) -> str:
    payload = {
        "contract_version": "market_predictor.serving.bundle_set.v1",
        "view_serving_bundle_ids": dict(sorted(view_bundle_ids.items())),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _canonical_horizon(value: str) -> str:
    return value.strip().lower()


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
        "feature_reference_profile_sha256": (
            model.feature_reference_profile_sha256
        ),
        "feature_reference_names_sha256": model.feature_reference_names_sha256,
    }
    if any(not _is_sha256(value) for value in values.values()):
        raise DataReadinessError("active model identity is incomplete for drift enforcement")
    return {field: str(value) for field, value in values.items()}


def _float_or_none(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(converted):
        return None
    return converted


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
