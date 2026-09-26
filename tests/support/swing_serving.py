"""A promoted ten-session swing service over synthetic live frames, never trading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

import market_predictor.serving.prediction_service as service_module
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.prediction_contracts import ModelInfo, PredictionRequest
from market_predictor.execution_policy import EXECUTION_POLICY_SHA256
from market_predictor.governance.drift.policy import DriftAssessmentV3, DriftPolicyV3
from market_predictor.governance.outcomes.contracts import content_sha256
from market_predictor.governance.promotion.bundle_contracts import (
    canonical_payload_sha256,
    ordered_values_sha256,
    validate_promoted_bundle,
)
from market_predictor.modeling.feature_reference import (
    feature_reference_names_sha256,
    feature_reference_profile_sha256,
)
from market_predictor.modeling.strategy_contract import StrategyContract, load_strategy_contract
from market_predictor.serving.prediction_service import PredictionService
from market_predictor.serving.routes import ServingRoute
from market_predictor.serving.swing_features import (
    SWING_LIVE_IDENTITY_COLUMNS,
    SWING_LIVE_REQUIRED_WATERMARKS,
    SwingLiveFeatureFrames,
    SwingLiveInputs,
)
from market_predictor.serving.swing_inference import LoadedSwingModelGeneration
from market_predictor.swing.contracts.model_artifact import SWING_CANDIDATE_MODEL_SCHEMA
from market_predictor.swing.features.panel import (
    SWING_FEATURE_PANEL_SCHEMA,
    swing_model_feature_columns,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 7, 8, 22, 5, tzinfo=UTC)
DECISION = pd.Timestamp("2026-07-08T22:00:00Z")
TEST_GATE_POLICY_SHA256 = canonical_payload_sha256({"test_fixture": True})
TEST_DRIFT_POLICY_SHA256 = DriftPolicyV3().sha256()


class _Estimator:
    def __init__(self, expected_width: int) -> None:
        self.expected_width = expected_width

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        assert values.shape[1] == self.expected_width
        probability = np.full(len(values), 0.72)
        return np.column_stack((1.0 - probability, probability))


class _Calibrator:
    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        probability = np.clip(values[:, 0], 0.0, 1.0)
        return np.column_stack((1.0 - probability, probability))


class _Fitted:
    calibrator = _Calibrator()

    def __init__(self, feature_columns: tuple[str, ...]) -> None:
        self.feature_columns = feature_columns
        self.estimator = _Estimator(len(feature_columns))


class LiveInputs:
    def load(
        self,
        *,
        as_of_utc: datetime,
        maximum_bytes: int | None = None,
        maximum_rows: int | None = None,
    ) -> SwingLiveInputs:
        del maximum_bytes, maximum_rows
        assert as_of_utc == NOW
        return SwingLiveInputs(
            stock_daily_bars=pd.DataFrame(),
            benchmark_daily_bars=pd.DataFrame(),
            point_in_time_memberships=pd.DataFrame(),
            catalyst_authority_directory=Path("unused"),
            catalyst_authority_sha256="c" * 64,
            manifest_path=Path("unused-manifest.json"),
            manifest_sha256="d" * 64,
            generated_at_utc=NOW,
            source_watermarks={key: DECISION.isoformat() for key in SWING_LIVE_REQUIRED_WATERMARKS},
            generation_id="e" * 64,
            pointer_sha256="f" * 64,
        )


class UnavailableInputs:
    def load(
        self,
        *,
        as_of_utc: datetime,
        maximum_bytes: int | None = None,
        maximum_rows: int | None = None,
    ) -> SwingLiveInputs:
        del as_of_utc, maximum_bytes, maximum_rows
        raise DataReadinessError("live input generation unavailable")


class GenerationCache:
    def __init__(self, generation: LoadedSwingModelGeneration) -> None:
        self.generation = generation
        self.current = True
        self.loads = 0

    def get(self, *_args: object, **_kwargs: object) -> LoadedSwingModelGeneration:
        self.loads += 1
        return self.generation

    def is_current(self, *_args: object, **_kwargs: object) -> bool:
        return self.current


@dataclass(frozen=True)
class SwingServing:
    service: PredictionService
    contract: StrategyContract
    bundle: dict[str, Any]
    generation_cache: GenerationCache


def swing_serving(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    generation_cache: type[GenerationCache] = GenerationCache,
    excluded_tickers: tuple[str, ...] = (),
    **service_options: Any,
) -> SwingServing:
    """Build the promoted swing service; drift enforcement stays on unless disabled."""
    contract_path = root / "configs" / "edge_rebuild_strategy_contract.toml"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_bytes((ROOT / "configs" / contract_path.name).read_bytes())
    contract = load_strategy_contract(contract_path)
    features = swing_model_feature_columns(contract=contract, catalyst=False)
    feature_reference = {
        feature: {"rows": 100, "observed": 100, "missing_rate": 0.0, "mean": 0.0, "std": 1.0}
        for feature in features
    }
    bundle_root = root / "models" / "swing"
    model_path = bundle_root / "model" / "model.joblib"
    evidence_path = model_path.with_suffix(model_path.suffix + ".promotion.attestation.json")
    model_path.parent.mkdir(parents=True)
    model_payload = {
        "schema": SWING_CANDIDATE_MODEL_SCHEMA,
        "status": "candidate",
        "promotion_permitted": False,
        "candidate_id": "swing-promoted-test",
        "model_family": "swing_baseline",
        "strategy_contract_sha256": contract.sha256(),
        "execution_policy_sha256": EXECUTION_POLICY_SHA256,
        "feature_columns": features,
        "feature_reference_profile": feature_reference,
        "feature_reference_profile_sha256": feature_reference_profile_sha256(feature_reference),
        "feature_reference_names_sha256": feature_reference_names_sha256(feature_reference),
        "ablation_profile": "technical_market",
        "probability_thresholds": {"classifier": 0.60},
        "fitted_models": {"classifier": _Fitted(features[:1])},
    }
    joblib.dump(model_payload, model_path)
    evidence_path.write_text('{"promotion":"passed"}\n', encoding="utf-8")
    bundle = {
        "schema_version": "edge_rebuild.promoted_bundle.v2",
        "mode": "swing",
        "strategy_id": "swing",
        "horizon_sessions": 10,
        "model_family": "swing_baseline",
        "feature_profile": "technical_market",
        "catalyst_policy": "confirmation_overlay",
        "model_id": "swing-promoted-test",
        "model_status": "promoted",
        "promotion_permitted": True,
        "model_artifact_path": "model/model.joblib",
        "model_artifact_sha256": file_sha256(model_path),
        "promotion_evidence_path": "model/model.joblib.promotion.attestation.json",
        "promotion_evidence_sha256": file_sha256(evidence_path),
        "promotion_attestation_id": "f" * 64,
        "promotion_gate_policy_sha256": TEST_GATE_POLICY_SHA256,
        "approved_by_principal_id": "test-approver",
        "promoted_at_utc": "2026-07-08T20:00:00Z",
        "feature_schema_version": SWING_FEATURE_PANEL_SCHEMA,
        "ordered_feature_columns": features,
        "ordered_feature_sha256": ordered_values_sha256(features),
        "strategy_contract_schema_version": contract.schema_version,
        "strategy_contract_sha256": contract.sha256(),
        "execution_policy_sha256": EXECUTION_POLICY_SHA256,
        "market_data_provider": "alpaca",
        "market_data_feed": "sip",
        "market_data_adjustment": "all",
        "model_source_families": [],
        "model_source_families_sha256": ordered_values_sha256(()),
        "catalyst_overlay_source_families": ["alpaca"],
        "catalyst_overlay_source_families_sha256": ordered_values_sha256(("alpaca",)),
        "catalyst_policy_sha256": "e" * 64,
        "global_context_policy": "ranking_overlay",
        "global_authority_schema_version": "edge_rebuild.global_event_authority.v1",
        "global_source_families": ["alpaca"],
        "global_source_families_sha256": ordered_values_sha256(("alpaca",)),
    }
    (bundle_root / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    promoted = validate_promoted_bundle(bundle, strategy_contract=contract)
    cache = generation_cache(
        LoadedSwingModelGeneration(
            generation_id=promoted.sha256(),
            pointer_sha256="a" * 64,
            bundle=promoted,
            model_payload=model_payload,
        )
    )
    live = live_frames(
        technical_features=features,
        catalyst_features=swing_model_feature_columns(contract=contract, catalyst=True),
        excluded_tickers=excluded_tickers,
    )
    monkeypatch.setattr(service_module, "build_live_swing_features", lambda *_args, **_kwargs: live)
    options: dict[str, Any] = {
        "swing_live_input_provider": LiveInputs(),
        "persist_snapshots": False,
        **service_options,
    }
    service = PredictionService(
        root,
        routes={
            "swing": {
                "10b": ServingRoute(
                    repository=Path("models/swing"),
                    attestation_trust_store=Path("unused.json"),
                    promotion_gate_policy_sha256=TEST_GATE_POLICY_SHA256,
                    drift_policy_sha256=TEST_DRIFT_POLICY_SHA256,
                    bar_timeframe="1Day",
                )
            }
        },
        swing_model_generation_cache=cache,
        **options,
    )
    return SwingServing(service=service, contract=contract, bundle=bundle, generation_cache=cache)


def swing_model_info(serving: SwingServing, monkeypatch: pytest.MonkeyPatch) -> ModelInfo:
    """The served model identity, read from one prediction with drift checks bypassed."""
    with monkeypatch.context() as patched:
        patched.setattr(serving.service, "_require_actionable_drift", lambda **_kwargs: None)
        response = serving.service.predict_swing(PredictionRequest(tickers=["T000"], as_of=NOW))
    return response.models["swing"]


def drift_assessment(
    model: ModelInfo,
    *,
    state: str,
    evaluated_at: datetime,
    policy_sha256: str = TEST_DRIFT_POLICY_SHA256,
) -> DriftAssessmentV3:
    stamp = evaluated_at.isoformat().replace("+00:00", "Z")
    content = {
        "contract_version": "market_predictor.drift_assessment.v3",
        "mode": "swing",
        "horizon": "10b",
        "model_release_id": model.release_id,
        "model_artifact_sha256": model.artifact_sha256,
        "prediction_policy_sha256": model.prediction_policy_sha256,
        "label_policy_sha256": model.label_policy_sha256,
        "execution_policy_sha256": model.execution_policy_sha256,
        "policy_sha256": policy_sha256,
        "performance_report_id": "1" * 64,
        "performance_cohort_id": "2" * 64,
        "feature_artifact_set_sha256": "3" * 64,
        "feature_drift_report_id": "4" * 64,
        "feature_reference_profile_sha256": model.feature_reference_profile_sha256,
        "feature_reference_names_sha256": model.feature_reference_names_sha256,
        "evaluated_at_utc": stamp,
        "state": state,
        "actionability": "actionable" if state == "stable" else "not_ready",
        "reasons": () if state == "stable" else ("selected_policy_performance_severe",),
        "feature_drift_status": "stable",
        "total_predictions": 50,
        "selected_predictions": 10,
        "matured_samples": 10,
        "independent_decision_groups": 10,
        "last_matured_outcome_utc": stamp,
    }
    return DriftAssessmentV3.model_validate({**content, "assessment_id": content_sha256(content)})


def live_frames(
    *,
    technical_features: tuple[str, ...],
    catalyst_features: tuple[str, ...],
    excluded_tickers: tuple[str, ...] = (),
) -> SwingLiveFeatureFrames:
    identities: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    for index in range(60):
        identity = {
            "decision_id": f"decision-{index}",
            "security_id": f"SEC-{index:03d}",
            "ticker": f"T{index:03d}",
            "session_date_et": "2026-07-08",
            "decision_time_utc": DECISION,
        }
        identities.append(identity)
        context_rows.append(
            {
                **identity,
                "decision_group_id": DECISION.isoformat(),
                "feature_available_at_utc": DECISION,
                "sector": f"Sector-{index % 10}",
                "primary_benchmark": f"XL{index % 10}",
                "market_regime": "risk_on",
                "market_cap_bucket": "large_cap",
                "liquidity_bucket": "liquid",
                "price_feed": "sip",
                "adjustment": "all",
                "daily_bar_count": 300,
                "close": 100.0,
                "atr_pct_14": 0.02,
                "return_1d": 0.01,
                "return_5d": 0.03,
                "return_20d": 0.08,
                "volume_z20": 1.2,
                "rel_return_20d_vs_spy": 0.04,
                "spy_return_5d": 0.01,
                "spy_return_20d": 0.04,
                "qqq_return_5d": 0.015,
                "qqq_return_20d": 0.05,
                "sector_return_5d": 0.02,
                "sector_return_20d": 0.06,
                "event_count_3d": 2.0,
                "sentiment_mean_3d": 0.4,
                "event_relevance_mean_3d": 0.8,
                "latest_event_feature_available_at_utc": DECISION - pd.Timedelta(hours=1),
            }
        )
    index = pd.MultiIndex.from_frame(
        pd.DataFrame(identities).loc[:, SWING_LIVE_IDENTITY_COLUMNS],
        names=SWING_LIVE_IDENTITY_COLUMNS,
    )
    return SwingLiveFeatureFrames(
        technical_market=pd.DataFrame(0.1, index=index, columns=technical_features),
        catalyst_full=pd.DataFrame(0.1, index=index, columns=catalyst_features),
        context=pd.DataFrame(context_rows, index=index),
        as_of_utc=pd.Timestamp(NOW),
        decision_time_utc=DECISION,
        session_date_et=DECISION.tz_convert("America/New_York").date(),
        excluded_tickers=excluded_tickers,
    )
