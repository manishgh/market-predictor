from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import market_predictor.serving.prediction_service as service_module
from market_predictor.api import create_app
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.prediction_contracts import (
    PredictionDriftBlockedError,
    PredictionReadinessError,
    PredictionRequest,
    PredictionValidationError,
)
from market_predictor.governance.promotion.bundle_contracts import (
    canonical_payload_sha256,
    ordered_values_sha256,
    validate_promoted_bundle,
)
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.serving.outcome_intents import maturation_intents_from_response
from market_predictor.serving.prediction_service import PredictionService
from market_predictor.serving.routes import ServingRoute
from market_predictor.serving.swing_features import (
    SWING_LIVE_IDENTITY_COLUMNS,
    SWING_LIVE_REQUIRED_WATERMARKS,
    SwingLiveFeatureFrames,
    SwingLiveInputs,
)
from market_predictor.serving.swing_inference import (
    LoadedSwingModelGeneration,
)
from market_predictor.swing.contracts.model_artifact import (
    SWING_CANDIDATE_MODEL_SCHEMA,
)
from market_predictor.swing.features.panel import (
    SWING_FEATURE_PANEL_SCHEMA,
    swing_model_feature_columns,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 7, 8, 22, 5, tzinfo=UTC)
DECISION = pd.Timestamp("2026-07-08T22:00:00Z")
TEST_GATE_POLICY_SHA256 = canonical_payload_sha256({"test_fixture": True})


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


class _Inputs:
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


class _UnavailableInputs:
    def load(
        self,
        *,
        as_of_utc: datetime,
        maximum_bytes: int | None = None,
        maximum_rows: int | None = None,
    ) -> SwingLiveInputs:
        del as_of_utc, maximum_bytes, maximum_rows
        raise DataReadinessError("live input generation unavailable")


class _GenerationCache:
    def __init__(self, generation: LoadedSwingModelGeneration) -> None:
        self.generation = generation
        self.current = True

    def get(self, *_args: object, **_kwargs: object) -> LoadedSwingModelGeneration:
        return self.generation

    def is_current(self, *_args: object, **_kwargs: object) -> bool:
        return self.current


def test_promoted_ten_session_swing_api_returns_human_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path = tmp_path / "configs" / "edge_rebuild_strategy_contract.toml"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes((ROOT / "configs" / contract_path.name).read_bytes())
    contract = load_strategy_contract(contract_path)
    features = swing_model_feature_columns(contract=contract, catalyst=False)
    bundle_root = tmp_path / "models" / "swing"
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
        "feature_columns": features,
        "ablation_profile": "technical_market",
        "probability_thresholds": {"classifier": 0.60},
        "fitted_models": {"classifier": _Fitted(features[:1])},
    }
    joblib.dump(model_payload, model_path)
    attestation_id = "f" * 64
    approver_id = "test-approver"
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
        "promotion_attestation_id": attestation_id,
        "promotion_gate_policy_sha256": TEST_GATE_POLICY_SHA256,
        "approved_by_principal_id": approver_id,
        "promoted_at_utc": "2026-07-08T20:00:00Z",
        "feature_schema_version": SWING_FEATURE_PANEL_SCHEMA,
        "ordered_feature_columns": features,
        "ordered_feature_sha256": ordered_values_sha256(features),
        "strategy_contract_schema_version": contract.schema_version,
        "strategy_contract_sha256": contract.sha256(),
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
    promoted_bundle = validate_promoted_bundle(
        bundle,
        strategy_contract=contract,
        expected_mode="swing",
    )
    generation = LoadedSwingModelGeneration(
        generation_id=promoted_bundle.sha256(),
        pointer_sha256="a" * 64,
        bundle=promoted_bundle,
        model_payload=model_payload,
    )
    live = _live_frames(
        technical_features=features,
        catalyst_features=swing_model_feature_columns(
            contract=contract,
            catalyst=True,
        ),
    )
    monkeypatch.setattr(service_module, "build_live_swing_features", lambda *_args, **_kwargs: live)
    monkeypatch.setattr(
        "market_predictor.governance.promotion.bundle_verification.verify_promotion_attestation",
        lambda *_args, **_kwargs: {
            "attestation_id": attestation_id,
            "promoted_at_utc": "2026-07-08T20:00:00+00:00",
            "candidate": {
                "artifact_sha256": bundle["model_artifact_sha256"],
                "model_run_id": bundle["model_id"],
                "model_schema_version": SWING_CANDIDATE_MODEL_SCHEMA,
            },
            "approver_principal": {"principal_id": approver_id},
            "ledger_receipt": {"result": "passed"},
            "gate_config_sha256": TEST_GATE_POLICY_SHA256,
        },
    )
    generation_cache = _GenerationCache(generation)
    service = PredictionService(
        tmp_path,
        routes={
            "swing": {
                "10b": ServingRoute(
                    repository=Path("models/swing"),
                    attestation_trust_store=Path("unused.json"),
                    promotion_gate_policy_sha256=TEST_GATE_POLICY_SHA256,
                    bar_timeframe="1Day",
                )
            }
        },
        swing_live_input_provider=_Inputs(),
        swing_model_generation_cache=generation_cache,
        persist_snapshots=False,
    )
    with pytest.raises(PredictionDriftBlockedError):
        service.predict_swing(
            PredictionRequest(tickers=["T000"], mode="swing", as_of=NOW)
        )
    drift_checks: list[dict[str, object]] = []
    monkeypatch.setattr(
        service,
        "_require_actionable_drift",
        lambda **kwargs: drift_checks.append(kwargs),
    )
    direct = service.predict_swing(
        PredictionRequest(
            tickers=["T000", "T059", "MISSING"],
            mode="swing",
            as_of=NOW,
        )
    )
    assert direct.predictions[0].swing is not None
    assert direct.evidence is not None
    assert direct.evidence.identity_status == "complete"
    model = direct.models["swing"]
    assert model.prediction_policy is not None
    assert model.prediction_policy_sha256 == direct.evidence.view_prediction_policy_sha256["swing"]
    assert model.prediction_policy["minimum_probability"] == 0.60
    assert (
        model.prediction_policy["maximum_predictions_per_decision"]
        == contract.swing.maximum_trades_per_decision
    )
    intents = maturation_intents_from_response(direct, snapshot_id="a" * 64)
    assert {intent.ticker for intent in intents} == {"T000", "T059"}

    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/predictions/swing",
            json={
                "tickers": ["T000", "T059", "MISSING"],
                "as_of": NOW.isoformat(),
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["horizon"] == "10b"
    scored = payload["predictions"][0]["swing"]
    assert scored["action"] == "watch_for_entry"
    assert scored["probability"] == 0.72
    assert scored["expected_horizon"] == "up to 10 trading sessions"
    assert len(scored["benchmark_context"]) == 3
    assert scored["managed_risk"]["entry_reference"] == "next_session_open"
    assert scored["managed_risk"]["price_levels_available"] is False
    assert scored["managed_risk"]["target_distance_fraction"] > 0
    assert "target_price" not in scored["managed_risk"]
    assert "stop_price" not in scored["managed_risk"]
    assert scored["catalyst"]["event_count"] == 2
    assert scored["lineage"]["model_artifact_sha256"] == bundle["model_artifact_sha256"]
    ranked = payload["predictions"][1]["swing"]
    assert ranked["selection_eligible"] is True
    assert ranked["selected_for_policy"] is False
    assert ranked["action"] == "observe_ranked_candidate"
    abstained = payload["predictions"][2]["swing"]
    assert abstained["action"] == "abstain"
    assert abstained["abstention_reasons"] == ["out_of_universe"]
    assert payload["evidence"]["identity_status"] == "complete"
    assert len(drift_checks) == 2
    assert all(check["checked_at"] == NOW for check in drift_checks)

    with pytest.raises(PredictionValidationError):
        service.predict_swing(
            PredictionRequest(
                tickers=["T000"],
                mode="swing",
                as_of=NOW,
                requested_models=["xgboost_regressor"],
            )
        )

    generation_cache.current = False
    with pytest.raises(PredictionReadinessError) as rollover:
        service.predict_swing(
            PredictionRequest(tickers=["T000"], mode="swing", as_of=NOW)
        )
    assert rollover.value.__cause__ is not None
    assert "generation changed" in str(rollover.value.__cause__)

    _, changed_policy_sha256 = service_module._swing_prediction_policy(
        probability_threshold=0.61,
        contract=contract,
    )
    assert changed_policy_sha256 != model.prediction_policy_sha256
    generation_cache.current = True

    service.swing_live_input_provider = _UnavailableInputs()
    with TestClient(create_app(service)) as client:
        unavailable = client.post(
            "/v1/predictions/swing",
            json={"tickers": ["T000"], "as_of": NOW.isoformat()},
        )
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "prediction_not_ready"


def _live_frames(
    *,
    technical_features: tuple[str, ...],
    catalyst_features: tuple[str, ...],
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
    technical = pd.DataFrame(0.1, index=index, columns=technical_features)
    catalyst = pd.DataFrame(0.1, index=index, columns=catalyst_features)
    context = pd.DataFrame(context_rows, index=index)
    return SwingLiveFeatureFrames(
        technical_market=technical,
        catalyst_full=catalyst,
        context=context,
        as_of_utc=pd.Timestamp(NOW),
        decision_time_utc=DECISION,
        session_date_et=DECISION.tz_convert("America/New_York").date(),
    )
