from __future__ import annotations

import json
import shutil
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.global_events.decision_authority import GlobalEventAuthority
from market_predictor.core.errors import (
    ArtifactIntegrityError,
    DataReadinessError,
    PromotionGateError,
    SchemaMismatchError,
)
from market_predictor.edge_rebuild.serving import (
    ACTIVE_GENERATION_SCHEMA,
    BenchmarkComparison,
    CatalystContextSnapshot,
    CatalystSourceSnapshot,
    GlobalContextSnapshot,
    PredictionResult,
    SwingModelGenerationCache,
    build_global_context_snapshot,
    canonical_payload_sha256,
    ordered_values_sha256,
    validate_batch_live_feature_parity,
    validate_file_backed_promoted_bundle,
    validate_ordered_feature_frame,
    validate_promoted_bundle,
)
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.edge_rebuild.swing_features import (
    SWING_FEATURE_PANEL_SCHEMA,
    swing_model_feature_columns,
)
from market_predictor.edge_rebuild.swing_training import MODEL_SCHEMA
from market_predictor.intraday.features.features import (
    CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
)
from market_predictor.promotion_attestation import (
    candidate_manifest_path_for,
    promotion_attestation_path_for,
)
from market_predictor.registry import write_model_manifest
from tests.r4_fixtures import (
    authorize_candidate_for_test,
    synthetic_identity_metrics,
)
from tests.r4_fixtures import (
    test_signing_material as _test_signing_material,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_strategy_contract(ROOT / "configs" / "edge_rebuild_strategy_contract.toml")
TRUST_STORE = ROOT / "configs" / "attestation_trust_store.example.json"
NOW = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
TEST_GATE_POLICY_SHA256 = canonical_payload_sha256({"test_fixture": True})


class _VerifiedFittedCandidate:
    estimator = object()
    calibrator = object()

    def __init__(self, feature_columns: tuple[str, ...]) -> None:
        self.feature_columns = feature_columns


def _base_bundle(*, mode: str) -> dict[str, object]:
    features = (
        swing_model_feature_columns(contract=CONTRACT, catalyst=False)
        if mode == "swing"
        else CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS
    )
    overlays = ("alpaca", "sec", "finviz")
    model_sources: tuple[str, ...] = ()
    global_sources = ("alpaca", "gdelt")
    payload: dict[str, object] = {
        "schema_version": "edge_rebuild.promoted_bundle.v2",
        "mode": mode,
        "model_id": f"{mode}-promoted-001",
        "model_status": "promoted",
        "promotion_permitted": True,
        "model_artifact_path": "model/model.bin",
        "model_artifact_sha256": "a" * 64,
        "promotion_evidence_path": "promotion/evidence.json",
        "promotion_evidence_sha256": "b" * 64,
        "promotion_attestation_id": "d" * 64,
        "promotion_gate_policy_sha256": TEST_GATE_POLICY_SHA256,
        "approved_by_principal_id": "test-approver",
        "promoted_at_utc": NOW,
        "ordered_feature_columns": features,
        "ordered_feature_sha256": ordered_values_sha256(features),
        "strategy_contract_schema_version": CONTRACT.schema_version,
        "strategy_contract_sha256": CONTRACT.sha256(),
        "market_data_provider": "alpaca",
        "market_data_feed": "sip",
        "market_data_adjustment": "all",
        "model_source_families": model_sources,
        "model_source_families_sha256": ordered_values_sha256(model_sources),
        "catalyst_overlay_source_families": overlays,
        "catalyst_overlay_source_families_sha256": ordered_values_sha256(overlays),
        "catalyst_policy_sha256": "c" * 64,
        "global_context_policy": "ranking_overlay",
        "global_authority_schema_version": "edge_rebuild.global_event_authority.v1",
        "global_source_families": global_sources,
        "global_source_families_sha256": ordered_values_sha256(global_sources),
    }
    if mode == "swing":
        payload.update(
            {
                "strategy_id": CONTRACT.swing.strategy_id,
                "horizon_sessions": 10,
                "model_family": "swing_baseline",
                "feature_schema_version": SWING_FEATURE_PANEL_SCHEMA,
                "feature_profile": "technical_market",
                "catalyst_policy": "confirmation_overlay",
            }
        )
    else:
        payload.update(
            {
                "strategy_id": CONTRACT.intraday.strategy_id,
                "horizon_minutes": 30,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_profile": "technical_market",
                "catalyst_policy": "confirmation_overlay",
            }
        )
    return payload


def test_default_serving_hash_matches_frozen_swing_gate_policy() -> None:
    policy = tomllib.loads(
        (ROOT / "configs" / "edge_rebuild_swing_promotion.toml").read_text(
            encoding="utf-8"
        )
    )["promotion_gate_policy"]
    default = tomllib.loads(
        (ROOT / "configs" / "default.toml").read_text(encoding="utf-8")
    )

    assert canonical_payload_sha256(policy) == default["prediction_serving"][
        "promotion_gate_policy_sha256"
    ]


def _publish_signed_swing_generation(
    repository: Path,
    *,
    candidate_id: str,
    marker: str,
    previous_generation_id: str | None = None,
) -> tuple[str, Path]:
    source = repository.parent / f"candidate-{candidate_id}"
    model_path = source / "model.joblib"
    model_path.parent.mkdir(parents=True)
    features = swing_model_feature_columns(contract=CONTRACT, catalyst=False)
    payload = {
        "schema": MODEL_SCHEMA,
        "status": "candidate",
        "promotion_permitted": False,
        "candidate_id": candidate_id,
        "model_family": "swing_baseline",
        "strategy_contract_sha256": CONTRACT.sha256(),
        "feature_columns": features,
        "ablation_profile": "technical_market",
        "probability_thresholds": {"classifier": 0.60},
        "fitted_models": {"classifier": _VerifiedFittedCandidate(features)},
        "marker": marker,
    }
    joblib.dump(payload, model_path)
    metrics = synthetic_identity_metrics(
        model_type="canonical_swing",
        model_run_id=candidate_id,
    )
    training = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "session_date_et": ["2026-01-02", "2026-01-05"],
            "target": [1, 0],
            features[0]: [0.1, -0.1],
        }
    )
    write_model_manifest(
        model_path=model_path,
        model_type="canonical_swing",
        schema_version=MODEL_SCHEMA,
        target_col="target",
        features=list(features),
        training_data=training,
        metrics=metrics,
        validation_split="session_purged_walk_forward_and_ticker_holdout",
        extra={"model_run_id": candidate_id},
    )
    authorize_candidate_for_test(model_path, metrics)
    attestation_path = promotion_attestation_path_for(model_path)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    approver = attestation["approver_principal"]
    bundle_payload = _base_bundle(mode="swing")
    bundle_payload.update(
        {
            "model_id": candidate_id,
            "model_artifact_path": "model/model.joblib",
            "model_artifact_sha256": file_sha256(model_path),
            "promotion_evidence_path": (
                "model/model.joblib.promotion.attestation.json"
            ),
            "promotion_evidence_sha256": file_sha256(attestation_path),
            "promotion_attestation_id": attestation["attestation_id"],
            "promotion_gate_policy_sha256": attestation["gate_config_sha256"],
            "approved_by_principal_id": approver["principal_id"],
            "promoted_at_utc": attestation["promoted_at_utc"],
        }
    )
    bundle = validate_promoted_bundle(
        bundle_payload,
        strategy_contract=CONTRACT,
        expected_mode="swing",
    )
    generation_id = bundle.sha256()
    generation = repository / "generations" / generation_id
    target_model = generation / "model" / "model.joblib"
    target_model.parent.mkdir(parents=True)
    shutil.copyfile(model_path, target_model)
    shutil.copyfile(
        candidate_manifest_path_for(model_path),
        candidate_manifest_path_for(target_model),
    )
    shutil.copyfile(attestation_path, promotion_attestation_path_for(target_model))
    bundle_path = generation / "bundle.json"
    bundle_path.write_text(
        json.dumps(bundle_payload, sort_keys=True, default=str),
        encoding="utf-8",
    )
    pointer: dict[str, object] = {
        "schema": ACTIVE_GENERATION_SCHEMA,
        "generation_id": generation_id,
        "bundle_file_sha256": file_sha256(bundle_path),
        "previous_generation_id": previous_generation_id,
        "activated_at_utc": NOW.isoformat(),
    }
    pointer["pointer_sha256"] = canonical_payload_sha256(pointer)
    repository.mkdir(parents=True, exist_ok=True)
    (repository / "active_generation.json").write_text(
        json.dumps(pointer, sort_keys=True),
        encoding="utf-8",
    )
    _, trust_store, _ = _test_signing_material()
    return generation_id, trust_store


def test_validates_strict_swing_and_intraday_promoted_bundles() -> None:
    swing = validate_promoted_bundle(
        _base_bundle(mode="swing"),
        strategy_contract=CONTRACT,
        expected_mode="swing",
    )
    intraday = validate_promoted_bundle(
        _base_bundle(mode="intraday"),
        strategy_contract=CONTRACT,
        expected_mode="intraday",
    )

    assert swing.horizon_sessions == 10
    assert swing.model_family == "swing_baseline"
    assert swing.catalyst_policy == "confirmation_overlay"
    assert intraday.horizon_minutes == 30
    assert intraday.catalyst_policy == "confirmation_overlay"
    assert len(swing.sha256()) == 64


@pytest.mark.parametrize(
    ("mode", "horizon_field", "legacy_horizon"),
    (("swing", "horizon_sessions", 5), ("intraday", "horizon_minutes", 60)),
)
def test_rejects_legacy_five_day_and_sixty_minute_artifacts(
    mode: str,
    horizon_field: str,
    legacy_horizon: int,
) -> None:
    payload = _base_bundle(mode=mode)
    payload[horizon_field] = legacy_horizon

    with pytest.raises(SchemaMismatchError, match="required horizon"):
        validate_promoted_bundle(payload, strategy_contract=CONTRACT)


def test_rejects_candidate_or_unbound_bundle() -> None:
    candidate = _base_bundle(mode="swing")
    candidate["model_status"] = "candidate"
    candidate["promotion_permitted"] = False
    with pytest.raises(PromotionGateError, match="promoted"):
        validate_promoted_bundle(candidate, strategy_contract=CONTRACT)

    stale = _base_bundle(mode="swing")
    stale["strategy_contract_sha256"] = "c" * 64
    with pytest.raises(ArtifactIntegrityError, match="active strategy contract"):
        validate_promoted_bundle(stale, strategy_contract=CONTRACT)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("feature_schema_version", "swing.features.v1"),
        ("ordered_feature_sha256", "d" * 64),
        ("catalyst_policy", "required_model_feature"),
        ("model_source_families", ("sec",)),
    ),
)
def test_rejects_unbound_swing_feature_or_catalyst_contract(
    field: str,
    value: object,
) -> None:
    payload = _base_bundle(mode="swing")
    payload[field] = value

    with pytest.raises(SchemaMismatchError):
        validate_promoted_bundle(payload, strategy_contract=CONTRACT)


@pytest.mark.parametrize("mode", ("swing", "intraday"))
def test_rejects_self_hashed_features_that_differ_from_active_schema(mode: str) -> None:
    payload = _base_bundle(mode=mode)
    ordered = payload["ordered_feature_columns"]
    assert isinstance(ordered, tuple)
    wrong = tuple(reversed(ordered))
    payload["ordered_feature_columns"] = wrong
    payload["ordered_feature_sha256"] = ordered_values_sha256(wrong)

    with pytest.raises(SchemaMismatchError, match="active .* estimator schema"):
        validate_promoted_bundle(payload, strategy_contract=CONTRACT)


def test_file_backed_bundle_verifies_model_and_promotion_evidence(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    model_path = root / "model" / "model.bin"
    evidence_path = root / "promotion" / "evidence.json"
    model_path.parent.mkdir(parents=True)
    evidence_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"verified model")
    evidence_path.write_text('{"promotion":"passed"}', encoding="utf-8")
    payload = _base_bundle(mode="intraday")
    payload["model_artifact_sha256"] = file_sha256(model_path)
    payload["promotion_evidence_sha256"] = file_sha256(evidence_path)

    bundle = validate_file_backed_promoted_bundle(
        payload,
        bundle_root=root,
        strategy_contract=CONTRACT,
        attestation_trust_store_path=TRUST_STORE,
        expected_mode="intraday",
    )

    assert bundle.model_artifact_path == "model/model.bin"
    assert bundle.promotion_evidence_path == "promotion/evidence.json"


def test_signed_swing_generation_is_cached_and_pointer_rollover_is_loaded(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    first_id, trust_store = _publish_signed_swing_generation(
        repository,
        candidate_id="signed-swing-one",
        marker="first",
    )
    cache = SwingModelGenerationCache(
        memory_budget_gib=4.0,
        memory_headroom_gib=0.5,
    )

    first = cache.get(
        repository,
        strategy_contract=CONTRACT,
        attestation_trust_store_path=trust_store,
        promotion_gate_policy_sha256=TEST_GATE_POLICY_SHA256,
        maximum_model_bytes=10_000_000,
        estimated_resident_gib=0.01,
    )
    repeated = cache.get(
        repository,
        strategy_contract=CONTRACT,
        attestation_trust_store_path=trust_store,
        promotion_gate_policy_sha256=TEST_GATE_POLICY_SHA256,
        maximum_model_bytes=10_000_000,
        estimated_resident_gib=0.01,
    )

    assert first.generation_id == first_id
    assert first.model_payload["marker"] == "first"
    assert repeated is first
    second_id, _ = _publish_signed_swing_generation(
        repository,
        candidate_id="signed-swing-two",
        marker="second",
        previous_generation_id=first_id,
    )
    second = cache.get(
        repository,
        strategy_contract=CONTRACT,
        attestation_trust_store_path=trust_store,
        promotion_gate_policy_sha256=TEST_GATE_POLICY_SHA256,
        maximum_model_bytes=10_000_000,
        estimated_resident_gib=0.01,
    )
    assert second.generation_id == second_id
    assert second.model_payload["marker"] == "second"
    assert second is not first


def test_signed_swing_generation_rejects_different_configured_gate_policy(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    _, trust_store = _publish_signed_swing_generation(
        repository,
        candidate_id="signed-swing-policy-mismatch",
        marker="mismatch",
    )

    with pytest.raises(PromotionGateError, match="gate policy"):
        SwingModelGenerationCache(
            memory_budget_gib=4.0,
            memory_headroom_gib=0.5,
        ).get(
            repository,
            strategy_contract=CONTRACT,
            attestation_trust_store_path=trust_store,
            promotion_gate_policy_sha256="e" * 64,
            maximum_model_bytes=10_000_000,
            estimated_resident_gib=0.01,
        )


def test_cached_swing_generation_revalidates_changed_strategy_contract(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    _, trust_store = _publish_signed_swing_generation(
        repository,
        candidate_id="signed-swing-contract-cache",
        marker="cached",
    )
    cache = SwingModelGenerationCache(
        memory_budget_gib=4.0,
        memory_headroom_gib=0.5,
    )
    cache.get(
        repository,
        strategy_contract=CONTRACT,
        attestation_trust_store_path=trust_store,
        promotion_gate_policy_sha256=TEST_GATE_POLICY_SHA256,
        maximum_model_bytes=10_000_000,
        estimated_resident_gib=0.01,
    )
    changed = CONTRACT.model_copy(
        update={
            "swing": CONTRACT.swing.model_copy(
                update={"minimum_expected_net_edge_bps": 6.0}
            )
        }
    )

    with pytest.raises(ArtifactIntegrityError, match="active strategy contract"):
        cache.get(
            repository,
            strategy_contract=changed,
            attestation_trust_store_path=trust_store,
            promotion_gate_policy_sha256=TEST_GATE_POLICY_SHA256,
            maximum_model_bytes=10_000_000,
            estimated_resident_gib=0.01,
        )


def test_signed_swing_generation_rejects_untrusted_attestation(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    _publish_signed_swing_generation(
        repository,
        candidate_id="signed-swing-untrusted",
        marker="untrusted",
    )
    untrusted = tmp_path / "untrusted.json"
    untrusted.write_text('{"schema":"promotion_attestation_trust.v1","signers":{}}', encoding="utf-8")

    with pytest.raises(PromotionGateError, match="attestation did not verify"):
        SwingModelGenerationCache(
            memory_budget_gib=4.0,
            memory_headroom_gib=0.5,
        ).get(
            repository,
            strategy_contract=CONTRACT,
            attestation_trust_store_path=untrusted,
            promotion_gate_policy_sha256=TEST_GATE_POLICY_SHA256,
            maximum_model_bytes=10_000_000,
            estimated_resident_gib=0.01,
        )


@pytest.mark.parametrize(
    "invalid_path",
    ("../outside.bin", "/absolute/model.bin", "C:/outside/model.bin", "model\\model.bin"),
)
def test_file_backed_bundle_rejects_artifact_path_escape(
    tmp_path: Path,
    invalid_path: str,
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    payload = _base_bundle(mode="intraday")
    payload["model_artifact_path"] = invalid_path

    with pytest.raises(ArtifactIntegrityError, match="path"):
        validate_file_backed_promoted_bundle(
            payload,
            bundle_root=root,
            strategy_contract=CONTRACT,
            attestation_trust_store_path=TRUST_STORE,
        )


def test_file_backed_bundle_rejects_missing_and_tampered_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    model_path = root / "model" / "model.bin"
    evidence_path = root / "promotion" / "evidence.json"
    model_path.parent.mkdir(parents=True)
    evidence_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"original model")
    evidence_path.write_text("{}", encoding="utf-8")
    payload = _base_bundle(mode="swing")
    payload["model_artifact_sha256"] = file_sha256(model_path)
    payload["promotion_evidence_sha256"] = file_sha256(evidence_path)

    model_path.write_bytes(b"tampered model")
    with pytest.raises(ArtifactIntegrityError, match="model artifact SHA256"):
        validate_file_backed_promoted_bundle(
            payload,
            bundle_root=root,
            strategy_contract=CONTRACT,
            attestation_trust_store_path=TRUST_STORE,
        )

    model_path.write_bytes(b"original model")
    evidence_path.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="promotion evidence artifact SHA256"):
        validate_file_backed_promoted_bundle(
            payload,
            bundle_root=root,
            strategy_contract=CONTRACT,
            attestation_trust_store_path=TRUST_STORE,
        )

    payload["model_artifact_path"] = "model/missing.bin"
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        validate_file_backed_promoted_bundle(
            payload,
            bundle_root=root,
            strategy_contract=CONTRACT,
            attestation_trust_store_path=TRUST_STORE,
        )


def test_global_context_is_built_from_exact_verified_authority_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "global-authority"
    directory.mkdir()
    authority_file = directory / "_authority.json"
    authority_file.write_text('{"state":"complete"}', encoding="utf-8")
    decisions = pd.DataFrame(
        {
            "decision_time_utc": [NOW],
            "global_source_complete_1d": [True],
            "global_source_complete_3d": [True],
            "global_event_count_1d": [4.0],
            "global_event_count_3d": [11.0],
            "global_sentiment_mean_1d": [-0.2],
            "global_sentiment_mean_3d": [-0.1],
            "global_sentiment_coverage_1d": [1.0],
            "global_sentiment_coverage_3d": [0.9],
            "global_latest_event_feature_available_at_utc_1d": [NOW],
            "global_latest_event_feature_available_at_utc_3d": [NOW],
        }
    )
    authority = GlobalEventAuthority(
        directory=directory,
        decisions=decisions,
        coverage=pd.DataFrame(),
        manifest={
            "production_ready": True,
            "required_historical_sources": ["alpaca", "gdelt"],
        },
        authority={"state": "complete"},
    )
    monkeypatch.setattr(
        "market_predictor.edge_rebuild.serving.load_global_event_authority",
        lambda *_args, **_kwargs: authority,
    )
    authority_sha256 = file_sha256(authority_file)

    snapshot = build_global_context_snapshot(
        authority,
        decision_time_utc=NOW,
        authority_sha256=authority_sha256,
    )

    assert snapshot.event_count_3d == 11
    assert snapshot.source_families == ("alpaca", "gdelt")
    assert "risk_score" not in GlobalContextSnapshot.model_fields
    assert "regime" not in GlobalContextSnapshot.model_fields
    unsourced = snapshot.model_dump()
    unsourced.update({"risk_score": -0.2, "regime": "risk_off"})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GlobalContextSnapshot.model_validate(unsourced)

    authority_file.write_text('{"state":"tampered"}', encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="SHA256"):
        build_global_context_snapshot(
            authority,
            decision_time_utc=NOW,
            authority_sha256=authority_sha256,
        )


def test_scored_result_exposes_technical_catalyst_global_and_benchmark_fields() -> None:
    result = PredictionResult(
        mode="swing",
        strategy_id="swing",
        model_id="swing-promoted-001",
        bundle_sha256="a" * 64,
        ticker="MSFT",
        as_of_utc=NOW,
        horizon_value=10,
        horizon_unit="sessions",
        status="scored",
        predicted_direction="up",
        model_score=0.76,
        technical_score=0.71,
        catalyst_overlay_status="incorporated",
        catalyst_context_available=True,
        catalyst_context=CatalystContextSnapshot(
            as_of_utc=NOW,
            authority_sha256="e" * 64,
            required_model_sources_complete=True,
            event_count_1d=2,
            event_count_3d=5,
            sentiment_mean_1d=0.4,
            sentiment_mean_3d=0.2,
            sentiment_coverage_1d=1.0,
            sentiment_coverage_3d=0.8,
            latest_event_feature_available_at_utc=NOW,
            sources=(
                CatalystSourceSnapshot(
                    source_family="alpaca",
                    coverage_known=True,
                    event_count_1d=2,
                    event_count_3d=5,
                ),
                CatalystSourceSnapshot(
                    source_family="sec",
                    coverage_known=False,
                ),
            ),
        ),
        global_context_available=True,
        global_context=GlobalContextSnapshot(
            as_of_utc=NOW,
            authority_sha256="d" * 64,
            source_coverage_complete=True,
            event_count_1d=4,
            event_count_3d=11,
            sentiment_mean_1d=-0.2,
            sentiment_mean_3d=-0.1,
            sentiment_coverage_1d=1.0,
            sentiment_coverage_3d=0.9,
            source_families=("alpaca",),
        ),
        benchmark_comparisons=(
            BenchmarkComparison(
                symbol="SPY",
                predicted_stock_return=0.03,
                predicted_benchmark_return=0.01,
                predicted_excess_return=0.02,
            ),
            BenchmarkComparison(
                symbol="QQQ",
                predicted_stock_return=0.03,
                predicted_benchmark_return=0.015,
                predicted_excess_return=0.015,
            ),
        ),
    )

    assert result.technical_score == pytest.approx(0.71)
    assert result.catalyst_overlay_status == "incorporated"
    assert result.global_context is not None
    assert result.benchmark_comparisons[0].predicted_excess_return == pytest.approx(0.02)


def test_unavailable_global_context_is_null_and_abstention_is_explicit() -> None:
    result = PredictionResult(
        mode="intraday",
        strategy_id="intraday",
        model_id="intraday-promoted-001",
        bundle_sha256="a" * 64,
        ticker="RGTI",
        as_of_utc=NOW,
        horizon_value=30,
        horizon_unit="minutes",
        status="abstained",
        predicted_direction=None,
        model_score=None,
        technical_score=None,
        catalyst_overlay_status="unavailable",
        catalyst_context_available=False,
        catalyst_context=None,
        global_context_available=False,
        global_context=None,
        abstention_reasons=("feature_value_unavailable",),
    )

    assert result.global_context is None
    assert result.abstention_reasons == ("feature_value_unavailable",)

    invalid = result.model_dump()
    invalid["global_context_available"] = True
    with pytest.raises(ValidationError, match="global_context must be null"):
        PredictionResult.model_validate(invalid)


def test_scored_result_requires_spy_and_qqq_and_abstention_cannot_leak_score() -> None:
    common: dict[str, object] = {
        "mode": "intraday",
        "strategy_id": "intraday",
        "model_id": "intraday-promoted-001",
        "bundle_sha256": "a" * 64,
        "ticker": "NVDA",
        "as_of_utc": NOW,
        "horizon_value": 30,
        "horizon_unit": "minutes",
        "catalyst_overlay_status": "neutral",
        "catalyst_context_available": False,
        "catalyst_context": None,
        "global_context_available": False,
        "global_context": None,
    }
    with pytest.raises(ValidationError, match="SPY"):
        PredictionResult(
            **common,
            status="scored",
            predicted_direction="up",
            model_score=0.65,
            technical_score=0.6,
            benchmark_comparisons=(
                BenchmarkComparison(
                    symbol="QQQ",
                    predicted_stock_return=0.01,
                    predicted_benchmark_return=0.005,
                    predicted_excess_return=0.005,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="QQQ"):
        PredictionResult(
            **common,
            status="scored",
            predicted_direction="up",
            model_score=0.65,
            technical_score=0.6,
            benchmark_comparisons=(
                BenchmarkComparison(
                    symbol="SPY",
                    predicted_stock_return=0.01,
                    predicted_benchmark_return=0.005,
                    predicted_excess_return=0.005,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="cannot expose a model score"):
        PredictionResult(
            **common,
            status="abstained",
            predicted_direction="down",
            model_score=0.2,
            technical_score=0.2,
            abstention_reasons=("stale_features",),
        )


def test_catalyst_context_preserves_unknown_source_coverage() -> None:
    with pytest.raises(ValidationError, match="counts must be null"):
        CatalystSourceSnapshot(
            source_family="sec",
            coverage_known=False,
            event_count_1d=0,
            event_count_3d=0,
        )

    with pytest.raises(ValidationError, match="aggregates must be null"):
        CatalystContextSnapshot(
            as_of_utc=NOW,
            authority_sha256="e" * 64,
            required_model_sources_complete=False,
            event_count_1d=0,
            event_count_3d=0,
            sentiment_mean_1d=0.0,
            sentiment_mean_3d=0.0,
            sentiment_coverage_1d=0.0,
            sentiment_coverage_3d=0.0,
            sources=(
                CatalystSourceSnapshot(
                    source_family="alpaca",
                    coverage_known=False,
                ),
            ),
        )


def test_batch_live_parity_accepts_tolerance_and_reports_hash() -> None:
    columns = ("return_1_bar", "session_vwap_distance_atr")
    batch = pd.DataFrame([[0.01, -0.5], [0.02, 0.25]], columns=columns)
    live = batch.copy()
    live.loc[1, "return_1_bar"] += 1e-13

    report = validate_batch_live_feature_parity(batch, live, columns)

    assert report.matched is True
    assert report.row_count == 2
    assert report.feature_count == 2
    assert report.ordered_feature_sha256 == ordered_values_sha256(columns)


def test_batch_live_parity_rejects_order_value_and_non_finite_drift() -> None:
    columns = ("a", "b")
    batch = pd.DataFrame([[1.0, 2.0]], columns=columns)

    with pytest.raises(SchemaMismatchError, match="promoted order"):
        validate_ordered_feature_frame(
            batch.loc[:, ["b", "a"]],
            columns,
            frame_name="live",
        )
    with pytest.raises(DataReadinessError, match="parity failed"):
        validate_batch_live_feature_parity(
            batch,
            pd.DataFrame([[1.0, 2.1]], columns=columns),
            columns,
        )
    with pytest.raises(DataReadinessError, match="non-finite"):
        validate_ordered_feature_frame(
            pd.DataFrame([[1.0, float("nan")]], columns=columns),
            columns,
            frame_name="live",
        )
    with pytest.raises(SchemaMismatchError, match="row identities"):
        validate_batch_live_feature_parity(
            pd.DataFrame([[1.0, 2.0]], columns=columns, index=["batch-row"]),
            pd.DataFrame([[1.0, 2.0]], columns=columns, index=["live-row"]),
            columns,
        )
