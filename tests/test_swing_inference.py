from __future__ import annotations

import json
import shutil
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import pytest

import market_predictor.serving.swing_inference as serving_module
from market_predictor.canonical.store import file_sha256
from market_predictor.core import path_integrity
from market_predictor.core.errors import (
    ArtifactIntegrityError,
    DataReadinessError,
    PromotionGateError,
    SchemaMismatchError,
)
from market_predictor.execution_policy import EXECUTION_POLICY_SHA256
from market_predictor.governance.promotion.bundle_contracts import (
    canonical_payload_sha256,
    ordered_values_sha256,
    validate_promoted_bundle,
)
from market_predictor.governance.promotion.bundle_verification import (
    resolve_verified_bundle_artifact,
    resolve_verified_bundle_root,
    validate_file_backed_promoted_bundle,
)
from market_predictor.intraday.features.features import (
    CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
)
from market_predictor.modeling.feature_reference import (
    feature_reference_names_sha256,
    feature_reference_profile_sha256,
)
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.promotion_attestation import (
    candidate_manifest_path_for,
    promotion_attestation_path_for,
)
from market_predictor.registry import write_model_manifest
from market_predictor.serving.swing_inference import (
    ACTIVE_GENERATION_SCHEMA,
    LoadedSwingModelGeneration,
    SwingInferenceEngine,
    SwingModelGenerationCache,
    validate_batch_live_feature_parity,
    validate_ordered_feature_frame,
)
from market_predictor.swing.contracts.model_artifact import (
    SWING_CANDIDATE_MODEL_SCHEMA,
)
from market_predictor.swing.features.panel import (
    SWING_FEATURE_PANEL_SCHEMA,
    swing_model_feature_columns,
)
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


def test_serving_does_not_export_governance_promotion_symbols() -> None:
    assert not hasattr(serving_module, "PromotedSwingBundle")
    assert not hasattr(serving_module, "PromotedIntradayBundle")
    assert not hasattr(serving_module, "validate_promoted_bundle")
    assert not hasattr(serving_module, "validate_file_backed_promoted_bundle")


def test_governance_owns_promoted_bundle_contracts() -> None:
    assert validate_promoted_bundle.__module__ == ("market_predictor.governance.promotion.bundle_contracts")
    assert validate_file_backed_promoted_bundle.__module__ == ("market_predictor.governance.promotion.bundle_verification")


class _VerifiedFittedCandidate:
    estimator = object()
    calibrator = object()

    def __init__(self, feature_columns: tuple[str, ...]) -> None:
        self.feature_columns = feature_columns


@pytest.mark.parametrize(
    "thresholds",
    (
        None,
        {"classifier": float("nan")},
        {"classifier": 0.60, "unexpected": 0.50},
    ),
)
def test_swing_inference_rejects_missing_or_unbound_thresholds(
    thresholds: dict[str, float] | None,
) -> None:
    bundle = validate_promoted_bundle(
        _base_bundle(mode="swing"),
        strategy_contract=CONTRACT,
        expected_mode="swing",
    )
    features = bundle.ordered_feature_columns
    feature_reference = _feature_reference(features)
    payload: dict[str, object] = {
        "schema": SWING_CANDIDATE_MODEL_SCHEMA,
        "status": "candidate",
        "promotion_permitted": False,
        "candidate_id": bundle.model_id,
        "model_family": bundle.model_family,
        "strategy_contract_sha256": bundle.strategy_contract_sha256,
        "execution_policy_sha256": bundle.execution_policy_sha256,
        "feature_columns": features,
        "feature_reference_profile": feature_reference,
        "feature_reference_profile_sha256": feature_reference_profile_sha256(
            feature_reference
        ),
        "feature_reference_names_sha256": feature_reference_names_sha256(
            feature_reference
        ),
        "ablation_profile": bundle.feature_profile,
        "fitted_models": {"classifier": _VerifiedFittedCandidate(features)},
    }
    if thresholds is not None:
        payload["probability_thresholds"] = thresholds

    with pytest.raises(SchemaMismatchError, match="threshold"):
        SwingInferenceEngine(
            LoadedSwingModelGeneration(
                generation_id=bundle.sha256(),
                pointer_sha256="a" * 64,
                bundle=bundle,
                model_payload=payload,
            )
        )


def _feature_reference(features: tuple[str, ...]) -> dict[str, dict[str, object]]:
    return {
        feature: {
            "rows": 100,
            "observed": 100,
            "missing_rate": 0.0,
            "mean": 0.0,
            "std": 1.0,
        }
        for feature in features
    }


def _base_bundle(*, mode: str) -> dict[str, object]:
    features = swing_model_feature_columns(contract=CONTRACT, catalyst=False) if mode == "swing" else CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS
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
        "execution_policy_sha256": EXECUTION_POLICY_SHA256,
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
    policy = tomllib.loads((ROOT / "configs" / "edge_rebuild_swing_promotion.toml").read_text(encoding="utf-8"))["promotion_gate_policy"]
    default = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))

    assert canonical_payload_sha256(policy) == default["prediction_serving"]["promotion_gate_policy_sha256"]


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
    feature_reference = _feature_reference(features)
    payload = {
        "schema": SWING_CANDIDATE_MODEL_SCHEMA,
        "status": "candidate",
        "promotion_permitted": False,
        "candidate_id": candidate_id,
        "model_family": "swing_baseline",
        "strategy_contract_sha256": CONTRACT.sha256(),
        "execution_policy_sha256": EXECUTION_POLICY_SHA256,
        "feature_columns": features,
        "feature_reference_profile": feature_reference,
        "feature_reference_profile_sha256": feature_reference_profile_sha256(
            feature_reference
        ),
        "feature_reference_names_sha256": feature_reference_names_sha256(
            feature_reference
        ),
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
        schema_version=SWING_CANDIDATE_MODEL_SCHEMA,
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
            "promotion_evidence_path": ("model/model.joblib.promotion.attestation.json"),
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


def test_governance_verifies_file_backed_swing_bundle_with_explicit_authority(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    generation_id, trust_store = _publish_signed_swing_generation(
        repository,
        candidate_id="governance-owned-swing",
        marker="governance",
    )
    generation = repository / "generations" / generation_id
    payload = json.loads((generation / "bundle.json").read_text(encoding="utf-8"))

    bundle = validate_file_backed_promoted_bundle(
        payload,
        bundle_root=generation,
        strategy_contract=CONTRACT,
        attestation_trust_store_path=trust_store,
        promotion_gate_policy_sha256=TEST_GATE_POLICY_SHA256,
        expected_mode="swing",
    )

    assert bundle.model_id == "governance-owned-swing"
    assert bundle.sha256() == generation_id


@pytest.mark.parametrize(
    ("trust_store", "policy_hash", "message"),
    (
        (None, TEST_GATE_POLICY_SHA256, "trust store"),
        (TRUST_STORE, None, "gate-policy hash"),
    ),
)
def test_governance_swing_verification_requires_explicit_authority(
    tmp_path: Path,
    trust_store: Path | None,
    policy_hash: str | None,
    message: str,
) -> None:
    repository = tmp_path / "repository"
    generation_id, actual_trust_store = _publish_signed_swing_generation(
        repository,
        candidate_id=f"missing-{message}",
        marker=message,
    )
    generation = repository / "generations" / generation_id
    payload = json.loads((generation / "bundle.json").read_text(encoding="utf-8"))
    configured_trust_store = actual_trust_store if trust_store is not None else None

    with pytest.raises(PromotionGateError, match=message):
        validate_file_backed_promoted_bundle(
            payload,
            bundle_root=generation,
            strategy_contract=CONTRACT,
            attestation_trust_store_path=configured_trust_store,
            promotion_gate_policy_sha256=policy_hash,
            expected_mode="swing",
        )


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
    changed = CONTRACT.model_copy(update={"swing": CONTRACT.swing.model_copy(update={"minimum_expected_net_edge_bps": 6.0})})

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


def test_promoted_bundle_root_rejects_reparse_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    original = path_integrity.is_reparse_point
    monkeypatch.setattr(
        path_integrity,
        "is_reparse_point",
        lambda path: path == root or original(path),
    )

    with pytest.raises(ArtifactIntegrityError, match="symlink or reparse point"):
        resolve_verified_bundle_root(root)


def test_promoted_bundle_artifact_rejects_child_reparse_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "bundle").resolve()
    artifact_directory = root / "model"
    artifact_directory.mkdir(parents=True)
    (artifact_directory / "model.bin").write_bytes(b"model")
    original = path_integrity.is_reparse_point
    monkeypatch.setattr(
        path_integrity,
        "is_reparse_point",
        lambda path: path == artifact_directory or original(path),
    )

    with pytest.raises(ArtifactIntegrityError, match="symlink or reparse point"):
        resolve_verified_bundle_artifact(
            root,
            "model/model.bin",
            label="model",
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
