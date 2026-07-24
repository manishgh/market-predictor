"""Deterministic synthetic promotion material for tests and CI smoke releases."""

from __future__ import annotations

import json
import os
import tempfile
from base64 import b64encode
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from market_predictor.execution_policy import EXECUTION_POLICY_SHA256
from market_predictor.hypothesis_registry import declare_hypothesis
from market_predictor.prediction_policy import prediction_policy_identity
from market_predictor.promotion_attestation import (
    ATTESTATION_TRUST_STORE_ENV,
    ATTESTATION_TRUST_STORE_SCHEMA,
    SIGNATURE_ALGORITHM,
    file_sha256,
)
from market_predictor.promotion_workflow import (
    PromotionTrustContext,
    evaluate_shadow_and_attest,
)
from market_predictor.registry import load_model_manifest
from market_predictor.shadow_ledger import write_shadow_bundle


def synthetic_identity_metrics(
    *,
    model_type: str,
    model_run_id: str,
    validation_split: str = "session_purged_walk_forward_and_ticker_holdout",
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "model_run_id": model_run_id,
        "validation_split": validation_split,
        "holdout_assignment_cutoff_utc": "2026-01-30T23:00:00+00:00",
        "holdout_ticker_summary_sha256": "1" * 64,
        "feature_set_sha256": "2" * 64,
        "reconciliation_sha256": "3" * 64,
        "dataset_label_config_sha256": "4" * 64,
        "calibration_method": "isotonic_prior_fold_only",
        "folds_causally_ordered": True,
        "execution_policy_sha256": EXECUTION_POLICY_SHA256,
        "dataset_sha256": "9" * 64,
        **prediction_policy_identity(),
    }
    if model_type == "canonical_swing":
        metrics["universe_identity_sha256"] = "5" * 64
    return metrics


def trust_context_for_candidate(
    root: Path,
    *,
    model_path: Path,
    metrics: dict[str, Any],
    model_type: str,
    hypothesis_suffix: str = "001",
    improvements: list[float] | None = None,
) -> PromotionTrustContext:
    signing_key, trust_store, signer_id = test_signing_material()
    run_id = str(metrics["model_run_id"])
    safe_run_id = "".join(character if character.isalnum() or character in "._-" else "-" for character in run_id)
    hypothesis_id = f"{safe_run_id}-{hypothesis_suffix}"
    family = f"{model_type.replace('_', '-')}-family"
    declared_at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    hypothesis = declare_hypothesis(
        root,
        hypothesis_id=hypothesis_id,
        hypothesis_family=family,
        model_type=model_type,
        baseline_id="frozen-baseline-v1",
        baseline_artifact_sha256="a" * 64,
        prediction_policy_sha256=str(metrics["prediction_policy_sha256"]),
        execution_policy_sha256=str(metrics["execution_policy_sha256"]),
        objective="Synthetic test declaration for the immutable promotion trust path.",
        declared_at=declared_at,
    )
    values = improvements or [0.02, 0.01, 0.03, 0.015]
    sessions = pd.DataFrame(
        {
            "session_date_et": pd.date_range("2026-02-02", periods=len(values), freq="B"),
            "candidate_benchmark_excess_return": values,
            "baseline_benchmark_excess_return": [0.0] * len(values),
        }
    )
    bundle = write_shadow_bundle(
        root,
        sessions,
        hypothesis=hypothesis,
        candidate_artifact_sha256=file_sha256(model_path),
        generated_at=declared_at + timedelta(days=60),
        bootstrap_iterations=200,
        bootstrap_seed=17,
    )
    return PromotionTrustContext(
        hypothesis_registry_root=root,
        hypothesis_id=hypothesis_id,
        shadow_bundle_path=bundle,
        build_identity="ci:test-build",
        approver_identity="reviewer:test",
        signing_private_key_path=signing_key,
        attestation_trust_store_path=trust_store,
        signer_id=signer_id,
        minimum_shadow_sessions=len(values),
        minimum_paired_improvement_ci_low=0.0,
    )


def authorize_candidate_for_test(model_path: Path, metrics: dict[str, Any]) -> Path:
    root = model_path.parent / f".{model_path.name}.promotion-test"
    manifest = load_model_manifest(model_path)
    context = trust_context_for_candidate(
        root,
        model_path=model_path,
        metrics=metrics,
        model_type=str(manifest["model_type"]),
    )
    evidence_manifest = root / "evidence.manifest.json"
    evidence_manifest.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")
    evidence_manifest.write_text(
        json.dumps(
            {
                "schema": (
                    "intraday_training_evidence.v1"
                    if manifest["model_type"] == "canonical_intraday"
                    else "swing_training_evidence.v1"
                ),
                "model_run_id": metrics["model_run_id"],
                "model_artifact_sha256": manifest["artifact_sha256"],
                "files": {
                    "metrics": {
                        "path": metrics_path.name,
                        "sha256": file_sha256(metrics_path),
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    outcome = evaluate_shadow_and_attest(
        model_path=model_path,
        evidence_manifest_path=evidence_manifest,
        metrics=metrics,
        gate_config={"test_fixture": True},
        context=context,
    )
    if outcome.attestation is None:
        raise AssertionError(f"synthetic promotion fixture failed: {outcome.failures}")
    return evidence_manifest


def test_signing_material() -> tuple[Path, Path, str]:
    root = Path(tempfile.gettempdir()) / f"market-predictor-r4-signing-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    key_path = root / "test-ed25519-private.pem"
    trust_store_path = root / "test-attestation-trust.json"
    signer_id = "test-ci-signer"
    if not key_path.exists():
        private_key = Ed25519PrivateKey.generate()
        key_path.write_bytes(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        trust_store_path.write_text(
            json.dumps(
                {
                    "schema": ATTESTATION_TRUST_STORE_SCHEMA,
                    "issuers": {
                        signer_id: {
                            "algorithm": SIGNATURE_ALGORITHM,
                            "public_key_base64": b64encode(public_key).decode("ascii"),
                        }
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    os.environ[ATTESTATION_TRUST_STORE_ENV] = str(trust_store_path)
    os.environ["MARKET_PREDICTOR_ALLOW_TEST_CLOCK"] = "1"
    return key_path, trust_store_path, signer_id
