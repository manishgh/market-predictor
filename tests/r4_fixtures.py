from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.execution_policy import EXECUTION_POLICY_SHA256
from market_predictor.hypothesis_registry import declare_hypothesis
from market_predictor.prediction_policy import PREDICTION_POLICY_SHA256
from market_predictor.promotion_attestation import (
    build_promotion_attestation,
    file_sha256,
    write_promotion_attestation,
)
from market_predictor.promotion_workflow import PromotionTrustContext
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
        "prediction_policy_sha256": PREDICTION_POLICY_SHA256,
        "execution_policy_sha256": EXECUTION_POLICY_SHA256,
        "dataset_sha256": "9" * 64,
    }
    if model_type == "swing_classifier_v1":
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
        generated_at=declared_at + timedelta(days=1),
        bootstrap_iterations=200,
        bootstrap_seed=17,
    )
    return PromotionTrustContext(
        hypothesis_registry_root=root,
        hypothesis_id=hypothesis_id,
        shadow_bundle_path=bundle,
        shadow_ledger_path=root / "shadow-ledger.jsonl",
        build_identity="ci:test-build",
        approver_identity="reviewer:test",
        minimum_shadow_sessions=len(values),
        minimum_paired_improvement_ci_low=0.0,
    )


def authorize_candidate_for_test(model_path: Path, metrics: dict[str, Any]) -> None:
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
    evidence_manifest.write_text(
        json.dumps(
            {
                "schema": "synthetic_training_evidence.v1",
                "model_run_id": metrics["model_run_id"],
                "model_artifact_sha256": manifest["artifact_sha256"],
                "files": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    from market_predictor.hypothesis_registry import load_hypothesis
    from market_predictor.shadow_ledger import load_shadow_bundle

    hypothesis = load_hypothesis(context.hypothesis_registry_root, context.hypothesis_id)
    shadow = load_shadow_bundle(context.shadow_bundle_path)
    attestation = build_promotion_attestation(
        model_path=model_path,
        evidence_manifest_path=evidence_manifest,
        metrics=metrics,
        hypothesis=hypothesis,
        shadow_bundle=shadow,
        gate_config={"test_fixture": True},
        build_identity=context.build_identity,
        approver_identity=context.approver_identity,
        promoted_at=datetime(2026, 2, 10, 12, 0, tzinfo=UTC),
    )
    write_promotion_attestation(model_path, attestation)
