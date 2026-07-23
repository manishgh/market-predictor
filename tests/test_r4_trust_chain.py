from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd

from market_predictor.hypothesis_registry import declare_hypothesis
from market_predictor.promotion_attestation import (
    build_promotion_attestation,
    file_sha256,
    verify_promotion_attestation,
    write_promotion_attestation,
)
from market_predictor.registry import write_model_manifest
from market_predictor.shadow_ledger import (
    consume_shadow_fingerprint,
    load_shadow_bundle,
    load_shadow_ledger,
    shadow_gate_failures,
    write_shadow_bundle,
)
from market_predictor.v3.errors import DataReadinessError


class R4TrustChainTests(unittest.TestCase):
    def test_hypothesis_is_predeclared_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            declared = _declare(root)
            repeated = _declare(root)
            self.assertEqual(repeated, declared)

            with self.assertRaisesRegex(DataReadinessError, "immutable"):
                declare_hypothesis(
                    root,
                    hypothesis_id="swing-alpha-001",
                    hypothesis_family="swing-alpha",
                    model_type="swing_classifier_v1",
                    baseline_id="technical-baseline-v1",
                    baseline_artifact_sha256="a" * 64,
                    prediction_policy_sha256="f" * 64,
                    objective="mutated after declaration",
                    declared_at=_declared_at(),
                )

    def test_shadow_fingerprint_is_one_use_and_failed_family_is_retired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hypothesis = _declare(root)
            bundle_path = _shadow(root, hypothesis, candidate_sha="c" * 64, improvements=[0.02, 0.01, 0.03, 0.015])
            bundle = load_shadow_bundle(bundle_path)
            self.assertEqual(
                shadow_gate_failures(bundle, minimum_independent_sessions=4, minimum_paired_improvement_ci_low=0.0),
                [],
            )
            ledger = root / "shadow-ledger.jsonl"
            consume_shadow_fingerprint(
                ledger,
                bundle=bundle,
                hypothesis=hypothesis,
                result="passed",
                attestation_id="d" * 64,
                consumed_at=_declared_at() + timedelta(days=3),
            )
            with self.assertRaisesRegex(DataReadinessError, "already been consumed"):
                consume_shadow_fingerprint(
                    ledger,
                    bundle=bundle,
                    hypothesis=hypothesis,
                    result="passed",
                    attestation_id="d" * 64,
                    consumed_at=_declared_at() + timedelta(days=3),
                )
            self.assertEqual(len(load_shadow_ledger(ledger)), 1)

            second = load_shadow_bundle(
                _shadow(root, hypothesis, candidate_sha="e" * 64, improvements=[-0.02, -0.01, -0.03, -0.015])
            )
            consume_shadow_fingerprint(
                ledger,
                bundle=second,
                hypothesis=hypothesis,
                result="failed",
                attestation_id=None,
                consumed_at=_declared_at() + timedelta(days=4),
            )
            third = load_shadow_bundle(
                _shadow(root, hypothesis, candidate_sha="1" * 64, improvements=[0.02, 0.01, 0.03, 0.015])
            )
            with self.assertRaisesRegex(DataReadinessError, "family was retired"):
                consume_shadow_fingerprint(
                    ledger,
                    bundle=third,
                    hypothesis=hypothesis,
                    result="passed",
                    attestation_id="2" * 64,
                    consumed_at=_declared_at() + timedelta(days=5),
                )

    def test_positive_point_with_nonpositive_lower_bound_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hypothesis = _declare(root)
            bundle = load_shadow_bundle(
                _shadow(root, hypothesis, candidate_sha="c" * 64, improvements=[0.10, -0.09, 0.10, -0.09])
            )
            interval = bundle["paired_improvement_interval"]
            self.assertGreater(float(interval["point"]), 0)
            self.assertLessEqual(float(interval["low"]), 0)
            failures = shadow_gate_failures(
                bundle,
                minimum_independent_sessions=4,
                minimum_paired_improvement_ci_low=0.0,
            )
            self.assertTrue(any("CI low" in failure for failure in failures))

    def test_attestation_binds_candidate_evidence_and_identity_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, metrics, evidence_manifest = _candidate(root)
            hypothesis = _declare(root)
            bundle = load_shadow_bundle(
                _shadow(root, hypothesis, candidate_sha=file_sha256(model), improvements=[0.02, 0.01, 0.03, 0.015])
            )
            attestation = build_promotion_attestation(
                model_path=model,
                evidence_manifest_path=evidence_manifest,
                metrics=metrics,
                hypothesis=hypothesis,
                shadow_bundle=bundle,
                gate_config={"minimum_shadow_sessions": 4, "minimum_ci_low": 0.0},
                build_identity="ci:test-build",
                approver_identity="reviewer:test",
                promoted_at=_declared_at() + timedelta(days=3),
            )
            write_promotion_attestation(model, attestation)
            verified = verify_promotion_attestation(model, evidence_manifest_path=evidence_manifest)
            self.assertEqual(verified["attestation_id"], attestation["attestation_id"])
            self.assertEqual(verified["identity_chain"]["dataset_sha256"], "9" * 64)

            evidence_manifest.write_text('{"mutated":true}', encoding="utf-8")
            with self.assertRaisesRegex(DataReadinessError, "evidence manifest changed"):
                verify_promotion_attestation(model, evidence_manifest_path=evidence_manifest)

    def test_attestation_rejects_any_bound_candidate_manifest_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, metrics, evidence_manifest = _candidate(root)
            hypothesis = _declare(root)
            bundle = load_shadow_bundle(
                _shadow(root, hypothesis, candidate_sha=file_sha256(model), improvements=[0.02, 0.01, 0.03, 0.015])
            )
            attestation = build_promotion_attestation(
                model_path=model,
                evidence_manifest_path=evidence_manifest,
                metrics=metrics,
                hypothesis=hypothesis,
                shadow_bundle=bundle,
                gate_config={"minimum_shadow_sessions": 4, "minimum_ci_low": 0.0},
                build_identity="ci:test-build",
                approver_identity="reviewer:test",
                promoted_at=_declared_at() + timedelta(days=3),
            )
            write_promotion_attestation(model, attestation)
            manifest_path = model.with_suffix(model.suffix + ".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "promoted"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(DataReadinessError, "manifest changed"):
                verify_promotion_attestation(model)


def _declared_at() -> datetime:
    return datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def _declare(root: Path) -> dict[str, object]:
    return declare_hypothesis(
        root,
        hypothesis_id="swing-alpha-001",
        hypothesis_family="swing-alpha",
        model_type="swing_classifier_v1",
        baseline_id="technical-baseline-v1",
        baseline_artifact_sha256="a" * 64,
        prediction_policy_sha256="f" * 64,
        objective="Improve benchmark-relative top-k swing return.",
        declared_at=_declared_at(),
    )


def _shadow(
    root: Path,
    hypothesis: dict[str, object],
    *,
    candidate_sha: str,
    improvements: list[float],
) -> Path:
    sessions = pd.DataFrame(
        {
            "session_date_et": pd.date_range("2026-07-10", periods=len(improvements), freq="B"),
            "candidate_benchmark_excess_return": improvements,
            "baseline_benchmark_excess_return": [0.0] * len(improvements),
        }
    )
    return write_shadow_bundle(
        root,
        sessions,
        hypothesis=hypothesis,
        candidate_artifact_sha256=candidate_sha,
        generated_at=_declared_at() + timedelta(days=2),
        bootstrap_iterations=200,
        bootstrap_seed=17,
    )


def _candidate(root: Path) -> tuple[Path, dict[str, object], Path]:
    model = root / "swing.joblib"
    joblib.dump({"model": "candidate"}, model)
    metrics: dict[str, object] = {
        "model_run_id": "swing-test-run",
        "validation_split": "session_purged_walk_forward_and_ticker_holdout",
        "holdout_assignment_cutoff_utc": "2026-01-30T23:00:00+00:00",
        "holdout_ticker_summary_sha256": "1" * 64,
        "feature_set_sha256": "2" * 64,
        "reconciliation_sha256": "3" * 64,
        "dataset_label_config_sha256": "4" * 64,
        "universe_identity_sha256": "5" * 64,
        "calibration_method": "isotonic_prior_fold_only",
        "folds_causally_ordered": True,
        "prediction_policy_sha256": "f" * 64,
        "execution_policy_sha256": "8" * 64,
        "dataset_sha256": "9" * 64,
    }
    training = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": pd.date_range("2026-01-01", periods=2),
            "return_1d": [0.01, -0.01],
            "target_net_positive_5d": [1, 0],
        }
    )
    manifest = write_model_manifest(
        model_path=model,
        model_type="swing_classifier_v1",
        schema_version="swing.model.v1",
        target_col="target_net_positive_5d",
        features=["return_1d"],
        training_data=training,
        metrics=metrics,
        validation_split="session_purged_walk_forward_and_ticker_holdout",
        extra={"model_run_id": "swing-test-run"},
    )
    evidence_manifest = root / "evidence.manifest.json"
    evidence_manifest.write_text(
        json.dumps(
            {
                "schema": "swing_training_evidence.v1",
                "model_run_id": "swing-test-run",
                "model_artifact_sha256": manifest["artifact_sha256"],
                "files": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return model, metrics, evidence_manifest


if __name__ == "__main__":
    unittest.main()
