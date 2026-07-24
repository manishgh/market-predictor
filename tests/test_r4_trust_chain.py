from __future__ import annotations

import hashlib
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
from market_predictor.promotion_workflow import PromotionTrustContext
from market_predictor.registry import write_model_manifest
from market_predictor.shadow_ledger import (
    consume_shadow_fingerprint,
    load_shadow_ledger,
    shadow_gate_failures,
)
from market_predictor.v3.errors import DataReadinessError
from tests.r4_fixtures import (
    load_test_shadow_bundle,
    test_authenticated_promotion_principals,
    test_promotion_identity_material,
    test_signing_material,
    write_test_shadow_bundle,
)


class R4TrustChainTests(unittest.TestCase):
    def test_promotion_context_rejects_shadow_outside_governance_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shadow = root / "outside" / f"{'a' * 64}.json"
            signing_key, trust_store, signer_id = test_signing_material()
            identity_config, identity_tokens = (
                test_promotion_identity_material()
            )

            with self.assertRaisesRegex(ValueError, "inside the hypothesis registry"):
                PromotionTrustContext(
                    hypothesis_registry_root=root,
                    hypothesis_id="swing-alpha-001",
                    shadow_bundle_path=shadow,
                    outcome_repository_root=root / "outcomes",
                    baseline_artifact_path=root / "baseline.joblib",
                    identity_config=identity_config,
                    identity_tokens=identity_tokens,
                    signing_private_key_path=signing_key,
                    attestation_trust_store_path=trust_store,
                    signer_id=signer_id,
                )

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
                    model_type="canonical_swing",
                    candidate_artifact_sha256="c" * 64,
                    baseline_id="technical-baseline-v1",
                    baseline_artifact_sha256="a" * 64,
                    prediction_policy_sha256="f" * 64,
                    execution_policy_sha256="8" * 64,
                    shadow_view="swing",
                    shadow_horizon="5d",
                    shadow_decision_group_ids=(
                        "2026-07-10T20:00:00+00:00",
                        "2026-07-13T20:00:00+00:00",
                    ),
                    shadow_minimum_tickers_per_group=1,
                    objective="mutated after declaration",
                    declared_at=_declared_at(),
                )

    def test_shadow_fingerprint_is_one_use_and_failed_family_is_retired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hypothesis = _declare(root)
            bundle_path = _shadow(root, hypothesis, candidate_sha="c" * 64, improvements=[0.02, 0.01, 0.03, 0.015])
            bundle = load_test_shadow_bundle(bundle_path)
            self.assertEqual(
                shadow_gate_failures(bundle, minimum_independent_sessions=4, minimum_paired_improvement_ci_low=0.0),
                [],
            )
            ledger = root / "shadow-ledger.jsonl"
            first_entry = consume_shadow_fingerprint(
                ledger,
                bundle=bundle,
                hypothesis=hypothesis,
                result="passed",
                attestation_id="d" * 64,
                transaction_id="e" * 64,
                consumed_at=_declared_at() + timedelta(days=3),
            )
            recovered = consume_shadow_fingerprint(
                ledger,
                bundle=bundle,
                hypothesis=hypothesis,
                result="passed",
                attestation_id="d" * 64,
                transaction_id="e" * 64,
                consumed_at=_declared_at() + timedelta(days=4),
            )
            self.assertEqual(recovered, first_entry)
            with self.assertRaisesRegex(DataReadinessError, "already been consumed"):
                consume_shadow_fingerprint(
                    ledger,
                    bundle=bundle,
                    hypothesis=hypothesis,
                    result="passed",
                    attestation_id="d" * 64,
                    transaction_id="f" * 64,
                    consumed_at=_declared_at() + timedelta(days=3),
                )
            self.assertEqual(len(load_shadow_ledger(ledger)), 1)

            second = load_test_shadow_bundle(
                _shadow(root, hypothesis, candidate_sha="e" * 64, improvements=[-0.02, -0.01, -0.03, -0.015])
            )
            consume_shadow_fingerprint(
                ledger,
                bundle=second,
                hypothesis=hypothesis,
                result="failed",
                attestation_id=None,
                transaction_id="1" * 64,
                consumed_at=_declared_at() + timedelta(days=4),
            )
            third = load_test_shadow_bundle(
                _shadow(root, hypothesis, candidate_sha="1" * 64, improvements=[0.02, 0.01, 0.03, 0.015])
            )
            with self.assertRaisesRegex(DataReadinessError, "family was retired"):
                consume_shadow_fingerprint(
                    ledger,
                    bundle=third,
                    hypothesis=hypothesis,
                    result="passed",
                    attestation_id="2" * 64,
                    transaction_id="3" * 64,
                    consumed_at=_declared_at() + timedelta(days=5),
                )

    def test_shadow_sessions_must_follow_hypothesis_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hypothesis = _declare(root)
            sessions = pd.DataFrame(
                {
                    "session_date_et": [
                        _declared_at().date(),
                        (_declared_at() + timedelta(days=1)).date(),
                    ],
                    "candidate_benchmark_excess_return": [0.01, 0.02],
                    "baseline_benchmark_excess_return": [0.0, 0.0],
                }
            )

            with self.assertRaisesRegex(DataReadinessError, "must follow"):
                write_test_shadow_bundle(
                    root,
                    sessions,
                    hypothesis=hypothesis,
                    candidate_artifact_sha256="c" * 64,
                    generated_at=_declared_at() + timedelta(days=2),
                    bootstrap_iterations=200,
                )

    def test_positive_point_with_nonpositive_lower_bound_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hypothesis = _declare(root)
            bundle = load_test_shadow_bundle(
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
            hypothesis = _declare(
                root,
                candidate_sha=file_sha256(model),
            )
            bundle = load_test_shadow_bundle(
                _shadow(root, hypothesis, candidate_sha=file_sha256(model), improvements=[0.02, 0.01, 0.03, 0.015])
            )
            bundle["source_rows_sha256"] = "e" * 64
            bundle["shadow_workload"] = hypothesis["shadow_workload"]
            ledger_entry = consume_shadow_fingerprint(
                root / "shadow-ledger.jsonl",
                bundle=bundle,
                hypothesis=hypothesis,
                result="passed",
                attestation_id=None,
                transaction_id="6" * 64,
            )
            signing_key, trust_store, signer_id = test_signing_material()
            build_principal, approver_principal = (
                test_authenticated_promotion_principals()
            )
            attestation = build_promotion_attestation(
                model_path=model,
                evidence_manifest_path=evidence_manifest,
                metrics=metrics,
                hypothesis=hypothesis,
                shadow_bundle=bundle,
                ledger_entry=ledger_entry,
                gate_config={"minimum_shadow_sessions": 4, "minimum_ci_low": 0.0},
                build_principal=build_principal,
                approver_principal=approver_principal,
                signing_private_key_path=signing_key,
                signer_id=signer_id,
                promoted_at=datetime.now(UTC),
            )
            write_promotion_attestation(
                model,
                attestation,
                trust_store_path=trust_store,
            )
            verified = verify_promotion_attestation(
                model,
                evidence_manifest_path=evidence_manifest,
                trust_store_path=trust_store,
            )
            self.assertEqual(verified["attestation_id"], attestation["attestation_id"])
            self.assertEqual(verified["identity_chain"]["dataset_sha256"], "9" * 64)

            forged = dict(attestation)
            forged["identity_chain"] = {}
            forged_payload = {
                key: value
                for key, value in forged.items()
                if key not in {"attestation_id", "signature"}
            }
            forged["attestation_id"] = hashlib.sha256(
                json.dumps(
                    forged_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            with self.assertRaisesRegex(DataReadinessError, "identity chain fields"):
                write_promotion_attestation(
                    root / "forged.joblib",
                    forged,
                    trust_store_path=trust_store,
                )

            untrusted = root / "untrusted.json"
            untrusted.write_text(
                json.dumps(
                    {
                        "schema": "market_predictor.attestation_trust_store.v1",
                        "issuers": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DataReadinessError, "signer is not trusted"):
                verify_promotion_attestation(
                    model,
                    evidence_manifest_path=evidence_manifest,
                    trust_store_path=untrusted,
                )

            evidence_manifest.write_text('{"mutated":true}', encoding="utf-8")
            with self.assertRaisesRegex(DataReadinessError, "evidence manifest changed"):
                verify_promotion_attestation(model, evidence_manifest_path=evidence_manifest)

    def test_attestation_rejects_any_bound_candidate_manifest_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, metrics, evidence_manifest = _candidate(root)
            hypothesis = _declare(
                root,
                candidate_sha=file_sha256(model),
            )
            bundle = load_test_shadow_bundle(
                _shadow(root, hypothesis, candidate_sha=file_sha256(model), improvements=[0.02, 0.01, 0.03, 0.015])
            )
            bundle["source_rows_sha256"] = "e" * 64
            bundle["shadow_workload"] = hypothesis["shadow_workload"]
            ledger_entry = consume_shadow_fingerprint(
                root / "shadow-ledger.jsonl",
                bundle=bundle,
                hypothesis=hypothesis,
                result="passed",
                attestation_id=None,
                transaction_id="7" * 64,
            )
            signing_key, trust_store, signer_id = test_signing_material()
            build_principal, approver_principal = (
                test_authenticated_promotion_principals()
            )
            attestation = build_promotion_attestation(
                model_path=model,
                evidence_manifest_path=evidence_manifest,
                metrics=metrics,
                hypothesis=hypothesis,
                shadow_bundle=bundle,
                ledger_entry=ledger_entry,
                gate_config={"minimum_shadow_sessions": 4, "minimum_ci_low": 0.0},
                build_principal=build_principal,
                approver_principal=approver_principal,
                signing_private_key_path=signing_key,
                signer_id=signer_id,
                promoted_at=datetime.now(UTC),
            )
            write_promotion_attestation(
                model,
                attestation,
                trust_store_path=trust_store,
            )
            manifest_path = model.with_suffix(model.suffix + ".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "promoted"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(DataReadinessError, "manifest changed"):
                verify_promotion_attestation(model, trust_store_path=trust_store)


def _declared_at() -> datetime:
    return datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def _declare(
    root: Path,
    *,
    candidate_sha: str = "c" * 64,
) -> dict[str, object]:
    test_signing_material()
    return declare_hypothesis(
        root,
        hypothesis_id="swing-alpha-001",
        hypothesis_family="swing-alpha",
        model_type="canonical_swing",
        candidate_artifact_sha256=candidate_sha,
        baseline_id="technical-baseline-v1",
        baseline_artifact_sha256="a" * 64,
        prediction_policy_sha256="f" * 64,
        execution_policy_sha256="8" * 64,
        shadow_view="swing",
        shadow_horizon="5d",
        shadow_decision_group_ids=(
            "2026-07-10T20:00:00+00:00",
            "2026-07-13T20:00:00+00:00",
            "2026-07-14T20:00:00+00:00",
            "2026-07-15T20:00:00+00:00",
        ),
        shadow_minimum_tickers_per_group=1,
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
    return write_test_shadow_bundle(
        root,
        sessions,
        hypothesis=hypothesis,
        candidate_artifact_sha256=candidate_sha,
        generated_at=_declared_at() + timedelta(days=20),
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
        "event_assignment_sha256": "a" * 64,
        "event_aggregate_sha256": "b" * 64,
        "label_material_sha256": "c" * 64,
        "label_source_reconciliation_sha256": "d" * 64,
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
        model_type="canonical_swing",
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
