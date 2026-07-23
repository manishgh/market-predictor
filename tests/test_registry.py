from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd

from market_predictor.registry import (
    feature_schema_hash,
    file_sha256,
    manifest_path_for,
    verify_model_artifact,
    write_model_manifest,
)
from tests.r4_fixtures import authorize_candidate_for_test, synthetic_identity_metrics


class ModelRegistryTests(unittest.TestCase):
    def test_verification_rejects_unregistered_and_modified_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump({"model": "placeholder"}, path)
            with self.assertRaisesRegex(FileNotFoundError, "Missing model manifest"):
                verify_model_artifact(path)

            _write_candidate(path)
            with path.open("ab") as handle:
                handle.write(b"modified")
            with self.assertRaisesRegex(ValueError, "integrity check failed"):
                verify_model_artifact(path)

    def test_writes_immutable_candidate_manifest_with_artifact_and_feature_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump({"model": "placeholder"}, path)
            _write_candidate(path)
            loaded = json.loads(manifest_path_for(path).read_text(encoding="utf-8"))

            self.assertEqual(loaded["schema"], "model_registry_manifest.v2")
            self.assertEqual(loaded["status"], "candidate")
            self.assertEqual(loaded["artifact_sha256"], file_sha256(path))
            self.assertEqual(loaded["dataset"]["feature_schema_hash"], feature_schema_hash(["return_1d", "volume_z20"]))
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                _write_candidate(path)

    def test_candidate_writer_has_no_status_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump({"model": "placeholder"}, path)
            with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'status'"):
                write_model_manifest(
                    model_path=path,
                    model_type="swing_classifier_v1",
                    schema_version="swing.model.v1",
                    target_col="target",
                    features=["return_1d"],
                    training_data=_training_frame(["return_1d"], rows=10, tickers=2),
                    metrics={},
                    validation_split="session_purged_walk_forward_and_ticker_holdout",
                    status="promoted",  # type: ignore[call-arg]
                )

    def test_promoted_is_derived_only_from_valid_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump({"model": "placeholder"}, path)
            metrics = _write_candidate(path)
            with self.assertRaisesRegex(ValueError, "not allowed"):
                verify_model_artifact(path, allowed_statuses={"promoted"})

            authorize_candidate_for_test(path, metrics)
            verified = verify_model_artifact(path, allowed_statuses={"promoted"})
            self.assertEqual(verified["status"], "promoted")
            self.assertIn("promotion_attestation", verified)
            self.assertEqual(json.loads(manifest_path_for(path).read_text(encoding="utf-8"))["status"], "candidate")

    def test_self_declared_promoted_manifest_is_rejected_even_with_matching_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump({"model": "placeholder"}, path)
            _write_candidate(path)
            manifest_path = manifest_path_for(path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "promoted"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must remain candidate"):
                verify_model_artifact(path, allowed_statuses={"promoted"})


def _write_candidate(path: Path) -> dict[str, object]:
    features = ["return_1d", "volume_z20"]
    metrics = {
        **synthetic_identity_metrics(model_type="swing_classifier_v1", model_run_id="registry-test-run"),
        "roc_auc": 0.7,
    }
    write_model_manifest(
        model_path=path,
        model_type="swing_classifier_v1",
        schema_version="swing.model.v1",
        target_col="target",
        features=features,
        training_data=_training_frame(features, rows=10, tickers=2),
        metrics=metrics,
        validation_split="session_purged_walk_forward_and_ticker_holdout",
        extra={"model_run_id": "registry-test-run"},
    )
    return metrics


def _training_frame(features: list[str], *, rows: int, tickers: int) -> pd.DataFrame:
    start = date(2026, 1, 1)
    records = []
    for idx in range(rows):
        row = {
            "ticker": f"T{idx % tickers:03d}",
            "date": start + timedelta(days=idx),
            "target": idx % 2,
        }
        for feature in features:
            row[feature] = float(idx % 7)
        records.append(row)
    return pd.DataFrame(records)


if __name__ == "__main__":
    unittest.main()
