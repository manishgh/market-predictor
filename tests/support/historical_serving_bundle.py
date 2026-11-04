"""Synthetic historical bundle bytes for read-only regression tests only.

No production command imports this fixture builder, and it cannot construct the
current active_generation serving contract or supply model promotion evidence.
"""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.evidence.hashing import json_sha256
from market_predictor.feature_store import LiveFeatureStore
from market_predictor.release import publish_local_release, verify_local_release
from market_predictor.serving.bundle import ACTIVE_SERVING_BUNDLE_SCHEMA, SERVING_BUNDLE_SCHEMA
from tests.r4_fixtures import test_signing_material as signing_material_for_test
from tests.support.swing_release import promoted_swing_candidate
from tests.test_feature_store import _frame as swing_frame
from tests.test_feature_store import _publish as publish_swing_features


def historical_bundle_timestamp() -> datetime:
    return datetime(2026, 7, 10, 22, 5, tzinfo=UTC)


def historical_bundle_inputs(root: Path, marker: str) -> tuple[Path, Path, str, Path]:
    repository = root / "repository"
    _, trust_store, _ = signing_material_for_test()
    model, evidence = promoted_swing_candidate(root / "models" / marker, marker)
    release = publish_local_release(repository, model_path=model, evidence_manifest_path=evidence,
        attestation_trust_store_path=trust_store, activate=False)
    store = LiveFeatureStore(root)
    publish_swing_features(store, swing_frame(), historical_bundle_timestamp())
    feature_path, _ = store.paths("swing")
    return repository, trust_store, str(release["release_id"]), feature_path


def historical_bundle_fixture(repository: Path, trust_store: Path, release_id: str, features: Path) -> dict[str, Any]:
    """Write one old-format snapshot and its recorded pointer inside a UNIT fixture."""
    release = verify_local_release(repository, release_id, attestation_trust_store_path=trust_store)
    released = repository / "releases" / release_id
    candidate = json.loads((released / release["candidate_manifest_path"]).read_text(encoding="utf-8"))
    metrics = candidate["metrics"]
    sidecar = manifest_path_for(features)
    feature = json.loads(sidecar.read_text(encoding="utf-8"))
    calibration = metrics["calibration_method"]
    identity = {"schema": SERVING_BUNDLE_SCHEMA, "mode": "swing", "horizon": "10b",
        "generated_at_utc": historical_bundle_timestamp().isoformat(), "model_release_id": release_id,
        "model_release_manifest_sha256": file_sha256(released / "release.json"),
        "model_artifact_sha256": candidate["artifact_sha256"], "calibration_method": calibration,
        "calibration_identity_sha256": json_sha256({"model_artifact_sha256": candidate["artifact_sha256"],
                                                   "calibration_method": calibration}),
        "prediction_policy_sha256": candidate.get("prediction_policy_sha256") or metrics["prediction_policy_sha256"],
        "label_policy_sha256": candidate.get("dataset_label_config_sha256") or metrics["dataset_label_config_sha256"],
        "execution_policy_sha256": candidate.get("execution_policy_sha256") or metrics["execution_policy_sha256"],
        "feature_path": "features/features.parquet", "feature_manifest_path": "features/features.parquet.manifest.json",
        "feature_manifest_sha256": file_sha256(sidecar), "feature_artifact_sha256": feature["artifact_sha256"],
        "feature_schema_version": feature["feature_schema_version"],
        "feature_source_artifact_sha256": feature["source_artifact_sha256"],
        "feature_source_artifact_type": feature["source_artifact_type"],
        "feature_columns_sha256": json_sha256({"columns": feature["columns"]})}
    bundle_id = json_sha256(identity)
    bundle = {**identity, "bundle_id": bundle_id}
    directory = repository / "serving_bundles" / bundle_id
    (directory / "features").mkdir(parents=True)
    shutil.copyfile(features, directory / identity["feature_path"])
    shutil.copyfile(sidecar, directory / identity["feature_manifest_path"])
    (directory / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    pointer = {"schema": ACTIVE_SERVING_BUNDLE_SCHEMA, "bundle_id": bundle_id,
        "bundle_manifest_sha256": file_sha256(directory / "bundle.json"), "previous_bundle_id": None,
        "activated_at_utc": historical_bundle_timestamp().isoformat()}
    (repository / "active_serving_bundle.json").write_text(
        json.dumps({**pointer, "pointer_sha256": json_sha256(pointer)}), encoding="utf-8")
    return bundle
