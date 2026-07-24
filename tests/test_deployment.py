from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.deployment import (
    DEFAULT_ACTIVE_POINTER,
    DeploymentRoute,
    publish_serving_release,
    rollback_serving_release,
    sync_active_serving_release,
)
from market_predictor.feature_store import LiveFeatureStore
from market_predictor.live_features import live_feature_columns
from market_predictor.registry import write_model_manifest
from market_predictor.swing.contracts import (
    SWING_FEATURE_SCHEMA_VERSION,
    SWING_MODEL_SCHEMA_VERSION,
    SWING_MODEL_TYPE,
)
from market_predictor.v3.errors import DataReadinessError
from tests.r4_fixtures import authorize_candidate_for_test, synthetic_identity_metrics


class InMemoryBlobStore:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.hashes: dict[str, str] = {}

    def upload_file(
        self,
        local_path: Path,
        blob_relative: str | Path,
        *,
        overwrite: bool = True,
    ) -> str:
        return self.upload_bytes(local_path.read_bytes(), blob_relative, overwrite=overwrite)

    def upload_bytes(
        self,
        data: bytes,
        blob_relative: str | Path,
        *,
        overwrite: bool = True,
    ) -> str:
        key = str(blob_relative).replace("\\", "/")
        if key in self.blobs and not overwrite:
            raise FileExistsError(key)
        self.blobs[key] = bytes(data)
        self.hashes[key] = hashlib.sha256(data).hexdigest()
        return key

    def download_file(
        self,
        blob_relative: str | Path,
        local_path: Path,
        *,
        overwrite: bool = True,
    ) -> Path:
        if local_path.exists() and not overwrite:
            return local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.download_bytes(blob_relative))
        return local_path

    def download_bytes(self, blob_relative: str | Path) -> bytes:
        return self.blobs[str(blob_relative).replace("\\", "/")]

    def blob_exists(self, blob_relative: str | Path) -> bool:
        return str(blob_relative).replace("\\", "/") in self.blobs

    def blob_sha256(self, blob_relative: str | Path) -> str | None:
        return self.hashes.get(str(blob_relative).replace("\\", "/"))


class ServingDeploymentTests(unittest.TestCase):
    def test_publish_rollback_and_sync_use_complete_immutable_releases(self) -> None:
        generated = datetime(2026, 7, 22, 22, 5, tzinfo=UTC)
        store = InMemoryBlobStore()
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            source = Path(source_dir)
            target = Path(target_dir)
            model = _write_promoted_model(source, marker="first")
            live_store = _write_live_features(source, generated)
            routes = {"swing": {"5d": DeploymentRoute(model=model.relative_to(source), bar_timeframe="1Day")}}

            first = publish_serving_release(
                store,
                root=source,
                routes=routes,
                live_feature_store=live_store,
                generated_at=generated,
            )
            first_release = str(first["release_id"])
            first_model_hash = file_sha256(model)

            second_model = _write_promoted_model(source, marker="second")
            routes = {"swing": {"5d": DeploymentRoute(model=second_model.relative_to(source), bar_timeframe="1Day")}}
            second = publish_serving_release(
                store,
                root=source,
                routes=routes,
                live_feature_store=live_store,
                generated_at=generated,
            )
            self.assertNotEqual(first_release, second["release_id"])
            self.assertEqual(second["previous_release_id"], first_release)

            rolled_back = rollback_serving_release(
                store,
                release_id=first_release,
                activated_at=generated,
            )
            self.assertEqual(rolled_back["release_id"], first_release)
            self.assertEqual(rolled_back["previous_release_id"], second["release_id"])

            synced = sync_active_serving_release(store, root=target)
            self.assertEqual(synced["release_id"], first_release)
            self.assertEqual(file_sha256(target / "models/swing-first.joblib"), first_model_hash)
            self.assertTrue((target / "models/swing-first.joblib.manifest.json").exists())
            self.assertTrue((target / "data/live/features/swing.parquet").exists())
            self.assertTrue((target / "data/live/.active_release.json").exists())

    def test_rollback_rejects_incomplete_release(self) -> None:
        generated = datetime(2026, 7, 22, 22, 5, tzinfo=UTC)
        store = InMemoryBlobStore()
        with tempfile.TemporaryDirectory() as source_dir:
            source = Path(source_dir)
            model = _write_promoted_model(source, marker="first")
            live_store = _write_live_features(source, generated)
            pointer = publish_serving_release(
                store,
                root=source,
                routes={"swing": {"5d": DeploymentRoute(model=model.relative_to(source))}},
                live_feature_store=live_store,
                generated_at=generated,
            )
            release_id = str(pointer["release_id"])
            release_blob = f"serving/releases/{release_id}/release.json"
            release = json.loads(store.blobs[release_blob].decode("utf-8"))
            missing_blob = f"serving/releases/{release_id}/{release['assets'][0]['destination']}"
            del store.blobs[missing_blob]
            del store.hashes[missing_blob]

            with self.assertRaisesRegex(DataReadinessError, "incomplete"):
                publish_serving_release(
                    store,
                    root=source,
                    routes={"swing": {"5d": DeploymentRoute(model=model.relative_to(source))}},
                    live_feature_store=live_store,
                    generated_at=generated,
                )
            with self.assertRaisesRegex(DataReadinessError, "incomplete"):
                rollback_serving_release(store, release_id=release_id, activated_at=generated)

    def test_publish_resumes_verified_partial_release_assets(self) -> None:
        generated = datetime(2026, 7, 22, 22, 5, tzinfo=UTC)
        store = InMemoryBlobStore()
        with tempfile.TemporaryDirectory() as source_dir:
            source = Path(source_dir)
            model = _write_promoted_model(source, marker="first")
            live_store = _write_live_features(source, generated)
            routes = {"swing": {"5d": DeploymentRoute(model=model.relative_to(source))}}
            first = publish_serving_release(
                store,
                root=source,
                routes=routes,
                live_feature_store=live_store,
                generated_at=generated,
            )
            release_blob = str(first["release_manifest_blob"])
            del store.blobs[release_blob]
            del store.hashes[release_blob]
            del store.blobs[DEFAULT_ACTIVE_POINTER]
            del store.hashes[DEFAULT_ACTIVE_POINTER]

            resumed = publish_serving_release(
                store,
                root=source,
                routes=routes,
                live_feature_store=live_store,
                generated_at=generated,
            )

        self.assertEqual(resumed["release_id"], first["release_id"])
        self.assertIn(release_blob, store.blobs)

    def test_active_pointer_is_published_after_release_manifest(self) -> None:
        generated = datetime(2026, 7, 22, 22, 5, tzinfo=UTC)
        store = InMemoryBlobStore()
        with tempfile.TemporaryDirectory() as source_dir:
            source = Path(source_dir)
            model = _write_promoted_model(source, marker="first")
            live_store = _write_live_features(source, generated)
            pointer = publish_serving_release(
                store,
                root=source,
                routes={"swing": {"5d": DeploymentRoute(model=model.relative_to(source))}},
                live_feature_store=live_store,
                generated_at=generated,
            )

        self.assertIn(DEFAULT_ACTIVE_POINTER, store.blobs)
        self.assertIn(str(pointer["release_manifest_blob"]), store.blobs)

    def test_publish_rejects_model_target_incompatible_with_route(self) -> None:
        generated = datetime(2026, 7, 22, 22, 5, tzinfo=UTC)
        store = InMemoryBlobStore()
        with tempfile.TemporaryDirectory() as source_dir:
            source = Path(source_dir)
            model = _write_promoted_model(source, marker="first")
            live_store = _write_live_features(source, generated)

            with self.assertRaisesRegex(ValueError, "target horizon 5d"):
                publish_serving_release(
                    store,
                    root=source,
                    routes={"swing": {"1d": DeploymentRoute(model=model.relative_to(source))}},
                    live_feature_store=live_store,
                    generated_at=generated,
                )


def _write_promoted_model(root: Path, *, marker: str) -> Path:
    model = root / f"models/swing-{marker}.joblib"
    model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"marker": marker}, model)
    training = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "date": pd.date_range("2025-01-01", periods=4),
            "return_1d": [0.1, -0.1, 0.2, -0.2],
            "target_net_positive_5d": [1, 0, 1, 0],
        }
    )
    model_run_id = f"deployment-{marker}"
    metrics = {
        **synthetic_identity_metrics(model_type=SWING_MODEL_TYPE, model_run_id=model_run_id),
        "roc_auc": 0.7,
    }
    write_model_manifest(
        model_path=model,
        model_type=SWING_MODEL_TYPE,
        schema_version=SWING_MODEL_SCHEMA_VERSION,
        target_col="target_net_positive_5d",
        features=["return_1d"],
        training_data=training,
        metrics=metrics,
        validation_split="session_purged_walk_forward_and_ticker_holdout",
        extra={"model_run_id": model_run_id},
    )
    authorize_candidate_for_test(model, metrics)
    return model


def _write_live_features(root: Path, generated: datetime) -> LiveFeatureStore:
    store = LiveFeatureStore(root)
    cutoff = generated - timedelta(minutes=5)
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": [generated.date(), generated.date()],
            "decision_time_utc": [cutoff, cutoff],
            "feature_available_at_utc": [cutoff - timedelta(minutes=5)] * 2,
            "bar_available_at_utc": [cutoff - timedelta(hours=1, minutes=45)] * 2,
            "prediction_cutoff_policy_id": ["xnys_1800_america_new_york_v1"] * 2,
            "daily_bar_count": [250, 250],
            "source_coverage_end_utc_alpaca": [cutoff - timedelta(minutes=10)] * 2,
            "price_feed": ["sip", "sip"],
        }
    )
    missing = {column: pd.Series(0.0, index=frame.index) for column in live_feature_columns("swing") if column not in frame}
    frame = pd.concat([frame, pd.DataFrame(missing)], axis=1)
    store.publish(
        "swing",
        frame,
        price_feed="sip",
        feature_schema_version=SWING_FEATURE_SCHEMA_VERSION,
        source_artifact_sha256="b" * 64,
        source_artifact_type="swing_inference_features",
        generated_at=generated,
    )
    return store


if __name__ == "__main__":
    unittest.main()
