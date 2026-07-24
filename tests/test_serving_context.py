from __future__ import annotations

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from market_predictor import serving_context as serving_context_module
from market_predictor.feature_store import LiveFeatureStore
from market_predictor.registry import write_model_manifest
from market_predictor.release import publish_local_release
from market_predictor.serving_bundle import (
    activate_serving_bundle,
    publish_serving_bundle,
)
from market_predictor.serving_context import (
    ActiveModelContextCache,
    ActiveReleaseRoute,
)
from market_predictor.swing.contracts import (
    SWING_FEATURE_SCHEMA_VERSION,
    SWING_MODEL_SCHEMA_VERSION,
    SWING_MODEL_TYPE,
)
from market_predictor.v3.errors import DataReadinessError
from tests.r4_fixtures import (
    authorize_candidate_for_test,
    synthetic_identity_metrics,
    test_signing_material,
)
from tests.test_feature_store import _frame, _publish


class ProbabilityEstimatorStub:
    def predict_proba(self, data: object) -> object:
        del data
        raise AssertionError("cache tests do not score the estimator")


class ActiveModelContextCacheTests(unittest.TestCase):
    def test_deserializes_active_release_once_and_reuses_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repository"
            _, trust_store, _ = test_signing_material()
            model, evidence = _promoted_swing_model(root / "source", "first")
            with patch.dict(
                os.environ,
                {"MARKET_PREDICTOR_ATTESTATION_TRUST_STORE": ""},
            ):
                release = publish_local_release(
                    repository,
                    model_path=model,
                    evidence_manifest_path=evidence,
                    attestation_trust_store_path=trust_store,
                    activate=False,
                )
                published = _publish_bundle(
                    root,
                    repository,
                    trust_store,
                    str(release["release_id"]),
                    generated_at=_timestamp(),
                )
                cache = ActiveModelContextCache(
                    root,
                    memory_budget_gib=4.0,
                    memory_headroom_gib=0.25,
                    max_contexts=1,
                )
                route = ActiveReleaseRoute(
                    repository=repository,
                    attestation_trust_store=trust_store,
                    bar_timeframe="1Day",
                )

                with patch(
                    "market_predictor.serving_context.joblib.load",
                    wraps=joblib.load,
                ) as load:
                    first = cache.get("swing", "5d", route)
                    second = cache.get("swing", "5d", route)

            self.assertIs(first, second)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(first.release_id, published["model_release_id"])
            self.assertEqual(first.serving_bundle_id, published["bundle_id"])
            self.assertEqual(cache.snapshot()["loaded_contexts"], 1)

    def test_atomically_replaces_context_after_active_pointer_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repository"
            _, trust_store, _ = test_signing_material()
            first_model, first_evidence = _promoted_swing_model(root / "first", "first")
            second_model, second_evidence = _promoted_swing_model(root / "second", "second")
            first_release = publish_local_release(
                repository,
                model_path=first_model,
                evidence_manifest_path=first_evidence,
                attestation_trust_store_path=trust_store,
                activate=False,
            )
            first_bundle = _publish_bundle(
                root,
                repository,
                trust_store,
                str(first_release["release_id"]),
                generated_at=_timestamp(),
            )
            cache = ActiveModelContextCache(
                root,
                memory_budget_gib=4.0,
                memory_headroom_gib=0.25,
                max_contexts=1,
            )
            route = ActiveReleaseRoute(
                repository=repository,
                attestation_trust_store=trust_store,
            )
            first = cache.get("swing", "5d", route)

            second_release = publish_local_release(
                repository,
                model_path=second_model,
                evidence_manifest_path=second_evidence,
                attestation_trust_store_path=trust_store,
                activate=False,
            )
            second_bundle = _publish_bundle(
                root,
                repository,
                trust_store,
                str(second_release["release_id"]),
                generated_at=_timestamp() + timedelta(minutes=1),
            )
            second = cache.get("swing", "5d", route)

            self.assertEqual(first.release_id, first_release["release_id"])
            self.assertEqual(second.release_id, second_release["release_id"])
            self.assertEqual(first.serving_bundle_id, first_bundle["bundle_id"])
            self.assertEqual(second.serving_bundle_id, second_bundle["bundle_id"])
            self.assertIsNot(first, second)
            self.assertEqual(cache.snapshot()["loaded_contexts"], 1)

    def test_pointer_flip_during_load_never_mixes_bundle_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repository"
            _, trust_store, _ = test_signing_material()
            first_model, first_evidence = _promoted_swing_model(root / "first", "first")
            second_model, second_evidence = _promoted_swing_model(root / "second", "second")
            first_release = publish_local_release(
                repository,
                model_path=first_model,
                evidence_manifest_path=first_evidence,
                attestation_trust_store_path=trust_store,
                activate=False,
            )
            first_bundle = _publish_bundle(
                root,
                repository,
                trust_store,
                str(first_release["release_id"]),
                generated_at=_timestamp(),
            )
            second_release = publish_local_release(
                repository,
                model_path=second_model,
                evidence_manifest_path=second_evidence,
                attestation_trust_store_path=trust_store,
                activate=False,
            )
            second_bundle = _publish_bundle(
                root,
                repository,
                trust_store,
                str(second_release["release_id"]),
                generated_at=_timestamp() + timedelta(minutes=1),
            )
            activate_serving_bundle(
                repository,
                str(first_bundle["bundle_id"]),
                attestation_trust_store_path=trust_store,
                activated_at=_timestamp() + timedelta(minutes=2),
            )
            cache = ActiveModelContextCache(
                root,
                memory_budget_gib=4.0,
                memory_headroom_gib=0.25,
                max_contexts=1,
            )
            route = ActiveReleaseRoute(
                repository=repository,
                attestation_trust_store=trust_store,
            )
            entered = threading.Event()
            release = threading.Event()
            original_load = serving_context_module._load_joblib_from_verified_handle

            def blocking_load(*args: object, **kwargs: object) -> object:
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release bundle load")
                return original_load(*args, **kwargs)  # type: ignore[arg-type]

            with patch(
                "market_predictor.serving_context._load_joblib_from_verified_handle",
                side_effect=blocking_load,
            ):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(cache.get, "swing", "5d", route)
                    self.assertTrue(entered.wait(timeout=5))
                    activate_serving_bundle(
                        repository,
                        str(second_bundle["bundle_id"]),
                        attestation_trust_store_path=trust_store,
                        activated_at=_timestamp() + timedelta(minutes=3),
                    )
                    release.set()
                    in_flight = pending.result(timeout=5)

            current = cache.get("swing", "5d", route)
            self.assertEqual(in_flight.serving_bundle_id, first_bundle["bundle_id"])
            self.assertEqual(in_flight.payload["marker"], "first")
            self.assertEqual(current.serving_bundle_id, second_bundle["bundle_id"])
            self.assertEqual(current.payload["marker"], "second")

    def test_rejects_payload_policy_that_conflicts_with_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repository"
            _, trust_store, _ = test_signing_material()
            model, evidence = _promoted_swing_model(
                root / "model",
                "mismatch",
                payload_prediction_policy_sha256="0" * 64,
            )
            release = publish_local_release(
                repository,
                model_path=model,
                evidence_manifest_path=evidence,
                attestation_trust_store_path=trust_store,
                activate=False,
            )
            _publish_bundle(
                root,
                repository,
                trust_store,
                str(release["release_id"]),
                generated_at=_timestamp(),
            )
            cache = ActiveModelContextCache(
                root,
                memory_budget_gib=4.0,
                memory_headroom_gib=0.25,
                max_contexts=1,
            )

            with self.assertRaisesRegex(
                DataReadinessError,
                "prediction policy is incompatible",
            ):
                cache.get(
                    "swing",
                    "5d",
                    ActiveReleaseRoute(
                        repository=repository,
                        attestation_trust_store=trust_store,
                    ),
                )

    def test_rejects_bundle_above_route_feature_row_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repository"
            _, trust_store, _ = test_signing_material()
            model, evidence = _promoted_swing_model(root / "model", "row-limit")
            release = publish_local_release(
                repository,
                model_path=model,
                evidence_manifest_path=evidence,
                attestation_trust_store_path=trust_store,
                activate=False,
            )
            _publish_bundle(
                root,
                repository,
                trust_store,
                str(release["release_id"]),
                generated_at=_timestamp(),
            )
            cache = ActiveModelContextCache(
                root,
                memory_budget_gib=4.0,
                memory_headroom_gib=0.25,
                max_contexts=1,
            )

            with self.assertRaisesRegex(DataReadinessError, "feature row limit"):
                cache.get(
                    "swing",
                    "5d",
                    ActiveReleaseRoute(
                        repository=repository,
                        attestation_trust_store=trust_store,
                        max_feature_rows=1,
                    ),
                )


def _promoted_swing_model(
    root: Path,
    marker: str,
    *,
    payload_prediction_policy_sha256: str | None = None,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    model = root / f"swing-{marker}.joblib"
    model_run_id = f"serving-context-{marker}"
    metrics = {
        **synthetic_identity_metrics(
            model_type=SWING_MODEL_TYPE,
            model_run_id=model_run_id,
        ),
        "roc_auc": 0.75,
    }
    joblib.dump(
        {
            "model_type": SWING_MODEL_TYPE,
            "model_schema_version": SWING_MODEL_SCHEMA_VERSION,
            "feature_schema_version": SWING_FEATURE_SCHEMA_VERSION,
            "features": ["return_1d"],
            "model": ProbabilityEstimatorStub(),
            "calibrator": object(),
            "target_col": "target_net_positive_5d",
            "calibration_method": "isotonic_prior_fold_only",
            "prediction_policy_sha256": (
                payload_prediction_policy_sha256
                or metrics["prediction_policy_sha256"]
            ),
            "marker": marker,
        },
        model,
    )
    training = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": pd.date_range("2026-01-01", periods=2),
            "return_1d": [0.01, -0.01],
            "target_net_positive_5d": [1, 0],
        }
    )
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
    evidence = authorize_candidate_for_test(model, metrics)
    return model, evidence


def _publish_bundle(
    root: Path,
    repository: Path,
    trust_store: Path,
    release_id: str,
    *,
    generated_at: datetime,
) -> dict[str, object]:
    store = LiveFeatureStore(root)
    _publish(store, _frame(), generated_at)
    feature_path, _ = store.paths("swing")
    return publish_serving_bundle(
        repository,
        mode="swing",
        horizon="5d",
        model_release_id=release_id,
        feature_path=feature_path,
        attestation_trust_store_path=trust_store,
        generated_at=generated_at,
    )


def _timestamp() -> datetime:
    return datetime(2026, 7, 10, 22, 5, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
