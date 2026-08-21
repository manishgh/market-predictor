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
from market_predictor.intraday.contracts import (
    INTRADAY_FEATURE_SCHEMA_VERSION,
    INTRADAY_MODEL_SCHEMA_VERSION,
    INTRADAY_MODEL_TYPE,
)
from market_predictor.live_features import live_feature_columns
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
from market_predictor.core.errors import DataReadinessError
from tests.r4_fixtures import (
    authorize_candidate_for_test,
    synthetic_identity_metrics,
)
from tests.r4_fixtures import test_signing_material as signing_material_for_test


class ProbabilityEstimatorStub:
    def predict_proba(self, data: object) -> object:
        del data
        raise AssertionError("cache tests do not score the estimator")


class ActiveModelContextCacheTests(unittest.TestCase):
    def test_deserializes_active_release_once_and_reuses_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repository"
            _, trust_store, _ = signing_material_for_test()
            model, evidence = _promoted_intraday_model(root / "source", "first")
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
                    bar_timeframe="5Min",
                )

                with patch(
                    "market_predictor.serving_context.joblib.load",
                    wraps=joblib.load,
                ) as load:
                    first = cache.get("intraday", "60m", route)
                    second = cache.get("intraday", "60m", route)

            self.assertIs(first, second)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(first.release_id, published["model_release_id"])
            self.assertEqual(first.serving_bundle_id, published["bundle_id"])
            self.assertEqual(cache.snapshot()["loaded_contexts"], 1)

    def test_atomically_replaces_context_after_active_pointer_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repository"
            _, trust_store, _ = signing_material_for_test()
            first_model, first_evidence = _promoted_intraday_model(root / "first", "first")
            second_model, second_evidence = _promoted_intraday_model(root / "second", "second")
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
            first = cache.get("intraday", "60m", route)

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
            second = cache.get("intraday", "60m", route)

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
            _, trust_store, _ = signing_material_for_test()
            first_model, first_evidence = _promoted_intraday_model(root / "first", "first")
            second_model, second_evidence = _promoted_intraday_model(root / "second", "second")
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
                    pending = pool.submit(cache.get, "intraday", "60m", route)
                    self.assertTrue(entered.wait(timeout=5))
                    activate_serving_bundle(
                        repository,
                        str(second_bundle["bundle_id"]),
                        attestation_trust_store_path=trust_store,
                        activated_at=_timestamp() + timedelta(minutes=3),
                    )
                    release.set()
                    in_flight = pending.result(timeout=5)

            current = cache.get("intraday", "60m", route)
            self.assertEqual(in_flight.serving_bundle_id, first_bundle["bundle_id"])
            self.assertEqual(in_flight.payload["marker"], "first")
            self.assertEqual(current.serving_bundle_id, second_bundle["bundle_id"])
            self.assertEqual(current.payload["marker"], "second")

    def test_rejects_payload_policy_that_conflicts_with_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repository"
            _, trust_store, _ = signing_material_for_test()
            model, evidence = _promoted_intraday_model(
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
                    "intraday",
                    "60m",
                    ActiveReleaseRoute(
                        repository=repository,
                        attestation_trust_store=trust_store,
                    ),
                )

    def test_rejects_bundle_above_route_feature_row_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repository"
            _, trust_store, _ = signing_material_for_test()
            model, evidence = _promoted_intraday_model(root / "model", "row-limit")
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
                    "intraday",
                    "60m",
                    ActiveReleaseRoute(
                        repository=repository,
                        attestation_trust_store=trust_store,
                        max_feature_rows=1,
                    ),
                )


def _promoted_intraday_model(
    root: Path,
    marker: str,
    *,
    payload_prediction_policy_sha256: str | None = None,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    model = root / f"intraday-{marker}.joblib"
    model_run_id = f"serving-context-{marker}"
    metrics = {
        **synthetic_identity_metrics(
            model_type=INTRADAY_MODEL_TYPE,
            model_run_id=model_run_id,
        ),
        "roc_auc": 0.75,
    }
    joblib.dump(
        {
            "model_type": INTRADAY_MODEL_TYPE,
            "model_schema_version": INTRADAY_MODEL_SCHEMA_VERSION,
            "feature_schema_version": INTRADAY_FEATURE_SCHEMA_VERSION,
            "features": ["return_1bar_5m"],
            "opportunity_target_col": "target_before_stop_60m",
            "downside_target_col": "stop_before_target_60m",
            "models": {
                "target_before_stop_60m": ProbabilityEstimatorStub(),
                "stop_before_target_60m": ProbabilityEstimatorStub(),
            },
            "calibrators": {
                "target_before_stop_60m": object(),
                "stop_before_target_60m": object(),
            },
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
            "return_1bar_5m": [0.01, -0.01],
            "target_before_stop_60m": [1, 0],
        }
    )
    write_model_manifest(
        model_path=model,
        model_type=INTRADAY_MODEL_TYPE,
        schema_version=INTRADAY_MODEL_SCHEMA_VERSION,
        target_col="target_before_stop_60m",
        features=["return_1bar_5m"],
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
    _publish_intraday(store, generated_at)
    feature_path, _ = store.paths("intraday")
    return publish_serving_bundle(
        repository,
        mode="intraday",
        horizon="60m",
        model_release_id=release_id,
        feature_path=feature_path,
        attestation_trust_store_path=trust_store,
        generated_at=generated_at,
    )


def _publish_intraday(
    store: LiveFeatureStore,
    generated_at: datetime,
) -> dict[str, object]:
    decision = pd.Timestamp(generated_at).tz_convert("UTC") - pd.Timedelta(minutes=1)
    frame = pd.DataFrame(
        {
            "ticker": ["MSFT", "AAPL"],
            "date": [decision, decision],
            "decision_time_utc": [decision, decision],
            "feature_available_at_utc": [decision, decision],
            "price_feed": ["sip", "sip"],
        }
    )
    missing = {
        column: pd.Series(0.0, index=frame.index)
        for column in live_feature_columns("intraday")
        if column not in frame
    }
    complete = pd.concat([frame, pd.DataFrame(missing)], axis=1)
    return store.publish(
        "intraday",
        complete,
        price_feed="sip",
        feature_schema_version=INTRADAY_FEATURE_SCHEMA_VERSION,
        source_artifact_sha256="a" * 64,
        source_artifact_type="intraday_inference_features",
        source_watermarks={"market:alpaca": decision.isoformat()},
        generated_at=generated_at,
    )


def _timestamp() -> datetime:
    return datetime(2026, 7, 10, 22, 5, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
