from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from market_predictor.registry import write_model_manifest
from market_predictor.release import publish_local_release
from market_predictor.serving_context import (
    ActiveModelContextCache,
    ActiveReleaseRoute,
)
from market_predictor.swing.contracts import (
    SWING_MODEL_SCHEMA_VERSION,
    SWING_MODEL_TYPE,
)
from tests.r4_fixtures import (
    authorize_candidate_for_test,
    synthetic_identity_metrics,
    test_signing_material,
)


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
                published = publish_local_release(
                    repository,
                    model_path=model,
                    evidence_manifest_path=evidence,
                    attestation_trust_store_path=trust_store,
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
            self.assertEqual(first.release_id, published["release_id"])
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
            )
            second = cache.get("swing", "5d", route)

            self.assertEqual(first.release_id, first_release["release_id"])
            self.assertEqual(second.release_id, second_release["release_id"])
            self.assertIsNot(first, second)
            self.assertEqual(cache.snapshot()["loaded_contexts"], 1)


def _promoted_swing_model(root: Path, marker: str) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    model = root / f"swing-{marker}.joblib"
    joblib.dump(
        {
            "model_type": SWING_MODEL_TYPE,
            "features": ["return_1d"],
            "model": ProbabilityEstimatorStub(),
            "target_col": "target_net_positive_5d",
            "marker": marker,
        },
        model,
    )
    model_run_id = f"serving-context-{marker}"
    metrics = {
        **synthetic_identity_metrics(
            model_type=SWING_MODEL_TYPE,
            model_run_id=model_run_id,
        ),
        "roc_auc": 0.75,
    }
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


if __name__ == "__main__":
    unittest.main()
