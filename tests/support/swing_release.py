"""Signed synthetic swing artifacts for release-admission tests, never trading."""

from pathlib import Path

import joblib
import pandas as pd

from market_predictor.registry import write_model_manifest
from market_predictor.swing.contracts import SWING_MODEL_SCHEMA_VERSION, SWING_MODEL_TYPE
from tests.r4_fixtures import authorize_candidate_for_test, synthetic_identity_metrics


def promoted_swing_candidate(root: Path, marker: str) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    model = root / f"swing-{marker}.joblib"
    joblib.dump({"test_fixture": marker}, model)
    run_id = f"swing-release-{marker}"
    metrics = synthetic_identity_metrics(model_type=SWING_MODEL_TYPE, model_run_id=run_id)
    training = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": pd.date_range("2026-01-01", periods=2),
            "return_1d": [0.01, -0.01],
            "target_10b": [1, 0],
        }
    )
    write_model_manifest(
        model_path=model,
        model_type=SWING_MODEL_TYPE,
        schema_version=SWING_MODEL_SCHEMA_VERSION,
        target_col="target_10b",
        features=["return_1d"],
        training_data=training,
        metrics=metrics,
        validation_split="session_purged_walk_forward_and_ticker_holdout",
        extra={"model_run_id": run_id},
    )
    return model, authorize_candidate_for_test(model, metrics)
