"""Signed synthetic candidates for release-admission tests, never trading."""

from pathlib import Path

import joblib
import pandas as pd

from market_predictor.registry import write_model_manifest
from market_predictor.swing.contracts import SWING_MODEL_SCHEMA_VERSION, SWING_MODEL_TYPE
from tests.r4_fixtures import authorize_candidate_for_test, synthetic_identity_metrics

# The retired day-trading model identity, as historical candidate manifests record it.
RETIRED_INTRADAY_MODEL_TYPE = "canonical_intraday"
RETIRED_INTRADAY_SCHEMA_VERSION = "intraday.model.v1"
RETIRED_INTRADAY_EVIDENCE_SCHEMA = "intraday_training_evidence.v1"


def promoted_swing_candidate(root: Path, marker: str) -> tuple[Path, Path]:
    return _signed_candidate(root, marker, SWING_MODEL_TYPE, SWING_MODEL_SCHEMA_VERSION, "swing_training_evidence.v1")


def retired_intraday_candidate(root: Path, marker: str) -> tuple[Path, Path]:
    """A verifiable signed candidate that declares the retired day-trading model type."""
    return _signed_candidate(
        root, marker, RETIRED_INTRADAY_MODEL_TYPE, RETIRED_INTRADAY_SCHEMA_VERSION, RETIRED_INTRADAY_EVIDENCE_SCHEMA
    )


def _signed_candidate(
    root: Path, marker: str, model_type: str, schema_version: str, evidence_schema: str
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    model = root / f"{model_type}-{marker}.joblib"
    joblib.dump({"test_fixture": marker}, model)
    run_id = f"release-{model_type}-{marker}"
    metrics = synthetic_identity_metrics(model_run_id=run_id)
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
        model_type=model_type,
        schema_version=schema_version,
        target_col="target_10b",
        features=["return_1d"],
        training_data=training,
        metrics=metrics,
        validation_split="session_purged_walk_forward_and_ticker_holdout",
        extra={"model_run_id": run_id},
    )
    return model, authorize_candidate_for_test(model, metrics, evidence_schema=evidence_schema)
