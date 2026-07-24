from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from market_predictor.registry import write_model_manifest
from market_predictor.release import publish_local_release
from market_predictor.swing.contracts import (
    SWING_MODEL_SCHEMA_VERSION,
    SWING_MODEL_TYPE,
)
from tests.r4_fixtures import (
    authorize_candidate_for_test,
    synthetic_identity_metrics,
    test_signing_material,
)


def build_smoke_release(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"smoke output must be empty: {output}")
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)

    features = ["return_1d"]
    estimator = LogisticRegression(random_state=17).fit(
        [[-0.02], [-0.01], [0.01], [0.02]],
        [0, 0, 1, 1],
    )
    model_path = source / "swing-container-smoke.joblib"
    joblib.dump(
        {
            "model_type": SWING_MODEL_TYPE,
            "features": features,
            "model": estimator,
            "target_col": "target_net_positive_5d",
        },
        model_path,
    )
    model_run_id = "container-smoke-swing"
    metrics = {
        **synthetic_identity_metrics(
            model_type=SWING_MODEL_TYPE,
            model_run_id=model_run_id,
        ),
        "roc_auc": 0.75,
    }
    training = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "date": pd.date_range("2026-01-05", periods=4, freq="B"),
            "return_1d": [-0.02, -0.01, 0.01, 0.02],
            "target_net_positive_5d": [0, 0, 1, 1],
        }
    )
    write_model_manifest(
        model_path=model_path,
        model_type=SWING_MODEL_TYPE,
        schema_version=SWING_MODEL_SCHEMA_VERSION,
        target_col="target_net_positive_5d",
        features=features,
        training_data=training,
        metrics=metrics,
        validation_split="session_purged_walk_forward_and_ticker_holdout",
        extra={"model_run_id": model_run_id},
    )
    evidence = authorize_candidate_for_test(model_path, metrics)
    _, trust_store, _ = test_signing_material()
    copied_trust_store = output / "attestation_trust_store.json"
    shutil.copy2(trust_store, copied_trust_store)
    publish_local_release(
        output / "releases",
        model_path=model_path,
        evidence_manifest_path=evidence,
        attestation_trust_store_path=copied_trust_store,
    )
    (output / "app_config.toml").write_text(
        "\n".join(
            [
                "[prediction_serving]",
                'attestation_trust_store = "/smoke/attestation_trust_store.json"',
                "",
                '[prediction_serving.routes.swing."5d"]',
                'release_repository = "/smoke/releases"',
                'bar_timeframe = "1Day"',
                "estimated_resident_gib = 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an ephemeral signed release for container startup smoke tests."
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build_smoke_release(arguments.output.resolve())


if __name__ == "__main__":
    main()
