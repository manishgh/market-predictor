from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

from market_predictor.primary_v2.contracts import (
    SWING_V2_ID,
    load_primary_v2_research_config,
)
from market_predictor.primary_v2.experiments import (
    _load_complete_candidate,
    _load_complete_run,
    _publish_candidate,
    _write_authority,
    _write_json,
)
from market_predictor.primary_v2.model import (
    PrimaryV2ExperimentResult,
    primary_v2_experiment_specs,
)
from market_predictor.v3.errors import DataReadinessError

CONFIG = load_primary_v2_research_config(
    Path("configs/primary_strategy_v2.toml")
)
SPEC = primary_v2_experiment_specs(SWING_V2_ID)[0]


def test_rejected_candidate_is_immutable_and_retains_no_model(
    tmp_path: Path,
) -> None:
    candidate_dir = tmp_path / SPEC.candidate_id
    result = _result(status="rejected", final_candidate=None)

    record = _publish_candidate(
        candidate_dir,
        result=result,
        run_request_sha256="run-hash",
        config=CONFIG,
    )

    assert record["status"] == "rejected"
    assert not (candidate_dir / "model.joblib").exists()
    verified = _load_complete_candidate(
        candidate_dir,
        expected_run_request_sha256="run-hash",
    )
    assert verified["candidate_id"] == SPEC.candidate_id

    (candidate_dir / "metrics.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="does not verify"):
        _load_complete_candidate(
            candidate_dir,
            expected_run_request_sha256="run-hash",
        )


def test_accepted_candidate_requires_and_publishes_fitted_bundle(
    tmp_path: Path,
) -> None:
    missing = _result(
        status="accepted_development",
        final_candidate=None,
    )
    with pytest.raises(DataReadinessError, match="no final fitted bundle"):
        _publish_candidate(
            tmp_path / "missing-model",
            result=missing,
            run_request_sha256="run-hash",
            config=CONFIG,
        )

    accepted = _result(
        status="accepted_development",
        final_candidate={"fitted": True},
    )
    candidate_dir = tmp_path / "accepted"
    record = _publish_candidate(
        candidate_dir,
        result=accepted,
        run_request_sha256="run-hash",
        config=CONFIG,
    )
    assert record["status"] == "accepted_development"
    assert (candidate_dir / "model.joblib").is_file()


def test_complete_run_replays_candidate_artifact_hashes(
    tmp_path: Path,
) -> None:
    candidate_dir = tmp_path / SPEC.candidate_id
    candidate = _publish_candidate(
        candidate_dir,
        result=_result(status="rejected", final_candidate=None),
        run_request_sha256="run-hash",
        config=CONFIG,
    )
    manifest = {
        "schema": "primary_strategy_v2.run.v1",
        "request_sha256": "run-hash",
        "candidates": [candidate],
    }
    _write_json(tmp_path / "_manifest.json", manifest)
    _write_authority(
        tmp_path,
        state="complete",
        request_sha256="run-hash",
        artifact="_manifest.json",
        artifact_sha256=_sha256(tmp_path / "_manifest.json"),
    )

    loaded = _load_complete_run(
        tmp_path,
        expected_request_sha256="run-hash",
    )
    assert loaded["request_sha256"] == "run-hash"

    (candidate_dir / "metrics.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="does not verify"):
        _load_complete_run(
            tmp_path,
            expected_request_sha256="run-hash",
        )


def _result(
    *,
    status: Literal["accepted_development", "rejected"],
    final_candidate: object | None,
) -> PrimaryV2ExperimentResult:
    return PrimaryV2ExperimentResult(
        spec=SPEC,
        status=status,
        rejection_reasons=("failed gate",) if status == "rejected" else (),
        predictions=pd.DataFrame({"ticker": ["TEST"]}),
        selected_predictions=pd.DataFrame({"ticker": ["TEST"]}),
        economics=pd.DataFrame({"selected_trades": [1]}),
        regime_evidence=pd.DataFrame({"market_regime": ["neutral"]}),
        calibration_evidence=pd.DataFrame({"rows": [1]}),
        incremental_evidence=pd.DataFrame(),
        fold_audit=pd.DataFrame({"fold": [1]}),
        metrics={"status": status},
        final_candidate=final_candidate,
    )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
