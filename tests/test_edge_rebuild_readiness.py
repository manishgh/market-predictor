import json
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.contracts import (
    load_edge_rebuild_readiness_config,
)
from market_predictor.edge_rebuild.readiness import (
    _intraday_fold_capacity,
    _json_sha256,
    _prepare_intraday_rows,
    _publish_audit,
    _swing_phase_capacity,
    _verify_intraday_coverage,
    load_complete_readiness_audit,
)
from market_predictor.v3.errors import DataReadinessError


def test_ten_session_phase_capacity_uses_independent_sessions() -> None:
    records = []
    for session_index in range(600):
        for ticker_index in range(20):
            records.append(
                {
                    "session_date_et": (
                        pd.Timestamp("2022-01-03")
                        + pd.offsets.BDay(session_index)
                    ).date(),
                    "decision_group_id": (
                        f"{session_index}-{ticker_index}"
                    ),
                    "ticker": f"T{ticker_index:02d}",
                    "source_usable": True,
                }
            )
    rows = pd.DataFrame(records)

    phases = _swing_phase_capacity(
        rows,
        phases=10,
        minimum_sessions=60,
        strategy_id="SWING.TEST.10D.V1",
    )

    assert len(phases) == 10
    assert set(phases["sessions"]) == {60}
    assert phases["status"].eq("pass").all()
    assert set(phases["source_rows"]) == {1_200}


def test_phase_capacity_does_not_confuse_rows_with_sessions() -> None:
    rows = pd.DataFrame(
        {
            "session_date_et": [pd.Timestamp("2025-01-02").date()] * 1_000,
            "decision_group_id": [f"D{i}" for i in range(1_000)],
            "ticker": [f"T{i % 50}" for i in range(1_000)],
            "source_usable": True,
        }
    )

    phases = _swing_phase_capacity(
        rows,
        phases=10,
        minimum_sessions=60,
        strategy_id="SWING.TEST.10D.V1",
    )

    assert phases["sessions"].sum() == 1
    assert phases["status"].eq("blocked").all()


def test_intraday_entry_at_decision_open_is_causally_valid() -> None:
    decision = pd.Timestamp("2025-01-02 15:31:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "session_date_et": ["2025-01-02"],
            "regime_risk_on": [False],
            "regime_risk_off": [False],
            "decision_time_utc": [decision],
            "feature_available_at_utc": [
                decision - pd.Timedelta(seconds=30)
            ],
            "entry_time_utc": [decision],
            "feature_eligible": [True],
            "one_minute_history_exact": [True],
            "observed_fraction_130": [1.0],
            "price_feed": ["sip"],
            "adjustment": ["all"],
            "ticker": ["TEST"],
            "setup_id": ["setup-1"],
            "universe_snapshot_id": ["snapshot-1"],
            "label_eligible": [True],
            "label_ineligible_reason": [""],
        }
    )

    rows, exclusions = _prepare_intraday_rows(
        frame,
        config=load_edge_rebuild_readiness_config(
            Path("configs/edge_rebuild_readiness.toml")
        ),
    )

    assert rows["source_usable"].all()
    entry_exclusion = exclusions.loc[
        exclusions["reason"].eq("entry_before_decision")
    ].iloc[0]
    assert entry_exclusion["excluded_rows"] == 0


def test_intraday_fold_capacity_counts_sessions_not_rows() -> None:
    rows = pd.DataFrame(
        {
            "session_date_et": [
                (pd.Timestamp("2023-01-02") + pd.offsets.BDay(index)).date()
                for index in range(240)
                for _ in range(50)
            ],
            "source_usable": True,
        }
    )

    folds = _intraday_fold_capacity(
        rows,
        folds=4,
        minimum_test_sessions=60,
        strategy_id="INTRADAY.TEST.30M.V1",
    )

    assert list(folds["test_sessions"]) == [60, 60, 60, 60]
    assert folds["status"].eq("pass").all()


def test_intraday_coverage_verification_detects_source_tampering(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "coverage" / "2025-01.parquet"
    artifact.parent.mkdir()
    pd.DataFrame({"exact": [True]}).to_parquet(artifact, index=False)
    record = {
        "path": "coverage/2025-01.parquet",
        "bytes": artifact.stat().st_size,
        "sha256": file_sha256(artifact),
    }
    manifest = {
        "schema": "intraday.specialist_coverage_audit.v1",
        "coverage_fingerprint": "f" * 64,
        "collection": {"manifest_sha256": "c" * 64},
        "summary": {"requirements": 1},
        "files": [record],
    }
    (tmp_path / "_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    verified = _verify_intraday_coverage(
        tmp_path,
        expected_collection_manifest_sha256="c" * 64,
    )
    assert verified["coverage_fingerprint"] == "f" * 64

    artifact.write_bytes(b"tampered")
    with pytest.raises(DataReadinessError, match="artifact changed"):
        _verify_intraday_coverage(
            tmp_path,
            expected_collection_manifest_sha256="c" * 64,
        )


def test_immutable_readiness_publication_detects_tampering(
    tmp_path: Path,
) -> None:
    request = {
        "schema": "edge_rebuild.readiness.run.v1",
        "training_performed": False,
    }
    request_sha256 = _json_sha256(request)
    evidence = {
        "blockers.csv": pd.DataFrame({"code": ["none"]}),
        "catalyst_readiness.csv": pd.DataFrame({"status": ["pass"]}),
        "cost_readiness.csv": pd.DataFrame({"cost": [0.001]}),
        "dimension_coverage.csv": pd.DataFrame({"dimension": ["year"]}),
        "exclusion_reasons.csv": pd.DataFrame({"reason": ["none"]}),
        "fold_capacity.csv": pd.DataFrame({"fold": [0]}),
        "phase_capacity.csv": pd.DataFrame({"phase": [0]}),
        "session_calendar.csv": pd.DataFrame({"session": ["2025-01-02"]}),
        "source_inventory.csv": pd.DataFrame({"source": ["test"]}),
    }
    root = tmp_path / "audit"

    _publish_audit(
        root,
        request=request,
        request_sha256=request_sha256,
        summary={"status": "ready_for_ER2"},
        evidence=evidence,
    )

    assert (
        load_complete_readiness_audit(
            root,
            expected_request_sha256=request_sha256,
        )["status"]
        == "ready_for_ER2"
    )
    (root / "phase_capacity.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="does not verify"):
        load_complete_readiness_audit(
            root,
            expected_request_sha256=request_sha256,
        )
