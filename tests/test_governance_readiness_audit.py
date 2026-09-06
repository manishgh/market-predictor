import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.training.swing_types import SwingPanelBinding
from market_predictor.governance.readiness.audit import (
    _SWING_PANEL_READINESS_COLUMNS,
    VerifiedCatalystSources,
    _assert_projected_dataframe_concat,
    _assert_projected_parquet_load,
    _bool_series,
    _build_summary,
    _catalyst_readiness,
    _intraday_fold_capacity,
    _prepare_intraday_rows,
    _read_current_swing_panel,
    _swing_phase_capacity,
    _verify_candidate_panel_binding,
    _verify_catalyst_sources,
    _verify_intraday_coverage,
)
from market_predictor.governance.readiness.contracts import (
    load_prediction_data_readiness_config,
)
from market_predictor.swing.features.panel import SWING_FEATURE_PROFILE


def test_candidate_must_bind_current_ten_session_panel(tmp_path: Path) -> None:
    binding = SwingPanelBinding(
        root=tmp_path,
        manifest={},
        manifest_sha256="a" * 64,
        authority_sha256="b" * 64,
        request_sha256="c" * 64,
        strategy_contract_sha256="d" * 64,
    )
    candidate = {
        "model_card": {
            "dataset": {
                "panel_manifest_sha256": "a" * 64,
                "panel_authority_sha256": "b" * 64,
                "panel_request_sha256": "c" * 64,
                "strategy_contract_sha256": "d" * 64,
            }
        }
    }

    _verify_candidate_panel_binding(candidate, binding)
    candidate["model_card"]["dataset"]["panel_manifest_sha256"] = "e" * 64
    with pytest.raises(DataReadinessError, match="active panel"):
        _verify_candidate_panel_binding(candidate, binding)


def test_readiness_reads_the_single_technical_swing_population(tmp_path: Path) -> None:
    relative_path = "panel/technical.parquet"
    partition = tmp_path / "final" / relative_path
    partition.parent.mkdir(parents=True)
    row = {column: 0 for column in _SWING_PANEL_READINESS_COLUMNS}
    row.update(
        {
            "decision_time_utc": pd.Timestamp("2025-01-02T21:00:00Z"),
            "entry_time_utc": pd.Timestamp("2025-01-03T14:30:00Z"),
            "feature_available_at_utc": pd.Timestamp("2025-01-02T21:00:00Z"),
            "label_available_at_utc": pd.Timestamp("2025-01-17T21:00:00Z"),
            "barrier_label_available_at_utc": pd.Timestamp("2025-01-17T21:00:00Z"),
            "feature_eligible": True,
            "cross_section_eligible": True,
            "label_eligible": True,
            "label_path_exact": True,
            "ticker": "TEST",
        }
    )
    pd.DataFrame([row]).to_parquet(partition, index=False)
    binding = SwingPanelBinding(
        root=tmp_path,
        manifest={
            "rows": 1,
            "files_by_profile": {SWING_FEATURE_PROFILE: [{"path": relative_path, "rows": 1}]},
        },
        manifest_sha256="a" * 64,
        authority_sha256="b" * 64,
        request_sha256="c" * 64,
        strategy_contract_sha256="d" * 64,
    )

    result = _read_current_swing_panel(
        binding,
        config=load_prediction_data_readiness_config(Path("configs/prediction_data_readiness.toml")),
    )

    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "TEST"


def test_ten_session_phase_capacity_uses_independent_sessions() -> None:
    records = []
    for session_index in range(600):
        for ticker_index in range(20):
            records.append(
                {
                    "session_date_et": (pd.Timestamp("2022-01-03") + pd.offsets.BDay(session_index)).date(),
                    "decision_group_id": (f"{session_index}-{ticker_index}"),
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
            "feature_available_at_utc": [decision - pd.Timedelta(seconds=30)],
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
        config=load_prediction_data_readiness_config(Path("configs/prediction_data_readiness.toml")),
    )

    assert rows["source_usable"].all()
    entry_exclusion = exclusions.loc[exclusions["reason"].eq("entry_before_decision")].iloc[0]
    assert entry_exclusion["excluded_rows"] == 0


def test_intraday_fold_capacity_counts_sessions_not_rows() -> None:
    rows = pd.DataFrame(
        {
            "session_date_et": [(pd.Timestamp("2023-01-02") + pd.offsets.BDay(index)).date() for index in range(240) for _ in range(50)],
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


@pytest.mark.parametrize("value", [2, -1, "yes", "truthy"])
def test_readiness_rejects_noncanonical_boolean_values(value: object) -> None:
    with pytest.raises(DataReadinessError, match="Boolean column"):
        _bool_series(pd.Series([value]))


def test_only_intraday_history_shortage_authorizes_collection() -> None:
    config = load_prediction_data_readiness_config(Path("configs/prediction_data_readiness.toml"))
    inventory = pd.DataFrame(
        [{"strategy_id": config.intraday.strategy_id, "valid_sessions": 500, "first_usable_decision_time_utc": "2025-01-02T15:30:00Z"}]
    )

    def summary_for(code: str, scope: str) -> dict[str, object]:
        evidence = {
            "blockers.csv": pd.DataFrame(
                [{"blocker_code": code, "scope": scope, "blocks_er2": True}]
            ),
            "source_inventory.csv": inventory,
        }
        return _build_summary(evidence, config=config, request_sha256="a" * 64)

    intraday = summary_for(
        "intraday_session_history_below_gate",
        config.intraday.strategy_id,
    )
    swing = summary_for(
        "swing_session_history_below_gate",
        config.swing.strategy_id,
    )

    assert intraday["acquisition_plan"]["authorized_by_audit"] is True
    assert swing["acquisition_plan"]["authorized_by_audit"] is False


def test_catalyst_readiness_counts_only_known_eligible_complete_coverage() -> None:
    config = load_prediction_data_readiness_config(Path("configs/prediction_data_readiness.toml"))
    catalyst = VerifiedCatalystSources(
        identity={},
        lineage_manifest={
            "channel_counts": {
                "direct_issuer": 1,
                "business_exposure": 1,
                "sector_context": 1,
                "global_context": 1,
            },
            "coverage": {"states": {"observed_complete": 2}},
            "source_event_rows": 2,
            "training_eligible_rows": 1,
        },
        feature_inventory={
            "availability_policy": "observed",
            "training_eligible_channels": ["direct_issuer"],
        },
        news_manifest={"availability_policy": "observed"},
        coverage=pd.DataFrame(
            {
                "source_family": ["alpaca", "alpaca", "alpaca"],
                "training_eligible": ["true", "false", "true"],
                "missingness_known": ["true", "true", "false"],
                "coverage_state": ["observed_complete"] * 3,
                "status": ["observed"] * 3,
            }
        ),
    )

    readiness = _catalyst_readiness(catalyst=catalyst, config=config)
    alpaca = readiness.loc[
        readiness["evidence_type"].eq("source_family")
        & readiness["evidence_value"].eq("alpaca")
    ].iloc[0]

    assert alpaca["observed_count"] == 1
    business_exposure = readiness.loc[
        readiness["evidence_type"].eq("relation_channel")
        & readiness["evidence_value"].eq("business_exposure")
    ].iloc[0]
    assert business_exposure["research_status"] == "pass"
    assert business_exposure["promotion_status"] == "blocked"


def test_catalyst_source_rejects_collection_lineage_availability_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    news = tmp_path / "news"
    news.mkdir()
    news_manifest = news / "_manifest.json"
    news_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "request_sha256": "a" * 64,
                "availability_policy": "observed",
            }
        ),
        encoding="utf-8",
    )
    verified = SimpleNamespace(
        manifest={"source_event_rows": 1},
        request={"collection_manifest_sha256": file_sha256(news_manifest)},
        feature_inventory={"availability_policy": "provider_publication_proxy"},
        coverage=pd.DataFrame(),
        manifest_sha256="b" * 64,
        request_sha256="c" * 64,
        coverage_sha256="d" * 64,
    )
    monkeypatch.setattr(
        "market_predictor.governance.readiness.audit.catalyst_source_verification.verify_completed_catalyst_lineage",
        lambda _path: verified,
    )

    with pytest.raises(DataReadinessError, match="news source does not verify"):
        _verify_catalyst_sources(lineage_dir=tmp_path / "lineage", news_dir=news)


def test_projected_parquet_memory_is_checked_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "rows.parquet"
    pd.DataFrame({"value": list(range(1_000))}).to_parquet(artifact, index=False)
    config = load_prediction_data_readiness_config(Path("configs/prediction_data_readiness.toml"))
    threshold_bytes = int((config.maximum_process_memory_gib - config.memory_guard_headroom_gib) * 1024**3)
    monkeypatch.setattr(
        "market_predictor.governance.readiness.audit.process_memory_snapshot",
        lambda: (threshold_bytes - 1, threshold_bytes - 1),
    )

    with pytest.raises(DataReadinessError, match="projected resident data"):
        _assert_projected_parquet_load(
            [artifact],
            config=config,
            stage="test load",
        )


def test_dataframe_concat_memory_is_checked_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_prediction_data_readiness_config(Path("configs/prediction_data_readiness.toml"))
    threshold_bytes = int((config.maximum_process_memory_gib - config.memory_guard_headroom_gib) * 1024**3)
    frame = pd.DataFrame({"text": ["x" * 1024] * 100})
    retained = int(frame.memory_usage(index=True, deep=True).sum() * 1.5)
    monkeypatch.setattr(
        "market_predictor.governance.readiness.audit.process_memory_snapshot",
        lambda: (threshold_bytes - retained + 1, threshold_bytes - retained + 1),
    )

    with pytest.raises(DataReadinessError, match="projected concat"):
        _assert_projected_dataframe_concat(
            [frame],
            config=config,
            stage="test concat",
        )


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
