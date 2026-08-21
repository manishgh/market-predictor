from __future__ import annotations
import market_predictor.edge_rebuild.swing_artifact_contracts as swing_artifact_contracts

import json
import os
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild import swing_training
from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.edge_rebuild.swing_features import (
    MANAGED_EXCESS_RETURN_COLUMNS,
    MANAGED_PATH_NET_RETURN_COLUMNS,
    MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
    SWING_BASELINE_ABLATION_ORDER,
    SWING_FEATURE_PANEL_SCHEMA,
    SWING_FEATURE_PROFILE,
    swing_baseline_feature_columns,
    swing_model_feature_columns,
)
from market_predictor.edge_rebuild.swing_training import (
    SwingPanelBinding,
    SwingProfileData,
    SwingTrainingConfig,
    load_swing_candidate_authority,
    load_swing_training_config,
    train_swing_edge_candidate,
)
from market_predictor.edge_rebuild.temporal_manifest import (
    TemporalFold,
    TemporalSchedule,
    build_temporal_schedule,
    load_temporal_manifest_config,
)
from market_predictor.edge_rebuild.training import evaluation, walk_forward
from market_predictor.process_memory import process_memory_snapshot, release_process_memory
from market_predictor.v3.errors import DataReadinessError


def test_repository_policy_is_frozen_for_ten_session_candidate_training() -> None:
    config = load_swing_training_config(
        Path("configs/edge_rebuild_swing_training.toml")
    )
    contract = _contract()

    assert config.decision_start_date == "2019-07-09"
    assert config.horizon_sessions == 10
    assert config.maximum_process_memory_gib == 5.0
    assert config.probability_thresholds == (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
    # The grid must fit the contract's six-candidate experiment budget, and the
    # constructor's arithmetic must agree with what the builder actually emits.
    specs = swing_training._candidate_specs(config, contract)
    assert len(specs) == 6
    assert len(specs) <= config.maximum_learned_candidates
    assert [spec.feature_group for spec in specs[:4]] == list(
        SWING_BASELINE_ABLATION_ORDER
    )
    temporal = load_temporal_manifest_config(
        Path("configs/edge_rebuild_temporal_manifest.toml")
    )
    assert contract.validation.swing_walk_forward_folds == 1
    assert temporal.modeled_decision_start == date(2019, 7, 9)
    assert temporal.initial_fit_start == date(2019, 7, 9)
    assert temporal.initial_fit_end == date(2024, 5, 28)
    assert temporal.initial_fit_expected_sessions == 1_231
    assert temporal.validation_expected_sessions == 252
    assert temporal.validation_embargo_expected_sessions == 10
    assert temporal.final_refit_expected_sessions == 1_493
    assert temporal.final_embargo_expected_sessions == 10
    assert temporal.locked_test_expected_sessions == 251
    assert contract.validation.unseen_ticker_holdout_fraction == 0.20
    assert config.bootstrap_block_sessions == 20
    assert config.bootstrap_samples >= 2_000


def test_walk_forward_uses_exact_governed_session_counts_and_dates() -> None:
    temporal = load_temporal_manifest_config(
        Path("configs/edge_rebuild_temporal_manifest.toml")
    )
    schedule = build_temporal_schedule(temporal)
    folds = swing_training._governed_folds(schedule)

    assert len(schedule.final_refit_sessions) == 1_493
    assert len(schedule.locked_test_sessions) == 251
    assert schedule.locked_test_sessions[0] == date(2025, 7, 1)
    assert schedule.locked_test_sessions[-1] == date(2026, 6, 30)
    assert all(len(fold.train_sessions) == 1_231 for fold in folds)
    assert all(len(fold.validation_sessions) == 252 for fold in folds)
    assert all(len(fold.purge_sessions) == 0 for fold in folds)
    assert all(len(fold.embargo_sessions) == 10 for fold in folds)
    assert all(
        not set(fold.train_sessions).intersection(fold.validation_sessions)
        for fold in folds
    )
    assert folds[0].train_sessions[0] == "2019-07-09"
    assert folds[0].train_sessions[-1] == "2024-05-28"
    assert folds[0].validation_sessions[0] == "2024-06-12"
    assert folds[0].validation_sessions[-1] == "2025-06-13"
    assert schedule.final_refit_sessions[0] == date(2019, 7, 9)
    assert schedule.final_refit_sessions[-1] == date(2025, 6, 13)


def test_baseline_ablation_contract_is_nested_and_excludes_catalysts() -> None:
    contract = _contract()
    technical = swing_model_feature_columns(contract=contract, catalyst=False)
    groups = [
        swing_baseline_feature_columns(group, contract=contract)
        for group in SWING_BASELINE_ABLATION_ORDER
    ]

    assert all(
        set(left).issubset(right)
        for left, right in zip(groups[:-1], groups[1:], strict=True)
    )
    assert groups[-1] == technical
    assert all("alpaca" not in column for group in groups for column in group)
    assert not any(
        swing_training._is_unapproved_source_feature(column)
        for group in groups
        for column in group
    )


def test_input_authority_rejects_pre_cutoff_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    final = tmp_path / "panel" / "final"
    final.mkdir(parents=True)
    manifest_path = final / "_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    authority = {
        "schema": swing_artifact_contracts.SWING_MATERIALIZATION_AUTHORITY_SCHEMA,
        "state": "complete",
        "artifact_sha256": file_sha256(manifest_path),
    }
    (final / "_authority.json").write_text(
        json.dumps(authority, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": swing_training.SWING_MATERIALIZATION_MANIFEST_SCHEMA,
        "strategy_contract_sha256": contract.sha256(),
        "feature_profiles": list(swing_training.ALLOWED_PROFILES),
        "first_session": "2018-05-29",
        "rows": 1_000,
        "securities": 30,
        "request_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        "market_predictor.edge_rebuild.training.data_io.load_complete_swing_feature_panel", lambda root: manifest
    )

    with pytest.raises(DataReadinessError, match="start exactly on 2019-07-09"):
        swing_training.load_swing_panel_binding(
            tmp_path / "panel",
            strategy_contract=contract,
            config=_config(),
        )


def test_development_partition_selection_physically_excludes_locked_test_months() -> None:
    records = [
        {
            "path": "panel/feature_profile=technical_market/month=2025-06/part.parquet",
            "partition_month": "2025-06",
            "first_session": "2025-06-02",
            "last_session": "2025-06-30",
        },
        {
            "path": "panel/feature_profile=technical_market/month=2025-07/part.parquet",
            "partition_month": "2025-07",
            "first_session": "2025-07-01",
            "last_session": "2025-07-31",
        },
    ]
    selected = swing_training._partition_records_for_sessions(
        records,
        ("2025-06-12", "2025-06-30"),
    )

    assert [record["partition_month"] for record in selected] == ["2025-06"]


def test_profile_session_coverage_requires_every_governed_session() -> None:
    governed = (
        "2019-07-09",
        "2019-07-10",
        "2019-07-11",
        "2019-07-12",
        "2019-07-15",
    )

    swing_training._validate_profile_session_coverage(set(governed), governed)
    with pytest.raises(DataReadinessError, match="missing governed sessions"):
        swing_training._validate_profile_session_coverage(
            {"2019-07-10", "2019-07-11", "2019-07-12", "2019-07-15"},
            governed,
        )


def test_probability_distribution_is_complete_and_finite() -> None:
    result = swing_training._probability_distribution(
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype="float64")
    )

    assert result["minimum"] == pytest.approx(0.1)
    assert result["median"] == pytest.approx(0.25)
    assert result["maximum"] == pytest.approx(0.4)
    with pytest.raises(DataReadinessError, match="finite vector"):
        swing_training._probability_distribution(np.asarray([np.nan]))

def test_trains_sequential_ablations_and_publishes_candidate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    config = _config()
    technical = _profile(contract)
    binding = _binding(tmp_path)
    accesses = _patch_inputs(monkeypatch, binding, technical)

    result = train_swing_edge_candidate(
        tmp_path / "panel",
        tmp_path / "candidate",
        strategy_contract=contract,
        config=config,
        temporal_policy_path=tmp_path / "temporal.toml",
    )

    assert result.evaluation["status"] == "candidate_only"
    assert result.evaluation["promotion_permitted"] is False
    assert result.evaluation["selection_basis"] == "validation_only"
    assert result.evaluation["outcome_contract"]["benchmark_excess_columns"] == {
        "SPY": "future_excess_return_10d_vs_spy",
        "QQQ": "future_excess_return_10d_vs_qqq",
        "sector": "future_excess_return_10d_vs_sector",
    }
    assert result.evaluation["selection_policy"]["auc_used_for_selection"] is False
    assert result.evaluation["test_access_count"] == 1
    assert result.evaluation["split"]["random_or_row_split_used"] is False
    assert result.evaluation["split"]["purge_sessions"] == 0
    assert result.evaluation["split"]["embargo_sessions"] == 10
    assert result.evaluation["overlap_audit"]["row_identity_overlap_total"] == 0
    assert result.evaluation["overlap_audit"]["all_temporal_partitions_disjoint"]
    assert {
        record["ablation_profile"]
        for record in result.evaluation["validation_candidates"]
    } == {SWING_FEATURE_PROFILE}
    assert {
        record["feature_group"]
        for record in result.evaluation["validation_candidates"]
    } == set(SWING_BASELINE_ABLATION_ORDER)
    assert result.evaluation["feature_ablation_order"] == list(
        SWING_BASELINE_ABLATION_ORDER
    )
    for record in result.evaluation["validation_candidates"]:
        assert set(record["selected_validation_metrics"]) == {
            "temporal_generalization_full_pit_cross_section",
            "unseen_security_generalization_stable_20pct",
        }
    assert result.evaluation["locked_test_outcomes_read"] is True
    assert any(
        set(_test_schedule(technical.frame).locked_test_sessions)
        == {date.fromisoformat(value) for value in access}
        & set(_test_schedule(technical.frame).locked_test_sessions)
        for access in accesses
    )
    assert set(result.evaluation["final_test"]) == {
        "temporal_generalization_full_pit_cross_section",
        "unseen_security_generalization_stable_20pct",
    }
    metrics = next(iter(result.evaluation["final_test"][
        "temporal_generalization_full_pit_cross_section"
    ].values()))
    assert {
        "roc_auc",
        "pr_auc",
        "selected_probability_lift",
        "expected_calibration_error",
        "selected_average_managed_net_return",
        "selected_win_rate_after_costs",
        "turnover",
        "daily_mark_to_market_max_drawdown_after_costs",
        "by_regime",
        "by_sector",
        "calendar_average_managed_exit_session_close_spy_excess",
        "calendar_average_managed_exit_session_close_qqq_excess",
        "calendar_average_managed_exit_session_close_sector_excess",
        "portfolio_daily_average_return",
    }.issubset(metrics)
    assert metrics["cost_deduction_count"] == 1
    assert metrics["drawdown_has_daily_mark_to_market"] is True
    assert set(metrics["binary_outcome_diagnostics"]) == {
        "estimator_target_top_sector_quantile",
        "managed_net_return_positive_after_costs",
        "ten_session_net_return_positive_after_costs",
        "ten_session_spy_excess_positive",
        "ten_session_qqq_excess_positive",
        "ten_session_sector_excess_positive",
    }
    assert metrics["negative_controls"]["label_permutation"]["passed"] is True
    assert result.evaluation["selected_bundle_id"].startswith(
        "swing_baseline_bundle."
    )

    replay = load_swing_candidate_authority(tmp_path / "candidate")
    assert replay["status"] == "candidate"
    assert replay["candidate_id"] == result.selected_candidate_id
    with pytest.raises(FileExistsError, match="immutable output"):
        train_swing_edge_candidate(
            tmp_path / "panel",
            tmp_path / "candidate",
            strategy_contract=contract,
            config=config,
            temporal_policy_path=tmp_path / "temporal.toml",
        )


def test_candidate_authority_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    technical = _profile(contract)
    binding = _binding(tmp_path)
    _patch_inputs(monkeypatch, binding, technical)
    output = tmp_path / "candidate"
    train_swing_edge_candidate(
        tmp_path / "panel",
        output,
        strategy_contract=contract,
        config=_config(),
        temporal_policy_path=tmp_path / "temporal.toml",
    )
    with (output / "model_card.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(DataReadinessError, match="does not verify"):
        load_swing_candidate_authority(output)


def test_validation_selection_is_unchanged_when_only_final_test_is_poisoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    config = _config()
    technical = _profile(contract)
    binding = _binding(tmp_path)
    _patch_inputs(monkeypatch, binding, technical)
    baseline = train_swing_edge_candidate(
        tmp_path / "panel",
        tmp_path / "baseline",
        strategy_contract=contract,
        config=config,
        temporal_policy_path=tmp_path / "temporal.toml",
    )

    poisoned_technical = replace(technical, frame=_poison_final_test(technical.frame, config))
    _patch_inputs(monkeypatch, binding, poisoned_technical)
    poisoned = train_swing_edge_candidate(
        tmp_path / "panel",
        tmp_path / "poisoned",
        strategy_contract=contract,
        config=config,
        temporal_policy_path=tmp_path / "temporal.toml",
    )

    assert baseline.selected_candidate_id == poisoned.selected_candidate_id
    assert baseline.evaluation["validation_candidates"] == poisoned.evaluation["validation_candidates"]
    assert baseline.evaluation["final_test"] != poisoned.evaluation["final_test"]


def test_profile_validation_rejects_double_cost_and_late_membership() -> None:
    contract = _contract()
    config = _config()
    profile = _profile(contract)
    invalid_cost = profile.frame.copy()
    invalid_cost["barrier_net_return"] -= 0.002
    with pytest.raises(DataReadinessError, match="cost exactly once"):
        swing_training._validate_profile_frame(
            invalid_cost,
            profile=SWING_FEATURE_PROFILE,
            feature_columns=profile.feature_columns,
            strategy_contract=contract,
            config=config,
        )
    late = profile.frame.copy()
    late["membership_available_at_utc"] = late["decision_time_utc"] + pd.Timedelta(seconds=1)
    with pytest.raises(DataReadinessError, match="membership was unavailable"):
        swing_training._validate_profile_frame(
            late,
            profile=SWING_FEATURE_PROFILE,
            feature_columns=profile.feature_columns,
            strategy_contract=contract,
            config=config,
        )


def test_profile_validation_preserves_bounded_feature_missingness() -> None:
    contract = _contract()
    config = _config()
    profile = _profile(contract)
    feature = profile.feature_columns[0]
    partially_missing = profile.frame.copy()
    partially_missing.loc[partially_missing.index[0], feature] = np.nan

    validated = swing_training._validate_profile_frame(
        partially_missing,
        profile=SWING_FEATURE_PROFILE,
        feature_columns=profile.feature_columns,
        strategy_contract=contract,
        config=config,
    )

    assert validated[feature].isna().sum() == 1
    entirely_missing = profile.frame.copy()
    entirely_missing[feature] = np.nan
    with pytest.raises(DataReadinessError, match="entirely missing"):
        swing_training._validate_profile_frame(
            entirely_missing,
            profile=SWING_FEATURE_PROFILE,
            feature_columns=profile.feature_columns,
            strategy_contract=contract,
            config=config,
        )
    infinite = profile.frame.copy()
    infinite.loc[infinite.index[0], feature] = np.inf
    with pytest.raises(DataReadinessError, match="contains infinity"):
        swing_training._validate_profile_frame(
            infinite,
            profile=SWING_FEATURE_PROFILE,
            feature_columns=profile.feature_columns,
            strategy_contract=contract,
            config=config,
        )


def test_stable_security_holdout_is_deterministic_and_disjoint() -> None:
    contract = _contract()
    frame = pd.DataFrame({"security_id": [f"sec:{index:04d}" for index in range(1_000)]})

    first = evaluation._security_holdout_mask(frame, contract)
    shuffled = frame.sample(frac=1.0, random_state=7)
    second = evaluation._security_holdout_mask(shuffled, contract)
    first_ids = set(frame.loc[first, "security_id"])
    second_ids = set(shuffled.loc[second, "security_id"])

    assert first_ids == second_ids
    assert 0.17 <= len(first_ids) / 1_000 <= 0.23
    assert first_ids.isdisjoint(set(frame.loc[~first, "security_id"]))


def test_constrained_selection_enforces_trade_and_sector_caps() -> None:
    rows = []
    sectors = ["A", "A", "A", "A", "A", "B", "C", "D", "E", "F"]
    for index, sector in enumerate(sectors):
        rows.append({
            "decision_group_id": "2024-01-02",
            "decision_time_utc": pd.Timestamp("2024-01-02T21:00:00Z"),
            "security_id": f"sec:{index}",
            "sector": sector,
            "__probability": 1.0 - index / 100.0,
        })

    selected = swing_training.select_constrained_swing_portfolio(
        pd.DataFrame(rows),
        maximum_trades=5,
        target_maximum_sector_weight=0.20,
        hard_maximum_sector_weight=1.0 / 3.0,
        minimum_distinct_sectors=3,
    )

    assert len(selected) == 5
    assert selected.groupby("sector").size().max() / len(selected) <= 0.20


@pytest.mark.parametrize(
    ("sectors", "expected_weight"),
    ((["A", "B", "C", "D"], 0.25), (["A", "B", "C"], 1.0 / 3.0)),
)
def test_constrained_selection_uses_bounded_sector_fallback(
    sectors: list[str],
    expected_weight: float,
) -> None:
    rows = [
        {
            "decision_group_id": "2024-01-02",
            "decision_time_utc": pd.Timestamp("2024-01-02T21:00:00Z"),
            "security_id": f"sec:{index}",
            "sector": sector,
            "__probability": 1.0 - index / 100.0,
        }
        for index, sector in enumerate(sectors)
    ]

    selected = swing_training.select_constrained_swing_portfolio(
        pd.DataFrame(rows),
        maximum_trades=25,
        target_maximum_sector_weight=0.20,
        hard_maximum_sector_weight=1.0 / 3.0,
        minimum_distinct_sectors=3,
    )

    assert len(selected) == len(sectors)
    assert selected["__effective_sector_weight_limit"].iloc[0] == pytest.approx(
        expected_weight
    )


def test_constrained_selection_rejects_fewer_than_three_sectors() -> None:
    frame = pd.DataFrame(
        {
            "decision_group_id": ["2024-01-02"] * 2,
            "decision_time_utc": [pd.Timestamp("2024-01-02T21:00:00Z")] * 2,
            "security_id": ["sec:1", "sec:2"],
            "sector": ["A", "B"],
            "__probability": [0.8, 0.7],
        }
    )

    selected = swing_training.select_constrained_swing_portfolio(
        frame,
        maximum_trades=25,
        target_maximum_sector_weight=0.20,
        hard_maximum_sector_weight=1.0 / 3.0,
        minimum_distinct_sectors=3,
    )

    assert selected.empty


def test_moving_block_bootstrap_is_deterministic_and_uses_frozen_block() -> None:
    values = np.sin(np.arange(200, dtype="float64") / 10.0) / 100.0

    first = swing_training._moving_block_bootstrap_mean_interval(values, 2_000, 20, 42)
    second = swing_training._moving_block_bootstrap_mean_interval(values, 2_000, 20, 42)

    assert first == second
    assert first["block_sessions"] == 20
    assert first["bootstrap_samples"] == 2_000


def test_session_economic_calendar_includes_zero_return_no_position_sessions() -> None:
    selected = pd.DataFrame(
        {
            "session_date_et": ["2024-01-03"],
            "barrier_net_return": [0.02],
            "approx_managed_exit_session_close_excess_vs_spy": [0.01],
            "approx_managed_exit_session_close_excess_vs_qqq": [0.008],
            "approx_managed_exit_session_close_excess_vs_sector": [0.012],
        }
    )

    blocks = swing_training._session_economic_blocks(
        selected,
        session_calendar=("2024-01-02", "2024-01-03", "2024-01-04"),
    )

    assert [record["barrier_net_return"] for record in blocks] == [0.0, 0.02, 0.0]


def test_economic_gate_uses_holding_aligned_benchmarks_and_portfolio_path() -> None:
    interval = {"estimate": 0.01, "low": 0.005, "high": 0.015}
    metrics = {
        "selected_average_managed_net_return": 0.01,
        "maximum_observed_sector_weight": 0.25,
        "double_cost_portfolio_daily_average_return": 0.005,
        "moving_block_bootstrap_95_ci": {
            "calendar_average_managed_net_return": interval,
            "portfolio_daily_return": interval,
            "double_cost_portfolio_daily_return": interval,
            "calendar_average_managed_exit_session_close_spy_excess": interval,
            "calendar_average_managed_exit_session_close_qqq_excess": interval,
            "calendar_average_managed_exit_session_close_sector_excess": interval,
        },
    }

    gate = swing_training._economic_gate(metrics, _contract())

    assert gate["passed"] is True
    assert "worst_holding_aligned_benchmark_ci_low_positive" in gate["checks"]
    assert "portfolio_daily_return_ci_low_positive" in gate["checks"]


def test_validation_threshold_requires_every_scope_economic_gate() -> None:
    assert not swing_training._validation_scopes_pass_economic_gates({})
    assert not swing_training._validation_scopes_pass_economic_gates(
        {
            "temporal": {"economic_gate": {"passed": True}},
            "unseen_security": {"economic_gate": {"passed": False}},
        }
    )
    assert swing_training._validation_scopes_pass_economic_gates(
        {
            "temporal": {"economic_gate": {"passed": True}},
            "unseen_security": {"economic_gate": {"passed": True}},
        }
    )


def test_no_candidate_evidence_is_immutable_and_does_not_open_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    technical = _profile(contract)
    frame = technical.frame.copy()
    frame["barrier_gross_return"] = -0.01
    frame["barrier_net_return"] = -0.012
    for column in MANAGED_EXCESS_RETURN_COLUMNS:
        frame[column] = -0.013
    for column in MANAGED_PATH_NET_RETURN_COLUMNS:
        frame[column] = -0.012
    failed_profile = replace(technical, frame=frame)
    binding = _binding(tmp_path)
    accesses = _patch_inputs(
        monkeypatch,
        binding,
        failed_profile,
    )

    result = train_swing_edge_candidate(
        tmp_path / "panel",
        tmp_path / "no-candidate",
        strategy_contract=contract,
        config=_config(),
        temporal_policy_path=tmp_path / "temporal.toml",
    )
    replay = load_swing_candidate_authority(tmp_path / "no-candidate")

    assert result.selected_candidate_id is None
    assert result.evaluation["test_access_count"] == 0
    assert result.evaluation["locked_test_outcomes_read"] is False
    locked = {
        value.isoformat()
        for value in _test_schedule(failed_profile.frame).locked_test_sessions
    }
    assert all(not locked.intersection(access) for access in accesses)
    assert replay["status"] == "no_candidate"
    assert not (tmp_path / "no-candidate" / "candidate.joblib").exists()


def test_production_technical_profile_memory_projection_stays_below_budget() -> None:
    one_profile = swing_training._projected_profile_memory_bytes(853_417, 138)
    safety_threshold = int(3.25 * 1024**3)

    assert one_profile < safety_threshold


@pytest.mark.skipif(
    os.environ.get("MARKET_PREDICTOR_RUN_MEMORY_STRESS") != "1",
    reason="explicit production-shape memory stress",
)
def test_realistic_single_profile_feature_matrix_stays_below_memory_budget() -> None:
    rows = 853_417
    features = 138
    matrix = np.empty((rows, features), dtype="float32")
    matrix.fill(0.125)
    holdout = np.arange(rows) % 5 == 0
    training = matrix[~holdout]
    validation = matrix[holdout]
    probabilities = np.empty(len(validation), dtype="float64")
    probabilities.fill(0.5)
    snapshot = process_memory_snapshot()
    try:
        assert snapshot is not None
        assert snapshot[0] < int(3.25 * 1024**3)
        print(
            f"production_shape_rss_bytes={snapshot[0]} "
            f"production_shape_peak_rss_bytes={snapshot[1]}"
        )
        assert training.shape == (rows - int(holdout.sum()), features)
        assert validation.shape == (int(holdout.sum()), features)
        assert len(probabilities) == len(validation)
    finally:
        del probabilities, validation, training, holdout, matrix
        release_process_memory()

def _config() -> SwingTrainingConfig:
    return SwingTrainingConfig(
        calibration_fraction=0.20,
        minimum_calibration_sessions=20,
        minimum_rows=1,
        minimum_securities=20,
        maximum_trades_per_decision=25,
        probability_thresholds=(0.10, 0.20),
        logistic_c_values=(1.0,),
        xgb_learning_rates=(0.05,),
        xgb_n_estimators=20,
        maximum_learned_candidates=6,
        bootstrap_samples=2_000,
        bootstrap_block_sessions=20,
    )


def _contract() -> StrategyContract:
    return load_strategy_contract(Path("configs/edge_rebuild_strategy_contract.toml"))


def _profile(contract: StrategyContract) -> SwingProfileData:
    sessions = pd.bdate_range("2019-07-09", periods=430, tz="UTC")
    securities = [f"SEC-{index:03d}" for index in range(30)]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(42)
    for session_index, session in enumerate(sessions[:420]):
        decision = session + pd.Timedelta(hours=20)
        label_time = sessions[session_index + 10] + pd.Timedelta(hours=21)
        barrier_exit = sessions[session_index + 5]
        for security_index, security in enumerate(securities):
            positive = security_index in {0, 1, 2, 3, 4, 5, 8, 14, 15, 22}
            rank_label = 1 if positive else (-1 if security_index >= 24 else 0)
            signal = 1.5 if rank_label == 1 else (-1.0 if rank_label == -1 else 0.0)
            gross = 0.012 * signal + rng.normal(0.0, 0.004)
            fixed_gross = gross + rng.normal(0.0, 0.002)
            rows.append(
                {
                    "decision_id": f"{session.date()}|{security}",
                    "decision_group_id": decision.isoformat(),
                    "ticker": f"T{security_index:03d}",
                    "security_id": security,
                    "sector": f"sector-{security_index}",
                    "primary_benchmark": f"XL{security_index}",
                    "market_regime": "risk_on" if session_index % 3 else "risk_off",
                    "session_date_et": session.date().isoformat(),
                    "decision_time_utc": decision,
                    "feature_available_at_utc": decision,
                    "label_available_at_utc": label_time,
                    "membership_effective_from_utc": pd.Timestamp("2010-01-01", tz="UTC"),
                    "membership_effective_to_utc": pd.NaT,
                    "membership_available_at_utc": pd.Timestamp("2010-01-01", tz="UTC"),
                    "entry_time_utc": decision + pd.Timedelta(hours=17, minutes=30),
                    "barrier_exit_session_date_et": barrier_exit.date().isoformat(),
                    "barrier_label_available_at_utc": barrier_exit + pd.Timedelta(hours=21),
                    "horizon_sessions": 10,
                    "feature_eligible": True,
                    "label_eligible": True,
                    "cross_section_eligible": True,
                    "sector_peer_count": 30,
                    "sector_rank_eligible": True,
                    "sector_rank_target_met": False,
                    "barrier_label": 1 if gross > 0.004 else (-1 if gross < -0.004 else 0),
                    "rank_label": rank_label,
                    "ranking_group_size": 30,
                    "ranking_reliability_weight": 0.6,
                    "target": 1 if rank_label == 1 else 0,
                    "barrier_holding_sessions": 5,
                    "barrier_gross_return": gross,
                    "barrier_cost": 0.002,
                    "barrier_net_return": gross - 0.002,
                    "relevance_score": 1.0,
                    "future_gross_return_10d": fixed_gross,
                    "future_net_return_10d": fixed_gross - 0.002,
                    "future_spy_return_10d": 0.001,
                    "future_qqq_return_10d": 0.0015,
                    "future_sector_return_10d": 0.0005,
                    "future_excess_return_10d_vs_spy": fixed_gross - 0.003,
                    "future_excess_return_10d_vs_qqq": fixed_gross - 0.0035,
                    "future_excess_return_10d_vs_sector": fixed_gross - 0.0025,
                    "managed_path_eligible": True,
                    "approx_managed_exit_session_close_spy_return": 0.001,
                    "approx_managed_exit_session_close_qqq_return": 0.0015,
                    "approx_managed_exit_session_close_sector_return": 0.0005,
                    "approx_managed_exit_session_close_excess_vs_spy": gross - 0.003,
                    "approx_managed_exit_session_close_excess_vs_qqq": gross - 0.0035,
                    "approx_managed_exit_session_close_excess_vs_sector": gross - 0.0025,
                    "swing_feature_panel_schema": SWING_FEATURE_PANEL_SCHEMA,
                    "strategy_contract_sha256": contract.sha256(),
                    "feature_signal": signal + rng.normal(0.0, 0.3),
                    "feature_trend": signal + rng.normal(0.0, 0.5),
                    "feature_pullback": signal + rng.normal(0.0, 0.7),
                    "feature_volume": signal + rng.normal(0.0, 0.9),
                    "feature_noise": rng.normal(),
                }
            )
            for offset, (ordinal_column, path_column) in enumerate(
                zip(
                    MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
                    MANAGED_PATH_NET_RETURN_COLUMNS,
                    strict=True,
                ),
                start=1,
            ):
                rows[-1][ordinal_column] = sessions[
                    session_index + offset
                ].date().toordinal()
                rows[-1][path_column] = (
                    gross - 0.002
                    if offset >= 5
                    else (gross * offset / 5.0 - 0.002)
                )
    frame = pd.DataFrame(rows)
    digest = swing_training._sequence_sha256(frame["decision_id"].astype(str))
    binding = SwingPanelBinding(
        root=Path("unused"),
        manifest={},
        manifest_sha256="1" * 64,
        authority_sha256="2" * 64,
        request_sha256="3" * 64,
        strategy_contract_sha256=contract.sha256(),
    )
    technical = SwingProfileData(
        frame=frame.copy(),
        profile=SWING_FEATURE_PROFILE,
        feature_columns=(
            "feature_signal",
            "feature_trend",
            "feature_pullback",
            "feature_volume",
            "feature_noise",
        ),
        decision_ids_sha256=digest,
        panel=binding,
    )
    return technical


def _binding(tmp_path: Path) -> SwingPanelBinding:
    return SwingPanelBinding(
        root=tmp_path / "panel",
        manifest={},
        manifest_sha256="a" * 64,
        authority_sha256="b" * 64,
        request_sha256="c" * 64,
        strategy_contract_sha256=_contract().sha256(),
    )


def _patch_inputs(
    monkeypatch: pytest.MonkeyPatch,
    binding: SwingPanelBinding,
    technical: SwingProfileData,
) -> list[set[str]]:
    policy = binding.root.parent / "temporal.toml"
    policy.write_text("test temporal policy\n", encoding="utf-8")
    schedule = _test_schedule(technical.frame)
    temporal = SimpleNamespace(
        schema_version="edge_rebuild.temporal_manifest.v2",
        modeled_decision_start=date(2019, 7, 9),
        validation_embargo_expected_sessions=10,
        final_embargo_expected_sessions=10,
        label_horizon_sessions=10,
        unseen_security_holdout_fraction=0.20,
        unseen_security_hash_seed=42,
        unseen_security_assignment="sha256_threshold_security_id_v1",
        sha256=lambda: "d" * 64,
    )
    monkeypatch.setattr(
        swing_training,
        "load_temporal_manifest_config",
        lambda path: temporal,
    )
    monkeypatch.setattr(
        swing_training,
        "build_temporal_schedule",
        lambda config: schedule,
    )
    monkeypatch.setattr(swing_training, "load_swing_panel_binding", lambda *args, **kwargs: binding)
    profiles = {SWING_FEATURE_PROFILE: technical}
    accesses: list[set[str]] = []

    def load_profile(
        binding: SwingPanelBinding,
        profile: str,
        *,
        sessions: tuple[str, ...],
        **kwargs: object,
    ) -> SwingProfileData:
        accesses.append(set(sessions))
        source = profiles[profile]
        frame = source.frame.loc[source.frame["session_date_et"].isin(sessions)].copy()
        return replace(
            source,
            frame=frame,
            decision_ids_sha256=swing_training._sequence_sha256(
                frame["decision_id"].astype(str)
            ),
        )

    monkeypatch.setattr(
        swing_training,
        "load_swing_profile",
        load_profile,
    )
    monkeypatch.setattr(
        swing_training,
        "_candidate_specs",
        _test_candidate_specs,
    )
    return accesses


def _test_candidate_specs(
    config: SwingTrainingConfig,
    strategy_contract: StrategyContract,
) -> tuple[swing_training.CandidateSpec, ...]:
    del strategy_contract
    nested = {
        "momentum_volatility": ("feature_signal",),
        "trend_confirmation": ("feature_signal", "feature_trend"),
        "pullback_timing": (
            "feature_signal",
            "feature_trend",
            "feature_pullback",
        ),
        "volume_liquidity": (
            "feature_signal",
            "feature_trend",
            "feature_pullback",
            "feature_volume",
            "feature_noise",
        ),
    }
    specs = [
        swing_training.CandidateSpec(
            candidate_id=f"test.{group}.logistic",
            profile=SWING_FEATURE_PROFILE,
            feature_group=group,
            feature_columns=columns,
            estimator_family="logistic",
            hyperparameters={"C": config.logistic_c_values[0], "solver": "lbfgs"},
        )
        for group, columns in nested.items()
    ]
    for family in ("xgboost_ranker", "xgboost_regressor"):
        specs.append(
            swing_training.CandidateSpec(
                candidate_id=f"test.volume_liquidity.{family}",
                profile=SWING_FEATURE_PROFILE,
                feature_group="volume_liquidity",
                feature_columns=nested["volume_liquidity"],
                estimator_family=family,
                hyperparameters={
                    "learning_rate": config.xgb_learning_rates[0],
                    "max_depth": config.xgb_max_depths[0],
                    "n_estimators": config.xgb_n_estimators,
                    "threads": 1,
                },
            )
        )
    return tuple(specs)


def _poison_final_test(frame: pd.DataFrame, config: SwingTrainingConfig) -> pd.DataFrame:
    poisoned = frame.copy()
    sessions = walk_forward._ordered_sessions(poisoned)
    test = sessions[-60:]
    selected = poisoned["session_date_et"].isin(test)
    poisoned.loc[selected, "barrier_gross_return"] *= -1
    poisoned.loc[selected, "barrier_net_return"] = (
        poisoned.loc[selected, "barrier_gross_return"] - 0.002
    )
    for column in (
        "future_excess_return_10d_vs_spy",
        "future_excess_return_10d_vs_qqq",
        "future_excess_return_10d_vs_sector",
    ):
        poisoned.loc[selected, column] *= -1
    shifted_noise = poisoned.loc[selected].groupby(
        "security_id", sort=False, observed=True
    )["feature_noise"].shift(1)
    poisoned.loc[selected, "feature_noise"] = shifted_noise.fillna(
        poisoned.loc[selected, "feature_noise"]
    )
    return poisoned


def _test_schedule(frame: pd.DataFrame) -> TemporalSchedule:
    sessions = tuple(
        date.fromisoformat(value) for value in walk_forward._ordered_sessions(frame)
    )
    return TemporalSchedule(
        target_sessions=sessions,
        warmup_sessions=(),
        folds=(
            TemporalFold(
                fold=1,
                train_sessions=sessions[:252],
                embargo_sessions=sessions[252:262],
                validation_sessions=sessions[262:322],
            ),
        ),
        final_refit_sessions=sessions[98:350],
        final_embargo_sessions=sessions[350:360],
        locked_test_sessions=sessions[360:420],
    )
