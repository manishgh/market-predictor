from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from market_predictor.edge_rebuild import intraday_development
from market_predictor.edge_rebuild.intraday_development import (
    IntradayDevelopmentConfig,
    baseline_profile,
    evaluate_future_intraday_holdout,
    load_intraday_development_config,
    train_intraday_development_candidate,
)
from market_predictor.edge_rebuild.intraday_training import MODEL_FEATURE_COLUMNS
from market_predictor.v3.errors import DataReadinessError
from tests.test_edge_rebuild_intraday_training import _publish_dataset, _training_frame


@pytest.fixture(scope="module")
def a43_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("a44-a43-authority")
    return _publish_dataset(
        root / "dataset",
        _training_frame(session_count=50, security_count=40),
    )


def test_repository_policy_freezes_complete_a44_contract() -> None:
    config = load_intraday_development_config(
        Path("configs/edge_rebuild_intraday_development.toml")
    )

    assert config.development_end_date == "2026-07-08"
    assert config.future_holdout_start_date == "2026-07-09"
    assert config.validation_folds == 4
    assert config.security_holdout_fraction == 0.20
    assert config.cost_curve_bps == (0.0, 5.0, 10.0, 20.0)
    assert config.stress_cost_bps == 20.0
    assert config.continuation_min_volume_return_1_bar == 0.0
    assert config.reversion_max_vwap_distance_atr == -0.5
    assert config.reversion_max_volume_rsi_14 == 45.0
    assert config.maximum_process_memory_gib == 4.0


def test_training_requires_an_explicit_supported_hypothesis(
    a43_dataset: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="hypothesis"):
        train_intraday_development_candidate(  # type: ignore[call-arg]
            a43_dataset,
            tmp_path / "missing-hypothesis",
            config=_rejecting_config(),
        )

    with pytest.raises(ValueError, match="continuation.*long-reversion"):
        train_intraday_development_candidate(
            a43_dataset,
            tmp_path / "unsupported-hypothesis",
            hypothesis="reversion",
            config=_rejecting_config(),
        )


def test_profile_identity_and_population_are_order_independent() -> None:
    config = _rejecting_config()
    frame = _training_frame(session_count=10, security_count=12)
    shuffled = frame.sample(frac=1.0, random_state=91).reset_index(drop=True)

    continuation = baseline_profile("continuation", config)
    reversion = baseline_profile("long-reversion", config)
    continuation_ids = set(
        frame.loc[
            intraday_development._profile_mask(frame, continuation),
            "decision_id",
        ]
    )
    shuffled_continuation_ids = set(
        shuffled.loc[
            intraday_development._profile_mask(shuffled, continuation),
            "decision_id",
        ]
    )
    reversion_ids = set(
        frame.loc[
            intraday_development._profile_mask(frame, reversion),
            "decision_id",
        ]
    )

    assert continuation.profile_id == "intraday_bar_continuation_long_v1"
    assert continuation.population_rule == {
        "volume_return_1_bar_gt": 0.0,
        "stock_return_20m_gt": 0.0,
        "session_vwap_distance_five_minute_atr_gte": 0.0,
    }
    assert reversion.profile_id == "intraday_bar_long_reversion_v1"
    assert reversion.population_rule == {
        "stock_return_20m_lt": 0.0,
        "session_vwap_distance_five_minute_atr_lte": -0.5,
        "volume_rsi_14_lte": 45.0,
    }
    assert continuation.sha256() != reversion.sha256()
    assert continuation_ids == shuffled_continuation_ids
    assert continuation_ids.isdisjoint(reversion_ids)


def test_fourth_walk_forward_fold_is_development_confirmation(
    a43_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _rejecting_config()
    data = _profile_frame(a43_dataset, config, "continuation")
    sessions = intraday_development._ordered_sessions(data)
    folds = intraday_development._walk_forward_folds(data, sessions, config)
    features = data.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype="float32")
    opportunity = data["net_return"].to_numpy(dtype="float64")
    downside = data["stop_hit"].to_numpy(dtype="int8")
    holdout = intraday_development._stable_security_holdout(
        data, config.security_holdout_fraction
    )

    def fake_fit_pair(
        _spec: object,
        frame: pd.DataFrame,
        _features: np.ndarray,
        _opportunity: np.ndarray,
        _downside: np.ndarray,
        training_sessions: tuple[str, ...],
        policy: IntradayDevelopmentConfig,
        **_kwargs: object,
    ) -> SimpleNamespace:
        fit, calibration = intraday_development._split_downside_calibration(
            frame, training_sessions, policy
        )
        return SimpleNamespace(
            fit_sessions=fit,
            calibration_sessions=calibration,
        )

    monkeypatch.setattr(intraday_development, "_fit_pair", fake_fit_pair)
    monkeypatch.setattr(
        intraday_development,
        "_predict_pair",
        lambda _fitted, matrix: (
            np.linspace(-0.001, 0.003, len(matrix), dtype="float64"),
            np.linspace(0.1, 0.4, len(matrix), dtype="float64"),
        ),
    )

    scored, records = intraday_development._walk_forward_predictions(
        intraday_development._candidate_specs(config)[0],
        data,
        features,
        opportunity,
        downside,
        folds,
        holdout,
        config,
    )

    assert len(folds) == 4
    assert len(records) == 4
    assert [record["role"] for record in records] == [
        "selection",
        "selection",
        "selection",
        "development_confirmation",
    ]
    assert set(scored["fold"]) == {0, 1, 2, 3}
    assert set(scored["validation_scope"]) == {
        "seen_security",
        "unseen_security",
    }


def test_downside_fit_and_calibration_are_chronological_purged_partitions(
    a43_dataset: Path,
) -> None:
    config = _rejecting_config()
    data = _profile_frame(a43_dataset, config, "long-reversion")
    sessions = intraday_development._ordered_sessions(data)
    training_sessions = sessions[: config.minimum_train_sessions]

    fit, calibration = intraday_development._split_downside_calibration(
        data, training_sessions, config
    )

    assert set(fit).isdisjoint(calibration)
    assert max(fit) < min(calibration)
    assert len(set(training_sessions) - set(fit) - set(calibration)) >= 1
    assert (
        data.loc[data["session_date_et"].isin(fit), "label_available_at_utc"].max()
        < data.loc[
            data["session_date_et"].isin(calibration), "decision_time_utc"
        ].min()
    )


def test_security_holdout_is_stable_under_row_and_security_order() -> None:
    config = _rejecting_config()
    frame = _training_frame(session_count=8, security_count=40)
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    first = intraday_development._stable_security_holdout(
        frame, config.security_holdout_fraction
    )
    second = intraday_development._stable_security_holdout(
        shuffled, config.security_holdout_fraction
    )

    assert first == second
    assert intraday_development._security_set_sha256(
        first
    ) == intraday_development._security_set_sha256(second)
    assert 0 < len(first) < frame["security_id"].nunique()


def test_final_candidate_fit_excludes_the_frozen_security_holdout(
    a43_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _rejecting_config()
    selected_spec = _limit_to_one_candidate(monkeypatch, config)
    profiled = _profile_frame(a43_dataset, config, "continuation")
    expected_holdout = intraday_development._stable_security_holdout(
        profiled, config.security_holdout_fraction
    )
    observed_exclusions: list[frozenset[str]] = []

    def recording_fit_pair(
        _spec: object,
        data: pd.DataFrame,
        _features: np.ndarray,
        _opportunity: np.ndarray,
        _downside: np.ndarray,
        sessions: tuple[str, ...],
        policy: IntradayDevelopmentConfig,
        *,
        excluded_securities: frozenset[str] = frozenset(),
    ) -> Any:
        observed_exclusions.append(excluded_securities)
        fit, calibration = intraday_development._split_downside_calibration(
            data, sessions, policy
        )
        return intraday_development._FittedPair(
            opportunity_estimator=_OpportunityEstimator(),
            downside_estimator=_DownsideEstimator(),
            downside_calibrator=_DownsideCalibrator(),
            fit_sessions=fit,
            calibration_sessions=calibration,
        )

    monkeypatch.setattr(intraday_development, "_fit_pair", recording_fit_pair)
    monkeypatch.setattr(
        intraday_development,
        "_evaluate_spec",
        lambda _spec, _scored, folds, _config, _cost: _passing_candidate_record(
            selected_spec, folds
        ),
    )

    result = train_intraday_development_candidate(
        a43_dataset,
        tmp_path / "holdout-excluded-candidate",
        hypothesis="continuation",
        config=config,
    )

    assert result.status == "candidate"
    assert len(observed_exclusions) == config.validation_folds + 1
    assert observed_exclusions[-1] == expected_holdout
    assert all(excluded == expected_holdout for excluded in observed_exclusions)


def test_paired_fit_produces_finite_row_aligned_opportunity_and_downside_scores(
    a43_dataset: Path,
) -> None:
    config = _rejecting_config()
    data = _profile_frame(a43_dataset, config, "continuation")
    sessions = intraday_development._ordered_sessions(data)
    features = data.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype="float32")
    opportunity = data["net_return"].to_numpy(dtype="float64")
    downside = data["stop_hit"].to_numpy(dtype="int8")
    holdout = intraday_development._stable_security_holdout(
        data, config.security_holdout_fraction
    )
    fitted = intraday_development._fit_pair(
        intraday_development._candidate_specs(config)[0],
        data,
        features,
        opportunity,
        downside,
        sessions[: config.minimum_train_sessions],
        config,
        excluded_securities=holdout,
    )

    opportunity_score, stop_probability = intraday_development._predict_pair(
        fitted, features[:25]
    )

    assert fitted.opportunity_estimator is not fitted.downside_estimator
    assert len(opportunity_score) == len(stop_probability) == 25
    assert np.isfinite(opportunity_score).all()
    assert np.isfinite(stop_probability).all()
    assert np.logical_and(stop_probability >= 0.0, stop_probability <= 1.0).all()
    assert set(fitted.fit_sessions).isdisjoint(fitted.calibration_sessions)


def test_worse_seen_or_unseen_scope_blocks_candidate_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _rejecting_config(expected_net_return_thresholds_bps=(0.0,))
    scored = _scored_policy_frame()
    passing = _passing_scopes()
    failing = _passing_scopes()
    failing["unseen_security"] = {
        "passed": False,
        "failed_gate_reasons": ["stop_hit_ece_above_gate"],
        "metrics": _passing_scope_metrics(),
    }
    responses = iter((passing, failing))
    monkeypatch.setattr(
        intraday_development,
        "_evaluate_scopes",
        lambda *_args, **_kwargs: next(responses),
    )

    result = intraday_development._evaluate_spec(
        intraday_development._candidate_specs(config)[0],
        scored,
        [{"fold": fold} for fold in range(4)],
        config,
        10.0,
    )

    assert result["validation_passed"] is False
    assert result["confirmation_policy_frozen_before_scoring"] is True
    assert result["failed_gate_reasons"] == [
        "unseen_security:stop_hit_ece_above_gate"
    ]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"rows": 99}, "insufficient_scope_rows"),
        (
            {"positive_net_return_roc_auc": 0.50},
            "positive_net_return_roc_auc_below_gate",
        ),
        ({"stop_hit_brier": 0.90}, "stop_hit_brier_above_gate"),
        (
            {"average_trade_net_return": -0.02},
            "average_trade_net_return_below_gate",
        ),
        ({"maximum_drawdown": 0.90}, "drawdown_above_gate"),
        ({"average_daily_round_trip_turnover": 2.0}, "turnover_above_gate"),
        ({"profitable_fold_fraction": 0.0}, "fold_stability_below_gate"),
    ],
)
def test_every_scope_gate_is_all_or_nothing(
    mutation: dict[str, object],
    reason: str,
) -> None:
    config = _gate_config()
    metrics = _passing_scope_metrics()
    metrics.update(mutation)

    passed, reasons = intraday_development._scope_gates(
        metrics, config, scope="seen_security"
    )

    assert passed is False
    assert reason in reasons


def test_no_candidate_publishes_evidence_without_model_or_future_access(
    a43_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _rejecting_config()
    _limit_to_one_candidate(monkeypatch, config)
    _patch_fast_pair(monkeypatch)
    output = tmp_path / "no-candidate"

    result = train_intraday_development_candidate(
        a43_dataset,
        output,
        hypothesis="continuation",
        config=config,
    )

    evaluation = _json(output / "evaluation.json")
    assert result.status == "no_candidate"
    assert result.selected_candidate_id is None
    assert not (output / "candidate.joblib").exists()
    assert (output / "validation_predictions.parquet").is_file()
    assert evaluation["future_holdout_opened"] is False
    assert evaluation["test_access_count"] == 0
    assert evaluation["memory"]["hard_budget_gib"] == 4.0
    assert evaluation["memory"]["safety_threshold_gib"] == 3.25
    assert evaluation["auditable_policy_ledger"]["selection_status"] == (
        "best_failed_diagnostic_only"
    )
    replayed = intraday_development.load_complete_intraday_development_output(output)
    assert replayed["state"] == "no_candidate"

    tampered = tmp_path / "tampered-output"
    shutil.copytree(output, tampered)
    tampered_evaluation = _json(tampered / "evaluation.json")
    tampered_evaluation["future_holdout_opened"] = True
    (tampered / "evaluation.json").write_text(
        json.dumps(tampered_evaluation, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(DataReadinessError, match="identity"):
        intraday_development.load_complete_intraday_development_output(tampered)

    extra_file = tmp_path / "extra-file-output"
    shutil.copytree(output, extra_file)
    (extra_file / "unexpected.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="immutable file set"):
        intraday_development.load_complete_intraday_development_output(extra_file)

    with pytest.raises(DataReadinessError, match="locked until validation"):
        evaluate_future_intraday_holdout(
            output,
            tmp_path / "must-not-be-opened",
            tmp_path / "future-output",
        )
    assert not (tmp_path / ".no-candidate.future-holdout-access.json").exists()


@pytest.mark.parametrize(
    ("event_subtype", "expected_family"),
    [
        (None, "intraday_event_confirmed_research"),
        ("bare_upgrade", "intraday_bare_upgrade_confirmed_research"),
    ],
)
def test_event_confirmed_training_binds_research_cohort_and_remains_non_promotable(
    a43_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event_subtype: str | None,
    expected_family: str,
) -> None:
    config = _rejecting_config()
    _limit_to_one_candidate(monkeypatch, config)
    _patch_fast_pair(monkeypatch)
    decision_ids = frozenset(
        intraday_development.load_published_intraday_dataset(a43_dataset)
        .frame["decision_id"]
        .astype(str)
    )
    event_directory = tmp_path / "event-preflight"
    event_directory.mkdir()
    cohort_identity = {
        "schema": "edge_rebuild.intraday_research_event_cohort.v1",
        "production_eligible": False,
        "serving_eligible": False,
        "future_holdout_opened": False,
        "catalyst_role": "confirmation_and_population_filter_not_model_feature",
        "event_subtype": event_subtype,
    }
    monkeypatch.setattr(
        intraday_development,
        "load_intraday_research_event_cohort",
        lambda _directory, **_kwargs: SimpleNamespace(
            decision_ids=decision_ids,
            identity=cohort_identity,
        ),
    )
    output = tmp_path / f"event-confirmed-{event_subtype or 'all'}"

    result = train_intraday_development_candidate(
        a43_dataset,
        output,
        hypothesis="continuation",
        config=config,
        research_event_preflight_directory=event_directory,
        research_event_subtype=event_subtype,
    )

    assert result.status == "no_candidate"
    evaluation = _json(output / "evaluation.json")
    model_card = _json(output / "model_card.json")
    assert evaluation["model_family"] == expected_family
    assert model_card["model_family"] == expected_family
    assert evaluation["dataset"]["research_event_cohort"] == cohort_identity
    assert evaluation["future_holdout_opened"] is False
    assert evaluation["promotion_permitted"] is False
    intraday_development.load_complete_intraday_development_output(output)


def test_event_subtype_requires_preflight_directory(
    a43_dataset: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(DataReadinessError, match="requires a historical event preflight"):
        train_intraday_development_candidate(
            a43_dataset,
            tmp_path / "invalid-directional-event",
            hypothesis="continuation",
            config=_rejecting_config(),
            research_event_subtype="bare_upgrade",
        )


def test_passing_development_candidate_still_keeps_future_closed(
    a43_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _rejecting_config()
    selected_spec = _limit_to_one_candidate(monkeypatch, config)
    _patch_fast_pair(monkeypatch)
    monkeypatch.setattr(
        intraday_development,
        "_evaluate_spec",
        lambda _spec, _scored, folds, _config, _cost: _passing_candidate_record(
            selected_spec, folds
        ),
    )
    output = tmp_path / "candidate"

    result = train_intraday_development_candidate(
        a43_dataset,
        output,
        hypothesis="long-reversion",
        config=config,
    )

    evaluation = _json(output / "evaluation.json")
    manifest = _json(output / "_manifest.json")
    candidate = joblib.load(output / "candidate.joblib")
    assert result.status == "candidate"
    assert intraday_development.load_complete_intraday_development_output(output)[
        "state"
    ] == "candidate"
    assert evaluation["future_holdout_opened"] is False
    assert evaluation["test_access_count"] == 0
    assert manifest["future_holdout_opened"] is False
    assert manifest["test_access_count"] == 0
    assert candidate["opportunity_estimator"] is not None
    assert candidate["downside_estimator"] is not None
    assert candidate["downside_calibrator"] is not None
    with pytest.raises(DataReadinessError, match="future holdout data does not exist"):
        evaluate_future_intraday_holdout(
            output,
            tmp_path / "future-missing",
            tmp_path / "future-output",
        )
    assert not (tmp_path / ".candidate.future-holdout-access.json").exists()


def test_portfolio_ledger_enforces_risk_capital_concurrency_cooldown_and_cost_once() -> None:
    rows: list[dict[str, object]] = []
    session = "2026-06-01"
    for decision_offset, securities in (
        (0, ("A", "B", "C")),
        (10, ("B", "C", "D")),
        (65, ("B", "C")),
    ):
        decision = pd.Timestamp("2026-06-01 14:30:00", tz="UTC") + pd.Timedelta(
            minutes=decision_offset
        )
        for rank, security in enumerate(securities):
            rows.append(
                {
                    "dataset_row_id": f"{decision_offset}-{security}",
                    "ticker": security,
                    "security_id": security,
                    "session_date_et": session,
                    "decision_group_id": decision.isoformat(),
                    "entry_time_utc": decision + pd.Timedelta(minutes=1),
                    "exit_bar_end_utc": decision + pd.Timedelta(minutes=31),
                    "predicted_net_return": 0.004 - rank * 0.0001,
                    "predicted_stop_probability": 0.50 if security == "A" else 0.20,
                    "gross_return": 0.002,
                    "spy_return": 0.0002,
                    "qqq_return": 0.0003,
                    "sector_return": 0.0001,
                    "entry_price": 100.0,
                    "stop_price": 98.5,
                    "fold": 0,
                }
            )
    config = _rejecting_config(
        maximum_candidates_per_decision=3,
        maximum_concurrent_positions=2,
        position_weight=0.5,
        per_security_cooldown_minutes=30,
    )

    ledger = intraday_development._position_ledger(
        pd.DataFrame(rows),
        0.0,
        0.35,
        10.0,
        config,
    )
    positions = ledger["position_records"]
    metrics = intraday_development._ledger_metrics(ledger)

    assert len(positions) == 4
    assert {row["security_id"] for row in positions} == {"B", "C"}
    assert not any(
        str(row["decision_group_id"]).startswith("2026-06-01T14:40")
        for row in positions
    )
    assert max(float(row["entry_weight"]) for row in positions) <= 0.5
    assert max(float(row["predicted_stop_probability"]) for row in positions) <= 0.35
    assert all(float(row["realized_net_return"]) == pytest.approx(0.001) for row in positions)
    assert all(
        float(row["realized_spy_excess_return"]) == pytest.approx(0.0008)
        for row in positions
    )
    assert metrics["maximum_concurrent_positions_observed"] == 2
    assert metrics["maximum_concurrent_positions_enforced"] is True
    assert metrics["capital_weights_enforced"] is True
    assert metrics["security_cooldown_enforced"] is True


def test_conservative_open_stop_marks_contribute_to_drawdown() -> None:
    first_decision = pd.Timestamp("2026-06-01 14:30:00", tz="UTC")
    second_decision = first_decision + pd.Timedelta(minutes=10)
    rows = [
        _ledger_row(
            security_id="A",
            decision=first_decision,
            exit_time=first_decision + pd.Timedelta(minutes=31),
            predicted_net_return=0.01,
            gross_return=0.0,
            stop_price=80.0,
        ),
        _ledger_row(
            security_id="B",
            decision=second_decision,
            exit_time=second_decision + pd.Timedelta(minutes=31),
            predicted_net_return=-0.01,
            gross_return=0.0,
            stop_price=99.0,
        ),
    ]
    config = _rejecting_config(
        maximum_candidates_per_decision=1,
        maximum_concurrent_positions=1,
        position_weight=1.0,
    )

    ledger = intraday_development._position_ledger(
        pd.DataFrame(rows),
        0.0,
        0.35,
        0.0,
        config,
    )
    metrics = intraday_development._ledger_metrics(ledger)

    assert min(ledger["equity_marks"]) == pytest.approx(0.80)
    assert metrics["maximum_drawdown"] == pytest.approx(0.20)


def test_simultaneous_exits_add_one_order_independent_post_batch_equity_mark() -> None:
    exit_time = pd.Timestamp("2026-06-01 15:01:00", tz="UTC")

    def close_in_order(security_order: tuple[str, ...]) -> tuple[float, list[float], list[str]]:
        gross_returns = {"A": 0.10, "B": -0.10}
        open_positions = [
            {
                "security_id": security_id,
                "exit_time_utc": exit_time,
                "gross_return": gross_returns[security_id],
                "notional": 0.5,
                "spy_return": 0.0,
                "qqq_return": 0.0,
                "sector_return": 0.0,
                "entry_price": 100.0,
                "stop_price": 100.0,
            }
            for security_id in security_order
        ]
        completed: list[dict[str, Any]] = []
        equity_marks = [1.0]

        _cash, equity = intraday_development._close_due_positions(
            open_positions,
            cutoff=exit_time,
            cash=0.0,
            cost_bps=0.0,
            cooldown_minutes=30,
            cooldown={},
            completed=completed,
            equity_marks=equity_marks,
        )
        return (
            equity,
            equity_marks,
            [str(position["security_id"]) for position in completed],
        )

    forward = close_in_order(("A", "B"))
    reversed_order = close_in_order(("B", "A"))

    assert forward == reversed_order
    assert forward[0] == pytest.approx(1.0)
    assert forward[1] == pytest.approx([1.0, 1.0, 1.0])
    assert forward[2] == ["A", "B"]


def test_capacity_gate_uses_per_decision_entries_not_session_total() -> None:
    config = _gate_config()
    metrics = _passing_scope_metrics()
    metrics["maximum_entries_per_session_observed"] = 6
    metrics["maximum_entries_per_decision_observed"] = 3

    passed, reasons = intraday_development._scope_gates(
        metrics, config, scope="seen_security"
    )

    assert passed is True, reasons
    assert "session_entry_capacity_breached" not in reasons


@pytest.mark.parametrize("output_relation", ["same", "inside", "parent"])
def test_training_output_cannot_overlap_a43_authority(
    a43_dataset: Path,
    output_relation: str,
) -> None:
    output = {
        "same": a43_dataset,
        "inside": a43_dataset / "candidate",
        "parent": a43_dataset.parent,
    }[output_relation]

    with pytest.raises(DataReadinessError, match="overlaps"):
        train_intraday_development_candidate(
            a43_dataset,
            output,
            hypothesis="continuation",
            config=_rejecting_config(),
        )


def test_intraday_policy_cannot_exceed_four_gib() -> None:
    assert _rejecting_config().maximum_process_memory_gib == 4.0
    with pytest.raises(ValueError, match=r"\(0, 4\] GiB"):
        _rejecting_config(maximum_process_memory_gib=4.01)


@pytest.mark.parametrize("output_owner", ["candidate", "future"])
def test_future_output_overlap_is_rejected_before_access_is_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_owner: str,
) -> None:
    candidate_directory = tmp_path / "candidate"
    future_directory = tmp_path / "future"
    candidate_directory.mkdir()
    future_directory.mkdir()
    config = _rejecting_config(
        future_access_registry_directory=str(tmp_path / "future-access-registry")
    )
    candidate = _future_candidate(config)
    monkeypatch.setattr(
        intraday_development,
        "_load_validation_passed_candidate",
        lambda _directory: (candidate, {"schema_version": "candidate-manifest"}),
    )
    access_consumed = False

    def consume_access(*_args: object) -> Path:
        nonlocal access_consumed
        access_consumed = True
        raise AssertionError("future access must not be consumed for overlapping output")

    monkeypatch.setattr(intraday_development, "_consume_future_access", consume_access)
    owner = candidate_directory if output_owner == "candidate" else future_directory

    with pytest.raises(DataReadinessError, match="overlap"):
        evaluate_future_intraday_holdout(
            candidate_directory,
            future_directory,
            owner / "future-evidence",
        )

    assert access_consumed is False


@pytest.mark.parametrize(
    ("identity_key", "message"),
    [
        ("strategy_contract_sha256", "strategy_contract_sha256"),
        ("transformation_sha256", "transformation_sha256"),
    ],
)
def test_future_holdout_requires_full_development_identity_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_key: str,
    message: str,
) -> None:
    config = _rejecting_config(
        future_access_registry_directory=str(tmp_path / "future-access-registry")
    )
    candidate = _future_candidate(config)
    future_directory = tmp_path / f"future-{identity_key}"
    future_directory.mkdir()
    published = _future_published(_training_frame(session_count=1, security_count=2))
    setattr(published, identity_key, "f" * 64)
    lock = tmp_path / f"{identity_key}.lock"
    lock.write_text("locked", encoding="utf-8")
    monkeypatch.setattr(
        intraday_development,
        "_load_validation_passed_candidate",
        lambda _directory: (candidate, {"schema_version": "candidate-manifest"}),
    )
    monkeypatch.setattr(intraday_development, "_require_output_isolated", lambda *_args: None)
    monkeypatch.setattr(intraday_development, "_consume_future_access", lambda *_args: lock)
    monkeypatch.setattr(
        intraday_development, "load_published_intraday_dataset", lambda _directory: published
    )

    with pytest.raises(DataReadinessError, match=message):
        evaluate_future_intraday_holdout(
            tmp_path / "candidate",
            future_directory,
            tmp_path / f"evidence-{identity_key}",
        )


@pytest.mark.parametrize(
    ("selected_security", "minimum_rows", "minimum_securities", "message"),
    [
        (None, 3, 2, "profile rows"),
        ("SEC00", 2, 2, "profile securities"),
    ],
)
def test_future_profile_population_must_meet_frozen_minimums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_security: str | None,
    minimum_rows: int,
    minimum_securities: int,
    message: str,
) -> None:
    config = _rejecting_config(
        future_access_registry_directory=str(tmp_path / "future-access-registry")
    )
    candidate = _future_candidate(
        config,
        minimum_rows=minimum_rows,
        minimum_securities=minimum_securities,
    )
    frame = _training_frame(session_count=2, security_count=2)
    profile_rows = (
        frame.index[:2]
        if selected_security is None
        else frame.index[frame["security_id"].eq(selected_security)]
    )
    frame.loc[:, "volume_return_1_bar"] = -1.0
    frame.loc[:, "stock_return_20m"] = -1.0
    frame.loc[:, "session_vwap_distance_five_minute_atr"] = -1.0
    frame.loc[profile_rows, "volume_return_1_bar"] = 1.0
    frame.loc[profile_rows, "stock_return_20m"] = 1.0
    frame.loc[profile_rows, "session_vwap_distance_five_minute_atr"] = 1.0
    future_directory = tmp_path / f"future-{message.replace(' ', '-')}"
    future_directory.mkdir()
    published = _future_published(frame)
    lock = tmp_path / f"{message.replace(' ', '-')}.lock"
    lock.write_text("locked", encoding="utf-8")
    monkeypatch.setattr(
        intraday_development,
        "_load_validation_passed_candidate",
        lambda _directory: (candidate, {"schema_version": "candidate-manifest"}),
    )
    monkeypatch.setattr(intraday_development, "_require_output_isolated", lambda *_args: None)
    monkeypatch.setattr(intraday_development, "_consume_future_access", lambda *_args: lock)
    monkeypatch.setattr(
        intraday_development, "load_published_intraday_dataset", lambda _directory: published
    )
    monkeypatch.setattr(
        intraday_development,
        "_validate_future_frame",
        lambda _published, _future_start, _development_end, _policy: frame.copy(),
    )

    with pytest.raises(DataReadinessError, match=message):
        evaluate_future_intraday_holdout(
            tmp_path / "candidate",
            future_directory,
            tmp_path / f"evidence-{message.replace(' ', '-')}",
        )


def test_future_access_registry_is_keyed_by_candidate_authority_hash(
    tmp_path: Path,
) -> None:
    first_candidate = tmp_path / "candidate-a"
    second_candidate = tmp_path / "relocated" / "candidate-b"
    future_directory = tmp_path / "future"
    registry = tmp_path / "registry"
    first_candidate.mkdir()
    future_directory.mkdir()
    (first_candidate / "_authority.json").write_text(
        json.dumps({"schema_version": "candidate-authority", "state": "candidate"}),
        encoding="utf-8",
    )
    (future_directory / "_authority.json").write_text(
        json.dumps({"schema_version": "future-dataset-authority", "state": "complete"}),
        encoding="utf-8",
    )
    second_candidate.parent.mkdir()
    shutil.copytree(first_candidate, second_candidate)
    authority_sha256 = intraday_development.file_sha256(
        first_candidate / "_authority.json"
    )

    lock = intraday_development._consume_future_access(
        first_candidate,
        future_directory,
        registry,
    )

    assert lock.parent.resolve() == registry.resolve()
    assert authority_sha256 in lock.name
    with pytest.raises(DataReadinessError, match="already consumed"):
        intraday_development._consume_future_access(
            second_candidate,
            future_directory,
            registry,
        )


def test_relative_future_registry_is_stable_home_based_and_embedded_in_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    first_working_directory = tmp_path / "working-a"
    second_working_directory = tmp_path / "working-b"
    home.mkdir()
    first_working_directory.mkdir()
    second_working_directory.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    config = _rejecting_config(
        future_access_registry_directory="state/intraday-future-access"
    )

    monkeypatch.chdir(first_working_directory)
    first_contract = intraday_development._future_data_contract(config)
    monkeypatch.chdir(second_working_directory)
    second_contract = intraday_development._future_data_contract(config)

    expected = (
        home / ".market-predictor" / "state" / "intraday-future-access"
    ).resolve()
    embedded = Path(str(first_contract["future_access_registry_directory"]))
    assert embedded.is_absolute()
    assert embedded == expected
    assert second_contract["future_access_registry_directory"] == str(expected)


def test_successful_future_evidence_replays_and_rejects_tamper_when_supported(
    tmp_path: Path,
) -> None:
    loader = getattr(
        intraday_development,
        "load_complete_intraday_future_evaluation_output",
        None,
    )
    if not callable(loader):
        pytest.skip("strict future-evidence loader is not implemented")
    output = tmp_path / "future-evidence"
    _publish_valid_future_evidence(output)

    replay = loader(output)
    assert replay["status"] == "locked_future_evaluated"

    tampered = tmp_path / "tampered-future-evidence"
    shutil.copytree(output, tampered)
    tampered_evaluation = _json(tampered / "future_evaluation.json")
    tampered_evaluation["selection_changed_after_future_observation"] = True
    (tampered / "future_evaluation.json").write_text(
        json.dumps(tampered_evaluation, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(DataReadinessError):
        loader(tampered)


def test_strict_future_loader_rejects_unexpected_nested_entry(tmp_path: Path) -> None:
    output = tmp_path / "future-evidence-with-nested-entry"
    _publish_valid_future_evidence(output)
    unexpected = output / "unexpected" / "nested.txt"
    unexpected.parent.mkdir()
    unexpected.write_text("not governed", encoding="utf-8")

    with pytest.raises(DataReadinessError, match="exact-file inventory"):
        intraday_development.load_complete_intraday_future_evaluation_output(output)


def _publish_valid_future_evidence(output: Path) -> None:
    evaluation = {
        "schema_version": intraday_development.FUTURE_EVALUATION_SCHEMA_VERSION,
        "status": "locked_future_evaluated",
        "promotion_permitted": False,
        "selection_changed_after_future_observation": False,
        "future_access_lock_sha256": "a" * 64,
        "candidate_authority_sha256": "b" * 64,
        "candidate_manifest_sha256": "c" * 64,
        "candidate_manifest_schema": "candidate-manifest",
        "future_dataset": {
            "authority_sha256": "d" * 64,
            "manifest_sha256": "e" * 64,
        },
        "future_session_first": "2026-07-09",
        "future_session_last": "2026-07-10",
        "metrics": {
            "position_ledger_rows": 1,
            "daily_ledger_rows": 1,
            "average_daily_net_return": 0.01,
            "compounded_net_return": 0.01,
            "negative_session_rate": 0.0,
            "maximum_entries_per_session_observed": 1,
            "average_trade_net_return": 0.02,
        },
    }
    intraday_development._publish_future_evaluation(
        output,
        evaluation,
        {
            "position_records": [{"notional": 0.5, "pnl": 0.01}],
            "daily_records": [{"daily_return": 0.01, "entries": 1}],
        },
    )


def _ledger_row(
    *,
    security_id: str,
    decision: pd.Timestamp,
    exit_time: pd.Timestamp,
    predicted_net_return: float,
    gross_return: float,
    stop_price: float,
) -> dict[str, object]:
    return {
        "dataset_row_id": f"{decision.isoformat()}-{security_id}",
        "ticker": security_id,
        "security_id": security_id,
        "session_date_et": "2026-06-01",
        "decision_group_id": decision.isoformat(),
        "entry_time_utc": decision + pd.Timedelta(minutes=1),
        "exit_bar_end_utc": exit_time,
        "predicted_net_return": predicted_net_return,
        "predicted_stop_probability": 0.10,
        "gross_return": gross_return,
        "spy_return": 0.0,
        "qqq_return": 0.0,
        "sector_return": 0.0,
        "entry_price": 100.0,
        "stop_price": stop_price,
        "fold": 0,
    }


def _future_candidate(
    config: IntradayDevelopmentConfig,
    *,
    minimum_rows: int = 1,
    minimum_securities: int = 2,
) -> dict[str, Any]:
    profile = baseline_profile("continuation", config)
    future_contract = intraday_development._future_data_contract(config)
    future_contract.update(
        {
            "minimum_sessions": 1,
            "minimum_rows": minimum_rows,
            "minimum_securities": minimum_securities,
        }
    )
    return {
        "model_family": "intraday_technical",
        "future_data_contract": future_contract,
        "training_config": asdict(config),
        "dataset": {
            "transformation_sha256": "c" * 64,
            "strategy_contract_sha256": "b" * 64,
            "ordered_feature_sha256": "d" * 64,
        },
        "frozen_round_trip_cost_bps": 10.0,
        "baseline_profile": asdict(profile),
        "baseline_profile_sha256": profile.sha256(),
        "opportunity_estimator": _OpportunityEstimator(),
        "downside_estimator": _DownsideEstimator(),
        "downside_calibrator": _DownsideCalibrator(),
        "expected_net_return_threshold_bps": 0.0,
        "maximum_stop_probability": 0.35,
    }


def _future_published(frame: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        frame=frame,
        frozen_round_trip_cost_bps=10.0,
        dataset_sha256="1" * 64,
        manifest_sha256="1" * 64,
        authority_sha256="2" * 64,
        request_sha256="3" * 64,
        transformation_sha256="c" * 64,
        session_unit_inventory_sha256="4" * 64,
        ordered_feature_sha256="d" * 64,
        strategy_contract_sha256="b" * 64,
    )


def _profile_frame(
    dataset: Path,
    config: IntradayDevelopmentConfig,
    hypothesis: str,
) -> pd.DataFrame:
    published = intraday_development.load_published_intraday_dataset(dataset)
    data = intraday_development._validate_development_frame(published, config)
    profile = baseline_profile(hypothesis, config)
    return data.loc[intraday_development._profile_mask(data, profile)].reset_index(
        drop=True
    )


def _rejecting_config(**overrides: object) -> IntradayDevelopmentConfig:
    values: dict[str, Any] = asdict(IntradayDevelopmentConfig())
    values.update(
        {
            "validation_folds": 4,
            "minimum_train_sessions": 30,
            "minimum_validation_sessions": 5,
            "minimum_rows": 100,
            "minimum_securities": 10,
            "minimum_calibration_sessions": 5,
            "maximum_candidates_per_decision": 3,
            "maximum_concurrent_positions": 3,
            "position_weight": 1.0 / 3.0,
            "per_security_cooldown_minutes": 30,
            "expected_net_return_thresholds_bps": (0.0,),
            "maximum_stop_probability_thresholds": (0.35,),
            "ridge_alphas": (1.0,),
            "logistic_c_values": (1.0,),
            "hgb_learning_rates": (0.05,),
            "hgb_max_leaf_nodes": (7,),
            "hgb_max_iter": 20,
            "hgb_max_bins": 31,
            "bootstrap_samples": 100,
            "bootstrap_block_sessions": 2,
            "minimum_validation_trades": 1,
            "minimum_validation_sessions_with_trades": 2,
            "minimum_scope_rows": 100,
            "minimum_scope_securities": 5,
            "minimum_average_trade_net_return_bps": 10_000.0,
            "cost_curve_bps": (0.0, 5.0, 10.0, 20.0),
            "maximum_process_memory_gib": 4.0,
            "memory_guard_headroom_gib": 0.75,
        }
    )
    values.update(overrides)
    return IntradayDevelopmentConfig(**values)


def _gate_config() -> IntradayDevelopmentConfig:
    return _rejecting_config(
        minimum_average_trade_net_return_bps=-100.0,
        minimum_average_daily_net_return_bps=-100.0,
        minimum_daily_return_ci_low_bps=-100.0,
        minimum_profit_factor=1.0,
        minimum_average_spy_excess_bps=-100.0,
        minimum_average_qqq_excess_bps=-100.0,
        minimum_average_sector_excess_bps=-100.0,
        minimum_stress_average_daily_return_bps=-100.0,
        maximum_drawdown=0.50,
        maximum_round_trip_turnover=1.0,
        minimum_profitable_fold_fraction=0.50,
        maximum_negative_session_rate=0.75,
        minimum_return_to_drawdown=0.0,
    )


def _passing_scope_metrics() -> dict[str, Any]:
    interval = {"estimate": 0.002, "low": 0.001, "high": 0.003}
    return {
        "rows": 200,
        "securities": 10,
        "positive_net_return_roc_auc": 0.90,
        "top_decile_positive_net_return_lift": 2.0,
        "stop_hit_roc_auc": 0.90,
        "stop_hit_brier": 0.10,
        "stop_hit_ece": 0.02,
        "stop_hit_brier_skill": 0.50,
        "trade_count": 100,
        "sessions_with_trades": 10,
        "average_trade_net_return": 0.002,
        "average_daily_net_return": 0.002,
        "moving_block_bootstrap_95_ci": {
            "average_daily_net_return": interval,
        },
        "profit_factor": 2.0,
        "maximum_drawdown": 0.05,
        "economic_rank_gain_over_exact_random_baseline": 0.002,
        "economic_rank_gain_bootstrap_95_ci": interval,
        "benchmark_excess_bootstrap_95_ci": {
            "spy": interval,
            "qqq": interval,
            "sector": interval,
        },
        "average_daily_round_trip_turnover": 0.25,
        "profitable_fold_fraction": 1.0,
        "negative_session_rate": 0.25,
        "return_to_drawdown": 1.0,
        "maximum_entries_per_session_observed": 3,
        "maximum_entries_per_decision_observed": 3,
        "maximum_concurrent_positions_observed": 3,
        "cost_curve": [
            {
                "round_trip_cost_bps": 20.0,
                "daily_return_bootstrap_95_ci": {
                    "average_daily_net_return": interval,
                },
            }
        ],
    }


def _passing_scopes() -> dict[str, Any]:
    return {
        scope: {
            "passed": True,
            "failed_gate_reasons": [],
            "metrics": _passing_scope_metrics(),
        }
        for scope in ("seen_security", "unseen_security")
    }


def _scored_policy_frame() -> pd.DataFrame:
    rows = []
    for fold in range(4):
        for scope in ("seen_security", "unseen_security"):
            rows.append(
                {
                    "fold": fold,
                    "validation_scope": scope,
                    "predicted_net_return": 0.002,
                    "predicted_stop_probability": 0.10,
                }
            )
    return pd.DataFrame(rows)


def _limit_to_one_candidate(
    monkeypatch: pytest.MonkeyPatch,
    config: IntradayDevelopmentConfig,
) -> Any:
    selected = intraday_development._candidate_specs(config)[0]
    monkeypatch.setattr(
        intraday_development,
        "_candidate_specs",
        lambda _config: (selected,),
    )
    return selected


class _OpportunityEstimator:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features[:, 0], dtype="float64") / 100.0


class _DownsideEstimator:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probability = np.full(len(features), 0.20, dtype="float64")
        return np.column_stack((1.0 - probability, probability))


class _DownsideCalibrator:
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        probability = np.full(len(logits), 0.20, dtype="float64")
        return np.column_stack((1.0 - probability, probability))


def _patch_fast_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    def fit_pair(
        _spec: object,
        data: pd.DataFrame,
        _features: np.ndarray,
        _opportunity: np.ndarray,
        _downside: np.ndarray,
        sessions: tuple[str, ...],
        config: IntradayDevelopmentConfig,
        **_kwargs: object,
    ) -> Any:
        fit, calibration = intraday_development._split_downside_calibration(
            data, sessions, config
        )
        return intraday_development._FittedPair(
            opportunity_estimator=_OpportunityEstimator(),
            downside_estimator=_DownsideEstimator(),
            downside_calibrator=_DownsideCalibrator(),
            fit_sessions=fit,
            calibration_sessions=calibration,
        )

    monkeypatch.setattr(intraday_development, "_fit_pair", fit_pair)


def _passing_candidate_record(spec: Any, folds: Any) -> dict[str, Any]:
    scopes = _passing_scopes()
    return {
        "candidate_id": spec.candidate_id,
        "family": spec.family,
        "hyperparameters": dict(spec.hyperparameters),
        "opportunity_training_target": "net_return",
        "downside_training_target": "stop_hit",
        "target_hit_used_as_training_target": False,
        "folds": list(folds),
        "selection_policies": [
            {
                "threshold_bps": 0.0,
                "maximum_stop_probability": 0.35,
                "selection_passed": True,
                "failed_gate_reasons": [],
                "selection_scopes": scopes,
            }
        ],
        "selection_passed": True,
        "validation_passed": True,
        "selected_threshold_bps": 0.0,
        "selected_maximum_stop_probability": 0.35,
        "selected_selection_scopes": scopes,
        "confirmation_scopes": scopes,
        "confirmation_policy_frozen_before_scoring": True,
        "failed_gate_reasons": [],
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
