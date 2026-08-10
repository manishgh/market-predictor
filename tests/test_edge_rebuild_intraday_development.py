from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild import intraday_development
from market_predictor.edge_rebuild.intraday_development import (
    IntradayDevelopmentConfig,
    evaluate_future_intraday_holdout,
    load_intraday_development_config,
    train_intraday_development_candidate,
)
from market_predictor.edge_rebuild.intraday_rejection import (
    publish_intraday_candidate_rejection,
)
from market_predictor.v3.errors import DataReadinessError
from tests.test_edge_rebuild_intraday_training import _publish_dataset, _training_frame


def test_repository_development_policy_freezes_future_boundary_and_economics() -> None:
    config = load_intraday_development_config(
        Path("configs/edge_rebuild_intraday_development.toml")
    )

    assert config.development_end_date == "2026-07-08"
    assert config.future_holdout_start_date == "2026-07-09"
    assert config.cost_curve_bps == (0.0, 5.0, 10.0, 20.0)
    assert config.stress_cost_bps == 20.0
    assert config.bootstrap_samples == 2_000
    assert config.maximum_process_memory_gib == 4.0


def test_development_run_publishes_no_candidate_without_opening_future(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _publish_dataset(tmp_path / "dataset", _training_frame())
    published = intraday_development.load_published_intraday_dataset(dataset)
    original = published.frame.copy(deep=True)
    monkeypatch.setattr(
        intraday_development,
        "load_published_intraday_dataset",
        lambda _: published,
    )
    output = tmp_path / "development"
    result = train_intraday_development_candidate(dataset, output, config=_rejecting_config())

    assert result.status == "no_candidate"
    assert result.selected_candidate_id is None
    assert not (output / "candidate.joblib").exists()
    assert (output / "position_ledger.parquet").is_file()
    assert (output / "daily_ledger.parquet").is_file()
    evaluation = _json(output / "evaluation.json")
    assert evaluation["future_holdout_opened"] is False
    assert evaluation["target_hit_used_as_training_target"] is False
    assert evaluation["raw_ndcg_reported"] is False
    assert evaluation["status"] == "no_candidate"
    assert all(not record["validation_passed"] for record in evaluation["validation_candidates"])
    assert all(
        record["training_target"] == "net_return"
        for record in evaluation["validation_candidates"]
    )
    assert evaluation["auditable_policy_ledger"]["selection_status"] == "best_failed_diagnostic_only"
    pd.testing.assert_frame_equal(published.frame, original)

    with pytest.raises(DataReadinessError, match="locked until validation"):
        evaluate_future_intraday_holdout(output, tmp_path / "must-not-be-opened", tmp_path / "future")


def test_validation_pass_publishes_candidate_but_missing_future_stays_locked(tmp_path: Path) -> None:
    dataset = _publish_dataset(tmp_path / "dataset", _training_frame())
    config = _rejecting_config(
        minimum_average_trade_net_return_bps=-10_000.0,
        minimum_average_daily_net_return_bps=-10_000.0,
        minimum_daily_return_ci_low_bps=-10_000.0,
        minimum_profit_factor=1.0,
        minimum_economic_rank_gain_bps=-10_000.0,
        maximum_drawdown=0.99,
        minimum_stress_average_daily_return_bps=-10_000.0,
    )
    output = tmp_path / "candidate"

    result = train_intraday_development_candidate(dataset, output, config=config)

    assert result.status == "candidate"
    assert result.selected_candidate_id is not None
    assert (output / "candidate.joblib").is_file()
    assert result.evaluation["auditable_policy_ledger"]["selection_status"] == "selected_candidate"
    with pytest.raises(DataReadinessError, match="future holdout data does not exist"):
        evaluate_future_intraday_holdout(output, tmp_path / "future-missing", tmp_path / "future-evidence")


def test_development_boundary_is_checked_before_model_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _training_frame()
    frame.loc[0, "session_date_et"] = date(2026, 7, 9)
    dataset = _publish_dataset(tmp_path / "dataset", frame)
    called = False

    def forbidden_fit(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("fit should not run")

    monkeypatch.setattr(intraday_development, "_fit", forbidden_fit)
    with pytest.raises(DataReadinessError, match="refuses observations after"):
        train_intraday_development_candidate(dataset, tmp_path / "out", config=_rejecting_config())
    assert called is False


def test_position_ledger_enforces_capital_concurrency_and_cooldown() -> None:
    rows: list[dict[str, object]] = []
    session = "2026-06-01"
    for decision_offset, securities in ((0, ("A", "B", "C")), (10, ("A", "B", "C")), (65, ("A", "C"))):
        decision = pd.Timestamp("2026-06-01 14:30:00", tz="UTC") + pd.Timedelta(minutes=decision_offset)
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
                    "gross_return": 0.002,
                    "net_return": 0.001,
                }
            )
    config = _rejecting_config(
        maximum_candidates_per_decision=3,
        maximum_concurrent_positions=2,
        position_weight=0.5,
        per_security_cooldown_minutes=30,
    )
    ledger = intraday_development._position_ledger(pd.DataFrame(rows), 0.0, 10.0, config)

    positions = ledger["position_records"]
    assert len(positions) == 4
    assert sum(row["decision_group_id"].startswith("2026-06-01T14:40") for row in positions) == 0
    assert max(float(row["entry_weight"]) for row in positions) <= 0.5
    assert intraday_development._ledger_metrics(ledger)["maximum_concurrent_positions_enforced"] is True


def test_economic_ranking_is_normalized_to_exact_random_baseline() -> None:
    scored = pd.DataFrame(
        {
            "decision_group_id": ["g", "g", "g"],
            "predicted_net_return": [0.03, 0.02, 0.01],
            "net_return": [0.03, 0.0, -0.03],
        }
    )
    metrics = intraday_development._economic_ranking_metrics(scored, 1)

    assert metrics["economic_rank_gain_over_exact_random_baseline"] == pytest.approx(0.03)
    assert metrics["economic_rank_capture_ratio"] == pytest.approx(1.0)
    assert metrics["raw_ndcg_reported"] is False


def test_moving_block_bootstrap_estimate_matches_headline_daily_estimand() -> None:
    values = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001, 0.003]).to_numpy()
    evidence = intraday_development._moving_block_bootstrap(
        values,
        samples=200,
        block_sessions=2,
        seed=7,
    )

    assert evidence["estimand"] == "daily_capital_weighted_portfolio_return"
    assert evidence["average_daily_net_return"]["estimate"] == pytest.approx(values.mean())
    assert evidence["compounded_net_return"]["estimate"] == pytest.approx(
        float((1.0 + values).prod() - 1.0)
    )


def test_rejection_evidence_binds_candidate_and_dataset_hashes(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    evaluation = {
        "dataset": {
            "authority_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "manifest_sha256": "b" * 64,
            "schema_version": "edge_rebuild.intraday_dataset.v2",
        }
    }
    (candidate / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    (candidate / "_manifest.json").write_text("{}", encoding="utf-8")
    policy = tmp_path / "rejection.toml"
    policy.write_text(
        _rejection_policy(candidate, evaluation["dataset"]),
        encoding="utf-8",
    )

    output = tmp_path / "rejected"
    evidence = publish_intraday_candidate_rejection(candidate, policy, output)

    assert evidence["status"] == "rejected"
    assert evidence["serving_eligible"] is False
    assert evidence["candidate_files"]["evaluation.json"]["sha256"] == file_sha256(
        candidate / "evaluation.json"
    )
    authority = _json(output / "_authority.json")
    assert authority["manifest_sha256"] == file_sha256(output / "_manifest.json")
    with pytest.raises(FileExistsError, match="immutable rejection"):
        publish_intraday_candidate_rejection(candidate, policy, output)


def _rejecting_config(**overrides: object) -> IntradayDevelopmentConfig:
    values: dict[str, object] = {
        "validation_folds": 2,
        "minimum_train_sessions": 20,
        "minimum_validation_sessions": 5,
        "minimum_rows": 100,
        "minimum_securities": 4,
        "maximum_candidates_per_decision": 3,
        "maximum_concurrent_positions": 3,
        "position_weight": 1.0 / 3.0,
        "per_security_cooldown_minutes": 30,
        "expected_net_return_thresholds_bps": (0.0, 5.0),
        "ridge_alphas": (1.0,),
        "hgb_learning_rates": (0.05,),
        "hgb_max_leaf_nodes": (7,),
        "hgb_max_iter": 20,
        "hgb_max_bins": 31,
        "bootstrap_samples": 100,
        "bootstrap_block_sessions": 2,
        "minimum_validation_trades": 10,
        "minimum_validation_sessions_with_trades": 5,
        "minimum_average_trade_net_return_bps": 10_000.0,
        "cost_curve_bps": (0.0, 5.0, 10.0, 20.0),
    }
    values.update(overrides)
    return IntradayDevelopmentConfig(**values)


def _rejection_policy(candidate: Path, dataset: dict[str, str]) -> str:
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in candidate.iterdir()
    }
    lines = [
        'candidate_reference_id = "test"',
        f'candidate_directory = "{candidate.as_posix()}"',
        "",
        "[dataset]",
        *(f'{key} = "{value}"' for key, value in dataset.items()),
        "",
    ]
    for name, record in files.items():
        lines.extend(
            [
                f'[candidate_files."{name}"]',
                f'bytes = {record["bytes"]}',
                f'sha256 = "{record["sha256"]}"',
                "",
            ]
        )
    lines.extend(
        [
            "[[reasons]]",
            'code = "negative_economics"',
            'detail = "candidate lost money after costs"',
        ]
    )
    return "\n".join(lines) + "\n"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
