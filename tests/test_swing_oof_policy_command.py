"""CLI preserves evidence pins and sequential action/source evaluation ownership."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import typer
from typer.testing import CliRunner

from market_predictor.commands import swing_oof_policy_evaluation as commands
from market_predictor.core.errors import DataReadinessError
from market_predictor.heavy_jobs import HeavyJobBusyError
from market_predictor.swing.contracts.holding_materialization import SourcePin


def _app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def root() -> None:
        pass

    commands.register_saved_oof_policy_command(app)
    return app


def _args() -> list[str]:
    values = ["evaluate-saved-swing-policy", "--root", "example-root", "--output", "data/research/evaluation"]
    for option, path in (
        ("run-manifest", "run.json"), ("feature-publication", "features.json"),
        ("target-config", "targets.toml"), ("strategy-contract", "strategy.toml"), ("research-contract", "research.toml"),
    ):
        values += [f"--{option}", path, f"--{option}-sha256", "a" * 64]
    return values + ["--learner", "regularized_linear_return", "--exit-policy", "stop_ten_session_timeout"]


def _patch(monkeypatch: pytest.MonkeyPatch, status: str = "computed") -> tuple[Mock, Mock, Mock]:
    load = Mock(return_value=SimpleNamespace(
        action_config=SourcePin(path="actions.toml", sha256="b" * 64),
        action_archive="data/raw/actions", action_audit_sha256="c" * 64,
    ))
    actions = Mock(return_value=object())
    evaluate = Mock(return_value={
        "status": status, "manifest_sha256": "d" * 64, "selected_rows": 12,
        "training_eligible": False, "promotion_eligible": False, "serving_eligible": False,
    })
    monkeypatch.setattr(commands, "load_corrected_outcome_policy", load)
    monkeypatch.setattr(commands, "load_corporate_action_evidence", actions)
    monkeypatch.setattr(commands, "publish_saved_oof_policy_evaluation", evaluate)
    monkeypatch.setattr(commands, "file_sha256", Mock(return_value="b" * 64))
    return load, actions, evaluate


@pytest.mark.parametrize("status,code", [("computed", 0), ("valuation_unavailable", 2)])
def test_saved_policy_command_preserves_inputs_and_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch, status: str, code: int,
) -> None:
    load, actions, evaluate = _patch(monkeypatch, status)
    sequence = Mock()
    sequence.attach_mock(load, "target")
    sequence.attach_mock(actions, "actions")
    sequence.attach_mock(evaluate, "evaluate")
    result = CliRunner().invoke(_app(), _args())
    assert result.exit_code == code, result.output
    assert json.loads(result.output)["serving_eligible"] is False
    assert Path(json.loads(result.output)["report_path"]).name == "report.json"
    assert [call[0] for call in sequence.mock_calls] == ["target", "actions", "evaluate"]
    load.assert_called_once_with(Path("example-root"), Path("targets.toml"), "a" * 64)
    actions.assert_called_once_with(
        root=Path("example-root"), config=Path("actions.toml"), archive=Path("data/raw/actions"), expected_audit_sha256="c" * 64,
    )
    evaluate.assert_called_once_with(
        root=Path("example-root"), run_manifest=SourcePin(path="run.json", sha256="a" * 64),
        feature_publication=SourcePin(path="features.json", sha256="a" * 64),
        target_config=SourcePin(path="targets.toml", sha256="a" * 64),
        strategy_contract=SourcePin(path="strategy.toml", sha256="a" * 64),
        research_contract=SourcePin(path="research.toml", sha256="a" * 64),
        learner="regularized_linear_return", exit_policy="stop_ten_session_timeout", evidence=actions.return_value,
        output=Path("data/research/evaluation"),
        historical_configuration_evidence=None,
    )


@pytest.mark.parametrize("option", ["--learner", "--exit-policy", "--run-manifest-sha256"])
def test_invalid_policy_or_pin_stops_before_action_replay(monkeypatch: pytest.MonkeyPatch, option: str) -> None:
    load, actions, evaluate = _patch(monkeypatch)
    arguments = _args()
    arguments[arguments.index(option) + 1] = "unapproved"
    result = CliRunner().invoke(_app(), arguments)
    assert result.exit_code == 2
    load.assert_not_called()
    actions.assert_not_called()
    evaluate.assert_not_called()


def test_action_config_pin_mismatch_stops_before_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    _, actions, evaluate = _patch(monkeypatch)
    monkeypatch.setattr(commands, "file_sha256", Mock(return_value="e" * 64))
    result = CliRunner().invoke(_app(), _args())
    assert result.exit_code == 2
    assert "action configuration differs" in result.output
    actions.assert_not_called()
    evaluate.assert_not_called()


@pytest.mark.parametrize("failure,code", [
    (HeavyJobBusyError("source scan owns workspace lease"), 75),
    (DataReadinessError("corporate action archive hash differs"), 2),
])
def test_action_replay_failure_does_not_open_evaluation(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, code: int,
) -> None:
    _, actions, evaluate = _patch(monkeypatch)
    actions.side_effect = failure
    result = CliRunner().invoke(_app(), _args())
    assert result.exit_code == code
    assert str(failure) in result.output
    evaluate.assert_not_called()


@pytest.mark.parametrize("extra", [
    ["--historical-configuration-evidence", "proof.json"],
    ["--historical-configuration-evidence-sha256", "a" * 64],
    ["--historical-configuration-evidence", "proof.json", "--historical-configuration-evidence-sha256", "invalid"],
])
def test_incomplete_historical_evidence_pair_stops_before_actions(
    monkeypatch: pytest.MonkeyPatch, extra: list[str],
) -> None:
    load, actions, evaluate = _patch(monkeypatch)
    result = CliRunner().invoke(_app(), [*_args(), *extra])
    assert result.exit_code == 2
    load.assert_not_called()
    actions.assert_not_called()
    evaluate.assert_not_called()


def test_explicit_historical_evidence_pin_reaches_actual_evaluator_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, evaluate = _patch(monkeypatch)
    result = CliRunner().invoke(_app(), [*_args(), "--historical-configuration-evidence", "proof.json",
        "--historical-configuration-evidence-sha256", "e" * 64])
    assert result.exit_code == 0, result.output
    assert evaluate.call_args.kwargs["historical_configuration_evidence"] == SourcePin(path="proof.json", sha256="e" * 64)
