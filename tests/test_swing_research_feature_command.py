from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from market_predictor.commands import swing_research_features as commands


@pytest.mark.parametrize("complete", [True, False])
def test_research_predictor_command_reports_checkpoint(monkeypatch: pytest.MonkeyPatch, complete: bool) -> None:
    calls: list[dict[str, object]] = []

    def build(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"status": "technical_inputs_complete_research_only" if complete else "partial_in_progress",
            "groups": {"one": {}}, "failed_groups": {}, "months": {}, "checkpoint_sha256": "a" * 64}

    monkeypatch.setattr(commands, "materialize_research_predictors", build)
    app = typer.Typer()
    commands.register_research_feature_commands(app)
    result = CliRunner().invoke(app, ["materialize-swing-research-predictors",
        "--expected-config-sha256", "b" * 64, "--maximum-groups-this-run", "1"])
    assert result.exit_code == (0 if complete else 2), result.output
    assert json.loads(result.output)["groups"] == 1
    assert calls[0]["maximum_groups_this_run"] == 1
    assert calls[0]["expected_config_sha256"] == "b" * 64


def test_research_predictor_command_rejects_zero_batch() -> None:
    app = typer.Typer()
    commands.register_research_feature_commands(app)
    result = CliRunner().invoke(app, ["materialize-swing-research-predictors",
        "--expected-config-sha256", "b" * 64, "--maximum-groups-this-run", "0"])
    assert result.exit_code == 2


@pytest.mark.parametrize("with_replay", [False, True])
def test_research_join_command_preserves_independent_input_pins(
    monkeypatch: pytest.MonkeyPatch, with_replay: bool,
) -> None:
    calls: list[dict[str, object]] = []

    def build(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"status": "complete_research_only", "rows": 2, "training_eligible": False,
            "promotion_eligible": False, "manifest_sha256": "f" * 64, "checkpoint_sha256": "e" * 64}

    monkeypatch.setattr(commands, "materialize_research_dataset", build)
    app = typer.Typer()
    commands.register_research_feature_commands(app)
    arguments = ["materialize-swing-research-dataset"]
    for name, suffix in (("decision-config", "toml"), ("strategy-config", "toml"),
        ("predictor-manifest", "json"), ("outcome-manifest", "json"), ("catalyst-manifest", "json")):
        arguments += [f"--{name}", f"{name}.{suffix}", f"--{name}-sha256", "a" * 64]
    if with_replay:
        for name in ("predictor-replay", "outcome-replay"):
            arguments += [f"--{name}", f"{name}.json", f"--{name}-sha256", "b" * 64]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.exception
    assert json.loads(result.output)["training_eligible"] is False
    assert str(calls[0]["output"]).endswith("swing_corrected_initial_fit_research")
    assert calls[0]["predictor_replay"] == (
        commands.SourcePin(path="predictor-replay.json", sha256="b" * 64) if with_replay else None)
    assert calls[0]["outcome_replay"] == (
        commands.SourcePin(path="outcome-replay.json", sha256="b" * 64) if with_replay else None)


@pytest.mark.parametrize("flag", ["--predictor-replay", "--predictor-replay-sha256",
    "--outcome-replay", "--outcome-replay-sha256"])
def test_join_cli_requires_complete_replay_pair(flag: str) -> None:
    app = typer.Typer()
    commands.register_research_feature_commands(app)
    arguments = ["materialize-swing-research-dataset"]
    for name in ("decision-config", "strategy-config", "predictor-manifest", "outcome-manifest", "catalyst-manifest"):
        arguments += [f"--{name}", f"{name}.json", f"--{name}-sha256", "a" * 64]
    arguments += [flag, "b" * 64 if flag.endswith("sha256") else "replay.json"]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 2
    assert "supplied together" in result.output


@pytest.mark.parametrize("complete", [True, False])
def test_predictor_replay_cli_preserves_pins_and_partial_status(
    monkeypatch: pytest.MonkeyPatch, complete: bool,
) -> None:
    calls: list[dict[str, object]] = []

    def run(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"status": "exact_replay_complete" if complete else "partial", "replay_complete": complete,
            "training_eligible": False, "promotion_eligible": False}

    monkeypatch.setattr(commands, "replay_predictor_publication", run)
    app = typer.Typer()
    commands.register_research_feature_commands(app)
    result = CliRunner().invoke(app, ["replay-swing-research-predictors", "--publication", "data/features/publication.json",
        "--publication-sha256", "a" * 64, "--migration-bindings", "data/evidence/bindings.json",
        "--migration-bindings-sha256", "b" * 64, "--implementation-snapshot", "data/evidence/snapshot.json",
        "--snapshot-sha256", "c" * 64, "--output", "data/reports/predictor-replay", "--maximum-groups", "1",
        "--feature-plan-snapshot", "data/evidence/plan.json", "--feature-plan-snapshot-sha256", "d" * 64])
    assert result.exit_code == (0 if complete else 2), result.output
    assert calls[0]["maximum_groups"] == 1
    assert calls[0]["publication"] == commands.SourcePin(path="data/features/publication.json", sha256="a" * 64)
    assert calls[0]["implementation_snapshot"] == commands.SourcePin(path="data/evidence/snapshot.json", sha256="c" * 64)
    assert calls[0]["feature_plan_snapshot"] == commands.SourcePin(path="data/evidence/plan.json", sha256="d" * 64)
