"""Command forwarding and failure reporting, without running data jobs."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import typer
from typer.testing import CliRunner

from market_predictor.commands import swing_issuer_reactions as commands
from market_predictor.core.errors import DataReadinessError
from market_predictor.heavy_jobs import HeavyJobBusyError
from market_predictor.swing.contracts.holding_materialization import SourcePin


def _app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def root() -> None:
        pass

    commands.register_issuer_reaction_commands(app)
    return app


def _report(status: str) -> dict[str, object]:
    return {
        "status": status, "rows": 10, "manifest_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64, "report_sha256": "c" * 64,
        "training_eligible": False, "promotion_eligible": False, "serving_eligible": False,
    }


def test_materialization_forwards_pins_and_bounded_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    publish = Mock(return_value=_report("complete_research_only"))
    monkeypatch.setattr(commands, "materialize_issuer_reactions", publish)
    result = CliRunner().invoke(_app(), [
        "materialize-swing-issuer-reactions", "--root", "example-root", "--config", "config.json",
        "--expected-config-sha256", "d" * 64, "--output", "output",
        "--expected-checkpoint-sha256", "e" * 64, "--maximum-months-this-run", "2",
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["serving_eligible"] is False
    publish.assert_called_once_with(
        root=Path("example-root"), config=Path("config.json"), expected_config_sha256="d" * 64,
        output=Path("output"), expected_checkpoint_sha256="e" * 64, maximum_months_this_run=2,
    )


def test_verification_forwards_exact_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    verify = Mock(return_value=_report("passed"))
    monkeypatch.setattr(commands, "verify_issuer_reaction_rows", verify)
    result = CliRunner().invoke(_app(), [
        "verify-swing-issuer-reactions", "--root", "example-root", "--publication", "publication.json",
        "--publication-sha256", "d" * 64, "--output", "receipt",
    ])
    assert result.exit_code == 0, result.output
    verify.assert_called_once_with(
        root=Path("example-root"), publication=SourcePin(path="publication.json", sha256="d" * 64),
        output=Path("receipt"),
    )


@pytest.mark.parametrize("command,worker", [
    ("materialize-swing-issuer-reactions", "materialize_issuer_reactions"),
    ("verify-swing-issuer-reactions", "verify_issuer_reaction_rows"),
])
@pytest.mark.parametrize("failure,code", [
    (HeavyJobBusyError("source scan owns workspace lease"), 75),
    (DataReadinessError("qualification authority hash differs"), 2),
])
def test_failures_are_reported_once(
    monkeypatch: pytest.MonkeyPatch, command: str, worker: str, failure: Exception, code: int,
) -> None:
    function = Mock(side_effect=failure)
    monkeypatch.setattr(commands, worker, function)
    arguments = [command, "--output", "output"]
    if worker == "materialize_issuer_reactions":
        arguments += ["--config", "config.json", "--expected-config-sha256", "a" * 64]
    else:
        arguments += ["--publication", "publication.json", "--publication-sha256", "a" * 64]
    result = CliRunner().invoke(_app(), arguments)
    assert result.exit_code == code
    assert str(failure) in result.output
    function.assert_called_once()


@pytest.mark.parametrize("command,worker,status", [
    ("materialize-swing-issuer-reactions", "materialize_issuer_reactions", "partial_research_only"),
    ("verify-swing-issuer-reactions", "verify_issuer_reaction_rows", "failed"),
])
def test_incomplete_outputs_do_not_report_success(
    monkeypatch: pytest.MonkeyPatch, command: str, worker: str, status: str,
) -> None:
    function = Mock(return_value=_report(status))
    monkeypatch.setattr(commands, worker, function)
    arguments = [command, "--output", "output"]
    if worker == "materialize_issuer_reactions":
        arguments += ["--config", "config.json", "--expected-config-sha256", "a" * 64]
    else:
        arguments += ["--publication", "publication.json", "--publication-sha256", "a" * 64]
    result = CliRunner().invoke(_app(), arguments)
    assert result.exit_code == 2
    assert json.loads(result.output)["status"] == status
