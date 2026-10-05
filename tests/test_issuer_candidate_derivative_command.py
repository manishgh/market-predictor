"""Unit-only CLI failure boundaries; no claim to run actual saved-source replay."""
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from market_predictor.commands import issuer_content_inventory as commands
from market_predictor.core.errors import DataReadinessError
from market_predictor.heavy_jobs import HeavyJobBusyError
from tests.test_issuer_content_qualification_command import _app


@pytest.mark.parametrize("command,owner,arguments", [
    ("derive-saved-issuer-candidates", "publish_issuer_candidate_derivative", [
        "--parent-population", "data/research/parent/_manifest.json", "--parent-population-sha256", "a" * 64,
        "--output", "data/research/derivative",
    ]),
    ("verify-saved-issuer-candidates", "verify_issuer_candidate_derivative", [
        "--publication", "data/research/derivative/_manifest.json", "--publication-sha256", "b" * 64,
    ]),
])
@pytest.mark.parametrize("failure,exit_code", [
    (HeavyJobBusyError("another actual source job owns the lease"), 75),
    (DataReadinessError("saved candidate differs from source-text replay"), 2),
])
def test_candidate_command_reports_failure_without_claiming_completion(
    monkeypatch: pytest.MonkeyPatch, command: str, owner: str, arguments: list[str],
    failure: Exception, exit_code: int,
) -> None:
    operation = Mock(side_effect=failure)
    monkeypatch.setattr(commands, owner, operation)
    result = CliRunner().invoke(_app(), [command, "--root", "example-root", *arguments])
    assert result.exit_code == exit_code
    assert str(failure) in result.output
    assert "passed_candidate_replay_only" not in result.output
    operation.assert_called_once()
    assert operation.call_args.kwargs["root"] == Path("example-root")


@pytest.mark.parametrize("command,owner,arguments", [
    ("derive-saved-issuer-candidates", "publish_issuer_candidate_derivative", [
        "--parent-population", "data/research/parent/_manifest.json", "--output", "data/research/derivative",
    ]),
    ("verify-saved-issuer-candidates", "verify_issuer_candidate_derivative", [
        "--publication", "data/research/derivative/_manifest.json",
    ]),
])
def test_candidate_command_requires_independent_input_hash(
    monkeypatch: pytest.MonkeyPatch, command: str, owner: str, arguments: list[str],
) -> None:
    operation = Mock()
    monkeypatch.setattr(commands, owner, operation)
    result = CliRunner().invoke(_app(), [command, *arguments])
    assert result.exit_code == 2
    operation.assert_not_called()
