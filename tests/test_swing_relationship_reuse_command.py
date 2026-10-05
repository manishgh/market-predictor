"""Source-proof command pins and failure behavior, without running data jobs."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import typer
from typer.testing import CliRunner

from market_predictor.commands import swing_return_relationships as commands
from market_predictor.core.errors import DataReadinessError
from market_predictor.heavy_jobs import HeavyJobBusyError


def _app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def root() -> None:
        pass

    commands.register_return_relationship_commands(app)
    return app


def _arguments() -> list[str]:
    return [
        "verify-swing-relationship-reuse", "--root", "example-root", "--config", "reuse.json",
        "--expected-config-sha256", "a" * 64, "--output", "data/reports/reuse.json",
    ]


@pytest.mark.parametrize("status,code,equal", [
    ("passed_exact_reuse", 0, True), ("failed_differences", 2, False),
])
def test_reuse_command_forwards_pins_and_keeps_admission_false(
    monkeypatch: pytest.MonkeyPatch, status: str, code: int, equal: bool,
) -> None:
    verify = Mock(return_value={
        "status": status, "rows": 10, "source_inputs_equal": equal, "additions_equal": equal,
        "inherited_columns_equal": True, "population_equal": True,
        "training_eligible": False, "promotion_eligible": False, "serving_eligible": False,
    })
    monkeypatch.setattr(commands, "verify_relationship_reuse", verify)
    result = CliRunner().invoke(_app(), _arguments())
    assert result.exit_code == code, result.output
    assert json.loads(result.output)["training_eligible"] is False
    assert json.loads(result.output)["source_inputs_equal"] is equal
    verify.assert_called_once_with(
        root=Path("example-root"), config=Path("reuse.json"), expected_config_sha256="a" * 64,
        output=Path("data/reports/reuse.json"),
    )


@pytest.mark.parametrize("failure,code", [
    (HeavyJobBusyError("source scan owns workspace lease"), 75),
    (DataReadinessError("historical sidecar hash differs"), 2),
])
def test_reuse_command_reports_source_failure_once(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, code: int,
) -> None:
    verify = Mock(side_effect=failure)
    monkeypatch.setattr(commands, "verify_relationship_reuse", verify)
    result = CliRunner().invoke(_app(), _arguments())
    assert result.exit_code == code
    assert str(failure) in result.output
    verify.assert_called_once()
