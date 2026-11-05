"""UNIT checks for the saved-outcome audit command boundary."""
from pathlib import Path
from unittest.mock import Mock

import pytest
import typer
from click.testing import Result
from typer.testing import CliRunner

from market_predictor.commands import swing_return_training as commands
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.heavy_jobs import HeavyJobBusyError


def invoke() -> Result:
    app = typer.Typer()
    commands.register_swing_return_training_command(app)
    return CliRunner().invoke(app, ["audit-saved-swing-outcomes", "--root", "source",
        "--directory", "data/research/run", "--manifest-sha256", "a" * 64,
        "--output", "reports/coverage.json", "--family", "shallow_boosted_return", "--scope", "security_transfer"])


def test_audit_command_keeps_explicit_scope_and_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    audit = Mock(return_value={"status": "diagnostic_complete", "report_sha256": "b" * 64})
    monkeypatch.setattr(commands, "audit_saved_outcomes", audit)
    result = invoke()
    assert result.exit_code == 0, result.exception
    audit.assert_called_once_with(root=Path("source"), directory=Path("data/research/run"),
        manifest_sha256="a" * 64, output=Path("reports/coverage.json"),
        family="shallow_boosted_return", scope="security_transfer")


@pytest.mark.parametrize(("error", "exit_code"), [
    (HeavyJobBusyError("busy"), 75), (DataReadinessError("changed evidence"), 2),
    (MemoryBudgetError("memory limit"), 2), (FileExistsError("existing report"), 2),
])
def test_audit_command_reports_failure_without_retry(monkeypatch: pytest.MonkeyPatch, error: Exception, exit_code: int) -> None:
    audit = Mock(side_effect=error)
    monkeypatch.setattr(commands, "audit_saved_outcomes", audit)
    result = invoke()
    assert result.exit_code == exit_code
    assert str(error) in result.output
    audit.assert_called_once()
