from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from market_predictor.commands import swing_return_relationships as commands
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.heavy_jobs import HeavyJobBusyError


def app() -> typer.Typer:
    result = typer.Typer()
    commands.register_return_relationship_commands(result)
    return result


@pytest.mark.parametrize("complete", [True, False])
def test_materializer_passes_pins_and_returns_partial_status(monkeypatch: pytest.MonkeyPatch, complete: bool) -> None:
    calls: list[dict[str, Any]] = []

    def run(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "complete_research_only" if complete else "partial_in_progress",
            "rows": 5, "checkpoint_sha256": "c" * 64, "training_eligible": False,
            "promotion_eligible": False, "serving_eligible": False}

    monkeypatch.setattr(commands, "materialize_return_relationships", run)
    result = CliRunner().invoke(app(), ["materialize-swing-return-relationships", "--config", "source.json",
        "--expected-config-sha256", "a" * 64, "--output", "data/features/new", "--root", "root",
        "--expected-checkpoint-sha256", "b" * 64, "--maximum-groups-this-run", "2"])
    assert result.exit_code == (0 if complete else 2), result.exception
    assert calls == [dict(root=Path("root"), config=Path("source.json"), expected_config_sha256="a" * 64,
        output=Path("data/features/new"), expected_checkpoint_sha256="b" * 64, maximum_groups_this_run=2)]
    assert json.loads(result.stdout)["training_eligible"] is False


def test_verifier_passes_independent_publication_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def run(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return dict(status="passed", manifest_sha256="a" * 64, rows=5, report_sha256="b" * 64,
            training_eligible=False, promotion_eligible=False, serving_eligible=False)

    monkeypatch.setattr(commands, "verify_return_relationship_rows", run)
    result = CliRunner().invoke(app(), ["verify-swing-return-relationships", "--publication", "new/_manifest.json",
        "--publication-sha256", "a" * 64, "--output", "data/reports/check.json"])
    assert result.exit_code == 0, result.exception
    assert calls == [dict(root=Path("."), publication=commands.SourcePin(path="new/_manifest.json", sha256="a" * 64),
        output=Path("data/reports/check.json"))]
    assert json.loads(result.stdout)["serving_eligible"] is False


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize(("error", "exit_code"), [(HeavyJobBusyError("busy"), 75),
    (DataReadinessError("invalid source"), 2), (MemoryBudgetError("memory limit"), 2),
    (FileExistsError("immutable output exists"), 2)])
def test_commands_report_expected_failure(monkeypatch: pytest.MonkeyPatch, verify: bool,
    error: Exception, exit_code: int,
) -> None:
    def fail(**kwargs: Any) -> dict[str, Any]:
        raise error

    name = "verify_return_relationship_rows" if verify else "materialize_return_relationships"
    monkeypatch.setattr(commands, name, fail)
    args = ["verify-swing-return-relationships", "--publication", "manifest.json", "--publication-sha256", "a" * 64] \
        if verify else ["materialize-swing-return-relationships", "--config", "source.json", "--expected-config-sha256", "a" * 64]
    result = CliRunner().invoke(app(), [*args, "--output", "data/output"])
    assert result.exit_code == exit_code, result.exception
    assert str(error) in result.output


def test_nonpositive_group_limit_does_not_call_materializer(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("invalid CLI input reached materializer")

    monkeypatch.setattr(commands, "materialize_return_relationships", unexpected)
    result = CliRunner().invoke(app(), ["materialize-swing-return-relationships", "--config", "source.json",
        "--expected-config-sha256", "a" * 64, "--output", "data/output", "--maximum-groups-this-run", "0"])
    assert result.exit_code == 2
