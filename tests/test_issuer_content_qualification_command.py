"""CLI orchestration fixtures; actual content qualification is tested separately."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import typer
from typer.testing import CliRunner

from market_predictor.commands import issuer_content_inventory as commands
from market_predictor.core.errors import DataReadinessError
from market_predictor.heavy_jobs import HeavyJobBusyError
from market_predictor.swing.contracts.holding_materialization import SourcePin


def _app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def root() -> None:
        pass

    commands.register_issuer_content_commands(app)
    return app


def _arguments() -> list[str]:
    return [
        "qualify-issuer-content", "--root", "example-root",
        "--population-authority", "data/research/population/_manifest.json", "--population-sha256", "a" * 64,
        "--reviewer-one", "data/research/reviewer-one.json", "--reviewer-one-sha256", "b" * 64,
        "--reviewer-two", "data/research/reviewer-two.json", "--reviewer-two-sha256", "c" * 64,
        "--output", "data/research/qualified-content",
    ]


def test_qualification_command_forwards_exact_evidence_and_nonadmission(monkeypatch: pytest.MonkeyPatch) -> None:
    publish = Mock(return_value={
        "status": "complete", "counts": {"qualified_versions": 0}, "research_feature_eligible": False,
        "training_eligible": False, "serving_eligible": False, "manifest_sha256": "d" * 64,
    })
    monkeypatch.setattr(commands, "publish_issuer_content_qualification", publish)
    result = CliRunner().invoke(_app(), _arguments())
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["research_feature_eligible"] is False
    publish.assert_called_once_with(
        root=Path("example-root"),
        population_authority=SourcePin(path="data/research/population/_manifest.json", sha256="a" * 64),
        reviewer_files=(SourcePin(path="data/research/reviewer-one.json", sha256="b" * 64),
                        SourcePin(path="data/research/reviewer-two.json", sha256="c" * 64)),
        output=Path("data/research/qualified-content"),
    )


@pytest.mark.parametrize("failure,exit_code", [
    (HeavyJobBusyError("source worker owns lease"), 75),
    (DataReadinessError("reviewer text differs from pinned source"), 2),
])
def test_qualification_command_reports_exact_failure_once(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, exit_code: int,
) -> None:
    publish = Mock(side_effect=failure)
    monkeypatch.setattr(commands, "publish_issuer_content_qualification", publish)
    result = CliRunner().invoke(_app(), _arguments())
    assert result.exit_code == exit_code
    assert str(failure) in result.output
    publish.assert_called_once()


def test_qualification_command_requires_both_reviewer_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    publish = Mock()
    monkeypatch.setattr(commands, "publish_issuer_content_qualification", publish)
    arguments = _arguments()
    position = arguments.index("--reviewer-two-sha256")
    del arguments[position:position + 2]
    result = CliRunner().invoke(_app(), arguments)
    assert result.exit_code == 2
    publish.assert_not_called()
