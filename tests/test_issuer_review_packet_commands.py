"""UNIT command forwarding; mocked replies are not source or model evidence."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
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


def test_publish_forwards_exact_config_and_does_not_claim_qualification(monkeypatch: pytest.MonkeyPatch) -> None:
    report = {"status": "complete_blind_review_packets_only", "counts": {"samples": 2},
              "manifest_sha256": "a" * 64, "qualification_established": False, "training_eligible": False,
              "serving_eligible": False, "promotion_eligible": False, "economic_eligible": False}
    publish = Mock(return_value=report)
    monkeypatch.setattr(commands, "publish_issuer_review_packets", publish)
    result = CliRunner().invoke(_app(), ["publish-issuer-review-packets", "--root", "unit-root",
        "--config", "unit-config.json", "--expected-config-sha256", "b" * 64, "--output", "unit-output"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == report
    publish.assert_called_once_with(root=Path("unit-root"), config=Path("unit-config.json"),
                                   expected_config_sha256="b" * 64, output=Path("unit-output"))


def test_verify_forwards_independent_manifest_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    pin = SourcePin(path="unit-packets/_manifest.json", sha256="a" * 64)
    verify = Mock(return_value=SimpleNamespace(publication=pin, counts={"samples": 2}))
    monkeypatch.setattr(commands, "verify_issuer_review_packets", verify)
    result = CliRunner().invoke(_app(), ["verify-issuer-review-packets", "--root", "unit-root",
        "--publication", pin.path, "--publication-sha256", pin.sha256])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["status"] == "passed_blind_review_packet_replay_only"
    assert report["manifest_sha256"] == pin.sha256
    assert all(report[field] is False for field in ("qualification_established", "training_eligible",
        "serving_eligible", "promotion_eligible", "economic_eligible"))
    verify.assert_called_once_with(root=Path("unit-root"), publication=pin)


@pytest.mark.parametrize("action,worker,arguments", [
    ("publish-issuer-review-packets", "publish_issuer_review_packets", [
        "--config", "unit-config.json", "--expected-config-sha256", "a" * 64, "--output", "unit-output"]),
    ("verify-issuer-review-packets", "verify_issuer_review_packets", [
        "--publication", "unit-packets/_manifest.json", "--publication-sha256", "a" * 64]),
])
@pytest.mark.parametrize("failure,code", [
    (HeavyJobBusyError("another unit job owns the lease"), 75),
    (DataReadinessError("unit source pin changed"), 2),
    (ValueError("unit configuration fields differ"), 2),
])
def test_packet_failures_are_reported_without_success(
    monkeypatch: pytest.MonkeyPatch, action: str, worker: str, arguments: list[str], failure: Exception, code: int,
) -> None:
    call = Mock(side_effect=failure)
    monkeypatch.setattr(commands, worker, call)
    result = CliRunner().invoke(_app(), [action, *arguments])
    assert result.exit_code == code
    assert str(failure) in result.output
    call.assert_called_once()
