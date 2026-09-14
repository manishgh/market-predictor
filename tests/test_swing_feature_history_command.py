from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from market_predictor.canonical.store import file_sha256
from market_predictor.commands import swing_feature_history as commands
from market_predictor.heavy_jobs import heavy_job_lease
from tests.test_swing_feature_history_plan import evidence as evidence


def _app() -> typer.Typer:
    app = typer.Typer()
    commands.register_feature_history_commands(app)
    return app


def _arguments(evidence: dict[str, Any]) -> list[str]:
    return ["--root", str(evidence["root"]), "--config", str(evidence["config"]),
        "--expected-policy-sha256", evidence["policy_pin"], "--plan-dir", "data/research/adjusted-plan"]


def test_plan_command_uses_shared_lease_before_loading(evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(commands, "_guard", lambda: None)
    owner = evidence["root"] / "data/runtime/heavy-job.owner.json"
    original = commands.feature_history_requirements

    def checked(*args: Any) -> Any:
        assert owner.is_file()
        return original(*args)

    monkeypatch.setattr(commands, "feature_history_requirements", checked)
    result = CliRunner().invoke(_app(), [*_arguments(evidence), "--publish-plan"])
    assert result.exit_code == 0, result.exception
    assert (evidence["root"] / "data/research/adjusted-plan/_authority.json").is_file()
    assert not owner.exists()


def test_competing_job_refuses_before_reading_policy(evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any) -> Any:
        pytest.fail("Policy read happened before lease acquisition")

    monkeypatch.setattr(commands, "feature_history_requirements", forbidden)
    with heavy_job_lease("test-other", runtime_dir=evidence["root"] / "data/runtime"):
        result = CliRunner().invoke(_app(), [*_arguments(evidence), "--publish-plan"])
    assert result.exit_code == 75


def test_source_is_closed_on_collection_error_and_units_are_sequential(
    evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(commands, "_guard", lambda: None)
    app, runner = _app(), CliRunner()
    assert runner.invoke(app, [*_arguments(evidence), "--publish-plan"]).exit_code == 0
    plan = evidence["root"] / "data/research/adjusted-plan"
    closed = []
    source = SimpleNamespace(settings=SimpleNamespace(alpaca_stock_feed="sip"),
        client=SimpleNamespace(session=SimpleNamespace(close=lambda: closed.append(True))))
    monkeypatch.setattr(commands, "get_settings", lambda: SimpleNamespace(has_alpaca=True, alpaca_stock_feed="sip"))
    monkeypatch.setattr(commands, "AlpacaSource", lambda settings: source)

    def fail(**kwargs: Any) -> Any:
        assert kwargs["maximum_units_this_run"] == 1
        kwargs["source_factory"]()
        raise RuntimeError("test provider failure")

    monkeypatch.setattr(commands, "collect_swing_history_plan", fail)
    result = runner.invoke(app, [*_arguments(evidence), "--out-dir", "data/raw/corrected-adjusted",
        "--expected-plan-sha256", file_sha256(plan / "_authority.json")])
    assert result.exit_code == 1 and "test provider failure" in str(result.exception)
    assert closed == [True]


@pytest.mark.parametrize("extra", [["--publish-plan", "--offline"], [], ["--publish-plan", "--out-dir", "data/raw/x"]])
def test_invalid_modes_rejected(evidence: dict[str, Any], extra: list[str]) -> None:
    result = CliRunner().invoke(_app(), [*_arguments(evidence), *extra])
    assert result.exit_code == 2


def test_guarded_source_checks_memory_before_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def full() -> None:
        raise MemoryError("test memory pressure")

    monkeypatch.setattr(commands, "_guard", full)
    source = commands._GuardedDailySource(SimpleNamespace(settings=SimpleNamespace(alpaca_stock_feed="sip")))
    from datetime import UTC, date, datetime

    with pytest.raises(MemoryError, match="memory pressure"):
        source.fetch_daily_page("SATS", datetime(2019, 1, 1, tzinfo=UTC), datetime(2020, 1, 1, tzinfo=UTC),
            page_token=None, asof=date(2020, 1, 1), adjustment="all")
