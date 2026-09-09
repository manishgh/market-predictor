from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from market_predictor.commands import swing_symbol_corrections as commands
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[typer.Typer, list[str], list[dict[str, Any]]]:
    policy = SimpleNamespace(parent_plan="parent/plan", parent_archive="parent/raw", document_archive="docs/raw",
        document_inventory="config/docs.toml", parent_plan_sha256="a" * 64,
        corrections=[SimpleNamespace(ticker="OLD", provider_symbol="NEW")])
    monkeypatch.setattr(commands, "load_symbol_correction_policy", lambda *a: policy)
    monkeypatch.setattr(commands, "pinned_object", lambda *a: {"request_sha256": "d" * 64,
        "policy_sha256": "b" * 64, "source_root": str(tmp_path), "policy_path": "config/policy.toml"})

    @contextmanager
    def lease(*args: Any, **kwargs: Any) -> Any:
        yield {}

    monkeypatch.setattr(commands, "verified_initial_fit_raw_share_plan", lease)
    app = typer.Typer()
    output: list[dict[str, Any]] = []
    commands.register_symbol_correction_commands(app, SimpleNamespace(print=output.append))
    # Keep a group even though this test surface registers only one command.
    app.callback()(lambda: None)
    args = ["collect-swing-symbol-corrections", "--root", str(tmp_path), "--config", "config/policy.toml",
        "--expected-policy-sha256", "b" * 64, "--plan-dir", "corrections/plan"]
    return app, args, output


def test_offline_never_initializes_alpaca(cli: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    app, args, output = cli
    monkeypatch.setattr(commands, "get_settings", lambda: pytest.fail("offline opened live settings"))
    monkeypatch.setattr(commands, "load_complete_swing_history_collection", lambda *a, **kw: {"status": "complete"})
    result = CliRunner().invoke(app, [*args, "--out-dir", "corrections/raw", "--expected-plan-sha256", "c" * 64, "--offline"])
    assert result.exit_code == 0, result.exception
    assert output == [{"status": "complete"}]


def test_busy_lease_reports_non_queueing_exit(cli: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    app, args, _ = cli

    def busy(*a: Any, **kw: Any) -> Any:
        raise HeavyJobBusyError("held by another test job")

    monkeypatch.setattr(commands, "verified_initial_fit_raw_share_plan", busy)
    for name in ("pinned_object", "get_settings", "publish_symbol_correction_plan", "load_complete_swing_history_collection"):
        monkeypatch.setattr(commands, name, lambda *a, **kw: pytest.fail("busy lease performed downstream work"))
    result = CliRunner().invoke(app, [*args, "--publish-plan"])
    assert result.exit_code == HEAVY_JOB_BUSY_EXIT_CODE


def test_policy_is_rechecked_after_bootstrap(cli: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    app, args, _ = cli
    original = commands.load_symbol_correction_policy
    calls = 0

    def policy(*a: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise commands.DataReadinessError("policy file changed after bootstrap")
        return original(*a)

    monkeypatch.setattr(commands, "load_symbol_correction_policy", policy)
    monkeypatch.setattr(commands, "publish_symbol_correction_plan", lambda *a: pytest.fail("changed policy reached publication"))
    result = CliRunner().invoke(app, [*args, "--publish-plan"])
    assert result.exit_code != 0 and calls == 2


@pytest.mark.parametrize("change_after_replay", [False, True])
def test_offline_supplied_archive_pin_is_enforced(cli: Any, monkeypatch: pytest.MonkeyPatch, change_after_replay: bool) -> None:
    app, args, _ = cli
    original = commands.pinned_object
    checks = 0
    replays = 0

    def checked(path: Path, *a: Any) -> Any:
        nonlocal checks
        if path.parent.name == "raw":
            checks += 1
            if not change_after_replay or checks == 2:
                raise commands.DataReadinessError("archive authority pin differs")
        return original(path, *a)

    def replay(*a: Any, **kw: Any) -> Any:
        nonlocal replays
        replays += 1
        return {"status": "complete"}

    monkeypatch.setattr(commands, "pinned_object", checked)
    monkeypatch.setattr(commands, "load_complete_swing_history_collection", replay)
    result = CliRunner().invoke(app, [*args, "--out-dir", "corrections/raw", "--expected-plan-sha256", "c" * 64,
        "--offline", "--expected-archive-sha256", "f" * 64])
    assert result.exit_code != 0
    assert replays == int(change_after_replay)


def test_live_rejects_archive_pin_before_provider_initialization(cli: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    app, args, _ = cli
    monkeypatch.setattr(commands, "get_settings", lambda: pytest.fail("invalid mode initialized provider"))
    result = CliRunner().invoke(app, [*args, "--out-dir", "corrections/raw", "--expected-plan-sha256", "c" * 64,
        "--expected-archive-sha256", "f" * 64])
    assert result.exit_code != 0


def test_offline_rejects_different_policy_than_cli(cli: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    app, args, _ = cli
    monkeypatch.setattr(commands, "pinned_object", lambda *a: {"request_sha256": "d" * 64, "policy_sha256": "e" * 64})
    result = CliRunner().invoke(app, [*args, "--out-dir", "corrections/raw", "--expected-plan-sha256", "c" * 64, "--offline"])
    assert result.exit_code != 0


@pytest.mark.parametrize("flags", [[], ["--offline"], ["--publish-plan", "--offline"],
    ["--publish-plan", "--out-dir", "corrections/raw"],
    ["--out-dir", "corrections/raw", "--expected-plan-sha256", "c" * 64, "--selection-file", "selection.json"]])
def test_cli_rejects_incomplete_or_conflicting_modes(cli: Any, flags: list[str]) -> None:
    app, args, _ = cli
    result = CliRunner().invoke(app, [*args, *flags])
    assert result.exit_code != 0


def test_output_cannot_contain_parent_evidence(cli: Any) -> None:
    app, args, _ = cli
    args[-1] = "parent"
    result = CliRunner().invoke(app, [*args, "--publish-plan"])
    assert result.exit_code != 0
