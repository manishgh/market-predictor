from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from market_predictor.canonical.store import file_sha256
from market_predictor.commands import swing_corrected_outcomes as commands
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.implementation_snapshot import create_implementation_snapshot
from market_predictor.heavy_jobs import heavy_job_lease
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets import corrected_outcomes, outcome_replay
from tests.test_swing_corrected_outcomes import publication as publication
from tests.test_swing_corrected_outcomes import run_publication


@pytest.fixture
def migration(publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = publication["root"]
    path = root / "src/market_predictor/swing/datasets/fixture.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"# Synthetic historical implementation.\r\n")
    publication["implementation_paths"] = [path.relative_to(root).as_posix()]
    result = run_publication(publication)
    snapshot = create_implementation_snapshot(root, {path.relative_to(root).as_posix(): file_sha256(path)},
        root / "data/evidence/fixture")
    path.write_bytes(b"# Synthetic replacement implementation.\n")
    monkeypatch.setattr(outcome_replay, "__file__", str(path))
    monkeypatch.setattr(corrected_outcomes, "_implementation", lambda _: {path.relative_to(root).as_posix(): file_sha256(path)})
    monkeypatch.setattr(outcome_replay, "load_corporate_action_evidence", lambda **kw: publication["evidence"])

    @contextmanager
    def sources(*args: Any) -> Any:
        with heavy_job_lease("test-outcome-migration", runtime_dir=root / "runtime"):
            yield publication["source"]

    monkeypatch.setattr(corrected_outcomes, "verified_corrected_research_sources", sources)
    return dict(publication, result=result, snapshot=snapshot, implementation=path)


def replay(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return outcome_replay.replay_corrected_outcome_publication(root=state["root"],
        config=SourcePin(path=str(state["config"]), sha256=state["pin"]),
        publication=SourcePin(path=str(state["output"] / "_manifest.json"), sha256=state["result"]["manifest_sha256"]),
        implementation_snapshot=SourcePin(path=state["snapshot"]["manifest_path"], sha256=state["snapshot"]["manifest_sha256"]),
        output=state["root"] / "data/reports/replay", **kwargs)


def test_exact_replay_preserves_original_publication(migration: dict[str, Any]) -> None:
    old = {p.relative_to(migration["output"]): p.read_bytes() for p in migration["output"].rglob("*") if p.is_file()}
    with pytest.raises(DataReadinessError, match="identity changed"):
        run_publication(migration, expected_output_sha256=migration["result"]["manifest_sha256"], replay=True)
    report = replay(migration)
    assert report["status"] == "exact_replay_complete"
    assert report["replay_complete"] is True
    assert report["training_eligible"] is report["promotion_eligible"] is False
    assert len(report["compared_months"]) == 2
    assert {p.relative_to(migration["output"]): p.read_bytes() for p in migration["output"].rglob("*") if p.is_file()} == old


def test_partial_replay_does_not_authorize_completion(migration: dict[str, Any]) -> None:
    report = replay(migration, maximum_months=1)
    assert report["replay_complete"] is False
    assert not (migration["root"] / "data/reports/replay/_manifest.json").exists()


def test_recomputed_mismatch_rejects(migration: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = corrected_outcomes._month

    def poisoned(*args: Any) -> dict[str, Any]:
        result = original(*args)
        return {**result, "rows": result["rows"] + 1}

    monkeypatch.setattr(corrected_outcomes, "_month", poisoned)
    with pytest.raises(DataReadinessError, match="differs from published"):
        replay(migration)
    assert not (migration["root"] / "data/reports/replay/_manifest.json").exists()


def test_changed_cohort_rejects(migration: dict[str, Any]) -> None:
    migration["source"]["request"]["cohort_sha256"] = "e" * 64
    with pytest.raises(DataReadinessError, match="cohort"):
        replay(migration)


@pytest.mark.parametrize("change", ["retrieval_clock", "reference", "policy"])
def test_simulation_retrieval_time_is_not_a_policy_override(
    migration: dict[str, Any], monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    loader = outcome_replay.load_trade_simulation_context

    def fresh(*args: Any, **kwargs: Any) -> Any:
        context = loader(*args, **kwargs)
        reference = context.policy_reference
        if change == "retrieval_clock":
            return context.model_copy(update={"policy_reference": reference.model_copy(
                update={"retrieved_at": reference.retrieved_at + timedelta(seconds=1)})})
        if change == "reference":
            return context.model_copy(update={"policy_reference": reference.model_copy(
                update={"record_locator": "different assumption source"})})
        return context.model_copy(update={"policy": context.policy.model_copy(update={"cost_policy": "altered"})})

    monkeypatch.setattr(outcome_replay, "load_trade_simulation_context", fresh)
    if change == "retrieval_clock":
        assert replay(migration)["replay_complete"] is True
    else:
        with pytest.raises(DataReadinessError, match="simulation differs"):
            replay(migration)


@pytest.mark.parametrize("fail_at_exit", [True, False])
def test_source_exit_failure_or_mutation_cannot_publish_success(
    migration: dict[str, Any], monkeypatch: pytest.MonkeyPatch, fail_at_exit: bool,
) -> None:
    @contextmanager
    def sources(*args: Any) -> Any:
        yield migration["source"]
        if fail_at_exit:
            raise DataReadinessError("source failed during context exit")
        migration["implementation"].write_bytes(b"# Changed between contexts.\n")

    original = (migration["output"] / "_manifest.json").read_bytes()
    monkeypatch.setattr(corrected_outcomes, "verified_corrected_research_sources", sources)
    with pytest.raises(DataReadinessError):
        replay(migration)
    assert not (migration["root"] / "data/reports/replay/_manifest.json").exists()
    assert (migration["output"] / "_manifest.json").read_bytes() == original


def test_altered_original_target_rejects(migration: dict[str, Any]) -> None:
    path = next(migration["output"].glob("*/targets.parquet"))
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(DataReadinessError, match="source changed"):
        replay(migration)


@pytest.mark.parametrize("complete,exit_code", [(True, 0), (False, 2)])
def test_cli_reports_partial_replay_as_incomplete(monkeypatch: pytest.MonkeyPatch, complete: bool, exit_code: int) -> None:
    app = typer.Typer()
    commands.register_corrected_outcome_commands(app)
    calls: list[dict[str, Any]] = []

    def run(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return dict(status="exact_replay_complete" if complete else "partial", replay_complete=complete,
            training_eligible=False, promotion_eligible=False)

    monkeypatch.setattr(commands, "replay_corrected_outcome_publication", run)
    result = CliRunner().invoke(app, ["replay-swing-corrected-outcomes", "--config", "config.toml",
        "--config-sha256", "a" * 64, "--publication", "data/labels/original/_manifest.json",
        "--publication-sha256", "b" * 64, "--implementation-snapshot", "data/evidence/code/_manifest.json",
        "--snapshot-sha256", "c" * 64, "--output", "data/reports/comparison"])
    assert result.exit_code == exit_code, result.output
    assert calls[0]["publication"].sha256 == "b" * 64
