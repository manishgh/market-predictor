from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets import corrected_outcomes, outcome_replay
from tests.test_swing_corrected_outcomes import publication as publication
from tests.test_swing_outcome_replay import migration as migration
from tests.test_swing_outcome_replay import replay


def test_verified_partial_replay_resumes_only_remaining_months(
    migration: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = replay(migration, maximum_months=1)
    directory = migration["root"] / "data/reports/replay"
    checkpoint = directory / "_checkpoint.json"
    assert first["source_checks_complete"] is True
    assert not first["replay_complete"]
    calls: list[Any] = []
    original = corrected_outcomes._month

    def count(*args: Any) -> Any:
        calls.append(args[2].session_date_et.min())
        return original(*args)

    monkeypatch.setattr(corrected_outcomes, "_month", count)
    result = replay(migration, expected_checkpoint_sha256=file_sha256(checkpoint), maximum_months=1)
    assert result["replay_complete"] is True
    assert len(calls) == 1
    assert len(result["compared_months"]) == 2
    assert (directory / "_manifest.json").exists()


def test_resume_requires_matching_external_checkpoint_pin(migration: dict[str, Any]) -> None:
    replay(migration, maximum_months=1)
    with pytest.raises(DataReadinessError):
        replay(migration)
    with pytest.raises(DataReadinessError):
        replay(migration, expected_checkpoint_sha256="f" * 64)


def test_resume_rejects_changed_current_implementation(migration: dict[str, Any]) -> None:
    replay(migration, maximum_months=1)
    checkpoint = migration["root"] / "data/reports/replay/_checkpoint.json"
    pin = file_sha256(checkpoint)
    migration["implementation"].write_bytes(b"# Different replay implementation.\n")
    with pytest.raises(DataReadinessError, match="request|implementation"):
        replay(migration, expected_checkpoint_sha256=pin)


def test_source_exit_failure_preserves_previous_verified_checkpoint(
    migration: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay(migration, maximum_months=1)
    directory = migration["root"] / "data/reports/replay"
    checkpoint = directory / "_checkpoint.json"
    before = checkpoint.read_bytes()

    @contextmanager
    def failing(*args: Any) -> Any:
        yield migration["source"]
        raise DataReadinessError("source exit failed")

    monkeypatch.setattr(corrected_outcomes, "verified_corrected_research_sources", failing)
    with pytest.raises(DataReadinessError, match="source exit"):
        replay(migration, expected_checkpoint_sha256=file_sha256(checkpoint))
    assert checkpoint.read_bytes() == before
    assert not (directory / "_manifest.json").exists()


def test_request_replacement_after_pinned_read_is_rejected(
    migration: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay(migration, maximum_months=1)
    directory = migration["root"] / "data/reports/replay"
    checkpoint = directory / "_checkpoint.json"
    original_checkpoint = checkpoint.read_bytes()
    loader = outcome_replay.pinned_object

    def replace_after_read(path: Any, pin: Any) -> Any:
        result = loader(path, pin)
        if path == directory / "_request.json":
            corrected_outcomes._write(path, {**result, "comparison": "different contract"})
        return result

    monkeypatch.setattr(outcome_replay, "pinned_object", replace_after_read)
    with pytest.raises(DataReadinessError, match="changed"):
        replay(migration, expected_checkpoint_sha256=file_sha256(checkpoint))
    assert checkpoint.read_bytes() == original_checkpoint
    assert not (directory / "_manifest.json").exists()


def test_manifest_write_failure_preserves_resumable_checkpoint(
    migration: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay(migration, maximum_months=1)
    directory = migration["root"] / "data/reports/replay"
    checkpoint = directory / "_checkpoint.json"
    original_checkpoint = checkpoint.read_bytes()
    writer = corrected_outcomes._write

    def fail_manifest(path: Any, value: Any) -> None:
        if path == directory / "_manifest.json":
            raise OSError("simulated failed manifest write")
        writer(path, value)

    monkeypatch.setattr(corrected_outcomes, "_write", fail_manifest)
    with pytest.raises(OSError, match="manifest write"):
        replay(migration, expected_checkpoint_sha256=file_sha256(checkpoint))
    assert checkpoint.read_bytes() == original_checkpoint
    monkeypatch.setattr(corrected_outcomes, "_write", writer)
    assert replay(migration, expected_checkpoint_sha256=file_sha256(checkpoint))["replay_complete"] is True


def verify(state: dict[str, Any]) -> outcome_replay.OutcomeReplayVerification:
    path = state["root"] / "data/reports/replay/_manifest.json"
    return outcome_replay.verify_outcome_replay(root=state["root"],
        publication=SourcePin(path=str(state["output"] / "_manifest.json"), sha256=state["result"]["manifest_sha256"]),
        replay=SourcePin(path=str(path), sha256=file_sha256(path)))


def test_completed_receipt_verifies_snapshot_and_current_code(migration: dict[str, Any]) -> None:
    replay(migration)
    verified = verify(migration)
    path = migration["implementation"].relative_to(migration["root"]).as_posix()
    assert verified.historical_implementation_files[path] != verified.live_evidence_files[path]
    assert any(name.endswith(".bin") for name in verified.live_evidence_files)


@pytest.mark.parametrize("change", ["implementation", "source", "publication", "partial"])
def test_receipt_cannot_hide_changed_or_partial_evidence(migration: dict[str, Any], change: str) -> None:
    replay(migration)
    if change == "implementation":
        migration["implementation"].write_bytes(b"# Changed verifier dependency.\n")
    elif change == "source":
        path = next(migration["output"].glob("*/targets.parquet"))
        path.write_bytes(path.read_bytes() + b"changed")
    elif change == "publication":
        path = migration["output"] / "_manifest.json"
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        path = migration["root"] / "data/reports/replay/_manifest.json"
        receipt = outcome_replay.pinned_object(path, file_sha256(path))
        receipt["replay_complete"] = False
        corrected_outcomes._write(path, receipt)
    with pytest.raises(DataReadinessError):
        verify(migration)
