"""Synthetic retained transport reconstruction; no provider calls or numerical targets."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.datasets import corporate_action_collection as collector
from market_predictor.swing.datasets import corporate_action_reconstruction as owner
from market_predictor.swing.datasets.action_evidence import load_corporate_action_evidence
from tests.test_swing_corporate_action_collection import (
    _attempt,
    _page,
    _read,
    _run,
    _write,
)
from tests.test_swing_corporate_action_collection import (
    inventory as inventory,
)


@pytest.fixture
def reconstruction(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    def fetch(params: dict[str, Any], maximum: int) -> Any:
        if params["symbols"] == "AAA" and "page_token" not in params:
            return _page(params, {"cash_dividends": [{"id": "synthetic-a"}]}, "opaque +/ next")
        return _page(params)

    report = _run(inventory, fetch)
    root = inventory["root"]
    config = root / "current.toml"
    config.write_bytes(inventory["config"].read_bytes())
    implementation = root / "synthetic_implementation.py"
    implementation.write_text("# pinned synthetic producer implementation\n", encoding="utf-8")
    monkeypatch.setattr(owner, "_guard", lambda: None)
    monkeypatch.setattr(owner, "assert_peak_memory_budget", lambda **kwargs: None)
    monkeypatch.setattr(owner, "heavy_job_runtime_dir", lambda: root / "runtime")
    monkeypatch.setattr(owner, "_implementation", lambda root, request:
        {implementation.relative_to(root).as_posix(): file_sha256(implementation)})
    original = inventory["output"]
    arguments = dict(root=root, original_archive=original, original_request_sha256=file_sha256(original / "_request.json"),
        original_audit_sha256=report["audit_sha256"], config=config, expected_config_sha256=file_sha256(config),
        output=root / "canonical")
    return dict(arguments=arguments, original=original, inventory=inventory, report=report,
        original_files={path: path.read_bytes() for path in original.rglob("*") if path.is_file()},
        stage=root / ".canonical.reconstructing", config=config)


def _reconstruct(fixture: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return owner.reconstruct_corporate_actions(**{**fixture["arguments"], **kwargs})


def _resume(fixture: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    directory = fixture["arguments"]["output"] if fixture["arguments"]["output"].exists() else fixture["stage"]
    return _reconstruct(fixture, expected_checkpoint_sha256=file_sha256(directory / "_checkpoint.json"), **kwargs)


def test_retained_bytes_clocks_and_current_offline_consumers_round_trip(reconstruction: dict[str, Any]) -> None:
    result = _reconstruct(reconstruction)
    output = reconstruction["arguments"]["output"]
    assert result["status"] == "collected_unreviewed" and result["acquired_tickers"] == result["requested_tickers"] == 2
    assert output.is_dir() and not reconstruction["stage"].exists()
    assert result["audit_sha256"] != reconstruction["report"]["audit_sha256"]
    for name in ("accounting_eligible", "ownership_admitted", "historical_announcement_availability_proven", "absence_of_actions_proven"):
        assert result[name] is False
    for path, body in reconstruction["original_files"].items():
        assert path.read_bytes() == body
        relative = path.relative_to(reconstruction["original"])
        if relative.parts[0] == "tickers":
            if path.name == "receipt.json":
                original, fresh = json.loads(body), _read(output / relative)
                for key in ("pages", "ticker", "attempt_id", "started_at_utc", "completed_at_utc", "state", "error_type"):
                    assert fresh[key] == original[key]
                assert fresh["request_sha256"] != original["request_sha256"]
            else:
                assert (output / relative).read_bytes() == body
    proof = _read(output / "_reconstruction.json")
    assert proof["original_request"]["sha256"] == reconstruction["arguments"]["original_request_sha256"]
    evidence = load_corporate_action_evidence(root=reconstruction["arguments"]["root"], config=reconstruction["config"],
        archive=output, expected_audit_sha256=result["audit_sha256"])
    assert set(proof["source_files"]).issubset(evidence.source_files)
    assert "canonical/_reconstruction.json" in evidence.source_files
    assert _resume(reconstruction) == result


def test_private_partial_resume_requires_independent_checkpoint(reconstruction: dict[str, Any]) -> None:
    partial = _reconstruct(reconstruction, maximum_tickers=1)
    assert partial["status"] == "partial_in_progress" and partial["completed_tickers"] == 1
    assert reconstruction["stage"].is_dir() and not reconstruction["arguments"]["output"].exists()
    with pytest.raises(DataReadinessError, match="independent checkpoint"):
        _reconstruct(reconstruction)
    assert _resume(reconstruction)["status"] == "collected_unreviewed"


@pytest.mark.parametrize("field", ["original_request_sha256", "original_audit_sha256", "expected_config_sha256"])
def test_independent_input_pins_required(reconstruction: dict[str, Any], field: str) -> None:
    with pytest.raises((DataReadinessError, FileNotFoundError)):
        _reconstruct(reconstruction, **{field: "0" * 64})
    assert not reconstruction["arguments"]["output"].exists()


@pytest.mark.parametrize("filename", ["000.bin", "000.json", "receipt.json"])
def test_original_transport_tamper_refused(reconstruction: dict[str, Any], filename: str) -> None:
    path = _attempt(reconstruction["inventory"]) / filename
    path.write_bytes(b"tampered")
    with pytest.raises(DataReadinessError):
        _reconstruct(reconstruction)
    assert not reconstruction["arguments"]["output"].exists()


def test_rehashed_current_query_change_cannot_reuse_original_transport(reconstruction: dict[str, Any]) -> None:
    config = reconstruction["config"]
    config.write_text(config.read_text().replace("page_limit = 10", "page_limit = 11"), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="queries differ"):
        _reconstruct(reconstruction, expected_config_sha256=file_sha256(config))


def test_interrupted_copy_resumes_without_replacing_originals(reconstruction: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    copy = owner.shutil.copyfile
    count = 0

    def interrupted(source: Path, destination: Path) -> Any:
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("synthetic copy interruption")
        return copy(source, destination)

    monkeypatch.setattr(owner.shutil, "copyfile", interrupted)
    with pytest.raises(OSError, match="copy interruption"):
        _reconstruct(reconstruction)
    assert not reconstruction["arguments"]["output"].exists()
    assert (reconstruction["stage"] / "_checkpoint.json").exists()
    monkeypatch.setattr(owner.shutil, "copyfile", copy)
    assert _resume(reconstruction)["status"] == "collected_unreviewed"


@pytest.mark.parametrize("target", ["source", "output", "proof", "current_config"])
def test_mutation_after_normal_replay_prevents_publication(reconstruction: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    replay = collector.collect_holding_corporate_actions

    def mutate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = replay(*args, **kwargs)
        directory = reconstruction["stage"]
        if target == "source":
            path = _attempt(reconstruction["inventory"]) / "000.bin"
        elif target == "output":
            path = next(directory.glob("tickers/*/*/000.bin"))
        elif target == "current_config":
            path = reconstruction["config"]
        else:
            path = directory / "_reconstruction.json"
        path.write_bytes(b"changed between replay and publication")
        return result

    monkeypatch.setattr(collector, "collect_holding_corporate_actions", mutate)
    with pytest.raises(DataReadinessError):
        _reconstruct(reconstruction)
    assert not reconstruction["arguments"]["output"].exists()


def test_failed_normal_offline_replay_never_publishes(reconstruction: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise DataReadinessError("synthetic normal reader failure")

    monkeypatch.setattr(collector, "collect_holding_corporate_actions", fail)
    with pytest.raises(DataReadinessError, match="normal reader failure"):
        _reconstruct(reconstruction)
    assert not reconstruction["arguments"]["output"].exists()


def test_original_clock_poison_rejected_even_with_new_audit_pin(reconstruction: dict[str, Any]) -> None:
    attempt = _attempt(reconstruction["inventory"])
    receipt = _read(attempt / "receipt.json")
    receipt["started_at_utc"] = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    receipt["completed_at_utc"] = receipt["started_at_utc"]
    receipt["receipt_sha256"] = json_sha256({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    _write(attempt / "receipt.json", receipt)
    report = json.loads(json.dumps(reconstruction["report"]))
    report["tickers"][0]["attempts"][0]["receipt_sha256"] = file_sha256(attempt / "receipt.json")
    report["audit_sha256"] = json_sha256({key: value for key, value in report.items() if key != "audit_sha256"})
    _write(reconstruction["original"] / "reports" / f"{report['audit_sha256']}.json", report)
    with pytest.raises(DataReadinessError, match="clock"):
        _reconstruct(reconstruction, original_audit_sha256=report["audit_sha256"])


def test_duplicate_successful_original_attempt_refused(reconstruction: dict[str, Any]) -> None:
    import shutil

    attempt = _attempt(reconstruction["inventory"])
    duplicate = attempt.with_name("duplicate")
    shutil.copytree(attempt, duplicate)
    receipt = _read(duplicate / "receipt.json")
    receipt["attempt_id"] = "duplicate"
    receipt["receipt_sha256"] = json_sha256({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    _write(duplicate / "receipt.json", receipt)
    with pytest.raises(DataReadinessError, match="duplicate successful"):
        _reconstruct(reconstruction)
