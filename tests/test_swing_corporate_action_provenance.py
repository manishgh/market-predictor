"""Synthetic reconstruction chains; original transport bytes are never rewritten."""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.datasets import action_evidence
from market_predictor.swing.datasets import corporate_action_collection as collector
from market_predictor.swing.datasets import corporate_action_provenance as provenance
from tests.test_swing_corporate_action_collection import _empty, _read, _resign, _run, _write
from tests.test_swing_corporate_action_collection import inventory as inventory


def _pin(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}


@pytest.fixture
def reconstructed(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root, original = inventory["root"], inventory["output"]
    original_report = _run(inventory, _empty)
    original_bytes = {path: path.read_bytes() for path in original.rglob("*") if path.is_file()}
    config = root / "current-policy.toml"
    config.write_bytes(inventory["config"].read_bytes())
    request = collector._prepare(root, config)
    assert request["request_sha256"] != original_report["request_sha256"]
    output = root / "reconstructed"
    output.mkdir()
    _write(output / "_request.json", request)
    shutil.copytree(original / "tickers", output / "tickers")
    implementation = root / "reconstructor.py"
    implementation.write_text("# synthetic reconstruction implementation", encoding="ascii")
    proof = {"schema": provenance.SCHEMA, "request_sha256": request["request_sha256"],
        "original_request": _pin(root, original / "_request.json"),
        "original_audit": _pin(root, original / "reports" / f"{original_report['audit_sha256']}.json"),
        "source_files": {path.relative_to(root).as_posix(): file_sha256(path) for path in original_bytes},
        "receipts": {path.relative_to(original).as_posix(): _pin(root, path) for path in original.glob("tickers/*/*/receipt.json")},
        "reconstructed_at_utc": datetime.now(UTC).isoformat(),
        "implementation_files": {implementation.name: file_sha256(implementation)}}
    _write(output / "_reconstruction.json", proof)
    for path in output.glob("tickers/*/*/receipt.json"):
        receipt = _read(path)
        receipt.update(request_sha256=request["request_sha256"],
            reconstruction_proof={"path": "_reconstruction.json", "sha256": file_sha256(output / "_reconstruction.json")},
            original_receipt=proof["receipts"][path.relative_to(output).as_posix()])
        _resign(path, receipt, "receipt_sha256")
    monkeypatch.setattr(action_evidence, "heavy_job_runtime_dir", lambda: Path("runtime"))
    monkeypatch.setattr(action_evidence, "assert_system_memory_available", lambda: None)
    return dict(root=root, original=original, output=output, config=config, request=request, proof=proof,
        original_bytes=original_bytes, implementation=implementation)


def _report(case: dict[str, Any]) -> dict[str, Any]:
    return collector._report(case["output"], case["request"], root=case["root"])


def _rewrite_proof(case: dict[str, Any]) -> None:
    path = case["output"] / "_reconstruction.json"
    _write(path, case["proof"])
    for receipt_path in case["output"].glob("tickers/*/*/receipt.json"):
        receipt = _read(receipt_path)
        receipt["reconstruction_proof"]["sha256"] = file_sha256(path)
        _resign(receipt_path, receipt, "receipt_sha256")


def test_reconstruction_replays_offline_and_propagates_original_input_pins(reconstructed: dict[str, Any]) -> None:
    case = reconstructed
    report = _report(case)
    replay = collector.collect_holding_corporate_actions(case["root"], case["config"], case["output"],
        expected_audit_sha256=report["audit_sha256"])
    assert replay == report
    assert not any(report[name] for name in ("historical_announcement_availability_proven", "absence_of_actions_proven",
        "ownership_admitted", "accounting_eligible"))
    evidence = action_evidence.load_corporate_action_evidence(root=case["root"], config=case["config"], archive=case["output"],
        expected_audit_sha256=report["audit_sha256"])
    expected = {**case["proof"]["source_files"], **case["proof"]["implementation_files"],
        "reconstructed/_reconstruction.json": file_sha256(case["output"] / "_reconstruction.json")}
    assert all(evidence.source_files.get(name) == digest for name, digest in expected.items())
    assert evidence.records_by_symbol == {"AAA": {}, "BBB": {}}
    evidence.recheck(case["root"])
    assert all(path.read_bytes() == value for path, value in case["original_bytes"].items())
    case["implementation"].write_bytes(b"changed after projection")
    with pytest.raises(DataReadinessError, match="changed"):
        evidence.recheck(case["root"])


@pytest.mark.parametrize("poison", ["request", "mapping", "missing_body", "missing_request", "cycle", "clock", "naive_clock"])
def test_proof_substitution_and_incomplete_original_inventory_fail(reconstructed: dict[str, Any], poison: str) -> None:
    case, proof = reconstructed, reconstructed["proof"]
    if poison == "request":
        proof["request_sha256"] = "a" * 64
    elif poison == "mapping":
        names = sorted(proof["receipts"])
        proof["receipts"][names[0]] = proof["receipts"][names[1]]
    elif poison == "missing_body":
        proof["source_files"].pop(next(name for name in proof["source_files"] if name.endswith(".bin")))
    elif poison == "missing_request":
        proof["source_files"].pop(proof["original_request"]["path"])
    elif poison == "cycle":
        proof["source_files"]["reconstructed/_request.json"] = file_sha256(case["output"] / "_request.json")
    elif poison == "clock":
        proof["reconstructed_at_utc"] = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    else:
        proof["reconstructed_at_utc"] = "2024-01-01T00:00:00"
    _rewrite_proof(case)
    with pytest.raises(DataReadinessError):
        _report(case)


@pytest.mark.parametrize("poison", ["proof_hash", "proof_path", "original_pin", "old_clock", "missing_proof", "body", "metadata"])
def test_fresh_receipt_cannot_replace_transport_or_provenance(reconstructed: dict[str, Any], poison: str) -> None:
    case = reconstructed
    path = next(case["output"].glob("tickers/AAA/*/receipt.json"))
    receipt = _read(path)
    if poison == "proof_hash":
        receipt["reconstruction_proof"]["sha256"] = "a" * 64
    elif poison == "proof_path":
        receipt["reconstruction_proof"]["path"] = "another-proof.json"
    elif poison == "original_pin":
        receipt["original_receipt"]["sha256"] = "a" * 64
    elif poison == "old_clock":
        receipt["started_at_utc"] = (datetime.fromisoformat(receipt["started_at_utc"]) - timedelta(seconds=1)).isoformat()
    elif poison == "missing_proof":
        receipt.pop("reconstruction_proof")
        receipt.pop("original_receipt")
    else:
        key = "body_path" if poison == "body" else "metadata_path"
        (path.parent / receipt["pages"][0][key]).write_bytes(b"changed")
    _resign(path, receipt, "receipt_sha256")
    with pytest.raises(DataReadinessError):
        _report(case)


def test_original_source_mutation_after_proof_load_is_rejected(reconstructed: dict[str, Any]) -> None:
    case = reconstructed
    validated = provenance.load_reconstruction_proof(case["root"], case["output"], case["request"])
    assert validated is not None
    original_body = next(case["original"].glob("tickers/*/*/*.bin"))
    original_body.write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="changed"):
        validated.recheck()


def test_normal_request_equality_is_not_relaxed_by_reconstruction(reconstructed: dict[str, Any]) -> None:
    case = reconstructed
    wrong = json.loads(json.dumps(case["request"]))
    wrong["policy"]["process_end"] = "2024-02-01"
    wrong["request_sha256"] = json_sha256({key: value for key, value in wrong.items() if key != "request_sha256"})
    with pytest.raises(DataReadinessError, match="request differs"):
        collector._archived_request(case["output"], wrong)


def test_atomic_rename_preserves_report_identity_and_normal_offline_replay(reconstructed: dict[str, Any]) -> None:
    case = reconstructed
    report = _report(case)
    destination = case["root"] / "published"
    case["output"].rename(destination)
    replay = collector.collect_holding_corporate_actions(case["root"], case["config"], destination,
        expected_audit_sha256=report["audit_sha256"])
    assert replay == report
    evidence = action_evidence.load_corporate_action_evidence(root=case["root"], config=case["config"], archive=destination,
        expected_audit_sha256=report["audit_sha256"])
    assert "published/_reconstruction.json" in evidence.source_files
    assert "reconstructed/_reconstruction.json" not in evidence.source_files
    evidence.recheck(case["root"])
