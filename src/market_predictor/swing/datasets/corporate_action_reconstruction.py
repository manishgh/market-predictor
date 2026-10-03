"""Offline, receipt-bound reconstruction of retained corporate-action transport."""
from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.locking import file_lock
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget
from market_predictor.swing.datasets import corporate_action_collection as collector
from market_predictor.swing.datasets.corporate_action_provenance import ValidatedReconstruction, load_reconstruction_proof

_CHECKPOINT_SCHEMA = "market_predictor.corporate_action_reconstruction_checkpoint"
_FLAGS = dict(historical_announcement_availability_proven=False, absence_of_actions_proven=False,
    ownership_admitted=False, accounting_eligible=False)


def _guard() -> None:
    assert_memory_budget(stage="corporate-action reconstruction", hard_budget_gib=5.0, headroom_gib=0.75)
    assert_system_memory_available()


def _check(root: Path, pins: dict[str, str]) -> None:
    collector._verify_sources(root, pins)


def _replace(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    write_json_object(temporary, value)
    temporary.replace(path)


def _put(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if collector._object(path) != value:
            raise DataReadinessError("uncheckpointed reconstruction metadata differs")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_object(path, value)


def _implementation(root: Path, request: dict[str, Any]) -> dict[str, str]:
    package = Path(collector.__file__).resolve().parents[2]
    pins = {(package / name).relative_to(root).as_posix(): digest
        for name, digest in request["implementation_files"].items()}
    for path in (Path(__file__).resolve(), Path(__file__).with_name("corporate_action_provenance.py").resolve()):
        pins[path.relative_to(root).as_posix()] = file_sha256(path)
    return pins


def _original(root: Path, source: Path, request_pin: str, audit_pin: str,
    current: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, str]]]:
    """Historical request interpretation is confined to this explicit importer."""
    request_path = source / "_request.json"
    _check(root, {request_path.relative_to(root).as_posix(): request_pin})
    request = collector._object(request_path)
    if request.get("request_sha256") != json_sha256({key: value for key, value in request.items() if key != "request_sha256"}):
        raise DataReadinessError("original corporate-action request semantic identity differs")
    if request.get("tickers") != current["tickers"] or not request["tickers"]:
        raise DataReadinessError("original and current corporate-action query populations differ")
    for ticker in current["tickers"]:
        if collector._params(request, ticker, None) != collector._params(current, ticker, None):
            raise DataReadinessError("original and current corporate-action queries differ")
    for name in ("numeric_end", "maximum_pages_per_ticker", "maximum_response_bytes"):
        if request["policy"].get(name) != current["policy"].get(name):
            raise DataReadinessError("original corporate-action query bounds or limits differ")
    report_path = source / "reports" / f"{audit_pin}.json"
    report = collector._object(report_path)
    if (report.get("audit_sha256") != audit_pin
            or json_sha256({key: value for key, value in report.items() if key != "audit_sha256"}) != audit_pin
            or report.get("status") != "collected_unreviewed"):
        raise DataReadinessError("original corporate-action audit is not independently pinned and complete")
    pins = {request_path.relative_to(root).as_posix(): request_pin,
        report_path.relative_to(root).as_posix(): file_sha256(report_path)}
    receipts: dict[str, dict[str, str]] = {}
    for ticker in report["tickers"]:
        for attempt in ticker["attempts"]:
            relative = f"tickers/{ticker['ticker']}/{attempt['attempt']}/receipt.json"
            path = inside(source, relative)
            _check(source, {relative: attempt["receipt_sha256"]})
            receipt = collector._object(path)
            receipts[relative] = {"path": path.relative_to(root).as_posix(), "sha256": attempt["receipt_sha256"]}
            pins[path.relative_to(root).as_posix()] = attempt["receipt_sha256"]
            for page in receipt["pages"]:
                for name, digest in ((page["body_path"], page["body_sha256"]),
                    (page["metadata_path"], page["metadata_sha256"])):
                    pins[inside(path.parent, name).relative_to(root).as_posix()] = digest
    _check(root, pins)
    # Replay every original page, exact query and terminal token with today's decoder.
    if collector._report(source, request, root=root) != report:
        raise DataReadinessError("original corporate-action report differs from current transport replay")
    _check(root, pins)
    return report, pins, receipts


def _copy_attempt(root: Path, directory: Path, relative: str, original: dict[str, str],
    request: dict[str, Any], reconstruction: ValidatedReconstruction,
) -> None:
    source_path = inside(root, original["path"])
    _check(root, {original["path"]: original["sha256"]})
    receipt = collector._object(source_path)
    target = inside(directory, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    for page in receipt["pages"]:
        for name, digest in ((page["body_path"], page["body_sha256"]),
            (page["metadata_path"], page["metadata_sha256"])):
            saved = inside(target.parent, name)
            source = inside(source_path.parent, name)
            if not saved.exists():
                shutil.copyfile(source, saved)
            _check(target.parent, {name: digest})
    fresh = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    fresh.update(request_sha256=request["request_sha256"],
        reconstruction_proof=reconstruction.proof_pin, original_receipt=original)
    fresh["receipt_sha256"] = json_sha256(fresh)
    _put(target, fresh)
    collector._check_attempt(target.parent, request, receipt["ticker"], reconstruction=reconstruction, root=root)


def _inventory(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): file_sha256(path)
        for path in directory.rglob("*") if path.is_file() and path != directory / "_checkpoint.json"}


def _verify_checkpoint(directory: Path, expected: str) -> dict[str, Any]:
    _check(directory, {"_checkpoint.json": expected})
    checkpoint = collector._object(directory / "_checkpoint.json")
    if checkpoint.get("schema") != _CHECKPOINT_SCHEMA:
        raise DataReadinessError("corporate-action reconstruction checkpoint schema differs")
    _check(directory, checkpoint["output_files"])
    return checkpoint


def _summary(directory: Path, checkpoint: dict[str, Any], report: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {**_FLAGS, "status": "partial_in_progress", "completed_tickers": len(checkpoint["completed_tickers"]),
        "requested_tickers": checkpoint["requested_tickers"], "checkpoint_sha256": file_sha256(directory / "_checkpoint.json"),
        "reconstruction_proof_sha256": checkpoint["proof_sha256"]}
    if report is not None:
        result.update({name: report[name] for name in ("status", "requested_tickers", "acquired_tickers", "audit_sha256")})
    return result


def reconstruct_corporate_actions(*, root: Path, original_archive: Path, original_request_sha256: str,
    original_audit_sha256: str, config: Path, expected_config_sha256: str, output: Path,
    expected_checkpoint_sha256: str | None = None, maximum_tickers: int | None = None,
) -> dict[str, Any]:
    """Rebind retained exact transport to current scope without a provider call.

    Original acquisition clocks remain unchanged. The proof records the distinct
    reconstruction clock. Only a complete, normally replayed private collection is
    renamed into the requested public path; no original bytes are overwritten.
    """
    for digest in (original_request_sha256, original_audit_sha256, expected_config_sha256, expected_checkpoint_sha256):
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("corporate-action reconstruction requires lowercase SHA256 pins")
    root = root.resolve()
    source, config, output = inside(root, original_archive), inside(root, config), inside(root, output)
    stage = output.with_name(f".{output.name}.reconstructing")
    runtime = inside(root, heavy_job_runtime_dir())
    if maximum_tickers is not None and (type(maximum_tickers) is not int or maximum_tickers < 1):
        raise ValueError("maximum_tickers must be a positive integer")
    if any(output.is_relative_to(path) or path.is_relative_to(output)
            or stage.is_relative_to(path) or path.is_relative_to(stage) for path in (source, config)):
        raise DataReadinessError("corporate-action reconstruction output overlaps source evidence")
    if output.exists() and stage.exists():
        raise DataReadinessError("both public and private reconstruction directories exist")
    directory = output if output.exists() else stage
    if directory.exists() != (expected_checkpoint_sha256 is not None):
        raise DataReadinessError("existing corporate-action reconstruction requires its independent checkpoint pin")
    with file_lock(output.parent / f".{output.name}.reconstruction", timeout=0):
        with heavy_job_lease("reconstruct-swing-corporate-actions", runtime_dir=runtime):
            _guard()
            _check(root, {config.relative_to(root).as_posix(): expected_config_sha256})
            request = collector._prepare(root, config)
            original_report, originals, receipts = _original(root, source, original_request_sha256,
                original_audit_sha256, request)
            implementation = _implementation(root, request)
            current_pins = {**request["bound_files"], **implementation, **originals,
                config.relative_to(root).as_posix(): expected_config_sha256}
            identity = dict(original_archive=source.relative_to(root).as_posix(),
                original_request_sha256=original_request_sha256, original_audit_sha256=original_audit_sha256,
                config=config.relative_to(root).as_posix(), config_sha256=expected_config_sha256,
                output=output.relative_to(root).as_posix(), request_sha256=request["request_sha256"])
            if directory.exists():
                assert expected_checkpoint_sha256 is not None
                checkpoint = _verify_checkpoint(directory, expected_checkpoint_sha256)
                if checkpoint["identity"] != identity or checkpoint["source_files"] != current_pins:
                    raise DataReadinessError("corporate-action reconstruction resume inputs changed")
                if collector._object(directory / "_request.json") != request:
                    raise DataReadinessError("corporate-action reconstruction request changed")
                proof = collector._object(directory / "_reconstruction.json")
                if (proof["source_files"] != originals or proof["receipts"] != receipts
                        or proof["implementation_files"] != implementation):
                    raise DataReadinessError("corporate-action reconstruction proof inputs changed")
            else:
                directory.mkdir(parents=True)
                _put(directory / "_request.json", request)
                audit_path = source / "reports" / f"{original_audit_sha256}.json"
                proof = dict(schema="market_predictor.corporate_action_reconstruction",
                    request_sha256=request["request_sha256"],
                    original_request=dict(path=(source / "_request.json").relative_to(root).as_posix(), sha256=original_request_sha256),
                    original_audit=dict(path=audit_path.relative_to(root).as_posix(),
                        sha256=originals[audit_path.relative_to(root).as_posix()]),
                    source_files=originals, receipts=receipts, reconstructed_at_utc=datetime.now(UTC).isoformat(),
                    implementation_files=implementation)
                _put(directory / "_reconstruction.json", proof)
                checkpoint = dict(schema=_CHECKPOINT_SCHEMA, identity=identity, source_files=current_pins,
                    proof_sha256=file_sha256(directory / "_reconstruction.json"), completed_tickers=[],
                    requested_tickers=len(request["tickers"]), output_files=_inventory(directory), **_FLAGS)
                _replace(directory / "_checkpoint.json", checkpoint)
            if (not set(checkpoint["completed_tickers"]).issubset(request["tickers"])
                    or len(set(checkpoint["completed_tickers"])) != len(checkpoint["completed_tickers"])):
                raise DataReadinessError("corporate-action checkpoint contains foreign or repeated tickers")
            reconstruction = load_reconstruction_proof(root, directory, request)
            if reconstruction is None or reconstruction.proof_pin["sha256"] != checkpoint["proof_sha256"]:
                raise DataReadinessError("corporate-action reconstruction proof pin differs")
            completed = set(checkpoint["completed_tickers"])
            attempted = 0
            for row in original_report["tickers"]:
                ticker = row["ticker"]
                if ticker in completed:
                    continue
                if maximum_tickers is not None and attempted >= maximum_tickers:
                    _check(root, current_pins)
                    return _summary(directory, checkpoint)
                _guard()
                for attempt in row["attempts"]:
                    relative = f"tickers/{ticker}/{attempt['attempt']}/receipt.json"
                    _copy_attempt(root, directory, relative, receipts[relative], request, reconstruction)
                    receipt_path = inside(directory, relative)
                    for path in receipt_path.parent.iterdir():
                        if path.is_file():
                            checkpoint["output_files"][path.relative_to(directory).as_posix()] = file_sha256(path)
                completed.add(ticker)
                checkpoint["completed_tickers"] = sorted(completed)
                _replace(directory / "_checkpoint.json", checkpoint)
                attempted += 1
            report = collector._report(directory, request, root=root)
            if report["status"] != "collected_unreviewed":
                raise DataReadinessError("reconstructed corporate-action collection is incomplete")
            _put(directory / "reports" / f"{report['audit_sha256']}.json", report)
            checkpoint["audit_sha256"] = report["audit_sha256"]
            report_path = directory / "reports" / f"{report['audit_sha256']}.json"
            checkpoint["output_files"][report_path.relative_to(directory).as_posix()] = file_sha256(report_path)
            if _inventory(directory) != checkpoint["output_files"]:
                raise DataReadinessError("corporate-action reconstruction contains unbound output files")
            _check(root, current_pins)
            _replace(directory / "_checkpoint.json", checkpoint)
            final_checkpoint_pin = file_sha256(directory / "_checkpoint.json")
        # The normal reader owns its own lease; never nest heavy-job leases.
        replay = collector.collect_holding_corporate_actions(root, config, directory,
            expected_audit_sha256=report["audit_sha256"])
        with heavy_job_lease("publish-swing-corporate-action-reconstruction", runtime_dir=runtime):
            _guard()
            if replay != report:
                raise DataReadinessError("normal corporate-action offline replay differs")
            final_checkpoint = _verify_checkpoint(directory, final_checkpoint_pin)
            if final_checkpoint != checkpoint or _inventory(directory) != checkpoint["output_files"]:
                raise DataReadinessError("corporate-action reconstruction output changed before publication")
            _check(root, current_pins)
            assert_peak_memory_budget(stage="corporate-action reconstruction publication", hard_budget_gib=5.0, headroom_gib=0.75)
            if directory == stage:
                if output.exists():
                    raise DataReadinessError("corporate-action public output appeared before publication")
                stage.rename(output)
            return _summary(output, checkpoint, report)
