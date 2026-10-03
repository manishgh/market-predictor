"""Explicit provenance for offline reconstruction of corporate-action transport."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority

SCHEMA = "market_predictor.corporate_action_reconstruction"
_FIELDS = {"schema", "request_sha256", "original_request", "original_audit", "source_files", "receipts",
    "reconstructed_at_utc", "implementation_files"}
_QUERY_FIELDS = ("process_start", "process_end", "page_limit", "maximum_pages_per_ticker", "maximum_response_bytes")


def _object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 8 * 1024**2:
        raise DataReadinessError("corporate-action reconstruction metadata exceeds limit")
    return parse_strict_json_object(path.read_bytes(), label="corporate-action reconstruction provenance")


def _pin(value: Any) -> dict[str, str]:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
            or not isinstance(value["path"], str) or not value["path"]
            or not isinstance(value["sha256"], str) or len(value["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in value["sha256"])):
        raise DataReadinessError("corporate-action reconstruction pin is malformed")
    return dict(value)


def _relative(root: Path, name: str) -> Path:
    path = resolve_inside_authority(root, name)
    if path.relative_to(root.resolve()).as_posix() != name:
        raise DataReadinessError("corporate-action reconstruction paths must be canonical relative paths")
    return path


@dataclass(frozen=True)
class ValidatedReconstruction:
    root: Path
    archive: Path
    proof_pin: dict[str, str]
    proof: dict[str, Any]
    original_request: dict[str, Any]
    source_files: dict[str, str]

    def recheck(self) -> None:
        """One full inventory pass, outside the per-receipt validation loop."""
        if self.source_files != {**self.proof["source_files"], **self.proof["implementation_files"]}:
            raise DataReadinessError("corporate-action reconstruction source inventory changed in memory")
        for name, digest in self.source_files.items():
            if file_sha256(_relative(self.root, name)) != digest:
                raise DataReadinessError("corporate-action reconstruction input changed")
        path = self.archive / self.proof_pin["path"]
        if file_sha256(path) != self.proof_pin["sha256"] or _object(path) != self.proof:
            raise DataReadinessError("corporate-action reconstruction proof changed in memory")


def load_reconstruction_proof(root: Path, archive: Path, request: dict[str, Any]) -> ValidatedReconstruction | None:
    """Verify the acyclic proof once per archive report, not once per receipt."""
    root, archive = root.resolve(), archive.resolve()
    path = archive / "_reconstruction.json"
    if not path.exists():
        return None
    if not archive.is_relative_to(root):
        raise DataReadinessError("corporate-action reconstruction archive escapes workspace")
    proof = _object(path)
    if (set(proof) != _FIELDS or proof["schema"] != SCHEMA or proof["request_sha256"] != request["request_sha256"]
            or request["request_sha256"] != json_sha256({key: value for key, value in request.items() if key != "request_sha256"})):
        raise DataReadinessError("corporate-action reconstruction proof request differs")
    try:
        stamp = datetime.fromisoformat(proof["reconstructed_at_utc"])
        if stamp.utcoffset() != UTC.utcoffset(None) or stamp > datetime.now(UTC):
            raise ValueError("not a completed UTC reconstruction")
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("corporate-action reconstruction clock differs") from exc
    files: dict[str, str] = {}
    for field in ("source_files", "implementation_files"):
        values = proof[field]
        if not isinstance(values, dict) or not values:
            raise DataReadinessError("corporate-action reconstruction inventory is empty")
        for name, digest in values.items():
            pin = _pin({"path": name, "sha256": digest})
            source = _relative(root, pin["path"])
            if source.is_relative_to(archive) or (name in files and files[name] != digest):
                raise DataReadinessError("corporate-action reconstruction inventory is cyclic or conflicting")
            if file_sha256(source) != digest:
                raise DataReadinessError("corporate-action reconstruction input changed")
            files[name] = digest
    originals = {}
    for field in ("original_request", "original_audit"):
        pin = _pin(proof[field])
        if proof["source_files"].get(pin["path"]) != pin["sha256"]:
            raise DataReadinessError("corporate-action reconstruction omits original authority")
        originals[field] = _object(_relative(root, pin["path"]))
    original = originals["original_request"]
    audit = originals["original_audit"]
    if (original.get("request_sha256") != json_sha256({k: v for k, v in original.items() if k != "request_sha256"})
            or audit.get("audit_sha256") != json_sha256({k: v for k, v in audit.items() if k != "audit_sha256"})
            or audit.get("request_sha256") != original["request_sha256"]
            or original.get("tickers") != request["tickers"]
            or any(original["policy"].get(key) != request["policy"].get(key) for key in _QUERY_FIELDS)):
        raise DataReadinessError("corporate-action reconstruction original query or audit differs")
    if [item["ticker"] for item in audit["tickers"]] != request["tickers"]:
        raise DataReadinessError("corporate-action reconstruction original ticker inventory differs")
    mapping = proof["receipts"]
    if not isinstance(mapping, dict) or not mapping:
        raise DataReadinessError("corporate-action reconstruction receipt mapping is empty")
    expected = {f"tickers/{ticker['ticker']}/{attempt['attempt']}/receipt.json": attempt["receipt_sha256"]
        for ticker in audit["tickers"] for attempt in ticker["attempts"]}
    if set(mapping) != set(expected) or len(expected) != sum(len(ticker["attempts"]) for ticker in audit["tickers"]):
        raise DataReadinessError("corporate-action reconstruction receipt mapping differs from original audit")
    original_directory = _relative(root, proof["original_request"]["path"]).parent
    for name, value in mapping.items():
        pin = _pin(value)
        source = _relative(root, pin["path"])
        if (source != original_directory / name or proof["source_files"].get(pin["path"]) != pin["sha256"]
                or expected[name] != pin["sha256"]):
            raise DataReadinessError("corporate-action reconstruction original receipt mapping differs")
        receipt = _object(source)
        for page in receipt["pages"]:
            for path_key, hash_key in (("body_path", "body_sha256"), ("metadata_path", "metadata_sha256")):
                page_path = resolve_inside_authority(source.parent, page[path_key])
                relative = page_path.relative_to(root).as_posix()
                if proof["source_files"].get(relative) != page[hash_key]:
                    raise DataReadinessError("corporate-action reconstruction omits original transport inputs")
    proof_pin = {"path": "_reconstruction.json", "sha256": file_sha256(path)}
    return ValidatedReconstruction(root, archive, proof_pin, proof, original, files)


def validate_reconstructed_receipt(attempt: Path, request: dict[str, Any], receipt: dict[str, Any],
    reconstruction: ValidatedReconstruction | None,
) -> None:
    """A fresh receipt may change only its request identity and explicit provenance."""
    marked = "reconstruction_proof" in receipt or "original_receipt" in receipt
    if reconstruction is None:
        if marked:
            raise DataReadinessError("corporate-action receipt lacks its reconstruction proof")
        return
    relative = (attempt / "receipt.json").relative_to(reconstruction.archive).as_posix()
    mapped = reconstruction.proof["receipts"].get(relative)
    if (receipt.get("reconstruction_proof") != reconstruction.proof_pin or mapped is None
            or receipt.get("original_receipt") != mapped
            or request["request_sha256"] != reconstruction.proof["request_sha256"]):
        raise DataReadinessError("corporate-action receipt reconstruction proof or mapping differs")
    source = _relative(reconstruction.root, mapped["path"])
    if file_sha256(source) != mapped["sha256"]:
        raise DataReadinessError("corporate-action original receipt changed")
    original = _object(source)
    excluded = {"request_sha256", "receipt_sha256", "reconstruction_proof", "original_receipt"}
    if (original.get("request_sha256") != reconstruction.original_request["request_sha256"]
            or original.get("receipt_sha256") != json_sha256({k: v for k, v in original.items() if k != "receipt_sha256"})
            or "reconstruction_proof" in original or "original_receipt" in original
            or {k: v for k, v in receipt.items() if k not in excluded} != {k: v for k, v in original.items() if k not in excluded}
            or datetime.fromisoformat(receipt["completed_at_utc"]) > datetime.fromisoformat(
                reconstruction.proof["reconstructed_at_utc"])):
        raise DataReadinessError("corporate-action reconstructed receipt changed original transport or clocks")
