"""Leased, resumable collection of selected SEC filing documents from a pinned SEC form inventory."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.sec_filings import document_collection as documents
from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research.issuer_content_inventory import _guard
from market_predictor.research.legacy_query_identity_proofs import pin_file
from market_predictor.research.sec_form_inventory import SEALED_RULE, load_sec_form_inventory
from market_predictor.sources.http import HttpByteResponse
from market_predictor.sources.sec import SecRequestGovernor, SecSource
from market_predictor.swing.datasets.symbol_corrections import pinned_object

SCHEMA = "market_predictor.sec_filing_document_collection"
REQUESTS_PER_SECOND = 5.0
FORBIDDEN_COOLDOWN_SECONDS = 600.0
RATE_LIMIT_COOLDOWN_SECONDS = 60.0
IMPLEMENTATION_PATHS = (
    "research/sec_filing_documents.py", "catalysts/sec_filings/document_collection.py", "sources/sec.py", "sources/http.py",
    "canonical/store.py", "evidence/hashing.py", "evidence/io.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def sec_fetch(settings: Settings) -> documents.Fetch:
    """One governed attempt per call with the fixed options every receipt assumes."""
    governor = SecRequestGovernor(requests_per_second=REQUESTS_PER_SECOND, forbidden_cooldown_seconds=FORBIDDEN_COOLDOWN_SECONDS,
                                  rate_limit_cooldown_seconds=RATE_LIMIT_COOLDOWN_SECONDS)
    client = SecSource(settings, governor=governor).client

    def fetch(url: str) -> HttpByteResponse:
        return client.get_bytes_with_metadata(url, retries=1, allow_redirects=False, raise_for_status=False,
                                              maximum_body_bytes=documents.MAXIMUM_BODY_BYTES)

    return fetch


def _units(root: Path, record: dict[str, str], accessions: tuple[str, ...], pins: dict[str, str]
           ) -> tuple[pd.DataFrame, dict[str, Any]]:
    filings, manifest = load_sec_form_inventory(root, record, pins, allow_sealed_collection=True)
    units = documents.work_list(filings)
    if accessions:
        missing = set(accessions) - set(units.accession_number)
        _require(not missing, f"pilot accessions are not selected filings: {sorted(missing)}")
        units = units.loc[units.accession_number.isin(accessions)].reset_index(drop=True)
    return units, manifest


def _totals(final: pd.DataFrame, receipts: pd.DataFrame, *, sealed: bool) -> dict[str, Any]:
    def pairs(frame: pd.DataFrame, keys: list[str]) -> dict[str, int]:
        return {"/".join(map(str, key)): int(count) for key, count in sorted(frame.groupby(keys).size().items())}

    units = {"units_by_phase_and_state": pairs(final, ["phase", "state"]),
             "attempts_by_state": {str(key): int(value) for key, value in sorted(receipts.state.value_counts().items())}}
    if sealed:
        return units  # Sealed collections report unit counts only.

    archived = receipts.loc[receipts.state.eq("archived")]
    kinds = final.loc[final.phase.eq("document")].assign(kind=lambda frame: frame.document_type.str.upper().where(
        frame.document_type.str.upper().str.startswith("EX-99"), "primary"))
    media = archived.content_type.fillna("none").str.split(";").str[0].str.strip()
    return {
        **units, "documents_by_type_and_state": pairs(kinds, ["kind", "state"]),
        "archived_bytes": int(pd.to_numeric(archived.body_length).sum()),
        "archived_content_types": {str(key): int(value) for key, value in sorted(media.value_counts().items())},
        "first_observed_utc": {"start": str(receipts.started_at_utc.min()), "end": str(receipts.completed_at_utc.max())},
    }


def open_sec_document_collection(root: Path, record: dict[str, str]) -> tuple[documents.Store, dict[str, Any]]:
    """The one reader of a completed collection; a sealed one stays closed until qualification rules are frozen."""
    manifest = pinned_object(pin_file(root, record, {}), record["sha256"])
    _require(manifest.get("schema") == SCHEMA and manifest.get("status") == "complete", "SEC document collection is not complete")
    _require(not manifest.get("sealed_until_rules_frozen"), f"SEC document collection is sealed: {SEALED_RULE}")
    output = inside(root, Path(record["path"]).parent)
    return documents.Store.open(output, str(manifest["checkpoint_sha256"]), str(manifest["request_sha256"])), manifest


def collect_sec_filing_documents(*, root: Path, inventory: dict[str, str], output: Path,
                                 resume_checkpoint_sha256: str | None = None, pilot_accessions: tuple[str, ...] = (),
                                 settings: Settings | None = None) -> dict[str, Any]:
    """Collect the detail page, primary document and EX-99 exhibits of every selected filing; never content review."""
    root = root.resolve()
    output = inside(root, output)
    _require(output.parent == root / "data/raw", "SEC document collection must be a direct child of data/raw")
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease("collect-sec-filing-documents", runtime_dir=runtime):
        _guard()
        _require(not (output / "_manifest.json").exists(), "completed SEC document collection is immutable")
        pins: dict[str, str] = {}
        units, manifest = _units(root, inventory, tuple(sorted(pilot_accessions)), pins)
        package = Path(__file__).resolve().parents[1]
        request: dict[str, Any] = {
            "schema": f"{SCHEMA}_request", "inventory": {"path": inside(root, inventory["path"]).relative_to(root).as_posix(),
                                                           "sha256": inventory["sha256"]},
            "inventory_mode": manifest["mode"], "sealed_until_rules_frozen": bool(manifest.get("sealed_until_rules_frozen")),
            "source_files": dict(sorted(pins.items())), "pilot_accessions": sorted(pilot_accessions),
            "work_list_sha256": json_sha256(units.to_dict("records")), "work_list_units": len(units),
            "selection": {"forms": list(documents.CURRENT_REPORTS), "items": sorted(documents.SELECTED_ITEMS),
                          "documents": "primary document and every EX-99 exhibit named by a verified detail page"},
            "fetch": {"retries": 1, "allow_redirects": False, "raise_for_status": False,
                      "maximum_body_bytes": documents.MAXIMUM_BODY_BYTES},
            "governor": {"requests_per_second": REQUESTS_PER_SECOND, "forbidden_cooldown_seconds": FORBIDDEN_COOLDOWN_SECONDS,
                         "rate_limit_cooldown_seconds": RATE_LIMIT_COOLDOWN_SECONDS},
            "maximum_attempts": documents.MAXIMUM_ATTEMPTS, "shard_attempts": documents.SHARD_ATTEMPTS,
            "shard_bytes": documents.SHARD_BYTES,
            "implementation_files": {f"market_predictor/{name}": file_sha256(package / name) for name in IMPLEMENTATION_PATHS},
            "availability": "retrieval time is first observation in 2026, never historical availability",
            "training_eligible": False, "serving_eligible": False, "promotion_eligible": False,
        }
        request["request_sha256"] = json_sha256(request)
        request_path = output / "_request.json"
        if request_path.exists():
            _require(resume_checkpoint_sha256 is not None, "resuming requires an independently pinned checkpoint SHA256")
            _require(pinned_object(request_path) == request, "SEC document request differs; use a new output directory")
            store = documents.Store.open(output, str(resume_checkpoint_sha256), request["request_sha256"])
        else:
            _require(resume_checkpoint_sha256 is None and not output.exists(), "SEC document output has files without a request")
            output.mkdir(parents=True)
            write_json_object(request_path, request)
            store = documents.Store(output, {})
            store.write_checkpoint(request["request_sha256"])
        result = documents.collect(store=store, units=units, fetch=sec_fetch(settings or Settings()),
                                   request_sha256=request["request_sha256"], memory_check=_guard,
                                   cooldowns={403: FORBIDDEN_COOLDOWN_SECONDS, 429: RATE_LIMIT_COOLDOWN_SECONDS})
        checkpoint = file_sha256(output / "_checkpoint.json")
        if result["status"] != "complete":
            return {**result, "checkpoint_sha256": checkpoint}
        _guard()
        final, receipts = documents.outcomes(store, units)
        store.verify(receipts)
        report = {
            "schema": SCHEMA, "status": "complete", "request_sha256": request["request_sha256"],
            "inventory_mode": request["inventory_mode"], "sealed_until_rules_frozen": request["sealed_until_rules_frozen"],
            "shards": dict(sorted(store.shards.items())), "checkpoint_sha256": checkpoint,
            "totals": _totals(final, receipts, sealed=request["sealed_until_rules_frozen"]),
            "document_scope": "retained bytes are unreviewed; content qualification decides meaning and use",
            "training_eligible": False, "serving_eligible": False, "promotion_eligible": False,
        }
        with tempfile.TemporaryDirectory(dir=output, prefix=".manifest-") as temporary:
            staged = Path(temporary) / "_manifest.json"
            write_json_object(staged, report)
            os.replace(staged, output / "_manifest.json")
        return {**report, "manifest_sha256": file_sha256(output / "_manifest.json")}
