"""Leased, resumable, governed collection of EDGAR pages and documents into a new direct child of data/raw."""
from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.sec_filings import document_collection as documents
from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.system_memory import system_memory_snapshot
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research.legacy_query_identity_proofs import pin_file
from market_predictor.resources import assert_memory_budget
from market_predictor.sources.http import HttpByteResponse, HttpClient
from market_predictor.sources.sec import SecRequestGovernor, SecSource
from market_predictor.swing.datasets.symbol_corrections import pinned_object

REQUESTS_PER_SECOND = 5.0
WORKERS = 4  # SEC answers in about 0.8 seconds, so four requests in flight reach the governed rate.
FORBIDDEN_COOLDOWN_SECONDS = 600.0
RATE_LIMIT_COOLDOWN_SECONDS = 60.0
RUNNER_PATHS = (
    "research/sec_page_collection.py", "catalysts/sec_filings/document_collection.py", "sources/sec.py", "sources/http.py",
    "canonical/store.py", "evidence/hashing.py", "evidence/io.py",
)
Plan = Callable[[dict[str, str]], tuple[pd.DataFrame, dict[str, Any]]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def memory_guard(stage: str) -> None:
    """The SEC jobs' memory policy: a 5 GiB process budget and system memory below 90 percent."""
    assert_memory_budget(stage=stage, hard_budget_gib=5.0, headroom_gib=0.75)
    snapshot = system_memory_snapshot()
    if snapshot is None or snapshot.used_percent >= 90.0:
        raise MemoryBudgetError(f"{stage} requires system memory below 90 percent")


def _guard() -> None:
    memory_guard("SEC page collection")


class GovernedFetch:
    """One governed attempt per call with the fixed options every receipt assumes. Each thread has its own HTTP
    session and all share one governor; once `stop` is set no request is sent, and a governor wait (such as SEC's
    cooldown after a 403) ends at once instead of sending when it expires."""

    def __init__(self, settings: Settings, stop: threading.Event) -> None:
        self._settings = settings
        self._stop = stop
        self._governor = SecRequestGovernor(
            requests_per_second=REQUESTS_PER_SECOND, forbidden_cooldown_seconds=FORBIDDEN_COOLDOWN_SECONDS,
            rate_limit_cooldown_seconds=RATE_LIMIT_COOLDOWN_SECONDS, sleeper=self._sleep)
        self._local = threading.local()
        self._clients: list[HttpClient] = []
        self._lock = threading.Lock()

    def _sleep(self, seconds: float) -> None:
        if self._stop.wait(seconds):
            raise documents.RequestsStopped

    def __call__(self, url: str) -> HttpByteResponse:
        if self._stop.is_set():
            raise documents.RequestsStopped
        client: HttpClient | None = getattr(self._local, "client", None)
        if client is None:
            client = SecSource(self._settings, governor=self._governor).client
            self._local.client = client
            with self._lock:
                self._clients.append(client)
        return client.get_bytes_with_metadata(url, retries=1, allow_redirects=False, raise_for_status=False,
                                              maximum_body_bytes=documents.MAXIMUM_BODY_BYTES)

    def close(self) -> None:
        with self._lock:
            for client in self._clients:
                client.session.close()
            self._clients.clear()


def sec_fetch(settings: Settings, stop: threading.Event) -> GovernedFetch:
    return GovernedFetch(settings, stop)


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


def open_collection(root: Path, record: dict[str, str], schema: str) -> tuple[documents.Store, dict[str, Any]]:
    """Reopen a completed collection of `schema` at its pinned manifest and checkpoint."""
    manifest = pinned_object(pin_file(root, record, {}), record["sha256"])
    _require(manifest.get("schema") == schema and manifest.get("status") == "complete", "SEC page collection is not complete")
    output = inside(root, Path(record["path"]).parent)
    return documents.Store.open(output, str(manifest["checkpoint_sha256"]), str(manifest["request_sha256"])), manifest


def run_collection(*, root: Path, output: Path, job: str, schema: str, plan: Plan, phases: tuple[str, ...],
                   implementation_paths: tuple[str, ...], resume_checkpoint_sha256: str | None,
                   settings: Settings | None) -> dict[str, Any]:
    """Leased, resumable collection of the planned units into a new direct child of data/raw.

    `plan` reads its pinned inputs under the lease, recording their hashes, and returns the
    units plus the request fields that describe them.
    """
    root = root.resolve()
    output = inside(root, output)
    _require(output.parent == root / "data/raw", "SEC page collection must be a direct child of data/raw")
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease(job, runtime_dir=runtime):
        _guard()
        _require(not (output / "_manifest.json").exists(), "completed SEC page collection is immutable")
        pins: dict[str, str] = {}
        units, fields = plan(pins)
        package = Path(__file__).resolve().parents[1]
        request: dict[str, Any] = {
            "schema": f"{schema}_request", **fields, "sealed_until_rules_frozen": bool(fields.get("sealed_until_rules_frozen")),
            "source_files": dict(sorted(pins.items())), "phases": list(phases),
            "work_list_sha256": json_sha256(units.to_dict("records")), "work_list_units": len(units),
            "fetch": {"retries": 1, "allow_redirects": False, "raise_for_status": False,
                      "maximum_body_bytes": documents.MAXIMUM_BODY_BYTES},
            "governor": {"requests_per_second": REQUESTS_PER_SECOND, "forbidden_cooldown_seconds": FORBIDDEN_COOLDOWN_SECONDS,
                         "rate_limit_cooldown_seconds": RATE_LIMIT_COOLDOWN_SECONDS, "workers": WORKERS},
            "maximum_attempts": documents.MAXIMUM_ATTEMPTS, "shard_attempts": documents.SHARD_ATTEMPTS,
            "shard_bytes": documents.SHARD_BYTES,
            "implementation_files": {f"market_predictor/{name}": file_sha256(package / name)
                                     for name in (*implementation_paths, *RUNNER_PATHS)},
            "availability": "retrieval time is first observation in 2026, never historical availability",
            "training_eligible": False, "serving_eligible": False, "promotion_eligible": False,
        }
        request["request_sha256"] = json_sha256(request)
        request_path = output / "_request.json"
        if request_path.exists():
            _require(resume_checkpoint_sha256 is not None, "resuming requires an independently pinned checkpoint SHA256")
            _require(pinned_object(request_path) == request, "SEC page request differs; use a new output directory")
            store = documents.Store.open(output, str(resume_checkpoint_sha256), request["request_sha256"])
        else:
            _require(resume_checkpoint_sha256 is None and not output.exists(), "SEC page output has files without a request")
            output.mkdir(parents=True)
            write_json_object(request_path, request)
            store = documents.Store(output, {})
            store.write_checkpoint(request["request_sha256"])
        stop = threading.Event()
        fetch = sec_fetch(settings or Settings(), stop)
        try:
            result = documents.collect(store=store, units=units, fetch=fetch, request_sha256=request["request_sha256"],
                                       memory_check=_guard,
                                       cooldowns={403: FORBIDDEN_COOLDOWN_SECONDS, 429: RATE_LIMIT_COOLDOWN_SECONDS},
                                       stop=stop, phases=phases, workers=WORKERS)
        finally:
            fetch.close()
        checkpoint = file_sha256(output / "_checkpoint.json")
        if result["status"] != "complete":
            return {**result, "checkpoint_sha256": checkpoint}
        _guard()
        final, receipts = documents.outcomes(store, units, phases)
        store.verify(receipts)
        report = {
            "schema": schema, "status": "complete", "request_sha256": request["request_sha256"],
            **({"inventory_mode": request["inventory_mode"]} if "inventory_mode" in request else {}),
            "sealed_until_rules_frozen": request["sealed_until_rules_frozen"],
            "shards": dict(sorted(store.shards.items())), "checkpoint_sha256": checkpoint,
            "totals": _totals(final, receipts, sealed=request["sealed_until_rules_frozen"]),
            "document_scope": "retained bytes are unreviewed; their consumers decide meaning and use",
            "training_eligible": False, "serving_eligible": False, "promotion_eligible": False,
        }
        with tempfile.TemporaryDirectory(dir=output, prefix=".manifest-") as temporary:
            staged = Path(temporary) / "_manifest.json"
            write_json_object(staged, report)
            os.replace(staged, output / "_manifest.json")
        return {**report, "manifest_sha256": file_sha256(output / "_manifest.json")}
