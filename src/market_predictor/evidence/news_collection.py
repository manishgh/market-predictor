"""Single-local-root raw news collection; no normalized coverage or PIT admission.

Attempts are logical transport invocations, which may contain transport retries.
The configured root is the coordination boundary, not a distributed owner lease.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Self, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.news_exchange import (
    NewsPageRequest,
    ValidatedNewsReceipt,
    read_news_receipt,
    validate_news_receipt,
)
from market_predictor.locking import file_lock

MAX_PLAN_BYTES = 4_194_304
MAX_LEDGER_RECORD_BYTES = 16_384
MAX_LEDGER_ENTRIES = 100_000
_DIGEST = r"^[0-9a-f]{64}$"
_NAME = r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$"
_ATTEMPT_NAME = re.compile(r"^[0-9]{6}$")


class NewsCollectionEvidenceError(ValueError):
    """Invalid or inconsistent collection evidence; never a transport retry."""


class NewsCollectionTransportError(RuntimeError):
    """An explicitly classified fetch failure that stops only its window."""


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _Owner(_Contract):
    schema_version: Literal["alpaca.news_collection_owner.v1"] = "alpaca.news_collection_owner.v1"
    owner: Literal["market_predictor"] = "market_predictor"
    coordination: Literal["single_local_root"] = "single_local_root"
    storage_root: str = Field(min_length=1, max_length=4096)


class NewsCollectionWindow(_Contract):
    window_id: str = Field(pattern=_NAME)
    request: NewsPageRequest

    @model_validator(mode="after")
    def initial_page(self) -> Self:
        reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
        if self.window_id.casefold() in reserved:
            raise ValueError("collection window name is reserved by the local filesystem")
        if self.request.page_token is not None:
            raise ValueError("collection windows must start with a null page token")
        return self


class NewsCollectionPlan(_Contract):
    schema_version: Literal["alpaca.news_collection_plan.v1"] = "alpaca.news_collection_plan.v1"
    owner: Literal["market_predictor"] = "market_predictor"
    coordination: Literal["single_local_root"] = "single_local_root"
    attempt_semantics: Literal["logical_transport_invocation"] = "logical_transport_invocation"
    producer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    windows: tuple[NewsCollectionWindow, ...] = Field(min_length=1, max_length=1000)
    max_pages_per_window: int = Field(ge=1, le=10_000)
    max_attempts_per_page: int = Field(default=3, ge=1, le=10)

    @field_validator("windows", mode="before")
    @classmethod
    def wire_array(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def unique_bounded_windows(self) -> Self:
        names = [window.window_id.casefold() for window in self.windows]
        queries = [(w.request.symbol, w.request.start_utc, w.request.end_utc) for w in self.windows]
        if len(set(names)) != len(names) or len(set(queries)) != len(queries):
            raise ValueError("collection window names and symbol/time windows must be unique")
        if len(self.windows) * self.max_pages_per_window * self.max_attempts_per_page > MAX_LEDGER_ENTRIES:
            raise ValueError("collection plan exceeds its total ledger budget")
        return self


class _Intent(_Contract):
    schema_version: Literal["alpaca.news_collection_intent.v1"] = "alpaca.news_collection_intent.v1"
    plan_sha256: str = Field(pattern=_DIGEST)
    window_id: str = Field(pattern=_NAME)
    attempt_number: int = Field(ge=1, le=MAX_LEDGER_ENTRIES)
    page_number: int = Field(ge=1, le=10_000)
    request: NewsPageRequest


class _Result(_Contract):
    schema_version: Literal["alpaca.news_collection_result.v1"] = "alpaca.news_collection_result.v1"
    intent_sha256: str = Field(pattern=_DIGEST)
    status: Literal["received", "transport_failure"]
    receipt_sha256: str | None = Field(default=None, pattern=_DIGEST)

    @model_validator(mode="after")
    def receipt_binding(self) -> Self:
        if (self.status == "received") != (self.receipt_sha256 is not None):
            raise ValueError("received results require exactly one independent receipt pin")
        return self


@dataclass(frozen=True, slots=True)
class NewsCollectedReceipt:
    attempt_id: str
    receipt_sha256: str
    manifest_path: Path
    payload_path: Path


WindowStatus = Literal["pending", "complete", "transport_failure", "pagination_cycle", "page_budget", "attempt_budget"]


@dataclass(frozen=True, slots=True)
class NewsCollectionWindowReport:
    window_id: str
    status: WindowStatus
    attempts: int
    ambiguous_attempts: int
    receipts: tuple[NewsCollectedReceipt, ...]


@dataclass(frozen=True, slots=True)
class NewsCollectionReport:
    plan_sha256: str
    windows: tuple[NewsCollectionWindowReport, ...]

    @property
    def complete(self) -> bool:
        return all(window.status == "complete" for window in self.windows)


@dataclass(slots=True)
class _WindowState:
    request: NewsPageRequest
    attempts: int = 0
    page_attempts: int = 0
    ambiguous_attempts: int = 0
    receipts: list[NewsCollectedReceipt] = field(default_factory=list)
    seen_tokens: set[str] = field(default_factory=set)
    terminal: Literal["complete", "pagination_cycle"] | None = None


_Model = TypeVar("_Model", bound=_Contract)


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        if value.utcoffset() != timedelta(0):
            raise NewsCollectionEvidenceError("collection clocks must be timezone-aware UTC")
        return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    raise TypeError(f"unsupported collection JSON value: {type(value).__name__}")


def _encode(model: _Contract) -> bytes:
    return json.dumps(model.model_dump(mode="python"), default=_json_default,
                      ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def news_collection_plan_bytes(plan: NewsCollectionPlan) -> bytes:
    # Revalidate even instances constructed through model_construct/model_copy.
    checked = NewsCollectionPlan.model_validate(parse_strict_json_object(_encode(plan), label="collection plan"))
    raw = _encode(checked)
    if len(raw) > MAX_PLAN_BYTES:
        raise NewsCollectionEvidenceError("collection plan exceeds its byte limit")
    return raw


def news_collection_plan_sha256(plan: NewsCollectionPlan) -> str:
    return hashlib.sha256(news_collection_plan_bytes(plan)).hexdigest()


def read_news_collection_plan(path: Path, *, expected_plan_sha256: str) -> NewsCollectionPlan:
    raw = _read_bounded(path, MAX_PLAN_BYTES)
    _check_pin(raw, expected_plan_sha256, "collection plan")
    plan = NewsCollectionPlan.model_validate(parse_strict_json_object(raw, label="collection plan"))
    if raw != news_collection_plan_bytes(plan):
        raise NewsCollectionEvidenceError("collection plan must use the canonical collection wire encoding")
    return plan


def _check_pin(raw: bytes, expected: str, label: str) -> None:
    if re.fullmatch(_DIGEST, expected) is None or hashlib.sha256(raw).hexdigest() != expected:
        raise NewsCollectionEvidenceError(f"{label} does not match its independent integrity pin")


def _plain_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise NewsCollectionEvidenceError("collection evidence cannot contain links or reparse points")


def _read_bounded(path: Path, maximum: int) -> bytes:
    _plain_path(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise NewsCollectionEvidenceError("collection ledger records must be regular files")
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if not 0 < len(raw) <= maximum:
        raise NewsCollectionEvidenceError("collection ledger record exceeds its byte limit")
    return raw


def _read_record(path: Path, model: type[_Model]) -> tuple[_Model, bytes]:
    raw = _read_bounded(path, MAX_LEDGER_RECORD_BYTES)
    value = model.model_validate(parse_strict_json_object(raw, label="collection ledger record"))
    if raw != _encode(value):
        raise NewsCollectionEvidenceError("collection ledger record has noncanonical bytes")
    return value, raw


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _mkdir(path: Path) -> None:
    _plain_path(path)
    if not path.exists():
        path.mkdir()
        _sync_directory(path.parent)
    if not path.is_dir():
        raise NewsCollectionEvidenceError("collection directory is not a directory")


def _write_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory(target: Path, files: dict[str, bytes]) -> None:
    if target.exists() or target.is_symlink():
        raise NewsCollectionEvidenceError("refusing to overwrite committed collection evidence")
    staging = target.with_name(f".pending-{uuid4().hex}")
    staging.mkdir()
    for name, raw in files.items():
        _write_new(staging / name, raw)
    _sync_directory(staging)
    # Same-volume publication, without replacement. Windows requests write-through
    # metadata publication; POSIX additionally fsyncs the containing directory.
    if os.name == "nt":
        move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move.restype = ctypes.c_int
        if not move(str(staging), str(target), 0x8):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.rename(staging, target)
    _sync_directory(target.parent)


def _entries(path: Path, maximum: int) -> list[Path]:
    _plain_path(path)
    if not path.exists():
        return []
    entries: list[Path] = []
    with os.scandir(path) as iterator:
        for entry in iterator:
            if len(entries) >= maximum:
                raise NewsCollectionEvidenceError("collection directory exceeds its entry budget")
            item = Path(entry.path)
            _plain_path(item)
            entries.append(item)
    return entries


def _pending(path: Path) -> bool:
    return re.fullmatch(r"\.pending-[0-9a-f]{32}", path.name) is not None and path.is_dir()


def _require_files(path: Path, names: set[str]) -> None:
    entries = _entries(path, len(names))
    if {entry.name for entry in entries} != names or any(not entry.is_file() for entry in entries):
        raise NewsCollectionEvidenceError("committed collection bundle has missing or unexpected files")


def _validate_receipt(receipt: ValidatedNewsReceipt, plan: NewsCollectionPlan, request: NewsPageRequest) -> ValidatedNewsReceipt:
    checked = validate_news_receipt(receipt.manifest_bytes, receipt.payload_bytes)
    if checked.manifest != receipt.manifest:
        raise NewsCollectionEvidenceError("receipt object disagrees with its original manifest bytes")
    if (checked.manifest.request != request or checked.manifest.producer != plan.owner
            or checked.manifest.producer_revision != plan.producer_revision):
        raise NewsCollectionEvidenceError("receipt request, producer or revision differs from the pinned plan")
    return checked


def _accept(state: _WindowState, receipt: ValidatedNewsReceipt, attempt_path: Path, intent_sha256: str) -> None:
    state.receipts.append(NewsCollectedReceipt(intent_sha256, receipt.receipt_id,
                                             attempt_path / "receipt" / "manifest.json",
                                             attempt_path / "receipt" / "payload.json"))
    token = parse_strict_json_object(receipt.payload_bytes, label="news receipt body").get("next_page_token")
    if token is None:
        state.terminal = "complete"
    elif not isinstance(token, str):
        raise NewsCollectionEvidenceError("validated receipt has a non-string pagination token")
    elif token in state.seen_tokens:
        state.terminal = "pagination_cycle"
    else:
        state.seen_tokens.add(token)
        state.request = state.request.model_copy(update={"page_token": token})
    state.page_attempts = 0


def _stop(state: _WindowState, plan: NewsCollectionPlan) -> WindowStatus | None:
    if state.terminal is not None:
        return state.terminal
    if len(state.receipts) >= plan.max_pages_per_window:
        return "page_budget"
    if state.page_attempts >= plan.max_attempts_per_page:
        return "attempt_budget"
    return None


def _recover(root: Path, plan: NewsCollectionPlan, pin: str, window: NewsCollectionWindow) -> _WindowState:
    state = _WindowState(window.request)
    directory = root / "attempts" / window.window_id
    maximum = plan.max_pages_per_window * plan.max_attempts_per_page
    entries = _entries(directory, min(MAX_LEDGER_ENTRIES, maximum * 2 + 16))
    attempts: list[Path] = []
    for entry in entries:
        if _pending(entry):
            continue
        if not _ATTEMPT_NAME.fullmatch(entry.name) or not entry.is_dir():
            raise NewsCollectionEvidenceError("unexpected collection attempt entry")
        attempts.append(entry)
    if len(attempts) > maximum:
        raise NewsCollectionEvidenceError("collection attempts exceed the pinned plan budget")
    for index, attempt_path in enumerate(sorted(attempts), 1):
        if attempt_path.name != f"{index:06d}" or _stop(state, plan) is not None:
            raise NewsCollectionEvidenceError("collection attempt sequence is discontinuous or exceeds its terminal boundary")
        intent, raw = _read_record(attempt_path / "intent.json", _Intent)
        expected = _Intent(plan_sha256=pin, window_id=window.window_id, attempt_number=index,
                           page_number=len(state.receipts) + 1, request=state.request)
        if intent != expected:
            raise NewsCollectionEvidenceError("collection intent does not match the reconstructed plan/page identity")
        intent_sha256 = hashlib.sha256(raw).hexdigest()
        state.attempts += 1
        state.page_attempts += 1
        for item in _entries(attempt_path, 32):
            if item.name not in {"intent.json", "receipt", "result"} and not _pending(item):
                raise NewsCollectionEvidenceError("unexpected collection attempt artifact")
        result_path = attempt_path / "result"
        if not result_path.exists():
            state.ambiguous_attempts += 1
            continue
        _require_files(result_path, {"result.json"})
        result, _ = _read_record(result_path / "result.json", _Result)
        if result.intent_sha256 != intent_sha256:
            raise NewsCollectionEvidenceError("collection result does not bind its exact intent")
        if result.status == "transport_failure":
            if (attempt_path / "receipt").exists():
                raise NewsCollectionEvidenceError("failed collection attempt unexpectedly contains a receipt")
            continue
        assert result.receipt_sha256 is not None
        bundle = attempt_path / "receipt"
        _require_files(bundle, {"manifest.json", "payload.json"})
        receipt = read_news_receipt(bundle / "manifest.json", bundle / "payload.json",
                                    expected_receipt_sha256=result.receipt_sha256)
        _accept(state, _validate_receipt(receipt, plan, state.request), attempt_path, intent_sha256)
    return state


def _local_root(root: Path) -> Path:
    resolved = root.resolve()
    if str(resolved).startswith(("\\\\", "//")):
        raise ValueError("shared news collection supports only a local filesystem root")
    if os.name == "nt":
        drive_type = ctypes.WinDLL("kernel32", use_last_error=True).GetDriveTypeW
        drive_type.argtypes = [ctypes.c_wchar_p]
        drive_type.restype = ctypes.c_uint
        if drive_type(resolved.anchor) != 3:  # DRIVE_FIXED, excluding mapped network drives.
            raise ValueError("shared news collection requires a fixed local drive")
    return resolved


def _empty_report(plan: NewsCollectionPlan, pin: str) -> NewsCollectionReport:
    return NewsCollectionReport(pin, tuple(NewsCollectionWindowReport(w.window_id, "pending", 0, 0, ()) for w in plan.windows))


def _verify_owner(root: Path, *, create: bool) -> bool:
    for entry in _entries(root, 1024):
        if entry.name not in {"collection-owner.lock", "owner", "runs"} and not _pending(entry):
            raise NewsCollectionEvidenceError("unexpected shared collection root artifact")
    owner_path = root / "owner"
    expected = _Owner(storage_root=str(root))
    if owner_path.exists():
        _require_files(owner_path, {"owner.json"})
        owner, _ = _read_record(owner_path / "owner.json", _Owner)
        if owner != expected:
            raise NewsCollectionEvidenceError("shared collection owner marker does not match this local root")
        return True
    if (root / "runs").exists():
        raise NewsCollectionEvidenceError("shared collection runs exist without a committed owner marker")
    if create:
        _publish_directory(owner_path, {"owner.json": _encode(expected)})
    return create


def collect_news_receipts(
    *,
    root: Path,
    plan: NewsCollectionPlan,
    expected_plan_sha256: str,
    fetch_page: Callable[[NewsPageRequest], ValidatedNewsReceipt] | None = None,
) -> NewsCollectionReport:
    """Collect bounded pages under one nonqueueing local owner lock.

    Only NewsCollectionTransportError is isolated to a window. All evidence and
    publication errors propagate. Receipt references contain the independent pins
    and unchanged v1 files consumed by Python and C# importers. Missing results
    remain ambiguous, including complete receipt bundles without a committed
    result; all their bytes are preserved and retry requires a new intent.
    fetch_page=None verifies existing evidence without filesystem writes and
    reports pending work. All plans share root/collection-owner.lock and root's
    immutable owner marker; each plan owns root/runs/{plan_sha256}.
    """
    raw_plan = news_collection_plan_bytes(plan)
    _check_pin(raw_plan, expected_plan_sha256, "collection plan")
    plan = NewsCollectionPlan.model_validate(parse_strict_json_object(raw_plan, label="collection plan"))
    root = _local_root(root)
    if not root.exists():
        if fetch_page is None:
            return _empty_report(plan, expected_plan_sha256)
        root.mkdir(parents=True, exist_ok=True)
        _sync_directory(root.parent)
    lock_path = root / "collection-owner.lock"
    _plain_path(lock_path)
    if fetch_page is None and (not lock_path.exists() or lock_path.stat().st_size == 0):
        # file_lock creates/initializes its lock file. Never do that in offline
        # mode, and never read an initialized store without the common lock.
        if _verify_owner(root, create=False):
            raise NewsCollectionEvidenceError("initialized collection root has no usable owner lock")
        return _empty_report(plan, expected_plan_sha256)
    with file_lock(root / "collection-owner", timeout=0.0):
        if not _verify_owner(root, create=fetch_page is not None):
            return _empty_report(plan, expected_plan_sha256)
        runs_root = root / "runs"
        run_root = runs_root / expected_plan_sha256
        _plain_path(runs_root)
        _plain_path(run_root)
        if fetch_page is not None:
            _mkdir(runs_root)
            _mkdir(run_root)
        published_plan = run_root / "plan"
        _plain_path(published_plan)
        if published_plan.exists():
            _require_files(published_plan, {"plan.json"})
            persisted = read_news_collection_plan(published_plan / "plan.json", expected_plan_sha256=expected_plan_sha256)
            if persisted != plan:
                raise NewsCollectionEvidenceError("collection run belongs to a different plan")
        else:
            if (run_root / "attempts").exists():
                raise NewsCollectionEvidenceError("collection run has attempts but no committed plan")
            if fetch_page is None:
                return _empty_report(plan, expected_plan_sha256)
            _publish_directory(published_plan, {"plan.json": raw_plan})
        attempts_root = run_root / "attempts"
        if fetch_page is not None:
            _mkdir(attempts_root)
        names = {window.window_id for window in plan.windows}
        for entry in _entries(attempts_root, len(names)):
            if entry.name not in names or not entry.is_dir():
                raise NewsCollectionEvidenceError("collection root contains an unplanned window")
        # Preflight every window before issuing any request, including windows
        # that were already terminal in a preceding invocation.
        states = [_recover(run_root, plan, expected_plan_sha256, window) for window in plan.windows]
        reports: list[NewsCollectionWindowReport] = []
        for window, state in zip(plan.windows, states, strict=True):
            directory = attempts_root / window.window_id
            status = _stop(state, plan)
            if fetch_page is None:
                reports.append(NewsCollectionWindowReport(window.window_id, status or "pending", state.attempts,
                                                          state.ambiguous_attempts, tuple(state.receipts)))
                continue
            _mkdir(directory)
            while status is None:
                number = state.attempts + 1
                intent = _Intent(plan_sha256=expected_plan_sha256, window_id=window.window_id,
                                 attempt_number=number, page_number=len(state.receipts) + 1, request=state.request)
                intent_bytes = _encode(intent)
                intent_sha256 = hashlib.sha256(intent_bytes).hexdigest()
                attempt_path = directory / f"{number:06d}"
                _publish_directory(attempt_path, {"intent.json": intent_bytes})
                state.attempts += 1
                state.page_attempts += 1
                try:
                    fetched = fetch_page(state.request)
                except NewsCollectionTransportError:
                    result = _Result(intent_sha256=intent_sha256, status="transport_failure")
                    _publish_directory(attempt_path / "result", {"result.json": _encode(result)})
                    status = "transport_failure"
                    break
                receipt = _validate_receipt(fetched, plan, state.request)
                _publish_directory(attempt_path / "receipt", {"manifest.json": receipt.manifest_bytes,
                                                            "payload.json": receipt.payload_bytes})
                result = _Result(intent_sha256=intent_sha256, status="received", receipt_sha256=receipt.receipt_id)
                _publish_directory(attempt_path / "result", {"result.json": _encode(result)})
                _accept(state, receipt, attempt_path, intent_sha256)
                status = _stop(state, plan)
            reports.append(NewsCollectionWindowReport(window.window_id, status, state.attempts,
                                                      state.ambiguous_attempts, tuple(state.receipts)))
        return NewsCollectionReport(expected_plan_sha256, tuple(reports))
