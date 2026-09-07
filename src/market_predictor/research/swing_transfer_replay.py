"""Bounded provider replay for historical index-transfer identity research.

Acquisition and exact-price comparison are evidence, not stock-class continuity,
total-return reconciliation, or permission to replace retained observations.
"""
from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import requests

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.locking import file_lock
from market_predictor.research.swing_transfer_sources import checked_transfer_file, prepare_transfer_replay_request
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget
from market_predictor.sources.alpaca import AlpacaBarsPage, decode_bars_page_response
from market_predictor.sources.http import HttpByteResponse

TransferPageFetcher = Callable[[dict[str, Any], str | None], AlpacaBarsPage]
_MAX_PAGE_BYTES = 32 * 1024 * 1024


def _object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 8 * 1024 * 1024:
        raise DataReadinessError("transfer replay metadata exceeds size limit")
    return parse_strict_json_object(path.read_bytes(), label=str(path))


def _request(root: Path, config: Path) -> dict[str, Any]:
    result = prepare_transfer_replay_request(root, config)
    package = Path(__file__).resolve().parents[1]
    result["implementation_sha256"] = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in (
            "research/swing_transfer_sources.py", "research/swing_transfer_replay.py",
            "sources/alpaca.py", "swing/labels/holding_paths.py",
        )
    }
    return {**result, "request_sha256": json_sha256(result)}


def _http_response(body: bytes, metadata: dict[str, Any]) -> HttpByteResponse:
    expected = {"requested_url", "final_url", "status_code", "retrieved_at_utc", "redirect_chain", "response_headers",
                "body_length", "body_sha256"}
    if set(metadata) != expected or not isinstance(metadata["response_headers"], dict):
        raise DataReadinessError("transfer page metadata contract differs")
    if len(body) != metadata["body_length"] or hashlib.sha256(body).hexdigest() != metadata["body_sha256"]:
        raise DataReadinessError("transfer page body hash or length differs")
    headers = metadata["response_headers"]
    return HttpByteResponse(
        body=body, requested_url=metadata["requested_url"], final_url=metadata["final_url"],
        redirect_chain=tuple(metadata["redirect_chain"]), status_code=metadata["status_code"],
        retrieved_at_utc=datetime.fromisoformat(metadata["retrieved_at_utc"]),
        content_type=headers.get("content-type"), content_encoding=headers.get("content-encoding"),
        etag=headers.get("etag"), last_modified=headers.get("last-modified"), body_length=len(body),
        sha256=metadata["body_sha256"], body_representation="http_entity_encoded", safe_headers=tuple(headers.items()),
    )


def _store_page(
    staging: Path, page: AlpacaBarsPage, unit: dict[str, Any], token: str | None, ordinal: int,
) -> tuple[dict[str, Any], AlpacaBarsPage]:
    if (page.raw_body is None or len(page.raw_body) > _MAX_PAGE_BYTES or page.requested_url is None
            or page.final_url is None or page.retrieved_at_utc is None or page.request_page_token != token):
        raise DataReadinessError("transfer page lacks mandatory bounded raw response provenance")
    metadata = {
        "requested_url": page.requested_url, "final_url": page.final_url, "status_code": page.status_code,
        "retrieved_at_utc": page.retrieved_at_utc.isoformat(), "redirect_chain": list(page.redirect_chain),
        "response_headers": page.response_headers, "body_length": len(page.raw_body),
        "body_sha256": hashlib.sha256(page.raw_body).hexdigest(),
    }
    params = {**unit["parameters"], **({"page_token": token} if token is not None else {})}
    decoded = decode_bars_page_response(_http_response(page.raw_body, metadata), expected_params=params)
    if page.next_page_token != decoded.next_page_token:
        raise DataReadinessError("transfer parsed page token differs from exact response bytes")
    body_name = f"{ordinal:03d}-{metadata['body_sha256']}.bin"
    metadata_name = f"{ordinal:03d}.json"
    (staging / body_name).write_bytes(page.raw_body)
    write_json_object(staging / metadata_name, metadata)
    return {
        "body_path": body_name, "metadata_path": metadata_name,
        "metadata_sha256": hashlib.sha256((staging / metadata_name).read_bytes()).hexdigest(),
    }, decoded


def _read_pages(
    root: Path, attempt: Path, receipt: dict[str, Any], unit: dict[str, Any], maximum_pages: int,
) -> tuple[list[dict[str, Any]], str | None]:
    records = receipt["pages"]
    if not isinstance(records, list) or len(records) > maximum_pages:
        raise DataReadinessError("transfer attempt page count exceeds contract")
    rows: list[dict[str, Any]] = []
    token = None
    seen: set[str] = set()
    expected_files = {"_receipt.json"}
    for ordinal, record in enumerate(records):
        if set(record) != {"body_path", "metadata_path", "metadata_sha256"} or record["metadata_path"] != f"{ordinal:03d}.json":
            raise DataReadinessError("transfer page ordinal or record contract differs")
        if ordinal and (token is None or token in seen):
            raise DataReadinessError("transfer page chain continues after terminal or repeated token")
        if token is not None:
            seen.add(token)
        metadata_path = resolve_inside_authority(root, attempt / record["metadata_path"])
        if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != record["metadata_sha256"]:
            raise DataReadinessError("transfer page metadata hash differs")
        metadata = _object(metadata_path)
        if record["body_path"] != f"{ordinal:03d}-{metadata['body_sha256']}.bin":
            raise DataReadinessError("transfer page body name differs")
        body_path = resolve_inside_authority(root, attempt / record["body_path"])
        if body_path.stat().st_size > _MAX_PAGE_BYTES:
            raise DataReadinessError("transfer page body exceeds limit")
        params = {**unit["parameters"], **({"page_token": token} if token is not None else {})}
        page = decode_bars_page_response(_http_response(body_path.read_bytes(), metadata), expected_params=params)
        if (page.retrieved_at_utc is None or not datetime.fromisoformat(receipt["started_at_utc"]) <= page.retrieved_at_utc
                <= datetime.fromisoformat(receipt["completed_at_utc"])):
            raise DataReadinessError("transfer response retrieval clock falls outside its attempt")
        rows.extend(page.bars.get(unit["ticker"], ()))
        token = page.next_page_token
        expected_files.update({record["metadata_path"], record["body_path"]})
    if {path.name for path in attempt.iterdir()} != expected_files:
        raise DataReadinessError("transfer attempt has unbound files")
    if receipt["state"] == "acquired" and (not records or token is not None):
        raise DataReadinessError("transfer acquired attempt lacks a terminal page")
    return rows, token


def compare_transfer_observations(unit: dict[str, Any], raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report exact OHLCV differences; missing data never becomes a neutral return."""
    observed: dict[str, dict[str, float]] = {}
    issues: list[dict[str, Any]] = []
    required = set(unit["required_sessions"])
    names = {"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}
    for raw in raw_rows:
        try:
            timestamp = pd.Timestamp(raw["t"])
            if timestamp.tzinfo is None or pd.isna(timestamp):
                raise ValueError("missing timezone")
            local = timestamp.tz_convert("America/New_York")
            session = local.date().isoformat()
            if local.time() != datetime.min.time():
                raise ValueError("daily source timestamp is not session midnight")
            values = {name: float(raw[key]) for name, key in names.items()}
        except (KeyError, ValueError, TypeError, OverflowError):
            issues.append({"code": "malformed_daily_observation"})
            continue
        if session not in required:
            issues.append({"code": "outside_required_sessions", "session": session})
            continue
        if session in observed:
            issues.append({"code": "duplicate_session", "session": session})
            continue
        if (not all(math.isfinite(value) and value > 0 for value in values.values())
                or values["low"] > min(values["open"], values["close"], values["high"])
                or values["high"] < max(values["open"], values["close"], values["low"])):
            issues.append({"code": "unusable_daily_observation", "session": session})
            continue
        observed[session] = values
    missing = sorted(required.difference(observed))
    differences: list[dict[str, Any]] = []
    for retained in unit["retained_rows"]:
        session = retained["session_date_et"]
        if session in observed:
            for name in names:
                if observed[session][name] != retained[name]:
                    differences.append({"session": session, "field": name, "retained": retained[name],
                                        "replay": observed[session][name], "delta": observed[session][name] - retained[name]})
    return {
        "status": "exact_match" if not issues and not missing and not differences else "review_required",
        "observed_rows": len(raw_rows), "valid_unique_sessions": len(observed), "required_sessions": len(required),
        "missing_sessions": missing, "issues": issues, "differences": differences,
        "retained_observations_replaced": False, "identity_admission": False, "accounting_eligible": False,
    }


def _verify_attempt(root: Path, attempt: Path, unit: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    path = resolve_inside_authority(root, attempt / "_receipt.json")
    receipt = _object(path)
    digest = receipt.pop("receipt_sha256", None)
    fields = {"schema_version", "request_sha256", "unit_sha256", "attempt_id", "started_at_utc",
              "completed_at_utc", "state", "error_type", "pages"}
    if (set(receipt) != fields or digest != json_sha256(receipt) or receipt.get("unit_sha256") != unit["unit_sha256"]
            or receipt.get("request_sha256") != request["request_sha256"] or receipt.get("attempt_id") != attempt.name
            or receipt.get("schema_version") != "market_predictor.swing_transfer_attempt"
            or receipt.get("state") not in {"acquired", "failed"}):
        raise DataReadinessError("transfer attempt identity or hash mismatch")
    if ((receipt["state"] == "acquired" and receipt.get("error_type") is not None)
            or (receipt["state"] == "failed" and not isinstance(receipt.get("error_type"), str))):
        raise DataReadinessError("transfer attempt failure status differs")
    started = datetime.fromisoformat(receipt["started_at_utc"])
    completed = datetime.fromisoformat(receipt["completed_at_utc"])
    if started.utcoffset() != UTC.utcoffset(None) or completed.utcoffset() != UTC.utcoffset(None) or started > completed:
        raise DataReadinessError("transfer attempt clocks must be ordered UTC timestamps")
    rows, _ = _read_pages(root, attempt, receipt, unit, request["maximum_pages_per_ticker"])
    comparison = compare_transfer_observations(unit, rows) if receipt["state"] == "acquired" else None
    return {
        "receipt_path": path.relative_to(root).as_posix(), "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "state": receipt["state"], "comparison": comparison,
    }


def _report(request: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    acquired = sum(int(row["acquired"]) for row in results)
    return {
        "schema_version": "market_predictor.swing_transfer_replay_report", "request_sha256": request["request_sha256"],
        "acquisition_status": "complete" if acquired == len(results) else "incomplete",
        "requested_units": len(results), "acquired_units": acquired, "units": results,
        "exact_match_units": sum(int(row["comparison"] is not None and row["comparison"]["status"] == "exact_match") for row in results),
        "identity_admission": False, "accounting_eligible": False, "retained_observations_replaced": False,
    }


def _unit_result(ticker: str, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [item for item in attempts if item["state"] == "acquired"]
    if len(successes) > 1 or len({item["receipt_path"] for item in attempts}) != len(attempts):
        raise DataReadinessError("transfer replay has duplicate or multiple successful attempts for one unit")
    return {"ticker": ticker, "acquired": bool(successes), "attempts": attempts,
            "comparison": successes[0]["comparison"] if successes else None}


def _verify_collection(output: Path, request: dict[str, Any]) -> dict[str, Any]:
    if _object(resolve_inside_authority(output, "_request.json")) != request:
        raise DataReadinessError("transfer replay request changed; use a new immutable directory")
    units_root = output / "units"
    tickers = {unit["ticker"] for unit in request["units"]}
    if units_root.exists() and any(path.name not in tickers or not path.is_dir() for path in units_root.iterdir()):
        raise DataReadinessError("transfer replay has unknown units")
    results = []
    for unit in request["units"]:
        parent = units_root / unit["ticker"]
        attempts = [_verify_attempt(output, path, unit, request) for path in sorted(parent.iterdir())] if parent.exists() else []
        results.append(_unit_result(unit["ticker"], attempts))
    report = _report(request, results)
    reports = output / "reports"
    for path in reports.iterdir() if reports.exists() else []:
        path = resolve_inside_authority(output, path)
        prior = _object(path)
        if path.name != f"{hashlib.sha256(path.read_bytes()).hexdigest()}.json" or prior.get("request_sha256") != request["request_sha256"]:
            raise DataReadinessError("transfer replay report hash or request differs")
        if len(prior.get("units", [])) != len(results):
            raise DataReadinessError("transfer replay report unit inventory differs")
        historical = []
        for old, current in zip(prior["units"], results, strict=True):
            if old["ticker"] != current["ticker"] or any(item not in current["attempts"] for item in old["attempts"]):
                raise DataReadinessError("transfer replay report receipt anchor differs")
            historical.append(_unit_result(old["ticker"], old["attempts"]))
        if prior != _report(request, historical):
            raise DataReadinessError("transfer replay report semantics differ from its receipts")
    return report


def _collect_unit(output: Path, request: dict[str, Any], unit: dict[str, Any], fetch: TransferPageFetcher) -> None:
    attempt_id = uuid4().hex
    staging = output / ".pending" / attempt_id
    staging.mkdir(parents=True)
    started = datetime.now(UTC)
    records: list[dict[str, Any]] = []
    token = None
    seen: set[str] = set()
    error_type = None
    for ordinal in range(request["maximum_pages_per_ticker"]):
        if token is not None:
            if token in seen:
                error_type = "RepeatedPaginationToken"
                break
            seen.add(token)
        try:
            page = fetch(unit, token)
        except (DataReadinessError, RuntimeError, ValueError, requests.RequestException, OSError) as exc:
            error_type = type(exc).__name__
            break
        try:
            record, decoded = _store_page(staging, page, unit, token, ordinal)
        except (DataReadinessError, RuntimeError, ValueError) as exc:
            error_type = type(exc).__name__
            break
        records.append(record)
        token = decoded.next_page_token
        del page, decoded
        assert_memory_budget(stage="transfer page archive", hard_budget_gib=5.0, headroom_gib=0.75)
        if token is None:
            break
    if error_type is None and token is not None:
        error_type = "PaginationPageCap"
    receipt = {
        "schema_version": "market_predictor.swing_transfer_attempt", "request_sha256": request["request_sha256"],
        "unit_sha256": unit["unit_sha256"], "attempt_id": attempt_id,
        "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(UTC).isoformat(),
        "state": "failed" if error_type else "acquired", "error_type": error_type, "pages": records,
    }
    write_json_object(staging / "_receipt.json", {**receipt, "receipt_sha256": json_sha256(receipt)})
    _verify_attempt(output, staging, unit, request)
    destination = output / "units" / unit["ticker"] / attempt_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, destination)


def run_swing_transfer_replay(
    *, root: Path, config_path: Path, output_directory: Path, fetch: TransferPageFetcher | None = None,
) -> dict[str, Any]:
    """Acquire/replay small evidence units under the shared nonqueueing root lease."""
    root = root.resolve()
    output = output_directory.resolve()
    if not output.is_relative_to(root):
        raise DataReadinessError("transfer replay output must stay inside the repository")
    runtime = heavy_job_runtime_dir()
    runtime = runtime if runtime.is_absolute() else root / runtime
    with heavy_job_lease("swing-transfer-replay", runtime_dir=runtime, config_path=config_path):
        with file_lock(output.parent / f".{output.name}.collection", timeout=0):
            request = _request(root, config_path)
            if not output.exists():
                if fetch is None:
                    raise DataReadinessError("offline transfer replay requires an existing collection")
                staging = output.parent / f".{output.name}.{uuid4().hex}.pending"
                staging.mkdir(parents=True)
                write_json_object(staging / "_request.json", request)
                os.rename(staging, output)
            report = _verify_collection(output, request)
            if fetch is None:
                return report
            complete = {unit["ticker"] for unit in report["units"] if unit["acquired"]}
            for unit in request["units"]:
                if unit["ticker"] not in complete:
                    _collect_unit(output, request, unit, fetch)
            for path, digest in request["bound_files"].items():
                checked_transfer_file(root, path, digest, {})
            report = _verify_collection(output, request)
            assert_peak_memory_budget(stage="transfer replay report", hard_budget_gib=5.0, headroom_gib=0.75)
            reports = output / "reports"
            reports.mkdir(exist_ok=True)
            temporary = output / ".pending" / f"{uuid4().hex}.json"
            temporary.parent.mkdir(exist_ok=True)
            write_json_object(temporary, report)
            report_path = reports / f"{hashlib.sha256(temporary.read_bytes()).hexdigest()}.json"
            if report_path.exists():
                if report_path.read_bytes() != temporary.read_bytes():
                    raise DataReadinessError("transfer report hash collision")
                temporary.unlink()
            else:
                os.rename(temporary, report_path)
            return {**report, "report_path": str(report_path)}
