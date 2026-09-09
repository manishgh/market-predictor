"""Resumable provider evidence, never inferred ownership or settlement accounting."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import requests
from pydantic import BaseModel, ConfigDict, Field

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.locking import file_lock
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget
from market_predictor.sources.alpaca_corporate_actions import corporate_action_parameters, decode_corporate_actions_page
from market_predictor.sources.http import HttpByteResponse
from market_predictor.swing.contracts.research_cohort import Sha256

ActionFetcher = Callable[[dict[str, Any], int], HttpByteResponse]


class _Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["market_predictor.swing_corporate_action_collection"]
    observation_inventory: str = Field(min_length=1)
    observation_audit_sha256: Sha256
    process_start: date
    process_end: date
    expected_securities: int = Field(ge=1, le=2000)
    maximum_pages_per_ticker: int = Field(ge=1, le=100)
    page_limit: int = Field(ge=1, le=1000)
    maximum_response_bytes: int = Field(ge=1024, le=4 * 1024**2)


def _object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 8 * 1024**2:
        raise DataReadinessError("corporate-action metadata exceeds size limit")
    return parse_strict_json_object(path.read_bytes(), label=str(path))


def _verify_sources(root: Path, files: dict[str, str]) -> None:
    for name, digest in files.items():
        if file_sha256(resolve_inside_authority(root, name)) != digest:
            raise DataReadinessError("corporate-action source binding differs")


def _prepare(root: Path, config: Path) -> dict[str, Any]:
    if config.stat().st_size > 1024**2:
        raise DataReadinessError("corporate-action configuration exceeds size limit")
    policy = _Policy.model_validate_json(json.dumps(tomllib.loads(config.read_text(encoding="utf-8"))))
    path = resolve_inside_authority(root, policy.observation_inventory)
    report = _object(path)
    if (report.get("audit_sha256") != policy.observation_audit_sha256
            or json_sha256({k: v for k, v in report.items() if k != "audit_sha256"}) != policy.observation_audit_sha256
            or report.get("scope") != "initial_fit_flagged_holding_observation_diagnostic"
            or report.get("securities") != policy.expected_securities
            or policy.process_end.isoformat() != report.get("numeric_end")
            or not date(2019, 7, 9) <= policy.process_start <= policy.process_end):
        raise DataReadinessError("corporate-action collection requires the pinned initial-fit inventory")
    bound = dict(report["source_files"])
    bound[path.relative_to(root).as_posix()] = file_sha256(path)
    bound[config.relative_to(root).as_posix()] = file_sha256(config)
    _verify_sources(root, bound)
    _verify_sources(path.parent, report["outputs"])
    for relative, digest in report["outputs"].items():
        bound[(path.parent / relative).relative_to(root).as_posix()] = digest
    cases = report["cases"]
    tickers = sorted({row["ticker"] for row in cases})
    if (not tickers or len(tickers) > 2000 or len({row["security_id"] for row in cases}) != policy.expected_securities
            or any(not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", ticker) for ticker in tickers)):
        raise DataReadinessError("corporate-action ticker inventory differs")
    package = Path(__file__).resolve().parents[2]
    implementation = {name: file_sha256(package / name) for name in (
        "swing/datasets/corporate_action_collection.py", "sources/alpaca_corporate_actions.py", "sources/http.py",
        "core/errors.py", "resources.py",
    )}
    request = {"policy": policy.model_dump(mode="json"), "tickers": tickers, "bound_files": bound,
        "implementation_files": implementation, "scope": "initial_fit_provider_process_date_evidence_only"}
    return {**request, "request_sha256": json_sha256(request)}


def _params(request: dict[str, Any], ticker: str, token: str | None) -> dict[str, Any]:
    policy = request["policy"]
    return corporate_action_parameters(ticker=ticker, start=date.fromisoformat(policy["process_start"]),
        end=date.fromisoformat(policy["process_end"]), page_token=token, limit=policy["page_limit"])


def _response(body: bytes, metadata: dict[str, Any]) -> HttpByteResponse:
    values = {**metadata, "body": body, "retrieved_at_utc": datetime.fromisoformat(metadata["retrieved_at_utc"]),
        "redirect_chain": tuple(metadata["redirect_chain"]), "safe_headers": tuple(tuple(row) for row in metadata["safe_headers"])}
    return HttpByteResponse(**values)


def _check_attempt(attempt: Path, request: dict[str, Any], ticker: str) -> dict[str, Any]:
    receipt = _object(attempt / "receipt.json")
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if (receipt.get("receipt_sha256") != json_sha256(unsigned)
            or receipt.get("request_sha256") != request["request_sha256"] or receipt.get("ticker") != ticker
            or receipt.get("attempt_id") != attempt.name or receipt.get("state") not in {"acquired", "failed"}
            or not isinstance(receipt.get("pages"), list)
            or len(receipt["pages"]) > request["policy"]["maximum_pages_per_ticker"]):
        raise DataReadinessError("corporate-action receipt identity differs")
    start, end = (datetime.fromisoformat(receipt[key]) for key in ("started_at_utc", "completed_at_utc"))
    if (start.utcoffset() != UTC.utcoffset(None) or end.utcoffset() != UTC.utcoffset(None)
            or not start <= end <= datetime.now(UTC)):
        raise DataReadinessError("corporate-action receipt clock differs")
    token = None
    seen_tokens: set[str] = set()
    seen_ids: set[str] = set()
    counts: Counter[str] = Counter()
    expected_files = {"receipt.json"}
    decode_error = None
    for ordinal, page in enumerate(receipt["pages"]):
        if (page.get("body_path") != f"{ordinal:03d}.bin" or page.get("metadata_path") != f"{ordinal:03d}.json"
                or ordinal and (token is None or token in seen_tokens) or decode_error is not None):
            raise DataReadinessError("corporate-action page chain differs")
        if token is not None:
            seen_tokens.add(token)
        metadata_path = resolve_inside_authority(attempt, page["metadata_path"])
        if file_sha256(metadata_path) != page["metadata_sha256"]:
            raise DataReadinessError("corporate-action metadata hash differs")
        body_path = resolve_inside_authority(attempt, page["body_path"])
        if body_path.stat().st_size > request["policy"]["maximum_response_bytes"]:
            raise DataReadinessError("corporate-action response exceeds limit")
        metadata = _object(metadata_path)
        body = body_path.read_bytes()
        if hashlib.sha256(body).hexdigest() != page["body_sha256"] or len(body) != page["body_length"]:
            raise DataReadinessError("corporate-action archived body differs")
        try:
            response = _response(body, metadata)
            if (response.retrieved_at_utc.utcoffset() != UTC.utcoffset(None)
                    or not start <= response.retrieved_at_utc <= end):
                raise DataReadinessError("corporate-action response clock outside attempt")
            families, token = decode_corporate_actions_page(response, expected_params=_params(request, ticker, token))
            for family, rows in families.items():
                for row in rows:
                    identity = row.get("id")
                    if isinstance(identity, str) and identity.strip():
                        if identity in seen_ids:
                            raise DataReadinessError("duplicate action ID across pages")
                        seen_ids.add(identity)
                counts[family] += len(rows)
        except (DataReadinessError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            decode_error = type(exc).__name__
        expected_files.update((page["metadata_path"], page["body_path"]))
    if {path.name for path in attempt.iterdir()} != expected_files:
        raise DataReadinessError("corporate-action attempt contains unbound files")
    if receipt["state"] == "acquired":
        if not receipt["pages"] or token is not None or decode_error is not None or receipt.get("error_type") is not None:
            raise DataReadinessError("corporate-action acquisition lacks valid terminal evidence")
    elif not isinstance(receipt.get("error_type"), str) or decode_error and decode_error != receipt["error_type"]:
        raise DataReadinessError("corporate-action failure receipt differs")
    return {"attempt": attempt.name, "receipt_sha256": file_sha256(attempt / "receipt.json"),
        "state": receipt["state"], "error_type": receipt["error_type"], "counts": dict(counts)}


def _collect(output: Path, request: dict[str, Any], ticker: str, fetch: ActionFetcher) -> None:
    attempt = output / ".pending" / uuid4().hex
    attempt.mkdir(parents=True)
    started = datetime.now(UTC)
    pages: list[dict[str, Any]] = []
    token = None
    seen_tokens: set[str] = set()
    seen_ids: set[str] = set()
    error = None
    for ordinal in range(request["policy"]["maximum_pages_per_ticker"]):
        if token is not None:
            if token in seen_tokens:
                error = "RepeatedPaginationToken"
                break
            seen_tokens.add(token)
        try:
            response = fetch(_params(request, ticker, token), request["policy"]["maximum_response_bytes"])
            if len(response.body) > request["policy"]["maximum_response_bytes"]:
                raise DataReadinessError("corporate-action response exceeds limit")
            metadata = asdict(response)
            del metadata["body"]
            metadata["retrieved_at_utc"] = response.retrieved_at_utc.isoformat()
            body_name, metadata_name = f"{ordinal:03d}.bin", f"{ordinal:03d}.json"
            (attempt / body_name).write_bytes(response.body)
            write_json_object(attempt / metadata_name, metadata)
            pages.append({"body_path": body_name, "metadata_path": metadata_name,
                "body_sha256": hashlib.sha256(response.body).hexdigest(), "body_length": len(response.body),
                "metadata_sha256": file_sha256(attempt / metadata_name)})
            if (response.retrieved_at_utc.utcoffset() != UTC.utcoffset(None)
                    or not started <= response.retrieved_at_utc <= datetime.now(UTC)):
                raise DataReadinessError("corporate-action response clock outside attempt")
            families, token = decode_corporate_actions_page(response, expected_params=_params(request, ticker, token))
            for rows in families.values():
                for row in rows:
                    identity = row.get("id")
                    if isinstance(identity, str) and identity.strip():
                        if identity in seen_ids:
                            raise DataReadinessError("duplicate action ID across pages")
                        seen_ids.add(identity)
        except MemoryBudgetError:
            raise
        except (DataReadinessError, RuntimeError, ValueError, TypeError, KeyError, requests.RequestException, OSError) as exc:
            error = type(exc).__name__
            break
        assert_memory_budget(stage="corporate-action collection", hard_budget_gib=5.0, headroom_gib=0.75)
        if token is None:
            break
    if error is None and token is not None:
        error = "PaginationPageCap"
    receipt = {"request_sha256": request["request_sha256"], "ticker": ticker, "attempt_id": attempt.name,
        "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(UTC).isoformat(),
        "state": "failed" if error else "acquired", "error_type": error, "pages": pages}
    write_json_object(attempt / "receipt.json", {**receipt, "receipt_sha256": json_sha256(receipt)})
    _check_attempt(attempt, request, ticker)
    parent = output / "tickers" / ticker
    parent.mkdir(parents=True, exist_ok=True)
    os.rename(attempt, parent / attempt.name)


def _report(output: Path, request: dict[str, Any]) -> dict[str, Any]:
    results = []
    if (output / "tickers").exists() and any(path.name not in request["tickers"] for path in (output / "tickers").iterdir()):
        raise DataReadinessError("corporate-action collection contains an unknown ticker")
    for ticker in request["tickers"]:
        parent = output / "tickers" / ticker
        attempts = [_check_attempt(path, request, ticker) for path in sorted(parent.iterdir())] if parent.exists() else []
        successes = [row for row in attempts if row["state"] == "acquired"]
        if len(successes) > 1:
            raise DataReadinessError("corporate-action collection contains duplicate successful attempts")
        results.append({"ticker": ticker, "acquired": bool(successes), "attempts": attempts,
            "counts": successes[0]["counts"] if successes else {}})
    counts: Counter[str] = Counter()
    for row in results:
        counts.update(row["counts"])
    report = {"request_sha256": request["request_sha256"], "tickers": results, "action_counts": dict(counts),
        "requested_tickers": len(results), "acquired_tickers": sum(row["acquired"] for row in results),
        "status": "collected_unreviewed" if all(row["acquired"] for row in results) else "incomplete",
        "historical_announcement_availability_proven": False, "absence_of_actions_proven": False,
        "ownership_admitted": False, "accounting_eligible": False}
    return {**report, "audit_sha256": json_sha256(report)}


def collect_holding_corporate_actions(
    root: Path, config_path: Path, output_directory: Path, *, fetch: ActionFetcher | None = None,
    expected_audit_sha256: str | None = None,
) -> dict[str, Any]:
    """Serialize acquisition; offline mode reconstructs receipts and checks an external pin."""
    root = root.resolve()
    config = resolve_inside_authority(root, str(config_path))
    output = (root / output_directory).resolve()
    runtime = (root / heavy_job_runtime_dir()).resolve()
    if not output.is_relative_to(root) or not runtime.is_relative_to(root):
        raise DataReadinessError("corporate-action paths must remain inside the workspace")
    with heavy_job_lease("collect-swing-holding-corporate-actions", runtime_dir=runtime):
        with file_lock(output.parent / f".{output.name}.collection", timeout=0):
            assert_memory_budget(stage="corporate-action input loading", hard_budget_gib=5.0, headroom_gib=0.75)
            request = _prepare(root, config)
            if output.exists() and expected_audit_sha256 is None:
                raise DataReadinessError("corporate-action resume requires an independent audit pin")
            if not output.exists():
                if fetch is None:
                    raise DataReadinessError("offline corporate-action replay requires an existing collection")
                staging = output.with_name(f".{output.name}.{uuid4().hex}.pending")
                staging.mkdir(parents=True)
                write_json_object(staging / "_request.json", request)
                os.rename(staging, output)
            if _object(output / "_request.json") != request:
                raise DataReadinessError("corporate-action collection request differs; use a new directory")
            report = _report(output, request)
            if expected_audit_sha256 is not None and report["audit_sha256"] != expected_audit_sha256:
                raise DataReadinessError("corporate-action independent audit pin differs")
            if fetch is None:
                if expected_audit_sha256 is None:
                    raise DataReadinessError("offline corporate-action replay requires an independent audit pin")
                return report
            completed = {row["ticker"] for row in report["tickers"] if row["acquired"]}
            for ticker in request["tickers"]:
                if ticker not in completed:
                    _collect(output, request, ticker, fetch)
            _verify_sources(root, request["bound_files"])
            _verify_sources(Path(__file__).resolve().parents[2], request["implementation_files"])
            report = _report(output, request)
            assert_peak_memory_budget(stage="corporate-action report", hard_budget_gib=5.0, headroom_gib=0.75)
            reports = output / "reports"
            reports.mkdir(exist_ok=True)
            path = reports / f"{report['audit_sha256']}.json"
            if path.exists() and _object(path) != report:
                raise DataReadinessError("immutable corporate-action report differs")
            if not path.exists():
                temporary = reports / f".{uuid4().hex}.pending"
                write_json_object(temporary, report)
                os.rename(temporary, path)
            return report
