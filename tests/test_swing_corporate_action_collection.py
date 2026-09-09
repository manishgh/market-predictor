"""Synthetic, independently pinned evidence tests; no provider or real-data access."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest
import requests

from market_predictor.canonical.store import file_sha256
from market_predictor.core import errors
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.sources.http import HttpByteResponse
from market_predictor.swing.datasets import corporate_action_collection as collector

Fetcher = Callable[[dict[str, Any], int], HttpByteResponse]


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resign(path: Path, value: dict[str, Any], field: str) -> None:
    value.pop(field, None)
    value[field] = json_sha256(value)
    _write(path, value)


def _config(fixture: dict[str, Any], **changes: Any) -> None:
    fixture["policy"].update(changes)
    fixture["config"].write_text(
        "".join(f"{key} = {json.dumps(value)}\n" for key, value in fixture["policy"].items()), encoding="utf-8",
    )


@pytest.fixture
def inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    source = tmp_path / "source.json"
    source.write_bytes(b'{"synthetic_source":true}')
    parent = tmp_path / "inventory"
    parent.mkdir()
    observations = parent / "observations.json"
    observations.write_bytes(b'{"synthetic_observations":[]}')
    manifest = parent / "_manifest.json"
    report = {
        "scope": "initial_fit_flagged_holding_observation_diagnostic", "securities": 2,
        "numeric_end": "2024-01-31", "source_files": {"source.json": file_sha256(source)},
        "outputs": {"observations.json": file_sha256(observations)},
        "cases": [{"ticker": "AAA", "security_id": "synthetic-a"}, {"ticker": "BBB", "security_id": "synthetic-b"}],
    }
    _resign(manifest, report, "audit_sha256")
    fixture = {
        "root": tmp_path, "config": tmp_path / "policy.toml", "output": tmp_path / "collection",
        "source": source, "observations": observations, "manifest": manifest,
        "policy": {
            "schema_version": "market_predictor.swing_corporate_action_collection",
            "observation_inventory": "inventory/_manifest.json", "observation_audit_sha256": report["audit_sha256"],
            "process_start": "2024-01-01", "process_end": "2024-01-31", "expected_securities": 2,
            "maximum_pages_per_ticker": 3, "page_limit": 10, "maximum_response_bytes": 2048,
        },
    }
    _config(fixture)
    monkeypatch.setattr(collector, "heavy_job_runtime_dir", lambda: tmp_path / "runtime")

    def no_network(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("synthetic collection must never access the network")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    return fixture


def _run(fixture: dict[str, Any], fetch: Fetcher | None = None, expected: str | None = None) -> dict[str, Any]:
    return collector.collect_holding_corporate_actions(
        fixture["root"], fixture["config"], fixture["output"], fetch=fetch, expected_audit_sha256=expected,
    )


def _page(params: dict[str, Any], actions: dict[str, Any] | None = None, token: str | None = None) -> HttpByteResponse:
    body = json.dumps({"corporate_actions": actions or {}, "next_page_token": token}).encode()
    url = "https://data.alpaca.markets/v1/corporate-actions?" + urlencode(params)
    return HttpByteResponse(
        body=body, requested_url=url, final_url=url, redirect_chain=(), status_code=200,
        retrieved_at_utc=datetime.now(UTC), content_type="application/json", content_encoding=None,
        etag=None, last_modified=None, body_length=len(body), sha256=sha256(body).hexdigest(),
        body_representation="http_entity_encoded", safe_headers=(("content-type", "application/json"),),
    )


def _empty(params: dict[str, Any], bound: int) -> HttpByteResponse:
    assert bound == 2048
    return _page(params)


def _forbidden(params: dict[str, Any], bound: int) -> HttpByteResponse:
    pytest.fail("completed or invalid evidence must not trigger a fetch")


def _ticker(report: dict[str, Any], ticker: str) -> dict[str, Any]:
    return next(row for row in report["tickers"] if row["ticker"] == ticker)


def _attempt(fixture: dict[str, Any], ticker: str = "AAA") -> Path:
    paths = list((fixture["output"] / "tickers" / ticker).iterdir())
    assert len(paths) == 1
    return paths[0]


def _receipt_rebound_pin(report: dict[str, Any], attempt: Path) -> str:
    # Deliberately pin corrupt chronology to test validation independently of hash rejection.
    rebound = json.loads(json.dumps(report))
    receipt = _read(attempt / "receipt.json")
    entry = _ticker(rebound, receipt["ticker"])["attempts"][0]
    assert entry["attempt"] == attempt.name
    entry["receipt_sha256"] = file_sha256(attempt / "receipt.json")
    rebound.pop("audit_sha256")
    return json_sha256(rebound)


@pytest.mark.parametrize("failure", ["exception", "http_503", "invalid_json"])
def test_one_ticker_failure_is_isolated_and_only_failure_retried(inventory: dict[str, Any], failure: str) -> None:
    calls = []

    def fetch(params: dict[str, Any], bound: int) -> HttpByteResponse:
        assert bound == 2048
        calls.append(params["symbols"])
        if params["symbols"] == "AAA":
            if failure == "exception":
                raise requests.ConnectionError("synthetic provider failure")
            response = _page(params)
            if failure == "http_503":
                return replace(response, status_code=503)
            body = b"not-json"
            return replace(response, body=body, body_length=len(body), sha256=sha256(body).hexdigest())
        return _page(params, {"cash_dividends": [{"id": "synthetic-b-1", "rate": "0.25"}]})

    report = _run(inventory, fetch)
    assert calls == ["AAA", "BBB"]
    assert report["status"] == "incomplete"
    assert report["acquired_tickers"] == 1
    assert _ticker(report, "AAA")["attempts"][0]["state"] == "failed"
    assert _ticker(report, "BBB")["acquired"] is True
    assert report["action_counts"] == {"cash_dividends": 1}
    assert _run(inventory, expected=report["audit_sha256"]) == report
    calls.clear()

    def retry(params: dict[str, Any], bound: int) -> HttpByteResponse:
        calls.append(params["symbols"])
        return _empty(params, bound)

    resumed = _run(inventory, retry, report["audit_sha256"])
    assert calls == ["AAA"]
    assert resumed["status"] == "collected_unreviewed"
    assert len(_ticker(resumed, "AAA")["attempts"]) == 2
    assert len(_ticker(resumed, "BBB")["attempts"]) == 1


def test_terminal_empty_counts_are_not_absence_or_admission_proof(inventory: dict[str, Any]) -> None:
    report = _run(inventory, _empty)
    assert report["requested_tickers"] == report["acquired_tickers"] == 2
    assert report["action_counts"] == {}
    assert report["status"] == "collected_unreviewed"
    for field in (
        "absence_of_actions_proven", "historical_announcement_availability_proven", "ownership_admitted", "accounting_eligible",
    ):
        assert report[field] is False
    assert len(_read(_attempt(inventory) / "receipt.json")["pages"]) == 1


def test_complete_pagination_exact_query_resume_and_independently_pinned_replay(inventory: dict[str, Any]) -> None:
    calls = []

    def fetch(params: dict[str, Any], bound: int) -> HttpByteResponse:
        assert bound == 2048
        calls.append(dict(params))
        assert params == {
            "symbols": params["symbols"], "start": "2024-01-01", "end": "2024-01-31", "region": "us",
            "data_quality": "all", "limit": 10, "sort": "asc",
            **({"page_token": "opaque+/= next"} if "page_token" in params else {}),
        }
        if params["symbols"] == "BBB":
            return _page(params)
        if "page_token" not in params:
            return _page(params, {"cash_mergers": [{"id": "synthetic-1", "process_date": "2024-01-10",
                "effective_date": "2024-01-09", "payable_date": "2024-01-12", "rate": "12.34"}]}, "opaque+/= next")
        return _page(params, {"unknown_family": [{"id": "synthetic-2", "uninterpreted": True}]})

    report = _run(inventory, fetch)
    assert [row["symbols"] for row in calls] == ["AAA", "AAA", "BBB"]
    assert report["action_counts"] == {"cash_mergers": 1, "unknown_family": 1}
    before = {path: file_sha256(path) for path in inventory["output"].rglob("*") if path.is_file()}
    assert _run(inventory, _forbidden, report["audit_sha256"]) == report
    assert _run(inventory, expected=report["audit_sha256"]) == report
    assert {path: file_sha256(path) for path in before} == before
    with pytest.raises(DataReadinessError, match="independent"):
        _run(inventory)
    with pytest.raises(DataReadinessError, match="pin"):
        _run(inventory, expected="0" * 64)


@pytest.mark.parametrize("target", ["body", "metadata", "receipt"])
def test_archived_bytes_metadata_and_receipt_tamper_rejected(inventory: dict[str, Any], target: str) -> None:
    report = _run(inventory, _empty)
    attempt = _attempt(inventory)
    if target == "body":
        (attempt / "000.bin").write_bytes(b"tampered")
    elif target == "metadata":
        path = attempt / "000.json"
        value = _read(path)
        value["status_code"] = 201
        _write(path, value)
    else:
        path = attempt / "receipt.json"
        value = _read(path)
        value["state"] = "failed"
        _write(path, value)
    with pytest.raises(DataReadinessError):
        _run(inventory, _forbidden, report["audit_sha256"])


@pytest.mark.parametrize("online", [False, True])
def test_rehashed_body_metadata_receipt_cannot_replace_external_counts_pin(inventory: dict[str, Any], online: bool) -> None:
    report = _run(inventory, _empty)
    attempt = _attempt(inventory)
    body = json.dumps({"corporate_actions": {"cash_dividends": [{"id": "forged"}]}, "next_page_token": None}).encode()
    (attempt / "000.bin").write_bytes(body)
    metadata = _read(attempt / "000.json")
    metadata.update(body_length=len(body), sha256=sha256(body).hexdigest())
    _write(attempt / "000.json", metadata)
    receipt = _read(attempt / "receipt.json")
    receipt["pages"][0].update(
        metadata_sha256=file_sha256(attempt / "000.json"), body_sha256=sha256(body).hexdigest(), body_length=len(body),
    )
    _resign(attempt / "receipt.json", receipt, "receipt_sha256")
    with pytest.raises(DataReadinessError, match="pin"):
        _run(inventory, _forbidden if online else None, report["audit_sha256"])


@pytest.mark.parametrize("kind", ["repeated_token", "page_cap", "duplicate_id"])
def test_invalid_pagination_is_failed_evidence_and_other_ticker_completes(inventory: dict[str, Any], kind: str) -> None:
    calls = []

    def fetch(params: dict[str, Any], bound: int) -> HttpByteResponse:
        assert bound == 2048
        calls.append(params["symbols"])
        if params["symbols"] == "BBB":
            return _page(params)
        ordinal = calls.count("AAA")
        if kind == "repeated_token":
            return _page(params, token="repeat")
        if kind == "page_cap":
            return _page(params, token=f"page-{ordinal}")
        return _page(params, {"cash_dividends": [{"id": "duplicate"}]}, "second" if ordinal == 1 else None)

    report = _run(inventory, fetch)
    assert calls.count("AAA") == (3 if kind == "page_cap" else 2)
    assert calls[-1] == "BBB"
    assert report["status"] == "incomplete"
    assert report["acquired_tickers"] == 1
    assert report["action_counts"] == {}
    assert _ticker(report, "AAA")["attempts"][0]["state"] == "failed"
    assert _run(inventory, expected=report["audit_sha256"]) == report


@pytest.mark.parametrize("target", ["source", "observations", "manifest", "config"])
def test_exact_input_file_pins_rejected_before_resume_fetch(inventory: dict[str, Any], target: str) -> None:
    report = _run(inventory, _empty)
    path = inventory[target]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(DataReadinessError):
        _run(inventory, _forbidden, report["audit_sha256"])


@pytest.mark.parametrize("target", ["source", "observations", "manifest"])
def test_input_hashes_verified_before_first_fetch(inventory: dict[str, Any], target: str) -> None:
    path = inventory[target]
    if target == "manifest":
        report = _read(path)
        report["securities"] = 7
        _resign(path, report, "audit_sha256")
    else:
        path.write_bytes(b"unbound replacement")
    with pytest.raises(DataReadinessError):
        _run(inventory, _forbidden)
    assert not inventory["output"].exists()


def test_input_report_requires_external_digest_in_config(inventory: dict[str, Any]) -> None:
    inventory["policy"].pop("observation_audit_sha256")
    _config(inventory)
    with pytest.raises(ValueError):
        _run(inventory, _forbidden)
    assert not inventory["output"].exists()


@pytest.mark.parametrize("changes", [
    {"process_start": "2019-07-08"}, {"process_start": "2024-02-01"},
    {"process_end": "2024-01-30"}, {"process_end": "2024-02-01"},
])
def test_scoped_dates_cannot_precede_decisions_or_differ_from_numeric_end(
    inventory: dict[str, Any], changes: dict[str, str],
) -> None:
    _config(inventory, **changes)
    with pytest.raises(DataReadinessError):
        _run(inventory, _forbidden)
    assert not inventory["output"].exists()


@pytest.mark.parametrize("kind", ["malformed", "naive", "reversed", "future", "response_outside_attempt"])
def test_rehashed_receipt_timestamp_corruption_rejected_before_fetch(inventory: dict[str, Any], kind: str) -> None:
    report = _run(inventory, _empty)
    attempt = _attempt(inventory)
    receipt = _read(attempt / "receipt.json")
    if kind == "malformed":
        receipt["started_at_utc"] = "not-a-timestamp"
    elif kind == "naive":
        receipt["started_at_utc"] = datetime.fromisoformat(receipt["started_at_utc"]).replace(tzinfo=None).isoformat()
    elif kind == "reversed":
        receipt["started_at_utc"] = (datetime.fromisoformat(receipt["completed_at_utc"]) + timedelta(seconds=1)).isoformat()
    else:
        metadata = _read(attempt / "000.json")
        metadata["retrieved_at_utc"] = (datetime.fromisoformat(metadata["retrieved_at_utc"]) + timedelta(days=365)).isoformat()
        _write(attempt / "000.json", metadata)
        receipt["pages"][0]["metadata_sha256"] = file_sha256(attempt / "000.json")
        if kind == "future":
            for key in ("started_at_utc", "completed_at_utc"):
                receipt[key] = (datetime.fromisoformat(receipt[key]) + timedelta(days=365)).isoformat()
    _resign(attempt / "receipt.json", receipt, "receipt_sha256")
    expected = _receipt_rebound_pin(report, attempt)
    with pytest.raises((DataReadinessError, ValueError), match="clock|time|future|UTC|isoformat|terminal"):
        _run(inventory, _forbidden, expected)


@pytest.mark.parametrize("target", ["source", "observations", "manifest", "config"])
def test_input_changed_during_fetch_prevents_report_publication(inventory: dict[str, Any], target: str) -> None:
    def fetch(params: dict[str, Any], bound: int) -> HttpByteResponse:
        path = inventory[target]
        path.write_bytes(path.read_bytes() + b"\n")
        return _empty(params, bound)

    with pytest.raises(DataReadinessError, match="source binding"):
        _run(inventory, fetch)
    assert not list(inventory["output"].glob("reports/*.json"))


def test_workspace_lease_precedes_input_load(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("collector loaded inputs before obtaining the workspace lease")

    monkeypatch.setattr(collector, "_prepare", forbidden)
    with heavy_job_lease("synthetic-other-worker", runtime_dir=inventory["root"] / "runtime"):
        with pytest.raises(HeavyJobBusyError):
            _run(inventory, _forbidden)
    assert not inventory["output"].exists()


def test_unknown_output_ticker_rejected_before_fetch(inventory: dict[str, Any]) -> None:
    report = _run(inventory, _empty)
    (inventory["output"] / "tickers" / "UNKNOWN").mkdir()
    with pytest.raises(DataReadinessError, match="unknown ticker"):
        _run(inventory, _forbidden, report["audit_sha256"])


def test_blank_or_missing_ids_across_pages_remain_uninterpreted_records(inventory: dict[str, Any]) -> None:
    def fetch(params: dict[str, Any], bound: int) -> HttpByteResponse:
        assert bound == 2048
        if params["symbols"] == "BBB":
            return _page(params)
        return _page(params, {"unknown_family": [{"id": ""}, {"id": " "}, {"id": None}, {}]},
            "second" if "page_token" not in params else None)

    report = _run(inventory, fetch)
    assert report["status"] == "collected_unreviewed"
    assert report["action_counts"] == {"unknown_family": 8}
    assert report["ownership_admitted"] is False
    assert _run(inventory, expected=report["audit_sha256"]) == report


@pytest.mark.parametrize("stage", ["input", "collection", "report"])
def test_memory_guard_aborts_globally_without_publishing_report(
    inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    calls = []

    def guard(*args: Any, **kwargs: Any) -> None:
        if stage in kwargs["stage"]:
            raise errors.MemoryBudgetError("synthetic memory budget exceeded")

    def fetch(params: dict[str, Any], bound: int) -> HttpByteResponse:
        calls.append(params["symbols"])
        return _empty(params, bound)

    monkeypatch.setattr(collector, "assert_memory_budget", guard)
    monkeypatch.setattr(collector, "assert_peak_memory_budget", guard)
    with pytest.raises(errors.MemoryBudgetError, match="synthetic memory budget exceeded"):
        _run(inventory, fetch)
    assert calls == {"input": [], "collection": ["AAA"], "report": ["AAA", "BBB"]}[stage]
    assert not list(inventory["output"].glob("reports/*.json"))


@pytest.mark.parametrize("complete", [False, True])
def test_online_resume_requires_external_pin_even_for_failed_attempts(inventory: dict[str, Any], complete: bool) -> None:
    def fetch(params: dict[str, Any], bound: int) -> HttpByteResponse:
        if params["symbols"] == "AAA" and not complete:
            raise requests.ConnectionError("synthetic retryable failure")
        return _empty(params, bound)

    report = _run(inventory, fetch)
    assert report["acquired_tickers"] == (2 if complete else 1)
    before = {path: file_sha256(path) for path in inventory["output"].rglob("*") if path.is_file()}
    with pytest.raises(DataReadinessError, match="independent|pin"):
        _run(inventory, _forbidden)
    with pytest.raises(DataReadinessError, match="pin"):
        _run(inventory, _forbidden, "0" * 64)
    assert {path: file_sha256(path) for path in before} == before


def test_resume_pin_includes_latest_failed_attempt_not_only_successes(inventory: dict[str, Any]) -> None:
    def fetch(params: dict[str, Any], bound: int) -> HttpByteResponse:
        if params["symbols"] == "AAA":
            raise requests.ConnectionError("synthetic retryable failure")
        return _empty(params, bound)

    first = _run(inventory, fetch)
    second = _run(inventory, fetch, first["audit_sha256"])
    assert len(_ticker(second, "AAA")["attempts"]) == 2
    assert second["audit_sha256"] != first["audit_sha256"]
    assert first["action_counts"] == second["action_counts"] == {}
    with pytest.raises(DataReadinessError, match="pin"):
        _run(inventory, _forbidden, first["audit_sha256"])
    assert _run(inventory, expected=second["audit_sha256"]) == second


@pytest.mark.parametrize("kind", ["sha256", "body_length", "naive_clock"])
def test_bad_incoming_metadata_is_archived_failed_evidence_and_other_ticker_continues(
    inventory: dict[str, Any], kind: str,
) -> None:
    calls = []
    received = []

    def fetch(params: dict[str, Any], bound: int) -> HttpByteResponse:
        assert bound == 2048
        calls.append(params["symbols"])
        response = _page(params)
        if params["symbols"] == "AAA":
            if kind == "sha256":
                response = replace(response, sha256="0" * 64)
            elif kind == "body_length":
                response = replace(response, body_length=len(response.body) + 1)
            else:
                response = replace(response, retrieved_at_utc=response.retrieved_at_utc.replace(tzinfo=None))
            received.append(response)
        return response

    report = _run(inventory, fetch)
    assert calls == ["AAA", "BBB"]
    assert report["status"] == "incomplete"
    assert report["acquired_tickers"] == 1
    assert _ticker(report, "AAA")["attempts"][0]["state"] == "failed"
    assert _ticker(report, "BBB")["acquired"] is True
    attempt = _attempt(inventory)
    receipt = _read(attempt / "receipt.json")
    page = receipt["pages"][0]
    body = (attempt / page["body_path"]).read_bytes()
    metadata = _read(attempt / page["metadata_path"])
    assert body == received[0].body
    assert page["body_sha256"] == sha256(body).hexdigest()
    assert page["body_length"] == len(body)
    assert metadata["sha256"] == received[0].sha256
    assert metadata["body_length"] == received[0].body_length
    assert metadata["retrieved_at_utc"] == received[0].retrieved_at_utc.isoformat()
    assert _run(inventory, expected=report["audit_sha256"]) == report
    resumed = _run(inventory, _empty, report["audit_sha256"])
    assert resumed["acquired_tickers"] == 2
    assert len(_ticker(resumed, "AAA")["attempts"]) == 2


@pytest.mark.parametrize("target", ["body", "metadata"])
def test_later_tamper_of_failed_response_archive_is_global_not_retryable(inventory: dict[str, Any], target: str) -> None:
    def fetch(params: dict[str, Any], bound: int) -> HttpByteResponse:
        response = _empty(params, bound)
        return replace(response, sha256="0" * 64) if params["symbols"] == "AAA" else response

    report = _run(inventory, fetch)
    assert report["status"] == "incomplete"
    attempt = _attempt(inventory)
    path = attempt / ("000.bin" if target == "body" else "000.json")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(DataReadinessError, match="hash|archived body"):
        _run(inventory, _forbidden, report["audit_sha256"])


def test_typed_memory_budget_exception_from_fetch_stops_all_tickers(inventory: dict[str, Any]) -> None:
    calls = []
    budget_error = errors.MemoryBudgetError("synthetic transport memory budget exceeded")
    assert isinstance(budget_error, DataReadinessError)

    def fetch(params: dict[str, Any], bound: int) -> HttpByteResponse:
        assert bound == 2048
        calls.append(params["symbols"])
        raise budget_error

    with pytest.raises(errors.MemoryBudgetError) as caught:
        _run(inventory, fetch)
    assert caught.value is budget_error
    assert calls == ["AAA"]
    assert not list(inventory["output"].glob("reports/*.json"))
    assert not list(inventory["output"].glob("tickers/*/*/receipt.json"))
