"""Synthetic durability tests, never market evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from market_predictor.evidence import news_collection as collection
from market_predictor.evidence.news_collection import (
    NewsCollectionPlan,
    NewsCollectionTransportError,
    NewsCollectionWindow,
    collect_news_receipts,
    news_collection_plan_bytes,
    news_collection_plan_sha256,
    read_news_collection_plan,
)
from market_predictor.evidence.news_exchange import NewsPageRequest, ValidatedNewsReceipt, read_news_receipt, validate_news_receipt
from market_predictor.locking import LockTimeout
from market_predictor.sources import news_collection as adapter

_CASE = json.loads((Path(__file__).parent / "fixtures/news_receipt_exchange.json").read_text(encoding="utf-8"))["cases"][0]
_SAVED = validate_news_receipt(_CASE["manifest_utf8"].encode(), _CASE["payload_utf8"].encode())


def _plan(*symbols: str, pages: int = 3, attempts: int = 3) -> NewsCollectionPlan:
    return NewsCollectionPlan(
        producer_revision="a" * 40, max_pages_per_window=pages, max_attempts_per_page=attempts,
        windows=tuple(NewsCollectionWindow(window_id=symbol, request=_SAVED.manifest.request.model_copy(update={"symbol": symbol}))
                      for symbol in (symbols or ("MSFT",))),
    )


def _receipt(request: NewsPageRequest, token: str | None = None) -> ValidatedNewsReceipt:
    manifest = json.loads(_SAVED.manifest_bytes)
    manifest["request"].update(symbol=request.symbol, page_token=request.page_token)
    body = json.dumps({"news": [], "next_page_token": token}).encode()
    manifest.update(payload_sha256=hashlib.sha256(body).hexdigest(), payload_bytes=len(body))
    return validate_news_receipt(json.dumps(manifest).encode(), body)


def _run(root: Path, plan: NewsCollectionPlan, fetch: object = None) -> collection.NewsCollectionReport:
    return collect_news_receipts(root=root, plan=plan, expected_plan_sha256=news_collection_plan_sha256(plan), fetch_page=fetch)


def _run_root(root: Path, plan: NewsCollectionPlan) -> Path:
    return root / "runs" / news_collection_plan_sha256(plan)


def test_exact_receipt_survives_offline_restart_and_portable_import(tmp_path: Path) -> None:
    plan = _plan()
    fetch = Mock(return_value=_SAVED)
    report = _run(tmp_path, plan, fetch)
    assert report.complete and fetch.call_count == 1
    assert _run(tmp_path, plan) == report
    assert _run(tmp_path, plan, fetch) == report
    assert fetch.call_count == 1
    ref = report.windows[0].receipts[0]
    imported = read_news_receipt(ref.manifest_path, ref.payload_path, expected_receipt_sha256=ref.receipt_sha256)
    assert imported == _SAVED
    assert not imported.observable_at(imported.manifest.request.end_utc)


def test_multiple_plans_share_one_local_owner_root(tmp_path: Path) -> None:
    first, second = _plan("MSFT"), _plan("NVDA")
    assert _run(tmp_path, first, _receipt).complete
    assert _run(tmp_path, second, _receipt).complete
    assert _run(tmp_path, first).complete
    assert _run(tmp_path, second).complete
    assert _run_root(tmp_path, first) != _run_root(tmp_path, second)


def test_offline_never_attempts_uncollected_work(tmp_path: Path) -> None:
    root = tmp_path / "absent"
    report = _run(root, _plan())
    assert not report.complete
    assert report.windows[0].attempts == 0
    assert not list(root.rglob("intent.json"))


def test_failure_isolated_then_resumed_without_refetching_success(tmp_path: Path) -> None:
    plan = _plan("MSFT", "NVDA")
    def fetch(request: NewsPageRequest) -> ValidatedNewsReceipt:
        if request.symbol == "MSFT":
            raise NewsCollectionTransportError("429 rate limited")
        return _receipt(request)
    first = _run(tmp_path, plan, fetch)
    assert [w.status for w in first.windows] == ["transport_failure", "complete"]
    retry = Mock(side_effect=_receipt)
    final = _run(tmp_path, plan, retry)
    assert final.complete and retry.call_count == 1
    assert retry.call_args.args[0].symbol == "MSFT"


@pytest.mark.parametrize("pages,token,expected,count", [(3, "again", "pagination_cycle", 2), (1, "next", "page_budget", 1)])
def test_pagination_is_bounded_and_not_false_complete(tmp_path: Path, pages: int, token: str, expected: str, count: int) -> None:
    plan = _plan(pages=pages)
    fetch = Mock(side_effect=lambda request: _receipt(request, token))
    report = _run(tmp_path, plan, fetch)
    assert report.windows[0].status == expected and not report.complete
    assert fetch.call_count == count
    assert _run(tmp_path, plan, fetch) == report and fetch.call_count == count


def test_retry_budget_includes_failed_logical_attempts(tmp_path: Path) -> None:
    plan = _plan(attempts=2)
    fetch = Mock(side_effect=NewsCollectionTransportError("unavailable"))
    _run(tmp_path, plan, fetch)
    _run(tmp_path, plan, fetch)
    report = _run(tmp_path, plan, fetch)
    assert report.windows[0].status == "attempt_budget" and fetch.call_count == 2


@pytest.mark.parametrize("stage", ["intent", "receipt", "result"])
def test_crash_during_publication_preserves_history_and_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    plan = _plan()
    publish = collection._publish_directory
    fired = False
    def interrupt(target: Path, files: dict[str, bytes]) -> None:
        nonlocal fired
        matching = (stage == "intent" and "intent.json" in files) or target.name == stage
        if matching and not fired:
            fired = True
            if stage != "intent":
                publish(target, files)
            raise KeyboardInterrupt("simulated process interruption")
        publish(target, files)
    monkeypatch.setattr(collection, "_publish_directory", interrupt)
    fetch = Mock(side_effect=_receipt)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, plan, fetch)
    monkeypatch.setattr(collection, "_publish_directory", publish)
    final = _run(tmp_path, plan, fetch)
    assert final.complete
    if stage == "result":
        assert fetch.call_count == 1
    elif stage == "receipt":
        assert final.windows[0].ambiguous_attempts == 1
        assert len(list(_run_root(tmp_path, plan).rglob("manifest.json"))) == 2
    else:
        assert final.windows[0].attempts == 1


def test_crash_after_fetch_before_receipt_keeps_indeterminate_attempt(tmp_path: Path) -> None:
    plan = _plan()
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, plan, Mock(side_effect=KeyboardInterrupt()))
    final = _run(tmp_path, plan, _receipt)
    assert final.complete
    assert final.windows[0].attempts == 2 and final.windows[0].ambiguous_attempts == 1


def test_partial_payload_write_never_becomes_committed_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan()
    write = collection._write_new
    def interrupt(path: Path, raw: bytes) -> None:
        if path.name == "payload.json":
            write(path, raw[:3])
            raise OSError("simulated interrupted write")
        write(path, raw)
    monkeypatch.setattr(collection, "_write_new", interrupt)
    with pytest.raises(OSError):
        _run(tmp_path, plan, _receipt)
    assert not list(_run_root(tmp_path, plan).rglob("result.json"))
    monkeypatch.setattr(collection, "_write_new", write)
    report = _run(tmp_path, plan, _receipt)
    assert report.complete and report.windows[0].ambiguous_attempts == 1
    assert report.windows[0].attempts == 2


def test_missing_committed_payload_and_changed_owner_are_fatal(tmp_path: Path) -> None:
    plan = _plan()
    report = _run(tmp_path, plan, _receipt)
    payload = report.windows[0].receipts[0].payload_path
    original = payload.read_bytes()
    payload.unlink()
    with pytest.raises((ValueError, OSError)):
        _run(tmp_path, plan)
    payload.write_bytes(original)
    owner = tmp_path / "owner" / "owner.json"
    owner.write_bytes(owner.read_bytes().replace(b"market_predictor", b"trading_flow"))
    fetch = Mock(side_effect=_receipt)
    with pytest.raises(ValueError):
        _run(tmp_path, plan, fetch)
    fetch.assert_not_called()


def test_plan_rejects_duplicate_windows_and_unbounded_work() -> None:
    window = _plan().windows[0]
    with pytest.raises(ValueError, match="unique"):
        NewsCollectionPlan(producer_revision="a" * 40, max_pages_per_window=3, windows=(window, window))
    with pytest.raises(ValueError, match="ledger budget"):
        _plan("MSFT", "NVDA", pages=10_000, attempts=10)


@pytest.mark.skipif(os.name != "nt", reason="Windows mapped-drive classification")
def test_mapped_network_drive_is_rejected_before_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = Mock()
    api.GetDriveTypeW.return_value = 4
    monkeypatch.setattr(collection.ctypes, "WinDLL", Mock(return_value=api))
    fetch = Mock()
    with pytest.raises(ValueError, match="fixed local drive"):
        _run(tmp_path, _plan(), fetch)
    fetch.assert_not_called()


def test_transport_adapter_does_not_downgrade_invalid_receipt_to_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = adapter.alpaca_news_receipt_fetcher(Mock(), producer_revision="a" * 40)
    failure = RuntimeError("request failed")
    failure.__cause__ = requests.ConnectionError("connection failed")
    monkeypatch.setattr(adapter, "fetch_news_receipt", Mock(side_effect=failure))
    with pytest.raises(NewsCollectionTransportError):
        fetch(_SAVED.manifest.request)
    monkeypatch.setattr(adapter, "fetch_news_receipt", Mock(side_effect=ValueError("integrity mismatch")))
    with pytest.raises(ValueError, match="integrity mismatch"):
        fetch(_SAVED.manifest.request)
    monkeypatch.setattr(adapter, "fetch_news_receipt", Mock(side_effect=RuntimeError("invalid news rows")))
    with pytest.raises(RuntimeError, match="invalid news rows") as rejected:
        fetch(_SAVED.manifest.request)
    assert not isinstance(rejected.value, NewsCollectionTransportError)


@pytest.mark.parametrize("name", ["payload.json", "manifest.json", "intent.json", "result.json", "plan.json"])
def test_tampered_committed_evidence_aborts_before_any_request(tmp_path: Path, name: str) -> None:
    plan = _plan("MSFT", "NVDA")
    _run(tmp_path, plan, _receipt)
    path = next(_run_root(tmp_path, plan).rglob(name))
    path.write_bytes(path.read_bytes() + b" ")
    fetch = Mock(side_effect=_receipt)
    with pytest.raises(ValueError):
        _run(tmp_path, plan, fetch)
    fetch.assert_not_called()


def test_receipt_cannot_be_assigned_to_different_query_or_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="request, producer or revision"):
        _run(tmp_path, _plan("NVDA"), Mock(return_value=_SAVED))


def test_pinned_plan_round_trip_and_substitution_rejection(tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "plan.json"
    path.write_bytes(news_collection_plan_bytes(plan))
    pin = news_collection_plan_sha256(plan)
    assert read_news_collection_plan(path, expected_plan_sha256=pin) == plan
    with pytest.raises(ValueError):
        read_news_collection_plan(path, expected_plan_sha256="0" * 64)
    fetch = Mock()
    with pytest.raises(ValueError):
        collect_news_receipts(root=tmp_path / "root", plan=plan, expected_plan_sha256="0" * 64, fetch_page=fetch)
    fetch.assert_not_called()


def test_second_process_cannot_fetch_and_owner_death_releases_lock(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    root = tmp_path / "receipts"
    script = (
        "import time\nfrom pathlib import Path\nfrom market_predictor.locking import file_lock\n"
        f"with file_lock(Path({str(root / 'collection-owner')!r})):\n"
        f" Path({str(ready)!r}).write_text('ready')\n time.sleep(60)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", script])
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), "child failed to acquire owner lock"
        fetch = Mock(side_effect=_receipt)
        with pytest.raises(LockTimeout):
            _run(root, _plan(), fetch)
        fetch.assert_not_called()
        process.kill()
        process.wait(timeout=5)
        assert _run(root, _plan(), fetch).complete
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
