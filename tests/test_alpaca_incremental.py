from __future__ import annotations

import base64
import gzip
import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock
from urllib.parse import urlencode

import pytest
import requests

from market_predictor.config import Settings
from market_predictor.core.errors import MemoryBudgetError
from market_predictor.heavy_jobs import heavy_job_lease
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.http import HttpByteResponse
from market_predictor.swing.datasets.alpaca_incremental import collect, collector
from market_predictor.swing.datasets.alpaca_incremental.__main__ import main
from market_predictor.swing.datasets.alpaca_incremental.config import load_config
from market_predictor.swing.datasets.alpaca_incremental.pages import Unit, page_record, validate_record
from market_predictor.swing.datasets.alpaca_incremental.storage import IntegrityError, publish, verified

DAY = date(2026, 7, 9)
NOW = datetime(2026, 7, 11, 12, tzinfo=UTC)
MEMORY_GUARD = collector.guard_memory


class Clock(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> Clock:
        return cls(NOW.year, NOW.month, NOW.day, NOW.hour, tzinfo=UTC)


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setattr(collector, "datetime", Clock)
    monkeypatch.setattr(collector, "guard_memory", Mock())
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(requests.Session, "request", Mock(side_effect=AssertionError("real network forbidden")))
    yield
    assert not (tmp_path / "runtime/heavy-job.owner.json").exists()


def config_file(tmp_path: Path, *, symbols: tuple[str, ...] = ("AAPL", "EMPTY"), **options: Any) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": [{"ticker": s} for s in symbols]}), encoding="utf-8")
    values: dict[str, Any] = {"start": DAY.isoformat(), "output": "archive", "revision_overlap_days": 0, **options}
    lines = [f"{key} = {json.dumps(value)}" for key, value in values.items()]
    lines.extend(["[[source_manifests]]", 'path = "manifest.json"', f'sha256 = "{sha256(manifest.read_bytes()).hexdigest()}"'])
    path = tmp_path / "collection.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class Transport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.bad_symbol: str | None = None
        self.fail_day: str | None = None
        self.pages = 1
        self.repeat_token = False
        self.mutate: Any = None
        self.bodies: list[bytes] = []

    def __call__(self, url: str, *, params: dict[str, Any], **kwargs: Any) -> HttpByteResponse:
        self.calls.append(dict(params))
        if self.bad_symbol in params["symbols"].split(",") or params["start"].startswith(self.fail_day or "never"):
            response = requests.Response()
            response.status_code = 400 if self.bad_symbol else 503
            raise requests.HTTPError("sensitive-failure-detail", response=response)
        index = int(params.get("page_token", "0"))
        next_token = "1" if self.repeat_token else str(index + 1) if index + 1 < self.pages else None
        symbols = [s for s in params["symbols"].split(",") if s != "EMPTY"]
        if "news" in url:
            payload: dict[str, Any] = {"news": [{"id": index + 1, "created_at": params["start"], "updated_at": params["start"],
                "symbols": symbols, "headline": "Fixture", "content": "raw untouched"}] if symbols else [], "next_page_token": next_token}
        else:
            payload = {"bars": {s: [{"t": params["start"], "o": 1, "h": 2, "l": 1, "c": 2, "v": 10}] for s in symbols},
                "next_page_token": next_token}
        if self.mutate is not None:
            self.mutate(payload)
        body = json.dumps(payload, indent=1).encode("utf-8") + b" \n"
        self.bodies.append(body)
        requested = f"{url}?{urlencode(params)}"
        return HttpByteResponse(body=body, requested_url=requested, final_url=requested, redirect_chain=(),
            status_code=200, retrieved_at_utc=NOW + timedelta(hours=1), content_type="application/json", content_encoding=None,
            etag="fixture-etag", last_modified=None, body_length=len(body), sha256=sha256(body).hexdigest(),
            body_representation="http_entity_encoded", safe_headers=(("etag", "fixture-etag"), ("content-type", "application/json")))


def source_with(transport: Transport) -> AlpacaSource:
    source = AlpacaSource(Settings(ALPACA_API_KEY_ID="fixture-key", ALPACA_API_SECRET_KEY="fixture-secret"))
    client = Mock()
    client.before_request = None
    client.get_bytes_with_metadata.side_effect = transport
    source.client = client
    return source


def test_resume_offline_exact_bytes_symbol_counts_and_portable_root(tmp_path: Path) -> None:
    path = config_file(tmp_path, symbols=("BRK-B", "EMPTY"))
    transport = Transport()
    source = source_with(transport)
    first = collect(Path(path.name), root=tmp_path, through=DAY, source=source)
    assert first["status"] == "complete"
    assert first["complete_through_utc_date"] == DAY.isoformat()
    assert len(transport.calls) == 3
    for family in ("bars_raw", "bars_all", "news"):
        assert first["families"][family]["symbols"]["EMPTY"]["no_data_days"] == 1
        assert first["families"][family]["symbols"]["BRK.B"]["rows"] == 1
    archived = list((tmp_path / "archive").rglob("*.jsonl.gz"))
    before = {p: p.read_bytes() for p in archived}
    for p in archived:
        record = json.loads(gzip.decompress(p.read_bytes()))
        assert base64.b64decode(record["body_base64"]) in transport.bodies
        assert record["retrieved_at_utc"] == (NOW + timedelta(hours=1)).isoformat()
        assert record["response_headers"]["etag"] == "fixture-etag"
    online = collect(path, through=DAY, source=source)
    assert online["reused_units"] == 3 and len(transport.calls) == 3
    snapshot = {p: p.read_bytes() for p in (tmp_path / "archive").rglob("*") if p.is_file()}
    offline = collect(path, through=DAY, offline=True, source=source)
    assert offline["status"] == "verified" and len(transport.calls) == 3
    assert snapshot == {p: p.read_bytes() for p in snapshot}
    assert before == {p: p.read_bytes() for p in archived}


def test_pagination_full_and_repeated_token_retry(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    transport = Transport()
    transport.repeat_token = True
    source = source_with(transport)
    result = collect(path, through=DAY, source=source)
    assert result["failed_units"] == 3 and result["complete_through_utc_date"] is None
    assert len(transport.calls) == 6
    assert not list((tmp_path / "archive").rglob("success.json"))
    transport.repeat_token, transport.pages = False, 3
    result = collect(path, through=DAY, source=source)
    assert result["status"] == "complete"
    assert result["families"]["news"]["rows"] == 3
    assert collect(path, through=DAY, offline=True)["status"] == "verified"


def test_batch_rejection_isolates_and_does_not_refetch_successful_siblings(tmp_path: Path) -> None:
    path = config_file(tmp_path, symbols=("GOOD", "RHT"))
    transport = Transport()
    transport.bad_symbol = "RHT"
    source = source_with(transport)
    result = collect(path, through=DAY, source=source)
    assert result["split_attempts"] == 3 and result["failed_units"] == 3
    assert result["families"]["news"]["symbols"]["RHT"]["failed_days"] == 1
    assert result["families"]["news"]["symbols"]["GOOD"]["rows"] == 1
    assert result["complete_through_utc_date"] is None
    transport.calls.clear()
    transport.bad_symbol = None
    result = collect(path, through=DAY, source=source)
    assert result["status"] == "complete"
    assert [c["symbols"] for c in transport.calls] == ["RHT"] * 3
    assert all("sensitive-failure-detail" not in p.read_text() for p in (tmp_path / "archive").rglob("failure.json"))


def test_watermark_never_skips_failed_day_and_extension_reuses_closed_prefix(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    transport = Transport()
    transport.fail_day = DAY.isoformat()
    source = source_with(transport)
    result = collect(path, through=DAY + timedelta(days=1), source=source)
    assert result["complete_through_utc_date"] is None
    assert result["fetched_units"] == 3
    transport.calls.clear()
    transport.fail_day = None
    result = collect(path, through=DAY + timedelta(days=1), source=source)
    assert result["complete_through_utc_date"] == "2026-07-10"
    assert len(transport.calls) == 3 and all(c["start"].startswith("2026-07-09") for c in transport.calls)
    assert all(c["end"].endswith("T23:59:59.999999+00:00") for c in transport.calls)
    assert all(c.get("asof", "2026-07-09") == "2026-07-09" for c in transport.calls)


def test_tamper_is_integrity_failure_not_refetched(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    transport = Transport()
    source = source_with(transport)
    collect(path, through=DAY, source=source)
    archive = next((tmp_path / "archive").rglob("*.jsonl.gz"))
    archive.write_bytes(archive.read_bytes() + b"tamper")
    transport.calls.clear()
    result = collect(path, through=DAY, source=source)
    assert result["status"] == "integrity_failed" and not transport.calls
    assert result["complete_through_utc_date"] is None


def test_replay_rejects_forged_counters_and_token_chain_even_with_new_file_hash(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    collect(path, through=DAY, source=source_with(Transport()))
    receipt_path = next((tmp_path / "archive").rglob("success.json"))
    receipt = verified(receipt_path)
    receipt["counts"]["AAPL"] = 999
    publish(receipt_path, receipt, replace=True)
    assert collect(path, through=DAY, offline=True)["status"] == "integrity_failed"
    receipt["counts"]["AAPL"] = 1
    archive = receipt_path.parent / receipt["archive"]
    record = json.loads(gzip.decompress(archive.read_bytes()))
    record["request_page_token"] = "unexpected"
    archive.write_bytes(gzip.compress(json.dumps(record).encode() + b"\n"))
    receipt["archive_sha256"] = sha256(archive.read_bytes()).hexdigest()
    publish(receipt_path, receipt, replace=True)
    assert collect(path, through=DAY, offline=True)["status"] == "integrity_failed"


def test_crash_before_success_receipt_leaves_retriable_orphan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = config_file(tmp_path)
    real_publish = publish

    def interrupted(target: Path, payload: dict[str, Any], *, replace: bool = False) -> None:
        if target.name == "success.json":
            raise KeyboardInterrupt
        real_publish(target, payload, replace=replace)

    monkeypatch.setattr(collector, "publish", interrupted)
    with pytest.raises(KeyboardInterrupt):
        collect(path, through=DAY, source=source_with(Transport()))
    orphan = next((tmp_path / "archive").rglob("*.jsonl.gz"))
    before = orphan.read_bytes()
    monkeypatch.setattr(collector, "publish", real_publish)
    assert collect(path, through=DAY, source=source_with(Transport()))["status"] == "complete"
    assert orphan.read_bytes() == before
    assert len(list((tmp_path / "archive").rglob("*.jsonl.gz"))) == 4


@pytest.mark.parametrize("option,value", [("batch_size", 51), ("batch_size", 0), ("batch_size", True),
    ("bars_limit", 10001), ("news_limit", 51), ("max_pages_per_unit", 0), ("max_pages_per_unit", 10001),
    ("revision_overlap_days", 8), ("unknown_option", 1)])
def test_bounded_config(tmp_path: Path, option: str, value: Any) -> None:
    with pytest.raises(ValueError):
        load_config(config_file(tmp_path, **{option: value}))


@pytest.mark.parametrize("value", [0, -1, True, 1000001])
def test_bounded_max_units(tmp_path: Path, value: int) -> None:
    with pytest.raises(ValueError):
        collect(config_file(tmp_path), through=DAY, max_units=value)


def test_unit_bound_and_offline_pending_are_not_no_data(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    transport = Transport()
    result = collect(path, through=DAY, max_units=1, source=source_with(transport))
    assert result["status"] == "bounded_incomplete" and len(transport.calls) == 1
    assert result["pending_units"] == 2 and result["families"]["news"]["no_data_units"] == 0
    assert collect(path, through=DAY, offline=True)["pending_units"] == 2


def test_config_provenance_tamper_and_scope_mutation_fail_closed(tmp_path: Path) -> None:
    path = config_file(tmp_path, start="2026-07-10")
    assert load_config(path)[0].start == date(2026, 7, 10)
    collect(path, through=date(2026, 7, 10), source=source_with(Transport()))
    path.write_text(path.read_text().replace('output = "archive"', 'output = "archive"\nbatch_size = 25'))
    with pytest.raises(IntegrityError):
        collect(path, through=date(2026, 7, 10), offline=True)
    (tmp_path / "manifest.json").write_text('{"artifacts": []}')
    with pytest.raises(ValueError, match="provenance"):
        load_config(path)


def test_per_page_memory_guard_pauses_without_losing_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = config_file(tmp_path)
    transport = Transport()
    source = source_with(transport)
    collect(path, through=DAY, max_units=1, source=source)
    before = list((tmp_path / "archive").rglob("success.json"))
    guards = Mock(side_effect=[None, None, None, MemoryBudgetError("pressure")])
    monkeypatch.setattr(collector, "guard_memory", guards)
    result = collect(path, through=DAY, source=source)
    assert result["status"] == "paused_memory"
    assert list((tmp_path / "archive").rglob("success.json")) == before
    assert source.client.before_request is None


def test_cli_busy_memory_and_invalid_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    path = config_file(tmp_path)
    with heavy_job_lease("fixture", runtime_dir=tmp_path / "runtime"):
        assert main(["--config", str(path), "--offline"]) == 75
    assert json.loads(capsys.readouterr().out)["status"] == "busy"
    monkeypatch.setattr(collector, "guard_memory", Mock(side_effect=MemoryBudgetError("pressure")))
    assert main(["--root", str(tmp_path), "--config", path.name, "--offline"]) == 75
    assert json.loads(capsys.readouterr().out)["status"] == "paused_memory"
    monkeypatch.setattr(collector, "guard_memory", Mock())
    assert main(["--config", str(path), "--max-units", "0", "--offline"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_config_or_io"


def test_partial_today_snapshot_and_revision_capture_offline_stable(tmp_path: Path) -> None:
    path = config_file(tmp_path, revision_overlap_days=3)
    transport = Transport()
    result = collect(path, through=NOW.date(), source=source_with(transport))
    assert result["status"] == "complete" and not result["today_complete"]
    assert result["complete_through_utc_date"] == "2026-07-10"
    assert result["partial_capture_count"] == 1
    assert result["revisions"]["news"]["observed_units"] == 2
    assert transport.calls[-1]["end"] == NOW.isoformat()
    snapshot = {p: p.read_bytes() for p in (tmp_path / "archive").rglob("*") if p.is_file()}
    result = collect(path, through=NOW.date(), offline=True)
    assert result["status"] == "verified" and result["partial_capture_count"] == 1
    assert snapshot == {p: p.read_bytes() for p in (tmp_path / "archive").rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        collect(path, through=NOW.date() + timedelta(days=1))


def test_offline_today_without_saved_capture_is_incomplete(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    collect(path, source=source_with(Transport()))
    result = collect(path, through=NOW.date(), offline=True)
    assert result["status"] == "bounded_incomplete" and result["partial_capture_count"] == 0


@pytest.mark.parametrize("mutation", ["updated_after", "created_after", "missing_created", "missing_id",
    "naive", "updated_before", "error"])
def test_news_wrong_window_or_identity_is_failure(tmp_path: Path, mutation: str) -> None:
    path = config_file(tmp_path)
    transport = Transport()

    def mutate(payload: dict[str, Any]) -> None:
        if "news" not in payload:
            return
        item = payload["news"][0]
        if mutation == "updated_after":
            item["updated_at"] = "2026-07-10T00:00:00Z"
        elif mutation == "created_after":
            item["created_at"] = "2026-07-10T00:00:00Z"
        elif mutation == "missing_id":
            del item["id"]
        elif mutation == "missing_created":
            del item["created_at"]
        elif mutation == "naive":
            item["created_at"] = "2026-07-09T00:00:00"
        elif mutation == "updated_before":
            item["updated_at"] = "2026-07-08T00:00:00Z"
        else:
            payload["error"] = "not successful"

    transport.mutate = mutate
    result = collect(path, through=DAY, source=source_with(transport))
    assert result["families"]["news"]["failed_units"] == 1
    assert result["families"]["news"]["no_data_units"] == 0
    assert result["complete_through_utc_date"] is None
    rejected = list((tmp_path / "archive").rglob("rejected-*.body.gz"))
    assert len(rejected) == 1 and gzip.decompress(rejected[0].read_bytes()) in transport.bodies


def test_older_publication_revised_inside_query_preserves_both_clocks(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    transport = Transport()

    def revision(payload: dict[str, Any]) -> None:
        if "news" in payload:
            payload["news"][0]["created_at"] = "2026-07-08T08:12:39Z"
            payload["news"][0]["updated_at"] = "2026-07-09T04:43:09Z"

    transport.mutate = revision
    result = collect(path, through=DAY, source=source_with(transport))
    assert result["families"]["news"]["failed_units"] == 0
    assert result["complete_through_utc_date"] == DAY.isoformat()
    assert result["news_query_timestamp"] == "updated_at"
    archives = list((tmp_path / "archive" / "units").glob("*-news-*/archive-*.jsonl.gz"))
    with gzip.open(archives[0], "rt") as stream:
        record = json.loads(stream.readline())
    body = json.loads(base64.b64decode(record["body_base64"]))
    assert body["news"][0]["created_at"] == "2026-07-08T08:12:39Z"
    assert body["news"][0]["updated_at"] == "2026-07-09T04:43:09Z"
    assert collect(path, through=DAY, offline=True)["status"] == "verified"


@pytest.mark.parametrize("mutation", ["query", "endpoint", "redirect", "status", "retrieved", "future_retrieved",
    "payload", "secret_header"])
def test_news_receipt_metadata_exactness(tmp_path: Path, mutation: str) -> None:
    path = config_file(tmp_path)
    config, _, _ = load_config(path)
    unit = Unit(DAY, "news", ("AAPL",))
    page = unit.fetch(source_with(Transport()), config, None)
    if mutation == "payload":
        with pytest.raises(ValueError):
            page_record(replace(page, raw_payload={}), unit, config, None)
        return
    record = page_record(page, unit, config, None)
    if mutation == "query":
        record["requested_url"] += "&unexpected=value"
    elif mutation == "endpoint":
        record["requested_url"] = record["requested_url"].replace("/v1beta1/news", "/v2/stocks/bars")
    elif mutation == "redirect":
        record["redirect_chain"] = [record["requested_url"]]
    elif mutation == "status":
        record["status_code"] = 201
    elif mutation == "retrieved":
        record["retrieved_at_utc"] = unit.start.isoformat()
    elif mutation == "future_retrieved":
        record["retrieved_at_utc"] = "2100-01-01T00:00:00+00:00"
    else:
        page = replace(page, response_headers={**page.response_headers, "Authorization": "fixture-sensitive"})
        assert "Authorization" not in page_record(page, unit, config, None)["response_headers"]
        return
    with pytest.raises(ValueError):
        validate_record(record, unit, config, None)


def test_page_limit_never_publishes_truncated_success(tmp_path: Path) -> None:
    path = config_file(tmp_path, max_pages_per_unit=1)
    transport = Transport()
    transport.pages = 2
    result = collect(path, through=DAY, source=source_with(transport))
    assert result["failed_units"] == 3 and result["fetched_units"] == 0
    assert not list((tmp_path / "archive").rglob("success.json"))
    assert len(transport.calls) == 3


def test_owned_session_closes_on_publication_error_injected_session_stays_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = config_file(tmp_path)
    source = source_with(Transport())
    monkeypatch.setattr(collector, "AlpacaSource", Mock(return_value=source))
    original = publish

    def failing(target: Path, payload: dict[str, Any], *, replace: bool = False) -> None:
        if target.name == "status.json":
            raise OSError("fixture publication interruption")
        original(target, payload, replace=replace)

    monkeypatch.setattr(collector, "publish", failing)
    with pytest.raises(OSError):
        collect(path, through=DAY)
    close = cast(Mock, source.client.session.close)
    close.assert_called_once()
    close.reset_mock()
    monkeypatch.setattr(collector, "publish", original)
    result = collect(path, through=DAY, source=source)
    assert result["reused_units"] == 3
    close.assert_not_called()


def test_cli_compact_counts_and_status_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    path = config_file(tmp_path)
    monkeypatch.setattr(collector, "AlpacaSource", Mock(return_value=source_with(Transport())))
    assert main(["--config", str(path), "--through", DAY.isoformat()]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "symbols" not in result["families"]["news"]
    status = verified(Path(result["status_path"]))
    assert status["families"]["news"]["symbols"]["EMPTY"]["no_data_days"] == 1
    assert verified(tmp_path / "archive/progress.json")["status"] == "complete"


def test_split_parent_consumes_max_unit_bound_and_resumes_children(tmp_path: Path) -> None:
    path = config_file(tmp_path, symbols=("GOOD", "RHT"))
    transport = Transport()
    transport.bad_symbol = "RHT"
    source = source_with(transport)
    result = collect(path, through=DAY, max_units=1, source=source)
    assert result["attempted_units"] == 1 and len(transport.calls) == 1
    assert len(list((tmp_path / "archive").rglob("split.json"))) == 1
    transport.calls.clear()
    transport.bad_symbol = None
    result = collect(path, through=DAY, max_units=1, source=source)
    assert result["attempted_units"] == 1 and transport.calls[0]["symbols"] == "GOOD"


def test_guard_thresholds_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    system = Mock()
    process = Mock()
    monkeypatch.setattr(collector, "assert_system_memory_available", system)
    monkeypatch.setattr(collector, "assert_memory_budget", process)
    MEMORY_GUARD()
    system.assert_called_once_with(minimum_available_gib=2.0, maximum_used_percent=85.0)
    process.assert_called_once_with(hard_budget_gib=5.0, headroom_gib=0.25, stage="alpaca incremental page")
    system.side_effect = MemoryBudgetError("unknown system measurement")
    with pytest.raises(MemoryBudgetError):
        MEMORY_GUARD()
