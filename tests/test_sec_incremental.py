from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, date, datetime
from hashlib import sha256
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.sec_filings.collection import (
    SecFilingCollectionConfig,
    collect_historical_sec_filings,
    load_sec_filing_collection,
)
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.sources.http import HttpClient
from market_predictor.sources.sec import SecFilingHistory, SecRawResponse, SecRequestGovernor
from market_predictor.swing.datasets import sec_incremental as inc

NOW = datetime(2026, 9, 11, 12, tzinfo=UTC)
CIKS = ["0000000001", "0000000002"]


class _Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is not None else NOW.replace(tzinfo=None)


class _Source:
    def __init__(self):
        self.calls = []
        self.fail = set()
        self.pressure = False
        self.client = None

    def fetch_cik_filing_history(self, cik, start, end, *, forms=None, ticker_hint="SEC"):
        self.calls.append((cik, start.date(), end.date()))
        if self.pressure:
            def fail_guard():
                raise MemoryBudgetError("fixture pressure")

            self.client.before_request.guard = fail_guard
        if self.client is not None:
            self.client.before_request()
        if cik in self.fail:
            raise RuntimeError("fixture provider failure")
        body = json.dumps({"cik": cik}).encode()
        digest = sha256(body).hexdigest()
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        raw = SecRawResponse(
            response_id=inc._json_sha256({
                "requested_url": url, "final_url": url, "retrieved_at_utc": NOW.isoformat(), "sha256": digest,
            }),
            requested_url=url, final_url=url, status_code=200, retrieved_at_utc=NOW,
            content_type="application/json", content_encoding="identity", etag=None, last_modified=None,
            body=body, body_sha256=digest, body_length=len(body), safe_headers=(),
        )
        return SecFilingHistory(
            ticker=ticker_hint, cik=cik, company_name="Fixture issuer", filings=(),
            submission_files=(f"CIK{cik}.json",), response_sha256=digest,
            raw_responses=(raw,), source_row_count=0,
        )


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = _Source()
    relations = pd.DataFrame([{
        "sec_cik": cik, "security_id": f"fixture:{cik}", "ticker": f"TEST{index}",
        "effective_from_utc": "2019-07-09T00:00:00Z", "effective_to_utc": "2026-07-09T00:00:00Z",
        "available_at_utc": "2019-07-09T00:00:00Z",
    } for index, cik in enumerate(CIKS)])
    relation_path = tmp_path / "relations.parquet"
    relations.to_parquet(relation_path, index=False)
    archive = tmp_path / "archive"
    collect_historical_sec_filings(
        relations, archive, source_factory=lambda: source,
        config=SecFilingCollectionConfig(
            start_date=date(2019, 7, 9), end_date=date(2026, 7, 8), forms=("8-K",), max_workers=1,
        ), clock=lambda: NOW,
    )
    source.calls.clear()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[sec_incremental]\narchive_directory = "archive"\n'
        f'archive_manifest_sha256 = "{file_sha256(archive / "_manifest.json")}"\n'
        'identity_relations = "relations.parquet"\n'
        f'identity_relations_sha256 = "{file_sha256(relation_path)}"\n'
        'output_root = "extension"\nextension_start = "2026-07-09"\n', encoding="utf-8",
    )
    monkeypatch.setattr(inc, "guard_memory", lambda: None)
    monkeypatch.setattr(inc, "datetime", _Clock)
    monkeypatch.setattr(inc, "Settings", lambda: SimpleNamespace(sec_user_agent="Fixture Research owner@research.test"))
    closes = []

    def factory(settings, *, governor, client):
        source.client = client
        monkeypatch.setattr(client.session, "close", lambda: closes.append(True))
        return source

    monkeypatch.setattr(inc, "SecSource", factory)

    def bounded_replay(path):
        assert path != archive, "historical archive must never load its event table"
        collection = load_sec_filing_collection(path)
        assert collection.manifest["issuer_count"] == 1
        return collection

    monkeypatch.setattr(inc, "load_sec_filing_collection", bounded_replay)
    return SimpleNamespace(
        root=tmp_path, path=config_path, source=source, archive=archive, closes=closes,
        config=inc.load_config(config_path, root=tmp_path),
    )


def _run(fixture, **kwargs):
    return inc.run(fixture.path, root=fixture.root, **kwargs)


def test_failure_is_immutable_others_continue_and_retry_has_no_gap(fixture):
    fixture.source.fail.add(CIKS[0])
    first = _run(fixture, through=date(2026, 7, 10))
    assert first["status"] == "partial"
    assert first["issuers"][0]["fully_successful_through"] == "2026-07-08"
    assert first["issuers"][1]["fully_successful_through"] == "2026-07-10"
    saved = {path: file_sha256(path) for path in fixture.config.output_root.rglob("*") if path.is_file()}
    failed = list((fixture.config.output_root / CIKS[0]).glob("*/collection/_manifest.json"))[0]
    assert json.loads(failed.read_text())["state"] == "complete"
    assert json.loads(failed.read_text())["failed_issuers"] == 1
    fixture.source.fail.clear()
    fixture.source.calls.clear()
    second = _run(fixture, through=date(2026, 7, 11))
    assert second["status"] == "complete"
    assert fixture.source.calls == [
        (CIKS[0], date(2026, 7, 9), date(2026, 7, 11)),
        (CIKS[1], date(2026, 7, 11), date(2026, 7, 11)),
    ]
    assert all(file_sha256(path) == digest for path, digest in saved.items() if path.name != "latest_status.json")
    assert fixture.closes == [True, True]


def test_max_issuers_counts_attempts_not_verified_skips_and_offline_is_read_only(fixture):
    first = _run(fixture, through=date(2026, 7, 10), max_issuers=1)
    assert first["attempted_issuers"] == 1 and first["status"] == "partial"
    second = _run(fixture, through=date(2026, 7, 10), max_issuers=1)
    assert second["attempted_issuers"] == 1 and second["status"] == "complete"
    fixture.source.calls.clear()
    saved = {path: file_sha256(path) for path in fixture.config.output_root.rglob("*") if path.is_file()}
    replay = _run(fixture, through=date(2026, 7, 10), offline=True)
    assert replay["status"] == "complete"
    assert replay["coverage_complete"] is False and replay["current_ownership_claimed"] is False
    assert not fixture.source.calls
    assert all(file_sha256(path) == digest for path, digest in saved.items() if path.name != "latest_status.json")
    assert _run(fixture, through=date(2026, 7, 11), offline=True)["status"] == "partial"


def test_current_day_snapshot_is_partial_and_never_advances_closed_checkpoint(fixture):
    report = _run(fixture, through=NOW.date())
    assert report["requested_through"] == "2026-09-11"
    assert report["status"] == "snapshot_collected" and report["partial_current_day"] is True
    assert report["current_day_collected"] is True and report["requested_window_fully_collected"] is False
    assert report["capture_cutoff_utc"] == NOW.isoformat()
    assert fixture.source.calls == [
        (CIKS[0], date(2026, 7, 9), date(2026, 9, 10)), (CIKS[0], NOW.date(), NOW.date()),
        (CIKS[1], date(2026, 7, 9), date(2026, 9, 10)), (CIKS[1], NOW.date(), NOW.date()),
    ]
    assert all(row["fully_successful_through"] == "2026-09-10" for row in report["issuers"])
    snapshots = list(fixture.config.output_root.glob("snapshots/*/*/*/manifest.*.json"))
    assert len(snapshots) == 2
    for path in snapshots:
        snapshot = inc._read_pinned(path)
        assert snapshot["status"] == "partial_observed"
        assert snapshot["request"]["capture_cutoff_utc"] == NOW.isoformat()
        assert snapshot["request"]["forms"] == ["8-K"]
        assert snapshot["coverage_complete"] is False
    fixture.source.calls.clear()
    offline = _run(fixture, through=NOW.date(), offline=True)
    assert offline["current_day_collected"] is True and not fixture.source.calls
    fixture.source.calls.clear()
    default = _run(fixture)
    assert default["requested_through"] == "2026-09-10" and default["status"] == "complete"
    assert not fixture.source.calls


def test_failed_snapshot_retries_new_attempt_at_shared_cutoff(fixture):
    _run(fixture)
    fixture.source.fail.add(CIKS[0])
    failed = _run(fixture, through=NOW.date())
    assert failed["current_day_collected"] is False
    saved = list(fixture.config.output_root.glob("snapshots/*/*/*/manifest.*.json"))
    pins = {path: file_sha256(path) for path in saved}
    fixture.source.fail.clear()
    fixture.source.calls.clear()
    report = _run(fixture, through=NOW.date())
    assert report["current_day_collected"] is True
    assert fixture.source.calls == [(CIKS[0], NOW.date(), NOW.date())]
    assert all(file_sha256(path) == digest for path, digest in pins.items())


def test_cli_snapshot_success_exits_zero_but_missing_and_failed_snapshots_exit_two(fixture, capsys):
    _run(fixture)
    args = ["--root", str(fixture.root), "--config", "config.toml", "--through", "2026-09-11"]
    assert inc.main([*args, "--offline"]) == 2
    assert json.loads(capsys.readouterr().out)["current_day_collected"] is False
    fixture.source.fail.add(CIKS[0])
    assert inc.main(args) == 2
    failed = json.loads(capsys.readouterr().out)
    assert failed["status"] == "partial" and failed["current_day_collected"] is False
    fixture.source.fail.clear()
    assert inc.main(args) == 0
    complete = json.loads(capsys.readouterr().out)
    assert complete["status"] == "snapshot_collected"
    assert complete["partial_current_day"] is True and complete["coverage_complete"] is False
    assert complete["current_day_collected"] is True and complete["requested_window_fully_collected"] is False


def test_snapshot_zip_tamper_fails_offline(fixture):
    _run(fixture, through=NOW.date())
    path = next(fixture.config.output_root.glob("snapshots/*/*/*/raw_responses.zip"))
    with path.open("ab") as handle:
        handle.write(b"poison")
    with pytest.raises(DataReadinessError, match="pin mismatch"):
        _run(fixture, through=NOW.date(), offline=True)


@pytest.mark.parametrize("target", ["archive", "manifest", "relation", "receipt", "request"])
def test_pin_tampering_fails_closed_without_network(fixture, target):
    _run(fixture, through=date(2026, 7, 10))
    paths = {
        "archive": fixture.archive / "raw_responses.zip",
        "manifest": next(fixture.config.output_root.glob("*/*/collection/_manifest.json")),
        "relation": fixture.config.identity_relations,
        "receipt": next(fixture.config.output_root.glob("*/*/result.*.json")),
        "request": next(fixture.config.output_root.glob("*/*/request.json")),
    }
    with paths[target].open("ab") as handle:
        handle.write(b" ")
    fixture.source.calls.clear()
    with pytest.raises(DataReadinessError):
        _run(fixture, through=date(2026, 7, 10), offline=True)
    assert not fixture.source.calls


def test_current_request_matching_precedes_skip_even_with_valid_result_pin(fixture):
    _run(fixture, through=date(2026, 7, 10))
    relations = inc.load_sec_identity_relations(fixture.config.identity_relations)
    subset = relations.loc[relations["sec_cik"].eq(CIKS[0])].reset_index(drop=True)
    attempt = next((fixture.config.output_root / CIKS[0]).iterdir())
    request = inc._read(attempt / "request.json")
    expected = dict(request["collection_request"])
    expected["forms"] = ["10-K"]
    with pytest.raises(DataReadinessError, match="current pinned request"):
        inc._replay(attempt, request, expected)
    subset.loc[:, "ticker"] = "CHANGED"
    with pytest.raises(DataReadinessError, match="current pinned request"):
        inc._checkpoint(fixture.config, subset, inc.verify_archive(fixture.config, guard=lambda: None),
                        date(2026, 7, 10), lambda: None)


def test_successful_interval_after_gap_cannot_advance_checkpoint(fixture):
    relations = inc.load_sec_identity_relations(fixture.config.identity_relations)
    subset = relations.loc[relations["sec_cik"].eq(CIKS[0])].reset_index(drop=True)
    archive = inc.verify_archive(fixture.config, guard=lambda: None)
    gate = inc._RequestGate(SecRequestGovernor(), lambda: None)
    assert inc._attempt(fixture.config, subset, inc._collection_config(date(2026, 7, 10), date(2026, 7, 10), archive),
                        date(2026, 7, 10), fixture.source, gate)
    with pytest.raises(DataReadinessError, match="gap"):
        _run(fixture, through=date(2026, 7, 10), offline=True)


def test_pressure_stops_run_without_later_issuer_and_closes_session(fixture):
    fixture.source.pressure = True
    report = _run(fixture, through=date(2026, 7, 10))
    assert report["status"] == "paused_memory_pressure" and report["attempted_issuers"] == 1
    assert report["issuers"][0]["fully_successful_through"] == "2026-07-08"
    assert len(fixture.source.calls) == 1 and fixture.closes == [True]
    manifest_path = next(fixture.config.output_root.glob("*/*/collection/_manifest.json"))
    assert json.loads(manifest_path.read_text())["failed_issuers"] == 1


def test_http_retry_checks_pressure_before_governor_cooldown(monkeypatch):
    guard_calls, acquisitions, sends = [], [], []

    def guard():
        guard_calls.append(True)
        if sends:
            raise MemoryBudgetError("pressure after first response")

    gate = inc._RequestGate(SimpleNamespace(acquire=lambda: acquisitions.append(True)), guard)
    client = HttpClient(before_request=gate, additional_retriable_statuses=frozenset({403}))
    response = requests.Response()
    response.status_code = 403
    response._content = b""
    response._content_consumed = True

    def get(*args, **kwargs):
        sends.append(True)
        return response

    monkeypatch.setattr(client.session, "get", get)
    monkeypatch.setattr("market_predictor.sources.http.time.sleep", lambda seconds: None)
    try:
        with pytest.raises(MemoryBudgetError):
            client.get_bytes_with_metadata("https://data.sec.gov/submissions/CIK0000000001.json")
    finally:
        client.session.close()
    assert gate.pressure and len(sends) == 1 and len(acquisitions) == 1 and len(guard_calls) == 3


def test_busy_lease_precedes_any_input_load(tmp_path, monkeypatch):
    @contextmanager
    def busy(*args, **kwargs):
        raise HeavyJobBusyError("fixture lease occupied")
        yield

    monkeypatch.setattr(inc, "heavy_job_lease", busy)
    monkeypatch.setattr(inc, "load_config", lambda *args, **kwargs: pytest.fail("loaded input before lease"))
    with pytest.raises(HeavyJobBusyError):
        inc.run(tmp_path / "missing.toml", root=tmp_path, offline=True)


def test_environment_lease_contention_precedes_input_load(tmp_path, monkeypatch):
    runtime = tmp_path / "shared-runtime"
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(runtime))
    monkeypatch.setattr(inc, "load_config", lambda *args, **kwargs: pytest.fail("loaded before shared lease"))
    with heavy_job_lease("fixture competing job", runtime_dir=runtime):
        with pytest.raises(HeavyJobBusyError):
            inc.run(tmp_path / "missing.toml", root=tmp_path, offline=True)


def test_pinned_json_roundtrip_uses_exact_bytes_on_host_platform(tmp_path):
    inc._write_pinned(tmp_path, "result", {"multiline": ["first", "second"], "value": True})
    path = next(tmp_path.glob("result.*.json"))
    assert b"\r\n" not in path.read_bytes()
    assert file_sha256(path) == path.name.split(".")[1]
    assert inc._read_pinned(path)["value"] is True
    with pytest.raises(FileExistsError):
        inc._write_pinned(tmp_path, "result", {"multiline": ["first", "second"], "value": True})


@pytest.mark.parametrize("stage", ["request", "result"])
def test_atomic_publish_crash_leaves_no_truncated_final_and_resume_recovers(fixture, monkeypatch, stage):
    original = inc.os.link
    crashed = False

    def crash_once(source, target, *args, **kwargs):
        nonlocal crashed
        if not crashed and target.name.startswith(stage):
            crashed = True
            raise OSError("fixture interrupted publish")
        return original(source, target, *args, **kwargs)

    monkeypatch.setattr(inc.os, "link", crash_once)
    with pytest.raises(OSError, match="interrupted publish"):
        _run(fixture, through=date(2026, 7, 10), max_issuers=1)
    assert not list(fixture.config.output_root.glob(f"*/*/{stage}*.json"))
    fixture.source.calls.clear()
    if stage == "result":
        assert list(fixture.config.output_root.glob("*/*/collection/_manifest.json"))
        recovered = _run(fixture, through=date(2026, 7, 10), offline=True)
        assert recovered["issuers"][0]["fully_successful_through"] == "2026-07-10"
        assert not fixture.source.calls
    else:
        assert _run(fixture, through=date(2026, 7, 10), max_issuers=1)["attempted_issuers"] == 1


def test_console_is_compact_and_persistent_status_contains_issuers(fixture, capsys):
    result = inc.main([
        "--root", str(fixture.root), "--config", "config.toml", "--through", "2026-07-08", "--offline",
    ])
    assert result == 0
    console = json.loads(capsys.readouterr().out)
    assert "issuers" not in console
    latest = inc._read(fixture.config.output_root / "latest_status.json")
    assert len(latest["issuers"]) == 2
    assert latest["requested_through"] == "2026-07-08"


def test_config_cli_validation_and_portable_root(fixture, monkeypatch):
    with pytest.raises(DataReadinessError):
        _run(fixture, through=date(2026, 9, 12))
    with pytest.raises(DataReadinessError):
        _run(fixture, max_issuers=0)
    assert inc.main(["--config", str(fixture.path), "--max-issuers", "0"]) == 2
    assert inc.main(["--root", str(fixture.root), "--config", "config.toml", "--offline", "--through", "2026-07-08"]) == 0
    monkeypatch.chdir(fixture.root)
    assert inc.main(["--config", "config.toml", "--offline", "--through", "2026-07-08"]) == 0
    text = fixture.path.read_text().replace('output_root = "extension"', 'output_root = "archive/new"')
    fixture.path.write_text(text)
    with pytest.raises(DataReadinessError, match="separate"):
        inc.load_config(fixture.path, root=fixture.root)
