from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import pytest
import requests
from typer.testing import CliRunner

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.research import swing_transfer_replay as replay
from market_predictor.sources.alpaca import decode_bars_page_response
from market_predictor.sources.http import HttpByteResponse


def _unit(ticker="ABC"):
    unit = {
        "ticker": ticker, "security_id": f"security:{ticker}", "sec_cik": "0000000001",
        "required_sessions": ["2020-06-09"],
        "retained_rows": [{"session_date_et": "2020-06-09", "open": 10., "high": 12.,
                           "low": 9., "close": 11., "volume": 100.}],
        "parameters": {"symbols": ticker, "timeframe": "1Day", "feed": "sip", "adjustment": "all",
                       "sort": "asc", "limit": 10000, "start": "2020-06-09T00:00:00-04:00",
                       "end": "2020-06-09T23:59:59.999999-04:00", "asof": "2020-06-08"},
    }
    return {**unit, "unit_sha256": json_sha256(unit)}


def _row(**updates):
    return {"t": "2020-06-09T04:00:00Z", "o": 10., "h": 12., "l": 9., "c": 11., "v": 100., **updates}


def _page(unit, token=None, *, rows=None, next_token=None, query_update=None):
    params = {**unit["parameters"], **({"page_token": token} if token is not None else {})}
    query = {**params, **(query_update or {})}
    url = f"https://data.alpaca.markets/v2/stocks/bars?{urlencode(query)}"
    body = json.dumps({"bars": {unit["ticker"]: [_row()] if rows is None else rows},
                       "next_page_token": next_token}).encode()
    response = HttpByteResponse(
        body=body, requested_url=url, final_url=url, redirect_chain=(), status_code=200,
        retrieved_at_utc=datetime.now(UTC), content_type="application/json", content_encoding=None,
        etag=None, last_modified=None, body_length=len(body), sha256=hashlib.sha256(body).hexdigest(),
        body_representation="http_entity_encoded", safe_headers=(("content-type", "application/json"),),
    )
    return decode_bars_page_response(response, expected_params=query)


@pytest.fixture
def collection(tmp_path, monkeypatch):
    request = {"units": [_unit(), _unit("DEF")], "maximum_pages_per_ticker": 3, "bound_files": {}}
    request["request_sha256"] = json_sha256(request)
    monkeypatch.setattr(replay, "_request", lambda *_: request)
    monkeypatch.setattr(replay, "heavy_job_runtime_dir", lambda: tmp_path / "runtime")
    return tmp_path, tmp_path / "archive", request


def _run(collection, fetch=None):
    root, output, _ = collection
    return replay.run_swing_transfer_replay(root=root, config_path=root / "config.toml",
                                          output_directory=output, fetch=fetch)


def test_exact_archive_and_offline_replay_do_not_admit_or_replace(collection):
    report = _run(collection, _page)
    assert report["acquisition_status"] == "complete"
    assert report["exact_match_units"] == 2
    assert not any(report[key] for key in ("identity_admission", "accounting_eligible", "retained_observations_replaced"))
    root, output, _ = collection
    before = {p.relative_to(root): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert _run(collection) == {key: value for key, value in report.items() if key != "report_path"}
    _run(collection, lambda *_: pytest.fail("successful evidence must not be fetched again"))
    assert before == {p.relative_to(root): p.read_bytes() for p in output.rglob("*") if p.is_file()}


def test_failure_is_isolated_and_resume_starts_failed_unit_only(collection):
    calls = []

    def fail_one(unit, token):
        calls.append((unit["ticker"], token))
        if unit["ticker"] == "ABC":
            if token is not None:
                raise requests.ConnectionError("test transport failure")
            return _page(unit, token, rows=[], next_token="second")
        return _page(unit, token)

    report = _run(collection, fail_one)
    assert report["acquired_units"] == 1
    failed = collection[1] / report["units"][0]["attempts"][0]["receipt_path"]
    assert len(json.loads(failed.read_bytes())["pages"]) == 1
    calls.clear()

    def resume(unit, token):
        calls.append((unit["ticker"], token))
        return _page(unit, token)

    assert _run(collection, resume)["acquired_units"] == 2
    assert calls == [("ABC", None)]
    assert len(_run(collection)["units"][0]["attempts"]) == 2


@pytest.mark.parametrize("kind", ["repeat", "cap"])
def test_pagination_is_bounded_and_retains_failed_prefix(collection, kind):
    calls = []

    def pages(unit, token):
        calls.append((unit["ticker"], token))
        return _page(unit, token, rows=[], next_token="same" if kind == "repeat" else str(len(calls)))

    report = _run(collection, pages)
    assert report["acquired_units"] == 0
    assert len(calls) == (4 if kind == "repeat" else 6)
    assert _run(collection)["acquisition_status"] == "incomplete"


@pytest.mark.parametrize("updates", [{"asof": "2026-01-01"}, {"symbols": "XYZ"}, {"feed": "iex"},
                                     {"start": "2020-06-08T00:00:00-04:00"}, {"page_token": "wrong"}])
def test_wrong_actual_query_is_a_failed_attempt(collection, updates):
    report = _run(collection, lambda unit, token: _page(unit, token, query_update=updates))
    assert report["acquired_units"] == 0
    assert all(item["attempts"][0]["state"] == "failed" for item in report["units"])


@pytest.mark.parametrize("rows,issue", [
    ([], None), ([_row(), _row()], "duplicate_session"), ([_row(v=0)], "unusable_daily_observation"),
    ([_row(h=8)], "unusable_daily_observation"), ([_row(c=float("nan"))], "unusable_daily_observation"),
    ([_row(t="2020-06-10T04:00:00Z")], "outside_required_sessions"),
    ([_row(t="2020-06-09T13:30:00Z")], "malformed_daily_observation"),
    ([_row(t="2020-06-09")], "malformed_daily_observation"),
])
def test_unusable_rows_remain_unavailable(rows, issue):
    result = replay.compare_transfer_observations(_unit(), rows)
    assert result["status"] == "review_required"
    if issue:
        assert result["issues"][0]["code"] == issue
    assert not result["identity_admission"]


def test_different_prices_are_reported_not_replaced(collection):
    report = _run(collection, lambda unit, token: _page(unit, token, rows=[_row(c=10.5)]))
    assert report["acquired_units"] == 2 and report["exact_match_units"] == 0
    assert report["units"][0]["comparison"]["differences"] == [
        {"session": "2020-06-09", "field": "close", "retained": 11., "replay": 10.5, "delta": -0.5}]
    assert collection[2]["units"][0]["retained_rows"][0]["close"] == 11.


@pytest.mark.parametrize("target", ["body", "metadata", "receipt", "request", "report"])
def test_byte_tampering_rejected(collection, target):
    report = _run(collection, _page)
    output = collection[1]
    receipt = output / report["units"][0]["attempts"][0]["receipt_path"]
    page = json.loads(receipt.read_bytes())["pages"][0]
    path = {"body": receipt.parent / page["body_path"], "metadata": receipt.parent / page["metadata_path"],
            "receipt": receipt, "request": output / "_request.json", "report": Path(report["report_path"])}[target]
    if target == "request":
        payload = json.loads(path.read_bytes())
        payload["request_sha256"] = "f" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(DataReadinessError):
        _run(collection)


@pytest.mark.parametrize("change", ["eligibility", "comparison", "counts", "duplicate_attempt"])
def test_rehashed_report_cannot_invent_eligibility_or_comparison(collection, change):
    report = _run(collection, _page)
    path = Path(report.pop("report_path"))
    if change == "eligibility":
        report["accounting_eligible"] = True
    elif change == "comparison":
        report["units"][0]["comparison"]["observed_rows"] += 1
    elif change == "counts":
        report["exact_match_units"] = 0
    else:
        report["units"][0]["attempts"].append(report["units"][0]["attempts"][0])
    path.unlink()
    body = json.dumps(report).encode()
    (path.parent / f"{hashlib.sha256(body).hexdigest()}.json").write_bytes(body)
    with pytest.raises(DataReadinessError):
        _run(collection)


def test_interrupted_publication_stays_pending_and_is_not_reused(collection, monkeypatch):
    rename = replay.os.rename

    def interrupt(source, destination):
        if "units" in Path(destination).parts:
            raise OSError("test disk interruption")
        return rename(source, destination)

    monkeypatch.setattr(replay.os, "rename", interrupt)
    with pytest.raises(OSError, match="interruption"):
        _run(collection, _page)
    monkeypatch.setattr(replay.os, "rename", rename)
    assert _run(collection)["acquired_units"] == 0
    assert _run(collection, _page)["acquired_units"] == 2


def test_resource_guard_stops_all_work_not_just_one_ticker(collection, monkeypatch):
    calls = []

    def fetch(unit, token):
        calls.append(unit["ticker"])
        return _page(unit, token)

    def exhausted(**_):
        raise DataReadinessError("test memory limit")

    monkeypatch.setattr(replay, "assert_memory_budget", exhausted)
    with pytest.raises(DataReadinessError, match="memory limit"):
        _run(collection, fetch)
    assert calls == ["ABC"]
    assert _run(collection)["acquired_units"] == 0


def test_lease_precedes_any_source_loading(collection, monkeypatch):
    @contextmanager
    def blocked(*_, **__):
        raise DataReadinessError("test lease busy")
        yield

    monkeypatch.setattr(replay, "heavy_job_lease", blocked)
    monkeypatch.setattr(replay, "_request", lambda *_: pytest.fail("input opened before lease"))
    with pytest.raises(DataReadinessError, match="lease busy"):
        _run(collection, _page)


def test_offline_requires_existing_archive(collection):
    with pytest.raises(DataReadinessError, match="existing collection"):
        _run(collection)


def test_output_must_stay_inside_root(collection):
    root, _, _ = collection
    with pytest.raises(DataReadinessError, match="inside the repository"):
        replay.run_swing_transfer_replay(root=root, config_path=root / "config.toml",
                                        output_directory=root.parent / "outside", fetch=_page)


def test_offline_cli_never_loads_credentials(monkeypatch):
    from market_predictor.commands import swing_research
    from market_predictor.research_cli import app

    monkeypatch.setattr(swing_research, "get_settings", lambda: pytest.fail("offline loaded credentials"))

    def run(**kwargs):
        assert kwargs["fetch"] is None
        return {"acquisition_status": "complete", "requested_units": 1, "acquired_units": 1,
                "exact_match_units": 1, "identity_admission": False, "accounting_eligible": False,
                "retained_observations_replaced": False}

    monkeypatch.setattr(swing_research, "run_swing_transfer_replay", run)
    result = CliRunner().invoke(app, ["replay-swing-transfer-history", "--output-directory", "archive", "--offline"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("ambiguity", ["field", "symbol", "symbol_case"])
def test_ambiguous_raw_response_rejected_at_network_and_offline_boundaries(collection, ambiguity):
    row = json.dumps(_row())
    if ambiguity == "field":
        ambiguous_row = row.replace('"c": 11.0', '"c": 7.0, "c": 11.0')
        body = ('{"bars":{"ABC":[' + ambiguous_row + ']},"next_page_token":null}').encode()
    else:
        duplicate = "ABC" if ambiguity == "symbol" else "abc"
        body = ('{"bars":{"ABC":[' + json.dumps(_row(c=7.)) + '],"' + duplicate
                + '":[' + row + ']},"next_page_token":null}').encode()
    report = _run(collection, _page)
    receipt_path = collection[1] / report["units"][0]["attempts"][0]["receipt_path"]
    receipt = json.loads(receipt_path.read_bytes())
    page_record = receipt["pages"][0]
    metadata_path = receipt_path.parent / page_record["metadata_path"]
    metadata = json.loads(metadata_path.read_bytes())
    metadata["body_length"] = len(body)
    metadata["body_sha256"] = hashlib.sha256(body).hexdigest()
    response = replay._http_response(body, metadata)
    with pytest.raises(RuntimeError):
        decode_bars_page_response(response, expected_params=collection[2]["units"][0]["parameters"])

    # Correctly rehash the test archive to prove semantic rejection, not just a checksum failure.
    (receipt_path.parent / page_record["body_path"]).unlink()
    page_record["body_path"] = f"000-{metadata['body_sha256']}.bin"
    (receipt_path.parent / page_record["body_path"]).write_bytes(body)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    page_record["metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = json_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError):
        _run(collection)


def test_archive_decoder_does_not_trust_preparsed_matching_values(collection):
    def poisoned(unit, token):
        valid = _page(unit, token)
        assert valid.raw_body is not None
        return replace(valid, raw_body=valid.raw_body.replace(b'"c": 11.0', b'"c": 7.0, "c": 11.0'))

    assert _run(collection, poisoned)["acquired_units"] == 0
