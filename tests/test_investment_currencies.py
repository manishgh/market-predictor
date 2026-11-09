from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.research import investment_currencies as currency
from market_predictor.research import investment_features as artifacts
from market_predictor.sources.http import HttpByteResponse
from market_predictor.swing.contracts.holding_materialization import SourcePin


def dividend(action_id="a", **updates):
    return {
        "id": action_id,
        "symbol": "AAA",
        "ex_date": "2024-05-01",
        "rate": 0.25,
        "foreign": False,
        "special": False,
        "record_date": "2024-05-02",
        "payable_date": "2024-05-15",
        **updates,
    }


def payload(*records, **updates):
    return {"corporate_actions": {"cash_dividends": list(records)}, **updates}


def test_only_exact_explicit_USD_currency_records_resolve():
    original = dividend()
    resolved, gaps = currency._decode_response(payload(dividend(currency="USD")), {"a": original})
    assert not gaps and resolved["a"][0]["currency"] == "USD"
    for change in (
        {"rate": 0.26},
        {"symbol": "BBB"},
        {"ex_date": "2024-05-02"},
        {"special": True},
        {"foreign": True},
        {"payable_date": "2024-05-16"},
        {"record_date": "2024-05-03"},
        {"currency": "EUR"},
        {"currency": None},
        {"foreign": 0},
    ):
        records, gaps = currency._decode_response(
            payload(dividend(currency="USD", **change) if "currency" not in change else dividend(**change)), {"a": original}
        )
        assert not records and "a" in gaps


def test_unrequested_duplicate_pagination_and_missing_ids_are_explicit():
    originals = {"a": dividend()}
    for source in (
        payload(dividend("b", currency="USD")),
        payload(dividend(currency="USD"), dividend(currency="USD")),
        payload(dividend(currency="USD"), next_page_token="next"),
    ):
        with pytest.raises(DataReadinessError):
            currency._decode_response(source, originals)
    assert currency._decode_response(payload(), originals)[1] == {"a": "requested_dividend_not_returned"}


def fixture(tmp_path, monkeypatch, records=None, response_transform=None):
    records = records or [dividend()]
    config = tmp_path / "actions.toml"
    config.write_text("pinned action config")
    pin = SourcePin(path=str(config), sha256=currency.file_sha256(config))
    calls, phases = [], []

    def replay(**kwargs):
        phases.append("replay")
        return SimpleNamespace(records_by_symbol={"AAA": {"cash_dividends": records}}, source_files={}, recheck=lambda root: None)

    monkeypatch.setattr(currency, "load_corporate_action_evidence", replay)

    @contextmanager
    def lease(*args, **kwargs):
        phases.append("lease")
        yield

    monkeypatch.setattr(currency, "heavy_job_lease", lease)
    monkeypatch.setattr(currency, "guard", lambda *args: None)
    monkeypatch.setattr(artifacts, "guard", lambda *args: None)

    def get(url, *, headers, params, maximum_body_bytes):
        calls.append(params)
        assert set(params) == {"ids"} and maximum_body_bytes == currency.MAXIMUM_RESPONSE_BYTES
        ids = params["ids"].split(",")
        fresh = [dict(r, currency="USD") for r in records if r["id"] in ids]
        source = payload(*fresh)
        if response_transform:
            source = response_transform(source, ids)
        body = json.dumps(source).encode()
        return HttpByteResponse(
            body=body,
            requested_url=url,
            final_url=url,
            redirect_chain=(),
            status_code=200,
            retrieved_at_utc=datetime(2026, 11, 7, tzinfo=UTC),
            content_type="application/json",
            content_encoding=None,
            etag=None,
            last_modified=None,
            body_length=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            body_representation="decoded",
            safe_headers=(),
        )

    source = SimpleNamespace(
        client=SimpleNamespace(get_bytes_with_metadata=get),
        corporate_actions_url="https://data.alpaca.markets/v1/corporate-actions",
        headers={"secret": "never-persist-this"},
    )
    monkeypatch.setattr(currency, "AlpacaSource", lambda settings: source)
    kwargs = dict(
        root=tmp_path,
        action_config=pin,
        action_archive=tmp_path / "old_archive",
        action_audit_sha256="a" * 64,
        output=tmp_path / "currency",
    )
    return kwargs, calls, phases


def test_collector_replays_before_lease_and_reader_binds_real_body_clock(tmp_path, monkeypatch):
    kwargs, calls, phases = fixture(tmp_path, monkeypatch)
    result = currency.collect_investment_dividend_currencies(**kwargs)
    assert phases[:2] == ["replay", "lease"] and len(calls) == 1
    manifest = tmp_path / "currency/final/_manifest.json"
    result = currency.load_investment_currency_evidence(
        tmp_path, manifest, currency.file_sha256(manifest), {}, interpretation_policy_sha256="b" * 64
    )
    record, evidence = result["a"]
    assert record["original_record_sha256"] == json_sha256(dividend())
    assert evidence.retrieved_at == evidence.available_at == datetime(2026, 11, 7, tzinfo=UTC)
    assert evidence.interpretation_policy_sha256 == "b" * 64
    assert currency.file_sha256(tmp_path / evidence.reference) == evidence.artifact_sha256
    assert evidence.reference.endswith("response.bin")
    raw = next((tmp_path / "currency/batches").glob("*/response.bin")).read_text()
    assert "original_record_sha256" not in raw
    assert "never-persist-this" not in "".join(p.read_text() for p in (tmp_path / "currency").rglob("*") if p.is_file())


def test_changed_provider_dividend_is_gap_not_replacement(tmp_path, monkeypatch):
    def changed(source, ids):
        source["corporate_actions"]["cash_dividends"][0]["rate"] = 99
        return source

    kwargs, _, _ = fixture(tmp_path, monkeypatch, response_transform=changed)
    currency.collect_investment_dividend_currencies(**kwargs)
    report = json.loads((tmp_path / "currency/final/_manifest.json").read_text())
    assert report["resolved_USD_actions"] == 0 and report["unresolved_actions"] == 1


def test_batch_failures_continue_and_counts_reconcile(tmp_path, monkeypatch):
    records = [dividend(str(i)) for i in range(51)]

    def fail_first(source, ids):
        if len(ids) == 50:
            raise RuntimeError("independent source batch failure")
        return source

    kwargs, calls, _ = fixture(tmp_path, monkeypatch, records, fail_first)
    currency.collect_investment_dividend_currencies(**kwargs)
    report = json.loads((tmp_path / "currency/final/_manifest.json").read_text())
    assert len(calls) == 2 and report["requested_actions"] == 51
    assert report["resolved_USD_actions"] == 1 and report["unresolved_actions"] == 50


def test_memory_refusal_is_fatal_and_does_not_publish_final(tmp_path, monkeypatch):
    def fail(source, ids):
        raise MemoryBudgetError("memory refusal")

    kwargs, _, _ = fixture(tmp_path, monkeypatch, response_transform=fail)
    with pytest.raises(MemoryBudgetError):
        currency.collect_investment_dividend_currencies(**kwargs)
    assert not (tmp_path / "currency/final").exists()


def test_reader_rejects_mutated_raw_response(tmp_path, monkeypatch):
    kwargs, _, _ = fixture(tmp_path, monkeypatch)
    currency.collect_investment_dividend_currencies(**kwargs)
    manifest = tmp_path / "currency/final/_manifest.json"
    next((tmp_path / "currency/batches").glob("*/response.bin")).write_text("{}")
    with pytest.raises(DataReadinessError, match="source changed"):
        currency.load_investment_currency_evidence(
            tmp_path, manifest, currency.file_sha256(manifest), {}, interpretation_policy_sha256="b" * 64
        )


def test_interrupted_collection_resumes_verified_batch_without_refetch(tmp_path, monkeypatch):
    records = [dividend(str(i)) for i in range(51)]
    interrupted = [True]

    def interrupt_last(source, ids):
        if len(ids) == 1 and interrupted[0]:
            raise MemoryBudgetError("stop after first saved batch")
        return source

    kwargs, calls, _ = fixture(tmp_path, monkeypatch, records, interrupt_last)
    with pytest.raises(MemoryBudgetError):
        currency.collect_investment_dividend_currencies(**kwargs)
    assert len(calls) == 2 and not (tmp_path / "currency/final").exists()
    interrupted[0] = False
    currency.collect_investment_dividend_currencies(**kwargs)
    assert len(calls) == 3
    report = json.loads((tmp_path / "currency/final/_manifest.json").read_text())
    assert report["resolved_USD_actions"] == 51 and report["unresolved_actions"] == 0


@pytest.mark.parametrize(
    "flag,value",
    [("cancelled", True), ("canceled", True), ("revised", True), ("revision", "2"), ("due_bill_redemption_date", "2024-06-01")],
)
def test_new_unsupported_refresh_terms_cannot_qualify_currency(flag, value):
    refreshed = dividend(currency="USD", **{flag: value})
    resolved, gaps = currency._decode_response(payload(refreshed), {"a": dividend()})
    assert not resolved and "unsupported_terms" in gaps["a"]


def test_refresh_must_retain_all_original_terms():
    original = dividend(process_date="2024-05-01")
    resolved, gaps = currency._decode_response(payload(dividend(currency="USD")), {"a": original})
    assert not resolved and gaps["a"].endswith("process_date")


def test_explicit_hashed_event_scope_collects_post_swing_end_without_changing_default(tmp_path, monkeypatch):
    original = dividend(ex_date="2025-03-24")
    kwargs, calls, _ = fixture(tmp_path, monkeypatch, [original])
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps({"schema": "market_predictor.investment_dividend_event_scope", "events": [{
        "action_id": "a", "original_record_sha256": json_sha256(original)}]}))
    kwargs["required_events"] = SourcePin(path=str(scope), sha256=currency.file_sha256(scope))
    currency.collect_investment_dividend_currencies(**kwargs)
    assert len(calls) == 1
    report = json.loads((tmp_path / "currency/final/_manifest.json").read_text())
    assert report["requested_actions"] == 1
    assert report["request"]["required_events"]["sha256"] == currency.file_sha256(scope)


def test_explicit_event_scope_rejects_wrong_original_before_provider_call(tmp_path, monkeypatch):
    kwargs, calls, _ = fixture(tmp_path, monkeypatch)
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps({"schema": "market_predictor.investment_dividend_event_scope", "events": [{
        "action_id": "a", "original_record_sha256": "f" * 64}]}))
    kwargs["required_events"] = SourcePin(path=str(scope), sha256=currency.file_sha256(scope))
    with pytest.raises(DataReadinessError, match="original hash"):
        currency.collect_investment_dividend_currencies(**kwargs)
    assert not calls


def test_explicit_scope_collection_honors_configured_shared_runtime(tmp_path, monkeypatch):
    configured = tmp_path / "shared-custom-runtime"
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(configured))
    kwargs, _, _ = fixture(tmp_path, monkeypatch)
    observed = []

    @contextmanager
    def lease(*args, **values):
        observed.append(values["runtime_dir"])
        yield
    monkeypatch.setattr(currency, "heavy_job_lease", lease)
    currency.collect_investment_dividend_currencies(**kwargs)
    assert observed == [configured]
