from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild import swing_history_collection as collector
from market_predictor.evidence.hashing import json_sha256
from market_predictor.sources.official_documents import collect_official_documents, verify_official_document_collection
from market_predictor.swing.datasets import symbol_corrections as owner
from market_predictor.swing.datasets.history_plan_publication import UNIT_COLUMNS
from market_predictor.swing.datasets.symbol_corrected_sources import (
    publish_or_verify_symbol_corrected_sources,
    reconstruct_symbol_corrected_sources,
)
from tests.test_official_document_collection import _fetch, _inventory
from tests.test_swing_history_collection import _FakeSource, _json, _plan, _test_transport_response, _write_json
from tests.test_swing_symbol_correction_collection import _resign

IDENTITIES = {"ECHO": "cik:0001415404", "FISV": "cik:0000798354"}


def _write_toml(path: Path, payload: dict[str, Any], table: str) -> None:
    lines = [f"{key} = {json.dumps(value)}" for key, value in payload.items() if key != table]
    for item in payload[table]:
        lines.extend(["", f"[[{table}]]", *(f"{key} = {json.dumps(value)}" for key, value in item.items())])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _TwoSessionSource(_FakeSource):
    """Synthetic transport only; no official facts or actual market prices."""

    def __init__(self, missing_benchmark_session: bool = False) -> None:
        super().__init__()
        self.missing_benchmark_session = missing_benchmark_session

    def fetch_daily_page(self, symbol: str, start: datetime, end_exclusive: datetime, *,
        page_token: str | None, asof: date, adjustment: str) -> collector.SwingDailyPage:
        page = super().fetch_daily_page(symbol, start, end_exclusive,
            page_token=page_token, asof=asof, adjustment=adjustment)
        bars = tuple(
            {**page.bars[0], "t": f"{day}T05:00:00Z"}
            for day in ("2020-01-02", "2020-01-03")
            if start <= datetime.fromisoformat(f"{day}T05:00:00+00:00") < end_exclusive
            and not (self.missing_benchmark_session and symbol == "SPY" and day == "2020-01-03")
        )
        payload = {"bars": {symbol: list(bars)}, "next_page_token": None}
        params = {"symbols": symbol, "timeframe": "1Day", "start": start.isoformat(),
            "end": (end_exclusive - timedelta(microseconds=1)).isoformat(), "feed": "sip", "limit": 10_000,
            "adjustment": adjustment, "sort": "asc", "asof": asof.isoformat()}
        receipt = replace(_test_transport_response(payload, params), retrieved_at_utc=datetime(2020, 1, 4, 6, tzinfo=UTC))
        return replace(page, bars=bars, raw_payload=payload, transport_response=receipt)


def _fixture(root: Path, *, missing_benchmark_session: bool = False) -> dict[str, Any]:
    plan = _plan(root, adjustment="raw")
    units = pd.DataFrame([
        {"security_id": IDENTITIES.get(ticker, f"benchmark:{ticker}"), "ticker": ticker,
         "start_date": "2020-01-02", "end_date": "2020-01-03",
         "role": "stock" if ticker in IDENTITIES else "benchmark"}
        for ticker in ("ECHO", "FISV", "SPY", "QQQ")
    ], columns=list(UNIT_COLUMNS))
    units.to_csv(plan / "daily_bar_units.csv", index=False, lineterminator="\n")
    request = _json(plan / "_request.json")
    request.update(scope="initial_fit_raw_share_acquisition", retained_security_ids=list(IDENTITIES.values()),
        provider_symbols={ticker: ticker for ticker in units.ticker},
        asof_policy="inclusive_unit_end_date_entity_mapping_not_ownership")
    _write_json(plan / "_request.json", request)
    manifest = _json(plan / "_manifest.json")
    manifest["scope"] = request["scope"]
    manifest["missing_session_ranges"][0].update(last_session="2020-01-03", sessions=2)
    manifest["daily_bars"].update(planned_units=4, stock_units=2, benchmark_units=2)
    _write_json(plan / "_manifest.json", manifest)
    plan_pin = _resign(plan)
    archive = root / "parent-archive"
    parent = collector.collect_swing_history_plan(
        plan_directory=plan, output_directory=archive,
        source_factory=lambda: _TwoSessionSource(missing_benchmark_session),
        provider_symbol_for=lambda ticker: ticker, expected_plan_authority_sha256=plan_pin,
    )
    assert parent["status"] == "complete"
    inventory = _inventory(4)
    inventory_path = root / "documents.toml"
    _write_toml(inventory_path, inventory.model_dump(mode="json"), "documents")
    documents = root / "documents"
    collect_official_documents(inventory=inventory, output_directory=documents, fetch=_fetch)
    policy = {
        "schema_version": "market_predictor.swing_symbol_correction_policy",
        "parent_plan": "plan", "parent_plan_sha256": plan_pin,
        "parent_archive": "parent-archive", "parent_archive_sha256": file_sha256(archive / "_authority.json"),
        "document_inventory": "documents.toml", "document_archive": "documents",
        "document_report_sha256": json_sha256(verify_official_document_collection(documents, inventory)),
        "corrections": [
            {"security_id": IDENTITIES[ticker], "ticker": ticker, "provider_symbol": provider,
             "parent_unit_id": next(row["unit_id"] for row in parent["unit_artifacts"] if row["ticker"] == ticker),
             "start_date": "2020-01-02" if ticker == "ECHO" else "2020-01-03", "end_date": "2020-01-03",
             "document_ids": [f"filing_{2 * index}", f"filing_{2 * index + 1}"],
             "interpretation": "Synthetic reviewed policy fixture, not an actual historical ownership assertion.",
             "record_locators": ["Synthetic cover fixture", "Synthetic transition fixture"]}
            for index, (ticker, provider) in enumerate((("ECHO", "SATS"), ("FISV", "FI")))
        ],
    }
    config = root / "corrections.toml"
    _write_toml(config, policy, "corrections")
    return {"root": root, "plan": plan, "archive": archive, "parent": parent,
        "config": config, "policy": policy, "policy_pin": file_sha256(config)}


@pytest.fixture
def evidence(tmp_path: Path) -> dict[str, Any]:
    return _fixture(tmp_path)


def _requirements(evidence: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    return owner.correction_requirements(evidence["root"], evidence["config"], evidence["policy_pin"],
        collector.load_complete_swing_history_collection)


def _publish(evidence: dict[str, Any]) -> tuple[Path, str]:
    path = evidence["root"] / "correction-plan"
    owner.publish_symbol_correction_plan(evidence["root"], evidence["config"], evidence["policy_pin"], path,
        collector.load_complete_swing_history_collection)
    return path, file_sha256(path / "_authority.json")


def _repin_policy(evidence: dict[str, Any]) -> None:
    _write_toml(evidence["config"], evidence["policy"], "corrections")
    evidence["policy_pin"] = file_sha256(evidence["config"])


def test_owner_reconstructs_two_units_and_replays_real_parent_without_network(evidence: dict[str, Any]) -> None:
    calls: list[dict[str, Any]] = []

    def replay(directory: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append({"directory": directory, **kwargs})
        return collector.load_complete_swing_history_collection(directory, **kwargs)

    request, manifest, units = owner.correction_requirements(
        evidence["root"], evidence["config"], evidence["policy_pin"], replay)
    assert calls == [{"directory": evidence["archive"], "plan_directory": evidence["plan"],
        "expected_adjustment": "raw", "expected_plan_authority_sha256": evidence["policy"]["parent_plan_sha256"]}]
    assert request["provider_symbols"] == {"ECHO": "SATS", "FISV": "FI"}
    assert request["policy_sha256"] == evidence["policy_pin"]
    assert request["policy"]["document_report_sha256"] == evidence["policy"]["document_report_sha256"]
    assert len(units) == 2 and units.role.eq("stock").all()
    assert {row["ticker"] for row in request["inherited_benchmarks"]} == {"SPY", "QQQ"}
    assert request["inherited_benchmarks"] == manifest["inherited_benchmarks"]
    assert [row["required_sessions"] for row in request["replacements"]] == [2, 1]
    assert request["replacements"][0]["required_sessions_sha256"] == json_sha256(["2020-01-02", "2020-01-03"])
    for key in ("accounting_eligible", "label_eligible", "promotion_eligible", "historical_availability_proven"):
        assert manifest[key] is False


def test_real_owner_integrates_with_correction_collection_and_offline_replay(evidence: dict[str, Any]) -> None:
    plan, pin = _publish(evidence)
    output = evidence["root"] / "correction-archive"
    source = _TwoSessionSource()
    manifest = collector.collect_swing_history_plan(
        plan_directory=plan, output_directory=output, source_factory=lambda: source,
        provider_symbol_for={"ECHO": "SATS", "FISV": "FI"}.__getitem__, expected_plan_authority_sha256=pin,
    )
    assert manifest["status"] == "complete" and manifest["requested_units"] == 2
    assert manifest["total_rows"] == 3
    assert {call["symbol"] for call in source.calls} == {"SATS", "FI"}
    source.calls.clear()
    assert collector.load_complete_swing_history_collection(
        output, plan_directory=plan, expected_adjustment="raw", expected_plan_authority_sha256=pin,
    ) == manifest
    assert source.calls == []


def test_real_correction_selection_discards_parent_intervals_and_rejects_rehashed_tamper(
    evidence: dict[str, Any],
) -> None:
    plan, pin = _publish(evidence)
    archive = evidence["root"] / "correction-archive"
    source = _TwoSessionSource()
    corrected = collector.collect_swing_history_plan(
        plan_directory=plan, output_directory=archive, source_factory=lambda: source,
        provider_symbol_for={"ECHO": "SATS", "FISV": "FI"}.__getitem__, expected_plan_authority_sha256=pin,
    )
    assert corrected["status"] == "complete" and corrected["total_rows"] == 3
    source.calls.clear()
    result = reconstruct_symbol_corrected_sources(
        root=evidence["root"], correction_plan=plan, correction_plan_sha256=pin,
        correction_archive=archive, correction_archive_sha256=file_sha256(archive / "_authority.json"),
        loader=collector.load_complete_swing_history_collection,
    )
    assert source.calls == []
    assert result["status"] == "source_selection_verified"
    assert result["selected_rows"] == 8 and result["discarded_parent_rows"] == 3
    assert result["missing_sessions"] == result["invalid_observations"] == 0
    assert result["exclusions_added"] == []
    segments = result["segments"]
    assert len(segments) == 5
    assert {
        (s["artifact"]["ticker"], s["artifact"]["provider_symbol"], s["selection"],
         s["first_session"], s["last_session"], s["rows"])
        for s in segments
    } == {
        ("ECHO", "SATS", "corrected", "2020-01-02", "2020-01-03", 2),
        ("FISV", "FISV", "parent", "2020-01-02", "2020-01-02", 1),
        ("FISV", "FI", "corrected", "2020-01-03", "2020-01-03", 1),
        ("SPY", "SPY", "parent", "2020-01-02", "2020-01-03", 2),
        ("QQQ", "QQQ", "parent", "2020-01-02", "2020-01-03", 2),
    }
    for segment in segments:
        is_corrected = segment["selection"] == "corrected"
        expected_archive = archive if is_corrected else evidence["archive"]
        expected_manifest = corrected if is_corrected else evidence["parent"]
        assert segment["archive"] == str(expected_archive)
        assert segment["artifact"] in expected_manifest["unit_artifacts"]
        assert segment["rows"] == segment["required_sessions"]
        assert file_sha256(expected_archive / segment["artifact"]["bars_path"]) == segment["artifact"]["bars_sha256"]
    for key in ("accounting_eligible", "label_eligible", "promotion_eligible", "historical_availability_proven"):
        assert result[key] is False
    assert result["audit_sha256"] == json_sha256({k: v for k, v in result.items() if k != "audit_sha256"})
    output = evidence["root"] / "selection.json"
    publish_or_verify_symbol_corrected_sources(output, result)
    publish_or_verify_symbol_corrected_sources(output, result, expected_sha256=file_sha256(output))
    tampered = _json(output)
    echo = next(s for s in tampered["segments"] if s["artifact"]["ticker"] == "ECHO")
    echo["selection"] = "parent"
    echo["archive"] = str(evidence["archive"])
    echo["artifact"] = next(a for a in evidence["parent"]["unit_artifacts"] if a["ticker"] == "ECHO")
    tampered["audit_sha256"] = json_sha256({k: v for k, v in tampered.items() if k != "audit_sha256"})
    _write_json(output, tampered)
    with pytest.raises(DataReadinessError, match="identical replay and independent file pin"):
        publish_or_verify_symbol_corrected_sources(output, result, expected_sha256=file_sha256(output))


def test_real_correction_selection_rejects_request_mutated_after_loader(evidence: dict[str, Any]) -> None:
    plan, pin = _publish(evidence)
    archive = evidence["root"] / "correction-archive"
    corrected = collector.collect_swing_history_plan(
        plan_directory=plan, output_directory=archive, source_factory=_TwoSessionSource,
        provider_symbol_for={"ECHO": "SATS", "FISV": "FI"}.__getitem__, expected_plan_authority_sha256=pin,
    )
    assert corrected["status"] == "complete"
    calls: list[Path] = []

    def replay_then_mutate(directory: Path, **kwargs: Any) -> dict[str, Any]:
        replayed = collector.load_complete_swing_history_collection(directory, **kwargs)
        assert replayed == corrected
        calls.append(directory)
        request = _json(plan / "_request.json")
        correction = request["policy"]["corrections"][0]
        assert correction["ticker"] == "ECHO" and correction["start_date"] == "2020-01-02"
        correction["start_date"] = "2020-01-03"
        _write_json(plan / "_request.json", request)
        return replayed

    output = evidence["root"] / "selection.json"
    with pytest.raises(DataReadinessError, match="independent file pin differs"):
        result = reconstruct_symbol_corrected_sources(
            root=evidence["root"], correction_plan=plan, correction_plan_sha256=pin,
            correction_archive=archive, correction_archive_sha256=file_sha256(archive / "_authority.json"),
            loader=replay_then_mutate,
        )
        publish_or_verify_symbol_corrected_sources(output, result)
    assert calls == [archive]
    assert file_sha256(plan / "_authority.json") == pin
    assert file_sha256(plan / "_request.json") != _json(plan / "_authority.json")["request_sha256"]
    assert not output.exists()


@pytest.mark.parametrize("fault", ["mapping", "embedded_policy", "security_id", "start_date", "end_date",
    "inherited_benchmark", "session_digest", "extra_row", "missing_row"])
def test_rehashed_manual_plan_edits_cannot_replace_frozen_reviewed_policy(evidence: dict[str, Any], fault: str) -> None:
    plan, original_pin = _publish(evidence)
    request = _json(plan / "_request.json")
    manifest = _json(plan / "_manifest.json")
    units = pd.read_csv(plan / "daily_bar_units.csv", dtype=str)
    if fault == "mapping":
        request["provider_symbols"]["ECHO"] = "WRONG"
    elif fault == "embedded_policy":
        request["policy"]["corrections"][0]["provider_symbol"] = "WRONG"
    elif fault in {"security_id", "start_date", "end_date"}:
        units.loc[units.ticker.eq("ECHO"), fault] = "cik:0001426945" if fault == "security_id" else "2020-01-03"
        if fault == "end_date":
            units.loc[units.ticker.eq("ECHO"), fault] = "2020-01-02"
    elif fault == "inherited_benchmark":
        request["inherited_benchmarks"] = manifest["inherited_benchmarks"] = []
    elif fault == "session_digest":
        request["replacements"][0]["required_sessions_sha256"] = "0" * 64
        manifest["replacements"][0]["required_sessions_sha256"] = "0" * 64
    elif fault == "extra_row":
        units = pd.concat([units, units.iloc[[0]].assign(ticker="OTHER")], ignore_index=True)
    else:
        units = units.iloc[:1]
    _write_json(plan / "_request.json", request)
    manifest["daily_bars"].update(planned_units=len(units), stock_units=len(units))
    _write_json(plan / "_manifest.json", manifest)
    units.to_csv(plan / "daily_bar_units.csv", index=False, lineterminator="\n")
    replacement_pin = _resign(plan)
    assert replacement_pin != original_pin
    with pytest.raises(DataReadinessError, match="pinned reviewed replacement requirements|provider identity policy"):
        collector._load_verified_plan(plan, expected_plan_authority_sha256=replacement_pin)


def test_policy_byte_change_cannot_be_hidden_by_plan_rehash(evidence: dict[str, Any]) -> None:
    plan, _ = _publish(evidence)
    evidence["policy"]["corrections"][0]["provider_symbol"] = "WRONG"
    _write_toml(evidence["config"], evidence["policy"], "corrections")
    replacement_pin = _resign(plan)
    with pytest.raises(DataReadinessError, match="independent reviewed policy pin"):
        collector._load_verified_plan(plan, expected_plan_authority_sha256=replacement_pin)


@pytest.mark.parametrize("field,value", [
    ("security_id", "cik:0001426945"), ("ticker", "OTHER"), ("provider_symbol", "ECHO"),
    ("parent_unit_id", "swing-daily-" + "0" * 24),
    ("start_date", "2019-12-31"), ("end_date", "2020-01-06"), ("start_date", "2020-01-04"),
    ("document_ids", ["filing_0", "absent_filing"]), ("document_ids", ["filing_0", "filing_0"]),
    ("record_locators", ["one", "two", "unmatched"]),
])
def test_reviewed_policy_must_still_match_parent_identity_dates_and_documents(
    evidence: dict[str, Any], field: str, value: Any,
) -> None:
    evidence["policy"]["corrections"][0][field] = value
    _repin_policy(evidence)
    with pytest.raises(DataReadinessError, match="identity, interval or reviewed source"):
        _requirements(evidence)


@pytest.mark.parametrize("key", ["security_id", "ticker"])
def test_two_corrections_cannot_alias_the_same_retained_identity(evidence: dict[str, Any], key: str) -> None:
    evidence["policy"]["corrections"][1][key] = evidence["policy"]["corrections"][0][key]
    _repin_policy(evidence)
    with pytest.raises(DataReadinessError, match="two distinct"):
        _requirements(evidence)


def test_inherited_benchmark_missing_session_fails_after_successful_parent_archive_replay(tmp_path: Path) -> None:
    evidence = _fixture(tmp_path, missing_benchmark_session=True)
    assert collector.load_complete_swing_history_collection(
        evidence["archive"], plan_directory=evidence["plan"], expected_adjustment="raw",
        expected_plan_authority_sha256=evidence["policy"]["parent_plan_sha256"],
    )["status"] == "complete"
    with pytest.raises(DataReadinessError, match="inherited benchmark exact session coverage"):
        _requirements(evidence)


@pytest.mark.parametrize("artifact", ["policy", "parent_plan_authority", "parent_archive_authority",
    "parent_manifest", "parent_benchmark_bars", "parent_stock_body", "document_body", "document_inventory"])
def test_hash_bound_input_tamper_blocks_owner(evidence: dict[str, Any], artifact: str) -> None:
    if artifact == "policy":
        path = evidence["config"]
    elif artifact == "parent_plan_authority":
        path = evidence["plan"] / "_authority.json"
    elif artifact == "parent_archive_authority":
        path = evidence["archive"] / "_authority.json"
    elif artifact == "parent_manifest":
        path = evidence["archive"] / "_manifest.json"
    elif artifact == "parent_benchmark_bars":
        benchmark = next(row for row in evidence["parent"]["unit_artifacts"] if row["ticker"] == "SPY")
        path = evidence["archive"] / benchmark["bars_path"]
    elif artifact == "parent_stock_body":
        stock = next(row for row in evidence["parent"]["unit_artifacts"] if row["ticker"] == "ECHO")
        unit = _json(evidence["archive"] / stock["unit_manifest_path"])
        path = evidence["archive"] / unit["pages"][0]["transport"]["body_path"]
    elif artifact == "document_body":
        path = next((evidence["root"] / "documents").glob("documents/*/*/body-*.bin"))
    else:
        inventory = _json(evidence["root"] / "documents" / "_request.json")
        inventory["inventory_sha256"] = "0" * 64
        _write_json(evidence["root"] / "documents" / "_request.json", inventory)
        with pytest.raises(DataReadinessError):
            _requirements(evidence)
        return
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(DataReadinessError):
        _requirements(evidence)


def test_semantic_document_report_pin_is_not_replaced_by_report_file_hash(evidence: dict[str, Any]) -> None:
    report = next((evidence["root"] / "documents" / "reports").glob("*.json"))
    evidence["policy"]["document_report_sha256"] = file_sha256(report)
    _repin_policy(evidence)
    with pytest.raises(DataReadinessError, match="reviewed document collection differs"):
        _requirements(evidence)


def test_parent_scope_cannot_be_a_correction_archive(evidence: dict[str, Any]) -> None:
    request = _json(evidence["plan"] / "_request.json")
    request["scope"] = owner.CORRECTION_SCOPE
    _write_json(evidence["plan"] / "_request.json", request)
    manifest = _json(evidence["plan"] / "_manifest.json")
    manifest["scope"] = owner.CORRECTION_SCOPE
    _write_json(evidence["plan"] / "_manifest.json", manifest)
    evidence["policy"]["parent_plan_sha256"] = _resign(evidence["plan"])
    _repin_policy(evidence)
    with pytest.raises(DataReadinessError, match="original initial-fit plan"):
        _requirements(evidence)


def test_published_correction_plan_is_immutable(evidence: dict[str, Any]) -> None:
    output, pin = _publish(evidence)
    with pytest.raises(DataReadinessError, match="must be new"):
        _publish(evidence)
    assert file_sha256(output / "_authority.json") == pin
