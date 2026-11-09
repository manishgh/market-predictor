"""Exact missing-session acquisition, quarantine, partial replay and immutable resume."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.investment import raw_acquisition as raw
from market_predictor.swing.datasets import history_archive as archive
from market_predictor.swing.datasets.history_plan_publication import AUTHORITY_SCHEMA, DAILY_BAR_UNITS_FILE
from market_predictor.swing.labels.holding_paths import holding_calendar
from tests.test_swing_history_collection import _test_transport_response


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pin(path: Path, root: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}


@pytest.fixture
def acquisition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", "data/runtime")
    lineage = {"universe_sha256": "1" * 64, "parent_lineage": {"fixture": "raw"}}
    parent = tmp_path / "parent"
    parent_request = {"scope": "initial_fit_raw_share_acquisition", "initial_fit_end": "2024-05-28",
        "membership_authority": lineage, "retained_security_ids": ["stock:AAA", "stock:BBB"], "excluded_security_ids": ["excluded"]}
    _write(parent / "_request.json", parent_request)
    _write(parent / "_manifest.json", {"daily_bars": {"adjustment": "raw"}})
    _write(parent / "_authority.json", {"schema": AUTHORITY_SCHEMA, "state": "complete", "artifact": "_manifest.json",
        "artifact_sha256": file_sha256(parent / "_manifest.json"), "request_sha256": file_sha256(parent / "_request.json")})
    segments: list[dict[str, Any]] = []
    for ticker, role, last in (("AAA", "stock", "2024-05-28"), ("BBB", "stock", "2024-05-24"),
                               ("SPY", "benchmark", "2024-05-28"), ("QQQ", "benchmark", "2024-05-28"),
                               ("XLK", "benchmark", "2024-05-28")):
        source = tmp_path / "retained" / ticker
        source.mkdir(parents=True)
        (source / "bars.parquet").write_bytes(b"retained immutable raw evidence")
        _write(source / "unit.json", {"retained": ticker})
        identity = f"{role}:{ticker}"
        artifact = {"security_id": identity, "ticker": ticker, "provider_symbol": ticker, "role": role, "status": "observed",
            "bars_path": "bars.parquet", "bars_sha256": file_sha256(source / "bars.parquet"),
            "unit_manifest_path": "unit.json", "unit_manifest_sha256": file_sha256(source / "unit.json")}
        segments.append({"archive": source.relative_to(tmp_path).as_posix(), "artifact": artifact,
            "first_session": "2024-04-30", "last_session": last, "missing_sessions": []})
    selection = {"schema": "market_predictor.swing_symbol_corrected_sources", "status": "source_selection_verified",
        "exclusions_added": [], "parent_plan_sha256": file_sha256(parent / "_authority.json"), "segments": segments}
    selection["audit_sha256"] = json_sha256(selection)
    _write(tmp_path / "selection.json", selection)
    rows = [{"security_id": "stock:AAA", "decision_id": "aaa-decision", "decision_session": "2024-05-28",
        "sector_benchmark": "XLK", "status": "eligible"},
        {"security_id": "stock:BBB", "decision_id": "bbb-decision", "decision_session": "2024-04-30",
        "sector_benchmark": "XLK", "status": "eligible"},
        {"security_id": "excluded", "decision_id": "excluded-decision", "decision_session": "2024-05-28",
        "sector_benchmark": "XLK", "status": "unavailable"}]
    _write(tmp_path / "roster.json", {"rows": rows})
    _write(tmp_path / "actions.json", {"policy": {"process_start": "2018-05-29", "process_end": "2026-09-11", "numeric_end": "2024-05-28"}})
    calendar = holding_calendar(date(2019, 7, 9), raw.OUTCOME_END)
    stock_gaps = []
    for row, segment in zip(rows[:2], segments[:2], strict=True):
        first = calendar.index(date.fromisoformat(row["decision_session"])) + 1
        required = set(calendar[first:first + 252])
        retained = set(holding_calendar(date(2024, 4, 30), date.fromisoformat(segment["last_session"])))
        missing = sorted(required - retained)
        stock_gaps.append({"security_id": row["security_id"], "ticker_at_latest_retained_segment": segment["artifact"]["ticker"],
            "provider_symbol_candidate": segment["artifact"]["provider_symbol"], "missing_source_session_count": len(missing),
            "ranges": [{"first_session": str(missing[0]), "last_session": str(missing[-1]), "sessions": len(missing)}]})
    stock_sessions = sum(record["missing_source_session_count"] for record in stock_gaps)
    gap = {"schema": "read_only_investment_raw_tail_assessment", "old_raw_end": "2024-05-28",
        "last_required_end_252": "2025-05-30", "last_decision": "2024-05-28", "no_prices_returns_or_financial_values_read": True,
        "source_pins": {name: file_sha256(tmp_path / name) for name in ("selection.json", "roster.json", "actions.json")},
        "missing_action_query_symbols": [], "eligible_security_months": 2, "eligible_unique_security_ids": 2,
        "benchmark_tickers": ["SPY", "QQQ", "XLK"], "identity_warnings": [{"security_id": "stock:BBB", "ticker": "BBB"}],
        "stock_tail_requirements": [{"security_id": "stock:AAA", "single_retained_identity_interval_covers_tail": True}],
        "all_horizon_metadata_coverage": {"missing_ranges": stock_gaps, "missing_stock_security_sessions_total": stock_sessions},
        "nominal_benchmark_sessions": 3 * 252, "nominal_full_missing_stock_plus_benchmark_sessions": stock_sessions + 3 * 252}
    _write(tmp_path / "gaps.json", gap)
    config = {"schema": "market_predictor.investment_raw_acquisition_request", "gap_inventory": _pin(tmp_path / "gaps.json", tmp_path),
        "source_selection": _pin(tmp_path / "selection.json", tmp_path), "roster": _pin(tmp_path / "roster.json", tmp_path),
        "parent_plan_authority": _pin(parent / "_authority.json", tmp_path), "action_query": _pin(tmp_path / "actions.json", tmp_path),
        "outcome_end": "2025-05-30"}
    _write(tmp_path / "request.json", config)
    return {"root": tmp_path, "request": tmp_path / "request.json", "plan": tmp_path / "plan", "archive": tmp_path / "archive"}


class _Source:
    def __init__(self, *, fail: str | None = None, empty: str | None = None, partial: str | None = None, memory: bool = False) -> None:
        self.fail, self.empty, self.partial, self.memory = fail, empty, partial, memory
        self.calls: list[str] = []
        self.thread_ids: set[int] = set()

    def fetch_daily_page(self, symbol: str, start: datetime, end_exclusive: datetime, *, page_token: str | None,
                         asof: date, adjustment: str) -> archive.SwingDailyPage:
        import threading
        self.thread_ids.add(threading.get_ident())
        self.calls.append(symbol)
        assert adjustment == "raw" and page_token is None
        assert asof == (end_exclusive - timedelta(days=1)).date()
        if self.memory:
            raise MemoryBudgetError("fixture memory stop")
        if symbol == self.fail:
            raise RuntimeError("fixture unavailable provider query")
        days = holding_calendar(start.date(), (end_exclusive - timedelta(days=1)).date())
        bars = () if symbol == self.empty else tuple({"t": pd.Timestamp(day).tz_localize("America/New_York").isoformat(),
            "o": 10, "h": 12, "l": 9, "c": 11, "v": 1000} for day in days)
        if symbol == self.partial:
            bars = bars[:1]
        payload = {"bars": {symbol: list(bars)}, "next_page_token": None}
        params = {"symbols": symbol, "timeframe": "1Day", "start": start.isoformat(),
            "end": (end_exclusive - timedelta(microseconds=1)).isoformat(), "feed": "sip", "limit": 10_000,
            "adjustment": "raw", "sort": "asc", "asof": asof.isoformat()}
        return archive.SwingDailyPage(page_token, None, symbol, "1Day", "sip", "raw", bars, {}, payload,
            _test_transport_response(payload, params))


def _publish(fixture: dict[str, Any]) -> str:
    result = raw.publish_investment_raw_acquisition_plan(fixture["root"], fixture["request"], fixture["plan"])
    return str(result["plan_authority_sha256"])


def _collect(fixture: dict[str, Any], source: _Source, pin: str) -> dict[str, Any]:
    return raw.collect_investment_raw_acquisition(fixture["root"], fixture["request"], fixture["plan"], fixture["archive"],
        source_factory=lambda: source, expected_plan_authority_sha256=pin)


def test_exact_missing_runs_preserve_historical_exit_and_quarantine_membership_covered_identity(acquisition: dict[str, Any]) -> None:
    _, manifest, units = raw.prepare_investment_raw_acquisition(acquisition["root"], acquisition["request"])
    assert len(units) == 5 and set(units.ticker) == {"AAA", "BBB", "SPY", "QQQ", "XLK"}
    assert units.loc[units.ticker.eq("BBB"), "start_date"].item() == "2024-05-28"
    assert units.loc[units.ticker.eq("AAA"), "start_date"].item() == "2024-05-29"
    assert manifest["requirements"]["historical_gap_security_ids"] == ["stock:BBB"]
    assert manifest["requirements"]["tail_identity_caution_ids"] == ["stock:BBB"]
    assert all(item["identity_admission"] == raw.QUARANTINE and not item["ownership_admitted"]
        for item in manifest["unit_requirements"].values() if item["identity"]["role"] == "stock")


@pytest.mark.parametrize("fault", ["end", "source_pin", "omit_gap", "gap_count", "provider_candidate"])
def test_changed_scope_sources_and_exact_gap_inventory_fail_before_publication(acquisition: dict[str, Any], fault: str) -> None:
    config = _read(acquisition["request"])
    gap_path = acquisition["root"] / "gaps.json"
    gap = _read(gap_path)
    if fault == "end":
        config["outcome_end"] = "2025-06-02"
    elif fault == "source_pin":
        (acquisition["root"] / "retained" / "AAA" / "bars.parquet").write_bytes(b"changed")
    else:
        records = gap["all_horizon_metadata_coverage"]["missing_ranges"]
        if fault == "omit_gap":
            records.pop()
        elif fault == "gap_count":
            records[0]["missing_source_session_count"] -= 1
        else:
            records[0]["provider_symbol_candidate"] = "WRONG"
        _write(gap_path, gap)
        config["gap_inventory"] = _pin(gap_path, acquisition["root"])
    _write(acquisition["request"], config)
    with pytest.raises(DataReadinessError):
        _publish(acquisition)
    assert not acquisition["plan"].exists()


def test_independent_plan_replay_rejects_self_resealed_unit_change(acquisition: dict[str, Any]) -> None:
    _publish(acquisition)
    units_path = acquisition["plan"] / DAILY_BAR_UNITS_FILE
    units = pd.read_csv(units_path, dtype=str)
    units.loc[units.ticker.eq("BBB"), "start_date"] = "2024-05-29"
    units.to_csv(units_path, index=False, lineterminator="\n")
    manifest = _read(acquisition["plan"] / "_manifest.json")
    manifest["daily_bars"]["units_artifact"].update(sha256=file_sha256(units_path), bytes=units_path.stat().st_size)
    _write(acquisition["plan"] / "_manifest.json", manifest)
    authority_path = acquisition["plan"] / "_authority.json"
    authority = _read(authority_path)
    authority.update(artifact_sha256=file_sha256(acquisition["plan"] / "_manifest.json"), units_sha256=file_sha256(units_path))
    _write(authority_path, authority)
    with pytest.raises(DataReadinessError, match="independent requirements"):
        archive._load_verified_plan(acquisition["plan"], expected_plan_authority_sha256=file_sha256(authority_path))


def test_serial_transport_nonempty_stocks_stay_quarantined_and_original_bytes_unchanged(acquisition: dict[str, Any]) -> None:
    before = {path: path.read_bytes() for path in (acquisition["root"] / "retained").rglob("*") if path.is_file()}
    pin = _publish(acquisition)
    source = _Source()
    result = _collect(acquisition, source, pin)
    assert result["archive_status"] == "complete" and result["missing_sessions"] == 0
    assert result["actual_source_end"] == "2025-05-30"
    assert result["requested_units"] == result["completed_units"] == len(source.calls) == 5
    assert len(source.thread_ids) == 1
    assert _read(acquisition["archive"] / "_request.json")["workers"] == 1
    assert result["parent_pins"]["source_selection"]["path"] == "selection.json"
    assert result["planned_benchmark_symbols"] == ["QQQ", "SPY", "XLK"]
    assert all(not item["ownership_admitted"] and item["identity_admission"] == raw.QUARANTINE
        for item in result["completed_artifacts"] if item["role"] == "stock")
    assert all(item["identity_admission"] == "captured_only" for item in result["completed_artifacts"] if item["role"] == "benchmark")
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert _collect(acquisition, _Source(fail="SPY"), pin)["captured_sessions"] == result["captured_sessions"]


def test_per_security_failure_retains_unrelated_artifacts_and_resume_does_not_redownload(acquisition: dict[str, Any]) -> None:
    pin = _publish(acquisition)
    failed = _Source(fail="AAA")
    result = _collect(acquisition, failed, pin)
    assert result["archive_status"] == "incomplete" and result["completed_units"] == 4
    assert set(failed.calls) == {"AAA", "BBB", "SPY", "QQQ", "XLK"}
    assert not (acquisition["archive"] / "_authority.json").exists()
    retained = {item["bars_path"]: (acquisition["archive"] / item["bars_path"]).read_bytes() for item in result["completed_artifacts"]}
    resumed = _Source()
    final = _collect(acquisition, resumed, pin)
    assert final["archive_status"] == "complete" and resumed.calls == ["AAA"]
    assert all((acquisition["archive"] / name).read_bytes() == payload for name, payload in retained.items())


@pytest.mark.parametrize("empty", ["AAA", "SPY"])
def test_unavailable_limit_and_mandatory_benchmark_do_not_fabricate_complete_archive(acquisition: dict[str, Any], empty: str) -> None:
    pin = _publish(acquisition)
    source = _Source(empty=empty)
    report = _collect(acquisition, source, pin)
    assert report["archive_status"] == "incomplete" and len(source.calls) == 5
    assert report["missing_sessions"] >= 252
    assert not (acquisition["archive"] / "_authority.json").exists()
    assert report["completed_units"] == (5 if empty == "AAA" else 4)
    assert report["planned_benchmark_symbols"] == ["QQQ", "SPY", "XLK"]


def test_raw_body_tamper_is_rejected_on_replay(acquisition: dict[str, Any]) -> None:
    pin = _publish(acquisition)
    result = _collect(acquisition, _Source(), pin)
    record = result["completed_artifacts"][0]
    unit = _read(acquisition["archive"] / record["unit_manifest_path"])
    (acquisition["archive"] / unit["pages"][0]["transport"]["body_path"]).write_bytes(b"{}")
    with pytest.raises(DataReadinessError):
        raw.verify_investment_raw_acquisition(acquisition["root"], acquisition["request"], acquisition["plan"], acquisition["archive"],
            expected_plan_authority_sha256=pin)


def test_nonempty_partial_benchmark_is_not_finalized_and_resume_retries_it(acquisition: dict[str, Any]) -> None:
    pin = _publish(acquisition)
    source = _Source(partial="SPY")
    report = _collect(acquisition, source, pin)
    assert report["archive_status"] == "incomplete" and report["completed_units"] == 4
    assert len(source.calls) == 5 and not (acquisition["archive"] / "_authority.json").exists()
    failures = report["collection_status"]["failed_units"]
    assert len(failures) == 1 and "missing=251" in failures[0]["error"]
    partial_unit = acquisition["archive"] / "units" / failures[0]["unit_id"]
    assert list(partial_unit.rglob("*.bin")) and list((partial_unit / "attempts").glob("*.json"))
    assert not (partial_unit / "_manifest.json").exists()
    resumed = _Source(partial="SPY")
    assert _collect(acquisition, resumed, pin)["archive_status"] == "incomplete"
    assert resumed.calls == ["SPY"]
    assert len(list(partial_unit.rglob("*.bin"))) == 2
    final_source = _Source()
    final = _collect(acquisition, final_source, pin)
    assert final["archive_status"] == "complete" and final["missing_sessions"] == 0
    assert final_source.calls == ["SPY"]


def test_public_plan_checks_lease_before_loading_missing_inputs(acquisition: dict[str, Any]) -> None:
    with heavy_job_lease("other-job", runtime_dir=acquisition["root"] / "data/runtime"):
        with pytest.raises(HeavyJobBusyError):
            raw.publish_investment_raw_acquisition_plan(acquisition["root"], Path("does-not-exist"), acquisition["plan"])


def test_memory_failure_stops_instead_of_collecting_unrelated_units(acquisition: dict[str, Any]) -> None:
    pin = _publish(acquisition)
    source = _Source(memory=True)
    with pytest.raises(MemoryBudgetError):
        _collect(acquisition, source, pin)
    assert len(source.calls) == 1
    report = raw.verify_investment_raw_acquisition(acquisition["root"], acquisition["request"], acquisition["plan"], acquisition["archive"],
        expected_plan_authority_sha256=pin)
    assert report["archive_status"] == "incomplete" and report["completed_units"] == 0
    assert report["collection_status"] is None and not report["status_receipt_present"]


def test_investment_scope_does_not_change_legacy_workers() -> None:
    assert archive._workers_for_scope(None) == archive.MAXIMUM_WORKERS == 2
    assert archive._workers_for_scope("initial_fit_raw_share_acquisition") == 2
    assert archive._workers_for_scope(raw.SCOPE) == 1


def test_system_memory_pressure_prevents_raw_input_read(acquisition: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def pressure(stage: str) -> None:
        calls.append(stage)
        raise MemoryBudgetError("fixture system RAM pressure")

    def forbidden_read(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("raw acquisition read source inputs before the system-memory guard")

    monkeypatch.setattr(raw, "runtime_memory_guard", pressure)
    monkeypatch.setattr(raw, "_object", forbidden_read)
    with pytest.raises(MemoryBudgetError, match="system RAM pressure"):
        raw.prepare_investment_raw_acquisition(acquisition["root"], acquisition["request"])
    assert calls == ["investment raw acquisition"]


def test_transport_system_memory_guard_prevents_next_provider_page(acquisition: dict[str, Any]) -> None:
    pin = _publish(acquisition)

    class PagedSource(_Source):
        def fetch_daily_page(self, *args: Any, **kwargs: Any) -> archive.SwingDailyPage:
            page = super().fetch_daily_page(*args, **kwargs)
            assert page.transport_response is not None
            payload = {"bars": {page.response_symbol: list(page.bars)}, "next_page_token": "continued"}
            params = dict(parse_qsl(urlsplit(page.transport_response.requested_url).query))
            return replace(page, next_page_token="continued", raw_payload=payload,
                transport_response=_test_transport_response(payload, params))

    source = PagedSource()
    calls = []

    def pressure_before_page() -> None:
        calls.append("guard")
        if len(calls) == 3:
            raise MemoryBudgetError("fixture system RAM pressure before page")

    with pytest.raises(MemoryBudgetError, match="before page"):
        archive.collect_swing_history_plan(plan_directory=acquisition["plan"], output_directory=acquisition["archive"],
            source_factory=lambda: source, provider_symbol_for=lambda symbol: symbol,
            expected_plan_authority_sha256=pin, memory_guard=pressure_before_page)
    # Guard 1 precedes plan loading, guard 2 permits page one, and guard 3
    # refuses page two after retaining the first original response/failed attempt.
    assert len(calls) == 3 and len(source.calls) == 1
    assert list(acquisition["archive"].glob("units/*/raw/*/*.bin"))
    assert not (acquisition["archive"] / "_authority.json").exists()


def test_investment_transport_cannot_omit_system_memory_guard(acquisition: dict[str, Any]) -> None:
    pin = _publish(acquisition)
    source = _Source()
    with pytest.raises(DataReadinessError, match="requires its system-memory guard"):
        archive.collect_swing_history_plan(plan_directory=acquisition["plan"], output_directory=acquisition["archive"],
            source_factory=lambda: source, provider_symbol_for=lambda symbol: symbol, expected_plan_authority_sha256=pin)
    assert source.calls == [] and not acquisition["archive"].exists()
