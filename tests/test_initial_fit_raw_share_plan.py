"""Synthetic, outcome-blind acquisition-plan contract and immutable replay tests."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.store import file_sha256
from market_predictor.commands import swing_collection as collection_commands
from market_predictor.commands import swing_research as research_commands
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild import swing_history_collection as history_collection
from market_predictor.edge_rebuild.swing_history_collection import (
    SwingDailyPage,
    collect_swing_history_plan,
    load_complete_swing_history_collection,
)
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError, heavy_job_lease
from market_predictor.swing.datasets import initial_fit_raw_share_plan as planner
from market_predictor.swing.datasets.history_plan_publication import AUTHORITY_SCHEMA, DAILY_BAR_UNITS_FILE, PLAN_SCHEMA
from market_predictor.swing.datasets.holding_observation_requirements import IDENTITY_COLUMNS, MEMBERSHIP_COLUMNS
from tests.test_swing_history_collection import _test_transport_response
from tests.test_swing_holding_identity_preflight import _bind, _run, _write_partition
from tests.test_swing_holding_identity_preflight import inventory as inventory

_SESSIONS = (
    "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09",
    "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-16", "2024-01-17", "2024-01-18",
)


def _write_request(fixture: dict[str, Any]) -> None:
    fixture["plan_config"].write_text(
        "\n".join(f"{key} = {json.dumps(value)}" for key, value in fixture["plan_request"].items()),
        encoding="utf-8",
    )


@pytest.fixture
def plan_fixture(inventory: dict[str, Any]) -> dict[str, Any]:
    # Two overlapping mature decisions cross the MLK holiday. s001 leaves the index,
    # s002 changes sector, and retained s018 has only held-out decision identities.
    inventory["request"]["initial_fit_end"] = "2024-01-19"
    members = inventory["memberships"].copy()
    members.loc[members.security_id.eq("s001"), "effective_to_utc"] = pd.Timestamp("2024-01-04T00:00:00Z")
    members.loc[members.security_id.eq("s018"), "effective_from_utc"] = pd.Timestamp("2024-02-01T00:00:00Z")
    changed = members.loc[members.security_id.eq("s002")].copy()
    members.loc[members.security_id.eq("s002"), "effective_to_utc"] = pd.Timestamp("2024-01-03T00:00:00Z")
    changed["effective_from_utc"] = pd.Timestamp("2024-01-03T00:00:00Z")
    changed["primary_benchmark"] = "XLF"
    changed["sector"] = "financials"
    members = pd.concat([members, changed], ignore_index=True)
    inventory["memberships"] = members
    members.to_parquet(inventory["root"] / "membership.parquet", index=False)
    inventory["combined"]["membership_authority"] = {
        "membership_artifact_sha256": file_sha256(inventory["root"] / "membership.parquet"),
        "universe_sha256": json_sha256(inventory["cohort_values"]["original_security_ids"]),
        "parent_lineage": {"synthetic_fixture": True},
    }
    inventory["cohort_values"]["combined_daily_inputs_sha256"] = json_sha256(inventory["combined"])
    for index, original in enumerate(inventory["frames"]):
        frame = original.loc[
            ~(original.security_id.eq("s001") & original.session_date_et.gt("2024-01-03"))
            & ~(original.security_id.eq("s018") & original.session_date_et.lt("2024-02-01"))
        ].copy().reset_index(drop=True)
        selected = frame.security_id.eq("s002") & frame.session_date_et.ge("2024-01-03")
        frame.loc[selected, "primary_benchmark"] = "XLF"
        frame.loc[selected, "sector"] = "financials"
        _write_partition(inventory, index, frame)
    _bind(inventory)
    preflight = _run(inventory)
    assert preflight["initial_fit"]["covered_matured"] == 32
    assert preflight["initial_fit"]["uncovered_matured"] == 2
    inventory["preflight"] = preflight
    inventory["plan_config"] = inventory["root"] / "raw-plan.toml"
    inventory["plan_output"] = inventory["root"] / "raw-plan"
    inventory["plan_request"] = {
        "schema": "market_predictor.swing_initial_fit_raw_share_plan_request",
        "preflight_path": inventory["output"].relative_to(inventory["root"]).as_posix(),
        "preflight_sha256": preflight["audit_sha256"],
        "parent_request_path": "parent.json",
    }
    _write_request(inventory)
    return inventory


def _plan(fixture: dict[str, Any], *, expected_plan_sha256: str | None = None) -> dict[str, Any]:
    return planner.run_initial_fit_raw_share_plan(
        fixture["root"], fixture["plan_config"], fixture["plan_output"], expected_plan_sha256=expected_plan_sha256,
    )


def _units(fixture: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(fixture["plan_output"] / DAILY_BAR_UNITS_FILE, dtype=str)


def _snapshot(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(directory).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*") if path.is_file()
    }


def test_exact_decision_plus_ten_xnys_sessions_keep_post_removal_tail(plan_fixture: dict[str, Any]) -> None:
    manifest = _plan(plan_fixture)
    units = _units(plan_fixture)
    assert list(units.columns) == ["security_id", "ticker", "start_date", "end_date", "role"]
    stocks = units.loc[units.role.eq("stock")]
    assert set(stocks.security_id) == {f"s{index:03}" for index in range(1, 18)}
    assert len(stocks) == 17
    for unit in stocks.itertuples(index=False):
        actual = tuple(day for day in _SESSIONS if unit.start_date <= day <= unit.end_date)
        assert actual == _SESSIONS
        for decision_index in (0, 1):
            assert set(_SESSIONS[decision_index:decision_index + 11]).issubset(actual)
        assert unit.start_date == "2024-01-02"
        assert unit.end_date == "2024-01-18"
    removed = stocks.loc[stocks.security_id.eq("s001")].iloc[0]
    assert removed.end_date > "2024-01-04"
    assert manifest["schema"] == PLAN_SCHEMA
    assert manifest["status"] == "ready_for_daily_history_collection"
    assert manifest["daily_bars"]["timeframe"] == "1Day"
    assert manifest["daily_bars"]["price_feed"] == "sip"
    assert manifest["daily_bars"]["adjustment"] == "raw"
    assert manifest["daily_bars"]["stock_units"] == 17


def test_market_and_pit_sector_benchmarks_cover_entire_initial_fit_window(plan_fixture: dict[str, Any]) -> None:
    manifest = _plan(plan_fixture)
    benchmarks = _units(plan_fixture).loc[lambda frame: frame.role.eq("benchmark")]
    assert set(benchmarks.ticker) == {"SPY", "QQQ", "XLK", "XLF"}
    assert len(benchmarks) == 4
    assert set(benchmarks.security_id) == {f"benchmark:{ticker}" for ticker in benchmarks.ticker}
    assert benchmarks.start_date.eq("2024-01-02").all()
    assert benchmarks.end_date.eq("2024-01-19").all()
    assert manifest["daily_bars"]["benchmark_units"] == 4
    assert manifest["daily_bars"]["planned_units"] == 21


def test_all_retained_ids_and_distinct_session_counts_are_preserved(plan_fixture: dict[str, Any]) -> None:
    manifest = _plan(plan_fixture)
    request = json.loads((plan_fixture["plan_output"] / "_request.json").read_text(encoding="utf-8"))
    assert request["retained_security_ids"] == [f"s{index:03}" for index in range(1, 19)]
    assert request["excluded_security_ids"] == ["s000", "s019"]
    counts = manifest["requirements"]
    assert counts["retained_securities"] == 18
    assert counts["in_window_securities"] == 17
    assert counts["zero_requirement_security_ids"] == ["s018"]
    assert counts["in_window_decisions"] == counts["mature_decisions"] == counts["decision_sessions"] == 34
    assert counts["holding_sessions"] == 17 * 11 == 187
    assert counts["decision_and_holding_sessions"] == 17 * 12 == 204
    assert counts["decision_only_sessions"] == 17
    assert counts["decision_and_holding_sessions"] == counts["holding_sessions"] + counts["decision_only_sessions"]
    assert [record["month"] for record in counts["partitions"]] == ["2024-01"]
    for key in ("outcomes_read", "ownership_admitted", "bar_coverage_verified", "accounting_eligible", "promotion_eligible"):
        assert manifest[key] is False
    assert request["decision_start"] == "2024-01-02"
    assert request["initial_fit_end"] == "2024-01-19"
    assert request["horizon_sessions"] == 10
    assert request["cohort_sha256"] == plan_fixture["preflight"]["cohort_sha256"]


def test_immature_cutoff_decision_does_not_create_partial_or_future_holding_path(plan_fixture: dict[str, Any]) -> None:
    frame = plan_fixture["frames"][0]
    extra = frame.loc[frame.security_id.eq("s003") & frame.session_date_et.eq("2024-01-03")].copy()
    extra["session_date_et"] = "2024-01-19"
    extra["decision_id"] = "s003:2024-01-19"
    extra["decision_time_utc"] = swing_prediction_cutoffs(extra.session_date_et)
    _write_partition(plan_fixture, 0, pd.concat([frame, extra], ignore_index=True))
    _bind(plan_fixture)
    plan_fixture["output"].unlink()
    preflight = _run(plan_fixture)
    plan_fixture["plan_request"]["preflight_sha256"] = preflight["audit_sha256"]
    _write_request(plan_fixture)
    manifest = _plan(plan_fixture)
    counts = manifest["requirements"]
    assert counts["mature_decisions"] == 34
    assert counts["in_window_decisions"] == counts["decision_sessions"] == 35
    assert counts["holding_sessions"] == 187
    assert counts["decision_and_holding_sessions"] == 205
    units = _units(plan_fixture)
    stock = units.loc[units.security_id.eq("s003")]
    assert len(stock) == 1
    assert stock.iloc[0].end_date == "2024-01-19"
    assert units.end_date.le("2024-01-19").all()


def test_publication_uses_existing_collector_artifacts_and_hashes(plan_fixture: dict[str, Any]) -> None:
    manifest = _plan(plan_fixture)
    output = plan_fixture["plan_output"]
    assert json.loads((output / "_manifest.json").read_text(encoding="utf-8")) == {
        key: value for key, value in manifest.items() if key not in {"plan_sha256", "provider_symbols"}
    }
    authority = json.loads((output / "_authority.json").read_text(encoding="utf-8"))
    assert authority["schema"] == AUTHORITY_SCHEMA
    assert authority["state"] == "complete"
    assert authority["artifact"] == "_manifest.json"
    assert authority["artifact_sha256"] == file_sha256(output / "_manifest.json")
    assert authority["request_sha256"] == file_sha256(output / "_request.json") == manifest["request_sha256"]
    assert authority["units_sha256"] == file_sha256(output / DAILY_BAR_UNITS_FILE)
    assert manifest["daily_bars"]["units_artifact"]["sha256"] == authority["units_sha256"]
    assert manifest["plan_sha256"] == file_sha256(output / "_authority.json")


def test_replay_requires_independent_authority_pin_and_preserves_all_bytes(plan_fixture: dict[str, Any]) -> None:
    source_paths = [plan_fixture["root"] / name for name in plan_fixture["preflight"]["source_files"]]
    source_hashes = {path: file_sha256(path) for path in source_paths}
    first = _plan(plan_fixture)
    output = plan_fixture["plan_output"]
    before = _snapshot(output)
    pin = file_sha256(output / "_authority.json")
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture)
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture, expected_plan_sha256="0" * 64)
    assert _plan(plan_fixture, expected_plan_sha256=pin) == first
    assert _snapshot(output) == before
    assert {path: file_sha256(path) for path in source_paths} == source_hashes


@pytest.mark.parametrize("artifact", ["_authority.json", "_request.json", "_manifest.json", DAILY_BAR_UNITS_FILE])
def test_published_artifact_tamper_cannot_be_repaired_by_replay(plan_fixture: dict[str, Any], artifact: str) -> None:
    _plan(plan_fixture)
    output = plan_fixture["plan_output"]
    pin = file_sha256(output / "_authority.json")
    target = output / artifact
    target.write_bytes(target.read_bytes() + b" ")
    tampered = _snapshot(output)
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture, expected_plan_sha256=pin)
    assert _snapshot(output) == tampered


@pytest.mark.parametrize("relative", [
    "preflight.toml", "cohort.json", "parent.json", "membership.parquet",
    "panel/manifest.json", "panel/january.parquet", "panel/february.parquet",
])
def test_every_frozen_source_is_bound_even_undecoded_heldout_partition(plan_fixture: dict[str, Any], relative: str) -> None:
    target = plan_fixture["root"] / relative
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture)
    assert not plan_fixture["plan_output"].exists()


def test_canonical_preflight_pin_accepts_json_whitespace_before_publication(plan_fixture: dict[str, Any]) -> None:
    report = json.loads(plan_fixture["output"].read_text(encoding="utf-8"))
    plan_fixture["output"].write_text(json.dumps(report, indent=3, sort_keys=True), encoding="utf-8")
    assert _plan(plan_fixture)["daily_bars"]["adjustment"] == "raw"


def test_preflight_pin_is_canonical_audit_not_json_file_hash(plan_fixture: dict[str, Any]) -> None:
    request = plan_fixture["plan_request"]
    assert request["preflight_sha256"] != file_sha256(plan_fixture["output"])
    request["preflight_sha256"] = file_sha256(plan_fixture["output"])
    _write_request(plan_fixture)
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture)
    assert not plan_fixture["plan_output"].exists()


def test_resigned_preflight_cannot_replace_the_frozen_pin(plan_fixture: dict[str, Any]) -> None:
    report = json.loads(plan_fixture["output"].read_text(encoding="utf-8"))
    report["request"]["initial_fit_end"] = "2024-01-22"
    report["audit_sha256"] = json_sha256({key: value for key, value in report.items() if key != "audit_sha256"})
    plan_fixture["output"].write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture)
    assert not plan_fixture["plan_output"].exists()


def test_mutated_preflight_contents_fail_even_with_unchanged_embedded_audit(plan_fixture: dict[str, Any]) -> None:
    report = json.loads(plan_fixture["output"].read_text(encoding="utf-8"))
    report["initial_fit"]["covered_matured"] += 1
    plan_fixture["output"].write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture)
    assert not plan_fixture["plan_output"].exists()


def test_replay_rechecks_sources_without_overwriting_existing_plan(plan_fixture: dict[str, Any]) -> None:
    _plan(plan_fixture)
    output = plan_fixture["plan_output"]
    before = _snapshot(output)
    pin = file_sha256(output / "_authority.json")
    target = plan_fixture["root"] / "panel/february.parquet"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture, expected_plan_sha256=pin)
    assert _snapshot(output) == before


def test_unrelated_existing_output_directory_is_not_overwritten(plan_fixture: dict[str, Any]) -> None:
    output = plan_fixture["plan_output"]
    output.mkdir()
    (output / "unrelated.txt").write_text("existing evidence", encoding="utf-8")
    before = _snapshot(output)
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture)
    assert _snapshot(output) == before


def test_parent_request_must_be_the_cohort_bound_source_not_an_identical_copy(plan_fixture: dict[str, Any]) -> None:
    root = plan_fixture["root"]
    (root / "alternate-parent.json").write_bytes((root / "parent.json").read_bytes())
    plan_fixture["plan_request"]["parent_request_path"] = "alternate-parent.json"
    _write_request(plan_fixture)
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture)
    assert not plan_fixture["plan_output"].exists()


@pytest.mark.parametrize("key,value", [
    ("decision_start", "2024-01-03"), ("initial_fit_end", "2024-01-22"), ("horizon_sessions", 9),
    ("excluded_security_ids", ["s001"]), ("adjustment", "all"), ("cohort_path", "cohort.json"),
])
def test_request_cannot_override_preflight_dates_cohort_horizon_or_raw_basis(
    plan_fixture: dict[str, Any], key: str, value: object,
) -> None:
    plan_fixture["plan_request"][key] = value
    _write_request(plan_fixture)
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture)
    assert not plan_fixture["plan_output"].exists()


@pytest.mark.parametrize("key", ["schema", "preflight_path", "preflight_sha256", "parent_request_path"])
def test_request_requires_every_declared_field(plan_fixture: dict[str, Any], key: str) -> None:
    del plan_fixture["plan_request"][key]
    _write_request(plan_fixture)
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture)
    assert not plan_fixture["plan_output"].exists()


@pytest.mark.parametrize("pin", ["", "0" * 63, "G" * 64, "0" * 64])
def test_invalid_or_wrong_preflight_pin_fails_closed(plan_fixture: dict[str, Any], pin: str) -> None:
    plan_fixture["plan_request"]["preflight_sha256"] = pin
    _write_request(plan_fixture)
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture)
    assert not plan_fixture["plan_output"].exists()


@pytest.mark.parametrize("field", ["preflight_path", "parent_request_path", "plan_config", "plan_output", "runtime"])
def test_all_paths_stay_inside_repository(
    plan_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    outside = plan_fixture["root"].parent / "outside-plan-authority"
    if field in plan_fixture["plan_request"]:
        plan_fixture["plan_request"][field] = "../outside-plan-authority"
        _write_request(plan_fixture)
    elif field == "runtime":
        monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(outside))
    else:
        plan_fixture[field] = outside
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture)
    assert not outside.exists()


def test_busy_lease_precedes_even_missing_config_or_preflight_reads(plan_fixture: dict[str, Any]) -> None:
    plan_fixture["plan_config"] = plan_fixture["root"] / "does-not-exist.toml"
    plan_fixture["output"].unlink()
    with heavy_job_lease("synthetic-other-job", runtime_dir=plan_fixture["root"] / "runtime"):
        with pytest.raises(HeavyJobBusyError):
            _plan(plan_fixture)
    assert not plan_fixture["plan_output"].exists()
    assert not (plan_fixture["root"] / "runtime" / "heavy-job.owner.json").exists()


def test_no_numeric_columns_or_heldout_partition_are_decoded(
    plan_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = pq.ParquetFile
    calls: list[tuple[str, tuple[str, ...]]] = []

    class ProjectedFile:
        def __init__(self, path: Path, *args: Any, **kwargs: Any) -> None:
            self.path = Path(path)
            assert self.path.name != "february.parquet", "held-out partition was opened for decoding"
            self.source = original(path, *args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.source, name)

        def __enter__(self) -> ProjectedFile:
            return self

        def __exit__(self, *args: Any) -> None:
            self.source.close()

        def read(self, *, columns: list[str], use_threads: bool) -> Any:
            expected = MEMBERSHIP_COLUMNS if self.path.name == "membership.parquet" else IDENTITY_COLUMNS
            assert columns and set(columns).issubset(expected)
            assert use_threads is False
            calls.append((self.path.name, tuple(columns)))
            return self.source.read(columns=columns, use_threads=use_threads)

    monkeypatch.setattr(pq, "ParquetFile", ProjectedFile)
    _plan(plan_fixture)
    assert {name for name, _ in calls} == {"membership.parquet", "january.parquet"}


def test_source_changing_after_projection_prevents_publication(
    plan_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = pq.ParquetFile.read
    changed = False

    def changing_read(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal changed
        result = original(self, *args, **kwargs)
        if not changed and "decision_id" in kwargs.get("columns", []):
            target = plan_fixture["root"] / "parent.json"
            target.write_bytes(target.read_bytes() + b" ")
            changed = True
        return result

    monkeypatch.setattr(pq.ParquetFile, "read", changing_read)
    with pytest.raises(DataReadinessError):
        _plan(plan_fixture)
    assert changed
    assert not plan_fixture["plan_output"].exists()
    assert not list(plan_fixture["root"].glob(".raw-plan.*.tmp"))


class _RawPlanSource:
    """Exact synthetic January sessions with original-response test receipts."""

    def __init__(self, runtime: Path) -> None:
        self.runtime = runtime
        self.calls: list[dict[str, Any]] = []

    def fetch_daily_page(
        self, symbol: str, start: datetime, end_exclusive: datetime, *,
        page_token: str | None, asof: date, adjustment: str,
    ) -> SwingDailyPage:
        assert (self.runtime / "heavy-job.owner.json").exists()
        assert page_token is None
        assert adjustment == "raw"
        self.calls.append({"symbol": symbol, "start": start, "end_exclusive": end_exclusive, "asof": asof})
        bars = tuple(
            {"t": f"{day}T05:00:00Z", "o": 10.0, "h": 12.0, "l": 9.0, "c": 11.0, "v": 1000}
            for day in (*_SESSIONS, "2024-01-19")
            if start <= datetime.fromisoformat(f"{day}T05:00:00+00:00") < end_exclusive
        )
        payload = {"bars": {symbol: list(bars)}, "next_page_token": None}
        params = {
            "symbols": symbol, "timeframe": "1Day", "start": start.isoformat(),
            "end": (end_exclusive - timedelta(microseconds=1)).isoformat(),
            "feed": "sip", "limit": 10_000, "adjustment": adjustment, "sort": "asc", "asof": asof.isoformat(),
        }
        transport = replace(
            _test_transport_response(payload, params), retrieved_at_utc=datetime(2024, 1, 20, 12, tzinfo=UTC),
        )
        return SwingDailyPage(
            request_page_token=None, next_page_token=None, response_symbol=symbol, response_timeframe="1Day",
            response_feed="sip", response_adjustment=adjustment, bars=bars, response_headers={},
            raw_payload=payload, transport_response=transport,
        )


@pytest.fixture
def collected_plan(plan_fixture: dict[str, Any]) -> dict[str, Any]:
    plan = _plan(plan_fixture)
    runtime = plan_fixture["root"] / "runtime"
    source = _RawPlanSource(runtime)
    output = plan_fixture["root"] / "raw-collection"
    with heavy_job_lease("synthetic-raw-plan-collection", runtime_dir=runtime):
        manifest = collect_swing_history_plan(
            plan_directory=plan_fixture["plan_output"], output_directory=output, source_factory=lambda: source,
            provider_symbol_for=lambda ticker: plan["provider_symbols"][ticker],
            expected_plan_authority_sha256=plan["plan_sha256"],
        )
    assert manifest["status"] == "complete", manifest
    return {**plan_fixture, "collection_output": output, "collection_manifest": manifest,
        "collection_source": source, "plan_pin": plan["plan_sha256"]}


def test_collector_consumes_published_raw_plan_and_replays_with_authority_file_pin(collected_plan: dict[str, Any]) -> None:
    fixture = collected_plan
    manifest = fixture["collection_manifest"]
    assert manifest["requested_units"] == manifest["observed_units"] == 21
    assert manifest["total_rows"] == 256
    assert manifest["failed_units"] == manifest["unattempted_units"] == manifest["unavailable_units"] == []
    source = fixture["collection_source"]
    assert len(source.calls) == 21
    expected = {
        (row.ticker, row.start_date, row.end_date) for row in _units(fixture).itertuples(index=False)
    }
    assert {
        (call["symbol"], call["start"].date().isoformat(), call["asof"].isoformat()) for call in source.calls
    } == expected
    assert all(call["end_exclusive"].date() == call["asof"] + timedelta(days=1) for call in source.calls)
    output = fixture["collection_output"]
    before = _snapshot(output)
    with heavy_job_lease("synthetic-raw-plan-replay", runtime_dir=fixture["root"] / "runtime"):
        replay = load_complete_swing_history_collection(
            output, plan_directory=fixture["plan_output"], expected_adjustment="raw",
            expected_plan_authority_sha256=fixture["plan_pin"],
        )
    assert replay == manifest
    assert _snapshot(output) == before
    authority = json.loads((output / "_authority.json").read_text(encoding="utf-8"))
    assert authority["plan_authority_sha256"] == fixture["plan_pin"] == file_sha256(fixture["plan_output"] / "_authority.json")


@pytest.mark.parametrize("operation", ["collect", "replay"])
@pytest.mark.parametrize("pin_kind", ["missing", "wrong", "manifest_file", "canonical_authority"])
def test_collector_pin_rejection_precedes_source_factory_or_collection_io(
    plan_fixture: dict[str, Any], operation: str, pin_kind: str,
) -> None:
    _plan(plan_fixture)
    plan_path = plan_fixture["plan_output"]
    output = plan_fixture["root"] / "must-not-be-created"
    authority = json.loads((plan_path / "_authority.json").read_text(encoding="utf-8"))
    pin = {
        "missing": None, "wrong": "0" * 64, "manifest_file": file_sha256(plan_path / "_manifest.json"),
        "canonical_authority": json_sha256(authority),
    }[pin_kind]
    assert pin != file_sha256(plan_path / "_authority.json")
    before = _snapshot(plan_fixture["root"])

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid independent plan pin reached provider construction or mapping")

    with heavy_job_lease("synthetic-invalid-plan-pin", runtime_dir=plan_fixture["root"] / "runtime"):
        with pytest.raises(DataReadinessError, match="authority pin"):
            if operation == "collect":
                collect_swing_history_plan(
                    plan_directory=plan_path, output_directory=output, source_factory=forbidden,
                    provider_symbol_for=forbidden, expected_plan_authority_sha256=pin,
                )
            else:
                load_complete_swing_history_collection(
                    output, plan_directory=plan_path, expected_adjustment="raw", expected_plan_authority_sha256=pin,
                )
    assert not output.exists()
    assert _snapshot(plan_fixture["root"]) == before


def test_collector_mapping_mismatch_rejects_before_source_factory_or_output_writes(plan_fixture: dict[str, Any]) -> None:
    plan = _plan(plan_fixture)
    output = plan_fixture["root"] / "must-not-be-created"
    before = _snapshot(plan_fixture["root"])

    def forbidden() -> Any:
        pytest.fail("provider mapping mismatch reached source construction")

    with heavy_job_lease("synthetic-provider-mapping-mismatch", runtime_dir=plan_fixture["root"] / "runtime"):
        with pytest.raises(DataReadinessError, match="provider symbols differ"):
            collect_swing_history_plan(
                plan_directory=plan_fixture["plan_output"], output_directory=output, source_factory=forbidden,
                provider_symbol_for=lambda ticker: "WRONG" if ticker == "T001" else plan["provider_symbols"][ticker],
                expected_plan_authority_sha256=plan["plan_sha256"],
            )
    assert not output.exists()
    assert _snapshot(plan_fixture["root"]) == before


def test_collector_replay_rejects_changed_provider_mapping_without_repairing_archive(collected_plan: dict[str, Any]) -> None:
    output = collected_plan["collection_output"]
    request_path = output / "_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["provider_symbols"]["T001"] = "WRONG"
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    before = _snapshot(output)
    with heavy_job_lease("synthetic-replay-mapping-mismatch", runtime_dir=collected_plan["root"] / "runtime"):
        with pytest.raises(DataReadinessError, match="provider symbols differ"):
            load_complete_swing_history_collection(
                output, plan_directory=collected_plan["plan_output"], expected_adjustment="raw",
                expected_plan_authority_sha256=collected_plan["plan_pin"],
            )
    assert _snapshot(output) == before


def _invoke_raw_cli(fixture: dict[str, Any], *, collect: bool, extra: list[str]) -> Any:
    from market_predictor.collection_cli import app as collection_app
    from market_predictor.research_cli import app as research_app

    common = ["--root", str(fixture["root"]), "--config", str(fixture["plan_config"])]
    if collect:
        args = ["collect-swing-initial-fit-raw-prices", *common,
            "--plan-dir", str(fixture["plan_output"]), "--out-dir", str(fixture["root"] / "raw-collection"), *extra]
    else:
        args = ["plan-swing-initial-fit-raw-prices", *common, "--output-directory", str(fixture["plan_output"]), *extra]
    return CliRunner().invoke(collection_app if collect else research_app, args)


def _forbid_provider_access(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("offline or rejected CLI invocation attempted credentials or provider construction")

    for module in (collection_commands, research_commands):
        monkeypatch.setattr(module, "get_settings", forbidden)
        monkeypatch.setattr(module, "AlpacaSource", forbidden)


@pytest.mark.parametrize("collect", [False, True])
def test_raw_cli_busy_lease_precedes_input_or_provider_access(
    plan_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch, collect: bool,
) -> None:
    _forbid_provider_access(monkeypatch)
    plan_fixture["plan_config"] = plan_fixture["root"] / "missing-request.toml"
    extra = ["--expected-plan-sha256", "0" * 64] if collect else []
    with heavy_job_lease("synthetic-other-cli-job", runtime_dir=plan_fixture["root"] / "runtime"):
        result = _invoke_raw_cli(plan_fixture, collect=collect, extra=extra)
    assert result.exit_code == HEAVY_JOB_BUSY_EXIT_CODE, result.output
    assert not plan_fixture["plan_output"].exists()
    assert not (plan_fixture["root"] / "raw-collection").exists()


@pytest.mark.parametrize("collect", [False, True])
def test_raw_cli_wrong_pin_rejects_before_settings_or_output_changes(
    plan_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch, collect: bool,
) -> None:
    _plan(plan_fixture)
    _forbid_provider_access(monkeypatch)
    before = _snapshot(plan_fixture["root"])
    result = _invoke_raw_cli(plan_fixture, collect=collect, extra=["--expected-plan-sha256", "0" * 64])
    assert result.exit_code != 0
    assert isinstance(result.exception, DataReadinessError), (result.output, result.exception)
    assert "pin" in str(result.exception)
    assert _snapshot(plan_fixture["root"]) == before


def test_raw_collection_cli_requires_explicit_pin_before_provider_access(
    plan_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_provider_access(monkeypatch)
    result = _invoke_raw_cli(plan_fixture, collect=True, extra=[])
    assert result.exit_code == 2, result.output
    assert "--expected-plan-sha256" in result.output
    assert not plan_fixture["plan_output"].exists()
    assert not (plan_fixture["root"] / "raw-collection").exists()


def test_raw_plan_cli_publishes_and_replays_without_provider_access(
    plan_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_provider_access(monkeypatch)
    result = _invoke_raw_cli(plan_fixture, collect=False, extra=[])
    assert result.exit_code == 0, (result.output, result.exception)
    before = _snapshot(plan_fixture["plan_output"])
    pin = file_sha256(plan_fixture["plan_output"] / "_authority.json")
    replay = _invoke_raw_cli(plan_fixture, collect=False, extra=["--expected-plan-sha256", pin])
    assert replay.exit_code == 0, (replay.output, replay.exception)
    assert _snapshot(plan_fixture["plan_output"]) == before


def test_raw_collection_cli_offline_replays_under_lease_without_credentials_or_network(
    collected_plan: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_provider_access(monkeypatch)
    runtime = collected_plan["root"] / "runtime"
    loader = collection_commands.load_complete_swing_history_collection
    calls = 0

    def leased_loader(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        assert kwargs["expected_adjustment"] == "raw"
        assert kwargs["expected_plan_authority_sha256"] == collected_plan["plan_pin"]
        with pytest.raises(HeavyJobBusyError):
            with heavy_job_lease("synthetic-nested-lease-probe", runtime_dir=runtime):
                pytest.fail("offline collection replay lost its workspace lease")
        return loader(*args, **kwargs)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("offline CLI dispatched online collection")

    monkeypatch.setattr(collection_commands, "load_complete_swing_history_collection", leased_loader)
    monkeypatch.setattr(collection_commands, "collect_swing_history_plan", forbidden)
    before = _snapshot(collected_plan["root"])
    result = _invoke_raw_cli(
        collected_plan, collect=True, extra=["--offline", "--expected-plan-sha256", collected_plan["plan_pin"]],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == 1
    assert _snapshot(collected_plan["root"]) == before


def _resign_synthetic_plan_metadata(output: Path) -> None:
    """Rehash local test artifacts without changing the independently saved pin."""
    manifest_path = output / "_manifest.json"
    authority_path = output / "_authority.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["request_sha256"] = file_sha256(output / "_request.json")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["request_sha256"] = manifest["request_sha256"]
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority_path.write_text(json.dumps(authority, sort_keys=True, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize("replacement_scope", [None, "generic_history_acquisition"])
def test_request_only_scope_downgrade_cannot_bypass_collector_pin(
    plan_fixture: dict[str, Any], replacement_scope: str | None,
) -> None:
    _plan(plan_fixture)
    plan_path = plan_fixture["plan_output"]
    request_path = plan_path / "_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if replacement_scope is None:
        del request["scope"]
    else:
        request["scope"] = replacement_scope
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    _resign_synthetic_plan_metadata(plan_path)
    manifest = json.loads((plan_path / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scope"] == "initial_fit_raw_share_acquisition"
    before = _snapshot(plan_fixture["root"])
    output = plan_fixture["root"] / "forbidden-downgraded-collection"

    def forbidden() -> Any:
        pytest.fail("request-only scope downgrade reached source construction without an external pin")

    with heavy_job_lease("synthetic-scope-downgrade", runtime_dir=plan_fixture["root"] / "runtime"):
        with pytest.raises(DataReadinessError):
            collect_swing_history_plan(
                plan_directory=plan_path, output_directory=output, source_factory=forbidden,
                provider_symbol_for=lambda ticker: ticker,
            )
    assert not output.exists()
    assert _snapshot(plan_fixture["root"]) == before


@pytest.mark.parametrize("race_point", ["first_authority_hash", "authority_snapshot", "provider_mapping"])
def test_collector_rejects_resigned_artifacts_after_first_authority_hash_before_dispatch(
    plan_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch, race_point: str,
) -> None:
    plan = _plan(plan_fixture)
    output = plan_fixture["root"] / "forbidden-raced-collection"
    plan_path = plan_fixture["plan_output"]
    authority_path = plan_path / "_authority.json"
    original_hash = history_collection.file_sha256
    original_parse = history_collection.parse_strict_json_object
    changed = False
    raced: dict[str, tuple[bytes, int]] = {}

    def replace_artifacts() -> None:
        nonlocal changed, raced
        if changed:
            return
        changed = True
        manifest_path = plan_path / "_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["resources"] = {"synthetic_race_replacement": True}
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        _resign_synthetic_plan_metadata(plan_path)
        raced = _snapshot(plan_path)

    def swapping_hash(path: Path) -> str:
        digest = original_hash(path)
        if Path(path).resolve() == authority_path.resolve() and not changed:
            assert digest == plan["plan_sha256"]
            replace_artifacts()
        return digest

    def swapping_parse(content: bytes, *, label: str) -> dict[str, Any]:
        result = original_parse(content, label=label)
        if Path(label).resolve() == authority_path.resolve():
            replace_artifacts()
        return result

    def provider_mapping(ticker: str) -> str:
        if race_point == "provider_mapping":
            replace_artifacts()
        return str(plan["provider_symbols"][ticker])

    def forbidden() -> Any:
        pytest.fail("artifact replacement after the independent pin check reached source dispatch")

    if race_point == "first_authority_hash":
        monkeypatch.setattr(history_collection, "file_sha256", swapping_hash)
    elif race_point == "authority_snapshot":
        monkeypatch.setattr(history_collection, "parse_strict_json_object", swapping_parse)
    with heavy_job_lease("synthetic-collector-authority-race", runtime_dir=plan_fixture["root"] / "runtime"):
        with pytest.raises(DataReadinessError):
            collect_swing_history_plan(
                plan_directory=plan_path, output_directory=output, source_factory=forbidden,
                provider_symbol_for=provider_mapping,
                expected_plan_authority_sha256=plan["plan_sha256"],
            )
    assert changed
    assert file_sha256(authority_path) != plan["plan_sha256"]
    assert not output.exists()
    assert _snapshot(plan_path) == raced


def test_replay_context_rejects_authority_change_during_reconstruction_before_yield(
    plan_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(plan_fixture)
    authority_path = plan_fixture["plan_output"] / "_authority.json"
    reconstruct = planner._reconstruct
    changed = False
    yielded = False
    raced: dict[str, tuple[bytes, int]] = {}

    def changing_reconstruct(*args: Any, **kwargs: Any) -> Any:
        nonlocal changed, raced
        result = reconstruct(*args, **kwargs)
        # Parsed authority and every internal hash remain valid; its independent
        # file identity nevertheless changes while reconstruction holds the lease.
        authority_path.write_bytes(authority_path.read_bytes() + b" ")
        changed = True
        raced = _snapshot(plan_fixture["plan_output"])
        return result

    monkeypatch.setattr(planner, "_reconstruct", changing_reconstruct)
    with pytest.raises(DataReadinessError):
        with planner.verified_initial_fit_raw_share_plan(
            plan_fixture["root"], plan_fixture["plan_config"], plan_fixture["plan_output"],
            expected_plan_sha256=plan["plan_sha256"],
        ):
            yielded = True
    assert changed
    assert not yielded, "replay yielded an authority different from the externally pinned file"
    assert file_sha256(authority_path) != plan["plan_sha256"]
    assert _snapshot(plan_fixture["plan_output"]) == raced
    assert not (plan_fixture["root"] / "runtime" / "heavy-job.owner.json").exists()


def test_per_unit_audit_binds_exact_csv_identity_counts_and_session_hashes(plan_fixture: dict[str, Any]) -> None:
    manifest = _plan(plan_fixture)
    request = json.loads((plan_fixture["plan_output"] / "_request.json").read_text(encoding="utf-8"))
    audit = manifest["unit_requirements"]
    assert request["unit_requirements_sha256"] == json_sha256(audit)
    expected: dict[str, Any] = {}
    for identity in _units(plan_fixture).to_dict(orient="records"):
        stock = identity["role"] == "stock"
        days = list(_SESSIONS) if stock else [*_SESSIONS, "2024-01-19"]
        unit_id = f"swing-daily-{json_sha256(identity)[:24]}"
        expected[unit_id] = {
            "identity": identity, "decision_sessions": 2 if stock else 0,
            "holding_sessions": 11 if stock else 0, "decision_and_holding_sessions": 12 if stock else 0,
            "benchmark_sessions": 0 if stock else 13, "required_sessions": len(days),
            "required_sessions_sha256": json_sha256(days),
        }
    assert audit == expected
    assert sum(record["holding_sessions"] for record in audit.values()) == 187
    assert sum(record["decision_and_holding_sessions"] for record in audit.values()) == 204
    assert sum(record["benchmark_sessions"] for record in audit.values()) == 52
    assert sum(record["required_sessions"] for record in audit.values()) == 256


@pytest.mark.parametrize("defect", [
    "identity", "decision_sessions", "holding_sessions", "decision_and_holding_sessions", "benchmark_sessions",
    "required_sessions", "required_sessions_sha256", "missing_unit", "unexpected_unit", "unit_id",
])
def test_resigned_per_unit_audit_tamper_cannot_override_pinned_source_reconstruction(
    plan_fixture: dict[str, Any], defect: str,
) -> None:
    _plan(plan_fixture)
    output = plan_fixture["plan_output"]
    manifest_path = output / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = manifest["unit_requirements"]
    key = next(key for key, record in audit.items() if record["identity"]["role"] == "stock")
    if defect == "identity":
        audit[key]["identity"]["ticker"] = "WRONG"
    elif defect == "required_sessions_sha256":
        audit[key][defect] = json_sha256([])
    elif defect == "missing_unit":
        del audit[key]
    elif defect in {"unexpected_unit", "unit_id"}:
        audit["swing-daily-" + "0" * 24] = audit[key]
        if defect == "unit_id":
            del audit[key]
    else:
        audit[key][defect] += 1
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    request_path = output / "_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["unit_requirements_sha256"] = json_sha256(audit)
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    _resign_synthetic_plan_metadata(output)
    new_pin = file_sha256(output / "_authority.json")
    before = _snapshot(output)
    with pytest.raises(DataReadinessError, match="replay"):
        _plan(plan_fixture, expected_plan_sha256=new_pin)
    assert _snapshot(output) == before


def test_disconnected_decision_runs_have_disjoint_exact_per_unit_audits(plan_fixture: dict[str, Any]) -> None:
    # January 31 decisions are immature; January 19-30 must not be fabricated
    # between the mature January 2/3 holding union and the late decision-only run.
    plan_fixture["request"]["initial_fit_end"] = "2024-01-31"
    _bind(plan_fixture)
    plan_fixture["output"].unlink()
    preflight = _run(plan_fixture)
    plan_fixture["plan_request"]["preflight_sha256"] = preflight["audit_sha256"]
    _write_request(plan_fixture)
    manifest = _plan(plan_fixture)
    units = _units(plan_fixture)
    stocks = units.loc[units.role.eq("stock")]
    assert len(stocks) == 33
    calendar = (*_SESSIONS, "2024-01-19", "2024-01-22", "2024-01-23", "2024-01-24", "2024-01-25",
        "2024-01-26", "2024-01-29", "2024-01-30", "2024-01-31")
    audit = manifest["unit_requirements"]
    for identity, group in stocks.groupby("security_id"):
        actual: set[str] = set()
        for record in group.to_dict(orient="records"):
            days = [day for day in calendar if record["start_date"] <= day <= record["end_date"]]
            assert not actual.intersection(days)
            actual.update(days)
            unit = audit[f"swing-daily-{json_sha256(record)[:24]}"]
            late = record["start_date"] == "2024-01-31"
            assert unit == {
                "identity": record, "decision_sessions": 1 if late else 2,
                "holding_sessions": 0 if late else 11, "decision_and_holding_sessions": 1 if late else 12,
                "benchmark_sessions": 0, "required_sessions": len(days), "required_sessions_sha256": json_sha256(days),
            }
        assert actual == set(_SESSIONS).union({"2024-01-31"} if identity != "s001" else set())
    for record in units.loc[units.role.eq("benchmark")].to_dict(orient="records"):
        unit = audit[f"swing-daily-{json_sha256(record)[:24]}"]
        assert unit["decision_sessions"] == unit["holding_sessions"] == unit["decision_and_holding_sessions"] == 0
        assert unit["benchmark_sessions"] == unit["required_sessions"] == 21
        assert unit["required_sessions_sha256"] == json_sha256(list(calendar))
    counts = manifest["requirements"]
    assert counts["decision_sessions"] == 50
    assert counts["mature_decisions"] == 34
    assert counts["holding_sessions"] == 187
    assert counts["decision_and_holding_sessions"] == 220


def test_nontrading_cutoff_keeps_inclusive_benchmark_request_but_hashes_only_xnys_sessions(
    plan_fixture: dict[str, Any],
) -> None:
    plan_fixture["request"]["initial_fit_end"] = "2024-01-21"
    _bind(plan_fixture)
    plan_fixture["output"].unlink()
    preflight = _run(plan_fixture)
    plan_fixture["plan_request"]["preflight_sha256"] = preflight["audit_sha256"]
    _write_request(plan_fixture)
    manifest = _plan(plan_fixture)
    request = json.loads((plan_fixture["plan_output"] / "_request.json").read_text(encoding="utf-8"))
    assert request["initial_fit_end"] == "2024-01-21"
    units = _units(plan_fixture)
    stocks = units.loc[units.role.eq("stock")]
    assert len(stocks) == 17
    assert stocks.end_date.eq("2024-01-18").all()
    benchmarks = units.loc[units.role.eq("benchmark")]
    assert len(benchmarks) == 4
    assert benchmarks.start_date.eq("2024-01-02").all()
    assert benchmarks.end_date.eq("2024-01-21").all()
    for identity in benchmarks.to_dict(orient="records"):
        unit = manifest["unit_requirements"][f"swing-daily-{json_sha256(identity)[:24]}"]
        assert unit["identity"] == identity
        assert unit["decision_sessions"] == unit["holding_sessions"] == unit["decision_and_holding_sessions"] == 0
        assert unit["benchmark_sessions"] == unit["required_sessions"] == 13
        assert unit["required_sessions_sha256"] == json_sha256([*_SESSIONS, "2024-01-19"])
    assert request["unit_requirements_sha256"] == json_sha256(manifest["unit_requirements"])
    assert manifest["requirements"]["mature_decisions"] == 34
    assert manifest["requirements"]["holding_sessions"] == 187
    assert manifest["requirements"]["decision_and_holding_sessions"] == 204
