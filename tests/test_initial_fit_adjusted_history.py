"""Synthetic source identity, reviewed document, and exact transport tests only."""
from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.swing.datasets import history_archive as archive
from market_predictor.swing.datasets import initial_fit_adjusted_history as owner
from market_predictor.swing.datasets.history_plan_publication import UNIT_COLUMNS
from tests import test_swing_feature_history_plan as feature_fixtures
from tests.test_swing_history_collection import _FakeSource, _json, _test_transport_response, _write_json
from tests.test_swing_symbol_corrections import _write_toml

BENCHMARKS = ("SPY", "QQQ", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")
reviewed_evidence = feature_fixtures.evidence


class _FullBenchmarkSource(_FakeSource):
    def __init__(self, fault: str | None = None) -> None:
        super().__init__()
        self.benchmark_fault = fault

    def fetch_daily_page(self, symbol: str, start: datetime, end_exclusive: datetime, *,
        page_token: str | None, asof: date, adjustment: str,
    ) -> archive.SwingDailyPage:
        page = super().fetch_daily_page(symbol, start, end_exclusive,
            page_token=page_token, asof=asof, adjustment=adjustment)
        if symbol not in BENCHMARKS:
            return page
        sessions = owner.holding_calendar(start.date(), asof)
        rows = [{**page.bars[0], "t": pd.Timestamp(day, tz="America/New_York").isoformat()} for day in sessions]
        if symbol == "SPY" and self.benchmark_fault == "zero_volume":
            rows[0]["v"] = 0
        if symbol == "SPY" and self.benchmark_fault == "missing":
            rows.pop(0)
        payload = {"bars": {symbol: rows}, "next_page_token": None}
        assert page.transport_response is not None
        params = dict(parse_qsl(urlsplit(page.transport_response.requested_url).query))
        transport = replace(_test_transport_response(payload, params), retrieved_at_utc=datetime(2025, 1, 1, tzinfo=UTC))
        return replace(page, bars=tuple(rows), raw_payload=payload, transport_response=transport)


def _refresh(state: dict[str, Any], units: pd.DataFrame) -> None:
    root, parent = state["root"], state["root"] / "plan"
    metadata_path = root / "parent-panel-request.json"
    if not metadata_path.exists():
        count = len(owner.holding_calendar(owner.WARMUP_START, owner.NUMERIC_END))
        coverage = [{"ticker": ticker, "action": "retain",
            "coverage_policy": "contiguous_pre_inception_prefix_allowed" if ticker == "XLC" else "exact_full_window",
            "first_observed_session": "2018-06-19" if ticker == "XLC" else "2018-05-29",
            "requested_first_session": "2018-05-29", "requested_last_session": "2024-05-28",
            "expected_session_count": count, "observed_session_count": count - (15 if ticker == "XLC" else 0),
            "missing_session_count": 15 if ticker == "XLC" else 0,
            "pre_inception_missing_session_count": 15 if ticker == "XLC" else 0,
            "first_missing_session": "2018-05-29" if ticker == "XLC" else None} for ticker in BENCHMARKS]
        _write_json(metadata_path, {"combined_daily_inputs": {"benchmark_coverage": coverage}})
    units.to_csv(parent / "daily_bar_units.csv", index=False, lineterminator="\n")
    original = _json(parent / "_request.json")
    original.update(retained_security_ids=sorted(set(units.loc[units.role.eq("stock"), "security_id"])),
        excluded_security_ids=[], decision_start="2019-07-09", initial_fit_end="2024-05-28",
        provider_symbols={ticker: ticker for ticker in units.ticker},
        asof_policy="inclusive_unit_end_date_entity_mapping_not_ownership")
    original["policy"] = {"parent_request_path": "parent-panel-request.json"}
    original["source_files"] = {"parent-panel-request.json": file_sha256(metadata_path)}
    _write_json(parent / "_request.json", original)
    manifest = _json(parent / "_manifest.json")
    manifest.update(scope=original["scope"], request_sha256=file_sha256(parent / "_request.json"))
    manifest["daily_bars"].update(planned_units=len(units), stock_units=int(units.role.eq("stock").sum()),
        benchmark_units=int(units.role.eq("benchmark").sum()), units_artifact={"path": "daily_bar_units.csv",
            "bytes": (parent / "daily_bar_units.csv").stat().st_size,
            "sha256": file_sha256(parent / "daily_bar_units.csv")})
    _write_json(parent / "_manifest.json", manifest)
    authority = _json(parent / "_authority.json")
    authority.update(request_sha256=manifest["request_sha256"], artifact_sha256=file_sha256(parent / "_manifest.json"),
        units_sha256=file_sha256(parent / "daily_bar_units.csv"))
    _write_json(parent / "_authority.json", authority)
    correction = state["correction"]
    correction["parent_plan_sha256"] = file_sha256(parent / "_authority.json")
    for item in correction["corrections"]:
        row = units.loc[units.security_id.eq(item["security_id"])].iloc[0].to_dict()
        item["parent_unit_id"] = "swing-daily-" + json_sha256(row)[:24]
    _write_toml(root / "corrections.toml", correction, "corrections")
    feature = tomllib.loads((root / "feature-history.toml").read_text())
    feature["correction_policy_sha256"] = file_sha256(root / "corrections.toml")
    (root / "feature-history.toml").write_text("\n".join(
        f"{key} = {json.dumps(value)}" for key, value in feature.items()), encoding="utf-8")
    policy = {"schema_version": "market_predictor.initial_fit_adjusted_history_policy",
        "parent_plan_authority": {"path": "plan/_authority.json", "sha256": file_sha256(parent / "_authority.json")},
        "reviewed_feature_history_policy": {"path": "feature-history.toml", "sha256": file_sha256(root / "feature-history.toml")},
        "warmup_start": "2018-05-29", "decision_start": "2019-07-09", "numeric_end": "2024-05-28",
        "adjustment": "all", "purpose": "source_queries_only_preserve_parent_decision_population",
        "benchmark_start_policy": "inherit_pinned_contiguous_prefix", "minimum_benchmark_warmup_sessions": 253}
    # Inline tables keep the fixture's strict nested SourcePin representation.
    (root / "adjusted.toml").write_text("\n".join(
        f'{key} = {{ path = "{value["path"]}", sha256 = "{value["sha256"]}" }}' if isinstance(value, dict)
        else f"{key} = {json.dumps(value)}" for key, value in policy.items()), encoding="utf-8")
    state.update(config=root / "adjusted.toml", policy_pin=file_sha256(root / "adjusted.toml"), units=units)


@pytest.fixture
def evidence(reviewed_evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state = dict(reviewed_evidence)
    records = [{"security_id": identity, "ticker": values[0], "start_date": "2019-07-09",
        "end_date": "2024-05-28", "role": "stock"} for identity, values in owner._CORRECTIONS.items()]
    records.append({"security_id": "security:AAA", "ticker": "AAA", "start_date": "2020-01-02",
        "end_date": "2020-01-03", "role": "stock"})
    records.extend({"security_id": f"benchmark:{ticker}", "ticker": ticker, "start_date": "2019-07-09",
        "end_date": "2024-05-28", "role": "benchmark"} for ticker in BENCHMARKS)
    _refresh(state, pd.DataFrame(records, columns=list(UNIT_COLUMNS)))
    monkeypatch.setattr(owner, "_EXPECTED_COUNTS", (16, 3, 3, 13))
    monkeypatch.setattr(owner, "_guard", lambda: None)
    return state


def _requirements(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    return owner.initial_fit_adjusted_history_requirements(state["root"], state["config"], state["policy_pin"])


def _publish(state: dict[str, Any]) -> tuple[Path, str]:
    plan = state["root"] / "adjusted-plan"
    owner.publish_initial_fit_adjusted_history_plan(state["root"], state["config"], state["policy_pin"], plan)
    return plan, file_sha256(plan / "_authority.json")


def test_identity_only_projection_preserves_population_and_query_windows(evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("numerical payload read")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    request, manifest, units = _requirements(evidence)
    assert len(units) == 16 and units.loc[units.ticker.ne("XLC"), "start_date"].eq("2018-05-29").all()
    assert units.loc[units.ticker.eq("XLC"), "start_date"].item() == "2018-06-19"
    assert units.end_date.tolist() == evidence["units"].end_date.tolist()
    assert units.security_id.tolist() == evidence["units"].security_id.tolist()
    assert set(units.ticker) == set(evidence["units"].ticker) - {"ECHO", "FISV"} | {"SATS", "FI"}
    assert request["parent_window_mapping"][0]["parent"]["ticker"] == "ECHO"
    assert request["parent_window_mapping"][0]["query"]["ticker"] == "SATS"
    assert manifest["daily_bars"]["adjustment"] == "all" and manifest["outcomes_read"] is False
    assert request["provider_symbols"]["FI"] == "FI"
    assert request["benchmark_start_sessions"]["XLC"] == "2018-06-19"
    assert request["benchmark_coverage_source"]["sha256"] == file_sha256(evidence["root"] / "parent-panel-request.json")
    assert len(request["source_files"]) >= 8


def test_frozen_564_window_count_with_distinct_aliases(evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    rows = evidence["units"].to_dict(orient="records")
    rows.extend({"security_id": f"security:{i}", "ticker": f"T{i:03d}", "role": "stock",
        "start_date": "2019-07-09", "end_date": "2024-05-28"} for i in range(542))
    rows.extend({"security_id": f"security:{i}", "ticker": f"OLD{i}", "role": "stock",
        "start_date": "2019-07-09", "end_date": "2020-01-03"} for i in range(6))
    _refresh(evidence, pd.DataFrame(rows, columns=list(UNIT_COLUMNS)))
    monkeypatch.setattr(owner, "_EXPECTED_COUNTS", (564, 551, 545, 13))
    _, manifest, units = _requirements(evidence)
    assert manifest["daily_bars"]["planned_units"] == len(units) == 564
    assert manifest["in_window_stock_securities"] == 545
    assert units.loc[units.ticker.eq("OLD0"), "end_date"].item() == "2020-01-03"
    assert units.loc[units.ticker.eq("XLC"), "start_date"].item() == "2018-06-19"
    assert len(owner.holding_calendar(date(2018, 6, 19), date(2019, 7, 8))) == 264


@pytest.mark.parametrize("poison", ["provider", "window", "adjustment", "membership", "population", "mapping", "implementation"])
def test_rehashed_local_plan_cannot_change_pinned_semantics(evidence: dict[str, Any], poison: str) -> None:
    request, manifest, units = _requirements(evidence)
    if poison == "provider":
        request["provider_symbols"]["FI"] = "FISV"
    elif poison == "window":
        units.loc[0, "end_date"] = "2024-05-29"
    elif poison == "adjustment":
        manifest["daily_bars"]["adjustment"] = "raw"
    elif poison == "membership":
        request["membership_authority"]["universe_sha256"] = "f" * 64
    elif poison == "population":
        request["retained_security_ids"].pop()
    elif poison == "mapping":
        request["parent_window_mapping"][0]["parent"]["ticker"] = "SATS"
    else:
        request["implementation_files"]["swing/datasets/initial_fit_adjusted_history.py"] = "f" * 64
    with pytest.raises(DataReadinessError, match="pinned requirements"):
        owner.validate_initial_fit_adjusted_history_collection_plan(directory=evidence["root"] / "plan",
            request=request, manifest=manifest, units=units)


@pytest.mark.parametrize("source", ["policy", "parent", "correction", "benchmark"])
def test_source_pin_and_reviewed_correction_changes_rejected(evidence: dict[str, Any], source: str) -> None:
    path = {"policy": evidence["config"], "parent": evidence["root"] / "plan/daily_bar_units.csv",
        "correction": evidence["root"] / "corrections.toml", "benchmark": evidence["root"] / "parent-panel-request.json"}[source]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(DataReadinessError):
        _requirements(evidence)


def test_rehashed_bad_correction_cannot_change_issuer(evidence: dict[str, Any]) -> None:
    evidence["correction"]["corrections"][0]["provider_symbol"] = "ECHO"
    _refresh(evidence, evidence["units"])
    with pytest.raises(DataReadinessError, match="provider identity"):
        _requirements(evidence)


def test_publication_and_current_collector_resume_roundtrip(evidence: dict[str, Any]) -> None:
    plan, pin = _publish(evidence)
    output = evidence["root"] / "archive"
    kwargs = dict(mode="collect", root=evidence["root"], config=evidence["config"], policy_sha256=evidence["policy_pin"],
        plan_directory=plan, expected_plan_sha256=pin, output_directory=output, source_factory=_FullBenchmarkSource)
    partial = owner.run_initial_fit_adjusted_history(**kwargs, maximum_units_this_run=1)
    assert partial["status"] == "incomplete"
    assert partial["unattempted_units"] == 15 and not output.exists()
    assert (evidence["root"] / ".archive.collecting/_request.json").is_file()
    assert not (evidence["root"] / ".archive.collecting/_authority.json").exists()
    complete = owner.run_initial_fit_adjusted_history(**kwargs, maximum_units_this_run=16)
    assert complete["status"] == "complete"
    assert complete["stock_missing_sessions"] > 0 and complete["stock_invalid_sessions"] == 0
    assert (output / "_authority.json").is_file() and not (evidence["root"] / ".archive.collecting").exists()
    replay = owner.run_initial_fit_adjusted_history(**{**kwargs, "mode": "offline"},
        expected_archive_sha256=complete["archive_sha256"])
    assert replay["archive_sha256"] == complete["archive_sha256"] and replay["observed_units"] == 16
    assert replay["stock_missing_sessions"] == complete["stock_missing_sessions"]
    with pytest.raises(DataReadinessError, match="independent authority pin"):
        owner.run_initial_fit_adjusted_history(**kwargs)
    with pytest.raises(DataReadinessError, match="must be new"):
        _publish(evidence)


def test_rehashed_archive_cannot_disable_transport_requirement(evidence: dict[str, Any]) -> None:
    plan, pin = _publish(evidence)
    output = evidence["root"] / "archive"
    archive.collect_swing_history_plan(plan_directory=plan, output_directory=output, source_factory=_FakeSource,
        provider_symbol_for=lambda ticker: ticker, expected_plan_authority_sha256=pin)
    request = _json(output / "_request.json")
    request["transport_receipts_required"] = False
    request["request_sha256"] = archive._json_sha256({key: value for key, value in request.items() if key != "request_sha256"})
    _write_json(output / "_request.json", request)
    manifest = _json(output / "_manifest.json")
    manifest["request_sha256"] = request["request_sha256"]
    for item in manifest["unit_artifacts"]:
        path = output / item["unit_manifest_path"]
        unit = _json(path)
        unit["request_sha256"] = request["request_sha256"]
        _write_json(path, unit)
        item["unit_manifest_sha256"] = file_sha256(path)
    manifest["unit_set_sha256"] = archive._unit_artifact_set_sha256(manifest["unit_artifacts"])
    _write_json(output / "_manifest.json", manifest)
    authority = _json(output / "_authority.json")
    authority.update(request_sha256=request["request_sha256"], artifact_sha256=file_sha256(output / "_manifest.json"),
        unit_set_sha256=manifest["unit_set_sha256"])
    _write_json(output / "_authority.json", authority)
    with pytest.raises(DataReadinessError, match="complete authority"):
        archive.load_complete_swing_history_collection(output, plan_directory=plan,
            expected_adjustment="all", expected_plan_authority_sha256=pin)


def test_cli_lease_precedes_input_loading(evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("input read before lease")

    monkeypatch.setattr(owner, "initial_fit_adjusted_history_requirements", forbidden)
    with heavy_job_lease("test-other", runtime_dir=evidence["root"] / "data/runtime"):
        with pytest.raises(HeavyJobBusyError):
            owner.run_initial_fit_adjusted_history(mode="publish-plan", root=evidence["root"], config=evidence["config"],
                policy_sha256=evidence["policy_pin"], plan_directory=Path("output"))


def test_policy_rejects_new_warmup_and_raw_adjustment(evidence: dict[str, Any]) -> None:
    raw = evidence["config"].read_text()
    for changed in (raw.replace("2018-05-29", "2018-05-28"), raw.replace('adjustment = "all"', 'adjustment = "raw"')):
        evidence["config"].write_text(changed)
        evidence["policy_pin"] = file_sha256(evidence["config"])
        with pytest.raises((ValueError, DataReadinessError)):
            _requirements(evidence)


def test_output_cannot_be_nested_in_preserved_raw_archive(evidence: dict[str, Any]) -> None:
    with pytest.raises(DataReadinessError, match="overlaps"):
        owner.publish_initial_fit_adjusted_history_plan(evidence["root"], evidence["config"], evidence["policy_pin"],
            evidence["root"] / "unused-original-raw/new-plan")


def test_standalone_cli_publication_and_missing_replay_pin(evidence: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    common = ["--root", str(evidence["root"]), "--config", str(evidence["config"]),
        "--expected-policy-sha256", evidence["policy_pin"], "--plan-dir", "adjusted-plan"]
    assert owner.main(["publish-plan", *common]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "plan_published"
    assert owner.main(["offline", *common, "--out-dir", "archive"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_or_incomplete"


@pytest.mark.parametrize("poison", ["unknown_policy", "gap", "non_prefix", "changed_first", "full_window", "missing_field",
    "duplicate", "short_history"])
def test_benchmark_metadata_rejects_unproven_prefix_or_insufficient_history(evidence: dict[str, Any], poison: str) -> None:
    metadata = _json(evidence["root"] / "parent-panel-request.json")
    coverage = metadata["combined_daily_inputs"]["benchmark_coverage"]
    row = next(item for item in coverage if item["ticker"] == "XLC")
    if poison == "unknown_policy":
        row["coverage_policy"] = "allow_sparse_gaps"
    elif poison == "gap":
        row["missing_session_count"] += 1
        row["observed_session_count"] -= 1
    elif poison == "non_prefix":
        row["first_missing_session"] = "2018-06-01"
    elif poison == "changed_first":
        row["first_observed_session"] = "2018-06-20"
    elif poison == "full_window":
        row["coverage_policy"] = "exact_full_window"
    elif poison == "missing_field":
        row.pop("pre_inception_missing_session_count")
    elif poison == "duplicate":
        row["ticker"] = "SPY"
    else:
        row["first_observed_session"] = "2018-07-09"
        sessions = tuple(owner.holding_calendar(owner.WARMUP_START, owner.NUMERIC_END))
        prefix = sessions.index(date(2018, 7, 9))
        row.update(missing_session_count=prefix, pre_inception_missing_session_count=prefix,
            observed_session_count=len(sessions) - prefix)
    with pytest.raises(DataReadinessError):
        owner._benchmark_starts(coverage, set(BENCHMARKS))


def test_rehashed_benchmark_query_start_cannot_restore_pre_inception_sessions(evidence: dict[str, Any]) -> None:
    request, manifest, units = _requirements(evidence)
    units.loc[units.ticker.eq("XLC"), "start_date"] = "2018-05-29"
    request["benchmark_start_sessions"]["XLC"] = "2018-05-29"
    with pytest.raises(DataReadinessError, match="pinned requirements"):
        owner.validate_initial_fit_adjusted_history_collection_plan(directory=evidence["root"] / "plan",
            request=request, manifest=manifest, units=units)


@pytest.mark.parametrize("fault", ["missing", "zero_volume"])
@pytest.mark.parametrize("mode", ["collect", "offline"])
def test_benchmark_gaps_and_invalid_candles_cannot_pass_publication_or_offline(
    evidence: dict[str, Any], fault: str, mode: str,
) -> None:
    plan, pin = _publish(evidence)
    output = evidence["root"] / "archive"
    def factory() -> _FullBenchmarkSource:
        return _FullBenchmarkSource(fault)
    archive_pin = None
    if mode == "offline":
        # The generic transport collector alone does not establish complete benchmark sessions.
        archive.collect_swing_history_plan(plan_directory=plan, output_directory=output,
            source_factory=factory, provider_symbol_for=lambda ticker: ticker, expected_plan_authority_sha256=pin)
        archive_pin = file_sha256(output / "_authority.json")
    with pytest.raises(DataReadinessError, match="benchmark has missing or invalid"):
        owner.run_initial_fit_adjusted_history(mode=mode, root=evidence["root"], config=evidence["config"],
            policy_sha256=evidence["policy_pin"], plan_directory=plan, expected_plan_sha256=pin,
            output_directory=output, source_factory=factory, maximum_units_this_run=16,
            expected_archive_sha256=archive_pin)
    if mode == "collect":
        assert not output.exists()
        assert (evidence["root"] / ".archive.collecting/_authority.json").is_file()


def test_source_mutation_during_last_fetch_never_publishes_final_archive(evidence: dict[str, Any]) -> None:
    import threading

    plan, pin = _publish(evidence)
    lock, calls = threading.Lock(), 0

    class ChangingSource(_FullBenchmarkSource):
        def fetch_daily_page(self, *args: Any, **kwargs: Any) -> archive.SwingDailyPage:
            nonlocal calls
            page = super().fetch_daily_page(*args, **kwargs)
            with lock:
                calls += 1
                if calls == 16:
                    path = evidence["root"] / "parent-panel-request.json"
                    path.write_bytes(path.read_bytes() + b"\n")
            return page

    with pytest.raises(DataReadinessError):
        owner.run_initial_fit_adjusted_history(mode="collect", root=evidence["root"], config=evidence["config"],
            policy_sha256=evidence["policy_pin"], plan_directory=plan, expected_plan_sha256=pin,
            output_directory=Path("archive"), source_factory=ChangingSource, maximum_units_this_run=16)
    assert calls == 16 and not (evidence["root"] / "archive").exists()
    assert (evidence["root"] / ".archive.collecting").is_dir()


def test_source_mutation_during_plan_writer_never_publishes_final_plan(
    evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = owner.publish_daily_history_plan

    def mutate(**kwargs: Any) -> dict[str, Any]:
        result = original(**kwargs)
        path = evidence["root"] / "parent-panel-request.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(owner, "publish_daily_history_plan", mutate)
    with pytest.raises(DataReadinessError):
        _publish(evidence)
    assert not (evidence["root"] / "adjusted-plan").exists()
    assert (evidence["root"] / ".adjusted-plan.planning/_authority.json").is_file()
