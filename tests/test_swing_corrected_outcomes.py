"""Tiny synthetic publication, ownership, action, cache and replay poison gates."""
from __future__ import annotations

import io
import json
import shutil
import time
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import exchange_calendars as xcals
import pandas as pd
import pytest
from typer.testing import CliRunner

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.reconciliation import stamp_canonical_decision_ids
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import heavy_job_lease
from market_predictor.swing.contracts.corrected_outcomes import DecisionCorrection
from market_predictor.swing.datasets import corrected_outcomes
from market_predictor.swing.datasets.action_evidence import CorporateActionEvidence
from market_predictor.swing.datasets.corrected_outcome_admission import (
    corrected_decisions,
    corrected_memberships,
    reconcile_actions,
    window_gaps,
)
from market_predictor.swing.datasets.corrected_outcomes import (
    MonthEngine,
    load_corrected_outcome_policy,
    materialize_corrected_outcomes,
)
from market_predictor.swing.evaluation.trade_simulation import load_trade_simulation_context
from market_predictor.swing.labels.holding_identity import membership_session_coverage

REPO = Path(__file__).resolve().parents[1]
CALENDAR = xcals.get_calendar("XNYS")


def action_evidence(records: dict[str, dict[str, list[dict[str, Any]]]], missing: tuple[str, ...] = ()) -> CorporateActionEvidence:
    digest = json_sha256({"records": records, "unavailable": list(missing), "source_files": {}, "replay": {}})
    return CorporateActionEvidence("a" * 64, "b" * 64, records, missing, {}, {}, digest)


def decision(identity: str, ticker: str, day: date) -> dict[str, Any]:
    row = {"security_id": identity, "ticker": ticker, "sector": "technology", "primary_benchmark": "XLK",
        "session_date_et": day, "decision_time_utc": swing_prediction_cutoffs(pd.Series([day])).iloc[0],
        "timeframe": "1Day", "bar_start_utc": pd.Timestamp(day, tz="America/New_York").tz_convert("UTC"),
        "prediction_cutoff_policy_id": "test-swing-cutoff"}
    return stamp_canonical_decision_ids(pd.DataFrame([row])).iloc[0].to_dict()


@pytest.fixture
def publication(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path
    (root / "configs").mkdir()
    for name in ("swing_corrected_outcomes.toml", "swing_research.toml", "swing_trade_simulation.toml"):
        shutil.copyfile(REPO / "configs" / name, root / "configs" / name)
    config = root / "configs/swing_corrected_outcomes.toml"
    pin = file_sha256(config)
    policy = load_corrected_outcome_policy(root, config, pin)
    raw = root / "data/raw/fixture"
    raw.mkdir(parents=True)
    segments = []
    sessions = tuple(s.date() for s in CALENDAR.sessions_in_range("2024-01-02", "2024-01-19"))
    for identity, ticker, role in (("common:A", "AAA", "stock"), ("common:B", "BBB", "stock"),
            ("benchmark:SPY", "SPY", "benchmark"), ("benchmark:QQQ", "QQQ", "benchmark"),
            ("benchmark:XLK", "XLK", "benchmark")):
        rows = [{"security_id": identity, "ticker": ticker, "session_date": str(day),
            "bar_start_utc": pd.Timestamp(day, tz="America/New_York").tz_convert("UTC"),
            "open": 100.0 + index, "high": 150.0, "low": 90.0, "close": 101.0 + index, "volume": 1000.0,
            "source": "alpaca", "timeframe": "1Day", "price_feed": "sip", "adjustment": "raw",
            "ingested_at_utc": datetime(2026, 9, 9, tzinfo=UTC)} for index, day in enumerate(sessions)]
        path = raw / f"{ticker}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        segments.append({"archive": str(raw), "first_session": "2024-01-02", "last_session": "2024-01-19",
            "artifact": {"security_id": identity, "ticker": ticker, "provider_symbol": ticker, "role": role,
                "unit_id": f"fixture-{ticker}", "bars_path": path.name, "bars_sha256": file_sha256(path)}})
    panel = root / "data/features/parent"
    panel.mkdir(parents=True)
    records, pins = [], {}
    for month, rows in (("2024-01", [decision("common:A", "AAA", date(2024, 1, 2)),
            decision("common:A", "AAA", date(2024, 1, 3)), decision("common:B", "BBB", date(2024, 1, 2))]),
            ("2024-05", [decision("common:A", "AAA", date(2024, 5, 28))])):
        path = panel / f"{month}.parquet"
        pd.DataFrame(rows).assign(forbidden_price_feature=999999).to_parquet(path, index=False)
        digest = file_sha256(path)
        pins[path.relative_to(root).as_posix()] = digest
        records.append({"partition_month": month, "path": path.name, "sha256": digest, "rows": len(rows),
            "first_session": str(min(r["session_date_et"] for r in rows)),
            "last_session": str(max(r["session_date_et"] for r in rows))})
    memberships = pd.DataFrame([{"security_id": identity, "ticker": ticker,
        "effective_from_utc": pd.Timestamp("2019-01-01", tz="UTC"), "effective_to_utc": pd.NaT}
        for identity, ticker in (("common:A", "AAA"), ("common:B", "BBB"))])
    evidence = action_evidence({symbol: {} for symbol in ("AAA", "BBB", "SPY", "QQQ", "XLK")})
    source = {"selection": {"segments": segments}, "memberships": memberships, "source_files": pins,
        "manifest_path": panel / "_manifest.json", "manifest": {"files": records},
        "request": {"retained_security_ids": ["common:A", "common:B"], "cohort_sha256": "f" * 64},
        "plan": {"requirements": {"in_window_decisions": 4}}, "action_index": ({}, ())}
    simulation = load_trade_simulation_context(root / policy.simulation_policy.path,
        expected_sha256=policy.simulation_policy.sha256)
    return {"root": root, "config": config, "pin": pin, "policy": policy, "source": source,
        "evidence": evidence, "simulation": simulation, "output": root / "data/labels/fixture"}


def run_publication(fixture: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    @contextmanager
    def sources(*args: Any, **options: Any) -> Any:
        with heavy_job_lease("test-corrected-outcomes", runtime_dir=fixture["root"] / "runtime"):
            yield fixture["source"]

    with patch.object(corrected_outcomes, "verified_corrected_research_sources", sources), patch.object(
            corrected_outcomes, "load_corporate_action_evidence", return_value=fixture["evidence"]), patch.object(
            corrected_outcomes, "_implementation", side_effect=lambda root: {
                name: file_sha256(root / name) for name in fixture.get("implementation_paths", ())}):
        return materialize_corrected_outcomes(root=fixture["root"], config=fixture["config"],
            expected_config_sha256=fixture["pin"], output=fixture["output"], **kwargs)


def engine(fixture: dict[str, Any]) -> MonthEngine:
    return MonthEngine(fixture["root"], fixture["policy"], fixture["pin"], fixture["source"],
        fixture["simulation"], date(2024, 1, 2), date(2024, 1, 19), io.BytesIO())


def test_reconcile_all_query_participants_not_query_tags() -> None:
    evidence = action_evidence({"AAA": {}, "ZZZ": {"stock_mergers": [{"id": "event", "acquiree_symbol": "AAA",
        "acquirer_symbol": "BBB", "effective_date": "2024-01-10"}]}})
    before = json.dumps(evidence.records_by_symbol, sort_keys=True)
    index, gaps = reconcile_actions(evidence, {"AAA", "ZZZ"})
    assert not gaps and {"AAA", "BBB", "ZZZ"}.issubset(index)
    assert index["AAA"][0].intersects(date(2024, 1, 3), date(2024, 1, 17))
    assert json.dumps(evidence.records_by_symbol, sort_keys=True) == before


@pytest.mark.parametrize("record", [{"id": "x", "symbol": "AAA"},
    {"id": "x", "unexpected_symbols": ["AAA"], "ex_date": "2024-01-09"}])
def test_unknown_economic_or_participant_fields_cannot_admit_absence(record: dict[str, Any]) -> None:
    evidence = action_evidence({"AAA": {"unknown_family": [record]}})
    index, gaps = reconcile_actions(evidence, {"AAA"})
    assert window_gaps(identity="common:A", symbol="AAA", first=date(2024, 1, 3), last=date(2024, 1, 17),
        actions=index, official=(), global_gaps=gaps)


def test_missing_query_and_conflicting_duplicate_are_symbol_isolated() -> None:
    evidence = action_evidence({"AAA": {"cash_dividends": [
        {"id": "same", "symbol": "AAA", "ex_date": "2024-01-09", "special": False},
        {"id": "same", "symbol": "AAA", "ex_date": "2024-01-10", "special": False}]}})
    index, gaps = reconcile_actions(evidence, {"AAA", "ZZZ"})
    assert not gaps
    assert any(w.unavailable_reason == "conflicting_action_record_identity" for w in index["AAA"])
    assert any(w.unavailable_reason == "action_query_unavailable" for w in index["ZZZ"])
    assert not window_gaps(identity="benchmark:SPY", symbol="SPY", first=date(2024, 1, 3), last=date(2024, 1, 17),
        actions=index, official=(), global_gaps=gaps)


def test_source_symbol_spin_off_reconciliation_is_not_global() -> None:
    evidence = action_evidence({"JEF": {"spin_offs": [{"id": "distribution", "source_symbol": "JEF",
        "new_symbol": "VTS", "ex_date": "2023-01-13"}]}, "SPY": {}})
    index, gaps = reconcile_actions(evidence, {"JEF", "SPY"})
    assert not gaps and set(index) == {"JEF", "VTS"}
    assert index["JEF"][0].intersects(date(2024, 1, 3), date(2024, 1, 17))


def test_identity_restamp_uses_all_metadata_and_preserves_parent() -> None:
    rule = DecisionCorrection(security_id="common:A", parent_ticker="AAA", ticker="BBB",
        first_session=date(2024, 1, 1), last_session=date(2024, 1, 31), document_ids=("reviewed",))
    row = decision("common:A", "AAA", date(2024, 1, 2))
    corrected = corrected_decisions(pd.DataFrame([row]), (rule,)).iloc[0]
    expected = decision("common:A", "BBB", date(2024, 1, 2))
    assert corrected.decision_id == expected["decision_id"] != row["decision_id"]
    assert corrected.parent_decision_id == row["decision_id"]
    poisoned = pd.DataFrame([{**row, "bar_start_utc": row["bar_start_utc"] + pd.Timedelta(seconds=1)}])
    with pytest.raises(DataReadinessError, match="canonical"):
        corrected_decisions(poisoned, (rule,))


def test_corrected_memberships_keep_competing_identity(publication: dict[str, Any]) -> None:
    rule = DecisionCorrection(security_id="common:A", parent_ticker="AAA", ticker="NEW",
        first_session=date(2024, 1, 1), last_session=date(2024, 1, 31), document_ids=("reviewed",))
    rival = {"security_id": "excluded:owner", "ticker": "NEW", "effective_from_utc": pd.Timestamp("2024-01-03", tz="UTC"),
        "effective_to_utc": pd.NaT}
    memberships = pd.concat([publication["source"]["memberships"], pd.DataFrame([rival])], ignore_index=True)
    corrected = corrected_memberships(memberships, (rule,))
    assert "excluded:owner" in set(corrected.security_id)
    coverage = membership_session_coverage(corrected.loc[corrected.ticker.eq("NEW")],
        sessions=(date(2024, 1, 3), date(2024, 1, 4)), security_ids=("common:A",))
    assert not coverage.membership_covered.any()


def test_real_config_pins_and_fbin_inventory_reference() -> None:
    path = REPO / "configs/swing_corrected_outcomes.toml"
    policy = load_corrected_outcome_policy(REPO, path, file_sha256(path))
    for pin in (policy.source_selection, policy.parent_config, policy.action_config, policy.symbol_corrections,
            policy.research_contract, policy.simulation_policy, *(w.document for w in policy.official_windows)):
        assert file_sha256(REPO / pin.path) == pin.sha256
    assert any("FBHS_" in w.document.path for w in policy.official_windows)
    assert len(policy.decision_corrections) == 3 and policy.dividend_policy.startswith("evidenced_marks")


def test_monthly_publication_nullable_population_and_specification_replay(publication: dict[str, Any]) -> None:
    result = run_publication(publication)
    assert result["rows"] == 4 and result["stock_source_admitted"] == 3 and result["fixed_comparisons_complete"] == 3
    assert not result["training_eligible"] and not result["promotion_eligible"]
    january = pd.read_parquet(publication["output"] / "2024-01/targets.parquet")
    terminal = pd.read_parquet(publication["output"] / "2024-05/targets.parquet")
    assert january.managed_horizon_net_return.isna().all() and terminal.fixed_horizon_net_return.isna().all()
    assert january.spy_security_id.eq("benchmark:SPY").all()
    assert "forbidden_price_feature" not in january
    replay = run_publication(publication, expected_output_sha256=result["manifest_sha256"], replay=True)
    assert replay == result


def test_benchmark_missing_preserves_standalone_stock(publication: dict[str, Any]) -> None:
    from market_predictor.swing.datasets.action_windows import ActionWindow

    publication["source"]["action_index"] = ({"SPY": (ActionWindow("unpriced", "cash_dividends",
        date(2024, 1, 9), date(2024, 1, 9), "claim_unavailable"),)}, ())
    row = engine(publication).row(decision("common:A", "AAA", date(2024, 1, 2)))
    assert row["fixed_horizon_net_return"] is not None
    assert row["spy_horizon_gross_return"] is None and row["spy_fixed_horizon_excess_return"] is None
    assert row["qqq_fixed_horizon_excess_return"] is not None and not row["fixed_comparisons_complete"]


def test_cache_parity_and_one_raw_read_per_unit(publication: dict[str, Any]) -> None:
    instance = engine(publication)
    with patch.object(corrected_outcomes, "read_bound_observations", wraps=corrected_outcomes.read_bound_observations) as read:
        first = instance.row(decision("common:A", "AAA", date(2024, 1, 2)))
        second = instance.row(decision("common:B", "BBB", date(2024, 1, 2)))
        instance.row(decision("common:A", "AAA", date(2024, 1, 3)))
        assert read.call_count == 5
    cold = engine(publication).row(decision("common:B", "BBB", date(2024, 1, 2)))
    assert second == cold and first["spy_component_id"] == second["spy_component_id"]
    assert len(instance.benchmarks) == 6


def test_future_poison_outside_holding_is_not_decoded_into_target(publication: dict[str, Any]) -> None:
    baseline = engine(publication).row(decision("common:A", "AAA", date(2024, 1, 2)))
    segment = publication["source"]["selection"]["segments"][0]
    path = Path(segment["archive"]) / segment["artifact"]["bars_path"]
    rows = pd.read_parquet(path)
    rows.loc[rows.session_date.eq("2024-01-19"), ["open", "high", "low", "close"]] = -999999
    rows.to_parquet(path, index=False)
    segment["artifact"]["bars_sha256"] = file_sha256(path)
    assert engine(publication).row(decision("common:A", "AAA", date(2024, 1, 2))) == baseline


def test_competing_owner_and_missing_path_remain_explicit(publication: dict[str, Any]) -> None:
    memberships = publication["source"]["memberships"]
    publication["source"]["memberships"] = pd.concat([memberships, pd.DataFrame([{
        "ticker": "AAA", "security_id": "excluded:owner", "effective_from_utc": pd.Timestamp("2024-01-04", tz="UTC"),
        "effective_to_utc": pd.Timestamp("2024-01-05", tz="UTC")}])], ignore_index=True)
    row = engine(publication).row(decision("common:A", "AAA", date(2024, 1, 2)))
    assert row["fixed_horizon_net_return"] is None and "competing" in row["stock_missing_reasons"]


def test_display_ticker_override_requires_explicit_interval_correction(publication: dict[str, Any]) -> None:
    rule = DecisionCorrection(security_id="common:A", parent_ticker="AAA", ticker="NEW",
        first_session=date(2024, 1, 1), last_session=date(2024, 1, 31), document_ids=("reviewed",))
    display = decision("common:A", "NEW", date(2024, 1, 2))
    memberships = publication["source"]["memberships"]
    publication["source"]["memberships"] = corrected_memberships(memberships, (rule,))
    unbound = engine(publication).row(display)
    assert unbound["fixed_horizon_net_return"] is None and "raw_logical_ticker" in unbound["stock_missing_reasons"]
    publication["policy"] = publication["policy"].model_copy(update={"decision_corrections": (rule,)})
    stream = io.BytesIO()
    instance = MonthEngine(publication["root"], publication["policy"], publication["pin"], publication["source"],
        publication["simulation"], date(2024, 1, 2), date(2024, 1, 19), stream)
    bound = instance.row(display)
    assert bound["ticker"] == "NEW" and bound["fixed_horizon_net_return"] is not None
    record = json.loads(stream.getvalue().splitlines()[0])
    assert record["source_ticker"] == "AAA" and record["provider_symbol"] == "AAA"
    assert record["specification"]["decision_id"] == display["decision_id"]


def test_invalid_midpath_observation_keeps_nullable_decision(publication: dict[str, Any]) -> None:
    segment = publication["source"]["selection"]["segments"][0]
    path = Path(segment["archive"]) / segment["artifact"]["bars_path"]
    rows = pd.read_parquet(path)
    rows.loc[rows.session_date.eq("2024-01-05"), "close"] = -1
    rows.to_parquet(path, index=False)
    segment["artifact"]["bars_sha256"] = file_sha256(path)
    row = engine(publication).row(decision("common:A", "AAA", date(2024, 1, 2)))
    assert row["fixed_horizon_net_return"] is None and "path_observation_invalid" in row["stock_missing_reasons"]


def test_one_month_pilot_publishes_checkpoint_not_completion(publication: dict[str, Any]) -> None:
    pilot = run_publication(publication, maximum_months=1)
    assert pilot["rows"] == 3 and pilot["manifest_sha256"] is None
    assert pilot["status"] == "partial_in_progress" and pilot["run_wall_seconds"] > 0
    assert not (publication["output"] / "_manifest.json").exists()
    progress = json.loads((publication["output"] / "_progress.json").read_text())
    assert progress["month"] == "2024-01" and progress["resources"]["peak_working_set_gib"] > 0
    resumed = run_publication(publication, expected_output_sha256=pilot["checkpoint_sha256"])
    assert resumed["rows"] == 4


def test_system_headroom_guard_prevents_output(publication: dict[str, Any]) -> None:
    from market_predictor.core.errors import MemoryBudgetError

    with patch.object(corrected_outcomes, "_guard", side_effect=MemoryBudgetError("test pressure")):
        with pytest.raises(MemoryBudgetError, match="pressure"):
            run_publication(publication)
    assert not publication["output"].exists()


def test_semantic_dependency_inventory_binds_contracts_cutoffs_and_paths() -> None:
    bound = corrected_outcomes._implementation(REPO)
    required = {f"src/market_predictor/{name}" for name in (
        "swing/contracts/holding_accounting.py", "swing/contracts/trade_simulation.py",
        "canonical/cutoffs.py", "swing/labels/holding_paths.py")}
    assert required.issubset(bound)
    assert all(bound[name] == file_sha256(REPO / name) for name in required)


@pytest.mark.parametrize("dependency", ["swing/contracts/holding_accounting.py", "swing/contracts/trade_simulation.py",
    "canonical/cutoffs.py", "swing/labels/holding_paths.py"])
def test_semantic_dependency_change_rejects_checkpoint_resume(publication: dict[str, Any], dependency: str) -> None:
    name = f"src/market_predictor/{dependency}"
    path = publication["root"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test fixture semantic identity before change", encoding="ascii")
    publication["implementation_paths"] = (name,)
    pilot = run_publication(publication, maximum_months=1)
    path.write_text("test fixture changed semantic identity", encoding="ascii")
    with pytest.raises(DataReadinessError, match="implementation identity changed"):
        run_publication(publication, expected_output_sha256=pilot["checkpoint_sha256"])


def test_published_target_tamper_and_external_pin_required(publication: dict[str, Any]) -> None:
    result = run_publication(publication)
    with pytest.raises(DataReadinessError, match="external"):
        run_publication(publication)
    path = publication["output"] / "2024-01/targets.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(DataReadinessError, match="source changed"):
        run_publication(publication, expected_output_sha256=result["manifest_sha256"])


def test_monthly_resume_does_not_recompile_completed_month(publication: dict[str, Any]) -> None:
    original = corrected_outcomes._month
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("test interruption")
        return original(*args, **kwargs)

    with patch.object(corrected_outcomes, "_month", side_effect=fail_second), pytest.raises(RuntimeError, match="interruption"):
        run_publication(publication)
    checkpoint = file_sha256(publication["output"] / "_checkpoint.json")
    with patch.object(corrected_outcomes, "_month", wraps=original) as compile_month:
        result = run_publication(publication, expected_output_sha256=checkpoint)
        assert compile_month.call_count == 1 and result["rows"] == 4


def test_collection_cli_registered_without_build() -> None:
    from typer.main import get_command

    from market_predictor.collection_cli import app

    result = CliRunner().invoke(app, ["materialize-swing-corrected-outcomes", "--help"], terminal_width=240)
    assert result.exit_code == 0
    command = get_command(app).commands["materialize-swing-corrected-outcomes"]
    assert "expected_config_sha256" in {parameter.name for parameter in command.params}


def test_100_component_runtime_gate(publication: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    instance = engine(publication)
    row = decision("common:A", "AAA", date(2024, 1, 2))
    sessions = tuple(s.date() for s in CALENDAR.sessions_window(pd.Timestamp(row["session_date_et"]), 11)[1:])
    with heavy_job_lease("test-corrected-100-components", runtime_dir=publication["root"] / "runtime"):
        corrected_outcomes._guard()
        started = time.perf_counter()
        for index in range(100):
            result, gaps, _ = instance.component({**row, "decision_id": f"test-perf-{index}"}, sessions)
            assert result is not None and not gaps
        elapsed = time.perf_counter() - started
    with capsys.disabled():
        print(f"\n100 ordinary components: {elapsed:.3f}s; 600000-component compute estimate: {elapsed * 6000 / 60:.2f} minutes")
