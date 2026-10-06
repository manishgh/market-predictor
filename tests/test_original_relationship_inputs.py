"""Synthetic unit-only inputs; no fixture substitutes for retained-data acceptance."""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import write_json_object
from market_predictor.heavy_jobs import HeavyJobBusyError
from market_predictor.research import original_relationship_inputs as owner
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_feature_profiles import ReturnRelationshipSources
from market_predictor.swing.contracts.return_relationship_reuse import RelationshipReusePolicy
from market_predictor.swing.datasets.relationship_historical_evidence import HistoricalFeatureEvidence, historical_bars
from market_predictor.swing.datasets.return_relationship_reuse import ReuseEvidence
from tests.test_swing_return_relationship_reuse import _historical_bar_fixture


def _pin(root: Path, path: str, data: bytes = b"unit evidence") -> SourcePin:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return SourcePin(path=path, sha256=file_sha256(target))


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    raw_request, raw_item, bars_path = _historical_bar_fixture(tmp_path)
    pins = {name: _pin(tmp_path, f"unit/{name}.json") for name in (
        "historical_parent_publication", "historical_parent_receipt", "historical_relationship_publication",
        "historical_relationship_receipt", "peer_reuse_report", "configuration_difference_report", "strategy_contract")}
    correction = _pin(tmp_path, "unit/corrections.toml")
    feature_lines = ['schema_version = "market_predictor.corrected_research_features"']
    for name, pin in {"outcome_source_config": correction, "strategy_contract": pins["strategy_contract"],
                      "adjusted_plan_authority": pins["strategy_contract"],
                      "adjusted_archive_authority": pins["strategy_contract"]}.items():
        feature_lines.extend((f"[{name}]", f'path = "{pin.path}"', f'sha256 = "{pin.sha256}"'))
    feature = _pin(tmp_path, "unit/features.toml", "\n".join(feature_lines).encode())
    reuse = RelationshipReusePolicy(schema_version="market_predictor.relationship_reuse_config", feature_config=feature, **pins)
    config_path = tmp_path / "unit/reuse.json"
    write_json_object(config_path, reuse.model_dump(mode="json"))
    config = SourcePin(path="unit/reuse.json", sha256=file_sha256(config_path))
    names = [f"feature_{index}" for index in range(124)]
    clocks = dict.fromkeys(names, "predictor_available_at_utc")
    rows = pd.DataFrame({"decision_id": ["april", "may-late", "may-early"], "security_id": "test-security",
        "ticker": "AAA", "parent_ticker": "AAA", "session_date_et": [date(2024, 4, 30), date(2024, 5, 28), date(2024, 5, 24)],
        "decision_time_utc": pd.to_datetime(["2024-04-30T20:15Z", "2024-05-28T20:15Z", "2024-05-24T20:15Z"]),
        "feature_profile": "technical_relationships", "unchanged_target": [0.1, None, -0.2]})
    frames = {"2024-04": rows.iloc[:1].reset_index(drop=True), "2024-05": rows.iloc[1:].reset_index(drop=True)}
    months = {month: {"rows": len(frame), "decision_ids_sha256": json_sha256(sorted(frame.decision_id)),
        "profiles": {"technical_relationships": {"model_columns": names, "availability_columns": clocks}}}
        for month, frame in frames.items()}
    key = json_sha256(["test-security", "AAA"])
    inventory = {key: {**raw_item, "source_group": "AAA", "rows": 3,
        "decision_ids_sha256": json_sha256(sorted(rows.decision_id)), "quarantine": None}}
    sources = ReturnRelationshipSources(baseline_authority_sha256=pins["historical_parent_publication"].sha256,
        stock_authority_sha256="1" * 64, spy_authority_sha256="2" * 64, availability_semantics="historical_proxy",
        availability_policy_id="market_interval_close", availability_policy_sha256="3" * 64,
        price_adjustment="all", price_basis_and_vintage_id="original-unit-only")
    files = {**raw_request["source_files"], **{pin.path: pin.sha256 for pin in (*pins.values(), correction, feature, config)}}
    request = {"stock_inventory": inventory, "source_files": files, "sources": sources.model_dump(mode="json"),
        "source_basis": {"original": "unit-only"}, "model_columns": names, "availability_columns": clocks}
    manifest = {"rows": 3, "months": months}
    original = HistoricalFeatureEvidence(tmp_path / pins["historical_relationship_publication"].path,
        manifest, request, {"additions_source_replayed": True}, files, "technical_relationships")
    parent = HistoricalFeatureEvidence(tmp_path / pins["historical_parent_publication"].path,
        manifest, {}, {}, files, "technical_market")
    evidence = ReuseEvidence(parent, original, files, {"old/producer.py": "4" * 64})
    receipt_path = tmp_path / "unit/replay.json"
    report = {"schema": owner.replay.SCHEMA, "status": "passed_original_snapshot_replay",
        "config": config.model_dump(mode="json"), "failed_current_comparison": pins["peer_reuse_report"].model_dump(mode="json"),
        "source_files": files, "original_sources": sources.model_dump(mode="json"), "rows": 3,
        "decision_ids_sha256": json_sha256(sorted(rows.decision_id)), "original_stock_inventory_sha256": json_sha256(inventory),
        "groups": {key: {"rows": 3, "decision_ids_sha256": json_sha256(sorted(rows.decision_id)), "quarantine": None}},
        "historical_implementation_files": {"old/producer.py": "4" * 64}}
    write_json_object(receipt_path, report)
    receipt_pin = SourcePin(path="unit/replay.json", sha256=file_sha256(receipt_path))
    policy = owner.OriginalRelationshipInputPins(pins["historical_relationship_publication"],
        pins["historical_relationship_receipt"], receipt_pin)
    calls = {"replay": 0, "lease": 0}
    def reproduce(**kwargs: Any) -> dict[str, Any]:
        calls["replay"] += 1
        return copy.deepcopy(report)
    def month_read(evidence: Any, month: str, columns: list[str] | None = None) -> pd.DataFrame:
        frame = frames[month].copy()
        return frame if columns is None else frame.loc[:, columns]
    def spy_read(root: Path, evidence: Any, consumed: dict[str, str]) -> pd.DataFrame:
        frame = historical_bars(root, raw_request, raw_item, consumed)
        return frame.assign(ticker="SPY", security_id="benchmark:SPY")
    @contextmanager
    def lease(*args: Any, **kwargs: Any) -> Any:
        calls["lease"] += 1
        yield
    monkeypatch.setattr(owner.replay, "_replay", reproduce)
    monkeypatch.setattr(owner, "inspect_reuse_evidence", lambda *args: evidence)
    monkeypatch.setattr(owner, "historical_month", month_read)
    monkeypatch.setattr(owner, "load_corrected_outcome_policy", lambda *args: SimpleNamespace(decision_corrections=[]))
    monkeypatch.setattr(owner, "implementation_files", lambda root: {})
    monkeypatch.setattr(owner, "_historical_spy", spy_read)
    monkeypatch.setattr(owner, "guard", lambda *args: None)
    monkeypatch.setattr(owner, "release_process_memory", lambda: None)
    monkeypatch.setattr(owner, "heavy_job_lease", lease)
    return {"root": tmp_path, "policy": policy, "report": report, "frames": frames, "key": key,
        "receipt_path": receipt_path, "calls": calls, "bars_path": bars_path, "evidence": evidence,
        "correction": correction, "source_files": files}


def test_one_context_reproduces_once_and_preserves_monthly_original_order(state: dict[str, Any]) -> None:
    with owner.verified_original_relationship_inputs(state["root"], state["policy"]) as context:
        assert len(context.parent.model_columns) == 124
        assert context.original_sources.baseline_authority_sha256 == state["report"]["original_sources"]["baseline_authority_sha256"]
        assert context.history_sessions[0] == date(2018, 5, 29)
        assert context.history_sessions[-1] == date(2024, 5, 28)
        for month, expected in state["frames"].items():
            pd.testing.assert_frame_equal(context.read_parent_month(month), expected)
            groups = list(context.iter_month_groups(month))
            assert len(groups) == 1 and groups[0][0] == state["key"]
            pd.testing.assert_frame_equal(groups[0][1], expected)
        assert context.read_stock(state["key"]).security_id.eq("test-security").all()
        assert context.read_spy().ticker.eq("SPY").all()
        assert state["calls"] == {"replay": 1, "lease": 1}
    with pytest.raises(DataReadinessError, match="closed"):
        context.read_spy()


def test_private_owner_verifies_without_nested_lease(state: dict[str, Any]) -> None:
    with owner._verified_original_relationship_inputs(state["root"], state["policy"]) as context:
        context.recheck()
    assert state["calls"] == {"replay": 1, "lease": 0}


@pytest.mark.parametrize("field", ["parent_publication", "parent_saved_row_verification"])
def test_wrong_parent_pin_rejects_before_full_replay(state: dict[str, Any], field: str) -> None:
    values = vars(state["policy"]).copy()
    values[field] = SourcePin(path="foreign.json", sha256="a" * 64)
    with pytest.raises(DataReadinessError, match="different parent"):
        with owner._verified_original_relationship_inputs(state["root"], owner.OriginalRelationshipInputPins(**values)):
            pytest.fail("wrong parent was admitted")
    assert state["calls"]["replay"] == 0


def test_rehashed_pass_must_independently_reproduce(state: dict[str, Any]) -> None:
    changed = {**state["report"], "rows": 4}
    state["receipt_path"].write_text(json.dumps(changed), encoding="utf-8")
    original = state["policy"]
    policy = owner.OriginalRelationshipInputPins(original.parent_publication, original.parent_saved_row_verification,
        SourcePin(path=original.original_snapshot_replay.path, sha256=file_sha256(state["receipt_path"])))
    with pytest.raises(DataReadinessError, match="independent reproduction"):
        with owner._verified_original_relationship_inputs(state["root"], policy):
            pytest.fail("rehashed false claim was admitted")


@pytest.mark.parametrize("filename", ["bars_path", "receipt_path"])
def test_mutation_after_last_read_rejects_context_exit(state: dict[str, Any], filename: str) -> None:
    with pytest.raises(DataReadinessError, match="changed"):
        with owner._verified_original_relationship_inputs(state["root"], state["policy"]) as context:
            context.read_stock(state["key"])
            path = state[filename]
            path.write_bytes(path.read_bytes() + b"changed")


def test_modern_repriced_stock_bytes_cannot_replace_original(state: dict[str, Any]) -> None:
    with pytest.raises(DataReadinessError, match="changed"):
        with owner._verified_original_relationship_inputs(state["root"], state["policy"]) as context:
            frame = pd.read_parquet(state["bars_path"])
            frame[["open", "high", "low", "close"]] *= 0.99
            frame.to_parquet(state["bars_path"], index=False)
            context.read_stock(state["key"])


def test_foreign_reader_pin_rejects_even_when_physical_values_are_valid(state: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    foreign = _pin(state["root"], "unit/foreign.bin")
    original = owner._historical_spy
    def substituted(root: Path, evidence: Any, consumed: dict[str, str]) -> pd.DataFrame:
        frame = original(root, evidence, consumed)
        consumed[foreign.path] = foreign.sha256
        return frame
    monkeypatch.setattr(owner, "_historical_spy", substituted)
    with owner._verified_original_relationship_inputs(state["root"], state["policy"]) as context:
        with pytest.raises(DataReadinessError, match="outside the reproduced"):
            context.read_spy()


@pytest.mark.parametrize("column,value", [("security_id", "foreign-issuer"), ("ticker", "OTHER"),
    ("high", 0.1), ("low", 999.0), ("volume", 0.0), ("source", "other"), ("price_feed", "iex"),
    ("available_at_utc", pd.Timestamp("2024-05-24T00:00Z")), ("ingested_at_utc", pd.NaT)])
def test_stock_full_metadata_and_explicit_identity_are_checked(
    state: dict[str, Any], monkeypatch: pytest.MonkeyPatch, column: str, value: Any,
) -> None:
    original = owner.original_stock_bars
    def poisoned(*args: Any) -> pd.DataFrame:
        frame = original(*args)
        frame.loc[0, column] = value
        return frame
    monkeypatch.setattr(owner, "original_stock_bars", poisoned)
    with owner._verified_original_relationship_inputs(state["root"], state["policy"]) as context:
        with pytest.raises(DataReadinessError):
            context.read_stock(state["key"])


def test_monthly_duplicate_ownership_is_rejected(state: dict[str, Any]) -> None:
    with owner._verified_original_relationship_inputs(state["root"], state["policy"]) as context:
        state["frames"]["2024-05"].loc[1, "decision_id"] = "may-late"
        with pytest.raises(DataReadinessError, match="overlaps or loses"):
            list(context.iter_month_groups("2024-05"))


def test_changed_original_quarantine_claim_cannot_be_rehashed_into_acceptance(state: dict[str, Any]) -> None:
    changed = copy.deepcopy(state["report"])
    changed["groups"][state["key"]]["quarantine"] = {"first_invalid_session": "2024-05-28"}
    state["receipt_path"].write_text(json.dumps(changed), encoding="utf-8")
    original = state["policy"]
    policy = owner.OriginalRelationshipInputPins(original.parent_publication, original.parent_saved_row_verification,
        SourcePin(path=original.original_snapshot_replay.path, sha256=file_sha256(state["receipt_path"])))
    with pytest.raises(DataReadinessError, match="independent reproduction"):
        with owner._verified_original_relationship_inputs(state["root"], policy):
            pytest.fail("changed quarantine was admitted")


def test_correction_configuration_mutation_rejects_at_exit(state: dict[str, Any]) -> None:
    with pytest.raises(DataReadinessError, match="changed"):
        with owner._verified_original_relationship_inputs(state["root"], state["policy"]) as context:
            context.read_stock(state["key"])
            (state["root"] / state["correction"].path).write_text("changed correction rules", encoding="utf-8")


def test_no_arbitrary_current_query_keys_or_later_months(state: dict[str, Any]) -> None:
    with owner._verified_original_relationship_inputs(state["root"], state["policy"]) as context:
        with pytest.raises(DataReadinessError, match="original verified group"):
            context.read_stock("swing-daily-fabricated-current-query")
        with pytest.raises(DataReadinessError, match="outside the frozen"):
            context.read_parent_month("2024-06")


def test_returned_metadata_cannot_rewrite_verified_ownership(state: dict[str, Any]) -> None:
    with owner._verified_original_relationship_inputs(state["root"], state["policy"]) as context:
        context.parent.request["stock_inventory"].clear()
        context.source_files.clear()
        assert list(context.iter_month_groups("2024-05"))[0][0] == state["key"]
        assert context.read_stock(state["key"]).ticker.eq("AAA").all()


def test_closed_context_and_direct_constructor_cannot_supply_source_approval(state: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="verified context"):
        owner.OriginalRelationshipInputs()
    with owner._verified_original_relationship_inputs(state["root"], state["policy"]) as context:
        pass
    with pytest.raises(DataReadinessError, match="closed"):
        context.recheck()


def test_busy_public_owner_does_not_read_or_reproduce(state: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def busy(*args: Any, **kwargs: Any) -> Any:
        raise HeavyJobBusyError("unit busy lease")
    monkeypatch.setattr(owner, "heavy_job_lease", busy)
    with pytest.raises(HeavyJobBusyError):
        with owner.verified_original_relationship_inputs(state["root"], state["policy"]):
            pytest.fail("busy lease admitted an input reader")
    assert state["calls"]["replay"] == 0
