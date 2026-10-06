"""Synthetic unit evidence only; never operational source substitutes."""
from __future__ import annotations

from contextlib import nullcontext
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
from market_predictor.research import original_relationship_replay as owner
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets import relationship_historical_evidence as historical
from market_predictor.swing.datasets.preserved_relationship_abstentions import preserved_physical_prefix
from market_predictor.swing.features.research_partition import ResearchFeaturePartition
from tests.test_swing_return_feature_profiles import Inputs, _build
from tests.test_swing_return_feature_profiles import contract as contract
from tests.test_swing_return_feature_profiles import inputs as inputs
from tests.test_swing_return_relationship_reuse import _historical_bar_fixture, _preserved_prefix_inputs, _scope_evidence_fixture


def _inventory(population: pd.DataFrame, *, corrected: bool = False) -> dict[str, Any]:
    item = {"security_id": "security-a", "source_group": "AAA", "kind": "corrected" if corrected else "combined",
        "rows": len(population), "decision_ids_sha256": json_sha256(sorted(population.decision_id))}
    return {json_sha256([item["security_id"], item["source_group"]]): item}


def test_original_owner_preserves_order_without_current_unit_ids(inputs: Inputs) -> None:
    population = inputs.decisions.assign(parent_ticker="AAA")
    groups = owner._assign_original_groups(population, _inventory(population))
    assert len(groups) == 1
    pd.testing.assert_frame_equal(next(iter(groups.values())), population)


@pytest.mark.parametrize("poison", ["ambiguous", "missing", "changed_id", "wrong_ticker"])
def test_original_ownership_rejects_ambiguous_or_changed_population(inputs: Inputs, poison: str) -> None:
    population = inputs.decisions.assign(parent_ticker="AAA")
    inventory = _inventory(population)
    if poison == "ambiguous":
        item = {**next(iter(inventory.values())), "source_group": "OTHER", "kind": "corrected"}
        inventory[json_sha256([item["security_id"], item["source_group"]])] = item
    elif poison == "missing":
        inventory = {}
    elif poison == "changed_id":
        population.loc[0, "decision_id"] = "substituted"
    else:
        population.loc[0, "parent_ticker"] = "OTHER"
    with pytest.raises(DataReadinessError):
        owner._assign_original_groups(population, inventory)


def test_exact_original_kernel_replay_does_not_claim_revised_price_equality(inputs: Inputs) -> None:
    # Expected is saved from the same real numerical owner, with unit-only values.
    expected = _build(inputs).rows
    baseline = ResearchFeaturePartition(inputs.baseline.rows.drop(columns="fixed_horizon_net_return"),
        inputs.baseline.model_columns, inputs.baseline.availability_columns, {})
    replay = owner.build_return_relationship_profile(expected_decisions=inputs.decisions, baseline=baseline,
        stock_bars=inputs.stocks, spy_bars=inputs.spy, history_sessions=inputs.sessions,
        contract=inputs.contract, sources=inputs.sources).rows
    assert owner.compare_frames(expected, replay, list(owner.ADDITIONS), scope="unit", check_dtype=True)["equal"] is True
    changed = inputs.spy.copy()
    changed["open"] *= 0.999
    revised = owner.build_return_relationship_profile(expected_decisions=inputs.decisions, baseline=baseline,
        stock_bars=inputs.stocks, spy_bars=changed, history_sessions=inputs.sessions,
        contract=inputs.contract, sources=inputs.sources).rows
    assert owner.compare_frames(expected, revised, list(owner.ADDITIONS), scope="unit", check_dtype=True)["equal"] is False


@pytest.mark.parametrize("column,value", [
    ("high", 0.5), ("low", 9999.0), ("volume", 0.0), ("high", "200.0"),
    ("source", "other"), ("availability_policy", "observed"), ("ticker", "QQQ"),
    ("ingested_at_utc", pd.NaT), ("bar_end_utc", pd.Timestamp("2022-01-03T21:01:00Z")),
])
def test_full_ohlcv_and_source_clock_validation(inputs: Inputs, column: str, value: Any) -> None:
    bars = inputs.spy.assign(high=inputs.spy.close + 1, low=inputs.spy.open - 1, schema_version="unit-physical")
    owner._validate_physical_history(bars, benchmark=True)
    bars[column] = bars[column].astype(object)
    bars.loc[0, column] = value
    with pytest.raises(DataReadinessError):
        owner._validate_physical_history(bars, benchmark=True)


def test_empty_quarantined_stock_history_retains_typed_clocks(inputs: Inputs) -> None:
    bars = inputs.spy.assign(high=inputs.spy.close + 1, low=inputs.spy.open - 1, schema_version="unit-physical").iloc[:0].copy()
    assert isinstance(bars.bar_start_utc.dtype, pd.DatetimeTZDtype)
    owner._validate_physical_history(bars, benchmark=False)
    with pytest.raises(DataReadinessError, match="SPY is absent"):
        owner._validate_physical_history(bars, benchmark=True)


def test_physical_clock_validation_still_rejects_naive_timestamps(inputs: Inputs) -> None:
    bars = inputs.spy.assign(high=inputs.spy.close + 1, low=inputs.spy.open - 1, schema_version="unit-physical")
    bars["ingested_at_utc"] = bars.ingested_at_utc.dt.tz_localize(None)
    with pytest.raises(DataReadinessError, match="timezone aware"):
        owner._validate_physical_history(bars, benchmark=False)


def test_actual_historical_bytes_are_pinned_and_later_rows_not_projected(tmp_path: Path) -> None:
    request, item, path = _historical_bar_fixture(tmp_path)
    files: dict[str, str] = {}
    rows = historical.historical_bars(tmp_path, request, item, files)
    assert len(rows) == 2
    assert rows.session_date_et.max().isoformat() == "2024-05-28"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(DataReadinessError, match="changed"):
        historical.historical_bars(tmp_path, request, item, {})


@pytest.mark.parametrize("poison", ["volume", "price", "clock", "missing_boundary"])
def test_original_boundary_cannot_be_repaired_or_shortened(poison: str) -> None:
    frame, fact, observation = _preserved_prefix_inputs()
    prefix = preserved_physical_prefix(frame, fact, observation, tuple(frame.session_date_et))
    assert len(prefix) == 1
    if poison == "missing_boundary":
        frame = frame.iloc[:1]
    elif poison == "clock":
        frame.loc[1, "available_at_utc"] += pd.Timedelta(nanoseconds=1)
    elif poison == "volume":
        frame.loc[1, "volume"] = 1.0
    else:
        frame.loc[1, "close"] += 1
    with pytest.raises(DataReadinessError):
        preserved_physical_prefix(frame, fact, observation, tuple(frame.session_date_et))


def test_original_unknown_identity_retains_empty_history() -> None:
    frame, fact, observation = _preserved_prefix_inputs()
    fact = fact.model_copy(update={"first_invalid_session": None, "boundary_observation_sha256": None,
        "quarantine": "entire_failed_group", "reason_code": "unverified_issuer_history"})
    assert preserved_physical_prefix(frame, fact, observation, tuple(frame.session_date_et)).empty


@pytest.mark.parametrize("poison", [None, "source", "first_boundary", "rows", "observation_owner"])
def test_original_inspector_owns_quarantine_without_current_query_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison: str | None,
) -> None:
    mapping, _, _ = _scope_evidence_fixture(tmp_path, monkeypatch)
    frame, _, _ = _preserved_prefix_inputs()
    fact = dict(mapping["historical_fact"])
    key = json_sha256([fact["security_id"], fact["symbol"]])
    fact["group_key"] = key
    source = fact["source_artifacts"][0]
    item = {"kind": "combined", "security_id": fact["security_id"], "source_group": fact["symbol"],
        "rows": fact["rows"], "quarantine": fact,
        "artifact": {"path": "raw.bin", "sha256": source["sha256"]}}
    declared = {source["path"]: source["sha256"], mapping["observation"]["path"]: mapping["observation"]["sha256"]}
    evidence = SimpleNamespace(request={"stock_inventory": {key: item}, "source_files": declared},
        receipt={"additions_source_replayed": True})
    if poison == "source":
        item["artifact"]["sha256"] = "a" * 64
    elif poison == "first_boundary":
        fact["boundary_observation_sha256"] = "b" * 64
    elif poison == "rows":
        frame = frame.iloc[:1]
    elif poison == "observation_owner":
        declared.pop(mapping["observation"]["path"])
    monkeypatch.setattr(historical, "historical_bars", lambda *args: frame)
    if poison is None:
        result = historical.original_stock_bars(tmp_path, evidence, key, {})
        pd.testing.assert_frame_equal(result, frame.iloc[:1].reset_index(drop=True))
    else:
        with pytest.raises(DataReadinessError):
            historical.original_stock_bars(tmp_path, evidence, key, {})


def test_source_mutation_at_last_replay_prevents_receipt_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "unit-source.bin"
    source.write_bytes(b"unit-original")
    before = file_sha256(source)
    def replay(**kwargs: Any) -> dict[str, Any]:
        source.write_bytes(b"changed-during-replay")
        return {"source_files": {"unit-source.bin": before}}
    monkeypatch.setattr(owner, "_replay", replay)
    monkeypatch.setattr(owner, "guard", lambda value: None)
    monkeypatch.setattr(owner, "heavy_job_lease", lambda *args, **kwargs: nullcontext())
    output = tmp_path / "data/reports/result.json"
    with pytest.raises(DataReadinessError, match="changed"):
        owner.replay_original_relationships(root=tmp_path, config=Path("config.json"),
            expected_config_sha256="1" * 64, failed_comparison=SourcePin(path="comparison.json", sha256="2" * 64), output=output)
    assert not output.exists()


def test_receipt_verifier_replays_instead_of_trusting_rehashed_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "receipt.json"
    saved = {"schema": owner.SCHEMA, "status": "passed_original_snapshot_replay", "rows": 10,
        "config": {"path": "config.json", "sha256": "1" * 64},
        "failed_current_comparison": {"path": "comparison.json", "sha256": "2" * 64}}
    write_json_object(path, saved)
    monkeypatch.setattr(owner, "heavy_job_lease", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(owner, "_replay", lambda **kwargs: {**saved, "rows": 9})
    with pytest.raises(DataReadinessError, match="reproduce exactly"):
        owner.verify_original_relationship_receipt(root=tmp_path,
            receipt=SourcePin(path="receipt.json", sha256=file_sha256(path)))


def test_busy_receipt_verifier_refuses_before_reading_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def busy(*args: Any, **kwargs: Any) -> Any:
        raise HeavyJobBusyError("another real job is running")

    def unexpected_read(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("receipt input was read before acquiring the job lease")

    monkeypatch.setattr(owner, "heavy_job_lease", busy)
    monkeypatch.setattr(owner, "read_object", unexpected_read)
    with pytest.raises(HeavyJobBusyError):
        owner.verify_original_relationship_receipt(root=tmp_path,
            receipt=SourcePin(path="absent.json", sha256="1" * 64))
