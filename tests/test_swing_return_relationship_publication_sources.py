"""Small physical-source fixtures for corrected ownership and nullable quarantine."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.contracts import CANONICAL_SCHEMA_VERSION
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_COLUMNS
from market_predictor.swing.datasets.adjusted_history_bindings import AdjustedHistoryBindings
from market_predictor.swing.datasets.adjusted_history_source import AdjustedHistorySource
from market_predictor.swing.datasets.predictor_abstention_derivation import ReviewedPredictorFailure
from market_predictor.swing.datasets.return_relationship_rows import PHYSICAL_COLUMNS, read_stock
from tests.test_swing_return_feature_profiles import Inputs, _build
from tests.test_swing_return_feature_profiles import contract as contract
from tests.test_swing_return_feature_profiles import inputs as inputs
from tests.test_swing_training_readiness import _json


def _physical(inputs: Inputs, ticker: str) -> pd.DataFrame:
    return inputs.stocks.assign(ticker=ticker, high=inputs.stocks.close * 1.01,
        low=inputs.stocks.open * 0.99, schema_version=CANONICAL_SCHEMA_VERSION)


def _source(root: Path, frame: pd.DataFrame, identity: str) -> tuple[Any, dict[str, Any]]:
    path = root / "stock.parquet"
    raw = frame.assign(security_id=identity, timeframe="1Day", session_date=frame.session_date_et.astype(str),
        bar_start_utc=pd.DatetimeIndex(frame.session_date_et).tz_localize("America/New_York").tz_convert("UTC"))
    raw.to_parquet(path,index=False)
    unit_path = root / "_manifest.json"
    _json(unit_path,{})
    record = dict(unit_id="stock-unit",security_id=identity,ticker=frame.ticker.iloc[0],provider_symbol=frame.ticker.iloc[0],
        role="stock",status="observed",start_date=str(frame.session_date_et.min()),end_date=str(frame.session_date_et.max()),
        rows=len(frame),bars_path=path.name,bars_sha256=file_sha256(path),
        unit_manifest_path=unit_path.name,unit_manifest_sha256=file_sha256(unit_path))
    sources = MappingProxyType({path.name:file_sha256(path),unit_path.name:file_sha256(unit_path)})
    source = AdjustedHistorySource(root,root,MappingProxyType({"stock-unit":record}),sources)
    window = {name:record[name] for name in ("security_id","ticker","role","start_date","end_date")}
    bindings = AdjustedHistoryBindings(source,MappingProxyType({"stock-unit":{"parent":window,"query":window}}),sources)
    return bindings,record


def _members(bindings: Any) -> pd.DataFrame:
    record = bindings.source.records["stock-unit"]
    return pd.DataFrame([dict(security_id=record["security_id"],ticker=record["ticker"],
        effective_from_utc=pd.Timestamp(record["start_date"],tz="America/New_York"),
        effective_to_utc=pd.Timestamp(record["end_date"],tz="America/New_York")+pd.Timedelta(days=1))])


def _identity(inputs: Inputs, identity: str, ticker: str) -> None:
    inputs.decisions = inputs.decisions.assign(security_id=identity, ticker=ticker)
    inputs.baseline = replace(inputs.baseline, rows=inputs.baseline.rows.assign(security_id=identity, ticker=ticker))


@pytest.mark.parametrize("ticker,identity", [("FI", "cik:0000798354"), ("SATS", "cik:0001415404")])
def test_verified_query_stream_uses_explicit_decision_corrections_and_refuses_substitution(tmp_path: Path, inputs: Inputs,
    ticker: str, identity: str,
) -> None:
    frame = _physical(inputs,ticker)
    bindings,record = _source(tmp_path,frame,identity)
    decision_ticker = "FISV" if ticker == "FI" else ticker
    corrections = [SimpleNamespace(security_id=identity,ticker=decision_ticker,
        first_session=inputs.sessions[0],last_session=inputs.sessions[-1])]
    context = SimpleNamespace(preserved_abstentions=None, bindings=bindings,memberships=_members(bindings),
        outcome_policy=SimpleNamespace(decision_corrections=corrections),
        facts=SimpleNamespace(failures=[]))
    item = dict(security_id=identity,source_group="stock-unit",artifact=record,quarantine=None)
    actual = read_stock(tmp_path,context,item)
    assert len(actual)==len(frame) and tuple(actual.session_date_et)==inputs.sessions
    assert actual.ticker.eq(decision_ticker).all()
    assert record["ticker"]==ticker
    pd.testing.assert_series_equal(actual.close,frame.close,check_names=False)
    _identity(inputs,identity,decision_ticker)
    inputs.stocks=actual
    assert _build(inputs).rows.iloc[-1][list(RETURN_RELATIONSHIP_COLUMNS)].notna().all()
    with pytest.raises(DataReadinessError,match="verified query"):
        read_stock(tmp_path,context,{**item,"source_group":"other"})
    path=tmp_path/record["bars_path"]
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(DataReadinessError,match="hash"):
        read_stock(tmp_path,context,item)


def _quarantine(root: Path, inputs: Inputs, ticker: str, boundary: int | None) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    identity = "security-" + ticker
    frame = _physical(inputs, ticker)
    invalid = []
    if boundary is not None:
        frame.loc[boundary:, "volume"] = 0.0
        row = frame.iloc[boundary]
        invalid = [dict(session_date_et=str(row.session_date_et), invalid_fields=["volume"],
            ohlcv={name: float(row[name]) for name in ("open", "high", "low", "close", "volume")},
            clocks={name: row[name].isoformat() for name in ("bar_start_utc", "bar_end_utc", "available_at_utc")})]
    bindings, record = _source(root,frame,identity)
    source = SourcePin(path=record["bars_path"], sha256=record["bars_sha256"])
    report = root / "observations.json"
    _json(report, dict(schema="market_predictor.predictor_source_failure_observations",
        numeric_first="2018-05-29", numeric_last="2024-05-28", observations=[dict(security_id=identity,
            ticker=ticker, source_path=source.path, source_sha256=source.sha256, bounded_rows=len(frame), invalid_rows=invalid)]))
    observation = SourcePin(path=report.name, sha256=file_sha256(report))
    fact = ReviewedPredictorFailure(group_key=json_sha256([identity, "stock-unit"]), security_id=identity, symbol=ticker,
        rows=len(inputs.decisions), parent_failure_sha256="b" * 64,
        reason_code="unverified_issuer_history" if boundary is None else "invalid_observation_stream",
        quarantine="entire_failed_group" if boundary is None else "suffix_from_first_invalid",
        first_invalid_session=None if boundary is None else inputs.sessions[boundary],
        boundary_observation_sha256=None if boundary is None else json_sha256(invalid[0]),
        source_artifacts=(source,), reviewed_evidence=(observation,), detail="Synthetic source branch regression")
    context = SimpleNamespace(preserved_abstentions=None, bindings=bindings,memberships=_members(bindings),
        outcome_policy=SimpleNamespace(decision_corrections=[]),
        facts=SimpleNamespace(failures=[fact], observations=observation))
    item = dict(security_id=identity, source_group="stock-unit", artifact=record, quarantine=fact.model_dump(mode="json"))
    _identity(inputs, identity, ticker)
    return context, item, frame


def test_wtw_decisions_survive_with_null_additions_and_unchanged_eligibility(tmp_path: Path, inputs: Inputs) -> None:
    context, item, _ = _quarantine(tmp_path, inputs, "WTW", None)
    original = inputs.baseline.rows.copy()
    inputs.stocks = read_stock(tmp_path, context, item)
    assert inputs.stocks.empty
    result = _build(inputs).rows
    assert result[list(RETURN_RELATIONSHIP_COLUMNS)].isna().all(axis=None)
    assert result[[f"available_at_{name}" for name in RETURN_RELATIONSHIP_COLUMNS]].isna().all(axis=None)
    assert result[[f"missing_reason_{name}" for name in RETURN_RELATIONSHIP_COLUMNS]].notna().all(axis=None)
    columns = original.columns.drop("feature_profile")
    pd.testing.assert_frame_equal(result[columns], original[columns], check_exact=True)


@pytest.mark.parametrize("ticker", ["ATVI", "INFO", "SBNY"])
def test_suffix_quarantine_preserves_exact_prefix_and_excludes_boundary_and_later(tmp_path: Path, inputs: Inputs,
    ticker: str,
) -> None:
    context, item, original = _quarantine(tmp_path, inputs, ticker, 252)
    actual = read_stock(tmp_path, context, item)
    assert len(actual) == 252 and actual.session_date_et.max() == inputs.sessions[251]
    pd.testing.assert_frame_equal(actual.loc[:, PHYSICAL_COLUMNS], original.loc[:251, PHYSICAL_COLUMNS], check_exact=True)
    inputs.stocks = actual
    result = _build(inputs).rows
    assert result[RETURN_RELATIONSHIP_COLUMNS[0]].iloc[:3].notna().all()
    assert result[list(RETURN_RELATIONSHIP_COLUMNS)].iloc[3:].isna().all(axis=None)
    columns = inputs.baseline.rows.columns.drop("feature_profile")
    pd.testing.assert_frame_equal(result[columns], inputs.baseline.rows[columns], check_exact=True)


def test_physical_missing_session_is_not_compressed_into_lag_positions(tmp_path: Path, inputs: Inputs) -> None:
    frame = _physical(inputs, "AAA").drop(index=248)
    bindings,record = _source(tmp_path,frame,"security-a")
    context = SimpleNamespace(preserved_abstentions=None, bindings=bindings,memberships=_members(bindings),
        outcome_policy=SimpleNamespace(decision_corrections=[]),facts=SimpleNamespace(failures=[]))
    item = dict(security_id="security-a",source_group="stock-unit",artifact=record,quarantine=None)
    inputs.stocks = read_stock(tmp_path, context, item)
    assert len(inputs.stocks) == 279
    result = _build(inputs).rows
    assert result[list(RETURN_RELATIONSHIP_COLUMNS[:3])].isna().all(axis=None)
    assert pd.notna(result.iloc[-1][RETURN_RELATIONSHIP_COLUMNS[3]])
    assert len(result) == len(inputs.decisions)




def test_query_history_cannot_extend_original_membership(tmp_path: Path, inputs: Inputs) -> None:
    frame = _physical(inputs, "AAA")
    bindings, record = _source(tmp_path, frame, "security-a")
    members = _members(bindings)
    first = inputs.sessions[3]
    members.loc[0, "effective_from_utc"] = pd.Timestamp(first, tz="America/New_York")
    context = SimpleNamespace(preserved_abstentions=None, bindings=bindings, memberships=members,
        outcome_policy=SimpleNamespace(decision_corrections=[]), facts=SimpleNamespace(failures=[]))
    item = dict(security_id="security-a", source_group="stock-unit", artifact=record, quarantine=None)
    actual = read_stock(tmp_path, context, item)
    assert tuple(actual.session_date_et) == inputs.sessions[3:]
    assert pd.notna(_build(inputs).rows.iloc[3][RETURN_RELATIONSHIP_COLUMNS[1]])
    inputs.stocks = actual
    result = _build(inputs).rows
    assert len(result) == len(inputs.decisions)
    assert pd.isna(result.iloc[3][RETURN_RELATIONSHIP_COLUMNS[1]])


def test_invalid_unreviewed_candle_remains_a_calendar_gap(tmp_path: Path, inputs: Inputs) -> None:
    frame = _physical(inputs, "AAA")
    frame.loc[248, "volume"] = 0
    bindings, record = _source(tmp_path, frame, "security-a")
    context = SimpleNamespace(preserved_abstentions=None, bindings=bindings, memberships=_members(bindings),
        outcome_policy=SimpleNamespace(decision_corrections=[]), facts=SimpleNamespace(failures=[]))
    item = dict(security_id="security-a", source_group="stock-unit", artifact=record, quarantine=None)
    inputs.stocks = read_stock(tmp_path, context, item)
    assert inputs.sessions[248] not in set(inputs.stocks.session_date_et)
    result = _build(inputs).rows
    assert len(result) == len(inputs.decisions)
    assert result[list(RETURN_RELATIONSHIP_COLUMNS[:3])].isna().all(axis=None)


def test_reviewed_failure_applies_only_to_its_query_window(tmp_path: Path, inputs: Inputs) -> None:
    from market_predictor.swing.datasets.return_relationship_publication import _validate_inventory
    from market_predictor.swing.datasets.return_relationship_sources import stock_inventory
    context, failed_item, frame = _quarantine(tmp_path, inputs, "WTW", None)
    identity = failed_item["security_id"]
    context.facts.failures = [context.facts.failures[0].model_copy(update={"rows":1})]
    healthy_root = tmp_path / "healthy"
    healthy_root.mkdir()
    healthy_bindings, healthy_record = _source(healthy_root, frame.iloc[140:].copy(), identity)
    healthy_record = {**healthy_record, "unit_id":"healthy-unit",
        "bars_path":"healthy/stock.parquet", "unit_manifest_path":"healthy/_manifest.json"}
    records = {"stock-unit":failed_item["artifact"], "healthy-unit":healthy_record}
    windows = {key:{kind:dict(value) for kind,value in window.items()}
        for key,window in context.bindings.windows.items()}
    windows["stock-unit"]["parent"]["end_date"] = str(inputs.sessions[139])
    windows["healthy-unit"] = healthy_bindings.windows["stock-unit"]
    files = {**context.bindings.source_files,
        **{"healthy/"+name:digest for name,digest in healthy_bindings.source_files.items()}}
    source = AdjustedHistorySource(tmp_path,tmp_path,MappingProxyType(records),MappingProxyType(files))
    context.bindings = AdjustedHistoryBindings(source,MappingProxyType(windows),MappingProxyType(files))
    population = pd.DataFrame(dict(decision_id=["failed-decision","healthy-decision"],security_id=identity,
        parent_ticker="WTW",ticker="WTW",session_date_et=[inputs.sessions[100],inputs.sessions[200]]))
    context.predictor_manifest = {"groups":{json_sha256([identity,unit]):dict(rows=1,decision_ids_sha256=json_sha256([decision]))
        for unit,decision in (("stock-unit","failed-decision"),("healthy-unit","healthy-decision"))}}
    context.predictor_request = {"decision_ids_sha256":json_sha256(sorted(population.decision_id))}
    context.groups = context.predictor_manifest["groups"]
    context.decision_ids_sha256 = context.predictor_request["decision_ids_sha256"]
    inventory = stock_inventory(population,context)
    _validate_inventory(inventory,context)
    assert inventory[json_sha256([identity,"stock-unit"])]["quarantine"] is not None
    healthy = inventory[json_sha256([identity,"healthy-unit"])]
    assert healthy["quarantine"] is None
    actual = read_stock(tmp_path,context,healthy)
    assert tuple(actual.session_date_et) == inputs.sessions[140:]
