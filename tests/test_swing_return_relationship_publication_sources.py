"""Small physical-source fixtures for corrected ownership and nullable quarantine."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_COLUMNS
from market_predictor.swing.datasets.predictor_abstention_derivation import ReviewedPredictorFailure
from market_predictor.swing.datasets.return_relationship_rows import PHYSICAL_COLUMNS, read_stock
from tests.test_swing_return_feature_profiles import Inputs, _build
from tests.test_swing_return_feature_profiles import contract as contract
from tests.test_swing_return_feature_profiles import inputs as inputs
from tests.test_swing_return_relationship_publication import _canonical
from tests.test_swing_training_readiness import _json


def _physical(inputs: Inputs, ticker: str) -> pd.DataFrame:
    return inputs.stocks.assign(ticker=ticker, high=inputs.stocks.close * 1.01,
        low=inputs.stocks.open * 0.99, schema_version="1.0")


def _combined(root: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path = root / "stock.parquet"
    _canonical(frame.loc[:, PHYSICAL_COLUMNS], path, "bars", "a" * 64)
    return dict(path=path.name, sha256=file_sha256(path), rows=len(frame), ticker=frame.ticker.iloc[0],
        canonical_manifest_sha256=file_sha256(manifest_path_for(path)))


def _identity(inputs: Inputs, identity: str, ticker: str) -> None:
    inputs.decisions = inputs.decisions.assign(security_id=identity, ticker=ticker)
    inputs.baseline = replace(inputs.baseline, rows=inputs.baseline.rows.assign(security_id=identity, ticker=ticker))


@pytest.mark.parametrize("ticker,identity", [("FI", "cik:0000798354"), ("SATS", "cik:0001415404")])
def test_corrected_issuer_uses_entire_stream_and_rejects_old_combined(tmp_path: Path, inputs: Inputs,
    ticker: str, identity: str,
) -> None:
    frame = _physical(inputs, ticker).assign(security_id=identity, timeframe="1Day",
        session_date=[str(day) for day in inputs.sessions],
        bar_start_utc=pd.to_datetime(inputs.sessions).tz_localize("America/New_York").tz_convert("UTC"))
    path = tmp_path / "corrected.parquet"
    frame.to_parquet(path, index=False)
    record = dict(bars_path=path.name, bars_sha256=file_sha256(path), security_id=identity, ticker=ticker, rows=len(frame))
    decision_ticker = "FISV" if ticker == "FI" else ticker
    corrections = [SimpleNamespace(security_id=identity, ticker=decision_ticker,
        first_session=inputs.sessions[0], last_session=inputs.sessions[-1])]
    context = SimpleNamespace(corrected_directory=tmp_path, corrected={identity: record},
        outcome_policy=SimpleNamespace(decision_corrections=corrections), facts=SimpleNamespace(failures=[]))
    item = dict(kind="corrected", security_id=identity, source_group=ticker, artifact=record)
    actual = read_stock(tmp_path, context, item)
    assert len(actual) == len(frame) and tuple(actual.session_date_et) == inputs.sessions
    assert actual.ticker.eq(decision_ticker).all()
    pd.testing.assert_series_equal(actual.close, frame.close, check_names=False)
    _identity(inputs, identity, decision_ticker)
    inputs.stocks = actual
    assert _build(inputs).rows.iloc[-1][list(RETURN_RELATIONSHIP_COLUMNS)].notna().all()
    with pytest.raises(DataReadinessError, match="old combined"):
        read_stock(tmp_path, context, {**item, "kind": "combined"})
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(DataReadinessError, match="source changed"):
        read_stock(tmp_path, context, item)


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
    record = _combined(root, frame)
    source = SourcePin(path=record["path"], sha256=record["sha256"])
    report = root / "observations.json"
    _json(report, dict(schema="market_predictor.predictor_source_failure_observations.v1",
        numeric_first="2018-05-29", numeric_last="2024-05-28", observations=[dict(security_id=identity,
            ticker=ticker, source_path=source.path, source_sha256=source.sha256, bounded_rows=len(frame), invalid_rows=invalid)]))
    observation = SourcePin(path=report.name, sha256=file_sha256(report))
    fact = ReviewedPredictorFailure(group_key=json_sha256([identity, ticker]), security_id=identity, symbol=ticker,
        rows=len(inputs.decisions), parent_failure_sha256="b" * 64,
        reason_code="unverified_issuer_history" if boundary is None else "invalid_observation_stream",
        quarantine="entire_failed_group" if boundary is None else "suffix_from_first_invalid",
        first_invalid_session=None if boundary is None else inputs.sessions[boundary],
        boundary_observation_sha256=None if boundary is None else json_sha256(invalid[0]),
        source_artifacts=(source,), reviewed_evidence=(observation,), detail="Synthetic source branch regression")
    context = SimpleNamespace(combined_directory=root, corrected={},
        facts=SimpleNamespace(failures=[fact], observations=observation))
    item = dict(kind="combined", security_id=identity, source_group=ticker, artifact=record, quarantine=fact.model_dump(mode="json"))
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
    record = _combined(tmp_path, frame)
    context = SimpleNamespace(combined_directory=tmp_path, corrected={}, facts=SimpleNamespace(failures=[]))
    item = dict(kind="combined", security_id="security-a", source_group="AAA", artifact=record)
    inputs.stocks = read_stock(tmp_path, context, item)
    assert len(inputs.stocks) == 279
    result = _build(inputs).rows
    assert result[list(RETURN_RELATIONSHIP_COLUMNS[:3])].isna().all(axis=None)
    assert pd.notna(result.iloc[-1][RETURN_RELATIONSHIP_COLUMNS[3]])
    assert len(result) == len(inputs.decisions)
