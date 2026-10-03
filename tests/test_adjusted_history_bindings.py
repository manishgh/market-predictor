from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import market_predictor.swing.datasets.adjusted_history_bindings as owner
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.adjusted_history_source import AdjustedHistorySource, AdjustedHistoryUnit


def _window(parent_ticker: str = "FISV", query_ticker: str = "FI", *, first: str = "2019-07-09",
    last: str = "2024-05-28", identity: str = "issuer",
) -> dict[str, dict[str, str]]:
    parent = dict(security_id=identity, ticker=parent_ticker, role="stock", start_date=first, end_date=last)
    return dict(parent=parent, query={**parent, "ticker": query_ticker, "start_date": "2018-05-29"})


def _bindings(tmp_path: Path, windows: dict[str, Any]) -> owner.AdjustedHistoryBindings:
    records = {key: {**value["query"], "unit_id": key} for key, value in windows.items()}
    return owner.AdjustedHistoryBindings(AdjustedHistorySource(tmp_path, tmp_path, records, {}), windows, {})


def _decisions(parent_ticker: str = "FISV", ticker: str = "FI") -> pd.DataFrame:
    return pd.DataFrame([dict(decision_id=f"d-{day}", security_id="issuer", parent_ticker=parent_ticker,
        ticker=ticker, session_date_et=date.fromisoformat(day), feature_eligible=False)
        for day in ("2023-06-06", "2023-06-07")])


@pytest.mark.parametrize(("parent_ticker", "query_ticker"), [("FISV", "FI"), ("ECHO", "SATS")])
def test_renamed_issuer_binds_parent_window_without_rewriting_decisions(tmp_path: Path,
    parent_ticker: str, query_ticker: str,
) -> None:
    bindings = _bindings(tmp_path, {"query-unit": _window(parent_ticker, query_ticker)})
    decisions = _decisions(parent_ticker, query_ticker)
    if parent_ticker == "FISV":
        decisions.loc[0, "ticker"] = "FISV"
    result = owner.bind_adjusted_history_decisions(decisions, bindings)
    pd.testing.assert_frame_equal(result.drop(columns="source_group"), decisions)
    assert result.source_group.tolist() == ["query-unit", "query-unit"]
    assert not result.feature_eligible.any()


def test_same_issuer_multiple_windows_bind_by_original_dates(tmp_path: Path) -> None:
    windows = {"old-window": _window(last="2023-06-06"), "new-window": _window(first="2023-06-07")}
    result = owner.bind_adjusted_history_decisions(_decisions(), _bindings(tmp_path, windows))
    assert result.source_group.tolist() == ["old-window", "new-window"]


@pytest.mark.parametrize("kind", ["overlap", "gap", "wrong_owner", "wrong_parent"])
def test_binding_refuses_ambiguous_or_unowned_decisions(tmp_path: Path, kind: str) -> None:
    windows = {"one": _window()}
    if kind == "overlap":
        windows["two"] = _window(first="2023-06-07")
    elif kind == "gap":
        windows["one"] = _window(last="2023-06-06")
    elif kind == "wrong_owner":
        windows["one"] = _window(identity="other")
    else:
        windows["one"] = _window(parent_ticker="FI")
    with pytest.raises(DataReadinessError, match="exactly one original parent"):
        owner.bind_adjusted_history_decisions(_decisions(), _bindings(tmp_path, windows))


def test_duplicate_decision_identity_refused(tmp_path: Path) -> None:
    decisions = _decisions()
    decisions.loc[1, "decision_id"] = decisions.loc[0, "decision_id"]
    with pytest.raises(DataReadinessError, match="unique complete"):
        owner.bind_adjusted_history_decisions(decisions, _bindings(tmp_path, {"one": _window()}))


def _memberships() -> pd.DataFrame:
    return pd.DataFrame([dict(ticker="FISV", security_id="issuer",
        effective_from_utc=pd.Timestamp("2023-06-05T04:00Z"), effective_to_utc=pd.Timestamp("2023-06-08T04:00Z")),
        dict(ticker="OTHER", security_id="issuer", effective_from_utc=pd.Timestamp("2018-05-29T04:00Z"),
            effective_to_utc=pd.NaT)])


def test_history_is_membership_bounded_and_never_splices_other_tickers(tmp_path: Path) -> None:
    bindings = _bindings(tmp_path, {"one": _window()})
    assert owner.expected_bound_history_sessions(bindings, "one", _memberships()) == (
        date(2023, 6, 5), date(2023, 6, 6), date(2023, 6, 7))


def test_overlapping_ticker_ownership_refused(tmp_path: Path) -> None:
    memberships = _memberships()
    memberships = pd.concat([memberships, memberships.iloc[:1].assign(security_id="different")], ignore_index=True)
    with pytest.raises(DataReadinessError, match="multiple securities"):
        owner.expected_bound_history_sessions(_bindings(tmp_path, {"one": _window()}), "one", memberships)


def test_normalized_history_clips_prices_and_retains_absent_invalid_dates(tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings(tmp_path, {"one": _window()})
    expected = owner.expected_bound_history_sessions(bindings, "one", _memberships())
    bars = pd.DataFrame(dict(bar_start_utc=pd.to_datetime(["2023-06-02T13:30Z", "2023-06-05T13:30Z"]),
        ticker=["FI", "FI"], ingested_at_utc=pd.Timestamp("2026-10-03T12:00Z"),
        available_at_utc=pd.to_datetime(["2023-06-02T20:15Z", "2023-06-05T20:15Z"])))
    unit = AdjustedHistoryUnit(bars, (date(2023, 6, 6), date(2023, 6, 8)), (date(2023, 6, 7),), bindings.source.records["one"])
    monkeypatch.setattr(owner, "read_adjusted_history_unit", lambda source, unit_id: unit)
    result = owner.read_bound_adjusted_history(bindings, "one", expected)
    assert len(result.bars) == 1 and result.missing_sessions == (date(2023, 6, 6),)
    assert result.invalid_sessions == (date(2023, 6, 7),)
    assert result.bars.ingested_at_utc.iloc[0] > result.bars.available_at_utc.iloc[0]
    assert result.record["ticker"] == "FI"


@pytest.mark.parametrize("poison", ["duplicate", "missing", "query_changed", "owner_changed"])
def test_loader_rejects_parent_mapping_poison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison: str) -> None:
    window = _window()
    record = {**window["query"], "unit_id": "one"}
    rows = [window]
    if poison == "duplicate":
        rows.append(window)
    elif poison == "missing":
        rows.clear()
    elif poison == "query_changed":
        window["query"] = {**window["query"], "ticker": "OTHER"}
    else:
        window["parent"] = {**window["parent"], "security_id": "other"}
    request = tmp_path / "_request.json"
    request.write_text(json.dumps(dict(parent_window_mapping=rows)), encoding="utf-8")
    source = AdjustedHistorySource(tmp_path, tmp_path, {"one": record}, {"_request.json": file_sha256(request)})
    monkeypatch.setattr(owner, "load_adjusted_history_source", lambda **kwargs: source)
    pin = SourcePin(path="_authority.json", sha256="a" * 64)
    with pytest.raises(DataReadinessError, match="parent-window|parent/query"):
        owner.load_adjusted_history_bindings(root=tmp_path, plan_authority=pin, archive_authority=pin)


def test_loader_uses_verified_request_hash_and_immutable_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window = _window()
    request = tmp_path / "_request.json"
    request.write_text(json.dumps(dict(parent_window_mapping=[window])), encoding="utf-8")
    source = AdjustedHistorySource(tmp_path, tmp_path, {"one": {**window["query"], "unit_id": "one"}},
        {"_request.json": file_sha256(request)})
    monkeypatch.setattr(owner, "load_adjusted_history_source", lambda **kwargs: source)
    pin = SourcePin(path="_authority.json", sha256="a" * 64)
    loaded = owner.load_adjusted_history_bindings(root=tmp_path, plan_authority=pin, archive_authority=pin)
    with pytest.raises(TypeError):
        loaded.windows["one"]["query"]["ticker"] = "changed"  # type: ignore[index]
    request.write_text("{}", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="independent file pin"):
        owner.load_adjusted_history_bindings(root=tmp_path, plan_authority=pin, archive_authority=pin)


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_benchmark_binding_requires_one_exact_unit(tmp_path: Path, kind: str) -> None:
    spy = dict(security_id="benchmark:SPY", ticker="SPY", role="benchmark", start_date="2018-05-29", end_date="2024-05-28")
    records = {} if kind == "missing" else {"one": spy, "two": spy}
    bindings = owner.AdjustedHistoryBindings(AdjustedHistorySource(tmp_path, tmp_path, records, {}), {}, {})
    with pytest.raises(DataReadinessError, match="benchmark"):
        owner.read_bound_benchmarks(bindings, {"SPY"})


def test_benchmark_reader_keeps_xlc_accepted_inception_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xlc = dict(security_id="benchmark:XLC", ticker="XLC", role="benchmark", start_date="2018-06-19", end_date="2024-05-28")
    source = AdjustedHistorySource(tmp_path, tmp_path, {"xlc-unit": xlc}, {})
    bindings = owner.AdjustedHistoryBindings(source, {}, {})

    def read(context: AdjustedHistorySource, unit_id: str) -> AdjustedHistoryUnit:
        assert context is source and unit_id == "xlc-unit"
        return AdjustedHistoryUnit(pd.DataFrame(dict(ticker=["XLC"],
            bar_start_utc=[pd.Timestamp("2018-06-19T13:30Z")])), (), (), xlc)

    monkeypatch.setattr(owner, "read_adjusted_history_unit", read)
    result = owner.read_bound_benchmarks(bindings, {"XLC"})
    assert len(result) == 1 and result.bar_start_utc.iloc[0] == pd.Timestamp("2018-06-19T13:30Z")
