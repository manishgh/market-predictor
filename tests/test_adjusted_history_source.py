from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets import adjusted_history_source as adapter
from market_predictor.swing.datasets.history_archive import collect_swing_history_plan
from market_predictor.swing.datasets.holding_raw_sources import RAW_COLUMNS
from tests.test_swing_history_collection import _FakeSource, _json, _plan, _write_json


def _pin(root: Path, path: Path) -> SourcePin:
    return SourcePin(path=path.relative_to(root).as_posix(), sha256=file_sha256(path))


@pytest.fixture
def collected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[SourcePin, SourcePin]:
    # Scope construction is independently tested with its plan producer. Only that
    # hook is replaced here; the collection, transport and authority readers run.
    module = ModuleType("market_predictor.swing.datasets.initial_fit_adjusted_history")
    module.validate_initial_fit_adjusted_history_collection_plan = lambda **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    plan = _plan(tmp_path)
    request = _json(plan / "_request.json")
    request["scope"] = "initial_fit_adjusted_history"
    request["source_root"] = str(tmp_path.resolve())
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")
    request["source_files"] = {"policy.json": file_sha256(policy)}
    units = pd.read_csv(plan / "daily_bar_units.csv")
    request["provider_symbols"] = dict(zip(units.ticker, units.ticker, strict=True))
    _write_json(plan / "_request.json", request)
    manifest = _json(plan / "_manifest.json")
    manifest["scope"] = request["scope"]
    manifest["request_sha256"] = file_sha256(plan / "_request.json")
    _write_json(plan / "_manifest.json", manifest)
    authority = _json(plan / "_authority.json")
    authority.update(artifact_sha256=file_sha256(plan / "_manifest.json"), request_sha256=manifest["request_sha256"])
    _write_json(plan / "_authority.json", authority)
    plan_pin = _pin(tmp_path, plan / "_authority.json")
    archive = tmp_path / "archive"
    collect_swing_history_plan(plan_directory=plan, output_directory=archive,
        source_factory=lambda: _FakeSource(empty_symbols={"AAA"}), provider_symbol_for=lambda ticker: ticker,
        expected_plan_authority_sha256=plan_pin.sha256)
    return plan_pin, _pin(tmp_path, archive / "_authority.json")


def test_strict_collection_load_preserves_all_units_and_raw_evidence(
    tmp_path: Path, collected: tuple[SourcePin, SourcePin],
) -> None:
    plan, archive = collected
    before = {path: file_sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    context = adapter.load_adjusted_history_source(root=tmp_path, plan_authority=plan, archive_authority=archive)
    assert len(context.records) == 22
    assert context.training_ready is context.promotion_ready is False
    assert context.source_files[plan.path] == plan.sha256
    assert context.source_files[archive.path] == archive.sha256
    assert context.source_files["policy.json"] == file_sha256(tmp_path / "policy.json")
    for path in (tmp_path / "archive").rglob("*.bin"):
        assert context.source_files[path.relative_to(tmp_path).as_posix()] == file_sha256(path)
    for path in (tmp_path / "archive").rglob("*.gz"):
        assert context.source_files[path.relative_to(tmp_path).as_posix()] == file_sha256(path)
    stock = next(record for record in context.records.values() if record["ticker"] == "AAA")
    empty = adapter.read_adjusted_history_unit(context, stock["unit_id"])
    assert empty.bars.empty and empty.invalid_sessions == ()
    assert empty.missing_sessions == (date(2020, 1, 2),)
    assert empty.security_id == "security:AAA" and empty.ticker == empty.provider_symbol == "AAA"
    observed = next(record for record in context.records.values() if record["ticker"] == "SPY")
    unit = adapter.read_adjusted_history_unit(context, observed["unit_id"])
    assert unit.missing_sessions == unit.invalid_sessions == ()
    assert list(unit.bars.timeframe) == ["1d"]
    assert unit.bars.available_at_utc.iloc[0] == pd.Timestamp("2020-01-02T21:15:00Z")
    assert unit.bars.ingested_at_utc.iloc[0] > unit.bars.available_at_utc.iloc[0]
    assert before == {path: file_sha256(path) for path in before}
    with pytest.raises(TypeError):
        context.records["new"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        stock["ticker"] = "POISON"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        context.training_ready = True  # type: ignore[misc]


@pytest.mark.parametrize("fault", ["plan_pin", "archive_pin", "escape", "scope", "transport", "counter"])
def test_load_refuses_wrong_authorities_scope_transport_and_counters(
    tmp_path: Path, collected: tuple[SourcePin, SourcePin], fault: str,
) -> None:
    plan, archive = collected
    if fault == "plan_pin":
        plan = plan.model_copy(update={"sha256": "0" * 64})
    elif fault == "archive_pin":
        archive = archive.model_copy(update={"sha256": "0" * 64})
    elif fault == "escape":
        plan = plan.model_copy(update={"path": "../outside/_authority.json"})
    elif fault == "scope":
        path = tmp_path / "plan/_request.json"
        payload = _json(path)
        payload["scope"] = "initial_fit_raw_share_acquisition"
        _write_json(path, payload)
    elif fault == "transport":
        path = tmp_path / "archive/_request.json"
        payload = _json(path)
        payload["transport_receipts_required"] = 1
        _write_json(path, payload)
    else:
        path = tmp_path / "archive/_manifest.json"
        payload = _json(path)
        payload["total_rows"] += 1
        _write_json(path, payload)
        authority_path = tmp_path / archive.path
        authority = _json(authority_path)
        authority["artifact_sha256"] = file_sha256(path)
        _write_json(authority_path, authority)
        archive = _pin(tmp_path, authority_path)
    with pytest.raises(DataReadinessError):
        adapter.load_adjusted_history_source(root=tmp_path, plan_authority=plan, archive_authority=archive)


def _unit(tmp_path: Path, *, role: str = "stock", fault: tuple[str, Any] | None = None,
    dates: tuple[str, ...] = ("2020-01-02",),
) -> adapter.AdjustedHistorySource:
    directory = tmp_path / "unit"
    directory.mkdir()
    frame = pd.DataFrame([{
        "security_id": "issuer", "ticker": "FI", "session_date": day,
        "bar_start_utc": pd.Timestamp(day, tz="America/New_York").tz_convert("UTC"),
        "open": 10., "high": 12., "low": 9., "close": 11., "volume": 1000.,
        "source": "alpaca", "timeframe": "1Day", "price_feed": "sip", "adjustment": "all",
        "ingested_at_utc": pd.Timestamp("2026-10-03T12:00:00Z"),
    } for day in dates], columns=RAW_COLUMNS)
    if fault:
        frame[fault[0]] = fault[1]
    path = directory / "bars.parquet"
    frame.to_parquet(path, index=False)
    manifest = directory / "_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    record = MappingProxyType({"unit_id": "unit", "security_id": "issuer", "ticker": "FI", "provider_symbol": "FISV",
        "role": role, "start_date": "2020-01-02", "end_date": "2020-01-03", "status": "observed",
        "rows": len(frame), "bars_path": "bars.parquet", "bars_sha256": file_sha256(path),
        "unit_manifest_path": "_manifest.json", "unit_manifest_sha256": file_sha256(manifest)})
    return adapter.AdjustedHistorySource(tmp_path, directory, MappingProxyType({"unit": record}), MappingProxyType({}))


def test_bounded_projection_preserves_gap_clocks_and_alias_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _unit(tmp_path)
    path = context.directory / "bars.parquet"
    original_hash = file_sha256(path)
    real_read = pd.read_parquet
    calls = []

    def checked(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(adapter.pd, "read_parquet", checked)
    result = adapter.read_adjusted_history_unit(context, "unit")
    assert calls == [{"columns": RAW_COLUMNS, "filters": [
        ("session_date", ">=", "2020-01-02"), ("session_date", "<=", "2020-01-03")]}]
    assert result.missing_sessions == (date(2020, 1, 3),) and result.invalid_sessions == ()
    assert result.security_id == "issuer" and result.ticker == "FI" and result.provider_symbol == "FISV"
    assert result.bars.ingested_at_utc.iloc[0] == pd.Timestamp("2026-10-03T12:00:00Z")
    assert result.bars.available_at_utc.iloc[0] == pd.Timestamp("2020-01-02T21:15:00Z")
    assert result.bars.bar_start_utc.iloc[0] == pd.Timestamp("2020-01-02T14:30:00Z")
    assert file_sha256(path) == original_hash


@pytest.mark.parametrize("column,value", [("volume", 0), ("volume", -1), ("volume", float("inf")),
    ("volume", float("nan")), ("open", 0), ("high", 8), ("close", float("inf"))])
def test_invalid_stock_candles_are_explicit_without_removing_the_unit(tmp_path: Path, column: str, value: float) -> None:
    context = _unit(tmp_path, fault=(column, value))
    result = adapter.read_adjusted_history_unit(context, "unit")
    assert result.invalid_sessions == (date(2020, 1, 2),)
    assert result.missing_sessions == (date(2020, 1, 3),)
    assert result.bars.empty and tuple(context.records) == ("unit",)


@pytest.mark.parametrize("fault", [None, ("volume", 0)])
def test_benchmark_missing_or_invalid_sessions_fail(tmp_path: Path, fault: tuple[str, Any] | None) -> None:
    context = _unit(tmp_path, role="benchmark", fault=fault,
        dates=("2020-01-02",) if fault is None else ("2020-01-02", "2020-01-03"))
    with pytest.raises(DataReadinessError, match="benchmark"):
        adapter.read_adjusted_history_unit(context, "unit")


@pytest.mark.parametrize("column,value", [("security_id", "other"), ("ticker", "FISV"), ("source", "other"),
    ("price_feed", "iex"), ("adjustment", "raw"), ("timeframe", "15Min"),
    ("bar_start_utc", "2020-01-02T12:00:00Z"), ("bar_start_utc", "2020-01-02T05:00:00"),
    ("ingested_at_utc", "2026-10-03T12:00:00")])
def test_identity_feed_basis_and_clock_poison_rejects(tmp_path: Path, column: str, value: str) -> None:
    context = _unit(tmp_path, fault=(column, value))
    with pytest.raises(DataReadinessError):
        adapter.read_adjusted_history_unit(context, "unit")


@pytest.mark.parametrize("dates", [("2020-01-02", "2020-01-02"), ("2020-01-04",), ("2020-01-01",)])
def test_duplicate_outside_and_non_calendar_sessions_reject(tmp_path: Path, dates: tuple[str, ...]) -> None:
    context = _unit(tmp_path, dates=dates)
    with pytest.raises(DataReadinessError):
        adapter.read_adjusted_history_unit(context, "unit")


def test_hash_change_during_projected_read_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _unit(tmp_path)
    real_read = pd.read_parquet

    def changed(path: Path, **kwargs: Any) -> pd.DataFrame:
        frame = real_read(path, **kwargs)
        path.write_bytes(path.read_bytes() + b"changed")
        return frame

    monkeypatch.setattr(adapter.pd, "read_parquet", changed)
    with pytest.raises(DataReadinessError, match="hash"):
        adapter.read_adjusted_history_unit(context, "unit")


@pytest.mark.parametrize("kind", ["transport", "page", "policy"])
def test_mutation_after_strict_collection_replay_cannot_enter_context(
    tmp_path: Path, collected: tuple[SourcePin, SourcePin], monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    original = adapter.history_archive.load_complete_swing_history_collection

    def changed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        path = (tmp_path / "policy.json" if kind == "policy" else
            next((tmp_path / "archive").rglob("*.bin" if kind == "transport" else "*.gz")))
        path.write_bytes(path.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(adapter.history_archive, "load_complete_swing_history_collection", changed)
    with pytest.raises(DataReadinessError, match="hash"):
        adapter.load_adjusted_history_source(root=tmp_path, plan_authority=collected[0], archive_authority=collected[1])
