"""Offline request preparation against tiny, actual hash-bound source artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.research.swing_transfer_sources import prepare_transfer_replay_request

_THANKSGIVING = ["2023-11-24", "2023-11-27", "2023-11-28", "2023-11-29", "2023-11-30",
                 "2023-12-01", "2023-12-04", "2023-12-05", "2023-12-06", "2023-12-07"]
_DST = ["2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14",
        "2024-03-15", "2024-03-18", "2024-03-19", "2024-03-20", "2024-03-21"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


@dataclass
class TransferFixture:
    root: Path
    frames: dict[str, pd.DataFrame]
    control: dict[str, Any]
    raw: dict[str, Any]
    config: dict[str, Any]

    @property
    def config_path(self) -> Path:
        return self.root / "replay.toml"

    def publish(self) -> Path:
        """Rebind deliberately changed fixture content, except in byte-tamper tests."""
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("membership", "relations", "bars"):
            self.frames[name].to_parquet(self.root / f"{name}.parquet", index=False)
        partitions: dict[str, str] = {}
        selected = self.frames["selected"]
        for month, frame in selected.groupby(selected["session_date_et"].str[:7]):
            relative = f"panel/feature_profile=technical_market/month={month}/part.parquet"
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            partitions[relative] = _sha(path)
        self.control["bound_files"] = partitions
        self.raw["artifacts"][0]["sha256"] = _sha(self.root / "bars.parquet")
        _json(self.root / "control.json", self.control)
        _json(self.root / "raw.json", self.raw)
        for key, filename in (("control_receipt", "control.json"), ("membership", "membership.parquet"),
                              ("sec_relations", "relations.parquet"), ("raw_manifest", "raw.json")):
            self.config[f"{key}_sha256"] = _sha(self.root / filename)
        self.write_config()
        return self.config_path

    def write_config(self) -> None:
        # JSON literals for this fixture's strings, integers and lists are also TOML literals.
        lines = [f"{key} = {json.dumps(value)}" for key, value in self.config.items() if key != "securities"]
        for security in self.config["securities"]:
            lines.append("\n[[securities]]")
            lines.extend(f"{key} = {json.dumps(value)}" for key, value in security.items())
        self.config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def prepare(self) -> dict[str, Any]:
        return prepare_transfer_replay_request(self.root, self.config_path)


def _fixture(root: Path, decisions: tuple[str, ...] = ("2023-11-22",)) -> TransferFixture:
    calendar = xcals.get_calendar("XNYS")
    start, end = date.fromisoformat(decisions[0]), date.fromisoformat(decisions[-1])
    sessions = calendar.sessions_in_range(start - timedelta(days=5), end + timedelta(days=45))
    selected = pd.DataFrame([
        {"decision_id": f"selected-{day}", "security_id": "security:one", "session_date_et": day,
         "decision_time_utc": pd.Timestamp(f"{day}T22:00:00Z"), "unused_feature": 999.0}
        for day in decisions
    ] + [{"decision_id": "unselected-other-security", "security_id": "security:other", "session_date_et": decisions[0],
          "decision_time_utc": pd.Timestamp(f"{decisions[0]}T22:00:00Z"), "unused_feature": -999.0}])
    membership = pd.DataFrame({
        "security_id": ["security:one"], "ticker": ["ONE"],
        "effective_from_utc": pd.to_datetime(["2019-07-09T00:00:00Z"], utc=True),
        # Membership ends before the first outcome bar; outcome history must survive removal.
        "effective_to_utc": pd.to_datetime([f"{end + timedelta(days=1)}T00:00:00Z"], utc=True),
    })
    bars = pd.DataFrame({
        "ticker": "ONE", "timeframe": "1d", "bar_start_utc": [calendar.session_open(day) for day in sessions],
        "bar_end_utc": [calendar.session_close(day) for day in sessions],
        "available_at_utc": [calendar.session_close(day) + pd.Timedelta(minutes=15) for day in sessions],
        "open": [100.0 + i for i in range(len(sessions))], "high": [102.0 + i for i in range(len(sessions))],
        "low": [99.0 + i for i in range(len(sessions))], "close": [101.0 + i for i in range(len(sessions))],
        "volume": [1000 + i for i in range(len(sessions))], "price_feed": "sip", "adjustment": "all",
    })
    return TransferFixture(
        root, {"selected": selected, "membership": membership, "relations": membership.assign(sec_cik="0000000001"), "bars": bars},
        {"schema_version": "market_predictor.swing_accounting_control.v1", "status": "blocked",
         "validation_or_test_outcomes_read": False, "scope": "initial_fit_deterministic_control_not_out_of_sample",
         "selected_decision_ids": [f"selected-{day}" for day in decisions], "decision_sessions": list(decisions),
         "valuation_sessions": [day.date().isoformat() for day in sessions]},
        {"artifacts": [{"ticker": "ONE", "price_feed": "sip", "adjustment": "all", "path": "bars.parquet", "sha256": ""}]},
        {"schema_version": "market_predictor.swing_transfer_replay", "control_receipt": "control.json",
         "membership_path": "membership.parquet", "sec_relations_path": "relations.parquet", "raw_manifest_path": "raw.json",
         "horizon_sessions": 10, "maximum_pages_per_ticker": 2, "expected_security_count": 1, "expected_decision_count": len(decisions),
         "securities": [{"ticker": "ONE", "security_id": "security:one", "sec_cik": "0000000001", "decision_dates": list(decisions)}]},
    )


@pytest.fixture
def sources(tmp_path: Path) -> TransferFixture:
    fixture = _fixture(tmp_path)
    fixture.publish()
    return fixture


def _required_index(sources: TransferFixture, day: str = "2023-11-24") -> int:
    dates = sources.frames["bars"]["bar_start_utc"].dt.date.astype(str)
    return int(dates[dates.eq(day)].index[0])


def _snapshot(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("decision,expected,start,end", [
    ("2023-11-22", _THANKSGIVING, "2023-11-24T00:00:00-05:00", "2023-12-07T23:59:59.999999-05:00"),
    ("2024-03-07", _DST, "2024-03-08T00:00:00-05:00", "2024-03-21T23:59:59.999999-04:00"),
])
def test_exact_ten_sessions_asof_and_exchange_boundaries(
    tmp_path: Path, decision: str, expected: list[str], start: str, end: str,
) -> None:
    fixture = _fixture(tmp_path, (decision,))
    fixture.publish()
    before = _snapshot(tmp_path)
    request = fixture.prepare()
    unit, = request["units"]
    assert unit["required_sessions"] == expected
    assert len(unit["retained_rows"]) == 10
    assert unit["decisions"] == [{"decision_id": f"selected-{decision}", "session_date_et": decision,
                                  "decision_time_utc": f"{decision}T22:00:00+00:00"}]
    assert (unit["ticker"], unit["security_id"], unit["sec_cik"]) == ("ONE", "security:one", "0000000001")
    assert unit["parameters"] == {"symbols": "ONE", "timeframe": "1Day", "feed": "sip", "adjustment": "all",
                                  "sort": "asc", "limit": 10000, "start": start, "end": end, "asof": decision}
    assert request["maximum_pages_per_ticker"] == 2
    assert request["retry_budget_per_page"] == 1
    assert request["identity_admission"] is request["accounting_eligible"] is False
    assert unit["unit_sha256"] == json_sha256({key: value for key, value in unit.items() if key != "unit_sha256"})
    assert all(_sha(tmp_path / name) == digest for name, digest in request["bound_files"].items())
    assert set(request["bound_files"]) == set(before)
    assert _snapshot(tmp_path) == before
    assert fixture.prepare() == request
    for row in unit["retained_rows"]:
        original = fixture.frames["bars"].iloc[_required_index(fixture, row["session_date_et"])]
        assert {name: row[name] for name in ("open", "high", "low", "close", "volume")} == {
            name: float(original[name]) for name in ("open", "high", "low", "close", "volume")
        }


def test_cross_month_overlapping_decisions_keep_exact_ids_and_union(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, ("2023-11-30", "2023-12-01"))
    fixture.frames["selected"] = fixture.frames["selected"].iloc[::-1]
    fixture.publish()
    unit, = fixture.prepare()["units"]
    assert [row["decision_id"] for row in unit["decisions"]] == ["selected-2023-11-30", "selected-2023-12-01"]
    assert unit["required_sessions"] == ["2023-12-01", "2023-12-04", "2023-12-05", "2023-12-06", "2023-12-07",
                                          "2023-12-08", "2023-12-11", "2023-12-12", "2023-12-13", "2023-12-14", "2023-12-15"]
    assert unit["parameters"]["asof"] == "2023-11-30"


@pytest.mark.parametrize("field,value", [
    ("schema_version", "other"), ("status", "passed"), ("scope", "validation"),
    ("validation_or_test_outcomes_read", True), ("validation_or_test_outcomes_read", 0),
])
def test_initial_fit_control_guard(sources: TransferFixture, field: str, value: Any) -> None:
    sources.control[field] = value
    sources.publish()
    with pytest.raises(DataReadinessError, match="initial-fit blocked control"):
        sources.prepare()


@pytest.mark.parametrize("filename", ["control.json", "raw.json", "membership.parquet", "relations.parquet", "bars.parquet",
                                      "panel/feature_profile=technical_market/month=2023-11/part.parquet"])
def test_tampered_bound_bytes_fail_without_modifying_retained_sources(sources: TransferFixture, filename: str) -> None:
    path = sources.root / filename
    path.write_bytes(path.read_bytes() + b"tampered")
    before = _snapshot(sources.root)
    with pytest.raises(DataReadinessError, match="hash mismatch"):
        sources.prepare()
    assert _snapshot(sources.root) == before


@pytest.mark.parametrize("field,value", [("decision_id", "not-selected"), ("security_id", "security:replacement"),
                                        ("session_date_et", "2023-11-21")])
def test_selected_rows_require_exact_frozen_identity_and_date(sources: TransferFixture, field: str, value: str) -> None:
    sources.frames["selected"].loc[0, field] = value
    sources.publish()
    with pytest.raises(DataReadinessError, match="exact frozen selected decisions"):
        sources.prepare()


@pytest.mark.parametrize("replacement", [[], ["some-other-id"]])
def test_selected_id_must_be_in_control(sources: TransferFixture, replacement: list[str]) -> None:
    sources.control["selected_decision_ids"] = replacement
    sources.publish()
    with pytest.raises(DataReadinessError, match="exact frozen selected decisions"):
        sources.prepare()


@pytest.mark.parametrize("invalid_id", [None, "", "   ", 7])
def test_selected_decision_ids_must_be_nonempty_strings_without_coercion(sources: TransferFixture, invalid_id: Any) -> None:
    selected = sources.frames["selected"].iloc[[0]].copy()
    selected["decision_id"] = invalid_id
    sources.frames["selected"] = selected
    sources.control["selected_decision_ids"] = [invalid_id]
    sources.publish()
    with pytest.raises(DataReadinessError):
        sources.prepare()


@pytest.mark.parametrize("clock", ["2023-11-23T22:00:00Z", "2023-11-22T01:00:00Z", "2023-11-22T22:00:00"])
def test_wrong_or_naive_decision_clock_is_rejected(sources: TransferFixture, clock: str) -> None:
    sources.frames["selected"]["decision_time_utc"] = clock
    sources.publish()
    with pytest.raises(DataReadinessError, match="decision clock"):
        sources.prepare()


@pytest.mark.parametrize("frame,field,value", [
    ("membership", "security_id", "wrong"), ("membership", "ticker", "WRONG"),
    ("relations", "security_id", "wrong"), ("relations", "ticker", "WRONG"), ("relations", "sec_cik", "0000000002"),
])
def test_wrong_membership_or_issuer_anchor(sources: TransferFixture, frame: str, field: str, value: str) -> None:
    sources.frames[frame].loc[0, field] = value
    sources.publish()
    with pytest.raises(DataReadinessError, match="anchor is missing or ambiguous"):
        sources.prepare()


@pytest.mark.parametrize("frame", ["membership", "relations"])
@pytest.mark.parametrize("field,clock", [("effective_to_utc", "2023-11-22T22:00:00Z"),
                                       ("effective_from_utc", "2023-11-22T22:00:01Z")])
def test_anchor_intervals_are_half_open_at_decision(
    sources: TransferFixture, frame: str, field: str, clock: str,
) -> None:
    sources.frames[frame].loc[0, field] = pd.Timestamp(clock)
    sources.publish()
    with pytest.raises(DataReadinessError, match="anchor is missing or ambiguous"):
        sources.prepare()


@pytest.mark.parametrize("frame", ["membership", "relations", "selected", "bars"])
def test_duplicate_relevant_rows_are_rejected(sources: TransferFixture, frame: str) -> None:
    index = _required_index(sources) if frame == "bars" else 0
    sources.frames[frame] = pd.concat([sources.frames[frame], sources.frames[frame].iloc[[index]]], ignore_index=True)
    sources.publish()
    with pytest.raises(DataReadinessError, match="ambiguous|exact frozen|duplicated sessions"):
        sources.prepare()


@pytest.mark.parametrize("day", ["2023-11-23", "2023-11-25"])
def test_holiday_and_weekend_decisions_are_not_exchange_sessions(tmp_path: Path, day: str) -> None:
    fixture = _fixture(tmp_path, (day,))
    fixture.publish()
    with pytest.raises(DataReadinessError, match="frozen exchange calendar"):
        fixture.prepare()


def test_decision_must_belong_to_control_calendar(sources: TransferFixture) -> None:
    sources.control["decision_sessions"] = ["2023-11-21"]
    sources.publish()
    with pytest.raises(DataReadinessError, match="frozen exchange calendar"):
        sources.prepare()


@pytest.mark.parametrize("day", ["2023-11-24", "2023-11-29", "2023-12-07"])
def test_entire_holding_window_must_remain_within_initial_fit(sources: TransferFixture, day: str) -> None:
    sources.control["valuation_sessions"].remove(day)
    sources.publish()
    with pytest.raises(DataReadinessError, match="within initial fit"):
        sources.prepare()


@pytest.mark.parametrize("defect", ["missing", "zero_volume", "negative_volume", "nan_volume", "nan_close", "zero_close", "invalid_high"])
@pytest.mark.parametrize("day", ["2023-11-24", "2023-12-07"])
def test_missing_or_unusable_bars_never_skip_to_later_observations(sources: TransferFixture, defect: str, day: str) -> None:
    index = _required_index(sources, day)
    if defect == "missing":
        sources.frames["bars"] = sources.frames["bars"].drop(index=index)
    else:
        field, value = {"zero_volume": ("volume", 0), "negative_volume": ("volume", -1), "nan_volume": ("volume", float("nan")),
                        "nan_close": ("close", float("nan")), "zero_close": ("close", 0), "invalid_high": ("high", 0)}[defect]
        sources.frames["bars"].loc[index, field] = value
    sources.publish()
    before = _snapshot(sources.root)
    with pytest.raises(DataReadinessError, match="price-complete population"):
        sources.prepare()
    assert _snapshot(sources.root) == before


@pytest.mark.parametrize("field,value", [("ticker", "OTHER"), ("timeframe", "1h"), ("price_feed", "iex"), ("adjustment", "raw")])
def test_retained_row_identity_must_match_request(sources: TransferFixture, field: str, value: str) -> None:
    sources.frames["bars"].loc[_required_index(sources), field] = value
    sources.publish()
    with pytest.raises(DataReadinessError, match="conflicting identity"):
        sources.prepare()


@pytest.mark.parametrize("field,value", [("ticker", "OTHER"), ("price_feed", "iex"), ("adjustment", "raw")])
def test_retained_manifest_identity_must_match_request(sources: TransferFixture, field: str, value: str) -> None:
    sources.raw["artifacts"][0][field] = value
    sources.publish()
    with pytest.raises(DataReadinessError, match="source identity or feed"):
        sources.prepare()


@pytest.mark.parametrize("field,clock", [("bar_start_utc", "2023-11-24T15:00:00Z"),
                                       ("bar_end_utc", "2023-11-24T21:00:00Z"),
                                       ("available_at_utc", "2023-11-24T17:59:00Z")])
def test_early_close_and_daily_bar_clock_contract(sources: TransferFixture, field: str, clock: str) -> None:
    sources.frames["bars"].loc[_required_index(sources), field] = pd.Timestamp(clock)
    sources.publish()
    with pytest.raises(DataReadinessError, match="session/availability mismatch"):
        sources.prepare()


def test_unneeded_old_or_future_observations_are_not_replaced_or_validated(sources: TransferFixture) -> None:
    before = sources.prepare()["units"][0]
    dates = sources.frames["bars"]["bar_start_utc"].dt.date.astype(str)
    unused = ~dates.isin(_THANKSGIVING)
    sources.frames["bars"].loc[unused, "volume"] = 0
    sources.frames["bars"].loc[unused, "close"] = float("nan")
    sources.publish()
    snapshot = _snapshot(sources.root)
    assert sources.prepare()["units"][0] == before
    assert _snapshot(sources.root) == snapshot


@pytest.mark.parametrize("field,value", [("sec_cik", "0000000002"), ("security_id", "other-security")])
def test_configuration_identity_is_not_authority(sources: TransferFixture, field: str, value: str) -> None:
    sources.config["securities"][0][field] = value
    sources.write_config()
    with pytest.raises(DataReadinessError, match="anchor|exact frozen"):
        sources.prepare()


@pytest.mark.parametrize("field,value", [("horizon_sessions", 9), ("maximum_pages_per_ticker", 4),
                                       ("expected_security_count", 2), ("expected_decision_count", 2)])
def test_config_counts_and_bounds_are_enforced(sources: TransferFixture, field: str, value: int) -> None:
    sources.config[field] = value
    sources.write_config()
    with pytest.raises((DataReadinessError, ValidationError)):
        sources.prepare()
