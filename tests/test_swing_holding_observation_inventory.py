"""End-to-end diagnostic publication uses synthetic, hash-bound source fixtures."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.swing.datasets import holding_observation_inventory as audit
from market_predictor.swing.labels.holding_paths import holding_calendar
from tests.test_swing_holding_observation_requirements import _bind
from tests.test_swing_holding_observation_requirements import inventory as requirements_inventory

upstream = requirements_inventory


@pytest.fixture
def inventory(upstream: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = upstream["root"]
    days = holding_calendar(date(2024, 1, 3), date(2024, 1, 17))
    calendar = xcals.get_calendar("XNYS")
    bars = pd.DataFrame({"ticker": "AAA", "timeframe": "1d", "price_feed": "sip", "adjustment": "all",
        "bar_start_utc": [calendar.session_open(pd.Timestamp(day)) for day in days],
        "bar_end_utc": [calendar.session_close(pd.Timestamp(day)) for day in days],
        "available_at_utc": [calendar.session_close(pd.Timestamp(day)) for day in days],
        "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, "volume": 1000.0})
    upstream.update(bars=bars, config=root / "inventory.toml", output=root / "diagnostic")
    _refresh(upstream)
    monkeypatch.setenv("MARKET_PREDICTOR_HEAVY_JOB_RUNTIME_DIR", str(root / "runtime"))
    monkeypatch.setattr(audit, "heavy_job_runtime_dir", lambda: root / "runtime")
    return upstream


def _refresh(fixture: dict[str, Any]) -> None:
    path = fixture["root"] / "raw/bars.parquet"
    fixture["bars"].to_parquet(path, index=False)
    records = fixture["raw_manifest"]["artifacts"]
    if records:
        records[0].update(path="raw/bars.parquet", sha256=file_sha256(path))
    _bind(fixture)
    fixture["config"].write_text(
        'schema = "market_predictor.swing_holding_observation_inventory_request"\n'
        'preflight_path = "preflight.json"\n'
        f'preflight_sha256 = "{fixture["preflight"]["audit_sha256"]}"\n'
        'parent_request_path = "parent.json"\nexpected_decisions = 1\nexpected_securities = 1\n', encoding="utf-8")


def _run(fixture: dict[str, Any], expected: str | None = None) -> dict[str, Any]:
    return audit.run_holding_observation_inventory(
        fixture["root"], fixture["config"], fixture["output"], expected_audit_sha256=expected,
    )


def test_publication_replay_preserves_unresolved_owner(inventory: dict[str, Any]) -> None:
    before = {path: file_sha256(path) for path in inventory["root"].rglob("*") if path.is_file()}
    report = _run(inventory)
    assert report["status"] == "diagnostic_complete"
    assert report["decisions"] == report["securities"] == 1
    assert report["required_security_ticker_sessions"] == 10
    assert report["counts"] == {"observation_valid": 10, "membership_supported": 9, "ownership_unresolved": 1}
    observations = pd.read_parquet(inventory["output"] / report["cases"][0]["output"])
    assert observations.loc[observations.ownership_status.eq("ownership_unresolved"), "security_id"].isna().all()
    assert observations.requested_security_id.eq("s1").all()
    assert {path: file_sha256(path) for path in before} == before
    assert _run(inventory, report["audit_sha256"]) == report
    for field in ("heldout_numeric_rows_read", "materialization_eligible", "accounting_eligible", "promotion_eligible"):
        assert report[field] is False


@pytest.mark.parametrize("kind", ["absent_artifact", "absent_row", "invalid_value", "bad_clock", "duplicate"])
def test_failure_dimensions_remain_explicit(inventory: dict[str, Any], kind: str) -> None:
    if kind == "absent_artifact":
        inventory["raw_manifest"].update(artifacts=[], artifact_count=0)
    elif kind == "absent_row":
        inventory["bars"] = inventory["bars"].iloc[1:].copy()
    elif kind == "invalid_value":
        inventory["bars"].loc[0, "close"] = 0.0
    elif kind == "bad_clock":
        inventory["bars"].loc[0, "bar_end_utc"] += pd.Timedelta(minutes=1)
    else:
        inventory["bars"] = pd.concat([inventory["bars"], inventory["bars"].iloc[:1]], ignore_index=True)
    _refresh(inventory)
    report = _run(inventory)
    counts = report["counts"]
    assert counts["ownership_unresolved"] == 1
    if kind in ("bad_clock", "duplicate"):
        assert counts["source_error"] == 10
        assert "observation_missing" not in counts
        assert report["status"] == "diagnostic_complete_with_source_errors"
    elif kind == "invalid_value":
        assert counts["observation_invalid"] == 1
        assert counts["observation_valid"] == 9
    else:
        assert counts["observation_missing"] == (10 if kind == "absent_artifact" else 1)


@pytest.mark.parametrize("target", ["raw", "published", "implementation", "manifest"])
def test_replay_tamper_fails(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    report = _run(inventory)
    if target == "raw":
        (inventory["root"] / "raw/bars.parquet").write_bytes(b"corrupt")
    elif target == "published":
        (inventory["output"] / report["cases"][0]["output"]).write_bytes(b"corrupt")
    elif target == "implementation":
        monkeypatch.setattr(audit, "_implementation_files", lambda: {"changed": "0" * 64})
    else:
        path = inventory["output"] / "_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["accounting_eligible"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataReadinessError):
        _run(inventory, report["audit_sha256"])


@pytest.mark.parametrize("field", ["outputs", "source_files", "counts"])
def test_rehashed_report_tamper_rejected_by_independent_pin(inventory: dict[str, Any], field: str) -> None:
    report = _run(inventory)
    expected = report.pop("audit_sha256")
    report[field] = {}
    report["audit_sha256"] = json_sha256(report)
    (inventory["output"] / "_manifest.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="replay identity differs"):
        _run(inventory, expected)


def test_replay_requires_external_pin(inventory: dict[str, Any]) -> None:
    _run(inventory)
    with pytest.raises(DataReadinessError, match="independently retained"):
        _run(inventory)


def test_lease_precedes_preparation(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("inputs opened while another heavy job owns the workspace")
    monkeypatch.setattr(audit, "prepare_holding_observation_requirements", forbidden)
    with heavy_job_lease("other", runtime_dir=inventory["root"] / "runtime"):
        with pytest.raises(HeavyJobBusyError):
            _run(inventory)


def test_publication_failure_does_not_publish_partial_inventory(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = pd.DataFrame.to_parquet
    calls = 0

    def fail_second(self: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic write failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_second)
    with pytest.raises(DataReadinessError, match="synthetic write failure"):
        _run(inventory)
    assert not inventory["output"].exists()
    assert not list(inventory["root"].glob(".diagnostic.*.tmp"))


def test_input_changed_during_scan_prevents_publication(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = audit.read_required_holding_observations

    def mutate(path: Path | None, **kwargs: Any) -> pd.DataFrame:
        result = original(path, **kwargs)
        assert path is not None
        path.write_bytes(b"changed during scan")
        return result
    monkeypatch.setattr(audit, "read_required_holding_observations", mutate)
    with pytest.raises(DataReadinessError, match="source hash differs"):
        _run(inventory)
    assert not inventory["output"].exists()


def test_one_corrupt_ticker_does_not_stop_other_observations(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    prepare = audit.prepare_holding_observation_requirements
    other = inventory["root"] / "raw/other.parquet"
    inventory["bars"].assign(ticker="BBB").to_parquet(other, index=False)

    def with_second_requirement(*args: Any, **kwargs: Any) -> Any:
        prepared = prepare(*args, **kwargs)
        extra = {**prepared.requirements[0], "ticker": "BBB", "security_id": "s2"}
        manifest = {**prepared.raw_manifest, "artifacts": [*prepared.raw_manifest["artifacts"], {
            "ticker": "BBB", "path": "raw/other.parquet", "sha256": file_sha256(other),
            "price_feed": "sip", "adjustment": "all",
        }]}
        return replace(prepared, requirements=[*prepared.requirements, extra], raw_manifest=manifest)

    monkeypatch.setattr(audit, "prepare_holding_observation_requirements", with_second_requirement)
    inventory["bars"].loc[0, "bar_end_utc"] += pd.Timedelta(minutes=1)
    _refresh(inventory)
    report = _run(inventory)
    assert report["counts"]["source_error"] == 10
    assert report["counts"]["observation_valid"] == 10
    assert report["cases"][1]["ticker"] == "BBB"
    assert report["cases"][1]["error"] is None
