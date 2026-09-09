"""Synthetic requirement replay; no real parent or numeric artifacts are opened."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.research_cohort import SwingResearchCohort
from market_predictor.swing.datasets import holding_observation_requirements as requirements
from market_predictor.swing.labels.holding_paths import holding_calendar


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def _sign(value: dict[str, Any], key: str) -> None:
    value[key] = json_sha256({name: item for name, item in value.items() if name != key})


def _bind(fixture: dict[str, Any], *, post_override: dict[str, str] | None = None) -> None:
    root = fixture["root"]
    fixture["memberships"].to_parquet(root / "memberships.parquet", index=False)
    for frame, record in zip(fixture["frames"], fixture["manifest"]["files"], strict=True):
        path = root / "panel" / record["path"]
        path.parent.mkdir(exist_ok=True)
        frame.to_parquet(path, index=False)
        dates = pd.to_datetime(frame.session_date_et)
        record.update(sha256=file_sha256(path), rows=len(frame), securities=frame.security_id.nunique(),
                      sessions=dates.nunique(), first_session=str(dates.min().date()), last_session=str(dates.max().date()))
    raw_payload = {key: value for key, value in fixture["raw_request"].items() if key != "request_sha256"}
    fixture["raw_request"]["request_sha256"] = hashlib.sha256(json.dumps(raw_payload, sort_keys=True).encode()).hexdigest()
    fixture["raw_manifest"]["request_sha256"] = fixture["raw_request"]["request_sha256"]
    _json(root / "raw/_request.json", fixture["raw_request"])
    _json(root / "raw/_manifest.json", fixture["raw_manifest"])
    post = fixture["parent"]["combined_daily_inputs"]["post_collection"]
    post.update(directory="raw", manifest_sha256=file_sha256(root / "raw/_manifest.json"),
                request_file_sha256=file_sha256(root / "raw/_request.json"),
                request_identity_sha256=fixture["raw_request"]["request_sha256"])
    post.update(post_override or {})
    _sign(fixture["parent"], "request_sha256")
    _json(root / "parent.json", fixture["parent"])
    fixture["manifest"]["request_sha256"] = fixture["parent"]["request_sha256"]
    _json(root / "panel/manifest.json", fixture["manifest"])
    source_paths = ["parent.json", "panel/manifest.json", "memberships.parquet",
                    *(f"panel/{row['path']}" for row in fixture["manifest"]["files"])]
    fixture["cohort"]["source_files"] = {name: file_sha256(root / name) for name in source_paths}
    fixture["cohort"]["combined_daily_inputs_sha256"] = json_sha256(fixture["parent"]["combined_daily_inputs"])
    cohort = SwingResearchCohort.model_validate_json(json.dumps(fixture["cohort"]))
    envelope = {"cohort": cohort.model_dump(mode="json"), "cohort_sha256": cohort.sha256(),
                "summary": cohort.summary(), "coverage": {"synthetic": True}}
    _sign(envelope, "audit_sha256")
    _json(root / "cohort.json", envelope)
    preflight = fixture["preflight"]
    preflight["request"]["cohort_sha256"] = cohort.sha256()
    preflight["cohort_sha256"] = cohort.sha256()
    preflight["source_files"] = {**cohort.source_files, "cohort.json": file_sha256(root / "cohort.json")}
    _save_preflight(fixture)


def _save_preflight(fixture: dict[str, Any]) -> None:
    _sign(fixture["preflight"], "audit_sha256")
    _json(fixture["root"] / "preflight.json", fixture["preflight"])


def _frame(days: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.DataFrame({"decision_id": [f"s1:{day}" for day in days], "security_id": "s1", "ticker": "AAA",
                          "sector": "technology", "primary_benchmark": "XLK", "session_date_et": days,
                          "close": "numeric-poison", "future_net_return_10d": "outcome-poison"})
    frame["decision_time_utc"] = swing_prediction_cutoffs(frame.session_date_et)
    return frame


@pytest.fixture
def inventory(tmp_path: Path) -> dict[str, Any]:
    memberships = pd.DataFrame({
        "security_id": ["s1", "s2", "s3", "w1", "s3"], "ticker": ["AAA", "BBB", "CCC", "DDD", "AAA"],
        "sector": "technology", "industry": "software", "market_cap_bucket": "large", "liquidity_bucket": "high",
        "primary_benchmark": "XLK", "universe_snapshot_id": "synthetic", "source": "synthetic",
        "effective_from_utc": pd.Timestamp("2023-01-01T00:00:00Z"),
        "effective_to_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
        "available_at_utc": pd.Timestamp("2022-12-01T00:00:00Z"), "close": "forbidden-numeric",
    })
    memberships.loc[4, "effective_from_utc"] = pd.Timestamp("2024-01-09T14:30:00Z")
    memberships.loc[4, "effective_to_utc"] = pd.Timestamp("2024-01-09T15:00:00Z")
    fixture: dict[str, Any] = {
        "root": tmp_path, "memberships": memberships,
        "frames": [_frame(("2024-01-02", "2024-01-03", "2024-01-31")), _frame(("2024-02-01", "2024-02-02"))],
        "manifest": {"feature_profiles": ["technical_market"], "files": [
            {"path": "january.parquet", "partition_month": "2024-01", "feature_profile": "technical_market"},
            {"path": "february.parquet", "partition_month": "2024-02", "feature_profile": "technical_market"}]},
        "parent": {"decision_start_date": "2024-01-02", "combined_daily_inputs": {"post_collection": {}}},
        "raw_request": {"schema": "swing.daily_history_collection.v1", "source": "alpaca", "price_feed": "sip",
                        "adjustment": "all", "timeframe": "1d", "start_date": "2024-01-02", "end_date": "2024-02-02",
                        "symbols": ["AAA", "BBB", "CCC", "DDD"]},
        "raw_manifest": {"schema": "swing.daily_history_manifest.v1", "artifact_count": 1, "artifacts": [
            {"ticker": "AAA", "price_feed": "sip", "adjustment": "all", "path": "raw/NEVER_OPEN.parquet",
             "sha256": "1" * 64}]},
        "cohort": {"schema_version": "market_predictor.swing_research_cohort", "scope": "retrospective_development_restriction",
                   "price_basis_status": "not_certified_by_cohort", "original_security_ids": ["s1", "s2", "s3"],
                   "inherited_excluded_security_ids": [], "warmup_only_security_ids": ["w1"],
                   "exclusions": [{"security_id": "s3", "tickers": ["CCC"], "reason": "unresolved_holding_identity"}],
                   "maximum_exclusion_bps": 4000, "cap_approval_reference": "synthetic"},
        "preflight": {"scope": "membership_identity_only_not_bar_or_return_admission", "status": "uncovered_holding_identity",
                      "initial_fit": {"uncovered_matured": 1}, "affected_securities": [
                          {"security_id": "s1", "initial_fit_decisions": 1, "first_decision": "2024-01-02",
                           "last_decision": "2024-01-03", "tickers": ["AAA"]},
                          {"security_id": "s2", "initial_fit_decisions": 0}],
                      "request": {"schema": "market_predictor.swing_holding_identity_preflight_request",
                                  "cohort_path": "cohort.json", "parent_manifest_path": "panel/manifest.json",
                                  "membership_path": "memberships.parquet", "decision_start": "2024-01-02",
                                  "snapshot_end": "2024-02-02", "initial_fit_end": "2024-01-17", "horizon_sessions": 10}},
    }
    _bind(fixture)
    return fixture


def _run(fixture: dict[str, Any], **kwargs: Any) -> requirements.PreparedHoldingObservations:
    arguments = {"preflight_path": Path("preflight.json"), "preflight_sha256": fixture["preflight"]["audit_sha256"],
                 "parent_request_path": Path("parent.json"), "expected_decisions": 1, "expected_securities": 1, **kwargs}
    return requirements.prepare_holding_observation_requirements(fixture["root"], **arguments)


def test_exact_initial_fit_maturity_and_full_excluded_memberships(inventory: dict[str, Any]) -> None:
    before = {path: file_sha256(path) for path in inventory["root"].rglob("*") if path.is_file()}
    prepared = _run(inventory)
    assert prepared.numeric_end == date(2024, 1, 17)
    assert prepared.decisions.decision_id.tolist() == ["s1:2024-01-02"]
    assert prepared.decisions.exit_session_date_et.tolist() == [date(2024, 1, 17)]
    assert list(prepared.decisions) == [*requirements.IDENTITY_COLUMNS, "exit_session_date_et"]
    assert prepared.requirements == [{"security_id": "s1", "ticker": "AAA",
                                      "sessions": list(holding_calendar(date(2024, 1, 3), date(2024, 1, 17)))}]
    assert len(prepared.requirements[0]["sessions"]) == 10
    assert len(prepared.memberships) == 5
    assert set(prepared.memberships.security_id) == {"s1", "s2", "s3", "w1"}
    assert set(prepared.memberships) == set(requirements.MEMBERSHIP_COLUMNS)
    assert prepared.raw_manifest == inventory["raw_manifest"]
    assert set(prepared.bound_files) == {*inventory["preflight"]["source_files"], "preflight.json",
                                        "raw/_request.json", "raw/_manifest.json"}
    assert {path: file_sha256(path) for path in before} == before


def test_projects_identity_only_and_skips_irrelevant_parent_month(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = requirements.pq.ParquetFile
    reads: list[str] = []

    class ProjectionSpy:
        def __init__(self, path: Path) -> None:
            assert path.name != "february.parquet"
            assert path.name in {"january.parquet", "memberships.parquet"}
            reads.append(path.name)
            self.source = original(path)
            self.metadata, self.schema_arrow = self.source.metadata, self.source.schema_arrow
            self.expected = requirements.MEMBERSHIP_COLUMNS if path.name == "memberships.parquet" else requirements.IDENTITY_COLUMNS

        def __enter__(self) -> ProjectionSpy:
            return self

        def __exit__(self, *args: Any) -> None:
            self.source.close()

        def read(self, *, columns: list[str], use_threads: bool) -> Any:
            assert columns == list(self.expected)
            assert use_threads is False
            return self.source.read(columns=columns, use_threads=use_threads)

    monkeypatch.setattr(requirements.pq, "ParquetFile", ProjectionSpy)
    _run(inventory)
    assert reads == ["memberships.parquet", "january.parquet"]


@pytest.mark.parametrize("relative", ["preflight.json", "cohort.json", "parent.json", "memberships.parquet",
                                       "panel/january.parquet", "panel/february.parquet", "raw/_request.json", "raw/_manifest.json"])
def test_source_tamper_before_identity_projection(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch, relative: str) -> None:
    path = inventory["root"] / relative
    if relative == "preflight.json":
        preflight = dict(inventory["preflight"], status="tampered")
        _json(path, preflight)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(requirements, "_projection", lambda *args: pytest.fail("projection before verification"))
    with pytest.raises(DataReadinessError):
        _run(inventory)


def test_canonical_audit_sha_is_not_file_sha(inventory: dict[str, Any]) -> None:
    with pytest.raises(DataReadinessError, match="canonical audit_sha256"):
        _run(inventory, preflight_sha256=file_sha256(inventory["root"] / "preflight.json"))


def test_cohort_owner_reverifies_source_root(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = requirements.load_swing_research_cohort
    calls: list[Path] = []

    def load(path: Path, *, source_root: Path) -> SwingResearchCohort:
        assert source_root == inventory["root"].resolve()
        calls.append(path)
        return original(path, source_root=source_root)

    monkeypatch.setattr(requirements, "load_swing_research_cohort", load)
    _run(inventory)
    assert calls == [inventory["root"] / "cohort.json"]


@pytest.mark.parametrize("payload", [b'{"audit_sha256":"a","audit_sha256":"b"}', b'{"number":NaN}'])
def test_strict_metadata_parser(inventory: dict[str, Any], payload: bytes) -> None:
    (inventory["root"] / "preflight.json").write_bytes(payload)
    with pytest.raises(DataReadinessError, match="duplicate|non-finite"):
        _run(inventory)


def test_source_escape_rejected(inventory: dict[str, Any]) -> None:
    inventory["preflight"]["source_files"]["../outside.json"] = "0" * 64
    _save_preflight(inventory)
    with pytest.raises(DataReadinessError, match="escapes authority"):
        _run(inventory)


def test_parent_request_must_be_in_preflight(inventory: dict[str, Any]) -> None:
    _json(inventory["root"] / "unbound.json", inventory["parent"])
    with pytest.raises(DataReadinessError, match="not bound"):
        _run(inventory, parent_request_path=Path("unbound.json"))


@pytest.mark.parametrize("field,value", [("source", "other"), ("price_feed", "iex"), ("adjustment", "raw"),
                                        ("timeframe", "5m"), ("start_date", "2024-01-03"), ("end_date", "2024-02-01")])
def test_raw_request_identity_semantics_fail_even_when_rebound(inventory: dict[str, Any], field: str, value: str) -> None:
    inventory["raw_request"][field] = value
    _bind(inventory)
    with pytest.raises(DataReadinessError, match="raw source"):
        _run(inventory)


@pytest.mark.parametrize("field", ["request_file_sha256", "manifest_sha256", "request_identity_sha256"])
def test_all_three_raw_source_pins_are_verified(inventory: dict[str, Any], field: str) -> None:
    _bind(inventory, post_override={field: "0" * 64})
    with pytest.raises(DataReadinessError, match="source hash|daily collection request hash"):
        _run(inventory)


def test_raw_collection_hash_uses_original_serialization(inventory: dict[str, Any]) -> None:
    request = inventory["raw_request"]
    payload = {key: value for key, value in request.items() if key != "request_sha256"}
    assert request["request_sha256"] == hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert request["request_sha256"] != json_sha256(payload)
    assert _run(inventory).raw_manifest["request_sha256"] == request["request_sha256"]


@pytest.mark.parametrize("count,security_count", [(2, 1), (1, 2), (True, 1), (1, 0)])
def test_expected_counts_fail_closed(inventory: dict[str, Any], count: Any, security_count: int) -> None:
    with pytest.raises(DataReadinessError):
        _run(inventory, expected_decisions=count, expected_securities=security_count)


def test_per_security_counts_recomputed_after_maturity(inventory: dict[str, Any]) -> None:
    inventory["preflight"]["affected_securities"][0]["initial_fit_decisions"] = 2
    inventory["preflight"]["initial_fit"]["uncovered_matured"] = 2
    _save_preflight(inventory)
    with pytest.raises(DataReadinessError, match="per-security"):
        _run(inventory, expected_decisions=2)


@pytest.mark.parametrize("removed", ["s3", "w1"])
def test_missing_excluded_or_warmup_membership_rejected(inventory: dict[str, Any], removed: str) -> None:
    inventory["memberships"] = inventory["memberships"].loc[inventory["memberships"].security_id.ne(removed)]
    _bind(inventory)
    with pytest.raises(DataReadinessError, match="full, unfiltered"):
        _run(inventory)


@pytest.mark.parametrize("kind", ["too_many", "duplicate_month", "partition_hash"])
def test_partition_inventory_bounds(inventory: dict[str, Any], kind: str) -> None:
    root = inventory["root"]
    manifest = inventory["manifest"]
    if kind == "too_many":
        manifest["files"] = [manifest["files"][0]] * 241
    elif kind == "duplicate_month":
        manifest["files"][1]["partition_month"] = "2024-01"
    else:
        manifest["files"][0]["sha256"] = "0" * 64
    _json(root / "panel/manifest.json", manifest)
    # Rebind the envelope without rebuilding the mutated inventory.
    cohort = dict(inventory["cohort"])
    cohort["source_files"]["panel/manifest.json"] = file_sha256(root / "panel/manifest.json")
    parsed = SwingResearchCohort.model_validate_json(json.dumps(cohort))
    envelope = {"cohort": parsed.model_dump(mode="json"), "cohort_sha256": parsed.sha256(),
                "summary": parsed.summary(), "coverage": {}}
    _sign(envelope, "audit_sha256")
    _json(root / "cohort.json", envelope)
    inventory["preflight"].update(cohort_sha256=parsed.sha256(), source_files={
        **parsed.source_files, "cohort.json": file_sha256(root / "cohort.json")})
    inventory["preflight"]["request"]["cohort_sha256"] = parsed.sha256()
    _save_preflight(inventory)
    with pytest.raises(DataReadinessError, match="partition"):
        _run(inventory)


def test_noncontiguous_union_does_not_compress_calendar(inventory: dict[str, Any]) -> None:
    inventory["frames"] = [_frame(("2024-01-02",)), _frame(("2024-02-01",))]
    competitor = inventory["memberships"].iloc[[-1]].copy()
    competitor["effective_from_utc"] = pd.Timestamp("2024-02-08T14:30:00Z")
    competitor["effective_to_utc"] = pd.Timestamp("2024-02-08T15:00:00Z")
    inventory["memberships"] = pd.concat([inventory["memberships"], competitor], ignore_index=True)
    inventory["preflight"]["request"].update(initial_fit_end="2024-02-16", snapshot_end="2024-03-01")
    inventory["raw_request"]["end_date"] = "2024-03-01"
    inventory["preflight"]["affected_securities"][0].update(initial_fit_decisions=2, last_decision="2024-02-01")
    inventory["preflight"]["initial_fit"]["uncovered_matured"] = 2
    _bind(inventory)
    prepared = _run(inventory, expected_decisions=2)
    expected = [*holding_calendar(date(2024, 1, 3), date(2024, 1, 17)),
                *holding_calendar(date(2024, 2, 2), date(2024, 2, 15))]
    assert prepared.requirements[0]["sessions"] == expected
    assert len(expected) == 20
    assert date(2024, 1, 18) not in expected


def test_metadata_size_limit(inventory: dict[str, Any]) -> None:
    (inventory["root"] / "preflight.json").write_bytes(b" " * (8 * 1024**2 + 1))
    with pytest.raises(DataReadinessError, match="bounded size"):
        _run(inventory)


def test_race_after_projection_rechecked(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = requirements.inspect_holding_membership_windows

    def poison(*args: Any, **kwargs: Any) -> pd.DataFrame:
        result = original(*args, **kwargs)
        path = inventory["root"] / "raw/_manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(requirements, "inspect_holding_membership_windows", poison)
    with pytest.raises(DataReadinessError, match="source hash"):
        _run(inventory)
