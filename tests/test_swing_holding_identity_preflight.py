"""Hash-bound synthetic integration tests; real histories are never read."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.research import swing_holding_identity_preflight as audit
from market_predictor.swing.contracts.research_cohort import SwingResearchCohort


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8")


def _config(fixture: dict[str, Any]) -> None:
    fixture["config"].write_text(
        "\n".join(f"{key} = {json.dumps(value)}" for key, value in fixture["request"].items()),
        encoding="utf-8",
    )


def _bind(fixture: dict[str, Any]) -> None:
    root = fixture["root"]
    _json(root / "parent.json", {"combined_daily_inputs": fixture["combined"]})
    _json(root / "panel" / "manifest.json", fixture["manifest"])
    cohort_values = fixture["cohort_values"]
    cohort_values["source_files"] = {
        name: file_sha256(root / name) for name in (
            "parent.json", "membership.parquet", "panel/manifest.json", "panel/january.parquet", "panel/february.parquet",
        )
    }
    cohort = SwingResearchCohort.model_validate_json(json.dumps(cohort_values))
    envelope = {
        "cohort": cohort.model_dump(mode="json"), "cohort_sha256": cohort.sha256(),
        "summary": cohort.summary(), "coverage": {"synthetic_fixture": True},
    }
    envelope["audit_sha256"] = json_sha256(envelope)
    _json(root / "cohort.json", envelope)
    fixture["request"]["cohort_sha256"] = cohort.sha256()
    _config(fixture)


def _write_partition(fixture: dict[str, Any], index: int, frame: pd.DataFrame) -> None:
    record = fixture["manifest"]["files"][index]
    path = fixture["root"] / "panel" / record["path"]
    frame.to_parquet(path, index=False)
    dates = pd.to_datetime(frame.session_date_et)
    record.update({
        "sha256": file_sha256(path), "rows": len(frame), "securities": frame.security_id.nunique(),
        "sessions": dates.nunique(), "first_session": str(dates.min().date()), "last_session": str(dates.max().date()),
    })
    fixture["frames"][index] = frame
    fixture["manifest"]["rows"] = sum(item["rows"] for item in fixture["manifest"]["files"])


@pytest.fixture
def inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    original = [f"s{index:03}" for index in range(20)]
    warmup = ["w000"]
    combined = {
        "modeled_security_count": 20, "modeled_security_ids_sha256": json_sha256(original),
        "excluded_security_ids": ["s000"], "excluded_security_ids_sha256": json_sha256(["s000"]),
        "excluded_security_count": 1, "retained_security_count": 19,
    }
    memberships = pd.DataFrame({
        "security_id": original + warmup, "ticker": [f"T{index:03}" for index in range(21)],
        "sector": "technology", "industry": "software", "market_cap_bucket": "large",
        "liquidity_bucket": "high", "primary_benchmark": "XLK", "universe_snapshot_id": "synthetic",
        "source": "synthetic-test-evidence",
        "effective_from_utc": pd.Timestamp("2023-01-01T00:00:00Z"),
        "effective_to_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
        "available_at_utc": pd.Timestamp("2022-12-30T00:00:00Z"),
        "numeric_poison": "not-a-price",
    })
    memberships.to_parquet(tmp_path / "membership.parquet", index=False)
    fixture: dict[str, Any] = {
        "root": tmp_path, "config": tmp_path / "preflight.toml", "output": tmp_path / "reports" / "identity.json",
        "combined": combined, "memberships": memberships, "frames": [None, None],
        "manifest": {"feature_profiles": ["technical_market"], "files": [
            {"path": "january.parquet", "partition_month": "2024-01", "feature_profile": "technical_market", "rows": 0},
            {"path": "february.parquet", "partition_month": "2024-02", "feature_profile": "technical_market", "rows": 0},
        ]},
        "cohort_values": {
            "schema_version": "market_predictor.swing_research_cohort", "scope": "retrospective_development_restriction",
            "price_basis_status": "not_certified_by_cohort", "combined_daily_inputs_sha256": json_sha256(combined),
            "original_security_ids": original, "inherited_excluded_security_ids": ["s000"],
            "warmup_only_security_ids": warmup,
            "exclusions": [{"security_id": "s019", "tickers": ["T019"], "reason": "unresolved_holding_identity"}],
            "maximum_exclusion_bps": 1000, "cap_approval_reference": "synthetic ten percent authorization",
        },
        "request": {
            "schema": "market_predictor.swing_holding_identity_preflight_request", "cohort_path": "cohort.json",
            "parent_manifest_path": "panel/manifest.json", "membership_path": "membership.parquet",
            "decision_start": "2024-01-02", "snapshot_end": "2024-02-02",
            "initial_fit_end": "2024-01-17", "horizon_sessions": 10,
        },
    }
    (tmp_path / "panel").mkdir()
    for index, dates in enumerate((("2024-01-02", "2024-01-03", "2024-01-31"), ("2024-02-01", "2024-02-02"))):
        frame = pd.DataFrame([
            {"decision_id": f"{identity}:{day}", "security_id": identity, "ticker": f"T{int(identity[1:]):03}",
             "sector": "technology", "primary_benchmark": "XLK", "session_date_et": day,
             "open": float("nan"), "close": float("inf"), "volume": -1,
             "future_net_return_10d": "forbidden-numeric-outcome", "prediction_score": "forbidden-feature"}
            for day in dates for identity in original[1:]
        ])
        frame["decision_time_utc"] = swing_prediction_cutoffs(frame.session_date_et)
        _write_partition(fixture, index, frame)
    _bind(fixture)
    return fixture


def _run(fixture: dict[str, Any]) -> dict[str, Any]:
    return audit.run_swing_holding_identity_preflight(fixture["root"], fixture["config"], fixture["output"])


def test_accepted_cap_counts_terminal_and_initial_fit_without_numeric_admission(inventory: dict[str, Any]) -> None:
    report = _run(inventory)
    assert report["status"] == "membership_identity_covered"
    assert report["all_history"] == {
        "decisions": 90, "terminal_immature": 54, "covered_matured": 36,
        "uncovered_matured": 0, "uncovered_session_instances": 0,
    }
    assert report["initial_fit"] == {
        "decisions": 18, "terminal_immature": 0, "covered_matured": 18,
        "uncovered_matured": 0, "uncovered_session_instances": 0,
    }
    assert report["affected_securities"] == []
    assert [entry["month"] for entry in report["partitions"]] == ["2024-01", "2024-02"]
    for key in ("numeric_features_read", "outcome_columns_read", "heldout_outcomes_read", "bar_coverage_verified",
                "benchmark_paths_verified", "accounting_eligible", "promotion_eligible"):
        assert report[key] is False
    assert report["audit_sha256"] == json_sha256({key: value for key, value in report.items() if key != "audit_sha256"})
    assert not (inventory["root"] / "runtime" / "heavy-job.owner.json").exists()


def test_replay_is_immutable_and_binds_all_sources(inventory: dict[str, Any]) -> None:
    root = inventory["root"]
    before = {path: file_sha256(path) for path in root.rglob("*") if path.is_file()}
    first = _run(inventory)
    content = inventory["output"].read_bytes()
    mtime = inventory["output"].stat().st_mtime_ns
    assert _run(inventory) == first
    assert inventory["output"].read_bytes() == content
    assert inventory["output"].stat().st_mtime_ns == mtime
    assert {path: file_sha256(path) for path in before} == before
    assert set(first["source_files"]) == {
        "preflight.toml", "cohort.json", "parent.json", "membership.parquet", "panel/manifest.json",
        "panel/january.parquet", "panel/february.parquet",
    }


def test_changed_request_cannot_overwrite_existing_report(inventory: dict[str, Any]) -> None:
    _run(inventory)
    before = inventory["output"].read_bytes()
    inventory["request"]["initial_fit_end"] = "2024-01-18"
    _config(inventory)
    with pytest.raises(DataReadinessError, match="conflict"):
        _run(inventory)
    assert inventory["output"].read_bytes() == before


def test_unapproved_lower_cap_fails_before_publication(inventory: dict[str, Any]) -> None:
    inventory["cohort_values"]["maximum_exclusion_bps"] = 500
    _bind(inventory)
    with pytest.raises(DataReadinessError, match="accepted research cohort"):
        _run(inventory)
    assert not inventory["output"].exists()


@pytest.mark.parametrize("relative", ["cohort.json", "parent.json", "membership.parquet", "panel/january.parquet"])
def test_altered_bound_source_or_cohort_is_rejected(inventory: dict[str, Any], relative: str) -> None:
    path = inventory["root"] / relative
    if relative == "cohort.json":
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["summary"]["retained_securities"] += 1
        _json(path, envelope)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(DataReadinessError):
        _run(inventory)
    assert not inventory["output"].exists()


def test_manifest_partition_hash_must_match_even_with_resigned_cohort(inventory: dict[str, Any]) -> None:
    inventory["manifest"]["files"][0]["sha256"] = "0" * 64
    _bind(inventory)
    with pytest.raises(DataReadinessError, match="partition hash"):
        _run(inventory)


def test_config_must_pin_cohort_identity(inventory: dict[str, Any]) -> None:
    inventory["request"]["cohort_sha256"] = "0" * 64
    _config(inventory)
    with pytest.raises(DataReadinessError, match="pinned"):
        _run(inventory)


def test_full_membership_keeps_excluded_competitor_and_reports_identity_only(inventory: dict[str, Any]) -> None:
    memberships = inventory["memberships"].copy()
    competitor = memberships.loc[memberships.security_id.eq("s019")].copy()
    competitor["ticker"] = "T001"
    competitor["effective_from_utc"] = pd.Timestamp("2024-01-09T14:30:00Z")
    competitor["effective_to_utc"] = pd.Timestamp("2024-01-09T15:00:00Z")
    pd.concat([memberships, competitor], ignore_index=True).to_parquet(inventory["root"] / "membership.parquet", index=False)
    _bind(inventory)
    report = _run(inventory)
    assert report["status"] == "uncovered_holding_identity"
    assert report["all_history"]["uncovered_matured"] == 2
    assert report["all_history"]["uncovered_session_instances"] == 2
    assert report["initial_fit"]["uncovered_matured"] == 1
    assert report["affected_securities"] == [{
        "security_id": "s001", "tickers": ["T001"], "decisions": 2, "initial_fit_decisions": 1,
        "first_decision": "2024-01-02", "last_decision": "2024-01-03", "uncovered_session_instances": 2,
    }]
    assert report["accounting_eligible"] is False


@pytest.mark.parametrize("removed", ["s000", "s019", "w000"])
def test_filtered_membership_authority_is_rejected(inventory: dict[str, Any], removed: str) -> None:
    memberships = inventory["memberships"]
    memberships.loc[~memberships.security_id.eq(removed)].to_parquet(inventory["root"] / "membership.parquet", index=False)
    _bind(inventory)
    with pytest.raises(DataReadinessError, match="full, unfiltered"):
        _run(inventory)


@pytest.mark.parametrize("field", ["cohort_path", "membership_path", "parent_manifest_path", "output", "runtime", "config"])
def test_paths_cannot_escape_authority(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    outside = inventory["root"].parent / "outside-authority"
    if field in inventory["request"]:
        inventory["request"][field] = "../outside-authority"
        _config(inventory)
    elif field == "runtime":
        monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(outside))
    else:
        inventory[field] = outside
    with pytest.raises(DataReadinessError, match="escape"):
        _run(inventory)


def test_busy_lease_precedes_input_loading(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("input loading began while another job held the lease")

    monkeypatch.setattr(audit, "load_swing_research_cohort", forbidden)
    monkeypatch.setattr(audit, "_object", forbidden)
    monkeypatch.setattr(audit, "_projection", forbidden)
    monkeypatch.setattr(audit, "_guard", forbidden)
    with heavy_job_lease("synthetic-other-job", runtime_dir=inventory["root"] / "runtime"):
        with pytest.raises(HeavyJobBusyError):
            _run(inventory)
    assert not inventory["output"].exists()


def test_parquet_reads_are_explicit_identity_projections(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = audit.pq.ParquetFile
    calls: list[tuple[str, tuple[str, ...]]] = []

    class ProjectedFile:
        def __init__(self, path: Path) -> None:
            self.path = Path(path)
            self.source = original(path)
            self.metadata = self.source.metadata

        def __enter__(self) -> ProjectedFile:
            return self

        def __exit__(self, *args: Any) -> None:
            self.source.close()

        def read(self, *, columns: list[str], use_threads: bool) -> Any:
            assert use_threads is False
            assert columns
            assert not set(columns).intersection({"open", "close", "volume", "future_net_return_10d", "prediction_score", "numeric_poison"})
            if self.path.name != "membership.parquet":
                assert tuple(columns) == audit.IDENTITY_COLUMNS
            calls.append((self.path.name, tuple(columns)))
            return self.source.read(columns=columns, use_threads=use_threads)

    monkeypatch.setattr(audit.pq, "ParquetFile", ProjectedFile)
    _run(inventory)
    assert [name for name, _ in calls] == ["membership.parquet", "january.parquet", "february.parquet"]


@pytest.mark.parametrize("defect", [
    "duplicate_id", "duplicate_security_session", "wrong_month", "outside_snapshot", "unknown_security", "wrong_cutoff",
])
def test_hash_valid_partition_identity_defects_are_rejected(inventory: dict[str, Any], defect: str) -> None:
    frame = inventory["frames"][0].copy()
    if defect == "duplicate_id":
        frame.loc[1, "decision_id"] = frame.loc[0, "decision_id"]
    elif defect == "duplicate_security_session":
        for field in ("security_id", "ticker"):
            frame.loc[1, field] = frame.loc[0, field]
    elif defect == "wrong_month":
        frame.loc[0, "session_date_et"] = "2024-02-01"
        frame["decision_time_utc"] = swing_prediction_cutoffs(frame.session_date_et)
    elif defect == "outside_snapshot":
        inventory["request"]["decision_start"] = "2024-01-03"
    elif defect == "unknown_security":
        frame.loc[0, "security_id"] = "not-in-the-cohort"
    else:
        frame.loc[0, "decision_time_utc"] += pd.Timedelta(seconds=1)
    _write_partition(inventory, 0, frame)
    _bind(inventory)
    with pytest.raises(DataReadinessError):
        _run(inventory)


def test_duplicate_decision_id_across_months_is_rejected(inventory: dict[str, Any]) -> None:
    frame = inventory["frames"][1].copy()
    frame.loc[0, "decision_id"] = inventory["frames"][0].loc[0, "decision_id"]
    _write_partition(inventory, 1, frame)
    _bind(inventory)
    with pytest.raises(DataReadinessError, match="decision|duplicate|identit"):
        _run(inventory)


def test_duplicate_month_inventory_is_rejected(inventory: dict[str, Any]) -> None:
    inventory["manifest"]["files"].append(dict(inventory["manifest"]["files"][0]))
    _bind(inventory)
    with pytest.raises(DataReadinessError, match="duplicate|monthly"):
        _run(inventory)


def test_manifest_partition_cannot_escape_authority(inventory: dict[str, Any]) -> None:
    inventory["manifest"]["files"][0]["path"] = "../../outside.parquet"
    _bind(inventory)
    with pytest.raises(DataReadinessError, match="escape"):
        _run(inventory)


def test_output_cannot_replace_bound_input(inventory: dict[str, Any]) -> None:
    inventory["output"] = inventory["root"] / "cohort.json"
    before = inventory["output"].read_bytes()
    with pytest.raises(DataReadinessError, match="overwrite input"):
        _run(inventory)
    assert inventory["output"].read_bytes() == before


@pytest.mark.parametrize("field,value", [("rows", 1), ("sessions", 50), ("securities", 200), ("first_session", "2024-01-03")])
def test_resigned_manifest_counts_must_match_projection(inventory: dict[str, Any], field: str, value: object) -> None:
    inventory["manifest"]["files"][0][field] = value
    _bind(inventory)
    with pytest.raises(DataReadinessError, match="identities, rows or dates"):
        _run(inventory)


def test_checked_in_config_pins_approved_cohort_and_original_development_calendar() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "swing_holding_identity_preflight.toml"
    values = tomllib.loads(path.read_text(encoding="utf-8"))
    assert values == {
        "schema": "market_predictor.swing_holding_identity_preflight_request",
        "cohort_path": "data/reports/swing_research_cohort/approved_research_population_audit.json",
        "cohort_sha256": "794bcf834501cbaa8ec54716e8b561a5aaf0137610fe67587f4db69f9189fca6",
        "parent_manifest_path": "data/features/edge_rebuild_swing_technical_panel_20190709_20260708_v1/final/_manifest.json",
        "membership_path": "data/canonical/index_membership/sp500_memberships_20180529_20260708_v1/memberships.parquet",
        "decision_start": "2019-07-09", "snapshot_end": "2026-07-08", "initial_fit_end": "2024-05-28",
        "horizon_sessions": 10,
    }
