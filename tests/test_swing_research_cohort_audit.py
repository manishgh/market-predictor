"""Synthetic identity-only inventories; no real research outcomes are inspected."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.research import swing_cohort as audit
from market_predictor.swing.contracts.research_cohort import load_swing_research_cohort


def _json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return file_sha256(path)


def _signed(path: Path, payload: dict[str, Any], field: str) -> str:
    payload.pop(field, None)
    payload[field] = json_sha256(payload)
    return _json(path, payload)


def _config(path: Path, values: dict[str, Any]) -> None:
    scalars = [f"{key} = {json.dumps(value)}" for key, value in values.items() if key != "exclusions"]
    for entry in values["exclusions"]:
        scalars.extend(["\n[[exclusions]]", *(f"{key} = {json.dumps(value)}" for key, value in entry.items())])
    path.write_text("\n".join(scalars), encoding="utf-8")


@pytest.fixture
def inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    original = [f"s{i:04}" for i in range(631)]
    warmup = [f"w{i:04}" for i in range(27)]
    inherited = original[:27]
    retained = original[27:]
    membership = tmp_path / "membership.parquet"
    pd.DataFrame({"security_id": original + warmup, "ticker": [f"T{i:04}" for i in range(658)]}).to_parquet(membership)
    combined = {
        "modeled_security_count": len(original), "modeled_security_ids_sha256": json_sha256(original),
        "retained_security_count": len(retained), "excluded_security_count": len(inherited),
        "excluded_security_ids": inherited, "excluded_security_ids_sha256": json_sha256(inherited),
        "warmup_only_security_ids": warmup, "warmup_only_security_count": len(warmup),
        "warmup_only_security_ids_sha256": json_sha256(warmup),
        "membership_authority": {"membership_artifact_sha256": file_sha256(membership)},
    }
    parent = {
        "combined_daily_inputs": combined, "modeled_security_count": len(retained),
        "security_count": len(retained), "modeled_security_ids_sha256": json_sha256(retained),
        "security_ids_sha256": json_sha256(retained), "warmup_only_security_ids": warmup,
        "warmup_only_security_count": len(warmup), "warmup_only_security_ids_sha256": json_sha256(warmup),
    }
    parent_sha = _signed(tmp_path / "parent.json", parent, "request_sha256")
    files = []
    for year in (2019, 2020):
        session = f"{year}-07-10"
        path = tmp_path / "final" / f"{year}-07.parquet"
        path.parent.mkdir(exist_ok=True)
        frame = pd.DataFrame({
            "security_id": retained, "session_date_et": session,
            "sector": ["healthcare" if int(x[1:]) % 2 else "technology" for x in retained],
            "future_net_return_10d": float("nan"), "feature_eligible": False,
            "private_numeric_outcome": -99999.0,
        })
        frame.to_parquet(path)
        files.append({"path": path.name, "sha256": file_sha256(path), "rows": len(frame),
                      "feature_profile": "technical_market", "partition_month": f"{year}-07",
                      "first_session": session, "last_session": session, "sessions": 1, "securities": len(retained)})
    manifest = {
        "request_sha256": parent["request_sha256"], "files": files, "rows": 2 * len(retained),
        "securities": len(retained), "modeled_security_count": len(retained),
        "modeled_security_ids_sha256": json_sha256(retained), "feature_profiles": ["technical_market"],
    }
    manifest_sha = _json(tmp_path / "final" / "manifest.json", manifest)
    control = {
        "scope": "initial_fit_deterministic_control_not_out_of_sample", "validation_or_test_outcomes_read": False,
        "bound_files": {"parent.json": parent_sha, "final/manifest.json": manifest_sha},
        "status": "blocked", "errors": {"code": "old_selected_path_incomplete"},
    }
    control_sha = _signed(tmp_path / "control.json", control, "audit_sha256")
    values = {
        "schema": "market_predictor.swing_research_cohort_request",
        "parent_request_path": "parent.json", "parent_request_sha256": parent_sha,
        "membership_path": "membership.parquet", "membership_sha256": file_sha256(membership),
        "parent_manifest_path": "final/manifest.json", "parent_manifest_sha256": manifest_sha,
        "control_report_path": "control.json", "control_report_sha256": control_sha,
        "maximum_exclusion_bps": 500, "cap_approval_reference": "existing frozen five percent ceiling",
        "exclusions": [{"security_id": sid, "tickers": [f"T{int(sid[1:]):04}"], "reason": "unresolved_holding_identity"}
                       for sid in original[27:45]],
    }
    config = tmp_path / "cohort.toml"
    _config(config, values)
    return {"root": tmp_path, "config": config, "output": tmp_path / "report.json", "values": values,
            "parent": parent, "manifest": manifest, "control": control, "retained": retained}


def _run(fixture: dict[str, Any]) -> dict[str, Any]:
    return audit.run_swing_research_cohort_audit(fixture["root"], fixture["config"], fixture["output"])


def _rebind(fixture: dict[str, Any], *, sign_parent: bool = True) -> None:
    root, values = fixture["root"], fixture["values"]
    parent, manifest, control = fixture["parent"], fixture["manifest"], fixture["control"]
    if sign_parent:
        _signed(root / "parent.json", parent, "request_sha256")
    else:
        _json(root / "parent.json", parent)
    values["parent_request_sha256"] = file_sha256(root / "parent.json")
    manifest["request_sha256"] = parent["request_sha256"]
    values["parent_manifest_sha256"] = _json(root / "final" / "manifest.json", manifest)
    control["bound_files"] = {"parent.json": values["parent_request_sha256"],
                              "final/manifest.json": values["parent_manifest_sha256"]}
    values["control_report_sha256"] = _signed(root / "control.json", control, "audit_sha256")
    _config(fixture["config"], values)


def test_cumulative_cap_and_identity_coverage_without_mutation(inventory: dict[str, Any]) -> None:
    root = inventory["root"]
    before = {path: file_sha256(path) for path in root.rglob("*") if path.is_file()}
    report = _run(inventory)
    summary = report["summary"]
    assert summary["status"] == "blocked_exclusion_cap"
    assert summary["original_securities"] == 631
    assert summary["inherited_excluded_securities"] == 27
    assert summary["additional_excluded_securities"] == 18
    assert summary["total_excluded_securities"] == 45
    assert summary["retained_securities"] == 586
    assert summary["accounting_eligible"] is False
    assert summary["promotion_eligible"] is False
    coverage = report["coverage"]
    assert coverage["observed_parent_securities"] == 604
    assert coverage["original_rows"] == 1208
    assert coverage["removed_rows"] == 36
    assert coverage["retained_rows"] == 1172
    assert len(coverage["by_calendar_year_and_sector"]) == 4
    assert sum(row["retained_rows"] for row in coverage["by_calendar_year_and_sector"]) == 1172
    assert {path: file_sha256(path) for path in before} == before
    cohort = load_swing_research_cohort(inventory["output"], source_root=root)
    assert len(cohort.source_files) == 7  # Four sources, the request configuration, two projected partitions.
    assert not cohort.within_cap
    assert _run(inventory) == report


def test_explicit_eight_percent_fixture_is_research_only(inventory: dict[str, Any]) -> None:
    inventory["values"]["maximum_exclusion_bps"] = 800
    inventory["values"]["cap_approval_reference"] = "synthetic test-only explicit eight percent approval"
    _config(inventory["config"], inventory["values"])
    report = _run(inventory)
    assert report["summary"]["status"] == "accepted_research_restriction"
    assert report["summary"]["promotion_eligible"] is False
    assert report["cohort"]["price_basis_status"] == "not_certified_by_cohort"


def test_coverage_is_bound_by_audit_hash(inventory: dict[str, Any]) -> None:
    report = _run(inventory)
    assert report["audit_sha256"] == json_sha256({k: v for k, v in report.items() if k != "audit_sha256"})
    report["coverage"]["retained_rows"] += 1
    _json(inventory["output"], report)
    with pytest.raises(DataReadinessError):
        load_swing_research_cohort(inventory["output"])


def test_control_must_bind_same_parent(inventory: dict[str, Any]) -> None:
    inventory["control"]["bound_files"]["parent.json"] = "0" * 64
    inventory["values"]["control_report_sha256"] = _signed(
        inventory["root"] / "control.json", inventory["control"], "audit_sha256",
    )
    _config(inventory["config"], inventory["values"])
    with pytest.raises(DataReadinessError, match="control does not bind"):
        _run(inventory)


def test_overlap_uses_union_not_double_count(inventory: dict[str, Any]) -> None:
    inventory["values"]["exclusions"] = [{"security_id": "s0000", "tickers": ["T0000"],
                                            "reason": "unresolved_corporate_action"}]
    _config(inventory["config"], inventory["values"])
    report = _run(inventory)
    assert report["summary"]["total_excluded_securities"] == 27
    assert report["summary"]["additional_excluded_securities"] == 0
    assert report["coverage"]["removed_rows"] == 0


@pytest.mark.parametrize("name", ["parent.json", "final/manifest.json", "control.json", "membership.parquet"])
def test_all_four_hashes_checked_before_membership_read(
    inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    path = inventory["root"] / name
    path.write_bytes(path.read_bytes() + b"tampered")
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("membership read preceded four-source verification")
    monkeypatch.setattr(audit.pd, "read_parquet", forbidden)
    with pytest.raises(DataReadinessError, match="hash differs"):
        _run(inventory)
    assert not inventory["output"].exists()


@pytest.mark.parametrize("change", ["unknown", "wrong_ticker", "warmup", "duplicates", "unsorted", "extra_field", "float_cap"])
def test_invalid_config_exclusions_fail(inventory: dict[str, Any], change: str) -> None:
    values = inventory["values"]
    if change == "unknown":
        values["exclusions"][0]["security_id"] = "unknown"
    elif change == "wrong_ticker":
        values["exclusions"][0]["tickers"] = ["OTHER"]
    elif change == "warmup":
        values["exclusions"] = [{"security_id": "w0000", "tickers": ["T0631"], "reason": "adjusted_price_mismatch"}]
    elif change == "duplicates":
        values["exclusions"].append(values["exclusions"][0])
    elif change == "unsorted":
        values["exclusions"].reverse()
    elif change == "extra_field":
        values["allow_bad_prices"] = True
    else:
        values["maximum_exclusion_bps"] = 500.0
    _config(inventory["config"], values)
    with pytest.raises(DataReadinessError):
        _run(inventory)


@pytest.mark.parametrize("change", ["denominator", "warmup_hash", "inherited_hash", "retained_hash", "canonical_hash"])
def test_parent_population_tamper_rejected(inventory: dict[str, Any], change: str) -> None:
    parent = inventory["parent"]
    if change == "denominator":
        parent["combined_daily_inputs"]["modeled_security_count"] = 604
    elif change == "warmup_hash":
        parent["warmup_only_security_ids"].pop()
    elif change == "inherited_hash":
        parent["combined_daily_inputs"]["excluded_security_ids"].pop()
    elif change == "retained_hash":
        parent["modeled_security_ids_sha256"] = "0" * 64
    else:
        parent["unexpected_metadata"] = "not covered by original canonical hash"
    _rebind(inventory, sign_parent=change != "canonical_hash")
    with pytest.raises(DataReadinessError):
        _run(inventory)


def test_projection_excludes_features_and_outcomes(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = audit.pq.ParquetFile.read
    projections = []
    def projected(self: Any, columns: list[str], **kwargs: Any) -> Any:
        projections.append(columns)
        assert set(columns) <= {"security_id", "session_date_et", "sector", "ticker"}
        return original(self, columns=columns, **kwargs)
    monkeypatch.setattr(audit.pq.ParquetFile, "read", projected)
    report = _run(inventory)
    assert len(projections) >= 2
    assert report["coverage"]["outcome_columns_read"] is False
    assert report["coverage"]["numeric_features_read"] is False
    assert report["coverage"]["heldout_outcomes_read"] is False


@pytest.mark.parametrize("change", ["hash", "missing_id", "duplicate_month", "rows", "unknown_id", "wrong_month"])
def test_partition_inventory_fails_closed(inventory: dict[str, Any], change: str) -> None:
    manifest = inventory["manifest"]
    record = manifest["files"][0]
    path = inventory["root"] / "final" / record["path"]
    if change == "hash":
        path.write_bytes(path.read_bytes() + b"tampered")
    elif change == "duplicate_month":
        manifest["files"].append(dict(record))
        _rebind(inventory)
    elif change == "rows":
        record["rows"] += 1
        _rebind(inventory)
    else:
        # Test setup may mutate synthetic outcomes; the audited builder may not read them.
        for entry in manifest["files"] if change == "missing_id" else [record]:
            target = inventory["root"] / "final" / entry["path"]
            frame = pd.read_parquet(target)
            if change == "missing_id":
                frame = frame.iloc[1:].copy()
                entry["rows"] -= 1
                entry["securities"] -= 1
                manifest["rows"] -= 1
            elif change == "unknown_id":
                frame.loc[0, "security_id"] = "unknown"
            else:
                frame.loc[0, "session_date_et"] = "2019-08-01"
            frame.to_parquet(target)
            entry["sha256"] = file_sha256(target)
        _rebind(inventory)
    with pytest.raises(DataReadinessError):
        _run(inventory)


def test_conflicting_config_cannot_overwrite(inventory: dict[str, Any]) -> None:
    _run(inventory)
    before = inventory["output"].read_bytes()
    inventory["values"]["maximum_exclusion_bps"] = 800
    inventory["values"]["cap_approval_reference"] = "explicit synthetic alternate policy"
    _config(inventory["config"], inventory["values"])
    with pytest.raises(DataReadinessError, match="immutable cohort output conflicts"):
        _run(inventory)
    assert inventory["output"].read_bytes() == before
    assert not list(inventory["root"].glob(".report.json.*.tmp"))


def test_lease_precedes_input_loading(inventory: dict[str, Any]) -> None:
    inventory["config"].unlink()
    runtime = inventory["root"] / "runtime"
    with heavy_job_lease("synthetic competing job", runtime_dir=runtime):
        with pytest.raises(HeavyJobBusyError):
            _run(inventory)
    assert not (runtime / "heavy-job.owner.json").exists()


def test_memory_release_for_each_partition(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    released = []
    guarded = []
    monkeypatch.setattr(audit, "release_process_memory", lambda: released.append(True))
    monkeypatch.setattr(audit, "assert_memory_budget", lambda **kwargs: guarded.append(kwargs))
    monkeypatch.setattr(audit, "assert_peak_memory_budget", lambda **kwargs: None)
    _run(inventory)
    assert len(released) == 3  # Membership plus two monthly frames.
    assert all(entry["hard_budget_gib"] == 5.0 and entry["headroom_gib"] == 0.75 for entry in guarded)


@pytest.mark.parametrize("target", ["config", "output", "source", "partition"])
def test_paths_cannot_escape_root(inventory: dict[str, Any], target: str) -> None:
    if target == "config":
        inventory["config"] = Path("../outside.toml")
    elif target == "output":
        inventory["output"] = Path("../outside.json")
    elif target == "source":
        inventory["values"]["membership_path"] = "../outside.parquet"
        _config(inventory["config"], inventory["values"])
    else:
        inventory["manifest"]["files"][0]["path"] = "../../outside.parquet"
        _rebind(inventory)
    with pytest.raises(DataReadinessError, match="escapes"):
        _run(inventory)


def test_cli_resolves_non_dot_relative_root_once(inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import typer
    from rich.console import Console
    from typer.testing import CliRunner

    from market_predictor.commands.swing_research import register_swing_research_commands

    root = inventory["root"]
    monkeypatch.chdir(root.parent)
    app = typer.Typer()
    register_swing_research_commands(app, Console())
    result = CliRunner().invoke(app, [
        "audit-swing-research-cohort", "--root", root.name,
        "--config", inventory["config"].name, "--output", inventory["output"].name,
    ])
    assert result.exit_code == 2, result.output
    assert "blocked_exclusion_cap" in result.output
    assert load_swing_research_cohort(inventory["output"], source_root=root).summary()["total_excluded_securities"] == 45
