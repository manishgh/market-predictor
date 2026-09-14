from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

import market_predictor.research.swing_training_readiness as owner
from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.store import file_sha256, manifest_path_for, write_canonical_artifact
from market_predictor.commands import swing_training_readiness as commands
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.features.panel import swing_model_feature_columns
from tests.test_fixed_horizon_readiness import inputs

REPO = Path(__file__).resolve().parents[1]


def _json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return file_sha256(path)


@pytest.fixture
def publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path
    package = root / "src/market_predictor"
    for name in owner.IMPLEMENTATION_PATHS:
        destination = package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "src/market_predictor" / name, destination)
    monkeypatch.setattr(owner, "__file__", str(package / "research/swing_training_readiness.py"))
    config = root / "configs/readiness.json"
    policy: dict[str, Any] = dict(schema_version="market_predictor.swing_training_readiness",
        scope="initial_fit_fixed_horizon_diagnostics")
    for key, name in (("research_contract", "swing_research.toml"),
            ("strategy_contract", "edge_rebuild_strategy_contract.toml"),
            ("temporal_contract", "edge_rebuild_temporal_manifest.toml")):
        destination = root / "configs" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "configs" / name, destination)
        policy[key] = dict(path=destination.relative_to(root).as_posix(), sha256=file_sha256(destination))
    strategy = load_strategy_contract(root / policy["strategy_contract"]["path"])
    days = (date(2019, 7, 9), date(2019, 7, 10), date(2019, 7, 31))
    calendar = xcals.get_calendar("XNYS")
    frame, _ = inputs()
    frame["session_date_et"] = list(days)
    frame["decision_time_utc"] = swing_prediction_cutoffs(frame.session_date_et)
    frame["research_label_mature_at"] = pd.to_datetime([
        calendar.session_close(calendar.sessions_window(pd.Timestamp(day), 11)[-1]) for day in days], utc=True)
    frame.loc[2, "stock_source_admitted"] = False
    frame.loc[2, "fixed_comparisons_complete"] = False
    frame.loc[2, "research_label_mature_at"] = pd.NaT
    frame.loc[2, "stock_component_id"] = None
    frame.loc[2, "stock_missing_reasons"] = '["initial_fit_terminal_immature"]'
    frame.loc[2, ["fixed_horizon_gross_return", "fixed_horizon_net_return"]] = np.nan
    for role in ("spy", "qqq", "sector"):
        frame.loc[2, [f"{role}_horizon_gross_return", f"{role}_fixed_horizon_excess_return"]] = np.nan
        frame.loc[2, f"{role}_component_id"] = None
        frame.loc[2, f"{role}_missing_reasons"] = '["initial_fit_terminal_immature"]'
    monkeypatch.setattr(owner, "load_temporal_manifest_config", lambda _: SimpleNamespace(
        calendar="XNYS", initial_fit_expected_sessions=3, label_horizon_sessions=10, warmup_sessions=250))
    monkeypatch.setattr(owner, "build_temporal_schedule", lambda _: SimpleNamespace(
        folds=(SimpleNamespace(train_sessions=days),)))
    monkeypatch.setattr(owner, "_guard", lambda: None)
    monkeypatch.setattr(owner, "release_process_memory", lambda: None)
    directory = root / "data/features/join"
    request = dict(schema="market_predictor.research_join_request", rows=3,
        historical_first_seen_proven=False, managed_outcomes_available=False,
        profiles=["technical_market", "catalyst_full"], cohort_sha256="c" * 64,
        source_files={policy[name]["path"]: policy[name]["sha256"] for name in ("research_contract", "strategy_contract")})
    request_pin = _json(directory / "_request.json", request)
    record: dict[str, Any] = dict(rows=3, decision_ids_sha256=json_sha256(sorted(frame.decision_id)), profiles={})
    paths = {}
    for profile in ("technical_market", "catalyst_full"):
        names = swing_model_feature_columns(contract=strategy, catalyst=profile == "catalyst_full")
        clocks = {name: f"available_at_{name}" for name in names}
        derived = pd.DataFrame({**{name: [1.0, 2.0, np.nan] for name in names},
            **{clock: frame.decision_time_utc for clock in clocks.values()}})
        joined = pd.concat((frame.drop(columns=["feature", "feature_clock"]), derived), axis=1)
        path = directory / "2019-07" / f"{profile}.parquet"
        audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="fixture", status="pass", failures=0,
            rows_checked=3, detail="Synthetic test-only evidence."),))
        write_canonical_artifact(joined, path, artifact_type="swing_research_join", audit=audit,
            inputs={"request_sha256": request_pin}, production_ready=False)
        paths[profile] = path
        record["profiles"][profile] = dict(path=f"2019-07/{profile}.parquet", sha256=file_sha256(path),
            manifest_sha256=file_sha256(manifest_path_for(path)), model_columns=list(names), availability_columns=clocks,
            audit=dict(training_eligible=False, promotion_eligible=False))
    manifest = dict(schema="market_predictor.research_join", status="complete_research_only", rows=3,
        request_sha256=request_pin, training_eligible=False, promotion_eligible=False, exclusions_added=[],
        months={"2019-07": record})
    manifest_pin = _json(directory / "_manifest.json", manifest)
    policy["publication"] = dict(path="data/features/join/_manifest.json", sha256=manifest_pin)
    receipt = dict(manifest_sha256=manifest_pin, status="passed", scope="published_join_population_clocks_original_targets",
        matched_profile_population=True, original_outcome_values_exact=True, outcome_filtered_rows=0,
        unique_decisions=3, months=1, training_eligible=False, promotion_eligible=False,
        profiles={profile: dict(rows=3, feature_eligible=3, complete_model_rows=2, clock_violations=0) for profile in paths})
    receipt_path = root / "data/reports/prior.json"
    policy["saved_row_verification"] = dict(path="data/reports/prior.json", sha256=_json(receipt_path, receipt))
    return dict(root=root, config=config, config_sha256=_json(config, policy), policy=policy, manifest=manifest,
        paths=paths, receipt=receipt, output=root / "data/reports/readiness.json")


def _run(state: dict[str, Any]) -> dict[str, Any]:
    return owner.audit_swing_training_readiness(**{name: state[name] for name in
        ("root", "config", "config_sha256", "output")})


def _repin(state: dict[str, Any]) -> None:
    policy = state["policy"]
    policy["publication"]["sha256"] = _json(state["root"] / policy["publication"]["path"], state["manifest"])
    state["receipt"]["manifest_sha256"] = policy["publication"]["sha256"]
    policy["saved_row_verification"]["sha256"] = _json(
        state["root"] / policy["saved_row_verification"]["path"], state["receipt"])
    state["config_sha256"] = _json(state["config"], policy)


def test_real_artifact_audit_preserves_populations_and_denies_training(publication: dict[str, Any]) -> None:
    before = {path: file_sha256(path) for path in publication["paths"].values()}
    report = _run(publication)
    assert report["status"] == "diagnostic_complete"
    assert report["rows_removed"] == 0
    assert report["training_eligible"] is False
    assert report["promotion_eligible"] is False
    assert report["managed_evaluation_eligible"] is False
    assert report["profiles"]["technical_market"]["complete_case_supervision"] == 2
    assert report["months"]["2019-07"]["paired_complete_case_supervision"] == 2
    assert len(report["missing_model_values"]["technical_market"]) == 120
    assert before == {path: file_sha256(path) for path in before}
    assert not (publication["root"] / "data/runtime/heavy-job.owner.json").exists()


@pytest.mark.parametrize("target", ["config", "publication", "saved_row_verification", "research_contract",
    "strategy_contract", "temporal_contract", "child", "sidecar", "request"])
def test_changed_pin_aborts_without_report(publication: dict[str, Any], target: str) -> None:
    if target == "config":
        path = publication["config"]
    elif target in ("child", "sidecar"):
        path = publication["paths"]["technical_market"]
        if target == "sidecar":
            path = manifest_path_for(path)
    elif target == "request":
        path = publication["root"] / "data/features/join/_request.json"
    else:
        path = publication["root"] / publication["policy"][target]["path"]
    with path.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(DataReadinessError):
        _run(publication)
    assert not publication["output"].exists()


@pytest.mark.parametrize("poison", ["reordered_features", "missing_month", "extra_profile", "training_flag", "receipt_scope"])
def test_rehashed_invalid_metadata_rejected(publication: dict[str, Any], poison: str) -> None:
    manifest = publication["manifest"]
    record = manifest["months"]["2019-07"]
    if poison == "reordered_features":
        record["profiles"]["technical_market"]["model_columns"].reverse()
    elif poison == "missing_month":
        manifest["months"] = {}
    elif poison == "extra_profile":
        record["profiles"]["bogus"] = {}
    elif poison == "training_flag":
        manifest["training_eligible"] = True
    else:
        publication["receipt"]["scope"] = "another_scope"
    _repin(publication)
    with pytest.raises(DataReadinessError):
        _run(publication)
    assert not publication["output"].exists()


def test_busy_lease_rejects_before_reading_config(publication: dict[str, Any]) -> None:
    publication["config"].unlink()
    with heavy_job_lease("other-job", runtime_dir=publication["root"] / "data/runtime"):
        with pytest.raises(HeavyJobBusyError):
            _run(publication)
    assert not publication["output"].exists()


def test_memory_guard_aborts_without_report(publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> None:
        raise MemoryBudgetError("fixture memory limit")
    monkeypatch.setattr(owner, "_guard", refuse)
    with pytest.raises(MemoryBudgetError):
        _run(publication)
    assert not publication["output"].exists()


def test_immutable_output_not_overwritten(publication: dict[str, Any]) -> None:
    publication["output"].write_text("protected", encoding="ascii")
    with pytest.raises(FileExistsError):
        _run(publication)
    assert publication["output"].read_text() == "protected"


def test_rehashed_profile_target_disagreement_rejected(publication: dict[str, Any]) -> None:
    path = publication["paths"]["catalyst_full"]
    frame = pd.read_parquet(path)
    # Arithmetic remains internally consistent, but the paired target changes.
    frame.loc[0, "fixed_horizon_gross_return"] = 0.2
    frame.loc[0, "fixed_horizon_net_return"] = 0.198
    for role in ("spy", "qqq", "sector"):
        frame.loc[0, f"{role}_fixed_horizon_excess_return"] = 0.198 - 0.02
    frame.to_parquet(path, index=False)
    sidecar_path = manifest_path_for(path)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["artifact_sha256"] = file_sha256(path)
    child = publication["manifest"]["months"]["2019-07"]["profiles"]["catalyst_full"]
    child["sha256"] = file_sha256(path)
    child["manifest_sha256"] = _json(sidecar_path, sidecar)
    _repin(publication)
    with pytest.raises(DataReadinessError, match="profiles differ"):
        _run(publication)
    assert not publication["output"].exists()


def test_input_mutation_during_read_is_detected(publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = owner.load_canonical_artifact
    def mutate(path: Path, **kwargs: Any) -> Any:
        loaded = original(path, **kwargs)
        with path.open("ab") as handle:
            handle.write(b"changed")
        return loaded
    monkeypatch.setattr(owner, "load_canonical_artifact", mutate)
    with pytest.raises(DataReadinessError, match="evidence changed"):
        _run(publication)
    assert not publication["output"].exists()


def test_output_outside_reports_rejected_before_input_read(publication: dict[str, Any]) -> None:
    publication["output"] = publication["root"] / "data/features/new.json"
    publication["config"].unlink()
    with pytest.raises(DataReadinessError, match="below data/reports"):
        _run(publication)


def test_rehashed_transform_policy_must_match_publication(publication: dict[str, Any]) -> None:
    reference = publication["policy"]["strategy_contract"]
    path = publication["root"] / reference["path"]
    original = path.read_text()
    changed = original.replace("cross_sectional_winsorize_quantile = 0.01", "cross_sectional_winsorize_quantile = 0.02")
    assert changed != original
    path.write_text(changed, encoding="utf-8")
    reference["sha256"] = file_sha256(path)
    publication["config_sha256"] = _json(publication["config"], publication["policy"])
    with pytest.raises(DataReadinessError, match="publication provenance"):
        _run(publication)


def test_cli_is_a_single_lease_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    def run(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return dict(status="diagnostic_complete", sessions=3, securities=1, profiles={},
            training_eligible=False, report_sha256="a" * 64)
    monkeypatch.setattr(commands, "audit_swing_training_readiness", run)
    app = typer.Typer()
    commands.register_training_readiness_command(app)
    result = CliRunner().invoke(app, ["--root", "repo", "--config", "input.json", "--config-sha256", "a" * 64,
        "--output", "data/reports/audit.json"])
    assert result.exit_code == 0, result.output
    assert calls == [dict(root=Path("repo"), config=Path("input.json"), config_sha256="a" * 64,
        output=Path("data/reports/audit.json"))]


def test_cli_busy_returns_75(monkeypatch: pytest.MonkeyPatch) -> None:
    def busy(**kwargs: Any) -> None:
        raise HeavyJobBusyError("owned")
    monkeypatch.setattr(commands, "audit_swing_training_readiness", busy)
    app = typer.Typer()
    commands.register_training_readiness_command(app)
    result = CliRunner().invoke(app, ["--config-sha256", "a" * 64, "--output", "data/reports/audit.json"])
    assert result.exit_code == 75
