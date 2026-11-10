"""Population/checkpoint orchestration tests; accounting is exercised in kernel tests."""
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.research import investment_labels as module
from market_predictor.swing.contracts.holding_accounting import EvidenceReference


def write(path: Path, value: Any) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": file_sha256(path)}


def setup_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, Path]:
    membership = write(tmp_path / "members.json", {"observed": "fixture"})
    coverage = write(tmp_path / "roster/request.json", {"membership": membership})
    rows = [{"decision_id": "eligible", "security_id": "issuer:one", "ticker": "ONE", "sector": "Technology",
        "sector_benchmark": "XLK", "decision_session": "2023-05-31", "decision_time_utc": "2023-05-31T20:00:00Z",
        "status": "eligible", "qualification_reasons": []},
        {"decision_id": "excluded", "security_id": "issuer:two", "decision_session": "2023-05-31",
            "status": "excluded_research_cohort", "qualification_reasons": ["excluded_research_cohort"]}]
    roster_rows = write(tmp_path / "roster/roster.json", {"rows": rows})
    roster = write(tmp_path / "roster/_manifest.json", {"schema": "market_predictor.investment_coverage_roster",
        "status": "roster_complete", "declared_rows": 2, "source_files": {},
        "files": {"request.json": coverage["sha256"], "roster.json": roster_rows["sha256"]}})
    config = write(tmp_path / "config.json", {})
    request = write(tmp_path / "request.json", {"roster_manifest": roster, "target_config": config,
        "nominal_unpaid_claim_valuation": True, "evaluation_as_of_utc": "2026-01-01T00:00:00Z"})
    monkeypatch.setattr(module, "guard", lambda *_, **__: None)
    monkeypatch.setattr(module, "_implementation", lambda: {})
    monkeypatch.setattr("market_predictor.research.investment_features.guard", lambda *_: None)
    monkeypatch.setattr(module, "load_corrected_outcome_policy", lambda *_: SimpleNamespace(
        action_config=SimpleNamespace(path="actions"), action_archive="archive", action_audit_sha256="a" * 64,
        official_windows=(), decision_corrections=(), numerical_end=datetime(2024, 5, 28, tzinfo=UTC).date()))
    monkeypatch.setattr(module, "load_corporate_action_evidence", lambda **_: object())

    @contextmanager
    def sources(*_: Any, **__: Any) -> Any:
        yield {"source_files": {}, "action_index": ({}, ()), "selection": {}}

    monkeypatch.setattr(module, "heavy_job_lease", sources)
    monkeypatch.setattr(module, "verified_corrected_research_sources", sources)
    monkeypatch.setattr(module, "InvestmentCorporateActions",
        lambda *_, **__: SimpleNamespace(validate_sources=lambda *_: None, recheck=lambda: None, source_files={}))
    monkeypatch.setattr(module, "InvestmentSourceReader", lambda *_: SimpleNamespace(recheck=lambda: None))
    monkeypatch.setattr(module, "_validate_membership_sources", lambda **_: None)

    def compile_row(**kwargs: Any) -> dict[str, Any]:
        row = kwargs["decision"]
        return {"decision_id": row["decision_id"], "security_id": row["security_id"], "horizon_sessions": kwargs["horizon"],
            "decision_session": row["decision_session"],
            "economic_maturity_utc": "2024-06-01T20:00:00Z", "label_available_at_utc": "2026-11-07T00:00:00Z",
            "status": "source_validated", "label_eligible": True, "missing_reasons": [], "components": [],
            "request": {"fixture": "exact request"}, "target": {"fixture": "accounting result"},
            "specifications": {"large": "drop"}, **module.FLAGS}

    monkeypatch.setattr(module, "compile_investment_label", compile_row)
    return Path(request["path"]), request["sha256"], tmp_path / "output"


def test_full_population_horizons_and_asof_are_separate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, digest, output = setup_job(tmp_path, monkeypatch)
    module.materialize_investment_labels(root=tmp_path, request=request, request_sha256=digest, output=output)
    report = json.loads((output / "final/_manifest.json").read_text())
    rows = [row for path in (output / "securities").glob("*/labels.json") for row in json.loads(path.read_text())["rows"]]
    assert len(rows) == 4
    for horizon in ("63", "252"):
        counts = report["counts_by_horizon"][horizon]
        assert counts["population_rows"] == 2 and counts["source_validated_rows"] == 1
        assert counts["currently_available_rows"] == 0 and counts["unresolved_rows"] == 1
    assert all("specifications" not in row for row in rows)
    assert all(row["membership_source_capture_at_utc"] is None for row in rows)
    assert report["training_ready"] is False
    assert report["inherited_security_checkpoint_count"] == 0
    assert report["new_security_checkpoint_count"] == 2
    assert report["new_label_rows"] == 4 and report["inherited_label_rows"] == 0
    assert all(item["origin"] == "new" for item in report["security_checkpoint_origins"])
    coverage = report["coverage_by_year_sector_and_roster_status"]
    assert sum(item["population_rows"] for item in coverage) == 4
    assert sum(item["source_validated_rows"] for item in coverage) == 2
    assert all(item["decision_year"] == "2023" for item in coverage)
    assert all(item["currently_available_rows"] == 0 for item in coverage)
    assert {item["roster_status"] for item in coverage} == {"eligible", "excluded_research_cohort"}
    assert {item["sector"] for item in coverage} == {"Technology", "unavailable"}
    for manifest_path in (output / "securities").glob("*/_manifest.json"):
        checkpoint = json.loads(manifest_path.read_text())
        assert sum(c["population_rows"] for c in checkpoint["counts_by_horizon"].values()) == 2


def test_only_monthly_row_checks_opt_out_of_explicit_reclamation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, digest, output = setup_job(tmp_path, monkeypatch)
    calls = []

    def checked(stage: str, *, reclaim_unused_memory: bool = True) -> None:
        calls.append((stage, reclaim_unused_memory))

    monkeypatch.setattr(module, "guard", checked)
    module.materialize_investment_labels(root=tmp_path, request=request, request_sha256=digest, output=output)
    assert [stage for stage, reclaim in calls if not reclaim] == ["investment monthly horizon label"] * 4
    assert [reclaim for stage, reclaim in calls if stage == "investment serial security labels"] == [True, True]
    assert calls[0] == ("investment label preparation", True)
    assert all(stage == "investment monthly horizon label" or reclaim for stage, reclaim in calls)


def test_monthly_cleanup_opt_out_still_refuses_before_label_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.core.errors import MemoryBudgetError

    request, digest, output = setup_job(tmp_path, monkeypatch)

    def checked(stage: str, *, reclaim_unused_memory: bool = True) -> None:
        if stage == "investment monthly horizon label":
            assert reclaim_unused_memory is False
            raise MemoryBudgetError("synthetic monthly process/system refusal")

    monkeypatch.setattr(module, "guard", checked)
    monkeypatch.setattr(module, "compile_investment_label", lambda **_: pytest.fail("label built after guard refusal"))
    with pytest.raises(MemoryBudgetError, match="monthly process/system refusal"):
        module.materialize_investment_labels(root=tmp_path, request=request, request_sha256=digest, output=output)
    assert not list((output / "securities").glob("*/_manifest.json"))


def test_resume_reuses_security_artifacts_and_detects_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, digest, output = setup_job(tmp_path, monkeypatch)
    original_publish = module._publish

    def interrupted(directory: Path, *args: Any, **kwargs: Any) -> Any:
        if directory.name == "final":
            raise RuntimeError("fixture interruption after security checkpoints")
        return original_publish(directory, *args, **kwargs)

    monkeypatch.setattr(module, "_publish", interrupted)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        module.materialize_investment_labels(root=tmp_path, request=request, request_sha256=digest, output=output)
    monkeypatch.setattr(module, "_publish", original_publish)
    checkpoint = next((output / "securities").glob("*/labels.json"))
    checkpoint.write_text('{"rows": []}', encoding="utf-8")
    with pytest.raises(DataReadinessError, match="hash differs"):
        module.materialize_investment_labels(root=tmp_path, request=request, request_sha256=digest, output=output)


@pytest.mark.parametrize("variant,source_validated", [("index_exit", True), ("future_available", False), ("competing_owner", False)])
def test_decision_membership_cutoff_and_index_exit(variant: str, source_validated: bool) -> None:
    row = {"security_id": "issuer:one", "ticker": "ONE", "sector": "Technology", "sector_benchmark": "XLK",
        "decision_time_utc": "2023-05-31T20:00:00Z"}
    member = {"security_id": "issuer:one", "ticker": "ONE", "sector": "Technology", "primary_benchmark": "XLK",
        "effective_from_utc": "2023-05-31T19:59:00Z", "effective_to_utc": "2023-06-01T00:00:00Z",
        "available_at_utc": "2023-05-31T20:00:00Z"}
    if variant == "future_available":
        member["available_at_utc"] = "2023-05-31T20:00:01Z"
    records = [member]
    if variant == "competing_owner":
        records.append({**member, "security_id": "issuer:other"})
    reference = EvidenceReference(reference="membership", artifact_sha256="a" * 64, record_locator="current verification",
        interpretation_policy_sha256="b" * 64, retrieved_at=datetime(2026, 11, 7, tzinfo=UTC),
        available_at=datetime(2026, 11, 7, tzinfo=UTC))
    result = module._validate_membership_sources(sources={"memberships": pd.DataFrame(records)}, row=row, horizon=252, evidence=reference)
    assert (not result.missing_reasons) == source_validated


@pytest.mark.parametrize("failure", ["lease", "pressure"])
def test_preparation_refusal_precedes_all_input_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    events = []

    @contextmanager
    def lease(*_: Any, **__: Any) -> Any:
        events.append("lease")
        if failure == "lease":
            raise RuntimeError("lease refused")
        yield {}

    def pressure(*_: Any) -> None:
        events.append("guard")
        raise RuntimeError("pressure refused")

    def forbidden_read(*_: Any, **__: Any) -> Any:
        pytest.fail("input read occurred before preparation admission")

    monkeypatch.setattr(module, "heavy_job_lease", lease)
    monkeypatch.setattr(module, "guard", pressure)
    monkeypatch.setattr(module, "_read", forbidden_read)
    monkeypatch.setattr(module, "_object", forbidden_read)
    monkeypatch.setattr(module, "_implementation", forbidden_read)
    with pytest.raises(RuntimeError, match=f"{failure} refused"):
        module.materialize_investment_labels(root=tmp_path, request=tmp_path / "unread.json", request_sha256="a" * 64,
            output=tmp_path / "output")
    assert events == (["lease"] if failure == "lease" else ["lease", "guard"])



def test_selected_decisions_retain_full_roster_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, _, output = setup_job(tmp_path, monkeypatch)
    parsed = json.loads(request.read_text())
    parsed["decision_ids"] = ["eligible"]
    pin = write(request, parsed)
    module.materialize_investment_labels(root=tmp_path, request=request, request_sha256=pin["sha256"], output=output)
    report = json.loads((output / "final/_manifest.json").read_text())
    assert report["scope"] == "selected_decisions"
    assert report["frozen_roster_rows"] == 2
    assert report["roster_rows"] == 1 and report["label_rows"] == 2
    assert report["selected_decision_ids"] == ["eligible"]
    assert report["training_ready"] is False
    assert len(report["security_checkpoints"]) == 1


@pytest.mark.parametrize("selection", [[], ["eligible", "eligible"], ["unknown"]])
def test_invalid_decision_selection_refuses_before_source_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: list[str]
) -> None:
    request, _, output = setup_job(tmp_path, monkeypatch)
    parsed = json.loads(request.read_text())
    parsed["decision_ids"] = selection
    pin = write(request, parsed)

    def forbidden(**_: Any) -> Any:
        pytest.fail("invalid selection reached independent source replay")

    monkeypatch.setattr(module, "load_corporate_action_evidence", forbidden)
    with pytest.raises((DataReadinessError, ValueError)):
        module.materialize_investment_labels(root=tmp_path, request=request, request_sha256=pin["sha256"], output=output)
    assert not output.exists()


def test_changed_selection_cannot_reuse_old_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, digest, output = setup_job(tmp_path, monkeypatch)
    original_publish = module._publish

    def interrupted(directory: Path, *args: Any, **kwargs: Any) -> Any:
        if directory.name == "final":
            raise RuntimeError("fixture interruption")
        return original_publish(directory, *args, **kwargs)

    monkeypatch.setattr(module, "_publish", interrupted)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        module.materialize_investment_labels(root=tmp_path, request=request, request_sha256=digest, output=output)
    parsed = json.loads(request.read_text())
    parsed["decision_ids"] = ["eligible"]
    pin = write(request, parsed)
    with pytest.raises(DataReadinessError, match="resume request or implementation differs"):
        module.materialize_investment_labels(root=tmp_path, request=request, request_sha256=pin["sha256"], output=output)


@pytest.mark.parametrize("collision", ["sector_spy", "sector_qqq", "stock_symbol", "stock_identity"])
def test_duplicate_comparison_is_local_before_any_compiler_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collision: str
) -> None:
    request, digest, output = setup_job(tmp_path, monkeypatch)
    request_data = json.loads(request.read_text())
    roster_path = tmp_path / "roster/roster.json"
    rows = json.loads(roster_path.read_text())
    row = rows["rows"][0]
    if collision.startswith("sector_"):
        row["sector_benchmark"] = collision.removeprefix("sector_").upper()
    elif collision == "stock_symbol":
        row["ticker"] = "SPY"
    else:
        row["security_id"] = "benchmark:QQQ"
    rows_pin = write(roster_path, rows)
    roster_path = tmp_path / "roster/_manifest.json"
    roster = json.loads(roster_path.read_text())
    roster["files"]["roster.json"] = rows_pin["sha256"]
    request_data["roster_manifest"] = write(roster_path, roster)
    pin = write(request, request_data)
    monkeypatch.setattr(module, "compile_investment_label", lambda **_: pytest.fail("duplicate reached compiler/source reads"))
    module.materialize_investment_labels(root=tmp_path, request=request, request_sha256=pin["sha256"], output=output)
    records = [row for path in (output / "securities").glob("*/labels.json") for row in json.loads(path.read_text())["rows"]]
    eligible = [row for row in records if row["roster_status"] == "eligible"]
    assert {row["horizon_sessions"] for row in eligible} == {63, 252}
    assert all(row["missing_reasons"] == ["required_comparison_security_or_symbol_not_distinct"] for row in eligible)
    assert all(row["status"] == "unresolved" and not row["label_eligible"] for row in eligible)
    assert all(row["request"] is row["target"] is row["request_sha256"] is None for row in eligible)
    assert all(row["components"] == [] and row["training_ready"] is False for row in eligible)


def test_comparison_precheck_uses_corrected_ticker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, digest, output = setup_job(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "corrected_ticker", lambda *_: "SPY")
    monkeypatch.setattr(module, "compile_investment_label", lambda **_: pytest.fail("uncorrected binding admitted"))
    module.materialize_investment_labels(root=tmp_path, request=request, request_sha256=digest, output=output)
    report = json.loads((output / "final/_manifest.json").read_text())
    assert report["counts_by_horizon"]["63"]["source_validated_rows"] == 0


def test_negative_precheck_does_not_hide_pinned_source_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, digest, output = setup_job(tmp_path, monkeypatch)
    (tmp_path / "members.json").write_text("tampered")
    monkeypatch.setattr(module, "corrected_ticker", lambda *_: "SPY")
    with pytest.raises(DataReadinessError, match="hash differs"):
        module.materialize_investment_labels(root=tmp_path, request=request, request_sha256=digest, output=output)
    assert not output.exists()
