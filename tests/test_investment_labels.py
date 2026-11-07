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
    monkeypatch.setattr(module, "guard", lambda *_: None)
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
        lambda *_, **__: SimpleNamespace(validate_sources=lambda *_: None, recheck=lambda: None))
    monkeypatch.setattr(module, "InvestmentSourceReader", lambda *_: SimpleNamespace(recheck=lambda: None))
    monkeypatch.setattr(module, "_validate_membership_sources", lambda **_: None)

    def compile_row(**kwargs: Any) -> dict[str, Any]:
        row = kwargs["decision"]
        return {"decision_id": row["decision_id"], "security_id": row["security_id"], "horizon_sessions": kwargs["horizon"],
            "economic_maturity_utc": "2024-06-01T20:00:00Z", "label_available_at_utc": "2026-11-07T00:00:00Z",
            "status": "source_validated", "label_eligible": True, "missing_reasons": [], "components": [],
            "request": {"fixture": "exact request"}, "target": {"fixture": "accounting result"}, "specifications": {"large": "drop"}}

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
