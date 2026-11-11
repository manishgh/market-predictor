"""Fixed-order selection, amendment, pair receipt and cohort coverage regressions."""
import json
from contextlib import nullcontext
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from market_predictor.canonical.store import file_sha256
from market_predictor.commands import investment_research
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.investment.corporate_actions import InvestmentCorporateActions
from market_predictor.investment.label_materialization import InvestmentSourceValidation
from market_predictor.investment.ranking import RANKING_SCORE
from market_predictor.investment.training import TARGET, investment_selection_diagnostics
from market_predictor.research import investment_selection_outcomes as module
from market_predictor.research.investment_features import ArtifactReference
from market_predictor.research.investment_labels import InvestmentLabelMaterializationRequest
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.action_evidence import CorporateActionEvidence
from tests.test_investment_label_materialization import Reader, decision, reference


def write(path: Path, value: Any) -> SourcePin:
    path.write_text(json.dumps(value), encoding="utf-8")
    return SourcePin(path=str(path), sha256=file_sha256(path))


def test_fixed_selection_replays_unknown_high_score_and_refuses_order_tamper() -> None:
    frame = pd.DataFrame({
        "decision_id": [f"decision-{index}" for index in range(11)], "security_id": [f"stock-{index:02}" for index in range(11)],
        "decision_session": "2024-05-28", RANKING_SCORE: range(11), "research_eligible": [True] * 10 + [False],
        TARGET: [0.1] * 10 + [None], "stock_net_return": [0.15] * 10 + [None], "spy_gross_return": [0.05] * 10 + [None],
        "qqq_gross_return": [0.06] * 10 + [None], "sector_gross_return": [0.07] * 10 + [None],
    })
    original = investment_selection_diagnostics(frame, RANKING_SCORE)["top_decile_by_date"]
    frozen = [{"decision_session": "2024-05-28", "security_ids": ["stock-10", "stock-09"]}]
    assert module._replay_ordered_selection(frame, RANKING_SCORE, original, frozen) == frozen
    assert original[0]["known_selected_outcomes"] == 1 and original[0]["mean_net_excess_vs_spy"] is None
    with pytest.raises(DataReadinessError, match="ordered selection"):
        module._replay_ordered_selection(frame, RANKING_SCORE, original, [{**frozen[0], "security_ids": ["stock-09", "stock-10"]}])


def test_incomplete_dates_refuse_full_period_mean_and_dedup_does_not_remove_placements() -> None:
    a, b = ("known", 63), ("missing", 63)
    outcomes = {a: {"status": "reused_original", "stock_net_return": .11, "spy_gross_return": .05, "net_excess_vs_spy": .06},
        b: {"status": "unresolved"}}
    one = {"decision_session": "2023-01-31", "security_ids": ["a"], "keys": [a]}
    two = {"decision_session": "2023-02-28", "security_ids": ["b"], "keys": [b]}
    results = module._compare_cohorts({"first": [one, two], "second": [one]}, outcomes)
    assert sum(view["selected_positions"] for view in results.values()) == 3
    assert results["first"]["complete_dates"] == 1 and results["first"]["mean_net_excess_vs_spy"] is None
    assert results["first"]["dates"][0]["mean_net_excess_vs_spy"] == .06
    assert results["first"]["dates"][1]["mean_net_excess_vs_spy"] is None
    assert results["second"]["mean_net_excess_vs_spy"] == .06


def test_currency_overlay_must_preserve_original_actions_and_only_add_selected_gaps(tmp_path: Path) -> None:
    old = write(tmp_path / "old.json", {"corroborations": [{"action_id": "prior", "rate": 1}]})
    new = write(tmp_path / "new.json", {"corroborations": [{"action_id": "prior", "rate": 1}, {"action_id": "selected"}]})
    request = module.InvestmentSelectionOutcomeRequest(
        schema_version="market_predictor.investment_selection_outcome_request", scope="retrospective_research",
        ranking_manifest=old, selection_inventory=old, label_corpus=old, label_request=old, currency_corroborations=new)
    original = InvestmentLabelMaterializationRequest(roster_manifest=ArtifactReference(path=old.path, sha256=old.sha256),
        target_config=ArtifactReference(path=old.path, sha256=old.sha256),
        currency_corroboration_manifest=ArtifactReference(path=old.path, sha256=old.sha256),
        nominal_unpaid_claim_valuation=True, evaluation_as_of_utc=datetime.now(UTC))
    rows = {("decision", 63): {"security_id": "a", "status": "unresolved",
        "original_missing_reasons": ["stock:action:cash_dividends:selected:dividend_usd_currency_unproven"]}}
    amended = module._amended_label_request(tmp_path, request, original, rows, {})
    assert amended.currency_corroboration_manifest is not None and amended.currency_corroboration_manifest.sha256 == new.sha256
    with pytest.raises(DataReadinessError, match="expands beyond selected"):
        module._amended_label_request(tmp_path, request, original, {}, {})
    changed = write(tmp_path / "changed.json", {"corroborations": [{"action_id": "prior", "rate": 2}]})
    with pytest.raises(DataReadinessError, match="replaces prior"):
        module._amended_label_request(tmp_path, request.model_copy(update={"currency_corroborations": changed}), original, rows, {})


def test_publication_retains_exact_pair_evidence_and_actual_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = write(tmp_path / "original.json", {"source_files": {}, "selection_sha256": "a" * 64, "nominal_unpaid_claim_valuation": True,
        "selections": {"view": [{"security_ids": ["issuer:test"]}]}})
    request = write(tmp_path / "request.json", {
        "schema_version": "market_predictor.investment_selection_outcome_request", "scope": "retrospective_research",
        **{name: original.model_dump(mode="json") for name in ("ranking_manifest", "selection_inventory", "label_corpus", "label_request")},
    })
    row = decision()
    key = (row["decision_id"], 63)
    outputs = {key: {**row, "status": "unresolved", "entry_time_utc": "2024-05-29T13:30:00Z",
        "economic_maturity_utc": "2024-08-27T20:00:00Z"}}
    cohorts = {"view": [{"decision_session": row["decision_session"], "security_ids": [row["security_id"]], "keys": [key]}]}
    monkeypatch.setattr(module, "_selected_rows", lambda *_: ({}, cohorts))
    monkeypatch.setattr(module, "_original_outcomes", lambda *_: outputs)
    monkeypatch.setattr(module, "_implementation", lambda: {})
    monkeypatch.setattr(module, "producer_origin", lambda *_: (None, None))
    monkeypatch.setattr(module, "guard", lambda *_, **__: None)
    monkeypatch.setattr("market_predictor.research.investment_features.guard", lambda *_: None)
    monkeypatch.setattr("market_predictor.research.investment_diagnostics.guard", lambda *_: None)
    monkeypatch.setattr(module, "heavy_job_lease", lambda *_, **__: nullcontext())
    inputs = SimpleNamespace(request=SimpleNamespace(decision_ids=None), policy=SimpleNamespace(decision_corrections=()),
        roster_rows=original, membership=original, rows=[{**row, "status": "eligible"}])
    monkeypatch.setattr(module, "read_investment_label_inputs", lambda *_: inputs)
    monkeypatch.setattr(module, "_amended_label_request", lambda *_: inputs.request)
    monkeypatch.setattr(module, "verified_investment_label_sources", lambda *_: nullcontext((None, None, {"source_files": {}})))
    action = SimpleNamespace(validate_sources=lambda *_: InvestmentSourceValidation((reference("actions"),)), recheck=lambda: None)
    def prepare_components(*args: Any) -> tuple[Any, dict[str, Any]]:
        policy_ref = args[-2]
        assert policy_ref.reference == original.path and policy_ref.artifact_sha256 == original.sha256
        assert policy_ref.interpretation_policy_sha256 == request.sha256
        projection = json_sha256({"records": {}, "unavailable": [], "source_files": {}, "replay": {}})
        empty_evidence = CorporateActionEvidence("a" * 64, "b" * 64, {}, (), {}, {}, projection)
        InvestmentCorporateActions(tmp_path, empty_evidence, ({}, ()), (), policy_ref, date(2025, 5, 30))
        return action, {}

    monkeypatch.setattr(module, "prepare_investment_label_components", prepare_components)
    monkeypatch.setattr(module, "InvestmentSourceReader", lambda *_: Reader())
    monkeypatch.setattr(module, "_validate_membership_sources", lambda **kwargs: InvestmentSourceValidation((kwargs["evidence"],)))
    output = tmp_path / "output"
    result = module.resolve_investment_selection_outcomes(root=tmp_path, request=Path(request.path),
        request_sha256=request.sha256, output=output)
    assert result["recovered_keys"] == 1 and result["unresolved_keys"] == 0
    recovered = json.loads((output / "outcomes.json").read_text())["rows"][0]
    path = Path(recovered["pair_proof"]["path"])
    assert path == output / "pairs" / key[0] / "63.json" and file_sha256(path) == recovered["pair_proof"]["sha256"]
    proof = json.loads(path.read_text())
    assert set(proof["specifications"]) == {"stock", "spy"} and proof["result"]["net_excess_vs_spy"] == pytest.approx(-.002)
    assert pd.Timestamp(recovered["outcome_available_at_utc"]) >= pd.Timestamp(proof["verified_at_utc"])
    assert file_sha256(Path(original.path)) == original.sha256


def test_cli_forwards_fixed_selection_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(investment_research, "resolve_investment_selection_outcomes",
        lambda **kwargs: calls.append(kwargs) or {"status": "fixture"})
    app = typer.Typer()
    investment_research.register_investment_research_commands(app)
    result = CliRunner().invoke(app, ["resolve-investment-selection-outcomes", "--root", str(tmp_path),
        "--request", str(tmp_path / "request"), "--request-sha256", "a" * 64, "--output", str(tmp_path / "output")])
    assert result.exit_code == 0, result.output
    assert calls == [{"root": tmp_path, "request": tmp_path / "request", "request_sha256": "a" * 64, "output": tmp_path / "output"}]
