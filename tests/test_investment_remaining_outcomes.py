"""Exact action relevance and fixed survivor recovery regressions."""
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from market_predictor.canonical.store import file_sha256
from market_predictor.commands import investment_research
from market_predictor.core.errors import DataReadinessError
from market_predictor.investment.action_relevance import action_relevance_reason
from market_predictor.investment.label_materialization import InvestmentSourceValidation, compile_investment_spy_pair
from market_predictor.research import investment_labels
from market_predictor.research import investment_remaining_outcomes as module
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.reviewed_action_dates import ReviewedActionDate
from market_predictor.swing.datasets.reviewed_parent_units import ReviewedParentSpinUnits
from tests.test_investment_corporate_actions import SESSIONS, build, dividend
from tests.test_investment_exclusion_evaluation import VIEW, B, C, _read, _repin, _run, _write
from tests.test_investment_exclusion_evaluation import parent as parent
from tests.test_investment_label_materialization import Reader, decision, reference

NAME = {"id": "name", "old_cusip": "934423104", "new_cusip": "934423104", "old_symbol": "OLD",
    "new_symbol": "TEST", "process_date": "2024-05-23"}
SPIN = {"id": "spin", "source_cusip": "075887109", "new_cusip": "941848103", "source_symbol": "TEST",
    "new_symbol": "DAUGHTER", "source_rate": 1, "new_rate": .25, "process_date": "2024-07-25",
    "ex_date": "2024-07-25", "record_date": "2024-07-20", "payable_date": "2024-07-24"}


@pytest.mark.parametrize("change", [
    {"old_cusip": "934423105", "new_cusip": "934423105"}, {"new_cusip": "075887109"},
    {"old_symbol": "TEST ", "new_symbol": "OTHER"}, {"process_date": None}, {"cash_rate": .5},
    {"cancelled": True}, {"effective_date": "2024-05-23"},
])
def test_name_rule_requires_exact_same_cusip_name_only_record(change: dict[str, Any]) -> None:
    assert action_relevance_reason("name_changes", NAME, "TEST", SESSIONS[0], SESSIONS[-1], None, None)
    assert action_relevance_reason("name_changes", {**NAME, **change}, "TEST", SESSIONS[0], SESSIONS[-1], None, None) is None


def test_official_name_change_is_strict_before_and_after_with_same_ticker() -> None:
    fact = ReviewedActionDate(action_id="name", family="name_changes", retained_record_sha256="a" * 64,
        economic_dates={"effective_date": date(2024, 5, 23)}, official_receipts=(SourcePin(path="receipt.json", sha256="b" * 64),))
    record = {**NAME, "old_symbol": "TEST", "new_cusip": "075887109"}
    assert action_relevance_reason("name_changes", record, "TEST", SESSIONS[0], SESSIONS[0], fact, None) == (
        "holding_ends_before_official_identity_change")
    assert action_relevance_reason("name_changes", record, "TEST", SESSIONS[2], SESSIONS[-1], fact, None) == (
        "new_same_symbol_raw_window_after_official_identity_change")
    assert action_relevance_reason("name_changes", record, "TEST", SESSIONS[0], SESSIONS[1], fact, None) is None
    assert action_relevance_reason("name_changes", record, "TEST", SESSIONS[1], SESSIONS[-1], fact, None) is None
    renamed = {**record, "old_symbol": "OLD"}
    assert action_relevance_reason("name_changes", renamed, "TEST", SESSIONS[2], SESSIONS[-1], fact, None) is None
    assert action_relevance_reason("name_changes", renamed, "TEST", SESSIONS[0], SESSIONS[0], fact, None) is None
    assert action_relevance_reason("name_changes", renamed, "OLD", SESSIONS[0], SESSIONS[0], fact, None)
    assert action_relevance_reason("name_changes", {**record, "id": "other"}, "TEST",
        SESSIONS[0], SESSIONS[0], fact, None) is None


@pytest.mark.parametrize("change", [
    {"record_date": None}, {"payable_date": "20240724"}, {"effective_date": "2024-05-28"},
    {"due_bill_redemption_date": "2024-05-25"}, {"record_date": "2024-07-26"}, {"new_symbol": "TEST"},
    {"new_cusip": "075887109"}, {"new_rate": 0}, {"cancelled": True}, {"unknown_economic_date": "2024-05-25"},
])
def test_provider_spin_is_parent_only_strictly_before_every_economic_date(change: dict[str, Any]) -> None:
    assert action_relevance_reason("spin_offs", SPIN, "TEST", SESSIONS[0], SESSIONS[-1], None, None) == (
        "holding_ends_before_distribution_economic_dates")
    assert action_relevance_reason("spin_offs", SPIN, "DAUGHTER", SESSIONS[0], SESSIONS[-1], None, None) is None
    assert action_relevance_reason("spin_offs", {**SPIN, **change}, "TEST", SESSIONS[0], SESSIONS[-1], None, None) is None
    assert action_relevance_reason("spin_offs", SPIN, "TEST", SESSIONS[0], date(2024, 7, 20), None, None) is None
    assert action_relevance_reason("spin_offs", SPIN, "TEST", date(2024, 8, 1), date(2024, 8, 2), None, None) is None


def test_reviewed_parent_spin_preserves_before_and_fresh_ex_date_but_never_crossing() -> None:
    fact = ReviewedParentSpinUnits(action_id="spin", retained_record_sha256="a" * 64, source_symbol="TEST",
        ex_date=date(2024, 7, 25), parent_units_per_owned_unit=1.0, regular_way_before_ex_includes_distribution=True,
        official_receipts=(SourcePin(path="receipt.json", sha256="b" * 64),))
    assert action_relevance_reason("spin_offs", SPIN, "TEST", date(2024, 7, 22), date(2024, 7, 24), None, fact)
    assert action_relevance_reason("spin_offs", SPIN, "TEST", date(2024, 7, 25), date(2024, 7, 26), None, fact)
    assert action_relevance_reason("spin_offs", SPIN, "TEST", date(2024, 7, 24), date(2024, 7, 26), None, fact) is None
    assert action_relevance_reason("spin_offs", SPIN, "DAUGHTER", SESSIONS[0], SESSIONS[-1], None, fact) is None


@pytest.mark.parametrize("collision", ["none", "same_family", "cross_family"])
def test_only_unique_exact_provider_version_omits_name_gap(tmp_path: Path, collision: str) -> None:
    names = [NAME]
    dividend_records = [dict(dividend(), currency=None)]
    if collision == "same_family":
        names.append({**NAME, "process_date": "2024-05-24"})
    if collision == "cross_family":
        dividend_records.append(dividend("name"))
    engine = build(tmp_path, {"name_changes": names, "cash_dividends": dividend_records},
        relevance_action_ids=frozenset({"name"}))
    default = engine.validate_sources("issuer:test", "TEST", SESSIONS, "shares")
    reviewed = engine.validate_sources("issuer:test", "TEST", SESSIONS, "shares", reviewed_relevance=True)
    reason = "action:name_changes:name:missing_action_economic_date"
    assert reason in default.missing_reasons
    assert (reason in reviewed.missing_reasons) == (collision != "none")
    assert "action:cash_dividends:div:dividend_usd_currency_unproven" in reviewed.missing_reasons
    evidence = [ref for ref in reviewed.evidence if ";relevance=" in ref.record_locator]
    assert len(evidence) == (1 if collision == "none" else 0)
    if evidence:
        assert evidence[0].available_at == datetime(2026, 11, 7, tzinfo=UTC)
        assert file_sha256(tmp_path / evidence[0].reference) == evidence[0].artifact_sha256


@pytest.mark.parametrize("gap", ["none", "raw", "alias", "membership"])
def test_reviewed_callback_requires_exact_stock_raw_and_membership(gap: str) -> None:
    class GateReader(Reader):
        def read(self, security_id: str, ticker: str, sessions: tuple[date, ...], *, benchmark: bool) -> Any:
            observations = super().read(security_id, ticker, sessions, benchmark=benchmark)
            if not benchmark and gap == "raw":
                return replace(observations, missing_reasons=("raw_gap",))
            if not benchmark and gap == "alias":
                return replace(observations, provider_symbol="OTHER")
            return observations

    calls = []
    def reviewed(identity: str, *_: Any) -> InvestmentSourceValidation:
        calls.append(identity)
        return InvestmentSourceValidation((reference("reviewed"),))

    result = compile_investment_spy_pair(decision=decision(), horizon=63, reader=GateReader(),
        membership=InvestmentSourceValidation((reference("membership"),), ("membership_gap",) if gap == "membership" else ()),
        action_source_validation=lambda *_: InvestmentSourceValidation((reference("ordinary"),)),
        reviewed_action_source_validation=reviewed)
    assert calls == (["issuer:test"] if gap == "none" else [])
    if gap == "none":
        assert {row["reference"] for row in result["specifications"]["stock"]["entry_evidence"]} >= {"reviewed"}
        assert {row["reference"] for row in result["specifications"]["spy"]["entry_evidence"]} >= {"ordinary"}


@pytest.mark.parametrize("status", ["recovered", "unresolved", "unattempted"])
def test_incremental_publication_keeps_fixed_survivors_and_original_clocks(
    tmp_path: Path, parent: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, status: str,
) -> None:
    request, ancestor_manifest = parent
    fixture_pin = _write(tmp_path / "source.json", {"nominal_unpaid_claim_valuation": True})
    ancestor_request = {
        "schema_version": "market_predictor.investment_selection_outcome_request", "scope": "retrospective_research",
        **dict.fromkeys(("ranking_manifest", "selection_inventory", "label_corpus", "label_request"), fixture_pin),
        "stock_identity": None, "currency_corroborations": None,
    }
    original_request = _write(tmp_path / "original-request.json", ancestor_request)
    original_report = _read(ancestor_manifest)
    original_report.update(request=original_request, request_sha256=original_request["sha256"],
        **{k: v for k, v in ancestor_request.items() if k not in {"schema_version", "scope"}})
    _write(ancestor_manifest, original_report)
    outcome_path = ancestor_manifest.parent / "outcomes.json"
    outcomes = _read(outcome_path)
    target = next(row for row in outcomes["rows"] if row["decision_id"] == "apr-c")
    target.update(entry_time_utc="2024-05-01T13:30:00+00:00", economic_maturity_utc="2024-07-31T20:00:00+00:00",
        pair_missing_reasons=["stock:action:name_changes:name:missing_action_economic_date"
            if status != "unattempted" else "stock:unrelated_gap"])
    _write(outcome_path, outcomes)
    _repin(request, ancestor_manifest, "outcomes.json")
    exclusion_result = _run(tmp_path, request)
    exclusion_manifest = tmp_path / "sensitivity" / "_manifest.json"
    parent_rows = _read(exclusion_manifest.parent / "outcomes.json")["rows"]
    parent_views = _read(exclusion_manifest.parent / "comparisons.json")["views"]
    parent_hash = file_sha256(exclusion_manifest)
    new_request = _write(tmp_path / "remaining-request.json", {
        "schema_version": "market_predictor.investment_remaining_outcome_request",
        "scope": "post_observation_sensitivity", "parent_manifest": {"path": str(exclusion_manifest), "sha256": parent_hash},
    })
    assert parent_hash == exclusion_result["manifest_sha256"]
    roster = [{**decision(), "decision_id": row["decision_id"], "ticker": row["ticker"], "security_id": row["security_id"],
        "decision_session": row["decision_session"], "decision_time_utc": row["decision_session"] + "T20:05:00Z", "status": "eligible"}
        for row in parent_rows]
    inputs = SimpleNamespace(request=SimpleNamespace(decision_ids=None), policy=SimpleNamespace(decision_corrections=()),
        roster_rows=SourcePin.model_validate(fixture_pin), membership=SourcePin.model_validate(fixture_pin), rows=roster)
    monkeypatch.setattr(module, "heavy_job_lease", lambda *_, **__: nullcontext())
    monkeypatch.setattr(module, "guard", lambda *_, **__: None)
    monkeypatch.setattr(module, "read_investment_label_inputs", lambda *_: inputs)
    monkeypatch.setattr(module, "InvestmentSourceReader", lambda *_: Reader())
    monkeypatch.setattr(module, "verified_investment_label_sources", lambda *_: nullcontext((None, None, {"source_files": {}})))
    monkeypatch.setattr("market_predictor.research.investment_selection_outcomes._validate_membership_sources",
        lambda **kwargs: InvestmentSourceValidation((kwargs["evidence"],)))

    def amendment(*args: Any) -> Any:
        assert set(args[-2]) == {("jan-c", 63), ("feb-a", 63), ("apr-c", 63)}
        assert all(row["status"] == "unresolved" for row in args[-2].values())
        return inputs.request

    monkeypatch.setattr(module, "_amended_label_request", amendment)
    calls = []
    def components(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        assert kwargs["relevance_action_ids"] == ({"name"} if status != "unattempted" else set())
        policy_ref = args[-2]
        evidence = policy_ref.model_copy(update={"record_locator": "family=name_changes;action_id=name;relevance=reviewed"})

        def validate(identity: str, *_: Any, reviewed_relevance: bool = False) -> InvestmentSourceValidation:
            calls.append((identity, reviewed_relevance))
            return InvestmentSourceValidation((evidence if reviewed_relevance else policy_ref,),
                ("companion_gap",) if reviewed_relevance and status == "unresolved" else ())

        return SimpleNamespace(validate_sources=validate, reviewed_relevance_ids=lambda *_: frozenset({"name"}), recheck=lambda: None), {}

    monkeypatch.setattr(module, "prepare_investment_label_components", components)
    output = tmp_path / "remaining"
    result = module.resolve_remaining_investment_outcomes(root=tmp_path, request=Path(new_request["path"]),
        request_sha256=new_request["sha256"], output=output)
    child_rows = _read(output / "outcomes.json")["rows"]
    attempts = _read(output / "attempts.json")["rows"]
    assert [row["decision_id"] for row in child_rows] == [row["decision_id"] for row in parent_rows]
    assert all(row == previous for row, previous in zip(child_rows, parent_rows, strict=True) if row["decision_id"] != "apr-c")
    assert len(attempts) == 1 and attempts[0]["status"] == status and attempts[0]["decision_id"] == "apr-c"
    assert calls == ([(C, True), ("benchmark:SPY", False)] if status != "unattempted" else [])
    assert result["newly_recovered_keys"] == int(status == "recovered")
    changed = next(row for row in child_rows if row["decision_id"] == "apr-c")
    view = _read(output / "comparisons.json")["views"][VIEW]
    assert view["mean_net_excess_vs_spy"] is None
    assert view["dates"][3]["security_ids"] == [C, B] and view["dates"][3]["removed_positions"] == []
    assert view["dates"][2] == parent_views[VIEW]["dates"][2]
    if status == "recovered":
        proof = Path(changed["pair_proof"]["path"])
        assert proof == output / "pairs" / "apr-c" / "63.json"
        assert changed["pair_proof"]["sha256"] == file_sha256(proof)
        assert changed["net_excess_vs_spy"] == pytest.approx(-.002)
        assert datetime.fromisoformat(changed["verified_at_utc"]) >= datetime.fromisoformat(target["verified_at_utc"])
        assert view["dates"][3]["complete"]
    else:
        assert changed == target
        assert not view["dates"][3]["complete"]
    assert file_sha256(exclusion_manifest) == parent_hash
    with pytest.raises(FileExistsError):
        module.resolve_remaining_investment_outcomes(root=tmp_path, request=Path(new_request["path"]),
            request_sha256=new_request["sha256"], output=output)


def test_repacked_survivor_clock_is_refused(tmp_path: Path, parent: tuple[Path, Path]) -> None:
    request, _ = parent
    _run(tmp_path, request)
    manifest = tmp_path / "sensitivity" / "_manifest.json"
    child = manifest.parent / "outcomes.json"
    rows = _read(child)
    rows["rows"][0]["verified_at_utc"] = "2026-11-11T00:00:00Z"
    child_pin = _write(child, rows)
    report = _read(manifest)
    report["files"]["outcomes.json"] = child_pin["sha256"]
    manifest_pin = SourcePin.model_validate(_write(manifest, report))
    with pytest.raises(DataReadinessError, match="original clocks"):
        module._read_survivors(tmp_path, manifest_pin, {})


def test_action_source_merge_cannot_repin_a_previously_consumed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin = SourcePin.model_validate(_write(tmp_path / "original.json", {"nominal_unpaid_claim_valuation": True}))
    monkeypatch.setattr(investment_labels, "InvestmentCorporateActions",
        lambda *args, **kwargs: SimpleNamespace(source_files={"original.json": "a" * 64}))
    parsed = SimpleNamespace(currency_manifest=None, currency_corroboration_manifest=None,
        dividend_amount_corrections_manifest=None)
    policy = SimpleNamespace(official_windows=(), numerical_end=SESSIONS[-1])
    pins = {pin.path: pin.sha256}
    with pytest.raises(DataReadinessError, match="source pins conflict"):
        investment_labels.prepare_investment_label_components(tmp_path, parsed, policy, pin, None, None,
            {"action_index": ({}, ())}, pins, reference("policy"), "a" * 64)
    assert pins == {pin.path: pin.sha256}


def test_cli_forwards_remaining_outcome_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(investment_research, "resolve_remaining_investment_outcomes",
        lambda **kwargs: calls.append(kwargs) or {"status": "fixture"})
    app = typer.Typer()
    investment_research.register_investment_research_commands(app)
    result = CliRunner().invoke(app, ["resolve-remaining-investment-outcomes", "--root", str(tmp_path),
        "--request", "request.json", "--request-sha256", "a" * 64, "--output", "remaining"])
    assert result.exit_code == 0, result.output
    assert calls == [{"root": tmp_path, "request": Path("request.json"), "request_sha256": "a" * 64, "output": Path("remaining")}]

