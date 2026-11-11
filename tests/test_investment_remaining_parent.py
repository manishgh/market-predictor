"""Pinned additive survivor ledger and mutation regressions."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.investment.action_relevance import RECIPIENT_PRE_DISTRIBUTION_RELEVANCE
from market_predictor.investment.label_materialization import InvestmentSourceValidation, compile_investment_spy_pair
from market_predictor.research import investment_remaining_parent as ledger
from market_predictor.research.investment_exclusion_evaluation import _coverage
from market_predictor.research.investment_selection_outcomes import publish_selected_pair
from market_predictor.research.investment_training import FLAGS
from market_predictor.swing.contracts.holding_materialization import SourcePin
from tests.test_investment_exclusion_evaluation import VIEW, B, _read, _repin, _run, _write
from tests.test_investment_exclusion_evaluation import parent as parent
from tests.test_investment_label_materialization import Reader, decision, reference


def _pin(path: Path) -> SourcePin:
    return SourcePin(path=str(path), sha256=file_sha256(path))


class _Reader(Reader):
    def __init__(self, verified: datetime) -> None:
        super().__init__()
        self.verified = verified

    def read(self, *args: Any, **kwargs: Any) -> Any:
        observations = super().read(*args, **kwargs)
        return replace(observations, evidence={day: item.model_copy(update={
            "retrieved_at": self.verified, "available_at": self.verified,
        }) for day, item in observations.evidence.items()})


def _write_ledger(
    root: Path, parent_pin: SourcePin, path: Path, recovery: str, verified: datetime,
    dates: SourcePin, recipient: SourcePin | None = None,
) -> Path:
    pins: dict[str, str] = {}
    before = ledger.read_remaining_outcome_parent(root, parent_pin, pins)
    if recipient is not None:
        pins[recipient.path] = recipient.sha256
    request = ledger.InvestmentRemainingOutcomeRequest(
        schema_version="market_predictor.investment_remaining_outcome_request", scope="post_observation_sensitivity",
        parent_manifest=parent_pin, reviewed_action_dates=dates, existing_recipient_stock_identity=recipient,
    )
    request_pin = SourcePin.model_validate(_write(root / (path.name + "-request.json"), request.model_dump(mode="json")))
    pins[request_pin.path] = request_pin.sha256
    rows = dict(before.rows)
    attempts = []
    path.mkdir()
    for key, previous in before.rows.items():
        if previous["status"] != "unresolved":
            continue
        attempt = {name: previous[name] for name in ledger.ATTEMPT_IDENTITY}
        attempts.append(attempt)
        if key[0] != recovery:
            attempt.update(status="unattempted", relevance_action_ids=[], attempted_at_utc=None, missing_reasons=[])
            continue
        family, action = ("spin_offs", "spin") if recipient is not None else ("name_changes", "name")
        reason = RECIPIENT_PRE_DISTRIBUTION_RELEVANCE if recipient is not None else "reviewed"
        evidence = reference(request_pin.path).model_copy(update={
            "artifact_sha256": request_pin.sha256, "interpretation_policy_sha256": request_pin.sha256,
            "record_locator": f"family={family};action_id={action};relevance={reason}",
            "retrieved_at": verified, "available_at": verified,
        })
        decision_fields = {**decision(), **{name: previous[name] for name in (
            "decision_id", "decision_session", "security_id", "ticker",
        )}}
        decision_fields["decision_time_utc"] = xcals.get_calendar("XNYS").session_close(previous["decision_session"]).isoformat()
        validation = InvestmentSourceValidation((evidence,))
        pair = compile_investment_spy_pair(decision=decision_fields, horizon=63, reader=_Reader(verified),
            membership=validation, action_source_validation=lambda *_, source=validation: source)
        recovered = dict(previous)
        publish_selected_pair(path, path, key, recovered, pair, verified, {})
        rows[key] = recovered
        attempt.update(status="recovered", relevance_action_ids=[action],
            attempted_at_utc=(verified + timedelta(seconds=1)).isoformat(), missing_reasons=[])
    views = ledger.compare_survivor_cohorts(before, rows)
    sidecars: dict[str, dict[str, Any]] = {
        "outcomes.json": {"rows": list(rows.values())},
        "comparisons.json": {"views": views},
        "attempts.json": {"rows": attempts},
    }
    files = {name: _write(path / name, content)["sha256"] for name, content in sidecars.items()}
    recovered_count = sum(attempt["status"] == "recovered" for attempt in attempts)
    manifest = path / "_manifest.json"
    _write(manifest, {
        "schema": ledger.SCHEMA, "scope": request.scope, "status": "complete_with_explicit_gaps",
        "request": request_pin.model_dump(mode="json"), **request.model_dump(mode="json", exclude={"schema_version", "scope"}),
        "parent_outcomes": before.outcomes_pin.model_dump(mode="json"),
        "parent_comparisons": before.comparisons_pin.model_dump(mode="json"),
        "ancestor_manifest": before.ancestor_pin.model_dump(mode="json"),
        **{name: before.manifest[name] for name in (
            "parent_selection_sha256", "exclusions", "comparison_views", "comparison_cohorts", "empty_cohorts",
        )},
        "round_trip_stock_cost_bps": 20, "benchmark_basis": "stock_net_minus_spy_gross",
        "verified_at_utc": verified.isoformat(), "completed_at_utc": (verified + timedelta(seconds=2)).isoformat(),
        "source_files": pins, "selection_changed": False, "models_refitted": False, "original_labels_changed": False,
        "training_ready": False, **FLAGS, "files": files,
        "attempted_keys": recovered_count, "unattempted_keys": len(attempts) - recovered_count,
        "newly_recovered_keys": recovered_count, "still_unresolved_keys": len(attempts) - recovered_count,
        "coverage": {"before": _coverage(before.rows.values(), before.comparisons), "after": _coverage(rows.values(), views)},
    })
    return manifest


@pytest.fixture
def chain(tmp_path: Path, parent: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(ledger, "guard", lambda *_, **__: None)
    request, ancestor = parent
    verified = datetime.now(UTC) - timedelta(minutes=1)
    selection = _read(request)
    selection["exclusions"] = [{"ticker": "B", "security_id": B, "reason": "user_requested_exclusion"}]
    _write(request, selection)
    path = ancestor.parent / "outcomes.json"
    outcomes = _read(path)
    calendar = xcals.get_calendar("XNYS")
    for row in outcomes["rows"]:
        if row["status"] != "unresolved":
            continue
        sessions = calendar.sessions_window(row["decision_session"], 64)[1:]
        reason = ("name_changes:name:missing_action_economic_date" if row["decision_id"] == "feb-a"
            else "spin_offs:spin:unbounded_entitlement_or_due_bill_dates")
        row.update(entry_time_utc=calendar.session_open(sessions[0]).isoformat(),
            economic_maturity_utc=calendar.session_close(sessions[-1]).isoformat(),
            verified_at_utc=(verified - timedelta(days=1)).isoformat(), pair_missing_reasons=[f"stock:action:{reason}"])
    _write(path, outcomes)
    _repin(request, ancestor, path.name)
    _run(tmp_path, request)
    original = _pin(tmp_path / "sensitivity" / "_manifest.json")
    dates = SourcePin.model_validate(_write(tmp_path / "dates.json", {"records": []}))
    recipient = SourcePin.model_validate(_write(tmp_path / "stock-identity.json", {"proofs": []}))
    first = _write_ledger(tmp_path, original, tmp_path / "first", "feb-a", verified, dates)
    second = _write_ledger(tmp_path, _pin(first), tmp_path / "second", "apr-c", verified + timedelta(seconds=10), dates, recipient)
    return first, second


def test_additive_children_preserve_prior_proof_clocks_and_ancestor_mask(
    tmp_path: Path, chain: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = chain
    before = ledger.read_remaining_outcome_parent(tmp_path, _pin(first), {})
    pins: dict[str, str] = {}
    guarded: list[tuple[str, bool]] = []
    monkeypatch.setattr(ledger, "guard", lambda stage, *, reclaim_unused_memory: guarded.append((stage, reclaim_unused_memory)))
    after = ledger.read_remaining_outcome_parent(tmp_path, _pin(second), pins)
    assert guarded == [("remaining ledger sidecars", False), ("remaining ledger pair proof", False)] * 2
    assert after.request is not None and after.request.existing_recipient_stock_identity is not None
    assert list(before.rows) == list(after.rows)
    assert after.original_unknowns == before.original_unknowns
    assert set(after.original_unknowns) == {("jan-c", 63), ("feb-a", 63), ("apr-c", 63)}
    assert before.rows["apr-c", 63]["status"] == "unresolved"
    assert after.rows["apr-c", 63]["status"] == "recovered"
    for key, row in before.rows.items():
        if key != ("apr-c", 63):
            assert after.rows[key] == row
    assert pins[str(first.parent / "pairs" / "feb-a" / "63.json")] == before.rows["feb-a", 63]["pair_proof"]["sha256"]
    assert len([path for path in pins if "pairs" in Path(path).parts]) == 2
    assert after.comparisons[VIEW]["mean_net_excess_vs_spy"] is None
    assert after.comparisons[VIEW]["dates"][-1]["complete"] is False


def _write_sidecar(manifest: Path, name: str, value: dict[str, Any]) -> None:
    report = _read(manifest)
    report["files"][name] = _write(manifest.parent / name, value)["sha256"]
    _write(manifest, report)


@pytest.mark.parametrize(("poison", "message"), [
    ("known_clock", "existing survivor"),
    ("known_proof", "existing survivor"),
    ("survivor_order", "existing survivor"),
    ("recovered_metadata", "frozen key"),
    ("attempt_inventory", "each prior unknown"),
    ("attempt_identity", "attempt order"),
    ("attempt_clock", "outside the verification window"),
    ("attempt_unrelated", "recorded eligible action"),
    ("attempt_count", "attempt counters"),
    ("parent_pin", "parent or ancestor"),
    ("ancestor_pin", "parent or ancestor"),
    ("scope", "scope"),
    ("cohort_order", "ordered cohort"),
    ("proof_path", "named sibling"),
    ("proof_hash", "hash differs"),
    ("proof_request", "request, policy"),
    ("proof_policy", "request, policy"),
    ("proof_interval", "identity or interval"),
    ("proof_return", "returns or availability"),
    ("proof_clock", "verification or availability clock"),
    ("extra_pair", "pair proof inventory"),
    ("extra_sidecar", "exact named sidecar"),
])
def test_repacked_remaining_ledger_is_refused(
    tmp_path: Path, chain: tuple[Path, Path], poison: str, message: str,
) -> None:
    _, manifest = chain
    path = manifest.parent
    outcomes = _read(path / "outcomes.json")
    target = next(row for row in outcomes["rows"] if row["decision_id"] == "apr-c")
    known = next(row for row in outcomes["rows"] if row["decision_id"] == "feb-a")
    if poison in {"known_clock", "known_proof", "survivor_order", "recovered_metadata", "proof_path", "proof_hash"}:
        if poison == "known_clock":
            known["verified_at_utc"] = "2024-12-30T00:00:00Z"
        elif poison == "known_proof":
            known["pair_proof"]["sha256"] = "a" * 64
        elif poison == "survivor_order":
            outcomes["rows"].reverse()
        elif poison == "recovered_metadata":
            target["original_status"] = "source_validated"
        elif poison == "proof_path":
            target["pair_proof"]["path"] = known["pair_proof"]["path"]
        else:
            target["pair_proof"]["sha256"] = "a" * 64
        _write_sidecar(manifest, "outcomes.json", outcomes)
    elif poison in {"attempt_inventory", "attempt_identity", "attempt_clock", "attempt_unrelated"}:
        attempts = _read(path / "attempts.json")
        row = attempts["rows"][0]
        if poison == "attempt_inventory":
            attempts["rows"].append(dict(row))
        elif poison == "attempt_identity":
            row["decision_session"] = "2024-05-01"
        elif poison == "attempt_clock":
            row["attempted_at_utc"] = "2024-12-30T00:00:00Z"
        else:
            row["relevance_action_ids"] = ["unrelated"]
        _write_sidecar(manifest, "attempts.json", attempts)
    elif poison in {"attempt_count", "parent_pin", "ancestor_pin", "scope"}:
        report = _read(manifest)
        if poison == "attempt_count":
            report["attempted_keys"] += 1
        elif poison == "parent_pin":
            report["parent_outcomes"]["sha256"] = "a" * 64
        elif poison == "ancestor_pin":
            report["ancestor_manifest"]["sha256"] = "a" * 64
        else:
            report["scope"] = "retrospective_research"
        _write(manifest, report)
    elif poison == "cohort_order":
        comparison = _read(path / "comparisons.json")
        comparison["views"][VIEW]["dates"].reverse()
        _write_sidecar(manifest, "comparisons.json", comparison)
    elif poison == "extra_pair":
        _write(path / "pairs" / "extra.json", {})
    elif poison == "extra_sidecar":
        _write(path / "extra.json", {})
    else:
        pair_path = Path(target["pair_proof"]["path"])
        proof = _read(pair_path)
        if poison == "proof_request":
            proof["result"]["request_sha256"] = "a" * 64
        elif poison == "proof_policy":
            proof["result"]["policy_sha256"] = "a" * 64
        elif poison == "proof_interval":
            proof["result"]["entry_time_utc"] = "2024-05-01T15:00:00Z"
        elif poison == "proof_return":
            target["stock_net_return"] += .01
        else:
            proof["verified_at_utc"] = "2024-12-30T00:00:00Z"
        target["pair_proof"] = _write(pair_path, proof)
        _write_sidecar(manifest, "outcomes.json", outcomes)
    with pytest.raises(DataReadinessError, match=message):
        ledger.read_remaining_outcome_parent(tmp_path, _pin(manifest), {})


@pytest.mark.parametrize(("poison", "message"), [
    ("cycle", "cycle"),
    ("outside", "escapes repository"),
    ("carried_dates", "carried action authority"),
    ("no_new_role", "added no completed cash"),
])
def test_remaining_lineage_refuses_bad_links_or_repeated_authority(
    tmp_path: Path, chain: tuple[Path, Path], poison: str, message: str,
) -> None:
    _, manifest = chain
    report = _read(manifest)
    request_path = Path(report["request"]["path"])
    request = _read(request_path)
    if poison in {"cycle", "outside"}:
        request["parent_manifest"]["path"] = str(manifest) if poison == "cycle" else "../outside/_manifest.json"
    elif poison == "carried_dates":
        request["reviewed_action_dates"] = _write(tmp_path / "different-dates.json", {"records": []})
    else:
        request["existing_recipient_stock_identity"] = None
    report["request"] = _write(request_path, request)
    report["source_files"][str(request_path)] = report["request"]["sha256"]
    report.update({name: value for name, value in request.items() if name not in {"schema_version", "scope"}})
    _write(manifest, report)
    with pytest.raises(DataReadinessError, match=message):
        ledger.read_remaining_outcome_parent(tmp_path, _pin(manifest), {})



@pytest.fixture
def failed_parent(tmp_path: Path, chain: tuple[Path, Path]) -> ledger.SurvivorOutcomes:
    first, second = chain
    report = _read(first)
    attempts = _read(first.parent / "attempts.json")
    failed = next(attempt for attempt in attempts["rows"] if attempt["decision_id"] == "apr-c")
    failed.update(status="unresolved", relevance_action_ids=["spin"],
        attempted_at_utc=(datetime.fromisoformat(report["verified_at_utc"]) + timedelta(seconds=1)).isoformat(),
        missing_reasons=["stock:action:cash_dividends:currency-target:dividend_usd_currency_unproven"])
    _write_sidecar(first, "attempts.json", attempts)
    report = _read(first)
    report["attempted_keys"] += 1
    report["unattempted_keys"] -= 1
    _write(first, report)
    alternate = _read(second)
    carry = _write_ledger(tmp_path, _pin(first), tmp_path / "carry", "", datetime.fromisoformat(alternate["verified_at_utc"]),
        SourcePin.model_validate(alternate["reviewed_action_dates"]),
        SourcePin.model_validate(alternate["existing_recipient_stock_identity"]))
    parent = ledger.read_remaining_outcome_parent(tmp_path, _pin(carry), {})
    assert parent.rows["apr-c", 63]["pair_missing_reasons"] == ["stock:action:spin_offs:spin:unbounded_entitlement_or_due_bill_dates"]
    latest = parent.latest_failed_attempts["apr-c", 63]
    assert latest.attempts_pin == _pin(first.parent / "attempts.json")
    assert latest.attempted_at_utc == datetime.fromisoformat(failed["attempted_at_utc"])
    assert parent.latest_reasons(("apr-c", 63)) == tuple(failed["missing_reasons"])
    return parent


def _currency_context(
    root: Path, parent: ledger.SurvivorOutcomes, *, owner_id: str = "cik:0000000003",
) -> tuple[ledger.SurvivorOutcomes, dict[str, Any], dict[str, Any]]:
    source = _write(root / "retained-source.json", {})
    owner = _write(root / "dated-class.json", {
        "security_id": owner_id, "ticker": "C", "sec_cik": "0000000003", "security_title": "Common Stock",
        "identity_authority": source, "membership_authority": source,
        "anchor": {"collection_manifest": source, "receipt_artifact": source, "unit_id": "retained",
            "body_sha256": "c" * 64, "filing_metadata": source},
    })
    captured = {"body": source, "receipt": source}
    item = {"method": "exact_tax_event", "action_id": "currency-target", "original_record_sha256": "a" * 64,
        "symbol": "C", "ex_date": "2024-05-17", "currency": "USD", "tax_value_difference": "0",
        "tax": {"source": captured, "gross_context": captured, "isin": "US7170811035", "entry_index": 0, "gross_amount": "1.25"},
        "identity": {"kind": "stock_composite", "cusip": "717081103", "proof": owner,
            "cusip_provenance": "provider_original_cash_action"}}
    prior = {**item, "action_id": "existing-assertion"}
    content = {"schema": ledger.CURRENCY_SCHEMA, "corroborations": [prior]}
    baseline = _write(root / "base-currency.json", content)
    parent = replace(parent, ancestor={**parent.ancestor, "currency_corroborations": baseline})
    return parent, content, item


def _prepare_currency(
    root: Path, parent: ledger.SurvivorOutcomes, content: dict[str, Any], pins: dict[str, str],
) -> tuple[ledger.InvestmentRemainingOutcomeRequest, ledger.RemainingEvidence]:
    assert parent.request is not None
    currency = SourcePin.model_validate(_write(root / "new-currency.json", content))
    request = parent.request.model_copy(update={"currency_corroborations": currency})
    return request, ledger.prepare_remaining_evidence(root, parent, request, pins)


@pytest.mark.parametrize("owner_id", ["cik:0000000003", "cik:0000000009"])
def test_currency_candidates_use_latest_pinned_failure_and_stable_owner(
    tmp_path: Path, failed_parent: ledger.SurvivorOutcomes, owner_id: str,
) -> None:
    parent, manifest, item = _currency_context(tmp_path, failed_parent, owner_id=owner_id)
    manifest["corroborations"].append(item)
    _, evidence = _prepare_currency(tmp_path, parent, manifest, {})
    key = ("apr-c", 63)
    assert evidence.candidate_ids(parent, key) == ({"currency-target"} if owner_id.endswith("3") else set())
    assert evidence.candidate_ids(replace(parent, latest_failed_attempts={}), key) == set()
    for poison in ("spy:action:cash_dividends:currency-target:dividend_usd_currency_unproven",
                   "stock:action:cash_dividends:another-id:dividend_usd_currency_unproven",
                   "stock:action:cash_dividends:currency-target:ordinary_domestic_cash_entitlement_unproven"):
        latest = replace(parent.latest_failed_attempts[key], missing_reasons=(poison,))
        assert evidence.candidate_ids(replace(parent, latest_failed_attempts={key: latest}), key) == set()


@pytest.mark.parametrize(("poison", "message"), [
    ("remove", "ordered prefix"),
    ("replace", "ordered prefix"),
    ("duplicate", "repeats an action ID"),
    ("shape", "manifest shape"),
    ("nested_pin", "conflicting source pins"),
    ("carried", "added no completed cash"),
    ("remove_carried", "removed carried currency"),
])
def test_currency_manifest_is_cumulative_and_append_only(
    tmp_path: Path, failed_parent: ledger.SurvivorOutcomes, poison: str, message: str,
) -> None:
    parent, manifest, item = _currency_context(tmp_path, failed_parent)
    if poison == "remove":
        manifest["corroborations"] = [item]
    elif poison == "replace":
        manifest["corroborations"][0] = {**manifest["corroborations"][0], "original_record_sha256": "b" * 64}
        manifest["corroborations"].append(item)
    elif poison == "duplicate":
        manifest["corroborations"].append(dict(item, action_id="existing-assertion"))
    elif poison == "shape":
        manifest["unreviewed"] = True
    elif poison == "nested_pin":
        tax = item["tax"]
        item["tax"] = {**tax, "source": {**tax["source"], "body": {**tax["source"]["body"], "sha256": "b" * 64}}}
        manifest["corroborations"].append(item)
    else:
        manifest["corroborations"].append(item)
        request, _ = _prepare_currency(tmp_path, parent, manifest, {})
        with pytest.raises(DataReadinessError, match=message):
            ledger.prepare_remaining_evidence(tmp_path, replace(parent, request=request),
                request if poison == "carried" else request.model_copy(update={"currency_corroborations": None}), {})
        return
    with pytest.raises(DataReadinessError, match=message):
        _prepare_currency(tmp_path, parent, manifest, {})


@pytest.mark.parametrize("poison", [
    "none", "entry_only", "entry_missing", "amount", "event_time", "stock_owner", "claim_mark", "wrong_pin",
    "wrong_policy", "backdate", "ex_open", "ex_open_claim",
])
def test_positive_currency_proof_is_bound_to_actual_stock_entitlement(
    tmp_path: Path, failed_parent: ledger.SurvivorOutcomes, poison: str,
) -> None:
    parent, manifest, item = _currency_context(tmp_path, failed_parent)
    manifest["corroborations"].append(item)
    pins: dict[str, str] = {}
    request, evidence = _prepare_currency(tmp_path, parent, manifest, pins)
    request_pin = SourcePin.model_validate(_write(tmp_path / "positive-request.json", request.model_dump(mode="json")))
    assert evidence.currency_pin is not None
    verified = datetime.now(UTC) - timedelta(seconds=1)
    ref = reference(evidence.currency_pin.path).model_copy(update={
        "artifact_sha256": evidence.currency_pin.sha256, "interpretation_policy_sha256": request_pin.sha256,
        "record_locator": "currency corroboration;action_id=currency-target;original_record_sha256=" + "a" * 64,
        "available_at": verified, "retrieved_at": verified,
    }).model_dump(mode="json")
    action = "holding:stock:shares:action:currency-target"
    entry = "2024-05-01T13:30:00Z"
    ex_open = "2024-05-17T13:30:00Z"
    security = "cik:0000000003"
    event = {"kind": "corporate_action", "event_id": action, "treatment": "distribution", "owned_security_id": security,
        "effective_at": ex_open, "evidence": [ref],
        "legs": [{"kind": "unpaid_proceeds", "position_id": action + ":unpaid", "security_id": security,
            "currency": "USD", "face_value_per_unit": 1.0, "units_per_owned_unit": 1.25}]}
    mark = {"status": "known", "position_id": action + ":unpaid", "currency": "USD", "value_per_unit": 1.0, "evidence": [ref]}
    specification = {"security_id": security, "initial_position_id": "holding:stock:shares", "initial_entry_timestamp": entry,
        "entry_evidence": [ref], "events": [event], "marks": [mark]}
    if poison == "entry_only":
        event["evidence"] = []
    elif poison == "entry_missing":
        specification["entry_evidence"] = []
    elif poison == "amount":
        event["legs"][0]["units_per_owned_unit"] = 9.0
    elif poison == "event_time":
        event["effective_at"] = "2024-05-17T14:00:00Z"
    elif poison == "stock_owner":
        event["owned_security_id"] = "cik:0000000009"
    elif poison == "claim_mark":
        mark["evidence"] = []
    elif poison == "wrong_pin":
        ref["artifact_sha256"] = "b" * 64
    elif poison == "wrong_policy":
        ref["interpretation_policy_sha256"] = "b" * 64
    elif poison == "backdate":
        ref["available_at"] = "2024-05-17T14:00:00Z"
    elif poison in {"ex_open", "ex_open_claim"}:
        specification["initial_entry_timestamp"] = ex_open
        if poison == "ex_open":
            specification.update(events=[], marks=[])
    if poison in {"none", "ex_open"}:
        ledger.verify_remaining_trigger_evidence(tmp_path, request_pin, evidence, specification, ["currency-target"], pins,
            verified_at=verified, outcome_available_at=verified)
    else:
        with pytest.raises(DataReadinessError, match="positive manifest proof|ex-open entry"):
            ledger.verify_remaining_trigger_evidence(tmp_path, request_pin, evidence, specification, ["currency-target"], pins,
                verified_at=verified, outcome_available_at=verified)

