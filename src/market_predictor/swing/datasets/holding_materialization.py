"""Compile selected raw observations and evidenced events into actual research lots."""
from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import date
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority
from market_predictor.resources import assert_memory_budget
from market_predictor.swing.contracts.holding_accounting import (
    CorporateActionEvent,
    EvidenceReference,
    HoldingSpecification,
    KnownMark,
    Mark,
    PaymentEvent,
    UnavailableMark,
)
from market_predictor.swing.contracts.holding_materialization import FixedHoldingBatch, FixedHoldingRequest
from market_predictor.swing.contracts.research import load_swing_research_contract
from market_predictor.swing.datasets.holding_raw_sources import read_bound_observations
from market_predictor.swing.datasets.symbol_corrected_sources import publish_or_verify_symbol_corrected_sources
from market_predictor.swing.datasets.symbol_corrections import inside, pinned_object
from market_predictor.swing.evaluation.holding_accounting import replay_holding

SelectionContext = Callable[[], AbstractContextManager[dict[str, Any]]]


def _verify_evidence(root: Path, evidence: tuple[EvidenceReference, ...], policy_sha256: str) -> None:
    for item in evidence:
        path = resolve_inside_authority(root, item.reference)
        if file_sha256(path) != item.artifact_sha256 or item.interpretation_policy_sha256 != policy_sha256:
            raise DataReadinessError("holding evidence bytes or interpretation policy differ")
        if item.available_at is not None and item.available_at > item.retrieved_at:
            raise DataReadinessError("evidenced availability cannot follow capture")


def _compile(root: Path, selection: dict[str, Any], request: FixedHoldingRequest,
    *, policy_sha256: str, research_sha256: str) -> dict[str, Any]:
    calendar = xcals.get_calendar("XNYS")
    day = pd.Timestamp(request.decision_session)
    if not calendar.is_session(day) or not date(2019, 7, 9) <= request.decision_session <= date(2024, 5, 13):
        raise DataReadinessError("holding decision is outside mature initial-fit sessions")
    sessions = calendar.sessions_window(day, 11)[1:]
    ends = tuple(calendar.session_close(session).to_pydatetime() for session in sessions)
    first, last = sessions[0].date(), sessions[-1].date()
    entry_at = calendar.session_open(sessions[0]).to_pydatetime()
    if len(ends) != 10 or last > date(2024, 5, 28):
        raise DataReadinessError("holding horizon crosses initial-fit boundary")
    classes = {request.initial_position_id: request.security_id}
    non_shares: set[str] = set()
    for event in request.events:
        _verify_evidence(root, event.evidence, policy_sha256)
        if isinstance(event, CorporateActionEvent):
            for leg in event.legs:
                if leg.kind == "tradable_shares":
                    classes[leg.position_id] = leg.security_id
                else:
                    non_shares.add(leg.position_id)
        elif isinstance(event, PaymentEvent):
            non_shares.add(event.pending_proceeds_id)
    sources: dict[tuple[str, date], tuple[Any, tuple[EvidenceReference, ...]]] = {}
    for binding in request.bindings:
        if classes.get(binding.position_id) != binding.security_id:
            raise DataReadinessError("source binding is not a declared owned share class")
        if binding.first_session < first or binding.last_session > last:
            raise DataReadinessError("position binding extends beyond the requested holding window")
        _verify_evidence(root, binding.ownership_evidence, policy_sha256)
        frame, references = read_bound_observations(root, selection, binding, first=first, last=last,
            interpretation_sha256=policy_sha256)
        for session in calendar.sessions_in_range(binding.first_session, binding.last_session):
            key = binding.position_id, session.date()
            if key in sources:
                raise DataReadinessError("overlapping holding position source bindings")
            observed = frame.loc[session.date()] if session.date() in frame.index else None
            evidence = (*binding.ownership_evidence, references[session.date()]) if observed is not None else ()
            sources[key] = observed, evidence
    initial = sources.get((request.initial_position_id, first))
    result: dict[str, Any] = {"decision_id": request.decision_id, "security_id": request.security_id,
        "specification": None, "diagnostic_replay": None, "materialization_gaps": [],
        "source_gaps": [{"code": "independent_source_admission_required", "reference_id": request.decision_id},
            *(gap.model_dump(mode="json") for gap in request.unresolved_source_gaps)],
        "reportable_gross_return": None, "reportable_net_return": None, "label_eligible": False,
        "source_admission_status": "independent_source_replay_required"}
    if initial is None or initial[0] is None or not bool(initial[0].outcome_observation_valid):
        code = ("ownership_unproven" if initial is None else
            "entry_observation_missing" if initial[0] is None else "entry_observation_invalid")
        result["materialization_gaps"].append({"code": code, "reference_id": request.initial_position_id})
        return result
    for identity in sorted(set(classes.values())):
        coverage = [item for item in request.action_coverage if item.security_id == identity
            and item.first_session <= first and item.last_session >= last]
        if not coverage:
            result["source_gaps"].append({"code": "action_coverage_unproven", "reference_id": identity})
    for coverage_item in request.action_coverage:
        _verify_evidence(root, coverage_item.evidence, policy_sha256)
    marks: list[Mark] = []
    for position in classes:
        for session, end in zip(sessions, ends, strict=True):
            source = sources.get((position, session.date()))
            reason = "ownership_unproven"
            if source is not None:
                observed, evidence = source
                reason = "observation_missing" if observed is None else "observation_invalid"
                if observed is not None and bool(observed.outcome_observation_valid):
                    marks.append(KnownMark(position_id=position, mark_at=end, value_per_unit=float(observed.close),
                        currency="USD", evidence=evidence))
                    continue
            marks.append(UnavailableMark(position_id=position, mark_at=end, reason=reason))
    for mark in request.non_share_marks:
        if mark.position_id not in non_shares or mark.mark_at not in ends:
            raise DataReadinessError("external mark must be for a declared non-share claim at an exact holding close")
        if isinstance(mark, KnownMark):
            _verify_evidence(root, mark.evidence, policy_sha256)
        marks.append(mark)
    spec = HoldingSpecification(research_contract_sha256=research_sha256, decision_id=request.decision_id,
        security_id=request.security_id, sector=request.sector, initial_position_id=request.initial_position_id,
        initial_entry_price=float(initial[0].open), initial_entry_timestamp=entry_at,
        price_basis="raw_with_no_adjustment", currency="USD", entry_evidence=initial[1],
        session_end_timestamps=ends, cost_prepaid_fraction=0.002, policy="fixed_horizon",
        events=request.events, marks=tuple(marks))
    outcome = replay_holding(spec)
    result["specification"] = spec.model_dump(mode="json")
    result["diagnostic_replay"] = outcome.model_dump(mode="json")
    mark_lookup = {(mark.position_id, mark.mark_at): mark for mark in spec.marks}
    for snapshot in outcome.snapshots:
        for residual in snapshot.residual_positions:
            residual_mark = mark_lookup.get((residual.position_id, snapshot.session_end_timestamp))
            if isinstance(residual_mark, UnavailableMark):
                result["materialization_gaps"].append({"code": residual_mark.reason,
                    "reference_id": f"{residual.position_id}:{snapshot.session_end_timestamp.isoformat()}"})
    # Hash-bound request facts are still interpretations, not independently admitted
    # source facts. They can never unlock reportable economics in this compiler.
    return result


def materialize_fixed_holdings(*, root: Path, request_path: Path, request_sha256: str,
    output: Path, selection_context: SelectionContext, expected_output_sha256: str | None = None) -> dict[str, Any]:
    """The context owns the existing workspace lease and reconstructs source selection.

    Dependency injection keeps provider/archive orchestration outside swing. The
    context must remain open through publication; no nested heavy lease is opened.
    """
    root = root.resolve()
    request_path, output = inside(root, request_path), inside(root, output)
    with selection_context() as selection:
        payload = pinned_object(request_path, request_sha256)
        batch = FixedHoldingBatch.model_validate_json(json.dumps(payload))
        selected_path = resolve_inside_authority(root, batch.source_selection.path)
        if pinned_object(selected_path, batch.source_selection.sha256) != selection:
            raise DataReadinessError("holding source selection differs from independent reconstruction")
        if selection.get("schema") != "market_predictor.swing_symbol_corrected_sources":
            raise DataReadinessError("holding compiler requires corrected source selection")
        if any(output.is_relative_to(inside(root, Path(segment["archive"]))) for segment in selection["segments"]):
            raise DataReadinessError("holding output cannot modify a selected raw archive")
        for pin in (batch.interpretation_policy, batch.research_contract):
            if file_sha256(resolve_inside_authority(root, pin.path)) != pin.sha256:
                raise DataReadinessError("holding compiler policy pin differs")
        contract = load_swing_research_contract(resolve_inside_authority(root, batch.research_contract.path))
        if contract.base_round_trip_cost_bps != 20.0:
            raise DataReadinessError("holding compiler requires the frozen 20 bps cost")
        rows = []
        for request in batch.requests:
            assert_memory_budget(hard_budget_gib=4.0, headroom_gib=0.75, stage="fixed holding materialization")
            rows.append(_compile(root, selection, request, policy_sha256=batch.interpretation_policy.sha256,
                research_sha256=contract.sha256()))
        pinned_object(request_path, request_sha256)
        pinned_object(selected_path, batch.source_selection.sha256)
        for pin in (batch.interpretation_policy, batch.research_contract):
            if file_sha256(resolve_inside_authority(root, pin.path)) != pin.sha256:
                raise DataReadinessError("holding compiler policy changed during materialization")
        checked: dict[str, str] = {}
        for row in rows:
            if row["specification"] is None:
                continue
            spec = HoldingSpecification.model_validate_json(json.dumps(row["specification"]))
            references = [*spec.entry_evidence, *(item for event in spec.events for item in event.evidence),
                *(item for mark in spec.marks if isinstance(mark, KnownMark) for item in mark.evidence)]
            for evidence in references:
                if evidence.reference in checked and checked[evidence.reference] != evidence.artifact_sha256:
                    raise DataReadinessError("conflicting source hashes within holding batch")
                checked[evidence.reference] = evidence.artifact_sha256
        for reference, digest in checked.items():
            if file_sha256(resolve_inside_authority(root, reference)) != digest:
                raise DataReadinessError("holding evidence changed before publication")
        package = Path(__file__).resolve().parents[2]
        implementation = {name: file_sha256(package / name) for name in (
            "swing/contracts/holding_materialization.py", "swing/datasets/holding_materialization.py",
            "swing/datasets/holding_raw_sources.py", "swing/contracts/holding_accounting.py",
            "swing/evaluation/holding_accounting.py", "swing/labels/holding_paths.py",
        )}
        result = {"schema": "market_predictor.fixed_holding_materialization", "request_sha256": request_sha256,
            "source_selection_sha256": batch.source_selection.sha256, "research_contract_sha256": contract.sha256(),
            "interpretation_policy_sha256": batch.interpretation_policy.sha256, "rows": rows,
            "implementation_files": implementation,
            "label_eligible": False, "accounting_eligible": False, "promotion_eligible": False,
            "status": "fixed_horizon_diagnostic_only", "exclusions_added": []}
        result["audit_sha256"] = json_sha256(result)
        publish_or_verify_symbol_corrected_sources(output, result, expected_sha256=expected_output_sha256)
        return result
