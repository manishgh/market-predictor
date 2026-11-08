"""Observed split/entitlement terms with explicit nominal unpaid-claim valuation."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import exchange_calendars as xcals

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority
from market_predictor.investment.dividend_currencies import CurrencyCorroboration
from market_predictor.investment.label_materialization import InvestmentSourceValidation
from market_predictor.swing.contracts.corrected_outcomes import OfficialWindow
from market_predictor.swing.contracts.holding_accounting import CorporateActionEvent, EvidenceReference, KnownMark, PositionLeg
from market_predictor.swing.datasets.action_evidence import CorporateActionEvidence
from market_predictor.swing.datasets.action_windows import ActionWindow
from market_predictor.swing.datasets.backtest_corporate_actions import _selected_pages


@dataclass(frozen=True)
class _Term:
    action_id: str
    symbol: str
    family: str
    day: date
    value: float
    evidence: tuple[EvidenceReference, ...]


def _positive(value: Any) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError("action value must be positive finite numeric")
    return float(value)


def _contains_record(value: object, expected: dict[str, Any]) -> bool:
    if isinstance(value, dict):
        return value == expected or any(_contains_record(child, expected) for child in value.values())
    if isinstance(value, list):
        return any(_contains_record(child, expected) for child in value)
    return False


class InvestmentCorporateActions:
    """Resolve only ordinary USD dividends and exact proportional share splits.

    Nominal receivables remain unpaid/nontradable forever in this marked-value
    research policy. A payable date never manufactures payment or spendable cash.
    """

    def __init__(self, root: Path, evidence: CorporateActionEvidence,
        action_index: tuple[dict[str, tuple[ActionWindow, ...]], tuple[str, ...]],
        official_windows: tuple[OfficialWindow, ...], policy_reference: EvidenceReference, numeric_end: date,
        currency_evidence: Mapping[str, tuple[dict[str, Any], EvidenceReference]] | None = None,
        currency_corroborations: Mapping[str, CurrencyCorroboration] | None = None) -> None:
        self.root, self.evidence = root.resolve(), evidence
        self.actions, self.global_gaps = action_index
        self.official = official_windows
        self.policy_reference, self.numeric_end = policy_reference, numeric_end
        self.source_files = dict(evidence.source_files)
        self.source_files[policy_reference.reference] = policy_reference.artifact_sha256
        self.terms: dict[tuple[str, str], _Term] = {}
        self.invalid: set[tuple[str, str]] = set()
        self.unresolved_terms: dict[tuple[str, str], str] = {}
        self.coverage: dict[str, tuple[EvidenceReference, ...]] = {}
        self.process_first: date | None = None
        self.process_last: date | None = None
        self.recheck()
        policy_path = resolve_inside_authority(self.root, policy_reference.reference)
        policy = parse_strict_json_object(policy_path.read_bytes(), label="investment nominal claim policy")
        if policy.get("nominal_unpaid_claim_valuation") is not True:
            raise DataReadinessError("investment policy must explicitly approve nominal unpaid claim valuation")
        refreshed_currencies = currency_evidence or {}
        independent_currencies = currency_corroborations or {}
        observed_corroborations: set[str] = set()
        for name in evidence.source_files:
            if not name.endswith("/_request.json"):
                continue
            request: dict[str, Any] = parse_strict_json_object(
                resolve_inside_authority(self.root, name).read_bytes(), label="action request")
            if request.get("request_sha256") != evidence.request_sha256:
                continue
            scope = request["policy"]
            self.process_first = date.fromisoformat(scope["process_start"])
            self.process_last = date.fromisoformat(scope["process_end"])
        accepted = {(symbol, family, json_sha256(record)) for symbol, families in evidence.records_by_symbol.items()
            for family, records in families.items() for record in records}
        observed: set[tuple[str, str, str]] = set()
        record_versions: dict[str, str] = {}
        calendar = xcals.get_calendar("XNYS")
        for name, (query, metadata_name, _, _) in _selected_pages(self.root, evidence, lambda: None).items():
            body_path = resolve_inside_authority(self.root, name)
            metadata_path = resolve_inside_authority(self.root, metadata_name)
            if body_path.stat().st_size > 8 * 1024**2 or metadata_path.stat().st_size > 256 * 1024:
                raise DataReadinessError("investment action page exceeds bounded metadata/body size")
            raw = body_path.read_bytes()
            payload: dict[str, Any] = parse_strict_json_object(raw, label="investment corporate-action body")
            metadata: dict[str, Any] = parse_strict_json_object(metadata_path.read_bytes(), label="action transport")
            if metadata.get("sha256") != evidence.source_files[name] or metadata.get("body_length") != len(raw):
                raise DataReadinessError("investment action transport differs from exact body")
            captured = datetime.fromisoformat(metadata["retrieved_at_utc"])
            if captured.utcoffset() is None:
                raise DataReadinessError("investment action capture clock must be aware")
            def ref(path: str, locator: str, capture: datetime = captured) -> EvidenceReference:
                return EvidenceReference(reference=path, artifact_sha256=evidence.source_files[path], record_locator=locator,
                    interpretation_policy_sha256=policy_reference.interpretation_policy_sha256,
                    retrieved_at=capture.astimezone(UTC), available_at=capture.astimezone(UTC))
            # Receipt bytes remain pinned in source_files; their creation clock is
            # not the page capture clock, so do not invent an observed clock for them.
            coverage = (ref(name, "complete selected provider action page"), ref(metadata_name, "actual action capture clock"))
            self.coverage[query] = (*self.coverage.get(query, ()), *coverage)
            for family, records in payload["corporate_actions"].items():
                for record in records:
                    record_hash = json_sha256(record)
                    key = query, family, record_hash
                    if key not in accepted:
                        raise DataReadinessError("investment action record absent from independently replayed projection")
                    observed.add(key)
                    action_id, symbol = record.get("id"), record.get("symbol")
                    if not isinstance(action_id, str) or not isinstance(symbol, str):
                        continue
                    identity = symbol, action_id
                    version = json_sha256({"family": family, "record": record})
                    if action_id in record_versions and record_versions[action_id] != version:
                        self.invalid.add(identity)
                    record_versions[action_id] = version
                    try:
                        ex = date.fromisoformat(record["ex_date"])
                        if record["ex_date"] != ex.isoformat() or not calendar.is_session(ex.isoformat()):
                            raise ValueError("non-session action")
                        if any(record.get(field) not in (None, "", False) for field in
                            ("cancelled", "canceled", "revised", "revision", "due_bill_redemption_date")):
                            raise ValueError("unresolved action revision or due bill")
                        currency_ref: EvidenceReference | None = None
                        corroboration_refs: tuple[EvidenceReference, ...] = ()
                        if family == "cash_dividends":
                            if record.get("special") is not False or record.get("foreign") is not False:
                                self.unresolved_terms[identity] = "ordinary_domestic_cash_entitlement_unproven"
                                continue
                            currency = record.get("currency")
                            corroboration = independent_currencies.get(action_id)
                            if corroboration is not None:
                                observed_corroborations.add(action_id)
                                if currency not in (None, "", "USD"):
                                    raise DataReadinessError("independent USD evidence contradicts provider currency")
                                corroboration_refs = corroboration.validates_against(
                                    record, self.root, policy_reference.interpretation_policy_sha256)
                                for source_ref in corroboration_refs:
                                    if (source_ref.reference in self.source_files
                                        and self.source_files[source_ref.reference] != source_ref.artifact_sha256):
                                        raise DataReadinessError("independent currency source hash conflicts")
                                    self.source_files[source_ref.reference] = source_ref.artifact_sha256
                            if currency != "USD" and not corroboration_refs:
                                refreshed = refreshed_currencies.get(action_id)
                                if currency not in (None, "") or refreshed is None:
                                    self.unresolved_terms[identity] = "dividend_usd_currency_unproven"
                                    continue
                                refreshed_record, currency_ref = refreshed
                                if (refreshed_record.get("original_record_sha256") != record_hash
                                    or refreshed_record.get("currency") != "USD"
                                    or any(refreshed_record.get(field) not in (None, "", False) for field in
                                        ("cancelled", "canceled", "revised", "revision", "due_bill_redemption_date"))
                                    or any(refreshed_record.get(field) != record.get(field)
                                        for field in ("id", "symbol", "ex_date", "rate", "foreign", "special"))
                                    or any(refreshed_record.get(field) != record[field] for field in record if field != "currency")):
                                    self.invalid.add(identity)
                                    self.unresolved_terms[identity] = "currency_refresh_economic_fields_differ"
                                    continue
                                if currency_ref.available_at is None or currency_ref.available_at > currency_ref.retrieved_at:
                                    self.invalid.add(identity)
                                    self.unresolved_terms[identity] = "currency_refresh_capture_unproven"
                                    continue
                                refreshed_path = resolve_inside_authority(self.root, currency_ref.reference)
                                if file_sha256(refreshed_path) != currency_ref.artifact_sha256:
                                    raise DataReadinessError("refreshed currency evidence hash differs")
                                if refreshed_path.stat().st_size > 8 * 1024**2:
                                    raise DataReadinessError("refreshed currency evidence exceeds bounded source size")
                                refreshed_payload = parse_strict_json_object(refreshed_path.read_bytes(), label="refreshed currency source")
                                provider_record = {key: item for key, item in refreshed_record.items() if key != "original_record_sha256"}
                                if not (_contains_record(refreshed_payload, provider_record)
                                    or _contains_record(refreshed_payload, refreshed_record)):
                                    raise DataReadinessError("refreshed USD record is absent from pinned source bytes")
                                if (currency_ref.reference in self.source_files
                                    and self.source_files[currency_ref.reference] != currency_ref.artifact_sha256):
                                    raise DataReadinessError("refreshed currency source hash conflicts")
                                self.source_files[currency_ref.reference] = currency_ref.artifact_sha256
                            value = _positive(record["rate"])
                        elif family in ("forward_splits", "reverse_splits", "unit_splits"):
                            value = _positive(record["new_rate"]) / _positive(record["old_rate"])
                        else:
                            continue
                        term_ref = ref(name, f"family={family};action_id={action_id};record_sha256={record_hash}")
                        extra = (*corroboration_refs, *((currency_ref,) if currency_ref is not None else ()))
                        self.terms[identity] = _Term(action_id, symbol, family, ex, value, (term_ref, *coverage[1:], *extra))
                    except (KeyError, TypeError, ValueError):
                        self.invalid.add(identity)
        if observed_corroborations != set(independent_currencies):
            raise DataReadinessError("independent currency evidence contains unused or unsupported action IDs")
        if observed != accepted:
            raise DataReadinessError("investment action selected pages do not cover exact projection")
        self.recheck()

    def recheck(self) -> None:
        self.evidence.recheck(self.root)
        for name, digest in self.source_files.items():
            if file_sha256(resolve_inside_authority(self.root, name)) != digest:
                raise DataReadinessError("investment corporate-action source changed")

    def validate_sources(self, identity: str, symbol: str, sessions: tuple[date, ...], position_id: str) -> InvestmentSourceValidation:
        first, last = sessions[0], sessions[-1]
        reasons = list(self.global_gaps)
        refs = (*self.coverage.get(symbol, ()), self.policy_reference)
        if (symbol not in self.coverage or self.process_first is None or self.process_last is None
            or first < self.process_first or last > min(self.process_last, self.numeric_end)):
            reasons.append("corporate_action_query_interval_unproven")
        selected: dict[str, _Term] = {}
        for window in self.actions.get(symbol, ()):
            if not window.intersects(first, last):
                continue
            term = self.terms.get((symbol, window.action_id))
            if (term is None or (symbol, window.action_id) in self.invalid
                or window.unavailable_reason != "corporate_terms_require_replay" or term.family != window.family):
                reason = self.unresolved_terms.get((symbol, window.action_id), window.unavailable_reason)
                reasons.append(f"action:{window.family}:{window.action_id}:{reason}")
            elif first < term.day <= last:
                selected[term.action_id] = term
            elif term.day != first:
                reasons.append(f"action:{window.action_id}:economic_interval_unresolved")
        for official in self.official:
            if official.security_id == identity and official.first_session <= last and official.last_session >= first:
                reasons.append(f"official:{official.reason}:{official.record_locator}")
        terms = sorted(selected.values(), key=lambda term: (term.day, term.action_id))
        for day in {term.day for term in terms}:
            if len([term for term in terms if term.day == day and term.family != "cash_dividends"]) > 1 or (
                any(term.day == day and term.family == "cash_dividends" for term in terms)
                and any(term.day == day and term.family != "cash_dividends" for term in terms)):
                reasons.append(f"simultaneous_split_distribution_basis_unproven:{day}")
        events: list[CorporateActionEvent] = []
        marks: list[KnownMark] = []
        positions = []
        calendar = xcals.get_calendar("XNYS")
        position = position_id
        claims: list[tuple[str, tuple[EvidenceReference, ...]]] = []
        for day in sessions:
            for term in (item for item in terms if item.day == day):
                event_id = position_id + ":action:" + term.action_id
                evidence = (*term.evidence, self.policy_reference)
                if term.family == "cash_dividends":
                    claim = event_id + ":unpaid"
                    leg = PositionLeg(position_id=claim, security_id=identity, kind="unpaid_proceeds",
                        units_per_owned_unit=term.value, currency="USD", face_value_per_unit=1.0)
                    treatment: Literal["distribution", "replace"] = "distribution"
                    claims.append((claim, evidence))
                else:
                    leg = PositionLeg(position_id=event_id + ":shares", security_id=identity, kind="tradable_shares",
                        units_per_owned_unit=term.value, currency="USD")
                    treatment = "replace"
                events.append(CorporateActionEvent(event_id=event_id,
                    effective_at=calendar.session_open(day.isoformat()).to_pydatetime(), order=len(events),
                    evidence=evidence, owned_position_id=position, owned_security_id=identity,
                    treatment=treatment, fractional_treatment="proportional", legs=(leg,)))
                if term.family != "cash_dividends":
                    position = leg.position_id
            positions.append(position)
            for claim, evidence in claims:
                marks.append(KnownMark(position_id=claim, mark_at=calendar.session_close(day.isoformat()).to_pydatetime(),
                    value_per_unit=1.0, currency="USD", evidence=evidence))
        return InvestmentSourceValidation(evidence=refs, missing_reasons=tuple(sorted(set(reasons))), events=tuple(events),
            non_share_marks=tuple(marks), share_position_ids=tuple(positions),
            valuation_basis=("source_verified_entitlement_policy_valued_unpaid_receivable" if claims
                else "observed_raw_prices_and_split_units"))
