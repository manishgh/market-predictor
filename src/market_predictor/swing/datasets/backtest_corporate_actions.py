"""Translate pinned ordinary dividends and splits into backtest holding events."""
from __future__ import annotations

import hashlib
import math
import tomllib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.swing.contracts.backtest_corporate_actions import (
    BacktestCorporateActionContext,
    BacktestCorporateActionPolicy,
)
from market_predictor.swing.contracts.holding_accounting import (
    CashAvailabilityEvent,
    CorporateActionEvent,
    Event,
    EvidenceReference,
    HoldingSpecification,
    KnownMark,
    PaymentEvent,
    PositionLeg,
    SimulationReference,
)
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.action_evidence import CorporateActionEvidence
from market_predictor.swing.datasets.action_windows import ActionWindow, action_window
from market_predictor.swing.datasets.reviewed_action_dates import load_reviewed_action_dates
from market_predictor.swing.datasets.reviewed_action_dispositions import load_reviewed_record_dispositions
from market_predictor.swing.datasets.reviewed_cash_payouts import load_reviewed_cash_payouts
from market_predictor.swing.datasets.reviewed_parent_units import load_reviewed_parent_spin_units
from market_predictor.swing.datasets.reviewed_spin_offs import load_reviewed_spin_offs


def load_backtest_corporate_action_context(root: Path, pin: SourcePin) -> BacktestCorporateActionContext:
    path = inside(root, pin.path)
    if path.stat().st_size > 65536 or file_sha256(path) != pin.sha256:
        raise DataReadinessError("backtest corporate-action policy size or hash differs")
    policy = BacktestCorporateActionPolicy.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    if file_sha256(path) != pin.sha256:
        raise DataReadinessError("backtest corporate-action policy changed while loading")
    return BacktestCorporateActionContext(policy=policy, policy_reference=EvidenceReference(
        reference=pin.path, artifact_sha256=pin.sha256, interpretation_policy_sha256=policy.sha256(),
        record_locator="human-approved backtest assumptions; not observed corporate cash receipts",
        retrieved_at=datetime.now(UTC), available_at=None))


@dataclass(frozen=True)
class _Terms:
    action_id: str
    family: str
    symbol: str
    session: date
    value: float
    evidence: EvidenceReference


def _positive(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("corporate action amount or split ratio must be positive and finite")
    return float(value)


def _selected_pages(root: Path, evidence: CorporateActionEvidence,
    memory_guard: Callable[[], None]) -> dict[str, tuple[str, str, str, str]]:
    """Select only acquired receipts belonging to the independently replayed request."""
    pages: dict[str, tuple[str, str, str, str]] = {}
    receipts: set[str] = set()
    for name, digest in evidence.source_files.items():
        if not name.endswith("/receipt.json") or "/tickers/" not in name:
            continue
        memory_guard()
        path = inside(root, name)
        if path.stat().st_size > 256 * 1024:
            raise DataReadinessError("backtest action receipt exceeds its bound")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise DataReadinessError("backtest action receipt changed")
        receipt: dict[str, Any] = parse_strict_json_object(raw, label="backtest action receipt")
        if receipt.get("request_sha256") != evidence.request_sha256 or receipt.get("state") != "acquired":
            continue
        symbol = receipt.get("ticker")
        if not isinstance(symbol, str) or symbol not in evidence.records_by_symbol or symbol in receipts:
            raise DataReadinessError("backtest action successful receipt is ambiguous")
        receipts.add(symbol)
        for page in receipt["pages"]:
            body = inside(path.parent, page["body_path"]).relative_to(root).as_posix()
            metadata = inside(path.parent, page["metadata_path"]).relative_to(root).as_posix()
            if (evidence.source_files.get(body) != page["body_sha256"]
                    or evidence.source_files.get(metadata) != page["metadata_sha256"] or body in pages):
                raise DataReadinessError("backtest action page differs from selected receipt")
            pages[body] = symbol, metadata, name, digest
    if receipts != set(evidence.records_by_symbol):
        raise DataReadinessError("backtest action successful receipt is missing")
    return pages


class BacktestCorporateActions:
    """Only exact independently replayed records can resolve an action window."""

    def __init__(self, root: Path, evidence: CorporateActionEvidence, context: BacktestCorporateActionContext,
        *, memory_guard: Callable[[], None]) -> None:
        self.context = context
        self.terms: dict[tuple[str, str], _Terms] = {}
        self._records: dict[str, tuple[str, dict[str, Any]]] = {}
        conflicting_records: set[str] = set()
        invalid: set[tuple[str, str]] = set()
        self.source_files: dict[str, str] = {}
        accepted = {(symbol, family, json_sha256(record)) for symbol, families in evidence.records_by_symbol.items()
            for family, records in families.items() for record in records}
        observed: set[tuple[str, str, str]] = set()
        calendar = xcals.get_calendar("XNYS")
        for name, (query, metadata_name, receipt_name, receipt_digest) in sorted(_selected_pages(root, evidence, memory_guard).items()):
            digest = evidence.source_files[name]
            memory_guard()
            path = inside(root, name)
            if path.stat().st_size > 8 * 1024**2:
                raise DataReadinessError("backtest corporate-action source size or hash differs")
            self.source_files[name] = digest
            self.source_files[receipt_name] = receipt_digest
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise DataReadinessError("backtest corporate-action source hash differs")
            payload = parse_strict_json_object(raw, label="backtest corporate-action body")
            metadata_digest = evidence.source_files.get(metadata_name)
            metadata_path = inside(root, metadata_name)
            if metadata_digest is None or metadata_path.stat().st_size > 256 * 1024:
                raise DataReadinessError("backtest action source lacks pinned transport metadata")
            metadata_bytes = metadata_path.read_bytes()
            if hashlib.sha256(metadata_bytes).hexdigest() != metadata_digest:
                raise DataReadinessError("backtest action transport metadata changed")
            metadata = parse_strict_json_object(metadata_bytes, label="backtest action transport")
            if metadata.get("sha256") != digest or metadata.get("body_length") != len(raw):
                raise DataReadinessError("backtest action transport does not bind the exact body")
            retrieved = metadata.get("retrieved_at_utc")
            if not isinstance(retrieved, str):
                raise DataReadinessError("backtest action capture clock missing")
            captured = datetime.fromisoformat(retrieved)
            if captured.utcoffset() is None:
                raise DataReadinessError("backtest action capture clock must be aware")
            self.source_files[metadata_name] = metadata_digest
            groups = payload.get("corporate_actions")
            if not isinstance(groups, dict):
                raise DataReadinessError("backtest corporate-action body has no family mapping")
            for family, records in groups.items():
                if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
                    raise DataReadinessError("backtest corporate-action family requires object records")
                for record in records:
                    identity = query, family, json_sha256(record)
                    if identity not in accepted:
                        raise DataReadinessError("backtest action page record is absent from successful projection")
                    observed.add(identity)
                    record_id = record.get("id")
                    if isinstance(record_id, str) and record_id:
                        prior_record = self._records.get(record_id)
                        if prior_record is not None and prior_record != (family, record):
                            conflicting_records.add(record_id)
                        self._records[record_id] = family, record
                    if family not in ("cash_dividends", "forward_splits", "reverse_splits"):
                        continue
                    try:
                        action_id, symbol = record["id"], record["symbol"]
                        if not isinstance(action_id, str) or not action_id or symbol != query:
                            continue
                        session = date.fromisoformat(record["ex_date"])
                        if record["ex_date"] != session.isoformat() or not calendar.is_session(session.isoformat()):
                            raise ValueError("action ex-date must be an exact exchange session")
                        if family == "cash_dividends":
                            if record.get("special") is not False or record.get("foreign") is not False:
                                continue
                            value = _positive(record["rate"])
                        else:
                            value = _positive(record["new_rate"]) / _positive(record["old_rate"])
                            _positive(value)
                        reference = EvidenceReference(reference=name, artifact_sha256=digest,
                            interpretation_policy_sha256=context.policy.sha256(),
                            record_locator=f"family={family};action_id={action_id};record_sha256={json_sha256(record)}",
                            retrieved_at=captured.astimezone(UTC), available_at=None)
                        term = _Terms(action_id, family, symbol, session, value, reference)
                        key = symbol, action_id
                        prior = self.terms.get(key)
                        if prior is not None and (prior.family, prior.session, prior.value) != (family, session, value):
                            invalid.add(key)
                        else:
                            self.terms[key] = prior or term
                    except (KeyError, TypeError, ValueError):
                        if isinstance(record.get("id"), str) and isinstance(record.get("symbol"), str):
                            invalid.add((record["symbol"], record["id"]))
        if observed != accepted:
            raise DataReadinessError("backtest action projection lacks matching successful source records")
        for conflicting_id in conflicting_records:
            del self._records[conflicting_id]
        self.reviewed_dates, reviewed_pins = load_reviewed_action_dates(root, context.policy.reviewed_action_dates,
            self._records, memory_guard=memory_guard)
        self.source_files.update(reviewed_pins)
        counts = Counter((term.symbol, term.session, term.family == "cash_dividends") for term in self.terms.values())
        mixed = {(symbol, session) for symbol, session, cash in counts if (symbol, session, not cash) in counts}
        for key, term in list(self.terms.items()):
            if (key in invalid or term.action_id in conflicting_records or (term.symbol, term.session) in mixed
                    or counts[term.symbol, term.session, term.family == "cash_dividends"] != 1):
                del self.terms[key]
        cash_facts, cash_pins = load_reviewed_cash_payouts(root, context.policy.reviewed_cash_payouts,
            self._records, documented_cash_approved=context.policy.documented_cash_payouts is not None,
            memory_guard=memory_guard)
        for name, digest in cash_pins.items():
            if name in self.source_files and self.source_files[name] != digest:
                raise DataReadinessError("reviewed cash cannot replace a different retained source pin")
        self.source_files.update(cash_pins)
        self.reviewed_parent_units, parent_pins = load_reviewed_parent_spin_units(root, context.policy.reviewed_parent_units,
            self._records, self.reviewed_dates, memory_guard=memory_guard)
        for name, digest in parent_pins.items():
            if name in self.source_files and self.source_files[name] != digest:
                raise DataReadinessError("reviewed parent units cannot replace a different retained source pin")
        self.source_files.update(parent_pins)
        self.reviewed_record_dispositions, disposition_pins = load_reviewed_record_dispositions(root,
            context.policy.reviewed_record_dispositions, self._records, memory_guard=memory_guard)
        for name, digest in disposition_pins.items():
            if name in self.source_files and self.source_files[name] != digest:
                raise DataReadinessError("reviewed disposition cannot replace a different retained source pin")
        self.source_files.update(disposition_pins)
        self.reviewed_cash_aliases: dict[tuple[str, str], str] = {}
        for fact in cash_facts:
            cash_identity = "reviewed-cash:" + json_sha256(fact.model_dump(mode="json"))
            fact_pin = context.policy.reviewed_cash_payouts
            if fact_pin is None:
                raise DataReadinessError("reviewed cash facts lack their explicit policy pin")
            reference = EvidenceReference(reference=fact_pin.path, artifact_sha256=fact_pin.sha256,
                interpretation_policy_sha256=context.policy.sha256(),
                record_locator=f"declared USD entitlement={fact.entitlement_id};ex_date={fact.ex_date.isoformat()}",
                retrieved_at=context.policy_reference.retrieved_at, available_at=None)
            for record in fact.provider_records:
                self.terms.pop((fact.symbol, record.action_id), None)
                self.reviewed_cash_aliases[fact.symbol, record.action_id] = cash_identity
            self.terms[fact.symbol, cash_identity] = _Terms(cash_identity, "cash_dividends", fact.symbol, fact.ex_date,
                fact.declared_rate_usd, reference)

        self.reviewed_spin_offs, spin_pins = load_reviewed_spin_offs(root, context.policy.reviewed_spin_offs,
            self._records, self.reviewed_dates, approved=context.policy.documented_spin_offs is not None,
            policy_sha256=context.policy.sha256(), retrieved_at=context.policy_reference.retrieved_at, memory_guard=memory_guard)
        for name, digest in spin_pins.items():
            if name in self.source_files and self.source_files[name] != digest:
                raise DataReadinessError("documented spin cannot replace a different retained source pin")
        self.source_files.update(spin_pins)
        for inputs in self.reviewed_spin_offs.values():
            for family, record in inputs.actions:
                child_action_id = record.get("id")
                if not isinstance(child_action_id, str) or not child_action_id.strip():
                    raise DataReadinessError("documented daughter action has no exact identity")
                existing = self._records.get(child_action_id)
                if existing is not None and (family != existing[0] or json_sha256(record) != json_sha256(existing[1])):
                    raise DataReadinessError("documented daughter action conflicts with retained record")
                self._records[child_action_id] = family, record
        for inputs in self.reviewed_spin_offs.values():
            spin_fact = inputs.fact
            first, last = spin_fact.ex_date, max(inputs.evidence, default=spin_fact.ex_date)
            for family, record in inputs.actions:
                if family == "spin_offs" and record["id"] == spin_fact.action_id:
                    continue
                participants = {value for key, value in record.items() if "symbol" in key and isinstance(value, str)}
                window = action_window(family, record)
                if (spin_fact.daughter_symbol not in participants or window.intersects(first, last)
                        and self.irrelevance_reason(spin_fact.daughter_symbol, window, first, last) is None):
                    raise DataReadinessError("documented daughter has an unresolved intervening corporate action")

    def irrelevance_reason(self, symbol: str, window: ActionWindow, first: date, last: date,
        *, price_history: bool = False) -> str | None:
        """Explain why an exact action cannot affect this newly opened position.

        This supplies neither action economics nor an identity mapping. The caller
        still verifies its raw source segment, membership, official and global gaps.
        Local reconciliation errors must never be removed by a valid sibling record.
        """
        if first > last:
            raise DataReadinessError("backtest action relevance interval is reversed")
        if window.unavailable_reason not in {"corporate_terms_require_replay",
                "unbounded_entitlement_or_due_bill_dates", "missing_action_economic_date"}:
            return None
        matched = self._records.get(window.action_id)
        if matched is None or matched[0] != window.family:
            return None
        family, record = matched
        disposition = self.reviewed_record_dispositions.get(window.action_id)
        if disposition is not None and family == disposition.family and symbol == disposition.symbol:
            return "audited_issuer_share_reconciliation_refutes_exact_retained_record"
        reviewed = self.reviewed_dates.get(window.action_id)
        parent_units = self.reviewed_parent_units.get(window.action_id)
        if parent_units is not None and family == "spin_offs" and symbol == parent_units.source_symbol:
            if (window.first_relevant_date is not None and window.first_relevant_date > parent_units.ex_date
                    or window.last_relevant_date is not None and window.last_relevant_date < parent_units.ex_date):
                return None
            if price_history:
                return "reviewed_spin_off_preserves_parent_raw_share_units"
            if first >= parent_units.ex_date:
                return "new_regular_way_parent_purchase_on_or_after_ex_date"
            if last < parent_units.ex_date:
                return "regular_way_parent_sold_before_distribution_entitlement"
        if family == "name_changes":
            if reviewed is not None and symbol in (record.get("old_symbol"), record.get("new_symbol")):
                effective = reviewed.economic_dates["effective_date"]
                if last < effective:
                    return "holding_ends_before_official_identity_change"
                # The caller independently binds a fresh raw-price window and
                # ownership. A same-ticker change strictly before that window
                # needs no conversion of an already-owned old security class.
                # Crossing the date, its unknown intraday clock, and ticker
                # aliases still require their separate ownership treatment.
                if first > effective and record.get("old_symbol") == record.get("new_symbol") == symbol:
                    return "new_same_symbol_raw_window_after_official_identity_change"
            old_cusip, new_cusip = record.get("old_cusip"), record.get("new_cusip")
            if (isinstance(old_cusip, str) and len(old_cusip) == 9 and old_cusip.isalnum()
                    and old_cusip == new_cusip and symbol in (record.get("old_symbol"), record.get("new_symbol"))):
                return "same_cusip_name_change_preserves_owned_units"
            return None
        if family in {"cash_mergers", "stock_mergers", "stock_and_cash_mergers"}:
            acquiree, acquirer = record.get("acquiree_symbol"), record.get("acquirer_symbol")
            old_cusip, new_cusip = record.get("acquiree_cusip"), record.get("acquirer_cusip")
            symbol_fields = {key for key, value in record.items() if "symbol" in key.lower() and value not in (None, "")}
            if (symbol == acquirer and isinstance(acquiree, str) and acquiree and acquiree != acquirer
                    and isinstance(old_cusip, str) and isinstance(new_cusip, str)
                    and len(old_cusip) == len(new_cusip) == 9 and old_cusip.isalnum() and new_cusip.isalnum()
                    and old_cusip != new_cusip and symbol_fields == {"acquiree_symbol", "acquirer_symbol"}):
                return "merger_acquirer_existing_shares_are_not_acquiree_consideration"
            return None
        if family not in {"cash_dividends", "spin_offs", "stock_dividends", "rights_distributions",
                "capital_gains_distributions"}:
            return None
        source = (record.get("symbol") if family in {"cash_dividends", "stock_dividends", "capital_gains_distributions"}
                  else record.get("source_symbol"))
        participants = (source, record.get("new_symbol")) if family == "spin_offs" else (source,)
        if symbol not in participants:
            return None
        if price_history and family == "cash_dividends":
            # A cash-only payout changes neither the units nor the identity of
            # shares in raw bars. Its price movement stays in the frozen raw ATR.
            # This does not admit a holding, value a claim or pay any cash.
            if type(record.get("special")) is bool and record.get("foreign") is False:
                try:
                    _positive(record["rate"])
                except (KeyError, TypeError, ValueError):
                    return None
                return "cash_distribution_does_not_change_raw_price_share_units"
        dates: dict[str, date] = {}
        try:
            for field in ("ex_date", "record_date", "payable_date", "due_bill_redemption_date", "effective_date"):
                value = record.get(field)
                if value in (None, ""):
                    continue
                if not isinstance(value, str):
                    return None
                parsed = date.fromisoformat(value)
                if value != parsed.isoformat():
                    return None
                dates[field] = parsed
        except (TypeError, ValueError):
            return None
        if reviewed is not None:
            dates.update((field, value) for field, value in reviewed.economic_dates.items())
        if "ex_date" not in dates:
            return None
        # The position starts with a new purchase, not an earlier owner's claims.
        # Keep the boundary day itself gated and retain every known later date.
        if first > max(dates.values()):
            return "new_purchase_after_distribution_economic_dates"
        if {"ex_date", "record_date", "payable_date"}.issubset(dates) and last < min(dates.values()):
            return "holding_ends_before_distribution_economic_dates"
        return None

    def resolves(self, symbol: str, window: ActionWindow) -> bool:
        spin = self.reviewed_spin_offs.get(window.action_id)
        if spin is not None and symbol == spin.fact.source_symbol and window.family == "spin_offs":
            return (window.unavailable_reason in {"corporate_terms_require_replay", "unbounded_entitlement_or_due_bill_dates"}
                and (window.first_relevant_date is None or window.first_relevant_date <= spin.fact.ex_date)
                and (window.last_relevant_date is None or spin.fact.ex_date <= window.last_relevant_date))
        reviewed = self.reviewed_cash_aliases.get((symbol, window.action_id))
        term = self.terms.get((symbol, reviewed or window.action_id))
        if reviewed is not None:
            return (term is not None and term.family == window.family == "cash_dividends"
                and window.unavailable_reason in {"corporate_terms_require_replay",
                    "unbounded_entitlement_or_due_bill_dates", "reviewed_cash_distribution_resolution_unavailable"}
                and (window.first_relevant_date is None or window.first_relevant_date <= term.session)
                and (window.last_relevant_date is None or term.session <= window.last_relevant_date))
        return (term is not None and term.family == window.family
            and window.unavailable_reason == "corporate_terms_require_replay"
            and window.first_relevant_date is not None
            and window.first_relevant_date <= term.session <= (window.last_relevant_date or term.session))

    def split_ratios(self, symbol: str, first: date, last: date) -> tuple[tuple[date, float], ...]:
        return tuple(sorted((term.session, term.value) for term in self.terms.values()
            if term.symbol == symbol and term.family != "cash_dividends" and first < term.session <= last))

    def specification(self, spec: HoldingSpecification, symbol: str, observations: pd.DataFrame,
        refs: dict[date, EvidenceReference]) -> HoldingSpecification:
        calendar = xcals.get_calendar("XNYS")
        days = tuple(stamp.date() for stamp in spec.session_end_timestamps)
        events: list[Event] = []
        marks: list[KnownMark] = []
        position = spec.initial_position_id
        terms = sorted((term for term in self.terms.values() if term.symbol == symbol
            and days[0] < term.session <= days[-1]), key=lambda term: (term.session, term.family == "cash_dividends", term.action_id))
        for day, close in zip(days, spec.session_end_timestamps, strict=True):
            opening = calendar.session_open(day.isoformat()).to_pydatetime()
            for term in (item for item in terms if item.session == day):
                prefix = ("simulated-dividend:" if term.family == "cash_dividends" else "backtest-split:") + json_sha256({
                    "decision": spec.decision_id, "action": term.action_id, "policy": self.context.policy.sha256()})
                if term.family != "cash_dividends":
                    replacement = prefix + ":shares"
                    events.append(CorporateActionEvent(event_id=prefix, effective_at=opening, order=len(events),
                        evidence=(term.evidence,), owned_position_id=position, owned_security_id=spec.security_id,
                        treatment="replace", fractional_treatment="proportional", legs=(PositionLeg(
                            position_id=replacement, security_id=spec.security_id, kind="tradable_shares",
                            units_per_owned_unit=term.value, currency="USD"),)))
                    position = replacement
                else:
                    assumption = SimulationReference(**self.context.policy_reference.model_dump(), basis="research_assumption")
                    evidence = (term.evidence, assumption)
                    claim, pending, payment = prefix + ":claim", prefix + ":pending", prefix + ":payment"
                    events.extend((CorporateActionEvent(event_id=prefix, effective_at=opening, order=len(events),
                        evidence=evidence, owned_position_id=position, owned_security_id=spec.security_id,
                        treatment="distribution", fractional_treatment="proportional", legs=(PositionLeg(
                            position_id=claim, security_id=spec.security_id, kind="unpaid_proceeds",
                            units_per_owned_unit=term.value, currency="USD", face_value_per_unit=1.0),)),
                        PaymentEvent(event_id=payment, effective_at=close, order=len(events) + 1, evidence=evidence,
                            claim_id=claim, amount_per_claim_unit=1.0, currency="USD", pending_proceeds_id=pending),
                        CashAvailabilityEvent(event_id=prefix + ":cash", effective_at=close, order=len(events) + 2,
                            evidence=evidence, payment_event_id=payment, currency="USD")))
            if day in observations.index and day in refs:
                marks.append(KnownMark(position_id=position, mark_at=close, value_per_unit=float(observations.loc[day, "close"]),
                    currency="USD", evidence=(refs[day],)))
        for inputs in self.reviewed_spin_offs.values():
            fact = inputs.fact
            opening = calendar.session_open(fact.ex_date.isoformat()).to_pydatetime()
            if fact.source_symbol != symbol or not spec.initial_entry_timestamp < opening <= spec.session_end_timestamps[-1]:
                continue
            prefix = "simulated-spin-off:" + json_sha256({"decision": spec.decision_id,
                "action": fact.action_id, "policy": self.context.policy.sha256()})
            pin = self.context.policy.reviewed_spin_offs
            assert pin is not None
            reference = EvidenceReference(reference=pin.path, artifact_sha256=pin.sha256,
                interpretation_policy_sha256=self.context.policy.sha256(), record_locator=f"documented spin={fact.action_id}",
                retrieved_at=self.context.policy_reference.retrieved_at, available_at=None)
            assumption = SimulationReference(**self.context.policy_reference.model_dump(), basis="research_assumption")
            daughter = prefix + ":daughter"
            events.append(CorporateActionEvent(event_id=prefix, effective_at=opening, order=len(events),
                evidence=(reference, assumption), owned_position_id=spec.initial_position_id, owned_security_id=spec.security_id,
                treatment="distribution", fractional_treatment="proportional", legs=(PositionLeg(position_id=daughter,
                    security_id=fact.daughter_security_id, kind="tradable_shares",
                    units_per_owned_unit=fact.daughter_units_per_parent, currency="USD"),)))
            for day in days:
                if day < fact.ex_date or day not in inputs.observations.index or day not in inputs.evidence:
                    continue
                row = inputs.observations.loc[day]
                if not bool(row.outcome_observation_valid):
                    continue
                for at, column in ((calendar.session_open(day.isoformat()).to_pydatetime(), "open"),
                        (calendar.session_close(day.isoformat()).to_pydatetime(), "close")):
                    marks.append(KnownMark(position_id=daughter, mark_at=at, value_per_unit=float(row[column]),
                        currency="USD", evidence=(inputs.evidence[day],)))
        return HoldingSpecification.model_validate({**spec.model_dump(), "events": tuple(events), "marks": tuple(marks)})
