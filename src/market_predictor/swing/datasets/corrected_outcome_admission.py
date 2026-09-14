"""Participant reconciliation and positive ownership, separate from lot economics."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

import pandas as pd

from market_predictor.canonical.reconciliation import stamp_canonical_decision_ids
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.corrected_outcomes import DecisionCorrection, OfficialWindow, ReviewedCashDistributionScope
from market_predictor.swing.datasets.action_evidence import CorporateActionEvidence
from market_predictor.swing.datasets.action_scope import apply_reviewed_cash_scope
from market_predictor.swing.datasets.action_windows import ActionWindow, action_window

PARTICIPANT_FIELDS = frozenset(("symbol", "source_symbol", "initiating_symbol", "target_symbol", "old_symbol", "new_symbol",
    "acquirer_symbol", "acquiree_symbol", "surviving_symbol", "distributing_symbol", "distributed_symbol"))


def reconcile_actions(evidence: CorporateActionEvidence, expected_symbols: set[str],
    reviewed_scopes: tuple[ReviewedCashDistributionScope, ...] = (),
    ) -> tuple[dict[str, tuple[ActionWindow, ...]], tuple[str, ...]]:
    """Every query participates; a response's query tag never restricts its actions."""
    gaps = []
    scopes = {scope.action_id: scope for scope in reviewed_scopes}
    if len(scopes) != len(reviewed_scopes):
        raise DataReadinessError("duplicate reviewed cash action scope")
    applied: set[str] = set()
    if set(evidence.records_by_symbol).difference(expected_symbols):
        raise DataReadinessError("action projection includes an unrequested query symbol")
    result: dict[str, list[ActionWindow]] = defaultdict(list)
    for symbol in expected_symbols.difference(evidence.records_by_symbol) | set(evidence.unavailable_symbols):
        result[symbol].append(ActionWindow(f"query:{symbol}", "query_coverage", None, None, "action_query_unavailable"))
    seen: set[tuple[str, str]] = set()
    identities: dict[str, tuple[str, set[str]]] = {}
    conflicting: set[str] = set()
    for query, families in sorted(evidence.records_by_symbol.items()):
        for family, records in sorted(families.items()):
            for record in records:
                digest = json_sha256({"family": family, "record": record})
                if (query, digest) in seen:
                    continue
                seen.add((query, digest))
                identity = record.get("id")
                participants = set()
                local_gaps = []
                for key, value in record.items():
                    if key in PARTICIPANT_FIELDS:
                        if value is not None and value != "":
                            if not isinstance(value, str) or value != value.strip().upper():
                                local_gaps.append("invalid_action_participant")
                            else:
                                participants.add(value)
                    elif "symbol" in key.lower() and value not in (None, "", []):
                        local_gaps.append("unsupported_action_participant_field")
                        candidates = value if isinstance(value, list) else [value]
                        participants.update(v for v in candidates if isinstance(v, str) and v and v == v.strip().upper())
                if not participants:
                    gaps.append("missing_action_participants")
                # Keep the query association as an additional conservative constraint,
                # not a substitute for assigning all participants across all responses.
                participants.add(query)
                if not isinstance(identity, str) or not identity:
                    local_gaps.append("unidentified_action_record")
                elif identity in identities:
                    previous_digest, previous_participants = identities[identity]
                    if previous_digest != digest:
                        conflicting.add(identity)
                    if identity in conflicting:
                        local_gaps.append("conflicting_action_record_identity")
                        participants.update(previous_participants)
                    previous_participants.update(participants)
                else:
                    identities[identity] = (digest, set(participants))
                window = action_window(family, record)
                if isinstance(identity, str) and identity in scopes:
                    window = apply_reviewed_cash_scope(window, record, digest, scopes[identity])
                    applied.add(identity)
                for symbol in participants:
                    result[symbol].append(window)
                    result[symbol].extend(ActionWindow(str(identity), family, None, None, reason) for reason in local_gaps)
    if applied != set(scopes):
        raise DataReadinessError("reviewed cash action scope is absent from replayed provider responses")
    return {key: tuple(value) for key, value in result.items()}, tuple(sorted(set(gaps)))


def corrected_decisions(frame: pd.DataFrame, rules: tuple[DecisionCorrection, ...]) -> pd.DataFrame:
    """Retain parent identity and restamp using the full canonical metadata inputs."""
    frame = stamp_canonical_decision_ids(frame)
    result = frame.rename(columns={"decision_id": "parent_decision_id"}).copy()
    result["parent_ticker"] = result.ticker
    for rule in rules:
        mask = (result.security_id.eq(rule.security_id) & result.parent_ticker.eq(rule.parent_ticker)
            & result.session_date_et.between(rule.first_session, rule.last_session))
        result.loc[mask, "ticker"] = rule.ticker
    return stamp_canonical_decision_ids(result)


def corrected_ticker(identity: str, source_ticker: str, session: date, rules: tuple[DecisionCorrection, ...]) -> str:
    matches = [rule for rule in rules if rule.security_id == identity and rule.parent_ticker == source_ticker
        and rule.first_session <= session <= rule.last_session]
    if len(matches) > 1:
        raise DataReadinessError("ambiguous corrected ticker interval")
    return matches[0].ticker if matches else source_ticker


def corrected_memberships(frame: pd.DataFrame, rules: tuple[DecisionCorrection, ...]) -> pd.DataFrame:
    """Split only explicitly reviewed ticker intervals; retain every competing owner."""
    rows: list[dict[str, Any]] = frame.to_dict("records")
    for rule in rules:
        lower = pd.Timestamp(rule.first_session, tz="America/New_York").tz_convert("UTC")
        upper = (pd.Timestamp(rule.last_session, tz="America/New_York") + pd.Timedelta(days=1)).tz_convert("UTC")
        updated = []
        for row in rows:
            start = pd.Timestamp(row["effective_from_utc"])
            raw_end = row["effective_to_utc"]
            end = None if pd.isna(raw_end) else pd.Timestamp(raw_end)
            if (row["security_id"] != rule.security_id or row["ticker"] != rule.parent_ticker
                    or start >= upper or end is not None and end <= lower):
                updated.append(row)
                continue
            if start < lower:
                updated.append({**row, "effective_to_utc": lower})
            updated.append({**row, "ticker": rule.ticker, "effective_from_utc": max(start, lower),
                "effective_to_utc": upper if end is None else min(end, upper)})
            if end is None or end > upper:
                updated.append({**row, "effective_from_utc": upper})
        rows = updated
    return pd.DataFrame(rows, columns=frame.columns)


def window_gaps(*, identity: str, symbol: str, first: date, last: date,
    actions: dict[str, tuple[ActionWindow, ...]], official: tuple[OfficialWindow, ...],
    global_gaps: tuple[str, ...]) -> tuple[str, ...]:
    if first > last:
        raise DataReadinessError("reversed admission window")
    return tuple(sorted(set((*global_gaps,
        *(f"action:{window.family}:{window.action_id}:{window.unavailable_reason}"
          for window in actions.get(symbol, ()) if window.intersects(first, last)),
        *(f"official:{window.reason}:{window.record_locator}" for window in official
          if window.security_id == identity and window.first_session <= last and window.last_session >= first)))))
