"""Evidence-backed identity proofs for legacy news-query IDs the CIK bridge cannot convert.

The target membership authority carries each security's latest ticker back through
history, so no rule attributes by comparing a historical ticker with it. Proofs
reproduce both IDs from pinned S&P event evidence, compare CIKs (whose ticker equality
can only reject, as a share-class guard), or match a CUSIP chain's last ticker only at
a target security's final instant. Every in-scope legacy spell is proven or rejected
with a reason; nothing is guessed. `issuer_news_identity.py` is pinned by
closed evidence, so its primitives are imported and its two translation loops are
re-expressed here as passes a differential test ties to the protected functions.
"""
from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.symbols import normalized_ticker
from market_predictor.universe.issuer_news_identity import (
    BRIDGE_COLUMNS,
    _append,
    _authority,
    _bounded,
    _cik,
    _frame,
    _hash,
    _Index,
    _index,
    _intersection,
    _overlaps,
    _required_utc,
    _sha,
    _Span,
    _text,
    _utc,
)
from market_predictor.universe.sp500.membership_authority import _historical_security_id
from market_predictor.universe.sp500.membership_history import (
    IndexChange,
    _optional_transition_text,
    _security_identity_for_interval,
    _transition_boolean,
)

LEGACY_STATUS = "legacy_proven"
AVAILABILITY_BASIS = "retrospective_membership_effective_proxy"
EVIDENCE_COMPLETE_DATE = (
    "earliest New York date by which sufficient cited evidence existed: the first publication of one reproducing "
    "event; for spell events the later side's first publication; null for CIK equality; for CUSIP chains the latest "
    "of the matched instant, a closing deletion's first publication and the cited transitions' effective dates, which "
    "are event dates because the transition snapshot carries no publication receipts")
PROOF_KINDS = ("company_ticker_hash_reproduced", "sp500_spell_events_reproduced", "cik_equal",
               "cusip_chain_end_ticker_match")
REJECTION_REASONS = ("unsupported_identity_namespace", "multiple_legacy_spells", "no_candidate", "ambiguous_candidates",
                     "legacy_chain_contradicts_transitions", "cik_equal_ticker_differs", "no_membership_intersection",
                     "corrected_security_uses_corrections_archive_only")
PROOF_COLUMNS = (
    "source_security_id", "ticker", "target_security_id", "effective_from_utc", "effective_to_utc", "available_at_utc",
    "legacy_spell_from_utc", "legacy_spell_to_utc", "proof_kind", "evidence_json", "evidence_complete_date",
    "proof_row_sha256",
)
REJECTION_COLUMNS = ("source_security_id", "ticker", "legacy_spell_from_utc", "legacy_spell_to_utc", "reason",
                     "detail_json")
_TIMES = ("effective_from_utc", "effective_to_utc", "available_at_utc", "legacy_spell_from_utc", "legacy_spell_to_utc")
_TRANSITION_COLUMNS = ("id", "old_symbol", "new_symbol", "old_cusip", "new_cusip", "identity_continuity",
                       "effective_date", "process_date")
_NEW_YORK = "America/New_York"
_NO_CURRENT = pd.DataFrame()


@dataclass(frozen=True)
class _Proof:
    target: str
    kind: str
    evidence: dict[str, Any]
    complete: date | None


@dataclass(frozen=True)
class _Rejection:
    reason: str
    detail: dict[str, Any]


@dataclass(frozen=True)
class _Change:
    source_id: str
    effective_at: pd.Timestamp
    old_ticker: str
    new_ticker: str


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _stamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def _events(changes: Sequence[IndexChange]) -> list[dict[str, Any]]:
    return [{"action": change.action, "company": change.company, "ticker": change.ticker,
             "effective_at_utc": _stamp(change.effective_at_utc).isoformat(),
             "sources": [{"source_url": source.source_url, "source_sha256": source.source_sha256,
                          "source_published_date": source.source_published_date.isoformat()}
                         for source in change.source_evidence()]}
            for change in sorted(changes, key=lambda item: (_stamp(item.effective_at_utc), item.action, item.company))]


def _first_published(changes: Sequence[IndexChange]) -> date:
    return min(source.source_published_date for change in changes for source in change.source_evidence())


def _chains(transitions: pd.DataFrame) -> dict[str, list[_Change]]:
    """Every identity-continuous ticker change per CUSIP chain, with the minting parser's effective time.

    `symbol_changes_from_transitions` drops old tickers ending in V as when-issued symbols, which
    hid Fiserv's FISV to FI change from the legacy build; verification must see every change.
    """
    frame = _frame(transitions, _TRANSITION_COLUMNS)
    if {"old_security_id", "new_security_id"}.intersection(frame.columns):
        raise DataReadinessError("transition evidence with explicit security IDs is not supported")
    chains: dict[str, list[_Change]] = {}
    for row in frame.to_dict("records"):
        old, new = normalized_ticker(str(row["old_symbol"])), normalized_ticker(str(row["new_symbol"]))
        if not _transition_boolean(row["identity_continuity"], field="identity_continuity") or old == new:
            continue
        cusip = _optional_transition_text(row["old_cusip"]).upper()
        if not cusip or _optional_transition_text(row["new_cusip"]).upper() != cusip:
            raise DataReadinessError(f"identity-continuous transition changes its CUSIP: {row['id']}")
        effective = _optional_transition_text(row["effective_date"]) or _optional_transition_text(row["process_date"])
        if not effective:
            raise DataReadinessError(f"transition has no effective date: {row['id']}")
        instant = pd.Timestamp(pd.Timestamp(effective).date(), tz=_NEW_YORK).tz_convert("UTC")
        chains.setdefault(f"cusip:{cusip}", []).append(_Change(_text(str(row["id"]), "transition_id"), instant, old, new))
    return chains


class _Builder:
    def __init__(self, legacy: dict[str, list[_Span]], members: list[_Span], changes: Sequence[IndexChange],
                 transitions: pd.DataFrame, corrected: Collection[str], cutoff: pd.Timestamp) -> None:
        self.legacy = legacy
        self.index = _index(members, lambda row: (row.security_id,))
        self.target_ids = {member.security_id for member in members}
        self.by_ticker: dict[str, list[_Span]] = {}
        for member in members:
            self.by_ticker.setdefault(member.ticker, []).append(member)
        self.changes: dict[str, list[IndexChange]] = {}
        for change in changes:
            self.changes.setdefault(change.ticker, []).append(change)
        self.chains = _chains(transitions)
        self.corrected = frozenset(corrected)
        self.cutoff = cutoff

    def resolve(self, source: str) -> _Proof | _Rejection:
        namespace = source.split(":", 1)[0]
        if namespace == "sp500-historical":
            return self._historical(source, self.legacy[source])
        if namespace == "cik":
            cik = _cik(source, self.legacy[source][0].ticker)
            target = f"cik:{cik}"
            if target not in self.target_ids:
                return _Rejection("no_candidate", {"cik": cik})
            return _Proof(target, "cik_equal", {"cik": cik}, None)
        if namespace == "cusip":
            return self._chain(source, self.legacy[source])
        return _Rejection("unsupported_identity_namespace", {"namespace": namespace})

    def _historical(self, source: str, spells: list[_Span]) -> _Proof | _Rejection:
        if len(spells) != 1:
            return _Rejection("multiple_legacy_spells", {"spells": len(spells)})
        spell = spells[0]
        candidates = self.changes.get(spell.ticker, [])
        named = [change for change in candidates if _security_identity_for_interval(
            ticker=spell.ticker, company=change.company, effective_from=spell.start, effective_to=spell.end,
            current=_NO_CURRENT, aliases=[]) == source]
        if not named:
            return _Rejection("no_candidate", {"reason": "no pinned event company reproduces the legacy ID"})
        targets = {_historical_security_id(change) for change in named} & self.target_ids
        if len(targets) > 1:
            return _Rejection("ambiguous_candidates", {"targets": sorted(targets)})
        if targets:
            return _Proof(targets.pop(), "company_ticker_hash_reproduced",
                          {"company": named[0].company.strip().lower(), "events": _events(named)}, _first_published(named))
        additions = [change for change in named
                     if change.action == "addition" and _stamp(change.effective_at_utc) == spell.start]
        deletions = [change for change in candidates if spell.end is not None and change.action == "deletion"
                     and _stamp(change.effective_at_utc) == spell.end]
        closing = [(change, _historical_security_id(change)) for change in deletions]
        exact = {target: [change for change, minted in closing if minted == target] for _, target in closing
                 if target in self.target_ids and any(row.start == spell.start and row.end == spell.end
                                                      for row in self.index[(target,)][1])}
        if not additions or not exact:
            return _Rejection("no_candidate", {"reason": "legacy ID reproduces only from events that bind no target spell"})
        if len(exact) > 1:
            return _Rejection("ambiguous_candidates", {"targets": sorted(exact)})
        target, closers = next(iter(exact.items()))
        return _Proof(target, "sp500_spell_events_reproduced",
                      {"addition": _events(additions), "deletion": _events(closers)},
                      max(_first_published(additions), _first_published(closers)))

    def _chain(self, source: str, spells: list[_Span]) -> _Proof | _Rejection:
        changes = self.chains.get(source, [])
        explained: list[_Change] = []
        for previous, current in zip(spells, spells[1:], strict=False):
            if previous.end != current.start or previous.ticker == current.ticker:
                continue
            matches = [change for change in changes if change.effective_at == current.start
                       and (change.old_ticker, change.new_ticker) == (previous.ticker, current.ticker)]
            if len(matches) != 1:
                return _Rejection("legacy_chain_contradicts_transitions",
                                  {"boundary_utc": current.start.isoformat(), "matching_transitions": len(matches)})
            explained.append(matches[0])
        for change in changes:
            instant = change.effective_at
            inside = [spell for spell in spells if spell.start <= instant and (spell.end is None or instant < spell.end)]
            if not inside or change in explained:
                continue
            spell = inside[0]
            contiguous = any(other.end == spell.start for other in spells)
            if not (instant == spell.start and change.new_ticker == spell.ticker and not contiguous):
                return _Rejection("legacy_chain_contradicts_transitions",
                                  {"transition_id": change.source_id, "effective_at_utc": instant.isoformat()})
            explained.append(change)
        last = spells[-1]
        matched: dict[str, tuple[_Span, pd.Timestamp, list[IndexChange]]] = {}
        for row in self.by_ticker.get(last.ticker, []):
            # Only a security's final row carries its true ticker; earlier rows of a rejoined one do not.
            if row != self.index[(row.security_id,)][1][-1]:
                continue
            if row.end is None:
                if last.end is None:
                    matched[row.security_id] = (row, self.cutoff, [])
            elif last.start < row.end and (last.end is None or row.end <= last.end):
                # A closed row's ticker is true at its end only where a pinned deletion documents it.
                closers = [change for change in self.changes.get(last.ticker, [])
                           if change.action == "deletion" and _stamp(change.effective_at_utc) == row.end]
                if closers:
                    matched[row.security_id] = (row, row.end, closers)
        if len(matched) != 1:
            return _Rejection("no_candidate" if not matched else "ambiguous_candidates",
                              {"ticker": last.ticker, "targets": sorted(matched)})
        target, (row, instant, closers) = next(iter(matched.items()))
        cited = sorted(explained, key=lambda change: (change.effective_at, change.source_id))
        dates = [change.effective_at.tz_convert(_NEW_YORK).date() for change in cited]
        if closers:
            dates.append(_first_published(closers))
        return _Proof(target, "cusip_chain_end_ticker_match", {
            "cusip_chain": source, "matched_ticker": last.ticker, "matched_instant_utc": instant.isoformat(),
            "matched_row_from_utc": row.start.isoformat(), "matched_row_open": row.end is None,
            "closing_deletion": _events(closers),
            "transitions": [{"id": change.source_id, "effective_at_utc": change.effective_at.isoformat(),
                             "old_ticker": change.old_ticker, "new_ticker": change.new_ticker} for change in cited]},
            max([*dates, instant.tz_convert(_NEW_YORK).date()]))


def _legacy(frames: Sequence[pd.DataFrame]) -> dict[str, list[_Span]]:
    merged: dict[str, list[_Span]] = {}
    for frame in frames:
        spans = _authority(frame)
        _index(spans, lambda row: (row.security_id, row.ticker))
        grouped: dict[str, list[_Span]] = {}
        for span in spans:
            grouped.setdefault(span.security_id, []).append(span)
        for security, rows in grouped.items():
            rows.sort(key=lambda row: row.start)
            if merged.setdefault(security, rows) != rows:
                raise DataReadinessError(f"legacy membership files disagree on one query identity: {security}")
    _index([span for rows in merged.values() for span in rows], lambda row: (row.security_id,))
    return merged


def _conflicts(proofs: list[dict[str, Any]], others: list[tuple[str, pd.Timestamp, pd.Timestamp | None, str]]) -> None:
    claims = others + [(row["target_security_id"], row["effective_from_utc"], row["effective_to_utc"],
                        f"proof:{row['source_security_id']}") for row in proofs]
    by_target: dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None, str]]] = {}
    for target, start, end, label in claims:
        by_target.setdefault(target, []).append((start, end, label))
    for row in proofs:
        for start, end, label in by_target[row["target_security_id"]]:
            if label == f"proof:{row['source_security_id']}":
                continue
            if (end is None or row["effective_from_utc"] < end) and (
                    row["effective_to_utc"] is None or start < row["effective_to_utc"]):
                raise DataReadinessError(
                    f"legacy identity proof overlaps another claim on {row['target_security_id']}: {label}")


def build_legacy_query_proofs(
    legacy_memberships: Sequence[pd.DataFrame],
    target_memberships: pd.DataFrame,
    changes: Sequence[IndexChange],
    transitions: pd.DataFrame,
    bridge: pd.DataFrame,
    *,
    corrected_security_ids: Collection[str],
    target_cutoff_utc: pd.Timestamp,
    evidence_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prove or reject every legacy spell whose ID is neither a CIK-bridge source nor a target ID.

    Target authorities must be global, not cohort-filtered. Corrected targets are
    rejected: only their separately corrected collection may supply them.
    """
    evidence = _sha(evidence_sha256)
    cutoff = _required_utc(target_cutoff_utc, "target_cutoff_utc")
    legacy = _legacy(legacy_memberships)
    members = _authority(target_memberships)
    for key in (lambda row: (row.ticker,), lambda row: (row.security_id,)):
        _index(members, key)
    bridge_rows = _authority(_frame(bridge, BRIDGE_COLUMNS).rename(columns={"target_security_id": "security_id"}))
    sources = {_text(value, "source_security_id") for value in bridge.source_security_id}
    targets = {member.security_id for member in members}
    builder = _Builder(legacy, members, changes, transitions, corrected_security_ids, cutoff)
    proofs: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []

    def reject(source: str, spell: _Span, outcome: _Rejection) -> None:
        _append(rejections, {"source_security_id": source, "ticker": spell.ticker, "legacy_spell_from_utc": spell.start,
                             "legacy_spell_to_utc": spell.end, "reason": outcome.reason, "detail_json": _json(outcome.detail)})

    for source in sorted(set(legacy) - sources - targets):
        spells = legacy[source]
        outcome = builder.resolve(source)
        if isinstance(outcome, _Proof) and outcome.target in builder.corrected:
            outcome = _Rejection("corrected_security_uses_corrections_archive_only", {"target": outcome.target})
        if isinstance(outcome, _Rejection):
            for spell in spells:
                reject(source, spell, outcome)
            continue
        rows = builder.index[(outcome.target,)][1]
        for spell in spells:
            overlapping = list(_overlaps(builder.index, (outcome.target,), spell.start, spell.end))
            # A share-class guard (a bare CIK target is one listed line): it can only reject, never attribute.
            if outcome.kind == "cik_equal" and any(row.ticker != spell.ticker for row in overlapping):
                reject(source, spell, _Rejection("cik_equal_ticker_differs", {"target": outcome.target,
                       "target_tickers": sorted({row.ticker for row in overlapping})}))
                continue
            if not overlapping:
                reject(source, spell, _Rejection("no_membership_intersection", {"target": outcome.target,
                       "target_rows": len(rows)}))
                continue
            for row in overlapping:
                proof = _intersection(spell, row)
                record = {"source_security_id": source, "ticker": spell.ticker, "target_security_id": outcome.target,
                          "effective_from_utc": proof.start, "effective_to_utc": proof.end,
                          "available_at_utc": proof.available, "legacy_spell_from_utc": spell.start,
                          "legacy_spell_to_utc": spell.end, "proof_kind": outcome.kind,
                          "evidence_json": _json(outcome.evidence),
                          "evidence_complete_date": outcome.complete.isoformat() if outcome.complete else None}
                record["proof_row_sha256"] = _hash("legacy_query_proof", evidence, record)
                _append(proofs, record)
    _conflicts(proofs, [(row.security_id, row.start, row.end, "cik_bridge") for row in bridge_rows]
               + [(source, span.start, span.end, "identity_equal") for source in sorted(set(legacy) & targets)
                  for span in legacy[source]])
    proof_frame = _typed(pd.DataFrame(proofs, columns=PROOF_COLUMNS), _TIMES)
    rejection_frame = _typed(pd.DataFrame(rejections, columns=REJECTION_COLUMNS), _TIMES[3:])
    for frame in (proof_frame, rejection_frame):
        _bounded(frame)
    proof_frame = proof_frame.sort_values(["source_security_id", "effective_from_utc", "target_security_id"],
                                          kind="stable").reset_index(drop=True)
    _proof_index(proof_frame)
    if not rejection_frame.reason.isin(REJECTION_REASONS).all():
        raise DataReadinessError("legacy identity rejections require known reasons")
    return proof_frame, rejection_frame.sort_values(["source_security_id", "legacy_spell_from_utc"],
                                                    kind="stable").reset_index(drop=True)


def _typed(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.as_unit("ns")
    return frame


def translation_index(frame: pd.DataFrame, *, source_column: str, digest_column: str) -> _Index:
    """Index target intervals by (source ID, query ticker); never overlapping per key."""
    spans = _authority(frame.rename(columns={"target_security_id": "security_id"}))
    rows: list[_Span] = []
    for span, row in zip(spans, frame.to_dict("records"), strict=True):
        source = _text(row[source_column], source_column)
        if source == span.security_id:
            raise DataReadinessError("identity translation requires a conversion")
        rows.append(replace(span, source_id=source, digest=_sha(row[digest_column])))
    return _index(rows, lambda row: (row.source_id, row.ticker))


def _proof_index(proofs: pd.DataFrame) -> tuple[_Index, dict[str, str]]:
    checked = _frame(proofs, PROOF_COLUMNS)
    if not checked.proof_kind.isin(PROOF_KINDS).all() or checked.proof_row_sha256.duplicated().any():
        raise DataReadinessError("legacy identity proofs require known kinds and unique row hashes")
    index = translation_index(checked, source_column="source_security_id", digest_column="proof_row_sha256")
    return index, dict(zip(checked.proof_row_sha256, checked.proof_kind, strict=True))


def relations_pass(output: pd.DataFrame, index: _Index, eligible: Sequence[bool], *, status: str,
                   digest_key: str, digest_column: str) -> None:
    """Translate eligible relation rows in place at event time; the protected relations loop, parameterized."""
    # An all-null stored column can be timezone-naive, and strings are valid input
    # clocks. Validate before conversion so normalization cannot legitimize naive dates.
    validated_clocks = {}
    clocks_normalized = False
    for column, nullable in (("identity_available_at_utc", True), ("feature_available_at_utc", False)):
        values = [_utc(value, column, nullable=nullable) for value in output[column]]
        validated_clocks[column] = pd.Series(values, index=output.index, dtype="datetime64[ns, UTC]")
    if output.relation_id.duplicated().any():
        raise DataReadinessError("relation IDs must be unique")
    for position, (row, allowed) in enumerate(zip(output.to_dict("records"), eligible, strict=True)):
        original = _text(row["relation_id"], "relation_id")
        key = (_text(row["target_security_id"], "target_security_id"), _text(row["target_ticker"], "target_ticker"))
        event = _required_utc(row["event_feature_available_at_utc"], "event_feature_available_at_utc")
        feature = _required_utc(row["feature_available_at_utc"], "feature_available_at_utc")
        identity = _utc(row["identity_available_at_utc"], "identity_available_at_utc", nullable=True)
        if feature < event or (identity is not None and feature < identity):
            raise DataReadinessError("relation feature availability precedes its dependencies")
        if not allowed:
            continue
        for span in _overlaps(index, key, event, None):
            if span.start > event:
                break
            if span.available > event or (identity is not None and identity > event):
                continue
            if not clocks_normalized:
                for column, normalized_values in validated_clocks.items():
                    output[column] = normalized_values
                clocks_normalized = True
            updates = {
                "target_security_id": span.security_id,
                "relation_id": _hash("relation", original, {digest_key: span.digest}),
                "identity_translation_status": status, digest_column: span.digest,
                "identity_available_at_utc": max(identity, span.available) if identity is not None else span.available,
                "feature_available_at_utc": max(feature, span.available),
            }
            for column, value in updates.items():
                output.iat[position, output.columns.get_loc(column)] = value
    if output.relation_id.duplicated().any():
        raise DataReadinessError("translated relation IDs collide")


def coverage_pass(checked: pd.DataFrame, index: _Index, eligible: Sequence[bool], *, status: str,
                  digest_key: str, digest_column: str) -> pd.DataFrame:
    """Split eligible coverage rows at proof boundaries; the protected coverage loop, parameterized."""
    if checked.chunk_id.duplicated().any():
        raise DataReadinessError("coverage chunk IDs must be unique")
    rows: list[dict[str, Any]] = []
    for row, allowed in zip(checked.to_dict("records"), eligible, strict=True):
        original = _text(row["chunk_id"], "chunk_id")
        key = (_text(row["security_id"], "security_id"), _text(row["ticker"], "ticker"))
        start = _required_utc(row["requested_start_utc"], "requested_start_utc")
        end = _required_utc(row["requested_end_utc"], "requested_end_utc")
        if end <= start:
            raise DataReadinessError("requested coverage must be a finite nonempty half-open interval")
        if not allowed:
            _append(rows, row)
            continue
        cursor = start
        segments: list[tuple[pd.Timestamp, pd.Timestamp, _Span | None]] = []
        for span in _overlaps(index, key, start, end):
            lower, upper = max(start, span.start, span.available), min(end, span.end) if span.end is not None else end
            if lower >= upper:
                continue
            if cursor < lower:
                segments.append((cursor, lower, None))
            segments.append((lower, upper, span))
            cursor = upper
        if cursor < end:
            segments.append((cursor, end, None))
        for lower, upper, proof in segments:
            segment = {**row, "requested_start_utc": lower, "requested_end_utc": upper}
            if proof is not None:
                segment.update({"security_id": proof.security_id, "identity_translation_status": status,
                                digest_column: proof.digest, "identity_available_at_utc": proof.available})
            if proof is not None or len(segments) > 1:
                segment["chunk_id"] = _hash("coverage", original, {
                    "start": lower, "end": upper, digest_key: proof.digest if proof is not None else None,
                })
            _append(rows, segment)
    result = pd.DataFrame(rows, columns=checked.columns)
    for column in ("requested_start_utc", "requested_end_utc", "identity_available_at_utc"):
        result[column] = pd.to_datetime(result[column], utc=True)
    if result.chunk_id.duplicated().any():
        raise DataReadinessError("translated coverage chunk IDs collide")
    _bounded(result)
    return result


_LEGACY_COLUMNS = ("identity_legacy_proof_kind", "identity_legacy_proof_row_sha256")


def _second_pass(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    checked = _frame(frame, (*columns, "identity_translation_status", "identity_bridge_row_sha256"))
    if set(_LEGACY_COLUMNS).intersection(checked.columns):
        raise DataReadinessError("legacy identity translation requires CIK-translated, legacy-untranslated input")
    if not checked.identity_translation_status.isin(("mapped", "unmapped")).all():
        raise DataReadinessError("legacy identity translation requires CIK-pass statuses")
    checked["identity_legacy_proof_row_sha256"] = ""
    return checked


def map_legacy_query_relations(relations: pd.DataFrame, proofs: pd.DataFrame) -> pd.DataFrame:
    """Translate only relations the CIK bridge left unmapped, at event time."""
    index, kinds = _proof_index(proofs)
    output = _second_pass(relations, ("relation_id", "target_security_id", "target_ticker",
                                      "event_feature_available_at_utc", "identity_available_at_utc",
                                      "feature_available_at_utc"))
    relations_pass(output, index, output.identity_translation_status.eq("unmapped").tolist(), status=LEGACY_STATUS,
                   digest_key="legacy_proof_row_sha256", digest_column="identity_legacy_proof_row_sha256")
    output["identity_legacy_proof_kind"] = output.identity_legacy_proof_row_sha256.map(kinds).fillna("")
    _bounded(output)
    return output


def map_legacy_query_coverage(coverage: pd.DataFrame, proofs: pd.DataFrame) -> pd.DataFrame:
    """Split only coverage the CIK bridge left unmapped, without inventing coverage or availability."""
    index, kinds = _proof_index(proofs)
    checked = _second_pass(coverage, ("chunk_id", "security_id", "ticker", "requested_start_utc",
                                      "requested_end_utc", "identity_available_at_utc"))
    result = coverage_pass(checked, index, checked.identity_translation_status.eq("unmapped").tolist(),
                           status=LEGACY_STATUS, digest_key="legacy_proof_row_sha256",
                           digest_column="identity_legacy_proof_row_sha256")
    result["identity_legacy_proof_kind"] = result.identity_legacy_proof_row_sha256.map(kinds).fillna("")
    return result
