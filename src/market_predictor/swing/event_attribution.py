from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError

ATTRIBUTION_POLICY_VERSION = "swing.event_attribution.v3"
RelationChannel = Literal["direct_issuer", "business_exposure", "sector_context"]
LabelType = Literal["offering", "driver", "end_market"]
RelationUse = Literal["exposure", "context"]

_TOKEN = re.compile(r"[a-z0-9]+")
_LEGAL_SUFFIXES = {
    "class",
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holdings",
    "inc",
    "incorporated",
    "limited",
    "ltd",
    "plc",
}
_AMBIGUOUS_BARE_TICKERS = frozenset(
    {
        "A",
        "AI",
        "ALL",
        "AM",
        "APP",
        "ARE",
        "BE",
        "BIG",
        "BOX",
        "CAN",
        "CAR",
        "CAT",
        "COST",
        "DAY",
        "DOC",
        "ELF",
        "FAST",
        "FOR",
        "HAS",
        "IT",
        "KEY",
        "LIFE",
        "LOVE",
        "LOW",
        "MAN",
        "NOW",
        "ON",
        "OUT",
        "PAY",
        "PLAY",
        "SEE",
        "SO",
        "T",
        "TAP",
        "TEAM",
        "UP",
        "YOU",
    }
)
_AMBIGUOUS_SINGLE_COMPANY_TERMS = frozenset(
    {
        "aerospace",
        "alphabet",
        "amazon",
        "apple",
        "ball",
        "bank",
        "booking",
        "connectivity",
        "dover",
        "dow",
        "fox",
        "gap",
        "global",
        "healthcare",
        "hunt",
        "industries",
        "match",
        "mobile",
        "news",
        "pool",
        "progressive",
        "smith",
        "snap",
        "southern",
        "target",
        "waters",
    }
)
_EVENT_REQUIRED = {
    "event_id",
    "security_id",
    "ticker",
    "feature_available_at_utc",
    "title",
    "summary",
    "text",
}
_LABEL_REQUIRED = {
    "security_id",
    "ticker",
    "company",
    "business_tag",
    "label_type",
    "match_terms",
    "tag_rank",
    "confidence",
    "relation_use",
    "effective_from_utc",
    "effective_to_utc",
    "available_at_utc",
}
_IDENTITY_REQUIRED = {
    "security_id",
    "ticker",
    "company",
    "effective_from_utc",
    "effective_to_utc",
    "available_at_utc",
}
RELATION_COLUMNS = (
    "relation_id",
    "event_id",
    "source_security_id",
    "source_ticker",
    "target_security_id",
    "target_ticker",
    "relation_channel",
    "relation_score",
    "relation_basis",
    "matched_business_labels",
    "matched_label_types",
    "matched_terms",
    "event_feature_available_at_utc",
    "identity_available_at_utc",
    "label_available_at_utc",
    "feature_available_at_utc",
    "attribution_policy_version",
    "attribution_policy_sha256",
    "business_label_assignment_sha256",
    "security_identity_registry_sha256",
)
_POLICY = {
    "version": ATTRIBUTION_POLICY_VERSION,
    "identity_join": "security_id",
    "identity_registry": "hash_bound_point_in_time_security_identity",
    "effective_interval": "[effective_from_utc,effective_to_utc)",
    "label_availability": "available_at_utc<=event.feature_available_at_utc",
    "feature_availability": "max(event.feature_available_at_utc,identity.available_at_utc,matched_label.available_at_utc)",
    "direct_rule": "provider_security_tag_and(explicit_ticker_or_full_company)",
    "exposure_rule": "active_exposure_enabled_offering_or_driver_term_match",
    "context_rule": "active_context_enabled_or_end_market_term_match_without_exposure_match",
    "short_ticker_rule": "one_or_two_character_tickers_require_explicit_marker",
    "ambiguous_bare_tickers": sorted(_AMBIGUOUS_BARE_TICKERS),
    "company_rule": "full_normalized_company_name_with_legal_suffix_required_for_ambiguous_single_terms",
    "ambiguous_single_company_terms": sorted(
        _AMBIGUOUS_SINGLE_COMPANY_TERMS
    ),
    "max_active_tags": 3,
    "scores": {
        "ticker": 0.99,
        "company": 0.96,
        "provider_plus_offering_or_driver": 0.82,
        "offering_and_driver_exposure": 0.82,
        "offering_exposure": 0.75,
        "driver_exposure": 0.68,
        "end_market_context": 0.40,
    },
}
ATTRIBUTION_POLICY_SHA256 = hashlib.sha256(json.dumps(_POLICY, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _BusinessLabel:
    row_id: int
    security_id: str
    ticker: str
    company: str
    business_tag: str
    label_type: LabelType
    terms: tuple[str, ...]
    tag_rank: int
    confidence: float
    relation_use: RelationUse
    effective_from_utc: pd.Timestamp
    effective_to_utc: pd.Timestamp | None
    available_at_utc: pd.Timestamp


@dataclass(frozen=True, slots=True)
class _SecurityIdentity:
    security_id: str
    ticker: str
    company: str
    effective_from_utc: pd.Timestamp
    effective_to_utc: pd.Timestamp | None
    available_at_utc: pd.Timestamp


def build_event_security_relations(
    events: pd.DataFrame,
    business_labels: pd.DataFrame,
    security_identities: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build point-in-time direct, exposure, and context relations.

    ``business_labels`` is metadata, not a prediction target. An assignment can
    contain zero to three active tags per security. Offering and driver tags may
    establish indirect exposure; an end-market match alone is context only.
    """

    _require_columns(events, _EVENT_REQUIRED, "canonical events")
    _require_columns(
        business_labels,
        _LABEL_REQUIRED,
        "business-label assignments",
    )
    prepared_events = _prepare_events(events)
    labels = _prepare_labels(business_labels)
    identities = _prepare_identities(
        business_labels
        if security_identities is None
        else security_identities
    )
    _validate_active_tag_limit(labels)
    assignment_sha256 = _label_assignment_sha256(labels)
    identity_sha256 = _identity_registry_sha256(identities)
    labels_by_security = _labels_by_security(labels)
    identities_by_security = _identities_by_security(
        identities
    )
    term_index = _term_index(labels)

    records: list[dict[str, Any]] = []
    for event in prepared_events.to_dict(orient="records"):
        event_time = pd.Timestamp(event["feature_available_at_utc"])
        normalized_text = _normalize(f"{event['title']} {event['summary']} {event['text']}")
        active_source = _active_labels(
            labels_by_security.get(str(event["security_id"]), ()),
            event_time,
        )
        active_identity = _active_identity(
            identities_by_security.get(
                str(event["security_id"]),
                (),
            ),
            event_time,
        )
        matched = _matched_active_labels(
            normalized_text,
            event_time,
            term_index,
        )
        matched_by_security = _group_by_security(matched)

        matched_source = matched_by_security.pop(
            str(event["security_id"]),
            [],
        )
        direct = _direct_relation(
            event,
            event_time=event_time,
            normalized_text=normalized_text,
            source_identity=active_identity,
            source_labels=active_source,
            matched_source=matched_source,
            assignment_sha256=assignment_sha256,
            identity_sha256=identity_sha256,
        )
        if direct is not None:
            records.append(direct)
        elif matched_source:
            source_relation = _indirect_relation(
                event,
                event_time=event_time,
                target_labels=matched_source,
                assignment_sha256=assignment_sha256,
                identity_sha256=identity_sha256,
            )
            if source_relation is not None:
                records.append(source_relation)

        for target_security_id in sorted(matched_by_security):
            target_labels = matched_by_security[target_security_id]
            relation = _indirect_relation(
                event,
                event_time=event_time,
                target_labels=target_labels,
                assignment_sha256=assignment_sha256,
                identity_sha256=identity_sha256,
            )
            if relation is not None:
                records.append(relation)

    if not records:
        return pd.DataFrame(columns=RELATION_COLUMNS)
    output = pd.DataFrame.from_records(records, columns=RELATION_COLUMNS)
    if bool(output["relation_id"].duplicated().any()):
        raise DataReadinessError("event attribution produced duplicate relations")
    return output.sort_values(
        ["event_id", "target_security_id", "relation_channel"],
        kind="stable",
    ).reset_index(drop=True)


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    output = events.loc[:, sorted(_EVENT_REQUIRED)].copy()
    output["event_id"] = output["event_id"].fillna("").astype(str).str.strip()
    output["security_id"] = output["security_id"].fillna("").astype(str).str.strip()
    output["ticker"] = output["ticker"].fillna("").astype(str).str.upper().str.strip()
    output["feature_available_at_utc"] = _strict_utc(
        output["feature_available_at_utc"],
        "canonical events.feature_available_at_utc",
    )
    if bool(output["event_id"].eq("").any() or output["security_id"].eq("").any() or output["ticker"].eq("").any()):
        raise DataReadinessError("canonical events contain empty identities")
    if bool(output["event_id"].duplicated().any()):
        raise DataReadinessError("canonical events contain duplicate event IDs")
    for column in ("title", "summary", "text"):
        output[column] = output[column].fillna("").astype(str)
    return output.sort_values(
        ["feature_available_at_utc", "event_id"],
        kind="stable",
    ).reset_index(drop=True)


def _prepare_labels(frame: pd.DataFrame) -> list[_BusinessLabel]:
    data = frame.loc[:, sorted(_LABEL_REQUIRED)].copy()
    for column in (
        "security_id",
        "ticker",
        "company",
        "business_tag",
        "label_type",
        "relation_use",
    ):
        data[column] = data[column].fillna("").astype(str).str.strip()
    data["ticker"] = data["ticker"].str.upper()
    data["label_type"] = data["label_type"].str.lower()
    invalid_types = sorted(set(data["label_type"]).difference({"offering", "driver", "end_market"}))
    if invalid_types:
        raise DataReadinessError(f"unsupported business label types: {invalid_types}")
    data["relation_use"] = data["relation_use"].str.lower()
    invalid_uses = sorted(set(data["relation_use"]).difference({"exposure", "context"}))
    if invalid_uses:
        raise DataReadinessError(f"unsupported business label relation uses: {invalid_uses}")
    data["effective_from_utc"] = _strict_utc(
        data["effective_from_utc"],
        "business labels.effective_from_utc",
    )
    data["effective_to_utc"] = _strict_utc(
        data["effective_to_utc"],
        "business labels.effective_to_utc",
        allow_null=True,
    )
    data["available_at_utc"] = _strict_utc(
        data["available_at_utc"],
        "business labels.available_at_utc",
    )
    ranks = pd.to_numeric(data["tag_rank"], errors="coerce")
    confidence = pd.to_numeric(data["confidence"], errors="coerce")
    if bool(ranks.isna().any() or ranks.mod(1).ne(0).any() or ranks.lt(1).any() or ranks.gt(3).any()):
        raise DataReadinessError("business tag ranks must be integers from 1 to 3")
    if bool(confidence.isna().any() or confidence.lt(0).any() or confidence.gt(1).any()):
        raise DataReadinessError("business tag confidence must be within [0, 1]")
    data["tag_rank"] = ranks.astype(int)
    data["confidence"] = confidence.astype(float)
    if bool(
        data["security_id"].eq("").any() or data["ticker"].eq("").any() or data["company"].eq("").any() or data["business_tag"].eq("").any()
    ):
        raise DataReadinessError("business labels contain empty required values")

    labels: list[_BusinessLabel] = []
    for row_id, row in enumerate(data.to_dict(orient="records")):
        effective_to = row["effective_to_utc"]
        if pd.notna(effective_to) and effective_to <= row["effective_from_utc"]:
            raise DataReadinessError("business label effective intervals must be half-open and positive")
        terms = _terms(row["match_terms"], str(row["business_tag"]))
        if not terms:
            raise DataReadinessError("business labels require match terms")
        labels.append(
            _BusinessLabel(
                row_id=row_id,
                security_id=str(row["security_id"]),
                ticker=str(row["ticker"]),
                company=str(row["company"]),
                business_tag=str(row["business_tag"]),
                label_type=str(row["label_type"]),  # type: ignore[arg-type]
                terms=terms,
                tag_rank=int(row["tag_rank"]),
                confidence=float(row["confidence"]),
                relation_use=str(row["relation_use"]),  # type: ignore[arg-type]
                effective_from_utc=pd.Timestamp(row["effective_from_utc"]),
                effective_to_utc=(None if pd.isna(effective_to) else pd.Timestamp(effective_to)),
                available_at_utc=pd.Timestamp(row["available_at_utc"]),
            )
        )
    return labels


def _prepare_identities(
    frame: pd.DataFrame,
) -> list[_SecurityIdentity]:
    _require_columns(
        frame,
        _IDENTITY_REQUIRED,
        "security identities",
    )
    data = frame.loc[:, sorted(_IDENTITY_REQUIRED)].copy()
    for column in ("security_id", "ticker", "company"):
        data[column] = data[column].fillna("").astype(str).str.strip()
    data["ticker"] = data["ticker"].str.upper()
    if bool(
        data["security_id"].eq("").any()
        or data["ticker"].eq("").any()
        or data["company"].eq("").any()
    ):
        raise DataReadinessError(
            "security identities contain empty required values"
        )
    data["effective_from_utc"] = _strict_utc(
        data["effective_from_utc"],
        "security identities.effective_from_utc",
    )
    data["effective_to_utc"] = _strict_utc(
        data["effective_to_utc"],
        "security identities.effective_to_utc",
        allow_null=True,
    )
    data["available_at_utc"] = _strict_utc(
        data["available_at_utc"],
        "security identities.available_at_utc",
    )
    identities: list[_SecurityIdentity] = []
    keys = [
        "security_id",
        "ticker",
        "company",
        "effective_from_utc",
        "effective_to_utc",
    ]
    for key, group in data.groupby(
        keys,
        dropna=False,
        sort=True,
        observed=True,
    ):
        (
            security_id,
            ticker,
            company,
            effective_from,
            effective_to,
        ) = key
        start = pd.Timestamp(effective_from)
        end = (
            None
            if pd.isna(effective_to)
            else pd.Timestamp(effective_to)
        )
        if end is not None and end <= start:
            raise DataReadinessError(
                "security identity intervals must be half-open and positive"
            )
        identities.append(
            _SecurityIdentity(
                security_id=str(security_id),
                ticker=str(ticker),
                company=str(company),
                effective_from_utc=start,
                effective_to_utc=end,
                available_at_utc=pd.Timestamp(
                    group["available_at_utc"].min()
                ),
            )
        )
    _validate_identity_intervals(identities)
    return identities


def _validate_active_tag_limit(labels: list[_BusinessLabel]) -> None:
    for security_id, security_labels in _labels_by_security(labels).items():
        points = (
            {label.effective_from_utc for label in security_labels}
            | {label.available_at_utc for label in security_labels}
            | {label.effective_to_utc for label in security_labels if label.effective_to_utc is not None}
        )
        for point in sorted(points):
            active = _active_labels(security_labels, point)
            if len(active) > 3:
                raise DataReadinessError(f"{security_id} has more than three active business tags")
            ranks = [label.tag_rank for label in active]
            if len(ranks) != len(set(ranks)):
                raise DataReadinessError(f"{security_id} has duplicate active business tag ranks")


def _active_labels(
    labels: Iterable[_BusinessLabel],
    timestamp: pd.Timestamp,
) -> list[_BusinessLabel]:
    return [
        label
        for label in labels
        if label.available_at_utc <= timestamp
        and label.effective_from_utc <= timestamp
        and (label.effective_to_utc is None or timestamp < label.effective_to_utc)
    ]


def _active_identity(
    identities: Iterable[_SecurityIdentity],
    timestamp: pd.Timestamp,
) -> _SecurityIdentity | None:
    active = [
        identity
        for identity in identities
        if identity.available_at_utc <= timestamp
        and identity.effective_from_utc <= timestamp
        and (
            identity.effective_to_utc is None
            or timestamp < identity.effective_to_utc
        )
    ]
    if len(active) > 1:
        raise DataReadinessError(
            "security identity intervals overlap at event time"
        )
    return active[0] if active else None


def _term_index(
    labels: list[_BusinessLabel],
) -> dict[str, tuple[tuple[str, _BusinessLabel], ...]]:
    by_first_token: dict[str, list[tuple[str, _BusinessLabel]]] = {}
    for label in labels:
        for term in label.terms:
            first = term.split(" ", maxsplit=1)[0]
            by_first_token.setdefault(first, []).append((term, label))
    return {token: tuple(sorted(values, key=lambda item: (item[0], item[1].row_id))) for token, values in by_first_token.items()}


def _matched_active_labels(
    normalized_text: str,
    timestamp: pd.Timestamp,
    term_index: Mapping[str, tuple[tuple[str, _BusinessLabel], ...]],
) -> list[tuple[_BusinessLabel, tuple[str, ...]]]:
    matched_terms: dict[int, set[str]] = {}
    labels: dict[int, _BusinessLabel] = {}
    for token in set(_tokens(normalized_text)):
        for term, label in term_index.get(token, ()):
            if (
                _contains_phrase(normalized_text, term)
                and label.available_at_utc <= timestamp
                and label.effective_from_utc <= timestamp
                and (label.effective_to_utc is None or timestamp < label.effective_to_utc)
            ):
                labels[label.row_id] = label
                matched_terms.setdefault(label.row_id, set()).add(term)
    return [(labels[row_id], tuple(sorted(matched_terms[row_id]))) for row_id in sorted(labels)]


def _group_by_security(
    matches: list[tuple[_BusinessLabel, tuple[str, ...]]],
) -> dict[str, list[tuple[_BusinessLabel, tuple[str, ...]]]]:
    grouped: dict[str, list[tuple[_BusinessLabel, tuple[str, ...]]]] = {}
    for match in matches:
        grouped.setdefault(match[0].security_id, []).append(match)
    return grouped


def _direct_relation(
    event: Mapping[str, Any],
    *,
    event_time: pd.Timestamp,
    normalized_text: str,
    source_identity: _SecurityIdentity | None,
    source_labels: list[_BusinessLabel],
    matched_source: list[tuple[_BusinessLabel, tuple[str, ...]]],
    assignment_sha256: str,
    identity_sha256: str,
) -> dict[str, Any] | None:
    ticker_match = _contains_ticker(
        f"{event['title']} {event['summary']} {event['text']}",
        str(event["ticker"]),
    )
    company_match = (
        source_identity is not None
        and _contains_any(
            normalized_text,
            _company_terms(source_identity.company),
        )
    )
    if not ticker_match and not company_match:
        return None

    basis = ["provider_security_tag"]
    score = 0.0
    if ticker_match:
        basis.append("ticker_text")
        score = max(score, 0.99)
    if company_match:
        basis.append("company_text")
        score = max(score, 0.96)
    dependencies = matched_source
    if company_match and not dependencies:
        dependencies = [(label, ()) for label in source_labels]
    return _relation_record(
        event,
        target_security_id=str(event["security_id"]),
        target_ticker=str(event["ticker"]),
        channel="direct_issuer",
        score=score,
        basis=basis,
        matches=dependencies,
        event_time=event_time,
        assignment_sha256=assignment_sha256,
        identity_sha256=identity_sha256,
        identity_available_at=(
            source_identity.available_at_utc
            if source_identity is not None
            else None
        ),
    )


def _indirect_relation(
    event: Mapping[str, Any],
    *,
    event_time: pd.Timestamp,
    target_labels: list[tuple[_BusinessLabel, tuple[str, ...]]],
    assignment_sha256: str,
    identity_sha256: str,
) -> dict[str, Any] | None:
    core = [match for match in target_labels if match[0].relation_use == "exposure" and match[0].label_type in {"offering", "driver"}]
    context = [match for match in target_labels if match[0].relation_use == "context" or match[0].label_type == "end_market"]
    if core:
        types = {match[0].label_type for match in core}
        base = 0.82 if types == {"offering", "driver"} else 0.75 if "offering" in types else 0.68
        confidence = max(match[0].confidence for match in core)
        score = base * (0.5 + 0.5 * confidence)
        if context:
            score = min(1.0, score + 0.05)
        return _relation_record(
            event,
            target_security_id=core[0][0].security_id,
            target_ticker=core[0][0].ticker,
            channel="business_exposure",
            score=score,
            basis=[
                "active_offering_or_driver_term",
                *(["supporting_context_term"] if context else []),
            ],
            matches=[*core, *context],
            event_time=event_time,
            assignment_sha256=assignment_sha256,
            identity_sha256=identity_sha256,
        )
    if context:
        confidence = max(match[0].confidence for match in context)
        return _relation_record(
            event,
            target_security_id=context[0][0].security_id,
            target_ticker=context[0][0].ticker,
            channel="sector_context",
            score=0.40 * (0.5 + 0.5 * confidence),
            basis=["end_market_term_only"],
            matches=context,
            event_time=event_time,
            assignment_sha256=assignment_sha256,
            identity_sha256=identity_sha256,
        )
    return None


def _relation_record(
    event: Mapping[str, Any],
    *,
    target_security_id: str,
    target_ticker: str,
    channel: RelationChannel,
    score: float,
    basis: list[str],
    matches: list[tuple[_BusinessLabel, tuple[str, ...]]],
    event_time: pd.Timestamp,
    assignment_sha256: str,
    identity_sha256: str,
    identity_available_at: pd.Timestamp | None = None,
) -> dict[str, Any]:
    matched_labels = sorted({match[0].business_tag for match in matches})
    matched_types = sorted({match[0].label_type for match in matches})
    matched_terms = sorted({term for _, terms in matches for term in terms})
    label_times = [match[0].available_at_utc for match in matches]
    label_available = max(label_times) if label_times else pd.NaT
    dependency_times = [
        *label_times,
        *(
            [identity_available_at]
            if identity_available_at is not None
            else []
        ),
    ]
    feature_available = max([event_time, *dependency_times])
    relation_material = {
        "event_id": str(event["event_id"]),
        "target_security_id": target_security_id,
        "channel": channel,
        "matched_business_labels": matched_labels,
        "matched_label_types": matched_types,
        "matched_terms": matched_terms,
        "identity_available_at_utc": (
            identity_available_at.isoformat()
            if identity_available_at is not None
            else None
        ),
        "policy_sha256": ATTRIBUTION_POLICY_SHA256,
        "assignment_sha256": assignment_sha256,
        "identity_sha256": identity_sha256,
    }
    relation_id = hashlib.sha256(
        json.dumps(
            relation_material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "relation_id": relation_id,
        "event_id": str(event["event_id"]),
        "source_security_id": str(event["security_id"]),
        "source_ticker": str(event["ticker"]),
        "target_security_id": target_security_id,
        "target_ticker": target_ticker,
        "relation_channel": channel,
        "relation_score": round(max(0.0, min(float(score), 1.0)), 8),
        "relation_basis": "+".join(sorted(set(basis))),
        "matched_business_labels": _json_list(matched_labels),
        "matched_label_types": _json_list(matched_types),
        "matched_terms": _json_list(matched_terms),
        "event_feature_available_at_utc": event_time,
        "identity_available_at_utc": (
            identity_available_at
            if identity_available_at is not None
            else pd.NaT
        ),
        "label_available_at_utc": label_available,
        "feature_available_at_utc": feature_available,
        "attribution_policy_version": ATTRIBUTION_POLICY_VERSION,
        "attribution_policy_sha256": ATTRIBUTION_POLICY_SHA256,
        "business_label_assignment_sha256": assignment_sha256,
        "security_identity_registry_sha256": identity_sha256,
    }


def _labels_by_security(
    labels: list[_BusinessLabel],
) -> dict[str, list[_BusinessLabel]]:
    grouped: dict[str, list[_BusinessLabel]] = {}
    for label in labels:
        grouped.setdefault(label.security_id, []).append(label)
    for values in grouped.values():
        values.sort(
            key=lambda label: (
                label.effective_from_utc,
                label.tag_rank,
                label.business_tag,
            )
        )
    return grouped


def _identities_by_security(
    identities: list[_SecurityIdentity],
) -> dict[str, list[_SecurityIdentity]]:
    grouped: dict[str, list[_SecurityIdentity]] = {}
    for identity in identities:
        grouped.setdefault(
            identity.security_id,
            [],
        ).append(identity)
    for values in grouped.values():
        values.sort(
            key=lambda identity: (
                identity.effective_from_utc,
                identity.ticker,
            )
        )
    return grouped


def _validate_identity_intervals(
    identities: list[_SecurityIdentity],
) -> None:
    for security_id, values in _identities_by_security(
        identities
    ).items():
        previous_end: pd.Timestamp | None = None
        previous_open = False
        for identity in values:
            if previous_open or (
                previous_end is not None
                and identity.effective_from_utc < previous_end
            ):
                raise DataReadinessError(
                    "security identity intervals overlap for "
                    f"{security_id}"
                )
            previous_open = identity.effective_to_utc is None
            previous_end = identity.effective_to_utc


def _label_assignment_sha256(labels: list[_BusinessLabel]) -> str:
    records = [
        {
            "security_id": label.security_id,
            "ticker": label.ticker,
            "company": label.company,
            "business_tag": label.business_tag,
            "label_type": label.label_type,
            "terms": list(label.terms),
            "tag_rank": label.tag_rank,
            "confidence": label.confidence,
            "relation_use": label.relation_use,
            "effective_from_utc": label.effective_from_utc.isoformat(),
            "effective_to_utc": (label.effective_to_utc.isoformat() if label.effective_to_utc is not None else None),
            "available_at_utc": label.available_at_utc.isoformat(),
        }
        for label in labels
    ]
    records.sort(
        key=lambda item: (
            str(item["security_id"]),
            str(item["effective_from_utc"]),
            int(item["tag_rank"]),
            str(item["business_tag"]),
        )
    )
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _identity_registry_sha256(
    identities: list[_SecurityIdentity],
) -> str:
    records = [
        {
            "security_id": identity.security_id,
            "ticker": identity.ticker,
            "company": identity.company,
            "effective_from_utc": (
                identity.effective_from_utc.isoformat()
            ),
            "effective_to_utc": (
                identity.effective_to_utc.isoformat()
                if identity.effective_to_utc is not None
                else None
            ),
            "available_at_utc": (
                identity.available_at_utc.isoformat()
            ),
        }
        for identity in identities
    ]
    records.sort(
        key=lambda item: (
            str(item["security_id"]),
            str(item["effective_from_utc"]),
            str(item["ticker"]),
        )
    )
    return hashlib.sha256(
        json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _terms(value: object, business_tag: str) -> tuple[str, ...]:
    raw: list[object]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raw = []
        elif stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DataReadinessError("business label match_terms contains invalid JSON") from exc
            if not isinstance(parsed, list):
                raise DataReadinessError("business label match_terms JSON must be a list")
            raw = parsed
        else:
            raw = [stripped]
    elif isinstance(value, Iterable):
        raw = list(value)
    else:
        raise DataReadinessError("business label match_terms must be a string or iterable")
    normalized = {_normalize(str(term).replace("_", " ")) for term in [*raw, business_tag]}
    normalized.discard("")
    return tuple(sorted(normalized, key=lambda term: (-len(term), term)))


def _company_terms(company: str) -> tuple[str, ...]:
    raw_tokens = [
        token
        for token in _tokens(company)
        if len(token) >= 2
    ]
    core_tokens = [
        token
        for token in raw_tokens
        if token not in _LEGAL_SUFFIXES and len(token) >= 3
    ]
    if not core_tokens:
        return ()
    if (
        len(core_tokens) == 1
        and core_tokens[0] in _AMBIGUOUS_SINGLE_COMPANY_TERMS
    ):
        if "class" in raw_tokens:
            raw_tokens = raw_tokens[: raw_tokens.index("class")]
        full = " ".join(raw_tokens)
        return (
            (full,)
            if full != core_tokens[0]
            else ()
        )
    return (" ".join(core_tokens),)


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in phrases)


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


def _contains_ticker(text: str, ticker: str) -> bool:
    canonical = ticker.strip().upper()
    explicit_patterns = (
        rf"\${re.escape(canonical)}(?![A-Z0-9])",
        rf"\({re.escape(canonical)}\)",
        rf"\b(?:NASDAQ|NYSE|AMEX)\s*:\s*{re.escape(canonical)}\b",
        rf"\bTICKER\s*:\s*{re.escape(canonical)}\b",
    )
    if any(re.search(pattern, str(text).upper()) for pattern in explicit_patterns):
        return True
    if len(canonical) <= 2 or canonical in _AMBIGUOUS_BARE_TICKERS:
        return False
    aliases = {canonical, canonical.replace("-", ".")}
    return any(
        re.search(
            rf"(?<![A-Z0-9])\$?{re.escape(alias)}(?![A-Z0-9])",
            str(text),
        )
        is not None
        for alias in aliases
        if alias
    )


def _normalize(value: str) -> str:
    return " ".join(_tokens(value))


def _tokens(value: str) -> list[str]:
    return _TOKEN.findall(str(value).lower())


def _strict_utc(
    values: pd.Series,
    name: str,
    *,
    allow_null: bool = False,
) -> pd.Series:
    parsed: list[pd.Timestamp] = []
    for value in values:
        if value is None or pd.isna(value):
            if allow_null:
                parsed.append(pd.NaT)
                continue
            raise DataReadinessError(f"{name} contains null timestamps")
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise DataReadinessError(f"{name} contains invalid timestamps") from exc
        if timestamp.tzinfo is None:
            raise DataReadinessError(f"{name} contains timezone-naive timestamps")
        parsed.append(timestamp.tz_convert("UTC"))
    return pd.Series(parsed, index=values.index, dtype="datetime64[ns, UTC]")


def _json_list(values: Sequence[str]) -> str:
    return json.dumps(values, separators=(",", ":"))


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SchemaMismatchError(f"{name} missing columns: {', '.join(missing)}")
