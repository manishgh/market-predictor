"""Deterministic, versioned issuer-event family classification.

The classifier is intentionally high precision.  It leaves unsupported or
ambiguous headlines unclassified instead of forcing every news item into a
specialist cohort.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd

from market_predictor.v3.errors import DataReadinessError

EVENT_FAMILY_POLICY_VERSION: Final = "swing.issuer_event_family.v1"
EVENT_FAMILIES: Final = (
    "earnings",
    "guidance",
    "sec_material_event",
    "analyst_revision",
    "offering",
    "merger_acquisition",
    "regulatory_decision",
    "product_event",
)
EVENT_FAMILY_COLUMNS: Final = (
    "event_id",
    "event_family",
    "classification_rule_id",
    "classification_basis",
    "matched_text",
    "event_feature_available_at_utc",
    "event_family_policy_version",
    "event_family_policy_sha256",
)


@dataclass(frozen=True, slots=True)
class _Rule:
    family: str
    rule_id: str
    patterns: tuple[str, ...]
    source_families: tuple[str, ...] = ()


_RULES: Final = (
    _Rule(
        "earnings",
        "earnings_reported_results",
        (
            r"\b(?:reports?|announces?|posts?)\b.{0,80}\b(?:quarterly|fiscal|full[ -]year|q[1-4])\b.{0,40}\b(?:results|earnings)\b",
            r"\b(?:quarterly|fiscal|full[ -]year|q[1-4])\b.{0,40}\b(?:results|earnings)\b",
        ),
    ),
    _Rule(
        "guidance",
        "guidance_changed_or_confirmed",
        (
            r"\b(?:raises?|lifts?|increases?|lowers?|cuts?|reduces?|withdraws?|suspends?|reaffirms?|reiterates?|initiates?|issues?)\b.{0,60}\b(?:guidance|outlook|forecast)\b",
            r"\b(?:guidance|outlook|forecast)\b.{0,60}\b(?:raised|lifted|increased|lowered|cut|reduced|withdrawn|suspended|reaffirmed|reiterated|initiated|issued)\b",
        ),
    ),
    _Rule(
        "analyst_revision",
        "analyst_rating_or_target_revision",
        (
            r"\b(?:upgrades?|downgrades?|initiates? coverage|resumes? coverage)\b",
            r"\bprice target\b.{0,40}\b(?:raised|lowered|increased|cut|boosted|reduced)\b",
            r"\b(?:raises?|lowers?|cuts?|boosts?|reduces?)\b.{0,30}\bprice target\b",
        ),
    ),
    _Rule(
        "offering",
        "capital_offering_announced",
        (
            r"\b(?:public|registered direct|at[ -]the[ -]market|atm|private placement|"
            r"convertible notes?|senior notes?|common stock)\b.{0,30}\boffering\b",
            r"\b(?:launches?|prices?|closes?|completes?)\b.{0,60}\boffering\b",
            r"\bfiles?\b.{0,30}\b(?:mixed|universal)?\s*shelf\b",
        ),
    ),
    _Rule(
        "merger_acquisition",
        "transaction_announced",
        (
            r"\b(?:agrees? to acquire|to acquire|will acquire|acquires|acquisition of|to be acquired)\b",
            r"\b(?:merger agreement|merges? with|takeover offer|buyout)\b",
        ),
    ),
    _Rule(
        "regulatory_decision",
        "regulator_decision_announced",
        (
            r"\b(?:fda|ema)\b.{0,70}\b(?:approves?|approval|clears?|clearance|grants?|accepts?|rejects?|declines?)\b",
            r"\b(?:approves?|approval|clears?|clearance|grants?|accepts?|rejects?|declines?)\b.{0,70}\b(?:fda|ema)\b",
            r"\bcomplete response letter\b",
            r"\b(?:ftc|department of justice|doj)\b.{0,70}\b(?:approves?|blocks?|challenges?|clears?|settles?)\b",
        ),
    ),
    _Rule(
        "product_event",
        "product_launch_or_material_contract",
        (
            r"\b(?:launches?|unveils?|introduces?)\b.{0,100}\b(?:product|platform|service|device|drug|system|chip|software)\b",
            r"\bannounces?\b.{0,50}\bgeneral availability\b",
            r"\b(?:wins?|awarded|receives?)\b.{0,80}\b(?:contract|order)\b",
        ),
    ),
)

_SEC_MATERIAL_FORMS: Final = frozenset(
    {"8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "6-K", "6-K/A"}
)
_SEC_OFFERING_FORMS: Final = frozenset(
    {
        "S-1",
        "S-1/A",
        "S-3",
        "S-3/A",
        "424B1",
        "424B2",
        "424B3",
        "424B4",
        "424B5",
    }
)
_POLICY_MATERIAL: Final = {
    "version": EVENT_FAMILY_POLICY_VERSION,
    "classification": "multi_label_high_precision_title_rules",
    "unmatched_policy": "unclassified_not_zero",
    "rules": [
        {
            "family": rule.family,
            "rule_id": rule.rule_id,
            "patterns": list(rule.patterns),
            "source_families": list(rule.source_families),
        }
        for rule in _RULES
    ],
    "sec_material_forms": sorted(_SEC_MATERIAL_FORMS),
    "sec_offering_forms": sorted(_SEC_OFFERING_FORMS),
}
EVENT_FAMILY_POLICY_SHA256: Final = hashlib.sha256(
    json.dumps(_POLICY_MATERIAL, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def classify_event_families(events: pd.DataFrame) -> pd.DataFrame:
    """Return zero or more high-confidence family records per canonical event."""

    required = {
        "event_id",
        "source_family",
        "feature_available_at_utc",
        "title",
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise DataReadinessError(
            "event-family classification input is missing columns: "
            + ", ".join(missing)
        )
    if events.empty:
        return pd.DataFrame(columns=EVENT_FAMILY_COLUMNS)

    data = events.copy()
    data["event_id"] = data["event_id"].fillna("").astype(str).str.strip()
    if bool(data["event_id"].eq("").any() or data["event_id"].duplicated().any()):
        raise DataReadinessError(
            "event-family classification requires unique non-empty event IDs"
        )
    data["source_family"] = (
        data["source_family"].fillna("").astype(str).str.lower().str.strip()
    )
    available = pd.to_datetime(
        data["feature_available_at_utc"], utc=True, errors="coerce"
    )
    if bool(available.isna().any()):
        raise DataReadinessError(
            "event-family classification contains invalid feature availability"
        )

    records: list[dict[str, object]] = []
    for position, event in enumerate(data.to_dict(orient="records")):
        event_id = str(event["event_id"])
        source_family = str(event["source_family"])
        title = _normalized_text(event.get("title"))
        matched_families: set[str] = set()

        sec_form = str(event.get("sec_form") or "").upper().strip()
        if source_family == "sec" and sec_form in _SEC_MATERIAL_FORMS:
            records.append(
                _record(
                    event_id=event_id,
                    family="sec_material_event",
                    rule_id="sec_material_form",
                    basis="structured_sec_form",
                    matched_text=sec_form,
                    available_at=available.iloc[position],
                )
            )
            matched_families.add("sec_material_event")
        if source_family == "sec" and sec_form in _SEC_OFFERING_FORMS:
            records.append(
                _record(
                    event_id=event_id,
                    family="offering",
                    rule_id="sec_offering_form",
                    basis="structured_sec_form",
                    matched_text=sec_form,
                    available_at=available.iloc[position],
                )
            )
            matched_families.add("offering")

        for rule in _RULES:
            if rule.source_families and source_family not in rule.source_families:
                continue
            match = _first_match(title, rule.patterns)
            if match is None or rule.family in matched_families:
                continue
            records.append(
                _record(
                    event_id=event_id,
                    family=rule.family,
                    rule_id=rule.rule_id,
                    basis="deterministic_title_rule",
                    matched_text=match,
                    available_at=available.iloc[position],
                )
            )
            matched_families.add(rule.family)

    if not records:
        return pd.DataFrame(columns=EVENT_FAMILY_COLUMNS)
    output = pd.DataFrame.from_records(records, columns=EVENT_FAMILY_COLUMNS)
    if bool(output.duplicated(["event_id", "event_family"]).any()):
        raise DataReadinessError("event-family classification produced duplicate rows")
    return output.sort_values(
        ["event_id", "event_family"], kind="stable"
    ).reset_index(drop=True)


def family_records_by_event(
    events: pd.DataFrame,
) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    """Return deterministic family records keyed by source event identity."""

    classified = classify_event_families(events)
    if classified.empty:
        return {}
    result: dict[str, tuple[Mapping[str, object], ...]] = {}
    for event_id, group in classified.groupby("event_id", sort=True):
        result[str(event_id)] = tuple(
            {str(key): value for key, value in row.items()}
            for row in group.to_dict(orient="records")
        )
    return result


def _record(
    *,
    event_id: str,
    family: str,
    rule_id: str,
    basis: str,
    matched_text: str,
    available_at: pd.Timestamp,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_family": family,
        "classification_rule_id": rule_id,
        "classification_basis": basis,
        "matched_text": matched_text,
        "event_feature_available_at_utc": available_at,
        "event_family_policy_version": EVENT_FAMILY_POLICY_VERSION,
        "event_family_policy_sha256": EVENT_FAMILY_POLICY_SHA256,
    }


def _first_match(text: str, patterns: Sequence[str]) -> str | None:
    for pattern in patterns:
        matched = re.search(pattern, text, flags=re.IGNORECASE)
        if matched is not None:
            return matched.group(0)
    return None


def _normalized_text(value: object) -> str:
    return " ".join(str(value or "").strip().split()).lower()
