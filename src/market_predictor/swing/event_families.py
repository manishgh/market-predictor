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

EVENT_FAMILY_POLICY_VERSION: Final = "swing.issuer_event_family.v2"
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
ALLOWED_SOURCE_FAMILIES_BY_FAMILY: Final[Mapping[str, tuple[str, ...]]] = {
    "earnings": ("alpaca", "finviz"),
    "guidance": ("alpaca", "finviz"),
    "sec_material_event": ("sec",),
    "analyst_revision": ("alpaca", "finviz"),
    "offering": ("alpaca", "finviz", "sec"),
    "merger_acquisition": ("alpaca", "finviz"),
    "regulatory_decision": ("alpaca", "finviz"),
    "product_event": ("alpaca", "finviz"),
}
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
    exclusion_patterns: tuple[str, ...] = ()
    issuer_targeted: bool = False


_RULES: Final = (
    _Rule(
        "earnings",
        "earnings_reported_results",
        (
            r"\b(?:reports?|announces?|posts?)\b.{0,80}\b(?:quarterly|fiscal|full[ -]year|q[1-4])\b.{0,40}\b(?:results|earnings)\b",
            r"\b(?:quarterly|fiscal|full[ -]year|q[1-4])\b.{0,40}\b(?:results|earnings)\b",
        ),
        (
            r"\b(?:earnings|results?)\s+(?:preview|outlook)\b",
            r"\b(?:preview|ahead of|what to expect from|estimates? for)\b.{0,50}\b(?:earnings|results?)\b",
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
            r"\b(?:upgrades?|downgrades?)\s+(?:shares?\s+(?:of|in)\s+)?{issuer}",
            r"{issuer}\s+(?:is\s+)?(?:upgraded|downgraded)\b",
            r"\b(?:initiates?|resumes?)\s+coverage\s+(?:on\s+)?{issuer}",
            r"\b(?:raises?|lowers?|cuts?|boosts?|reduces?)\s+(?:the\s+)?price target\s+(?:on|for)\s+{issuer}",
            r"\bprice target\s+(?:on|for)\s+{issuer}.{0,30}\b(?:raised|lowered|increased|cut|boosted|reduced)\b",
            r"{issuer}(?:'s)?\s+price target.{0,30}\b(?:raised|lowered|increased|cut|boosted|reduced)\b",
        ),
        issuer_targeted=True,
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
            r"\b(?:agrees? to acquire|will acquire|acquires|acquisition of|to be acquired)\b",
            r"\b(?:merger agreement|merges? with|takeover offer|buyout)\b",
        ),
        (
            r"\b(?:could|may|might|would|likely to|plans? to|seeks? to|considering|"
            r"explores?|evaluates?|potential|possible|rumou?red)\b.{0,80}"
            r"\b(?:acquire|acquisition|merger|buyout)\b",
            r"\b(?:in talks?|weighs?|mulls?|bid)\b.{0,80}\b(?:acquire|acquisition|merger|buyout)\b",
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
_COMPANY_SUFFIXES: Final = frozenset(
    {
        "co",
        "company",
        "corp",
        "corporation",
        "inc",
        "incorporated",
        "limited",
        "ltd",
        "plc",
    }
)
_AMBIGUOUS_COMPANY_PREFIXES: Final = frozenset(
    {
        "advanced",
        "american",
        "first",
        "general",
        "global",
        "international",
        "national",
        "new",
        "united",
    }
)
_POLICY_MATERIAL: Final = {
    "version": EVENT_FAMILY_POLICY_VERSION,
    "classification": "multi_label_high_precision_issuer_targeted_title_rules",
    "issuer_binding": {
        "structured": "provider_security_id_and_ticker_or_causal_company",
        "title": "explicit_ticker_or_causally_available_company_alias",
        "analyst": "action_syntactically_targets_issuer_alias",
    },
    "unmatched_policy": "unclassified_not_zero",
    "allowed_source_families_by_family": {
        family: list(ALLOWED_SOURCE_FAMILIES_BY_FAMILY[family])
        for family in EVENT_FAMILIES
    },
    "rules": [
        {
            "family": rule.family,
            "rule_id": rule.rule_id,
            "patterns": list(rule.patterns),
            "exclusion_patterns": list(rule.exclusion_patterns),
            "issuer_targeted": rule.issuer_targeted,
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
        "security_id",
        "ticker",
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
    data["security_id"] = data["security_id"].fillna("").astype(str).str.strip()
    data["ticker"] = data["ticker"].fillna("").astype(str).str.upper().str.strip()
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
        issuer_patterns = _issuer_patterns(
            event,
            event_available_at=available.iloc[position],
        )
        title_has_issuer = _first_match(title, issuer_patterns) is not None
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
            if source_family not in ALLOWED_SOURCE_FAMILIES_BY_FAMILY[rule.family]:
                continue
            if not title_has_issuer or _first_match(title, rule.exclusion_patterns) is not None:
                continue
            patterns = (
                tuple(
                    pattern.replace("{issuer}", _issuer_alternation(issuer_patterns))
                    for pattern in rule.patterns
                )
                if rule.issuer_targeted
                else rule.patterns
            )
            match = _first_match(title, patterns)
            if match is None or rule.family in matched_families:
                continue
            records.append(
                _record(
                    event_id=event_id,
                    family=rule.family,
                    rule_id=rule.rule_id,
                    basis=(
                        "issuer_targeted_title_rule"
                        if rule.issuer_targeted
                        else "issuer_anchored_title_rule"
                    ),
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


def _issuer_patterns(
    event: Mapping[str, object],
    *,
    event_available_at: pd.Timestamp,
) -> tuple[str, ...]:
    security_id = str(event.get("security_id") or "").strip()
    ticker = str(event.get("ticker") or "").upper().strip()
    company = _normalized_text(event.get("issuer_company"))
    if not security_id:
        raise DataReadinessError("event-family classification requires issuer security_id")

    patterns: set[str] = set()
    if ticker:
        escaped_ticker = re.escape(ticker.lower())
        patterns.update(
            {
                rf"\${escaped_ticker}(?![a-z0-9])",
                rf"\({escaped_ticker}\)",
                rf"\b(?:nasdaq|nyse|amex)\s*:\s*{escaped_ticker}\b",
                rf"\bticker\s*:\s*{escaped_ticker}\b",
            }
        )
    if company:
        company_available = pd.to_datetime(
            event.get("issuer_company_available_at_utc"),
            utc=True,
            errors="coerce",
        )
        if pd.isna(company_available) or pd.Timestamp(company_available) > event_available_at:
            raise DataReadinessError(
                "event-family issuer company availability is missing or post-event"
            )
        patterns.update(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            for alias in _company_aliases(company)
        )
    if not patterns:
        raise DataReadinessError(
            "event-family classification requires ticker or causal issuer company"
        )
    return tuple(sorted(patterns, key=lambda value: (-len(value), value)))


def _company_aliases(company: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", company)
    while tokens and tokens[-1] in _COMPANY_SUFFIXES:
        tokens.pop()
    if not tokens:
        return set()
    aliases = {" ".join(tokens)}
    if (
        len(tokens) > 1
        and len(tokens[0]) >= 4
        and tokens[0] not in _AMBIGUOUS_COMPANY_PREFIXES
    ):
        aliases.add(tokens[0])
    return aliases


def _issuer_alternation(patterns: Sequence[str]) -> str:
    return "(?:" + "|".join(patterns) + ")"


def _normalized_text(value: object) -> str:
    return " ".join(str(value or "").strip().split()).lower()
