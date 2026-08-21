"""Verify that a membership interval's symbol claim is supported by evidence.

The point-in-time universe is built by taking each security's current ticker and
back-filling it, breaking the back-fill only where the provider's
corporate-actions feed reports a change. Where that feed is incomplete the
back-fill silently becomes a hindsight assertion: the artifact claims a security
traded under a symbol in 2021 on no evidence beyond it trading under that symbol
today.

Such a row is structurally valid and factually wrong, so hashes, authorities, and
non-overlap checks all pass. It also defeats the two existing defences, because
membership filtering applies a wrong interval faithfully and identity grouping
cannot help when the contaminated rows carry the correct security id.

This module checks the claim against observed bars. A symbol held by one
continuous security produces one continuous price series; a symbol that changed
hands produces a gap, a level break, or both. An interval whose claim cannot be
supported is excluded rather than corrected, because correcting it would mean
substituting an inferred rename date for evidence.
"""
from __future__ import annotations



import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditReport,
    audit_universe_memberships,
)
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    write_canonical_artifact,
)
from market_predictor.edge_rebuild.corpus_integrity import IntegrityThresholds
from market_predictor.core.errors import DataReadinessError

MEMBERSHIP_IDENTITY_SCHEMA = "edge_rebuild.membership_identity.v1"
REQUIRED_MEMBERSHIP_COLUMNS = (
    "ticker",
    "security_id",
    "effective_from_utc",
    "effective_to_utc",
)
REQUIRED_EVIDENCE_COLUMNS = ("session", "ticker", "bars", "last_close")


@dataclass
class MembershipVerification:
    """Which membership intervals are supported by bar evidence, and which are not."""

    schema: str = MEMBERSHIP_IDENTITY_SCHEMA
    verified: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    unevaluated: list[dict[str, Any]] = field(default_factory=list)

    @property
    def excluded_securities(self) -> set[str]:
        return {str(item["security_id"]) for item in self.excluded}

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verified_intervals": len(self.verified),
            "excluded_intervals": len(self.excluded),
            "unevaluated_intervals": len(self.unevaluated),
            "excluded_securities": sorted(self.excluded_securities),
            "exclusions": self.excluded,
        }


def publish_verified_universe(
    *,
    memberships_path: Path,
    evidence: pd.DataFrame,
    output_path: Path,
    audit_path: Path,
    thresholds: IntegrityThresholds | None = None,
) -> dict[str, Any]:
    """Publish a universe containing only evidence-supported membership intervals.

    The source artifact is never mutated. Excluded intervals are recorded with
    their reason so the reduction is auditable rather than silent.
    """

    if output_path.exists():
        raise DataReadinessError(f"verified universe output must be new: {output_path}")
    memberships, source_manifest = load_canonical_artifact(
        memberships_path,
        expected_type="memberships",
        allow_research=True,
    )
    memberships["effective_from_utc"] = pd.to_datetime(
        memberships["effective_from_utc"], utc=True
    )
    memberships["effective_to_utc"] = pd.to_datetime(
        memberships["effective_to_utc"], utc=True, errors="coerce"
    )
    result = verify_membership_identity(memberships, evidence, thresholds=thresholds)
    excluded_securities = result.excluded_securities
    kept = memberships[~memberships["security_id"].astype(str).isin(excluded_securities)]
    if kept.empty:
        raise DataReadinessError("verification excluded every membership interval")
    dropped_share = 1.0 - (kept["security_id"].nunique() / memberships["security_id"].nunique())
    validate_security_exclusion_share(
        source_securities=int(memberships["security_id"].nunique()),
        excluded_securities=len(excluded_securities),
        thresholds=thresholds,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Historical membership availability is a provider-publication proxy, so the
    # verified universe stays research-only and cannot authorize promotion.
    write_canonical_artifact(
        kept.reset_index(drop=True),
        output_path,
        artifact_type="memberships",
        audit=CanonicalAuditReport(
            # Historical membership availability is a provider-publication
            # proxy, never a prospectively observed fact, so the observed gate
            # is disabled exactly as it is on the source artifact.
            checks=audit_universe_memberships(kept, require_observed=False)
        ),
        inputs={
            "source_memberships_sha256": str(source_manifest["artifact_sha256"]),
            "identity_rule": MEMBERSHIP_IDENTITY_SCHEMA,
        },
        production_ready=False,
    )
    audit = {
        "schema": MEMBERSHIP_IDENTITY_SCHEMA,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_path": str(memberships_path),
        "source_sha256": str(source_manifest["artifact_sha256"]),
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "source_intervals": int(len(memberships)),
        "kept_intervals": int(len(kept)),
        "source_securities": int(memberships["security_id"].nunique()),
        "kept_securities": int(kept["security_id"].nunique()),
        "excluded_security_share": round(dropped_share, 6),
        "maximum_excluded_security_share": (
            thresholds or IntegrityThresholds()
        ).maximum_excluded_symbol_share,
        **result.to_record(),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def validate_security_exclusion_share(
    *,
    source_securities: int,
    excluded_securities: int,
    thresholds: IntegrityThresholds | None = None,
) -> float:
    """Allow audited whole-security exclusions only through the frozen 5% cap."""

    if source_securities < 1:
        raise ValueError("source security count must be positive")
    if not 0 <= excluded_securities <= source_securities:
        raise ValueError("excluded security count is outside the source universe")
    limits = thresholds or IntegrityThresholds()
    share = excluded_securities / source_securities
    if share > limits.maximum_excluded_symbol_share:
        raise DataReadinessError(
            "membership verification excluded "
            f"{excluded_securities} of {source_securities} securities "
            f"({share:.2%}), above the frozen "
            f"{limits.maximum_excluded_symbol_share:.2%} ceiling"
        )
    return share


def verify_membership_identity(
    memberships: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    thresholds: IntegrityThresholds | None = None,
) -> MembershipVerification:
    """Check each membership interval's symbol claim against observed bars.

    `evidence` carries one row per observed (ticker, session) with `bars` and
    `last_close`. Intervals with no overlapping evidence are reported as
    unevaluated rather than excluded, because absence of bars in a window the
    corpus never covered proves nothing either way.
    """

    limits = thresholds or IntegrityThresholds()
    for frame, columns, name in (
        (memberships, REQUIRED_MEMBERSHIP_COLUMNS, "membership"),
        (evidence, REQUIRED_EVIDENCE_COLUMNS, "evidence"),
    ):
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise DataReadinessError(
                f"{name} input lacks required columns: {missing}"
            )

    observed = evidence.loc[evidence["bars"] > 0].copy()
    observed["session"] = observed["session"].astype(str)
    observed = observed.sort_values(["ticker", "session"])
    by_ticker = {
        str(ticker): group for ticker, group in observed.groupby("ticker", sort=False)
    }
    covered = (
        (observed["session"].min(), observed["session"].max())
        if not observed.empty
        else (None, None)
    )
    rank = (
        exchange_session_rank(str(covered[0]), str(covered[1]))
        if covered[0] is not None and covered[1] is not None
        else {}
    )

    result = MembershipVerification()
    for row in memberships.itertuples():
        ticker = str(row.ticker)
        interval: dict[str, Any] = {
            "ticker": ticker,
            "security_id": str(row.security_id),
            "effective_from": str(pd.Timestamp(row.effective_from_utc).date()),
            "effective_to": (
                None
                if pd.isna(row.effective_to_utc)
                else str(pd.Timestamp(row.effective_to_utc).date())
            ),
        }
        group = by_ticker.get(ticker)
        window = _interval_evidence(group, interval)
        if window is None or len(window) < 2:
            bucket, reason = _classify_sparse_interval(group, window, covered, interval)
            interval["reason"] = reason
            (result.excluded if bucket == "excluded" else result.unevaluated).append(interval)
            continue
        breach = _identity_breach(window, limits, rank)
        if breach is None:
            result.verified.append({**interval, "evidence_sessions": len(window)})
        else:
            result.excluded.append({**interval, **breach, "evidence_sessions": len(window)})
    return result


def _interval_evidence(
    group: pd.DataFrame | None,
    interval: dict[str, Any],
) -> pd.DataFrame | None:
    if group is None or group.empty:
        return None
    inside = group[group["session"] >= interval["effective_from"]]
    if interval["effective_to"] is not None:
        inside = inside[inside["session"] < interval["effective_to"]]
    return inside


def _classify_sparse_interval(
    group: pd.DataFrame | None,
    window: pd.DataFrame | None,
    covered: tuple[str | None, str | None],
    interval: dict[str, Any],
) -> tuple[str, str]:
    """Decide whether too-little evidence is unevaluable or itself a defect.

    An interval the corpus never spans is unevaluable, and excluding it would
    discard a security for a gap in our own collection window. But a symbol with
    no bars inside its claimed interval and bars *outside* it was held by
    someone else, which is the same reuse defect a within-interval continuity
    check cannot see precisely because there is no series to check.
    """

    start, end = covered
    if start is None:
        return "unevaluated", "no_bar_corpus"
    if interval["effective_to"] is not None and interval["effective_to"] <= start:
        return "unevaluated", "interval_precedes_corpus"
    if interval["effective_from"] >= end:
        return "unevaluated", "interval_follows_corpus"
    inside = 0 if window is None else len(window)
    outside = 0 if group is None else len(group) - inside
    if inside == 0 and outside > 0:
        return "excluded", "symbol_reused_outside_interval"
    if inside == 0:
        return "excluded", "no_bar_evidence"
    return "excluded", "insufficient_evidence"


def _identity_breach(
    window: pd.DataFrame,
    limits: IntegrityThresholds,
    rank: Mapping[str, int],
) -> dict[str, Any] | None:
    """One continuous security produces one continuous series."""

    closes = pd.to_numeric(window["last_close"], errors="coerce")
    previous = closes.shift()
    pair = pd.concat([closes, previous], axis=1)
    ratio = pair.max(axis=1) / pair.min(axis=1)
    worst = ratio.max()
    if pd.notna(worst) and float(worst) > limits.maximum_session_close_ratio:
        position = ratio.idxmax()
        return {
            "reason": "symbol_changed_hands",
            "detail": "close_ratio",
            "ratio": round(float(worst), 6),
            "at_session": str(window.loc[position, "session"]),
        }
    sessions = list(window["session"])
    gap = _longest_session_gap(sessions, rank)
    if gap > limits.maximum_interior_gap_sessions:
        return {
            "reason": "symbol_unproven_for_interval",
            "detail": "interior_gap",
            "gap_sessions": gap,
            "at_session": sessions[min(gap, len(sessions) - 1)],
        }
    return None


def exchange_session_rank(
    first_session: str,
    last_session: str,
    *,
    calendar_name: str = "XNYS",
) -> dict[str, int]:
    """Ordinal position of every exchange session in a range.

    Gaps are counted in sessions, never calendar days. Market holidays fall on
    weekdays, so calendar arithmetic both overstates gaps across weekends and
    understates them across holiday weeks.
    """

    calendar = xcals.get_calendar(calendar_name)
    sessions = calendar.sessions_in_range(first_session, last_session)
    return {str(pd.Timestamp(session).date()): index for index, session in enumerate(sessions)}


def _longest_session_gap(sessions: list[str], rank: Mapping[str, int]) -> int:
    """Longest run of absent exchange sessions between observations.

    A date absent from the calendar cannot be ranked and is skipped, so a bar
    stamped on a non-session date can never manufacture a false gap.
    """

    ranks = sorted(rank[session] for session in sessions if session in rank)
    if len(ranks) < 2:
        return 0
    return int(max(later - earlier - 1 for earlier, later in zip(ranks, ranks[1:], strict=False)))
