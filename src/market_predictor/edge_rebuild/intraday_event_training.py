"""Research-only event cohort binding for intraday development training."""
from __future__ import annotations



from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.intraday_event_preflight import (
    load_intraday_event_preflight,
)
from market_predictor.edge_rebuild.intraday_training import PublishedIntradayDataset
from market_predictor.edge_rebuild.issuer_event_precision_audit import (
    issuer_event_rule_variant,
)
from market_predictor.resources import release_process_memory
from market_predictor.core.errors import DataReadinessError

EVENT_COHORT_SCHEMA: Final = "edge_rebuild.intraday_research_event_cohort.v1"
MINIMUM_EVENT_EPISODES: Final = 1_000
DIRECTIONAL_EVENT_SUBTYPES: Final = frozenset(
    {"bare_upgrade", "bare_downgrade", "coverage"}
)
MINIMUM_DIRECTIONAL_EVENT_EPISODES: Final = 500
MINIMUM_DIRECTIONAL_SECURITIES: Final = 200
MINIMUM_DIRECTIONAL_SESSIONS: Final = 200
MINIMUM_DIRECTIONAL_EVENTS_PER_VALIDATION_FOLD: Final = 100
MINIMUM_DIRECTIONAL_UNSEEN_SECURITY_EVENTS: Final = 100
EXPECTED_PROXY_BLOCKERS: Final = frozenset(
    {
        "historical_availability_proxy_only",
        "no_production_eligible_events",
        "no_production_eligible_decisions",
    }
)


@dataclass(frozen=True, slots=True)
class IntradayResearchEventCohort:
    """Exact A4.3 decisions occurring within 24 hours after a broker event."""

    decision_ids: frozenset[str]
    identity: dict[str, Any]


def load_intraday_research_event_cohort(
    directory: Path,
    *,
    verified_dataset: PublishedIntradayDataset | None = None,
    event_subtype: str | None = None,
) -> IntradayResearchEventCohort:
    """Strictly load a proxy-time cohort that can never authorize production."""

    authority = load_intraday_event_preflight(
        directory,
        verified_dataset=verified_dataset,
        retain_verified_parent_events=event_subtype is not None,
    )
    manifest = authority.manifest
    blockers = manifest.get("blockers")
    if (
        manifest.get("status") != "blocked"
        or manifest.get("training_eligible") is not False
        or manifest.get("serving_eligible") is not False
        or manifest.get("future_holdout_opened") is not False
        or not isinstance(blockers, list)
        or not EXPECTED_PROXY_BLOCKERS.issubset(set(map(str, blockers)))
    ):
        raise DataReadinessError(
            "intraday historical event cohort must remain proxy-time research-only"
        )
    attached = authority.attachments.loc[
        authority.attachments["decision_id"].astype(str).ne("")
        & authority.attachments["research_eligible"].astype(bool)
    ].copy()
    if attached.empty:
        raise DataReadinessError("intraday historical event cohort has no attached decisions")
    if not attached["identity_alignment"].astype(str).eq(
        "exact_ticker_cik_compatible"
    ).all():
        raise DataReadinessError("intraday historical event cohort identity is not exact")
    available = pd.to_datetime(attached["feature_available_at_utc"], utc=True, errors="coerce")
    decision_time = pd.to_datetime(attached["decision_time_utc"], utc=True, errors="coerce")
    if available.isna().any() or decision_time.isna().any() or available.gt(decision_time).any():
        raise DataReadinessError("intraday historical event cohort uses future evidence")
    capacity: dict[str, Any] | None = None
    if event_subtype is not None:
        attached, capacity = _directional_attachments(
            attached,
            decisions=authority.decisions,
            verified_parent_events=getattr(authority, "verified_parent_events", None),
            event_subtype=event_subtype,
        )
    event_episodes = int(attached["family_event_id"].astype(str).nunique())
    if event_subtype is None and event_episodes < MINIMUM_EVENT_EPISODES:
        raise DataReadinessError(
            f"intraday historical event cohort has {event_episodes} events; "
            f"requires {MINIMUM_EVENT_EPISODES}"
        )
    decision_ids = frozenset(attached["decision_id"].astype(str))
    if not decision_ids:
        raise DataReadinessError("intraday historical event cohort decision set is empty")
    result = IntradayResearchEventCohort(
        decision_ids=decision_ids,
        identity={
            "schema": EVENT_COHORT_SCHEMA,
            "preflight_directory": str(directory.resolve()),
            "preflight_authority_sha256": file_sha256(directory / "_authority.json"),
            "preflight_manifest_sha256": file_sha256(directory / "_manifest.json"),
            "preflight_request_sha256": str(manifest.get("request_sha256", "")),
            "event_episodes": event_episodes,
            "event_decision_attachment_rows": int(len(attached)),
            "unique_decision_rows": len(decision_ids),
            "event_subtype": event_subtype,
            "directional_capacity": capacity,
            "availability_policy": "provider_publication_proxy_research_only",
            "catalyst_role": "confirmation_and_population_filter_not_model_feature",
            "production_eligible": False,
            "serving_eligible": False,
            "future_holdout_opened": False,
        },
    )
    del authority, attached
    release_process_memory()
    return result


def _directional_attachments(
    attached: pd.DataFrame,
    *,
    decisions: pd.DataFrame,
    verified_parent_events: pd.DataFrame | None,
    event_subtype: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if event_subtype not in DIRECTIONAL_EVENT_SUBTYPES:
        raise DataReadinessError(
            f"unsupported intraday analyst-event subtype: {event_subtype}"
        )
    event_subtypes = _load_event_subtypes(verified_parent_events)
    output = attached.merge(
        event_subtypes,
        on="family_event_id",
        how="left",
        validate="many_to_one",
    )
    if output["event_subtype"].isna().any():
        raise DataReadinessError("intraday event subtype lineage is incomplete")
    output = output.loc[output["event_subtype"].astype(str).eq(event_subtype)].copy()
    if output.empty:
        raise DataReadinessError(
            f"intraday analyst-event subtype has no attached decisions: {event_subtype}"
        )
    required_decisions = {
        "decision_id",
        "security_id",
        "session_date_et",
        "development_fold",
        "validation_scope",
    }
    missing = required_decisions.difference(decisions.columns)
    if missing:
        raise DataReadinessError(
            f"intraday directional capacity spine is incomplete: {sorted(missing)}"
        )
    spine = decisions.loc[:, sorted(required_decisions)].copy()
    if spine["decision_id"].astype(str).duplicated().any():
        raise DataReadinessError("intraday directional decision identity is duplicated")
    output = output.merge(spine, on="decision_id", how="left", validate="many_to_one")
    if output["security_id_y"].isna().any():
        raise DataReadinessError("intraday directional decisions are absent from preflight")
    if output["security_id_x"].astype(str).ne(output["security_id_y"].astype(str)).any():
        raise DataReadinessError("intraday directional security identity differs")
    event_ids = output["family_event_id"].astype(str)
    fold_counts = {
        str(int(fold)): int(group["family_event_id"].astype(str).nunique())
        for fold, group in output.loc[output["development_fold"].astype(int).ge(0)].groupby(
            "development_fold", sort=True
        )
    }
    validation_folds = tuple(str(index) for index in range(4))
    event_episodes = int(event_ids.nunique())
    decision_rows = int(output["decision_id"].astype(str).nunique())
    security_count = int(output["security_id_y"].astype(str).nunique())
    session_count = int(output["session_date_et"].astype(str).nunique())
    events_by_validation_fold = {
        fold: int(fold_counts.get(fold, 0)) for fold in validation_folds
    }
    unseen_security_events = int(
        output.loc[
            output["validation_scope"].astype(str).eq("unseen_security"),
            "family_event_id",
        ]
        .astype(str)
        .nunique()
    )
    capacity = {
        "event_episodes": event_episodes,
        "decision_rows": decision_rows,
        "securities": security_count,
        "sessions": session_count,
        "events_by_validation_fold": events_by_validation_fold,
        "unseen_security_events": unseen_security_events,
    }
    failures: list[str] = []
    if event_episodes < MINIMUM_DIRECTIONAL_EVENT_EPISODES:
        failures.append(
            f"events={event_episodes}<{MINIMUM_DIRECTIONAL_EVENT_EPISODES}"
        )
    if security_count < MINIMUM_DIRECTIONAL_SECURITIES:
        failures.append(
            f"securities={security_count}<{MINIMUM_DIRECTIONAL_SECURITIES}"
        )
    if session_count < MINIMUM_DIRECTIONAL_SESSIONS:
        failures.append(
            f"sessions={session_count}<{MINIMUM_DIRECTIONAL_SESSIONS}"
        )
    for fold in validation_folds:
        count = events_by_validation_fold[fold]
        if count < MINIMUM_DIRECTIONAL_EVENTS_PER_VALIDATION_FOLD:
            failures.append(
                f"fold_{fold}_events={count}<"
                f"{MINIMUM_DIRECTIONAL_EVENTS_PER_VALIDATION_FOLD}"
            )
    if unseen_security_events < MINIMUM_DIRECTIONAL_UNSEEN_SECURITY_EVENTS:
        failures.append(
            f"unseen_security_events={unseen_security_events}<"
            f"{MINIMUM_DIRECTIONAL_UNSEEN_SECURITY_EVENTS}"
        )
    if failures:
        raise DataReadinessError(
            f"intraday analyst-event subtype {event_subtype} lacks governed capacity: "
            + ", ".join(failures)
        )
    return output.drop(columns=["security_id_y"]).rename(
        columns={"security_id_x": "security_id"}
    ), capacity


def _load_event_subtypes(
    verified_parent_events: pd.DataFrame | None,
) -> pd.DataFrame:
    if verified_parent_events is None:
        raise DataReadinessError("intraday verified event subtype parents are missing")
    output = _classify_event_subtypes(verified_parent_events)
    if output["family_event_id"].astype(str).duplicated().any():
        raise DataReadinessError("intraday event subtype identity is duplicated")
    return output


def _classify_event_subtypes(events: pd.DataFrame) -> pd.DataFrame:
    required = {
        "family_event_id",
        "event_family",
        "classification_rule_id",
        "matched_text",
    }
    missing = required.difference(events.columns)
    if missing:
        raise DataReadinessError(
            f"intraday verified event subtype fields are missing: {sorted(missing)}"
        )
    output = events.loc[:, sorted(required)].copy()
    if not output["event_family"].astype(str).eq("analyst_revision").all():
        raise DataReadinessError("intraday event subtype input contains another family")
    output["event_subtype"] = [
        issuer_event_rule_variant(row) for row in output.to_dict(orient="records")
    ]
    return output.loc[:, ["family_event_id", "event_subtype"]]


def filter_to_research_event_cohort(
    frame: pd.DataFrame,
    cohort: IntradayResearchEventCohort,
) -> pd.DataFrame:
    """Filter A4.3 rows without duplicating decisions that have multiple events."""

    if "decision_id" not in frame.columns:
        raise DataReadinessError("intraday training frame lacks decision identity")
    output = frame.loc[frame["decision_id"].astype(str).isin(cohort.decision_ids)].copy()
    if output.empty:
        raise DataReadinessError("event cohort does not overlap the intraday dataset")
    if output["decision_id"].astype(str).duplicated().any():
        raise DataReadinessError("event cohort filtering duplicated an intraday decision")
    if not set(output["decision_id"].astype(str)).issubset(cohort.decision_ids):
        raise DataReadinessError("event cohort filtering changed decision identity")
    return output.reset_index(drop=True)
