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
from market_predictor.resources import release_process_memory
from market_predictor.v3.errors import DataReadinessError

EVENT_COHORT_SCHEMA: Final = "edge_rebuild.intraday_research_event_cohort.v1"
MINIMUM_EVENT_EPISODES: Final = 1_000
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
) -> IntradayResearchEventCohort:
    """Strictly load a proxy-time cohort that can never authorize production."""

    authority = load_intraday_event_preflight(
        directory,
        verified_dataset=verified_dataset,
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
    event_episodes = int(attached["family_event_id"].astype(str).nunique())
    if event_episodes < MINIMUM_EVENT_EPISODES:
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
