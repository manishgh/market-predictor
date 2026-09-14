"""Bind adjusted predictor history and raw dollar volume to corrected decisions."""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.canonical.normalize import canonicalize_bars
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import resolve_inside_authority
from market_predictor.swing.contracts.corrected_outcomes import CorrectedOutcomePolicy
from market_predictor.swing.contracts.holding_accounting import EvidenceReference
from market_predictor.swing.contracts.holding_materialization import PositionSourceBinding
from market_predictor.swing.datasets.corrected_outcome_admission import corrected_ticker
from market_predictor.swing.datasets.holding_raw_sources import RAW_COLUMNS, read_bound_observations
from market_predictor.swing.features.research_join import DECISION_KEYS


def corrected_adjusted_bars(archive: Path, record: dict[str, Any]) -> pd.DataFrame:
    """Consume one record from the independently replayed two-issuer collection."""
    path = resolve_inside_authority(archive, record["bars_path"])
    if file_sha256(path) != record["bars_sha256"]:
        raise DataReadinessError("corrected adjusted feature source changed")
    frame = pd.read_parquet(path, columns=RAW_COLUMNS,
        filters=[("session_date", ">=", "2018-05-29"), ("session_date", "<=", "2024-05-28")])
    if file_sha256(path) != record["bars_sha256"] or len(frame) != record["rows"]:
        raise DataReadinessError("corrected adjusted feature inventory differs")
    if (not frame.security_id.eq(record["security_id"]).all() or not frame.ticker.eq(record["ticker"]).all()
            or not frame.source.eq("alpaca").all() or not frame.price_feed.eq("sip").all()
            or not frame.adjustment.eq("all").all() or not frame.timeframe.eq("1Day").all()):
        raise DataReadinessError("corrected adjusted feature issuer/feed/basis mismatch")
    dates = pd.to_datetime(frame.session_date, errors="raise").dt.date
    if not dates.between(date(2018, 5, 29), date(2024, 5, 28)).all():
        raise DataReadinessError("corrected adjusted history escapes warm-up/initial-fit bounds")
    return canonicalize_bars(frame, timeframe="1d", availability_policy="market_interval_close")


def raw_dollar_volume_inputs(
    *, root: Path, selection: dict[str, Any], decisions: pd.DataFrame,
    policy: CorrectedOutcomePolicy, policy_sha256: str,
) -> pd.DataFrame:
    """Read exact raw decision observations, not adjusted dollar-volume proxies.

    Raw archives and ticker interpretation must already have passed the shared source
    verification context. A missing or unusable observation is left absent for the
    feature builder to represent as unavailable on the preserved decision row.
    """
    columns = [*DECISION_KEYS, "close", "volume", "available_at_utc", "price_feed", "adjustment"]
    if decisions.empty or decisions.security_id.nunique() != 1:
        raise DataReadinessError("raw feature binding requires one non-empty issuer decision group")
    identity = str(decisions.security_id.iloc[0])
    first, last = min(decisions.session_date_et), max(decisions.session_date_et)
    parts: list[pd.DataFrame] = []
    for segment in selection["segments"]:
        artifact = segment["artifact"]
        if artifact["security_id"] != identity or artifact["role"] != "stock":
            continue
        lower = max(first, date.fromisoformat(segment["first_session"]))
        upper = min(last, date.fromisoformat(segment["last_session"]))
        if lower > upper:
            continue
        reference = EvidenceReference(reference=policy.source_selection.path,
            artifact_sha256=policy.source_selection.sha256, interpretation_policy_sha256=policy_sha256,
            record_locator=f"raw decision-volume source unit {artifact['unit_id']}; not a trade or availability receipt",
            retrieved_at=datetime.now(UTC), available_at=None)
        binding = PositionSourceBinding(position_id=f"feature:{artifact['unit_id']}", security_id=identity,
            unit_id=artifact["unit_id"], provider_symbol=artifact["provider_symbol"], bars_sha256=artifact["bars_sha256"],
            first_session=lower, last_session=upper, ownership_evidence=(reference,))
        observed, _ = read_bound_observations(root, selection, binding, first=lower, last=upper,
            interpretation_sha256=policy_sha256)
        if observed.empty:
            continue
        observed = observed.loc[observed.outcome_observation_valid].reset_index()
        observed["ticker"] = [corrected_ticker(identity, artifact["ticker"], day, policy.decision_corrections)
            for day in observed.session_date_et]
        # Match the established historical daily-bar proxy, not retrospective retrieval.
        observed["available_at_utc"] = observed.bar_end_utc + pd.Timedelta(minutes=15)
        payload = observed.loc[:, ["session_date_et", "ticker", "close", "volume", "available_at_utc", "price_feed", "adjustment"]]
        merged = decisions.loc[:, [*DECISION_KEYS, "session_date_et"]].merge(
            payload, on=["ticker", "session_date_et"], how="inner", validate="one_to_one")
        parts.append(merged.loc[:, columns])
    output = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    if output.decision_id.duplicated().any():
        raise DataReadinessError("more than one selected raw source owns a feature decision")
    return output
