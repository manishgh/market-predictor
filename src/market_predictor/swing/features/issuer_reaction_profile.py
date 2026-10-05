"""One deterministic reviewed-event selector and reaction projection for batch/live.

The publishing caller verifies event qualification, duplicate proofs, source coverage
and all authority bytes. This pure transform never grants source or model admission.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.contracts.issuer_reaction import REACTION_COLUMNS, REACTION_EVENT_COLUMNS, IssuerReactionSources
from market_predictor.swing.contracts.issuer_reaction_profile import (
    ISSUER_REACTION_PROFILE,
    LOOKBACK,
    QUALIFIED_EVENT_COLUMNS,
    REACTION_COVERAGE_COLUMNS,
    issuer_reaction_profile_sha256,
)
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_PROFILE
from market_predictor.swing.features.issuer_reaction import build_issuer_reactions
from market_predictor.swing.features.research_partition import ResearchFeaturePartition
from market_predictor.swing.features.return_relationships import _numeric, _require, _session_dates, _sessions, _utc

_STRINGS = ("security_id", "ticker", "source_family", "event_id", "event_version_sha256",
            "qualification_status", "qualification_authority_sha256", "duplicate_group_id")
_SELECTED = ("source_family", "event_id", "event_version_sha256", "event_family", "duplicate_group_id",
             "qualification_authority_sha256", "event_available_at_utc", "identity_available_at_utc")
_INTERVALS = ("reaction_session_date_et", "reaction_start_utc", "reaction_end_utc")


def _strings(frame: pd.DataFrame, names: Sequence[str], label: str) -> None:
    for name in names:
        if not frame[name].map(lambda value: isinstance(value, str) and bool(value) and value.strip() == value).all():
            raise DataReadinessError(f"{label} contains invalid {name}")


def _parent(baseline: ResearchFeaturePartition, sources: IssuerReactionSources) -> pd.DataFrame:
    issuer_reaction_profile_sha256(baseline.model_columns, sources)
    rows = baseline.rows.copy()
    _require(rows, ("decision_id", "security_id", "ticker", "session_date_et", "decision_time_utc",
                    "feature_profile", "feature_eligible", *baseline.model_columns,
                    *baseline.availability_columns.values()), "reaction parent")
    if rows.empty or rows.decision_id.duplicated().any() or rows.duplicated(["security_id", "session_date_et"]).any():
        raise DataReadinessError("reaction parent requires unique nonempty original decisions")
    _strings(rows, ("decision_id", "security_id", "ticker"), "reaction parent")
    _session_dates(rows.session_date_et, "reaction parent")
    cutoff = _utc(rows.decision_time_utc, "reaction parent cutoff")
    if (cutoff.isna().any() or not cutoff.eq(swing_prediction_cutoffs(rows.session_date_et)).all()
            or rows.session_date_et.lt(date(2019, 7, 9)).any()):
        raise DataReadinessError("reaction parent requires canonical initial swing decision clocks")
    if not rows.feature_profile.eq(RETURN_RELATIONSHIP_PROFILE).all():
        raise DataReadinessError("reaction parent must be the unchanged relationship profile")
    if not rows.feature_eligible.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise DataReadinessError("reaction parent requires explicit feature eligibility")
    if not set(baseline.model_columns).issubset(baseline.availability_columns):
        raise DataReadinessError("reaction parent omits feature clocks")
    for name in baseline.model_columns:
        values = _numeric(rows[name], f"reaction parent.{name}")
        clocks = _utc(rows[baseline.availability_columns[name]], f"reaction parent.{name} clock")
        if (values.abs().gt(np.finfo(np.float32).max).any()
                or (values.notna() & (clocks.isna() | clocks.gt(cutoff))).any()):
            raise DataReadinessError(f"reaction parent feature {name} is unavailable at decision")
    return rows


def _events(events: pd.DataFrame, sources: IssuerReactionSources) -> pd.DataFrame:
    _require(events, QUALIFIED_EVENT_COLUMNS, "reviewed event history")
    data = events.loc[:, list(QUALIFIED_EVENT_COLUMNS)].copy()
    _strings(data, _STRINGS, "reviewed event history")
    if (not data.source_family.isin(("alpaca", "sec")).all()
            or not data.qualification_status.isin(("qualified", "rejected", "unclassified")).all()
            or not data.qualification_authority_sha256.eq(sources.event_authority_sha256).all()
            or not data.event_version_sha256.str.fullmatch(r"[0-9a-f]{64}").all()
            or not data.loc[data.qualification_status.eq("qualified"), "event_family"].isin(("earnings", "guidance")).all()):
        raise DataReadinessError("reviewed event source, qualification or authority binding differs")
    for name in ("event_available_at_utc", "identity_available_at_utc"):
        data[name] = _utc(data[name], f"reviewed event {name}").astype("datetime64[ns, UTC]")
        if data[name].isna().any():
            raise DataReadinessError("reviewed events require nonnull version and identity clocks")
    keys = ["source_family", "event_id", "security_id", "event_version_sha256"]
    data = data.drop_duplicates().reset_index(drop=True)
    if data.duplicated(keys).any():
        raise DataReadinessError("reviewed event version has contradictory evidence")
    if data.groupby(["source_family", "event_id", "event_version_sha256"]).event_available_at_utc.nunique().gt(1).any():
        raise DataReadinessError("reviewed event version has contradictory availability")
    return data


def _coverage(coverage: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    _require(coverage, REACTION_COVERAGE_COLUMNS, "reaction source coverage")
    data = coverage.loc[:, list(REACTION_COVERAGE_COLUMNS)].copy()
    if data.decision_id.duplicated().any() or set(data.decision_id) != set(rows.decision_id):
        raise DataReadinessError("reaction coverage must bind every original decision exactly once")
    if not data.coverage_status.isin(("known", "unknown")).all():
        raise DataReadinessError("reaction coverage has unsupported status")
    data = data.set_index("decision_id").loc[rows.decision_id].reset_index()
    clocks = _utc(data.available_at_utc, "reaction coverage clock").astype("datetime64[ns, UTC]")
    cutoff = _utc(rows.decision_time_utc.reset_index(drop=True), "reaction cutoff")
    if ((data.coverage_status.eq("known") & clocks.isna()).any() or clocks.gt(cutoff).any()):
        raise DataReadinessError("reaction coverage is unavailable at decision")
    data["available_at_utc"] = clocks
    return data


def _select(events: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    selected: list[dict[str, object]] = []
    groups = {security: group for security, group in events.groupby("security_id", sort=False)}
    tie = ["source_family", "event_id", "event_version_sha256"]
    for decision in rows.itertuples(index=False):
        candidates = groups.get(decision.security_id)
        if candidates is None:
            continue
        cutoff = pd.Timestamp(decision.decision_time_utc)
        candidates = candidates.loc[candidates.event_available_at_utc.le(cutoff)]
        # A newer known rejected/unclassified revision suppresses its older text.
        candidates = candidates.sort_values(["event_available_at_utc", *tie], ascending=[False, True, True, True])
        candidates = candidates.drop_duplicates(["source_family", "event_id"], keep="first")
        candidates = candidates.loc[candidates.qualification_status.eq("qualified")
            & candidates.identity_available_at_utc.le(cutoff)]
        # Duplicate groups are explicit caller-proven identities, never fuzzy matches.
        candidates = candidates.sort_values(["event_available_at_utc", *tie]).drop_duplicates("duplicate_group_id")
        candidates = candidates.loc[candidates.event_available_at_utc.gt(cutoff - pd.Timedelta(LOOKBACK))]
        if candidates.empty:
            continue
        winner = candidates.sort_values(["event_available_at_utc", *tie], ascending=[False, True, True, True]).iloc[0]
        if winner.ticker != decision.ticker:
            raise DataReadinessError("selected event differs from original decision ticker identity")
        record = winner.to_dict()
        record.update(decision_id=decision.decision_id, decision_time_utc=cutoff)
        selected.append(record)
    return pd.DataFrame(selected, columns=[*QUALIFIED_EVENT_COLUMNS, "decision_id", "decision_time_utc"])


def build_issuer_reaction_profile(
    *, baseline: ResearchFeaturePartition, qualified_events: pd.DataFrame, coverage: pd.DataFrame,
    stock_bars: pd.DataFrame, spy_bars: pd.DataFrame, history_sessions: Sequence[date],
    sources: IssuerReactionSources,
    purpose: Literal["historical_research", "live_construction"] = "historical_research",
) -> ResearchFeaturePartition:
    """Append two measurements; callers independently establish source qualification.

    ``qualified_events`` includes all reviewed version dispositions, not only the
    qualified subset. Coverage is diagnostic and never filters the original cohort.
    """
    if purpose not in ("historical_research", "live_construction"):
        raise DataReadinessError("unsupported issuer reaction profile purpose")
    if purpose == "live_construction" and any(value != "observed" for value in (
        sources.bars.availability_semantics, sources.event_availability_semantics, sources.identity_availability_semantics,
    )):
        raise DataReadinessError("live reaction profile cannot consume historical proxy availability")
    rows = _parent(baseline, sources)
    sessions = _sessions(history_sessions)
    if not rows.session_date_et.isin(sessions).all():
        raise DataReadinessError("reaction history must cover every decision session")
    events = _events(qualified_events, sources)
    if not events.security_id.isin(rows.security_id).all():
        raise DataReadinessError("reviewed events contain foreign security identities")
    cov = _coverage(coverage, rows)
    selected = _select(events, rows)
    additions = (*REACTION_COLUMNS, *(f"available_at_{name}" for name in REACTION_COLUMNS),
                 *(f"missing_reason_{name}" for name in REACTION_COLUMNS), *_INTERVALS,
                 *(f"selected_{name}" for name in _SELECTED), "reaction_coverage_status", "reaction_coverage_available_at_utc")
    if set(additions).intersection(rows.columns):
        raise DataReadinessError("reaction profile output columns already exist in parent")
    rows["reaction_coverage_status"] = cov.coverage_status.to_numpy()
    rows["reaction_coverage_available_at_utc"] = cov.available_at_utc.array
    reason = np.where(cov.coverage_status.eq("known"), "no_qualified_event_in_lookback", "unknown_source_coverage")
    for name in REACTION_COLUMNS:
        rows[name] = np.full(len(rows), np.nan, dtype=np.float32)
        rows[f"available_at_{name}"] = pd.array([pd.NaT] * len(rows), dtype="datetime64[ns, UTC]")
        rows[f"missing_reason_{name}"] = reason.copy()
    for name in (*_INTERVALS, *(f"selected_{field}" for field in _SELECTED)):
        rows[name] = (pd.array([pd.NaT] * len(rows), dtype="datetime64[ns, UTC]")
                      if name.endswith("_utc") else pd.array([None] * len(rows), dtype=object))
    if not selected.empty:
        securities = set(selected.security_id)
        measured = build_issuer_reactions(events=selected.loc[:, list(REACTION_EVENT_COLUMNS)],
            stock_bars=stock_bars.loc[stock_bars.security_id.isin(securities)], spy_bars=spy_bars,
            history_sessions=history_sessions, sources=sources, purpose=purpose)
        for name in _SELECTED:
            measured[f"selected_{name}"] = selected[name].array
        indexed = measured.set_index("decision_id").reindex(rows.decision_id)
        matched = rows.decision_id.isin(selected.decision_id).to_numpy()
        for name in additions:
            if name not in indexed:
                continue
            current = rows[name].copy()
            current.iloc[np.flatnonzero(matched)] = indexed[name].iloc[np.flatnonzero(matched)].array
            rows[name] = current.array
    rows["feature_profile"] = ISSUER_REACTION_PROFILE
    # Pandas string inference otherwise gives a mixed batch a different dtype
    # from concatenated all-missing and populated single-decision batches.
    text_columns = ("feature_profile", "reaction_coverage_status",
                    *(f"missing_reason_{name}" for name in REACTION_COLUMNS),
                    *(f"selected_{name}" for name in _SELECTED if not name.endswith("_utc")))
    for name in text_columns:
        rows[name] = pd.Series(rows[name], index=rows.index, dtype=pd.StringDtype(storage="python"))
    rows["reaction_session_date_et"] = pd.Series(
        [None if pd.isna(value) else value for value in rows.reaction_session_date_et], index=rows.index, dtype=object,
    )
    names = (*baseline.model_columns, *REACTION_COLUMNS)
    clocks = {**baseline.availability_columns, **{name: f"available_at_{name}" for name in REACTION_COLUMNS}}
    audit = {"feature_profile": ISSUER_REACTION_PROFILE, "profile_sha256": issuer_reaction_profile_sha256(baseline.model_columns, sources),
        "rows": len(rows), "selected_event_rows": len(selected), "model_feature_count": len(names),
        "population_preserved": True, "outcome_filtered_rows": 0, "baseline_model_inputs_unchanged": True,
        "construction_purpose": purpose, "source_admission": "required_from_publishing_caller",
        "training_eligible": False, "promotion_eligible": False, "serving_eligible": False,
        "sources": sources.model_dump(mode="json"), "baseline_audit": dict(baseline.audit)}
    return ResearchFeaturePartition(rows, names, clocks, audit)
