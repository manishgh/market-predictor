"""Causal event-time selection for the intraday research universe.

The screen observes canonical regular-session five-minute bars in timestamp
order.  A stock activates on the first completed bar where its cumulative
observed volume is at least twice the median cumulative volume at the same
five-minute slot over the prior twenty sessions.  Current-session bars never
enter either historical denominator.
"""
from __future__ import annotations



from collections import deque
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.canonical.audits import audit_universe_memberships
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
)
from market_predictor.intraday.datasets.history import (
    json_sha256,
    load_plan_json,
    write_plan_json,
)
from market_predictor.edge_rebuild.strategy_contract import (
    IntradayUniverseContract,
    StrategyContract,
)
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.core.errors import DataReadinessError

INTRADAY_SELECTION_SCHEMA = "edge_rebuild.intraday_universe_selection.v3"
INTRADAY_SELECTION_AUTHORITY_SCHEMA = "edge_rebuild.intraday_universe_selection_authority.v3"
SP500_MEMBERSHIP_REQUEST_SCHEMA = "edge_rebuild.sp500_membership_request.v1"
SP500_MEMBERSHIP_MANIFEST_SCHEMA = "edge_rebuild.sp500_membership_manifest.v1"
SP500_MEMBERSHIP_AUTHORITY_SCHEMA = "edge_rebuild.sp500_membership_authority.v1"
DEFAULT_SP500_MEMBERSHIP_AUTHORITY_DIR = Path("data/canonical/index_membership/sp500_memberships_20180529_20260708_v1")
EXCHANGE_TIMEZONE = ZoneInfo("America/New_York")
ACTIVITY_AUDIT_COLUMNS = (
    "ticker",
    "session_date_et",
    "observed_5m_bars",
    "prior_sessions_available",
    "average_volume_prior_sessions",
    "median_volume_prior_sessions",
    "average_bar_continuity_prior_sessions",
    "exact_slot_baseline_ready",
)
SELECTION_COLUMNS = (
    "ticker",
    "session_date_et",
    "activation_time_utc",
    "activation_rank",
    "relative_volume_at_activation",
    "average_volume_prior_sessions",
    "median_volume_prior_sessions",
    "price_at_activation",
)
_REQUIRED_BAR_COLUMNS = frozenset(
    {
        "ticker",
        "timeframe",
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "close",
        "volume",
        "price_feed",
        "adjustment",
    }
)
_MEMORY_BUDGET_GIB = 4.0
_MEMORY_HEADROOM_GIB = 0.75
_MEMBERSHIP_PARENT_LINEAGE_KEYS = frozenset(
    {
        "raw_authority_sha256",
        "raw_manifest_sha256",
        "event_authority_sha256",
        "event_set_sha256",
        "transition_authority_sha256",
        "transition_set_sha256",
        "anchor_file_sha256",
        "anchor_semantic_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class IntradaySelectionResult:
    """Causal screen diagnostics and unique selected stock-sessions."""

    # The name remains neutral for callers constructing test artifacts.  The
    # table now contains event-time audit rows, not end-of-day liquidity rows.
    liquidity: pd.DataFrame
    selection: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _SessionProfile:
    session_date_et: date
    total_observed_volume: float
    cumulative_volume_by_slot: Mapping[int, float]
    bar_continuity: float


@dataclass(frozen=True, slots=True)
class _MembershipEligibility:
    sessions_by_ticker: Mapping[str, frozenset[date]]
    cold_start_sessions_by_ticker: Mapping[str, frozenset[date]]
    authority_directory: Path
    authority_sha256: str
    manifest_sha256: str
    membership_table_sha256: str
    universe_sha256: str
    universe_snapshot_id: str
    parent_lineage: Mapping[str, str]


def select_intraday_activations(
    bars: pd.DataFrame,
    *,
    universe: IntradayUniverseContract,
    session_eligibility: Mapping[str, Collection[date]] | None = None,
    expected_bars_by_session: Mapping[date, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select activations from canonical five-minute rows in causal order."""

    if bars.empty:
        return _empty_activity(), _empty_selection()
    normalized = _validate_canonical_five_minute_rows(bars)
    expected = (
        _normalize_expected_bars(expected_bars_by_session)
        if expected_bars_by_session is not None
        else {
            session_date: int(
                frame.groupby("ticker", observed=True)["slot"].nunique().max()
            )
            for session_date, frame in normalized.groupby(
                "session_date_et", sort=True
            )
        }
    )
    sessions = ((session_date, frame.reset_index(drop=True)) for session_date, frame in normalized.groupby("session_date_et", sort=True))
    return _screen_sessions(
        sessions,
        universe=universe,
        session_eligibility=_normalize_session_eligibility(session_eligibility),
        expected_bars_by_session=expected,
    )


def _normalize_expected_bars(value: Mapping[date, int]) -> dict[date, int]:
    normalized: dict[date, int] = {}
    for session_date, raw_count in value.items():
        count = int(raw_count)
        if not isinstance(session_date, date) or count < 1:
            raise DataReadinessError("expected five-minute session bars are invalid")
        normalized[session_date] = count
    return normalized


def _normalize_session_eligibility(
    value: Mapping[str, Collection[date]] | None,
) -> dict[str, frozenset[date]] | None:
    if value is None:
        return None
    normalized: dict[str, frozenset[date]] = {}
    for raw_ticker, raw_sessions in value.items():
        ticker = str(raw_ticker).upper().strip()
        if not ticker or any(not isinstance(item, date) for item in raw_sessions):
            raise DataReadinessError("session eligibility contains invalid identity")
        normalized[ticker] = frozenset(raw_sessions)
    return normalized


def _screen_sessions(
    sessions: Iterable[tuple[date, pd.DataFrame]],
    *,
    universe: IntradayUniverseContract,
    session_eligibility: Mapping[str, frozenset[date]] | None = None,
    cold_start_sessions: Mapping[str, frozenset[date]] | None = None,
    expected_bars_by_session: Mapping[date, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookback = universe.relative_volume_lookback_sessions
    history: dict[str, deque[_SessionProfile]] = {}
    previous_market_sessions: deque[date] = deque(maxlen=lookback)
    activity_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for session_date, session_rows in sessions:
        expected_bar_count = expected_bars_by_session.get(session_date)
        if expected_bar_count is None or expected_bar_count < 1:
            raise DataReadinessError(
                f"missing expected five-minute bar count for {session_date}"
            )
        if session_eligibility is not None:
            for ticker in tuple(history):
                if session_date not in session_eligibility.get(ticker, frozenset()) or session_date in (cold_start_sessions or {}).get(
                    ticker, frozenset()
                ):
                    del history[ticker]
        if session_rows.empty:
            previous_market_sessions.append(session_date)
            continue
        if not session_rows["session_date_et"].eq(session_date).all():
            raise DataReadinessError("five-minute session partition contains another date")
        if bool(session_rows.duplicated(["ticker", "slot"]).any()):
            raise DataReadinessError("canonical five-minute input contains duplicate symbol-slot rows")

        prior_dates = tuple(previous_market_sessions)
        activations: list[dict[str, object]] = []
        current_profiles: dict[str, _SessionProfile] = {}
        for ticker, ticker_rows in session_rows.groupby("ticker", sort=True):
            normalized_ticker = str(ticker)
            if session_eligibility is not None and session_date not in session_eligibility.get(normalized_ticker, frozenset()):
                continue
            ordered = ticker_rows.sort_values("slot", kind="stable").reset_index(drop=True)
            profile = _session_profile(
                session_date,
                ordered,
                expected_bar_count=expected_bar_count,
            )
            current_profiles[normalized_ticker] = profile
            prior = tuple(history.get(normalized_ticker, ()))
            baseline_ready = (
                len(prior_dates) == lookback and len(prior) == lookback and tuple(item.session_date_et for item in prior) == prior_dates
            )
            average_volume = float(np.mean([item.total_observed_volume for item in prior])) if baseline_ready else np.nan
            median_volume = float(np.median([item.total_observed_volume for item in prior])) if baseline_ready else np.nan
            average_continuity = (
                float(np.mean([item.bar_continuity for item in prior]))
                if baseline_ready
                else np.nan
            )
            activity_rows.append(
                {
                    "ticker": str(ticker),
                    "session_date_et": session_date,
                    "observed_5m_bars": int(len(ordered)),
                    "prior_sessions_available": int(len(prior)),
                    "average_volume_prior_sessions": average_volume,
                    "median_volume_prior_sessions": median_volume,
                    "average_bar_continuity_prior_sessions": average_continuity,
                    "exact_slot_baseline_ready": baseline_ready,
                }
            )
            if (
                not baseline_ready
                or average_volume < universe.minimum_average_volume_shares
                or average_continuity < universe.minimum_bar_continuity
            ):
                continue
            activation = _first_activation(
                ordered,
                prior=prior,
                average_volume=average_volume,
                median_volume=median_volume,
                universe=universe,
            )
            if activation is not None:
                activations.append(activation)

        _append_capped_decision_activations(
            selected_rows,
            activations,
            session_date=session_date,
            maximum_candidates_per_decision=(universe.maximum_candidates_per_decision),
        )

        for ticker, profile in current_profiles.items():
            values = history.setdefault(ticker, deque(maxlen=lookback))
            values.append(profile)
        previous_market_sessions.append(session_date)

    activity = pd.DataFrame(activity_rows, columns=list(ACTIVITY_AUDIT_COLUMNS))
    selection = pd.DataFrame(selected_rows, columns=list(SELECTION_COLUMNS))
    return activity, selection


def _append_capped_decision_activations(
    destination: list[dict[str, object]],
    activations: list[dict[str, object]],
    *,
    session_date: date,
    maximum_candidates_per_decision: int,
) -> None:
    """Cap only activations first available at the same decision timestamp."""

    if not activations:
        return
    ordered = sorted(
        activations,
        key=lambda row: (
            pd.Timestamp(row["activation_time_utc"]),
            -float(cast(float, row["relative_volume_at_activation"])),
            str(row["ticker"]),
        ),
    )
    frame = pd.DataFrame(ordered)
    for _, decision_rows in frame.groupby("activation_time_utc", sort=True):
        for rank, activation in enumerate(
            decision_rows.head(maximum_candidates_per_decision).to_dict(orient="records"),
            start=1,
        ):
            destination.append(
                {
                    **activation,
                    "session_date_et": session_date,
                    "activation_rank": rank,
                }
            )


def _session_profile(
    session_date: date,
    rows: pd.DataFrame,
    *,
    expected_bar_count: int,
) -> _SessionProfile:
    observed_bar_count = int(rows["slot"].nunique())
    if observed_bar_count > expected_bar_count:
        raise DataReadinessError(
            f"observed five-minute bars exceed the XNYS session for {session_date}"
        )
    cumulative = pd.to_numeric(rows["volume"], errors="raise").cumsum()
    return _SessionProfile(
        session_date_et=session_date,
        total_observed_volume=float(cumulative.iloc[-1]),
        cumulative_volume_by_slot={int(slot): float(value) for slot, value in zip(rows["slot"], cumulative, strict=True)},
        bar_continuity=observed_bar_count / expected_bar_count,
    )


def _first_activation(
    rows: pd.DataFrame,
    *,
    prior: tuple[_SessionProfile, ...],
    average_volume: float,
    median_volume: float,
    universe: IntradayUniverseContract,
) -> dict[str, object] | None:
    cumulative_observed = 0.0
    for row in rows.itertuples(index=False):
        cumulative_observed += float(cast(float, row.volume))
        historical = [profile.cumulative_volume_by_slot.get(int(row.slot)) for profile in prior]
        # Exact-slot matching means one absent observation invalidates this
        # decision slot.  Filling or carrying a value forward is prohibited.
        if any(value is None for value in historical):
            continue
        denominator = float(np.median(np.asarray(historical, dtype="float64")))
        if denominator <= 0:
            continue
        relative_volume = cumulative_observed / denominator
        price = float(row.close)
        if relative_volume >= universe.minimum_relative_volume and universe.minimum_price <= price <= universe.maximum_price:
            return {
                "ticker": str(row.ticker),
                "activation_time_utc": pd.Timestamp(row.bar_end_utc) + pd.Timedelta(seconds=universe.activation_delay_seconds),
                "relative_volume_at_activation": relative_volume,
                "average_volume_prior_sessions": average_volume,
                "median_volume_prior_sessions": median_volume,
                "price_at_activation": price,
            }
    return None


def _validate_canonical_five_minute_rows(bars: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_REQUIRED_BAR_COLUMNS.difference(bars.columns))
    if missing:
        raise DataReadinessError(f"five-minute bars are missing columns: {missing}")
    frame = bars.loc[:, sorted(_REQUIRED_BAR_COLUMNS)].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["bar_start_utc"] = pd.to_datetime(frame["bar_start_utc"], utc=True)
    frame["bar_end_utc"] = pd.to_datetime(frame["bar_end_utc"], utc=True)
    frame["available_at_utc"] = pd.to_datetime(frame["available_at_utc"], utc=True)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")

    normalized_timeframe = frame["timeframe"].astype(str).str.lower().str.strip()
    normalized_feed = frame["price_feed"].astype(str).str.lower().str.strip()
    normalized_adjustment = frame["adjustment"].astype(str).str.lower().str.strip()
    if bool(normalized_timeframe.ne("5m").any()) or bool(normalized_feed.ne("sip").any()) or bool(normalized_adjustment.ne("all").any()):
        raise DataReadinessError("intraday activity selection requires canonical 5m SIP/all rows")
    if (
        bool(frame["ticker"].eq("").any())
        or bool(frame["close"].isna().any())
        or bool(frame["close"].le(0).any())
        or bool(frame["volume"].isna().any())
        or bool(frame["volume"].lt(0).any())
    ):
        raise DataReadinessError("five-minute bars contain invalid identity or values")

    starts = frame["bar_start_utc"]
    ends = frame["bar_end_utc"]
    expected_available = ends + pd.Timedelta(seconds=60)
    local = starts.dt.tz_convert(EXCHANGE_TIMEZONE)
    minutes_after_midnight = local.dt.hour * 60 + local.dt.minute
    regular_slot = (minutes_after_midnight - (9 * 60 + 30)) // 5
    aligned = (
        minutes_after_midnight.ge(9 * 60 + 30)
        & minutes_after_midnight.lt(16 * 60)
        & minutes_after_midnight.sub(9 * 60 + 30).mod(5).eq(0)
        & local.dt.second.eq(0)
        & local.dt.microsecond.eq(0)
    )
    if bool(frame.duplicated(["ticker", "bar_start_utc"]).any()):
        raise DataReadinessError("canonical five-minute input contains duplicate symbol-slot rows")
    if (
        bool((ends - starts).ne(pd.Timedelta(minutes=5)).any())
        or bool(frame["available_at_utc"].ne(expected_available).any())
        or not bool(aligned.all())
    ):
        raise DataReadinessError("intraday activity selection requires exact regular-session 5m slots available at bar end plus 60 seconds")
    frame["session_date_et"] = local.dt.date
    frame["slot"] = regular_slot.astype("int16")
    return frame.sort_values(["session_date_et", "bar_start_utc", "ticker"], kind="stable").reset_index(drop=True)


def build_intraday_selection(
    *,
    canonical_dir: Path,
    contract: StrategyContract,
    first_session: date,
    last_session: date,
    membership_authority_dir: Path = DEFAULT_SP500_MEMBERSHIP_AUTHORITY_DIR,
    exclude_tickers: frozenset[str] = frozenset(),
) -> IntradaySelectionResult:
    """Stream the verified per-symbol regular five-minute corpus causally."""

    if first_session > last_session:
        raise ValueError("first_session must not be after last_session")
    _manifest, records = _verify_canonical_regular_store(canonical_dir)
    calendar = xcals.get_calendar("XNYS")
    market_sessions = tuple(pd.Timestamp(value).date() for value in calendar.sessions_in_range(first_session, last_session))
    if not market_sessions:
        raise DataReadinessError("activity screen window contains no XNYS sessions")
    expected_bars_by_session = _xnys_expected_five_minute_bars(
        calendar,
        market_sessions,
    )
    membership = _load_sp500_membership_eligibility(
        membership_authority_dir,
        market_sessions=market_sessions,
        calendar=calendar,
    )
    observed_tickers: set[str] = set()
    rows_read = 0
    activity_parts: list[pd.DataFrame] = []
    activation_parts: list[pd.DataFrame] = []
    root = canonical_dir.resolve()
    empty = pd.DataFrame(columns=[*sorted(_REQUIRED_BAR_COLUMNS), "session_date_et", "slot"])
    for record in records:
        ticker = str(record["ticker"]).upper()
        if ticker in exclude_tickers:
            continue
        path = (root / str(record["path"])).resolve()
        frame = pd.read_parquet(path, columns=sorted(_REQUIRED_BAR_COLUMNS))
        rows_read += len(frame)
        normalized = _validate_canonical_five_minute_rows(frame)
        normalized = normalized.loc[pd.Series(normalized["session_date_et"]).between(first_session, last_session)].reset_index(drop=True)
        by_date = {value: part.reset_index(drop=True) for value, part in normalized.groupby("session_date_et", sort=True)}
        activity, activations = _screen_sessions(
            ((value, by_date.get(value, empty)) for value in market_sessions),
            universe=contract.intraday_universe,
            session_eligibility=membership.sessions_by_ticker,
            cold_start_sessions=membership.cold_start_sessions_by_ticker,
            expected_bars_by_session=expected_bars_by_session,
        )
        if not activity.empty:
            activity_parts.append(activity)
        if not activations.empty:
            activation_parts.append(activations.drop(columns="activation_rank"))
        observed_tickers.add(ticker)
        del frame, normalized, by_date, activity, activations
        assert_memory_budget(
            hard_budget_gib=_MEMORY_BUDGET_GIB,
            headroom_gib=_MEMORY_HEADROOM_GIB,
            stage=f"intraday event-time screen {ticker}",
        )

    activity = pd.concat(activity_parts, ignore_index=True) if activity_parts else _empty_activity()
    uncapped = (
        pd.concat(activation_parts, ignore_index=True)
        if activation_parts
        else pd.DataFrame(columns=[name for name in SELECTION_COLUMNS if name != "activation_rank"])
    )
    selected_rows: list[dict[str, object]] = []
    if not uncapped.empty:
        for session_date, group in uncapped.groupby("session_date_et", sort=True):
            _append_capped_decision_activations(
                selected_rows,
                group.drop(columns="session_date_et").to_dict(orient="records"),
                session_date=cast(date, session_date),
                maximum_candidates_per_decision=(contract.intraday_universe.maximum_candidates_per_decision),
            )
    selection = pd.DataFrame(selected_rows, columns=list(SELECTION_COLUMNS))
    release_process_memory()
    if bool(selection.duplicated(["ticker", "session_date_et"]).any()):
        raise DataReadinessError("event-time screen emitted duplicate stock-sessions")
    per_session = selection.groupby("session_date_et", sort=True).size()
    eligible = activity["average_volume_prior_sessions"].ge(
        contract.intraday_universe.minimum_average_volume_shares
    ) & activity["average_bar_continuity_prior_sessions"].ge(
        contract.intraday_universe.minimum_bar_continuity
    )
    audit: dict[str, Any] = {
        "schema": INTRADAY_SELECTION_SCHEMA,
        "strategy_id": contract.intraday.strategy_id,
        "strategy_contract_sha256": contract.sha256(),
        "canonical_dir": str(canonical_dir.resolve()),
        "canonical_manifest_sha256": file_sha256(canonical_dir / "_manifest.json"),
        "membership_authority_dir": str(membership.authority_directory),
        "membership_authority_sha256": membership.authority_sha256,
        "membership_manifest_sha256": membership.manifest_sha256,
        "membership_table_sha256": membership.membership_table_sha256,
        "membership_universe_sha256": membership.universe_sha256,
        "membership_universe_snapshot_id": membership.universe_snapshot_id,
        "membership_parent_lineage": dict(membership.parent_lineage),
        "membership_cold_start_policy": "reset_on_each_membership_entry",
        "first_session_et": first_session.isoformat(),
        "last_session_et": last_session.isoformat(),
        "excluded_tickers": sorted(exclude_tickers),
        "symbols_read": len(observed_tickers),
        "five_minute_rows_read": rows_read,
        "activity_rows": int(len(activity)),
        "sessions_in_window": int(activity["session_date_et"].nunique()),
        "layer_one": {
            "minimum_average_volume_shares": (contract.intraday_universe.minimum_average_volume_shares),
            "minimum_bar_continuity": contract.intraday_universe.minimum_bar_continuity,
            "average_volume_lookback_sessions": (contract.intraday_universe.average_volume_lookback_sessions),
            "stock_sessions": int(eligible.sum()),
        },
        "layer_two": {
            "selection_timing": contract.intraday_universe.selection_timing,
            "minimum_relative_volume": (contract.intraday_universe.minimum_relative_volume),
            "maximum_candidates_per_decision": (contract.intraday_universe.maximum_candidates_per_decision),
            "stock_sessions": int(len(selection)),
            "symbols": int(selection["ticker"].nunique()),
            "sessions_with_candidates": int(len(per_session)),
            "per_session_maximum": int(per_session.max()) if len(per_session) else 0,
        },
        "memory": memory_audit(
            hard_budget_gib=_MEMORY_BUDGET_GIB,
            headroom_gib=_MEMORY_HEADROOM_GIB,
        ).to_record(),
    }
    return IntradaySelectionResult(
        liquidity=activity,
        selection=selection,
        audit=audit,
    )


def _xnys_expected_five_minute_bars(
    calendar: Any,
    market_sessions: Collection[date],
) -> dict[date, int]:
    expected: dict[date, int] = {}
    for session_date in market_sessions:
        session = pd.Timestamp(session_date)
        opened = pd.Timestamp(calendar.session_open(session))
        closed = pd.Timestamp(calendar.session_close(session))
        duration = closed - opened
        count = int(duration / pd.Timedelta(minutes=5))
        if count < 1 or duration != count * pd.Timedelta(minutes=5):
            raise DataReadinessError(
                f"XNYS session is not divisible into exact five-minute bars: {session_date}"
            )
        expected[session_date] = count
    return expected


def _verify_canonical_regular_store(
    directory: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_plan_json(directory / "_manifest.json")
    authority = load_plan_json(directory / "_authority.json")
    if (
        manifest.get("schema") != "edge_rebuild.intraday_materialization.v1"
        or authority.get("schema") != "edge_rebuild.intraday_materialization_authority.v1"
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(directory / "_manifest.json")
    ):
        raise DataReadinessError(f"intraday canonical store lacks matching authority: {directory}")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise DataReadinessError("intraday canonical manifest has no files")
    records: list[dict[str, Any]] = []
    root = directory.resolve()
    for raw in raw_files:
        if not isinstance(raw, Mapping) or raw.get("store") != "regular":
            continue
        record = {str(key): value for key, value in raw.items()}
        path = (root / str(record.get("path", ""))).resolve()
        if root not in path.parents or file_sha256(path) != record.get("sha256"):
            raise DataReadinessError(f"canonical regular file failed its hash: {path}")
        records.append(record)
    if not records:
        raise DataReadinessError("intraday canonical store has no regular files")
    return manifest, sorted(records, key=lambda item: str(item["ticker"]))


def _load_sp500_membership_eligibility(
    directory: Path,
    *,
    market_sessions: tuple[date, ...],
    calendar: Any,
) -> _MembershipEligibility:
    root = directory.resolve()
    request_path = root / "_request.json"
    manifest_path = root / "_manifest.json"
    authority_path = root / "_authority.json"
    request = load_plan_json(request_path)
    manifest = load_plan_json(manifest_path)
    authority = load_plan_json(authority_path)

    request_payload = {str(key): value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = json_sha256(request_payload)
    parent_lineage = request_payload.get("parent_lineage")
    if (
        request.get("schema") != SP500_MEMBERSHIP_REQUEST_SCHEMA
        or request.get("request_sha256") != request_sha256
        or not _valid_parent_lineage(parent_lineage)
    ):
        raise DataReadinessError("S&P membership request or parent lineage is invalid")

    if (
        authority.get("schema") != SP500_MEMBERSHIP_AUTHORITY_SCHEMA
        or authority.get("state") != "membership_complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != request_sha256
        or authority.get("parent_lineage") != parent_lineage
    ):
        raise DataReadinessError("S&P membership authority is invalid")
    if (
        manifest.get("schema") != SP500_MEMBERSHIP_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or manifest.get("parent_lineage") != parent_lineage
        or int(manifest.get("benchmark_session_exclusions", -1)) != 0
    ):
        raise DataReadinessError("S&P membership manifest lineage is invalid")

    start_date = _iso_date(manifest.get("start_date"), field="start_date")
    cutoff_date = _iso_date(manifest.get("cutoff_date"), field="cutoff_date")
    if (
        request_payload.get("start_date") != start_date.isoformat()
        or request_payload.get("cutoff_date") != cutoff_date.isoformat()
        or market_sessions[0] < start_date
        or market_sessions[-1] > cutoff_date
    ):
        raise DataReadinessError("S&P membership authority does not cover screen window")

    membership_record = _artifact_inventory_record(manifest.get("membership_artifact"), role="membership")
    exclusion_record = _artifact_inventory_record(manifest.get("exclusion_artifact"), role="exclusion")
    membership_path = _verified_inventory_artifact(root, membership_record)
    _verified_inventory_artifact(root, exclusion_record)
    sidecar_path = manifest_path_for(membership_path)
    if not sidecar_path.is_file() or manifest.get("membership_manifest_sha256") != file_sha256(sidecar_path):
        raise DataReadinessError("canonical S&P membership sidecar hash is invalid")

    memberships, sidecar = load_canonical_artifact(
        membership_path,
        expected_type="memberships",
        allow_research=True,
    )
    if (
        _record_int(manifest.get("membership_intervals")) != len(memberships)
        or _record_int(authority.get("membership_intervals")) != len(memberships)
        or _record_int(sidecar.get("rows")) != len(memberships)
        or _record_int(manifest.get("security_count")) != memberships["security_id"].nunique()
        or _record_int(authority.get("security_count")) != memberships["security_id"].nunique()
        or _record_int(manifest.get("ticker_count")) != memberships["ticker"].nunique()
    ):
        raise DataReadinessError("S&P membership row or identity counts are invalid")

    sidecar_inputs = sidecar.get("inputs")
    expected_inputs = {
        **cast(dict[str, str], parent_lineage),
        "reconstruction_schema": str(request_payload.get("reconstruction_schema", "")),
        "request_sha256": request_sha256,
    }
    if not isinstance(sidecar_inputs, Mapping) or any(sidecar_inputs.get(key) != value for key, value in expected_inputs.items()):
        raise DataReadinessError("canonical S&P membership lineage is invalid")

    universe_sha256 = _membership_semantic_sha256(memberships)
    snapshot_values = set(memberships["universe_snapshot_id"].astype(str))
    universe_snapshot_id = str(manifest.get("universe_snapshot_id", ""))
    if (
        not universe_snapshot_id
        or snapshot_values != {universe_snapshot_id}
        or manifest.get("universe_sha256") != universe_sha256
        or authority.get("universe_sha256") != universe_sha256
    ):
        raise DataReadinessError("S&P membership universe identity is invalid")

    checks = audit_universe_memberships(memberships, require_observed=False)
    if any(check.status != "pass" for check in checks):
        raise DataReadinessError("canonical S&P membership rows failed semantic audit")
    sources = set(memberships["source"].astype(str))
    if sources != {"spglobal_official_point_in_time"}:
        raise DataReadinessError("S&P membership source is not point-in-time evidence")

    normalized = memberships.copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper().str.strip()
    for column in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
        normalized[column] = pd.to_datetime(normalized[column], utc=True, errors="coerce")
    if bool(normalized["ticker"].eq("").any()) or bool(normalized[["effective_from_utc", "available_at_utc"]].isna().any().any()):
        raise DataReadinessError("S&P membership timing contains invalid values")

    sessions_by_ticker: dict[str, set[date]] = {}
    cold_start_sessions_by_ticker: dict[str, set[date]] = {}
    session_opens = {session: pd.Timestamp(calendar.session_open(pd.Timestamp(session))).tz_convert("UTC") for session in market_sessions}
    for row in normalized.itertuples(index=False):
        effective_from = pd.Timestamp(row.effective_from_utc)
        effective_to = None if pd.isna(row.effective_to_utc) else pd.Timestamp(row.effective_to_utc)
        available_at = pd.Timestamp(row.available_at_utc)
        ticker = str(row.ticker)
        eligible = sessions_by_ticker.setdefault(ticker, set())
        interval_sessions = [
            session
            for session, session_open in session_opens.items()
            if effective_from <= session_open and available_at <= session_open and (effective_to is None or session_open < effective_to)
        ]
        eligible.update(interval_sessions)
        if interval_sessions:
            cold_start_sessions_by_ticker.setdefault(ticker, set()).add(interval_sessions[0])

    return _MembershipEligibility(
        sessions_by_ticker={ticker: frozenset(sessions) for ticker, sessions in sessions_by_ticker.items()},
        cold_start_sessions_by_ticker={ticker: frozenset(sessions) for ticker, sessions in cold_start_sessions_by_ticker.items()},
        authority_directory=root,
        authority_sha256=file_sha256(authority_path),
        manifest_sha256=file_sha256(manifest_path),
        membership_table_sha256=file_sha256(membership_path),
        universe_sha256=universe_sha256,
        universe_snapshot_id=universe_snapshot_id,
        parent_lineage=cast(dict[str, str], parent_lineage),
    )


def _valid_parent_lineage(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != _MEMBERSHIP_PARENT_LINEAGE_KEYS:
        return False
    return all(
        isinstance(key, str)
        and bool(key)
        and isinstance(item, str)
        and len(item) == 64
        and all(character in "0123456789abcdef" for character in item)
        for key, item in value.items()
    )


def _iso_date(value: object, *, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise DataReadinessError(f"S&P membership {field} is invalid") from exc


def _record_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return -1
    try:
        return int(value)
    except ValueError:
        return -1


def _artifact_inventory_record(value: object, *, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DataReadinessError(f"S&P {role} artifact inventory is invalid")
    return {str(key): item for key, item in value.items()}


def _verified_inventory_artifact(root: Path, record: Mapping[str, Any]) -> Path:
    path = (root / str(record.get("path", ""))).resolve()
    if (
        root not in path.parents
        or not path.is_file()
        or record.get("sha256") != file_sha256(path)
        or int(record.get("bytes", -1)) != path.stat().st_size
    ):
        raise DataReadinessError("S&P membership artifact inventory is invalid")
    return path


def _membership_semantic_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["ticker", "effective_from_utc", "security_id"], kind="stable")
    records: list[dict[str, Any]] = []
    for raw in ordered.to_dict(orient="records"):
        record = {str(key): value for key, value in raw.items()}
        for field in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
            value = record[field]
            record[field] = None if pd.isna(value) else pd.Timestamp(value).tz_convert("UTC").isoformat()
        records.append(record)
    return json_sha256(records)


def publish_intraday_selection(
    result: IntradaySelectionResult,
    *,
    output_directory: Path,
) -> dict[str, Any]:
    """Write the event-time screen as a hash-bound research artifact."""

    if (output_directory / "_authority.json").exists():
        raise DataReadinessError(f"published intraday selection is immutable: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    request = {
        key: result.audit[key]
        for key in (
            "schema",
            "strategy_id",
            "strategy_contract_sha256",
            "canonical_dir",
            "membership_authority_dir",
            "membership_authority_sha256",
            "membership_manifest_sha256",
            "membership_table_sha256",
            "membership_universe_sha256",
            "membership_universe_snapshot_id",
            "membership_parent_lineage",
            "membership_cold_start_policy",
            "first_session_et",
            "last_session_et",
            "excluded_tickers",
        )
    }
    request_sha256 = json_sha256(request)
    write_plan_json(
        output_directory / "_request.json",
        {**request, "request_sha256": request_sha256},
    )
    tables = {
        "session_activity_audit.parquet": result.liquidity,
        "selected_stock_sessions.parquet": result.selection,
    }
    records: list[dict[str, Any]] = []
    for name, frame in tables.items():
        path = output_directory / name
        frame.to_parquet(path, index=False)
        records.append(
            {
                "path": name,
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        **result.audit,
        "request_sha256": request_sha256,
        "published_at_utc": datetime.now(UTC).isoformat(),
        "tables": records,
    }
    write_plan_json(output_directory / "_manifest.json", manifest)
    write_plan_json(
        output_directory / "_authority.json",
        {
            "schema": INTRADAY_SELECTION_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(output_directory / "_manifest.json"),
            "request_sha256": request_sha256,
        },
    )
    return manifest


def load_complete_intraday_selection(directory: Path) -> dict[str, Any]:
    """Verify a published screen against its authority before any read."""

    request = load_plan_json(directory / "_request.json")
    manifest = load_plan_json(directory / "_manifest.json")
    authority = load_plan_json(directory / "_authority.json")
    request_sha256 = str(request.get("request_sha256", ""))
    payload = {key: value for key, value in request.items() if key != "request_sha256"}
    if (
        json_sha256(payload) != request_sha256
        or manifest.get("schema") != INTRADAY_SELECTION_SCHEMA
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != INTRADAY_SELECTION_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(directory / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
    ):
        raise DataReadinessError(f"intraday selection lacks matching complete authority: {directory}")
    tables = manifest.get("tables")
    if not isinstance(tables, list) or not tables:
        raise DataReadinessError(f"intraday selection manifest has no tables: {directory}")
    for record in tables:
        path = directory / str(record["path"])
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise DataReadinessError(f"intraday selection table failed its hash: {path}")
        if len(pd.read_parquet(path, columns=["ticker"])) != int(record["rows"]):
            raise DataReadinessError(f"intraday selection table row count moved: {path}")
    return manifest


def _empty_activity() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ACTIVITY_AUDIT_COLUMNS))


def _empty_selection() -> pd.DataFrame:
    return pd.DataFrame(columns=list(SELECTION_COLUMNS))
