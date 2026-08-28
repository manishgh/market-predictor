"""Causal intraday activity selection and anti-leakage poison tests."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest

import market_predictor.intraday.datasets.selection as selection_module
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.datasets.history import (
    json_sha256,
    load_plan_json,
    write_plan_json,
)
from market_predictor.intraday.datasets.selection import (
    build_intraday_selection,
    select_intraday_activations,
)
from market_predictor.modeling.strategy_contract import (
    IntradayUniverseContract,
    load_strategy_contract,
)

LOOKBACK = 20
SLOTS = 78
CONTRACT_PATH = Path("configs/edge_rebuild_strategy_contract.toml")


def _universe(**overrides: object) -> IntradayUniverseContract:
    payload: dict[str, object] = {
        "scope": "sp500_point_in_time",
        "index_restricted": True,
        "minimum_average_volume_shares": 1_000_000,
        "average_volume_lookback_sessions": LOOKBACK,
        "minimum_price": 5.0,
        "maximum_price": 500.0,
        "minimum_bar_continuity": 0.95,
        "minimum_relative_volume": 2.0,
        "relative_volume_lookback_sessions": LOOKBACK,
        "relative_volume_excludes_current_session": True,
        "selection_timing": "cumulative_to_decision",
        "activity_timeframe": "5Min",
        "activity_numerator": "cumulative_observed_volume",
        "activity_baseline": "median_cumulative_same_slot_prior_sessions",
        "exact_slot_matching": True,
        "activity_resets_each_session": True,
        "imputation_allowed": False,
        "activation_delay_seconds": 60,
        "maximum_candidates_per_decision": 30,
        "exclude_exchange_traded_products": True,
    }
    payload.update(overrides)
    return IntradayUniverseContract.model_validate(payload)


def _session_rows(
    session: pd.Timestamp,
    *,
    ticker: str,
    volumes: Iterable[float],
    close: float = 25.0,
) -> pd.DataFrame:
    start = pd.Timestamp(session.date(), tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
    volumes_list = list(volumes)
    starts = pd.date_range(start, periods=len(volumes_list), freq="5min")
    ends = starts + pd.Timedelta(minutes=5)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "timeframe": "5m",
            "bar_start_utc": starts.tz_convert("UTC"),
            "bar_end_utc": ends.tz_convert("UTC"),
            "available_at_utc": (ends + pd.Timedelta(seconds=60)).tz_convert("UTC"),
            "close": close,
            "volume": volumes_list,
            "price_feed": "sip",
            "adjustment": "all",
        }
    )


def _history_and_current(
    *,
    ticker: str = "AAA",
    current_volumes: list[float] | None = None,
    current_close: float = 25.0,
) -> pd.DataFrame:
    sessions = pd.bdate_range("2024-01-02", periods=LOOKBACK + 1)
    rows = [
        _session_rows(
            session,
            ticker=ticker,
            volumes=[20_000.0] * SLOTS,
        )
        for session in sessions[:-1]
    ]
    rows.append(
        _session_rows(
            sessions[-1],
            ticker=ticker,
            volumes=current_volumes or [40_000.0] + [20_000.0] * (SLOTS - 1),
            close=current_close,
        )
    )
    return pd.concat(rows, ignore_index=True)


def test_activation_uses_same_slot_prior_session_median() -> None:
    activity, selected = select_intraday_activations(_history_and_current(), universe=_universe())

    row = selected.iloc[0]
    assert len(selected) == 1
    assert row["relative_volume_at_activation"] == pytest.approx(2.0)
    assert row["average_volume_prior_sessions"] == pytest.approx(1_560_000.0)
    assert row["median_volume_prior_sessions"] == pytest.approx(1_560_000.0)
    assert row["price_at_activation"] == pytest.approx(25.0)
    assert row["activation_rank"] == 1
    assert pd.Timestamp(row["activation_time_utc"]) == pd.Timestamp("2024-01-30 14:36:00+00:00")
    assert int(activity.iloc[-1]["prior_sessions_available"]) == LOOKBACK
    assert float(activity.iloc[-1]["average_bar_continuity_prior_sessions"]) == pytest.approx(1.0)


def test_future_same_session_volume_cannot_change_earlier_activation() -> None:
    """Poisoning bars after activation must not alter time, rank, or features."""

    ordinary = [40_000.0] + [20_000.0] * (SLOTS - 1)
    poisoned = ordinary.copy()
    poisoned[-20:] = [2_000_000_000.0] * 20
    _, original = select_intraday_activations(_history_and_current(current_volumes=ordinary), universe=_universe())
    _, changed = select_intraday_activations(_history_and_current(current_volumes=poisoned), universe=_universe())

    pd.testing.assert_frame_equal(original, changed)


def test_late_activation_is_not_compared_with_an_earlier_decision_group() -> None:
    """The cap applies per decision timestamp, never across future timestamps."""

    sessions = pd.bdate_range("2024-01-02", periods=LOOKBACK + 1)
    frames: list[pd.DataFrame] = []
    for index in range(31):
        ticker = f"T{index:02d}"
        for session in sessions[:-1]:
            frames.append(_session_rows(session, ticker=ticker, volumes=[20_000.0] * SLOTS))
        current = [10_000.0, 4_000_000.0] + [20_000.0] * (SLOTS - 2) if index == 30 else [40_000.0] + [20_000.0] * (SLOTS - 1)
        frames.append(_session_rows(sessions[-1], ticker=ticker, volumes=current))

    _, selected = select_intraday_activations(pd.concat(frames, ignore_index=True), universe=_universe())

    assert len(selected) == 31
    assert "T30" in set(selected["ticker"])
    late = selected[selected["ticker"] == "T30"].iloc[0]
    assert late["activation_rank"] == 1


def test_same_time_activations_use_only_contemporaneous_relative_volume() -> None:
    frames = []
    for ticker, first_volume in (("AAA", 40_000.0), ("BBB", 60_000.0)):
        frames.append(
            _history_and_current(
                ticker=ticker,
                current_volumes=[first_volume] + [20_000.0] * (SLOTS - 1),
            )
        )
    _, selected = select_intraday_activations(pd.concat(frames, ignore_index=True), universe=_universe())

    assert selected["ticker"].tolist() == ["BBB", "AAA"]
    assert selected["activation_rank"].tolist() == [1, 2]


def test_price_band_is_evaluated_at_activation() -> None:
    _, selected = select_intraday_activations(_history_and_current(current_close=3.0), universe=_universe())

    assert selected.empty


def test_missing_exact_historical_slot_is_not_imputed() -> None:
    bars = _history_and_current(current_volumes=[40_000.0] + [0.0] * (SLOTS - 1))
    prior_session = sorted(pd.to_datetime(bars["bar_start_utc"], utc=True).dt.tz_convert("America/New_York").dt.date.unique())[0]
    local = pd.to_datetime(bars["bar_start_utc"], utc=True).dt.tz_convert("America/New_York")
    bars = bars.loc[~((local.dt.date == prior_session) & (local.dt.hour == 9) & (local.dt.minute == 30))]

    _, selected = select_intraday_activations(bars, universe=_universe())

    assert selected.empty


def test_trailing_bar_continuity_gate_uses_only_prior_sessions() -> None:
    bars = _history_and_current()
    local = pd.to_datetime(bars["bar_start_utc"], utc=True).dt.tz_convert(
        "America/New_York"
    )
    sessions = sorted(local.dt.date.unique())
    historical = local.dt.date != sessions[-1]
    slot = ((local.dt.hour * 60 + local.dt.minute) - (9 * 60 + 30)) // 5
    bars = bars.loc[~(historical & slot.ge(70))].reset_index(drop=True)

    activity, selected = select_intraday_activations(
        bars,
        universe=_universe(),
        expected_bars_by_session={session: SLOTS for session in sessions},
    )

    assert selected.empty
    assert float(activity.iloc[-1]["average_bar_continuity_prior_sessions"]) == pytest.approx(70 / 78)


def test_early_close_full_session_counts_as_complete() -> None:
    calendar = xcals.get_calendar("XNYS")
    sessions = [
        pd.Timestamp(value).date()
        for value in calendar.sessions_in_range("2024-11-01", "2024-12-02")
    ][-(LOOKBACK + 1) :]
    expected = {
        session: (42 if session == date(2024, 11, 29) else SLOTS)
        for session in sessions
    }
    frames = [
        _session_rows(
            pd.Timestamp(session),
            ticker="AAA",
            volumes=[20_000.0] * expected[session],
        )
        for session in sessions[:-1]
    ]
    frames.append(
        _session_rows(
            pd.Timestamp(sessions[-1]),
            ticker="AAA",
            volumes=[40_000.0] + [20_000.0] * (SLOTS - 1),
        )
    )

    activity, selected = select_intraday_activations(
        pd.concat(frames, ignore_index=True),
        universe=_universe(),
        expected_bars_by_session=expected,
    )

    assert len(selected) == 1
    assert float(activity.iloc[-1]["average_bar_continuity_prior_sessions"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("column", "value"),
    (("timeframe", "1m"), ("price_feed", "iex"), ("adjustment", "raw")),
)
def test_input_identity_must_be_five_minute_sip_all(column: str, value: str) -> None:
    bars = _history_and_current()
    bars[column] = value

    with pytest.raises(DataReadinessError, match="5m SIP/all"):
        select_intraday_activations(bars, universe=_universe())


def test_availability_must_be_bar_end_plus_sixty_seconds() -> None:
    bars = _history_and_current()
    bars.loc[bars.index[-1], "available_at_utc"] = bars.loc[bars.index[-1], "bar_end_utc"]

    with pytest.raises(DataReadinessError, match="plus 60 seconds"):
        select_intraday_activations(bars, universe=_universe())


def test_duplicate_symbol_slot_is_rejected() -> None:
    bars = _history_and_current()
    duplicate = bars.iloc[[-1]].copy()

    with pytest.raises(DataReadinessError, match="duplicate"):
        select_intraday_activations(pd.concat([bars, duplicate], ignore_index=True), universe=_universe())


def test_baseline_does_not_bleed_across_symbols() -> None:
    quiet = _history_and_current(ticker="AAA")
    loud = _history_and_current(ticker="BBB")
    historical = loud.index < LOOKBACK * SLOTS
    loud.loc[historical, "volume"] = 200_000.0
    _, selected = select_intraday_activations(pd.concat([quiet, loud], ignore_index=True), universe=_universe())
    by_ticker = selected.set_index("ticker")

    assert by_ticker.loc["AAA", "average_volume_prior_sessions"] == pytest.approx(1_560_000.0)
    assert "BBB" not in by_ticker.index


def test_pre_addition_rows_are_excluded_and_membership_entry_starts_cold() -> None:
    bars = _history_and_current()
    sessions = _session_dates(bars)

    activity, selected = select_intraday_activations(
        bars,
        universe=_universe(),
        session_eligibility={"AAA": sessions[1:]},
    )

    assert sessions[0] not in set(activity["session_date_et"])
    assert selected.empty
    assert int(activity.iloc[-1]["prior_sessions_available"]) == LOOKBACK - 1


def test_post_removal_rows_are_excluded() -> None:
    bars = _history_and_current()
    sessions = _session_dates(bars)

    activity, selected = select_intraday_activations(
        bars,
        universe=_universe(),
        session_eligibility={"AAA": sessions[:-1]},
    )

    assert sessions[-1] not in set(activity["session_date_et"])
    assert selected.empty


def test_reentry_resets_history_and_requires_a_new_full_lookback() -> None:
    sessions = pd.bdate_range("2024-01-02", periods=LOOKBACK + 2)
    bars = pd.concat(
        [
            _session_rows(
                session,
                ticker="AAA",
                volumes=([40_000.0] + [20_000.0] * (SLOTS - 1) if index == LOOKBACK + 1 else [20_000.0] * SLOTS),
            )
            for index, session in enumerate(sessions)
        ],
        ignore_index=True,
    )
    eligible = [session.date() for session in sessions[:LOOKBACK]] + [sessions[-1].date()]

    activity, selected = select_intraday_activations(
        bars,
        universe=_universe(),
        session_eligibility={"AAA": eligible},
    )

    reentry = activity.loc[activity["session_date_et"].eq(sessions[-1].date())].iloc[0]
    assert int(reentry["prior_sessions_available"]) == 0
    assert not bool(reentry["exact_slot_baseline_ready"])
    assert selected.empty


def test_future_membership_transition_cannot_change_prior_activations() -> None:
    bars = _history_and_current()
    sessions = _session_dates(bars)
    future = pd.Timestamp(sessions[-1]) + pd.Timedelta(days=30)
    _, original = select_intraday_activations(
        bars,
        universe=_universe(),
        session_eligibility={"AAA": sessions},
    )
    _, poisoned = select_intraday_activations(
        bars,
        universe=_universe(),
        session_eligibility={"AAA": [*sessions, future.date()]},
    )

    pd.testing.assert_frame_equal(original, poisoned)


def test_production_screen_streams_verified_canonical_symbol_files(
    tmp_path: Path,
) -> None:
    corpus, first_session, last_session = _canonical_store(tmp_path)
    membership = _membership_authority(
        tmp_path,
        intervals=[("AAA", first_session, None)],
    )
    contract = load_strategy_contract(CONTRACT_PATH)

    result = build_intraday_selection(
        canonical_dir=corpus,
        contract=contract,
        first_session=first_session,
        last_session=last_session,
        membership_authority_dir=membership,
    )

    assert result.audit["symbols_read"] == 1
    assert result.audit["canonical_manifest_sha256"] == file_sha256(corpus / "_manifest.json")
    assert result.audit["membership_authority_sha256"] == file_sha256(membership / "_authority.json")
    assert result.audit["membership_universe_snapshot_id"] == "sp500-test"
    assert result.audit["membership_cold_start_policy"] == ("reset_on_each_membership_entry")
    assert result.selection["ticker"].tolist() == ["AAA"]


def test_production_screen_rejects_modified_canonical_file(tmp_path: Path) -> None:
    corpus, first_session, last_session = _canonical_store(tmp_path)
    membership = _membership_authority(
        tmp_path,
        intervals=[("AAA", first_session, None)],
    )
    path = corpus / "regular" / "5m" / "AAA.parquet"
    changed = pd.read_parquet(path)
    changed.loc[0, "volume"] = 99_999_999
    changed.to_parquet(path, index=False)

    with pytest.raises(DataReadinessError, match="failed its hash"):
        build_intraday_selection(
            canonical_dir=corpus,
            contract=load_strategy_contract(CONTRACT_PATH),
            first_session=first_session,
            last_session=last_session,
            membership_authority_dir=membership,
        )


def test_production_membership_reentry_resets_history_at_interval_boundary(
    tmp_path: Path,
) -> None:
    corpus, first_session, last_session = _canonical_store(tmp_path)
    membership = _membership_authority(
        tmp_path,
        intervals=[
            ("AAA", first_session, last_session),
            ("AAA", last_session, None),
        ],
    )

    result = build_intraday_selection(
        canonical_dir=corpus,
        contract=load_strategy_contract(CONTRACT_PATH),
        first_session=first_session,
        last_session=last_session,
        membership_authority_dir=membership,
    )

    reentry = result.liquidity.loc[result.liquidity["session_date_et"].eq(last_session)].iloc[0]
    assert int(reentry["prior_sessions_available"]) == 0
    assert result.selection.empty


@pytest.mark.parametrize("poison", ["authority", "table"])
def test_production_screen_rejects_tampered_membership_authority(
    tmp_path: Path,
    poison: str,
) -> None:
    corpus, first_session, last_session = _canonical_store(tmp_path)
    membership = _membership_authority(
        tmp_path,
        intervals=[("AAA", first_session, None)],
    )
    if poison == "authority":
        authority = load_plan_json(membership / "_authority.json")
        authority["universe_sha256"] = "0" * 64
        write_plan_json(membership / "_authority.json", authority)
    else:
        table = membership / "memberships.parquet"
        changed = pd.read_parquet(table)
        changed.loc[0, "ticker"] = "POISON"
        changed.to_parquet(table, index=False)

    with pytest.raises(DataReadinessError, match="membership|artifact|integrity"):
        build_intraday_selection(
            canonical_dir=corpus,
            contract=load_strategy_contract(CONTRACT_PATH),
            first_session=first_session,
            last_session=last_session,
            membership_authority_dir=membership,
        )


def _canonical_store(tmp_path: Path) -> tuple[Path, date, date]:
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range("2024-01-02", "2024-02-15")[: LOOKBACK + 1]
    frames = []
    for index, session in enumerate(sessions):
        volumes = [40_000.0] + [20_000.0] * (SLOTS - 1) if index == LOOKBACK else [20_000.0] * SLOTS
        frames.append(_session_rows(pd.Timestamp(session), ticker="AAA", volumes=volumes))
    root = tmp_path / "canonical"
    path = root / "regular" / "5m" / "AAA.parquet"
    path.parent.mkdir(parents=True)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    manifest = {
        "schema": "edge_rebuild.intraday_materialization.v1",
        "files": [
            {
                "path": "regular/5m/AAA.parquet",
                "rows": sum(len(frame) for frame in frames),
                "sha256": file_sha256(path),
                "store": "regular",
                "ticker": "AAA",
            }
        ],
    }
    write_plan_json(root / "_manifest.json", manifest)
    write_plan_json(
        root / "_authority.json",
        {
            "schema": "edge_rebuild.intraday_materialization_authority.v1",
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(root / "_manifest.json"),
        },
    )
    return root, sessions[0].date(), sessions[-1].date()


def _session_dates(bars: pd.DataFrame) -> list[date]:
    return sorted(pd.to_datetime(bars["bar_start_utc"], utc=True).dt.tz_convert("America/New_York").dt.date.unique())


def _membership_authority(
    tmp_path: Path,
    *,
    intervals: list[tuple[str, date, date | None]],
) -> Path:
    root = tmp_path / "memberships"
    root.mkdir()
    parent_lineage = {
        "raw_authority_sha256": "1" * 64,
        "raw_manifest_sha256": "2" * 64,
        "event_authority_sha256": "3" * 64,
        "event_set_sha256": "4" * 64,
        "transition_authority_sha256": "5" * 64,
        "transition_set_sha256": "6" * 64,
        "anchor_file_sha256": "7" * 64,
        "anchor_semantic_sha256": "8" * 64,
    }
    request_payload: dict[str, Any] = {
        "schema": "edge_rebuild.sp500_membership_request.v1",
        "reconstruction_schema": "edge_rebuild.sp500_membership_reconstruction.v1",
        "start_date": "2018-05-29",
        "cutoff_date": "2026-07-08",
        "maximum_security_exclusion_fraction": 0.05,
        "security_exclusions_sha256": None,
        "parent_lineage": parent_lineage,
    }
    request_sha256 = json_sha256(request_payload)
    write_plan_json(
        root / "_request.json",
        {**request_payload, "request_sha256": request_sha256},
    )

    frame = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "security_id": f"security:{ticker}",
                "effective_from_utc": _membership_moment(start),
                "effective_to_utc": (pd.NaT if end is None else _membership_moment(end)),
                "available_at_utc": _membership_moment(start),
                "sector": "Information Technology",
                "industry": "Software",
                "market_cap_bucket": "large_cap_sp500",
                "liquidity_bucket": "sp500_constituent",
                "primary_benchmark": "XLK",
                "universe_snapshot_id": "sp500-test",
                "source": "spglobal_official_point_in_time",
                "availability_policy": "provider_publication_proxy",
                "schema_version": "market_data.v1",
            }
            for ticker, start, end in intervals
        ]
    )
    membership_path = root / "memberships.parquet"
    frame.to_parquet(membership_path, index=False)
    sidecar = {
        "schema": "market_data.artifact_manifest.v1",
        "canonical_schema_version": "market_data.v1",
        "artifact_type": "memberships",
        "artifact_path": str(membership_path),
        "artifact_sha256": file_sha256(membership_path),
        "created_at_utc": "2026-08-01T00:00:00+00:00",
        "rows": len(frame),
        "columns": list(frame.columns),
        "first_available_at_utc": frame["available_at_utc"].min().isoformat(),
        "last_available_at_utc": frame["available_at_utc"].max().isoformat(),
        "inputs": {
            **parent_lineage,
            "reconstruction_schema": request_payload["reconstruction_schema"],
            "request_sha256": request_sha256,
        },
        "audit": [],
        "production_ready": False,
    }
    sidecar_path = root / "memberships.parquet.manifest.json"
    write_plan_json(sidecar_path, sidecar)
    exclusions_path = root / "security_exclusions.json"
    write_plan_json(exclusions_path, [])
    universe_sha256 = selection_module._membership_semantic_sha256(frame)
    artifact = {
        "path": membership_path.name,
        "bytes": membership_path.stat().st_size,
        "sha256": file_sha256(membership_path),
    }
    exclusion_artifact = {
        "path": exclusions_path.name,
        "bytes": exclusions_path.stat().st_size,
        "sha256": file_sha256(exclusions_path),
    }
    manifest = {
        "schema": "edge_rebuild.sp500_membership_manifest.v1",
        "status": "complete",
        "request_sha256": request_sha256,
        "parent_lineage": parent_lineage,
        "benchmark_session_exclusions": 0,
        "start_date": "2018-05-29",
        "cutoff_date": "2026-07-08",
        "membership_artifact": artifact,
        "exclusion_artifact": exclusion_artifact,
        "membership_manifest_sha256": file_sha256(sidecar_path),
        "membership_intervals": len(frame),
        "security_count": frame["security_id"].nunique(),
        "ticker_count": frame["ticker"].nunique(),
        "universe_sha256": universe_sha256,
        "universe_snapshot_id": "sp500-test",
    }
    write_plan_json(root / "_manifest.json", manifest)
    write_plan_json(
        root / "_authority.json",
        {
            "schema": "edge_rebuild.sp500_membership_authority.v1",
            "state": "membership_complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(root / "_manifest.json"),
            "request_sha256": request_sha256,
            "parent_lineage": parent_lineage,
            "membership_intervals": len(frame),
            "security_count": frame["security_id"].nunique(),
            "universe_sha256": universe_sha256,
        },
    )
    return root


def _membership_moment(value: date) -> pd.Timestamp:
    return pd.Timestamp(value, tz="America/New_York").tz_convert("UTC")
