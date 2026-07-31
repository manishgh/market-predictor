"""Causal intraday activity selection and anti-leakage poison tests."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.intraday_history import write_plan_json
from market_predictor.edge_rebuild.intraday_selection import (
    build_intraday_selection,
    select_intraday_activations,
)
from market_predictor.edge_rebuild.strategy_contract import (
    IntradayUniverseContract,
    load_strategy_contract,
)
from market_predictor.v3.errors import DataReadinessError

LOOKBACK = 20
SLOTS = 78
CONTRACT_PATH = Path("configs/edge_rebuild_strategy_contract.toml")


def _universe(**overrides: object) -> IntradayUniverseContract:
    payload: dict[str, object] = {
        "scope": "broad_us_point_in_time",
        "index_restricted": False,
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
    start = pd.Timestamp(session.date(), tz="America/New_York") + pd.Timedelta(
        hours=9, minutes=30
    )
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
    activity, selected = select_intraday_activations(
        _history_and_current(), universe=_universe()
    )

    row = selected.iloc[0]
    assert len(selected) == 1
    assert row["relative_volume_at_activation"] == pytest.approx(2.0)
    assert row["average_volume_prior_sessions"] == pytest.approx(1_560_000.0)
    assert row["median_volume_prior_sessions"] == pytest.approx(1_560_000.0)
    assert row["price_at_activation"] == pytest.approx(25.0)
    assert row["activation_rank"] == 1
    assert pd.Timestamp(row["activation_time_utc"]) == pd.Timestamp(
        "2024-01-30 14:36:00+00:00"
    )
    assert int(activity.iloc[-1]["prior_sessions_available"]) == LOOKBACK


def test_future_same_session_volume_cannot_change_earlier_activation() -> None:
    """Poisoning bars after activation must not alter time, rank, or features."""

    ordinary = [40_000.0] + [20_000.0] * (SLOTS - 1)
    poisoned = ordinary.copy()
    poisoned[-20:] = [2_000_000_000.0] * 20
    _, original = select_intraday_activations(
        _history_and_current(current_volumes=ordinary), universe=_universe()
    )
    _, changed = select_intraday_activations(
        _history_and_current(current_volumes=poisoned), universe=_universe()
    )

    pd.testing.assert_frame_equal(original, changed)


def test_late_activation_is_not_compared_with_an_earlier_decision_group() -> None:
    """The cap applies per decision timestamp, never across future timestamps."""

    sessions = pd.bdate_range("2024-01-02", periods=LOOKBACK + 1)
    frames: list[pd.DataFrame] = []
    for index in range(31):
        ticker = f"T{index:02d}"
        for session in sessions[:-1]:
            frames.append(
                _session_rows(session, ticker=ticker, volumes=[20_000.0] * SLOTS)
            )
        current = (
            [10_000.0, 4_000_000.0] + [20_000.0] * (SLOTS - 2)
            if index == 30
            else [40_000.0] + [20_000.0] * (SLOTS - 1)
        )
        frames.append(_session_rows(sessions[-1], ticker=ticker, volumes=current))

    _, selected = select_intraday_activations(
        pd.concat(frames, ignore_index=True), universe=_universe()
    )

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
    _, selected = select_intraday_activations(
        pd.concat(frames, ignore_index=True), universe=_universe()
    )

    assert selected["ticker"].tolist() == ["BBB", "AAA"]
    assert selected["activation_rank"].tolist() == [1, 2]


def test_price_band_is_evaluated_at_activation() -> None:
    _, selected = select_intraday_activations(
        _history_and_current(current_close=3.0), universe=_universe()
    )

    assert selected.empty


def test_missing_exact_historical_slot_is_not_imputed() -> None:
    bars = _history_and_current(current_volumes=[40_000.0] + [0.0] * (SLOTS - 1))
    prior_session = sorted(
        pd.to_datetime(bars["bar_start_utc"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date.unique()
    )[0]
    local = pd.to_datetime(bars["bar_start_utc"], utc=True).dt.tz_convert(
        "America/New_York"
    )
    bars = bars.loc[~((local.dt.date == prior_session) & (local.dt.hour == 9) & (local.dt.minute == 30))]

    _, selected = select_intraday_activations(bars, universe=_universe())

    assert selected.empty


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
    bars.loc[bars.index[-1], "available_at_utc"] = bars.loc[
        bars.index[-1], "bar_end_utc"
    ]

    with pytest.raises(DataReadinessError, match="plus 60 seconds"):
        select_intraday_activations(bars, universe=_universe())


def test_duplicate_symbol_slot_is_rejected() -> None:
    bars = _history_and_current()
    duplicate = bars.iloc[[-1]].copy()

    with pytest.raises(DataReadinessError, match="duplicate"):
        select_intraday_activations(
            pd.concat([bars, duplicate], ignore_index=True), universe=_universe()
        )


def test_baseline_does_not_bleed_across_symbols() -> None:
    quiet = _history_and_current(ticker="AAA")
    loud = _history_and_current(ticker="BBB")
    historical = loud.index < LOOKBACK * SLOTS
    loud.loc[historical, "volume"] = 200_000.0
    _, selected = select_intraday_activations(
        pd.concat([quiet, loud], ignore_index=True), universe=_universe()
    )
    by_ticker = selected.set_index("ticker")

    assert by_ticker.loc["AAA", "average_volume_prior_sessions"] == pytest.approx(
        1_560_000.0
    )
    assert "BBB" not in by_ticker.index


def test_production_screen_streams_verified_canonical_symbol_files(
    tmp_path: Path,
) -> None:
    corpus, first_session, last_session = _canonical_store(tmp_path)
    contract = load_strategy_contract(CONTRACT_PATH)

    result = build_intraday_selection(
        canonical_dir=corpus,
        contract=contract,
        first_session=first_session,
        last_session=last_session,
    )

    assert result.audit["symbols_read"] == 1
    assert result.audit["canonical_manifest_sha256"] == file_sha256(
        corpus / "_manifest.json"
    )
    assert result.selection["ticker"].tolist() == ["AAA"]


def test_production_screen_rejects_modified_canonical_file(tmp_path: Path) -> None:
    corpus, first_session, last_session = _canonical_store(tmp_path)
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
        )


def _canonical_store(tmp_path: Path) -> tuple[Path, date, date]:
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range("2024-01-02", "2024-02-15")[: LOOKBACK + 1]
    frames = []
    for index, session in enumerate(sessions):
        volumes = (
            [40_000.0] + [20_000.0] * (SLOTS - 1)
            if index == LOOKBACK
            else [20_000.0] * SLOTS
        )
        frames.append(
            _session_rows(pd.Timestamp(session), ticker="AAA", volumes=volumes)
        )
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
