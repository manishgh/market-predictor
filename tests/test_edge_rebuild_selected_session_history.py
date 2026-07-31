from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from market_predictor.edge_rebuild.history_collection import (
    collect_intraday_history,
    load_complete_intraday_history_collection,
)
from market_predictor.edge_rebuild.history_contracts import (
    SELECTED_SESSION_PLAN_SCHEMA,
    load_collection_transport_config,
    load_selected_session_history_config,
)
from market_predictor.edge_rebuild.history_materialization import (
    selected_ticker_sessions,
    session_bounds_for,
)
from market_predictor.edge_rebuild.intraday_history import (
    load_complete_intraday_history_plan,
)
from market_predictor.edge_rebuild.intraday_selection import (
    IntradaySelectionResult,
    publish_intraday_selection,
)
from market_predictor.edge_rebuild.selected_session_history import (
    build_selected_session_history_plan,
    verify_selected_stock_sessions,
)
from market_predictor.v3.errors import DataReadinessError

POLICY = Path("configs/edge_rebuild_selected_session_history.toml")
# 2024-07-03 is a half day closing at 13:00 ET; 2024-07-05 is a full session.
EARLY_CLOSE = "2024-07-03"
FULL_SESSION = "2024-07-05"


def _publish_selection(
    directory: Path,
    *,
    rows: list[tuple[str, str]] | None = None,
) -> Path:
    records = rows or [
        (EARLY_CLOSE, "AAA"),
        (EARLY_CLOSE, "BBB"),
        (FULL_SESSION, "AAA"),
        (FULL_SESSION, "CCC"),
    ]
    selection = pd.DataFrame(
        {
            "ticker": [ticker for _, ticker in records],
            "session_date_et": [session for session, _ in records],
            "session_volume": [5_000_000] * len(records),
            "average_volume_prior_sessions": [1_500_000.0] * len(records),
            "relative_volume": [3.3] * len(records),
            "session_close": [42.5] * len(records),
            "baseline_sessions": [20.0] * len(records),
            "session_rank": list(range(1, len(records) + 1)),
        }
    )
    audit = {
        "schema": "edge_rebuild.intraday_universe_selection.v1",
        "strategy_id": "INTRADAY.VWAP_EXHAUSTION_REVERSAL.30M.V1",
        "strategy_contract_sha256": "0" * 64,
        "collection_dir": str(directory / "daily"),
        "first_session_et": EARLY_CLOSE,
        "last_session_et": FULL_SESSION,
        "excluded_tickers": [],
    }
    publish_intraday_selection(
        IntradaySelectionResult(
            liquidity=selection,
            selection=selection,
            audit=audit,
        ),
        output_directory=directory,
    )
    return directory


def test_plan_requests_one_unit_per_session_at_real_session_bounds(
    tmp_path: Path,
) -> None:
    selection = _publish_selection(tmp_path / "screen")
    plan_dir = tmp_path / "plan"

    manifest = build_selected_session_history_plan(
        selection_directory=selection,
        policy_path=POLICY,
        output_directory=plan_dir,
        config=load_selected_session_history_config(POLICY),
    )
    verified = load_complete_intraday_history_plan(plan_dir)
    units = pd.concat(
        [pd.read_parquet(p) for p in (plan_dir / "units" / "5Min").glob("*.parquet")],
        ignore_index=True,
    ).set_index("session_date_et")

    assert verified["schema"] == SELECTED_SESSION_PLAN_SCHEMA
    assert manifest["summary"]["acquisition_units"] == 2
    assert manifest["summary"]["planned_history_sessions"] == 2
    assert manifest["summary"]["benchmark_tickers"] == 0
    assert manifest["summary"]["early_close_sessions"] == 1
    # The half day holds 42 regular five-minute bars, the full session 78.
    early = units.loc[pd.Timestamp(EARLY_CLOSE).date()]
    full = units.loc[pd.Timestamp(FULL_SESSION).date()]
    assert int(early["expected_bars_per_symbol"]) == 42
    assert int(full["expected_bars_per_symbol"]) == 78
    assert early["requested_end_utc"] == pd.Timestamp("2024-07-03 17:00", tz="UTC")
    assert full["requested_end_utc"] == pd.Timestamp("2024-07-05 20:00", tz="UTC")
    # Only the names selected for that session, and no benchmark is added.
    assert json.loads(str(early["canonical_symbols_json"])) == ["AAA", "BBB"]
    assert json.loads(str(full["canonical_symbols_json"])) == ["AAA", "CCC"]


def test_generic_collector_accepts_the_registered_plan_schema(
    tmp_path: Path,
) -> None:
    plan_dir = tmp_path / "plan"
    build_selected_session_history_plan(
        selection_directory=_publish_selection(tmp_path / "screen"),
        policy_path=POLICY,
        output_directory=plan_dir,
        config=load_selected_session_history_config(POLICY),
    )
    output = tmp_path / "collection"

    result = collect_intraday_history(
        plan_directory=plan_dir,
        policy_path=POLICY,
        output_directory=output,
        config=load_collection_transport_config(POLICY),
        source_factory=_FakeAlpacaSource,
    )
    verified = load_complete_intraday_history_collection(output)

    assert result["status"] == "transport_complete"
    assert result["completed_units"] == 2
    assert verified["observed_symbols"] == ["AAA", "BBB", "CCC"]


def test_plan_refuses_a_date_that_is_not_an_exchange_session(
    tmp_path: Path,
) -> None:
    selection = _publish_selection(
        tmp_path / "screen",
        rows=[("2024-07-04", "AAA"), (FULL_SESSION, "AAA")],
    )

    with pytest.raises(DataReadinessError, match="not exchange sessions"):
        build_selected_session_history_plan(
            selection_directory=selection,
            policy_path=POLICY,
            output_directory=tmp_path / "plan",
            config=load_selected_session_history_config(POLICY),
        )


def test_plan_refuses_a_selection_edited_after_publication(
    tmp_path: Path,
) -> None:
    selection = _publish_selection(tmp_path / "screen")
    table = selection / "selected_stock_sessions.parquet"
    frame = pd.read_parquet(table)
    frame.loc[0, "ticker"] = "ZZZ"
    frame.to_parquet(table, index=False)

    with pytest.raises(DataReadinessError, match="failed its hash"):
        verify_selected_stock_sessions(selection)


def test_screen_makes_non_index_stock_sessions_eligible(tmp_path: Path) -> None:
    """A screened name is eligible on its selected session and on no other."""

    selection = _publish_selection(tmp_path / "screen")
    bounds = session_bounds_for(EARLY_CLOSE, FULL_SESSION)

    eligible, identity = selected_ticker_sessions(selection, bounds)

    assert eligible["AAA"] == {EARLY_CLOSE, FULL_SESSION}
    assert eligible["BBB"] == {EARLY_CLOSE}
    assert eligible["CCC"] == {FULL_SESSION}
    assert identity["stock_sessions_inside_window"] == 4
    assert identity["symbols"] == 3


class _FakeAlpacaSource:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(alpaca_stock_feed="sip")
        self.client = SimpleNamespace(timeout=30)

    def fetch_bars_page(
        self,
        symbols: tuple[str, ...],
        start: object,
        end: object,
        **kwargs: object,
    ) -> object:
        from market_predictor.sources.alpaca import AlpacaBarsPage

        del end
        assert kwargs["timeframe"] == "5Min"
        timestamps = [
            pd.Timestamp(str(start)),
            pd.Timestamp(str(start)) + pd.Timedelta(minutes=5),
        ]
        return AlpacaBarsPage(
            request_page_token=None,
            next_page_token=None,
            bars={
                symbol: tuple(
                    {
                        "t": timestamp.isoformat(),
                        "o": 100.0,
                        "h": 101.0,
                        "l": 99.0,
                        "c": 100.5,
                        "v": 1000,
                    }
                    for timestamp in timestamps
                )
                for symbol in symbols
            },
            response_headers={},
        )
