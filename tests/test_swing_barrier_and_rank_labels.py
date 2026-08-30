from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import market_predictor.modeling.label_outcomes as label_outcomes
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.labels.barrier_and_rank import (
    BARRIER_COLUMNS,
    RANK_COLUMNS,
    BarrierSpec,
    apply_cross_sectional_rank,
    apply_triple_barrier,
    forward_return_from_barrier,
)

SPEC = BarrierSpec(target_atr_multiple=3.0, stop_atr_multiple=1.5, horizon_sessions=5)


def _json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def test_barrier_contract_ownership_and_values_are_frozen() -> None:
    assert BarrierSpec.__module__ == (
        "market_predictor.swing.labels.barrier_and_rank"
    )
    assert pickle.loads(pickle.dumps(SPEC)) == SPEC
    assert _json_hash(asdict(SPEC)) == (
        "1a6bee0ccb2e5c0b8c54b6ff45b9d5e641d4e57da747092a605da39baceb960f"
    )
    assert _json_hash(BARRIER_COLUMNS) == (
        "2513343a01863d35bbca80c97b980666f20a2ef381c1e9ff00e619c957469cfa"
    )
    assert _json_hash(RANK_COLUMNS) == (
        "f89aa0051ff32e5a4b7d8efae2aa9e9ef4876e3a03a71182bc1c40b09fd59b56"
    )


def test_representative_barrier_and_managed_return_output_is_stable() -> None:
    rows = _flat(10)
    rows[1] = ("2024-01-02", 100.0, 104.0, 99.0, 103.0)
    rows[3] = ("2024-01-04", 100.0, 101.0, 98.0, 99.0)
    entries = pd.DataFrame(
        {
            "session": ["2024-01-01", "2024-01-03", "2024-01-05"],
            "atr": [1.0, 1.0, 1.0],
        }
    )
    output = apply_triple_barrier(_bars(rows), entries, spec=SPEC)
    output["managed_return"] = forward_return_from_barrier(
        output,
        pd.Series([100.0, 100.0, 100.0]),
    )
    payload = output.to_json(orient="split", double_precision=15)

    assert tuple(output.columns) == (
        "session",
        *BARRIER_COLUMNS,
        "managed_return",
    )
    assert tuple(map(str, output.dtypes)) == (
        "str",
        "int64",
        "str",
        "float64",
        "int64",
        "float64",
        "float64",
        "float64",
    )
    assert hashlib.sha256(payload.encode()).hexdigest() == (
        "47ea63e0186f0509f1bb2e3ebf9f697026968fc20fbbd3a01344d62e346faa3d"
    )


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["session", "open", "high", "low", "close"]
    )


def _flat(sessions: int, price: float = 100.0) -> list[tuple[str, float, float, float, float]]:
    return [
        (f"2024-01-{day:02d}", price, price, price, price)
        for day in range(1, sessions + 1)
    ]


def test_target_hit_resolves_to_the_target_price() -> None:
    rows = _flat(8)
    rows[3] = ("2024-01-04", 100.0, 106.0, 99.0, 105.0)  # +6 clears a +3 target
    entries = pd.DataFrame({"session": ["2024-01-01"], "atr": [1.0]})

    out = apply_triple_barrier(_bars(rows), entries, spec=SPEC)

    assert out.loc[0, "barrier_label"] == label_outcomes.TARGET_HIT
    assert out.loc[0, "exit_price"] == pytest.approx(103.0)
    assert out.loc[0, "exit_session"] == "2024-01-04"


def test_stop_hit_resolves_to_the_stop_price() -> None:
    rows = _flat(8)
    rows[2] = ("2024-01-03", 100.0, 101.0, 98.0, 98.5)  # -2 breaches a -1.5 stop
    entries = pd.DataFrame({"session": ["2024-01-01"], "atr": [1.0]})

    out = apply_triple_barrier(_bars(rows), entries, spec=SPEC)

    assert out.loc[0, "barrier_label"] == label_outcomes.STOP_HIT
    assert out.loc[0, "exit_price"] == pytest.approx(98.5)


def test_gap_through_stop_fills_at_worse_open() -> None:
    rows = _flat(8)
    rows[2] = ("2024-01-03", 96.0, 97.0, 95.0, 96.5)
    entries = pd.DataFrame({"session": ["2024-01-01"], "atr": [1.0]})

    out = apply_triple_barrier(_bars(rows), entries, spec=SPEC)

    assert out.loc[0, "barrier_label"] == label_outcomes.STOP_HIT
    assert out.loc[0, "stop_price"] == pytest.approx(98.5)
    assert out.loc[0, "exit_price"] == pytest.approx(96.0)


def test_gap_through_target_uses_conservative_resting_limit_fill() -> None:
    rows = _flat(8)
    rows[2] = ("2024-01-03", 106.0, 107.0, 105.0, 106.5)
    entries = pd.DataFrame({"session": ["2024-01-01"], "atr": [1.0]})

    out = apply_triple_barrier(_bars(rows), entries, spec=SPEC)

    assert out.loc[0, "barrier_label"] == label_outcomes.TARGET_HIT
    assert out.loc[0, "target_price"] == pytest.approx(103.0)
    assert out.loc[0, "exit_price"] == pytest.approx(103.0)


def test_a_bar_touching_both_barriers_resolves_to_the_stop() -> None:
    """The bar records that both prices traded, not which came first."""

    rows = _flat(8)
    rows[1] = ("2024-01-02", 100.0, 110.0, 90.0, 100.0)
    entries = pd.DataFrame({"session": ["2024-01-01"], "atr": [1.0]})

    out = apply_triple_barrier(_bars(rows), entries, spec=SPEC)

    assert out.loc[0, "barrier_label"] == label_outcomes.STOP_HIT
    assert out.loc[0, "holding_sessions"] == 1


def test_untouched_barriers_time_out_at_the_horizon_close() -> None:
    entries = pd.DataFrame({"session": ["2024-01-01"], "atr": [1.0]})

    out = apply_triple_barrier(_bars(_flat(8)), entries, spec=SPEC)

    assert out.loc[0, "barrier_label"] == label_outcomes.TIMEOUT
    assert out.loc[0, "holding_sessions"] == SPEC.horizon_sessions


def test_entry_never_uses_the_decision_bar() -> None:
    """Entry is the next session's open, so the decision bar cannot price it."""

    rows = _flat(8)
    rows[0] = ("2024-01-01", 100.0, 999.0, 1.0, 100.0)  # violent decision bar
    rows[1] = ("2024-01-02", 50.0, 50.0, 50.0, 50.0)
    entries = pd.DataFrame({"session": ["2024-01-01"], "atr": [1.0]})

    out = apply_triple_barrier(_bars(rows), entries, spec=SPEC)

    # Barriers are struck from the 50.0 open, not from anything on 2024-01-01.
    assert out.loc[0, "target_price"] == pytest.approx(53.0)
    assert out.loc[0, "stop_price"] == pytest.approx(48.5)


def test_a_horizon_running_past_the_data_is_unresolved_not_a_timeout() -> None:
    """Labelling an unknown outcome zero would invent an observation."""

    entries = pd.DataFrame({"session": ["2024-01-03"], "atr": [1.0]})

    out = apply_triple_barrier(_bars(_flat(5)), entries, spec=SPEC)

    assert pd.isna(out.loc[0, "barrier_label"])


def test_barrier_spec_rejects_incoherent_geometry() -> None:
    with pytest.raises(ValueError, match="must exceed stop"):
        BarrierSpec(target_atr_multiple=1.0, stop_atr_multiple=1.5, horizon_sessions=5)
    with pytest.raises(ValueError, match="conservative"):
        BarrierSpec(
            target_atr_multiple=3.0,
            stop_atr_multiple=1.5,
            horizon_sessions=5,
            same_bar_resolution="target_first",
        )


def _panel(returns: list[float], session: str = "2024-01-02") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": [session] * len(returns),
            "ticker": [f"T{i:03d}" for i in range(len(returns))],
            "sector": ["Tech"] * len(returns),
            "forward_return": returns,
        }
    )


def test_rank_splits_the_cross_section_into_thirds_by_quantile() -> None:
    panel = _panel([i / 100 for i in range(100)])

    out = apply_cross_sectional_rank(
        panel,
        top_quantile=0.2,
        bottom_quantile=0.2,
        within_sector=False,
        minimum_cross_section=50,
    )

    assert int((out["rank_label"] == label_outcomes.RANK_TOP).sum()) == 20
    assert int((out["rank_label"] == label_outcomes.RANK_BOTTOM).sum()) == 20
    assert int((out["rank_label"] == label_outcomes.RANK_MIDDLE).sum()) == 60


def test_rank_is_relative_to_the_session_not_to_a_fixed_threshold() -> None:
    """Every stock falling still yields a top fifth: the best of a bad day."""

    panel = _panel([-0.10 + i / 1000 for i in range(100)])

    out = apply_cross_sectional_rank(
        panel,
        top_quantile=0.2,
        bottom_quantile=0.2,
        within_sector=False,
        minimum_cross_section=50,
    )

    top = out[out["rank_label"] == label_outcomes.RANK_TOP]
    assert len(top) == 20
    assert (top["forward_return"] < 0).all()


def test_a_cross_section_below_the_minimum_is_left_unlabelled() -> None:
    """Quantiles of a handful of rows describe the handful, not the market."""

    out = apply_cross_sectional_rank(
        _panel([0.01, 0.02, 0.03]),
        top_quantile=0.2,
        bottom_quantile=0.2,
        within_sector=False,
        minimum_cross_section=50,
    )

    assert out["rank_label"].isna().all()


def test_sector_ranking_compares_a_stock_with_its_peers() -> None:
    """A whole sector rallying must not read as stock selection."""

    tech = pd.DataFrame(
        {
            "session": ["2024-01-02"] * 60,
            "sector": ["Tech"] * 60,
            "forward_return": [0.10 + i / 1000 for i in range(60)],
        }
    )
    utilities = pd.DataFrame(
        {
            "session": ["2024-01-02"] * 60,
            "sector": ["Utilities"] * 60,
            "forward_return": [-0.10 + i / 1000 for i in range(60)],
        }
    )
    out = apply_cross_sectional_rank(
        pd.concat([tech, utilities], ignore_index=True),
        top_quantile=0.2,
        bottom_quantile=0.2,
        within_sector=True,
        minimum_cross_section=50,
    )

    # Both sectors contribute winners and losers despite opposite sector moves.
    for sector in ("Tech", "Utilities"):
        part = out[out["sector"] == sector]
        assert int((part["rank_label"] == label_outcomes.RANK_TOP).sum()) == 12
        assert int((part["rank_label"] == label_outcomes.RANK_BOTTOM).sum()) == 12


def test_representative_sector_rank_output_is_stable() -> None:
    tech = pd.DataFrame(
        {
            "session": ["2024-01-02"] * 60,
            "sector": ["Tech"] * 60,
            "forward_return": [0.10 + i / 1000 for i in range(60)],
        }
    )
    utilities = pd.DataFrame(
        {
            "session": ["2024-01-02"] * 60,
            "sector": ["Utilities"] * 60,
            "forward_return": [-0.10 + i / 1000 for i in range(60)],
        }
    )
    output = apply_cross_sectional_rank(
        pd.concat([tech, utilities], ignore_index=True),
        top_quantile=0.2,
        bottom_quantile=0.2,
        within_sector=True,
        minimum_cross_section=50,
    )
    columns = ("session", "sector", *RANK_COLUMNS, "ranking_group_size")
    representative = output.loc[:, columns]
    payload = representative.to_json(orient="split", double_precision=15)

    assert tuple(map(str, representative.dtypes)) == (
        "str",
        "str",
        "int64",
        "float64",
        "float64",
        "int32",
    )
    assert hashlib.sha256(payload.encode()).hexdigest() == (
        "72fbbb05ec4fedd5fbe7cc271e2f233ed33d48c7d0a8cd3592b60f48bdfe5c14"
    )


def test_missing_columns_fail_closed() -> None:
    with pytest.raises(DataReadinessError, match="missing columns"):
        apply_cross_sectional_rank(
            pd.DataFrame({"session": ["2024-01-02"]}),
            top_quantile=0.2,
            bottom_quantile=0.2,
            within_sector=False,
            minimum_cross_section=50,
        )


def test_non_finite_atr_is_unresolved() -> None:
    entries = pd.DataFrame({"session": ["2024-01-01"], "atr": [np.nan]})

    out = apply_triple_barrier(_bars(_flat(8)), entries, spec=SPEC)

    assert pd.isna(out.loc[0, "barrier_label"])


def test_appending_future_bars_does_not_change_resolved_barrier() -> None:
    entries = pd.DataFrame({"session": ["2024-01-01"], "atr": [1.0]})
    original = _bars(_flat(8))
    extended = pd.concat(
        [
            original,
            _bars(
                [
                    ("2024-01-09", 999.0, 999.0, 1.0, 999.0),
                    ("2024-01-10", 1.0, 999.0, 1.0, 1.0),
                ]
            ),
        ],
        ignore_index=True,
    )

    expected = apply_triple_barrier(original, entries, spec=SPEC)
    actual = apply_triple_barrier(extended, entries, spec=SPEC)

    pdt.assert_frame_equal(actual, expected)


def test_appending_a_future_session_does_not_change_prior_ranks() -> None:
    first_session = _panel([i / 100 for i in range(100)])
    future_session = _panel(
        [1.0 - i / 100 for i in range(100)],
        session="2024-01-03",
    )
    expected = apply_cross_sectional_rank(
        first_session,
        top_quantile=0.2,
        bottom_quantile=0.2,
        within_sector=False,
        minimum_cross_section=50,
    )
    actual = apply_cross_sectional_rank(
        pd.concat([first_session, future_session], ignore_index=True),
        top_quantile=0.2,
        bottom_quantile=0.2,
        within_sector=False,
        minimum_cross_section=50,
    )

    pdt.assert_frame_equal(actual.iloc[: len(expected)].reset_index(drop=True), expected)
