from __future__ import annotations

import pandas as pd
import pytest

from market_predictor.edge_rebuild.corpus_integrity import (
    IntegrityThresholds,
    verify_corpus_integrity,
)
from market_predictor.core.errors import DataReadinessError


def _row(
    session: str,
    ticker: str,
    bars: int,
    *,
    segment: str = "regular",
    expected: int = 78,
    zero_volume: int = 0,
    distinct_close: int = 40,
    last_close: float = 100.0,
    total_volume: int = 1_000_000,
) -> dict[str, object]:
    return {
        "session": session,
        "ticker": ticker,
        "segment": segment,
        "bars": bars,
        "expected_bars": expected,
        "zero_volume_bars": zero_volume,
        "distinct_close": distinct_close,
        "last_close": last_close,
        "total_volume": total_volume,
    }


def _clean(sessions: int = 6, tickers: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(f"2024-01-{day:02d}", f"T{i}", 78, last_close=100.0 + i)
            for day in range(2, 2 + sessions)
            for i in range(tickers)
        ]
    )


def test_clean_corpus_reports_no_defect() -> None:
    report = verify_corpus_integrity(_clean(), label="clean")

    assert report.defect_count == 0
    assert report.checked_bars == 78 * 30
    report.raise_if_defective("clean")


def test_truncated_session_is_caught_even_though_every_ticker_has_bars() -> None:
    """The 2022-03-08 shape: every ticker present, median one bar."""

    frame = _clean()
    frame.loc[frame["session"] == "2024-01-04", "bars"] = 1

    report = verify_corpus_integrity(frame, label="truncated")

    assert [item["session"] for item in report.truncated_sessions] == ["2024-01-04"]
    assert report.truncated_sessions[0]["median_completeness"] < 0.02
    with pytest.raises(DataReadinessError, match="truncated sessions"):
        report.raise_if_defective("truncated")


def test_single_truncated_ticker_session_is_caught() -> None:
    frame = _clean(sessions=30)
    frame.loc[(frame["session"] == "2024-01-05") & (frame["ticker"] == "T2"), "bars"] = 3

    report = verify_corpus_integrity(frame, label="hole")

    assert len(report.truncated_ticker_sessions) == 1
    assert report.truncated_ticker_sessions[0]["ticker"] == "T2"
    assert not report.truncated_sessions


def test_consistently_thin_symbol_is_not_a_defect() -> None:
    """AutoZone prints in roughly half the buckets every session.

    That is correct data about a high-priced, low-share-volume symbol. Whether
    it can be traded intraday is decided by the eligibility filter, not here.
    """

    # A realistic cross-section: mostly liquid symbols, one persistently thin.
    frame = pd.DataFrame(
        [
            _row(f"2024-01-{day:02d}", f"T{i}", 78)
            for day in range(2, 30)
            for i in range(8)
        ]
        + [_row(f"2024-01-{day:02d}", "AZO", 37) for day in range(2, 30)]
    )

    report = verify_corpus_integrity(frame, label="thin")

    assert report.defect_count == 0
    report.raise_if_defective("thin")


def test_liquid_symbol_collapsing_to_a_tenth_is_still_caught() -> None:
    """The failure worth catching: normally full coverage, suddenly almost none."""

    rows = [_row(f"2024-01-{day:02d}", "AAPL", 78) for day in range(2, 30)]
    rows.append(_row("2024-01-30", "AAPL", 6))
    frame = pd.DataFrame(rows)

    report = verify_corpus_integrity(frame, label="collapse")

    assert len(report.truncated_ticker_sessions) == 1
    finding = report.truncated_ticker_sessions[0]
    assert finding["session"] == "2024-01-30"
    assert finding["symbol_typical_completeness"] == pytest.approx(1.0)
    assert finding["share_of_typical"] < 0.1


def test_isolated_holes_are_recorded_but_do_not_block_a_large_corpus() -> None:
    """Two quiet symbol-days in 450,000 rows must not refuse the whole corpus."""

    rows = [
        _row(f"2024-{month:02d}-{day:02d}", f"T{i}", 78)
        for month in range(1, 13)
        for day in range(1, 25)
        for i in range(40)
    ]
    rows[0] = _row(rows[0]["session"], rows[0]["ticker"], 5)  # type: ignore[index]
    frame = pd.DataFrame(rows)

    report = verify_corpus_integrity(frame, label="large")

    assert len(report.truncated_ticker_sessions) == 1
    assert report.isolated_defects_tolerated
    assert report.blocking_defect_count == 0
    report.raise_if_defective("large")  # must not raise


def test_clustered_truncation_still_blocks() -> None:
    """A broken transport drags a whole session down and must refuse the build."""

    rows = [
        _row(f"2024-{month:02d}-{day:02d}", f"T{i}", 78)
        for month in range(1, 13)
        for day in range(1, 25)
        for i in range(40)
    ]
    frame = pd.DataFrame(rows)
    frame.loc[frame["session"] == "2024-06-12", "bars"] = 1

    report = verify_corpus_integrity(frame, label="clustered")

    assert report.truncated_sessions
    assert report.blocking_defect_count > 0
    with pytest.raises(DataReadinessError):
        report.raise_if_defective("clustered")


def test_frozen_price_outside_the_regular_session_is_not_a_defect() -> None:
    """A liquid symbol resting at one price across a few pre-market prints."""

    frame = pd.DataFrame(
        [
            _row(
                "2024-01-02",
                "PRU",
                12,
                segment="premarket",
                expected=66,
                distinct_close=1,
            )
        ]
    )

    report = verify_corpus_integrity(frame, label="premarket-flat")

    assert report.defect_count == 0


def test_reused_symbol_price_jump_is_caught() -> None:
    """The observed shape: a 3.15 series resuming at 115.88, a 36.8x jump."""

    frame = pd.DataFrame(
        [_row("2024-01-02", "FI", 78, last_close=3.15)]
        + [_row("2024-01-03", "FI", 78, last_close=115.88)]
    )

    report = verify_corpus_integrity(frame, label="reuse")

    breaks = [b for b in report.identity_breaks if b["reason"] == "close_ratio"]
    assert len(breaks) == 1
    assert breaks[0]["ratio"] == pytest.approx(36.787, rel=1e-3)


def test_long_interior_gap_is_caught_as_identity_break() -> None:
    members = {"FI": [f"2024-01-{day:02d}" for day in range(2, 16)]}
    frame = pd.DataFrame(
        [_row("2024-01-02", "FI", 78, last_close=3.15)]
        + [_row("2024-01-15", "FI", 78, last_close=3.20)]
    )

    report = verify_corpus_integrity(frame, label="gap", member_sessions=members)

    gaps = [b for b in report.identity_breaks if b["reason"] == "interior_gap"]
    assert len(gaps) == 1
    assert gaps[0]["gap_sessions"] == 12


def test_zero_volume_and_frozen_price_bars_are_rejected() -> None:
    """Provider placeholders for a dead symbol are not observations."""

    frame = pd.DataFrame(
        [
            _row(
                "2024-01-02",
                "DEAD",
                78,
                zero_volume=78,
                distinct_close=1,
                last_close=3.15,
                total_volume=0,
            )
        ]
    )

    report = verify_corpus_integrity(frame, label="fabricated")

    reasons = sorted(item["reason"] for item in report.fabricated_bars)
    assert reasons == ["frozen_price", "zero_volume"]


def test_extended_hours_sparsity_is_never_a_defect() -> None:
    """Thin pre/post-market volume is observed absence, not missing data.

    Treating it as a defect would filter the cross-section by liquidity, which
    is the bias the rebuild exists to remove.
    """

    frame = pd.DataFrame(
        [
            _row("2024-01-02", "MCO", 1, segment="premarket", expected=66),
            _row("2024-01-02", "VZ", 31, segment="premarket", expected=66),
            _row("2024-01-02", "MCO", 0, segment="postmarket", expected=48),
        ]
    )

    report = verify_corpus_integrity(frame, label="extended")

    assert report.defect_count == 0
    assert report.benign_thin_trading == 3
    report.raise_if_defective("extended")


def test_thresholds_reject_incoherent_limits() -> None:
    with pytest.raises(ValueError, match="completeness floor"):
        IntegrityThresholds(minimum_regular_completeness=0.0)
    with pytest.raises(ValueError, match="close-ratio ceiling"):
        IntegrityThresholds(maximum_session_close_ratio=1.0)
    with pytest.raises(ValueError, match="excluded-symbol share"):
        IntegrityThresholds(maximum_excluded_symbol_share=1.01)
    with pytest.raises(ValueError, match="frozen at 5%"):
        IntegrityThresholds(maximum_excluded_symbol_share=0.04)


def test_missing_input_columns_fail_closed() -> None:
    with pytest.raises(DataReadinessError, match="lacks columns"):
        verify_corpus_integrity(
            pd.DataFrame({"session": ["2024-01-02"], "ticker": ["T0"]}),
            label="partial",
        )


def test_a_flat_price_on_real_volume_is_trading_not_a_placeholder() -> None:
    """A par-pegged preferred prints a whole session at one price on 15M shares."""

    frame = pd.DataFrame(
        [
            _row(
                "2024-01-02",
                "STRC",
                78,
                distinct_close=1,
                last_close=95.99,
                total_volume=15_824_325,
            )
        ]
    )

    report = verify_corpus_integrity(frame, label="pegged")

    assert report.defect_count == 0


def test_a_level_jump_across_sparse_coverage_is_not_an_identity_break() -> None:
    """Screened symbols are collected only on selected sessions.

    Two consecutive rows can sit months apart, so the ratio between them
    measures months of drift rather than one move.
    """

    frame = pd.DataFrame(
        [
            _row("2024-01-02", "CAPR", 78, last_close=6.36),
            _row("2024-06-14", "CAPR", 78, last_close=29.96),
        ]
    )

    report = verify_corpus_integrity(frame, label="sparse")

    assert not [b for b in report.identity_breaks if b["reason"] == "close_ratio"]
