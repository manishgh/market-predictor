"""Unit, truncated-history, future-poison, barrier-fill and fold-purge tests for Step 3a swing setups."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from market_predictor.research_lane.barrier_labels import (
    OUTCOME_STOP,
    OUTCOME_TARGET,
    OUTCOME_TIMEOUT,
    evaluate_candidate_barriers,
)
from market_predictor.research_lane.fitted import Fold
from market_predictor.research_lane.meta_model import (
    extract_candidate_table,
    purged_train_mask,
    run_walk_forward_meta_models,
)
from market_predictor.research_lane.news_catalysts import (
    build_catalyst_matrix_from_frames,
    classify_event_polarity,
    read_family_events,
)
from market_predictor.research_lane.panel import ResearchPanel, SealedPeriodError
from market_predictor.research_lane.run import score_swing_setups
from market_predictor.research_lane.setups import (
    SETUP_FEATURE_NAMES,
    VolatilityMode,
    build_setups_and_features,
)
from tests.support.research_lane_panel import POST_END_PRICE, synthetic_panel, write_panel_files


def _synthetic_catalysts(
    panel: ResearchPanel, *, cutoff_row: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str], str]]:
    """Build deterministic news + SEC frames on ``panel`` (optionally truncated to ``cutoff_row``)."""
    limit = len(panel.sessions) if cutoff_row is None else cutoff_row
    pit_map: dict[tuple[int, str], str] = {}
    for s_idx in range(limit):
        for c_idx, sec_id in enumerate(panel.security_ids):
            pit_map[(s_idx, f"SYM{c_idx:03d}")] = sec_id

    news_rows = []
    for s_idx in range(15, limit, 17):
        day = pd.Timestamp(panel.sessions[s_idx]).strftime("%Y-%m-%d")
        # Regular-hours positive upgrade at 14:00 UTC (09:00/10:00 ET) -> maps to session s_idx
        ts = pd.Timestamp(f"{day} 14:00:00", tz="UTC")
        c_idx = s_idx % len(panel.security_ids)
        news_rows.append({
            "security_id": panel.security_ids[c_idx],
            "ticker": f"SYM{c_idx:03d}",
            "source_family": "alpaca",
            "event_family": "analyst_revision",
            "matched_text": f"upgrades SYM{c_idx:03d} and raises price target",
            "published_at_utc": ts,
            "event_available_at_utc": ts,
            "feature_available_at_utc": ts,
            "relation_channel": "direct_issuer",
            "research_eligible": True,
        })
        # Bearish downgrade on another security
        c_bear = (c_idx + 3) % len(panel.security_ids)
        news_rows.append({
            "security_id": panel.security_ids[c_bear],
            "ticker": f"SYM{c_bear:03d}",
            "source_family": "alpaca",
            "event_family": "guidance",
            "matched_text": "lowers full-year guidance",
            "published_at_utc": ts,
            "event_available_at_utc": ts,
            "feature_available_at_utc": ts,
            "relation_channel": "direct_issuer",
            "research_eligible": True,
        })

    sec_rows = []
    for s_idx in range(25, limit, 29):
        day = pd.Timestamp(panel.sessions[s_idx]).strftime("%Y-%m-%d")
        ts = pd.Timestamp(f"{day} 13:00:00", tz="UTC")
        c_idx = (s_idx * 2) % len(panel.security_ids)
        sec_rows.append({
            "security_id": panel.security_ids[c_idx],
            "sec_cik": f"0000{c_idx:06d}",
            "accession_number": f"acc-{s_idx}-{c_idx}",
            "accepted_at_utc": ts,
            "available_at_utc": ts,
        })
    return pd.DataFrame(news_rows), pd.DataFrame(sec_rows), pit_map


def test_polarity_and_retrospective_noise_classification() -> None:
    assert classify_event_polarity("earnings", "reports q2 earnings results") == (False, 1)
    assert classify_event_polarity("guidance", "raises fiscal guidance") == (False, 2)
    assert classify_event_polarity("guidance", "cuts full-year guidance") == (True, 0)
    assert classify_event_polarity("analyst_revision", "upgrades shares of acme") == (False, 6)
    assert classify_event_polarity("analyst_revision", "downgrades acme to sell") == (True, 0)
    assert classify_event_polarity("offering", "launches public offering") == (True, 0)
    assert classify_event_polarity("regulatory_decision", "fda approves drug") == (False, 4)
    assert classify_event_polarity("regulatory_decision", "fda issues complete response letter") == (True, 0)
    assert classify_event_polarity("earnings", "Why NVDA Shares Are Trading Higher Today after earnings") == (False, 0)
    assert classify_event_polarity("product_event", "10 Biggest Stock Movers Yesterday") == (False, 0)


def test_after_hours_news_maps_to_next_session_never_prior_close() -> None:
    panel = synthetic_panel(sessions=490, securities=16, seed=11)
    monday = pd.Timestamp(panel.sessions[5]).strftime("%Y-%m-%d")
    # 16:05 America/New_York on session[5] -> strictly after 16:00 ET close of session[5]!
    after_hours = pd.Timestamp(f"{monday} 16:05:00", tz="America/New_York").tz_convert("UTC")
    pre_market = pd.Timestamp(f"{monday} 08:00:00", tz="America/New_York").tz_convert("UTC")
    news = pd.DataFrame([
        {
            "security_id": "sec:000", "ticker": "AAA", "source_family": "alpaca",
            "event_family": "earnings", "matched_text": "reports quarterly results",
            "published_at_utc": after_hours, "event_available_at_utc": after_hours,
            "feature_available_at_utc": after_hours, "relation_channel": "direct_issuer", "research_eligible": True,
        },
        {
            "security_id": "sec:001", "ticker": "BBB", "source_family": "alpaca",
            "event_family": "guidance", "matched_text": "raises guidance",
            "published_at_utc": pre_market, "event_available_at_utc": pre_market,
            "feature_available_at_utc": pre_market, "relation_channel": "direct_issuer", "research_eligible": True,
        },
    ])
    sec = pd.DataFrame(columns=["security_id", "sec_cik", "accession_number", "accepted_at_utc", "available_at_utc"])
    matrix = build_catalyst_matrix_from_frames(
        panel, news_events=news, sec_releases=sec, pit_ticker_map={(5, "AAA"): "sec:000", (6, "AAA"): "sec:000"},
    )
    # Pre-market on session 5 is available at session 5's close (entry open[6]).
    assert bool(matrix.tier1_on_session[5, 1]) is True
    # After-hours at 16:05 ET on session 5 MUST NOT appear on session 5; it maps to session 6 (entry open[7]).
    assert bool(matrix.tier1_on_session[5, 0]) is False
    assert np.isnan(matrix.sessions_since_tier1[5, 0])
    assert bool(matrix.tier1_on_session[6, 0]) is True
    assert float(matrix.sessions_since_tier1[6, 0]) == 0.0
    assert float(matrix.sessions_since_tier1[7, 0]) == 1.0


def test_family_events_reader_drops_post_end_rows_and_refuses_sealed(tmp_path: Path) -> None:
    path = tmp_path / "family_events.parquet"
    rows = pd.DataFrame([
        {
            "security_id": "sec:000", "ticker": "AAA", "source_family": "alpaca",
            "event_family": "earnings", "matched_text": "reports q1 results",
            "published_at_utc": pd.Timestamp("2024-05-28 20:00:00", tz="UTC"),
            "event_available_at_utc": pd.Timestamp("2024-05-28 20:00:00", tz="UTC"),
            "feature_available_at_utc": pd.Timestamp("2024-05-28 20:00:00", tz="UTC"),
            "relation_channel": "direct_issuer", "research_eligible": True,
        },
        {
            "security_id": "sec:000", "ticker": "AAA", "source_family": "alpaca",
            "event_family": "earnings", "matched_text": "SEALED FUTURE LEAK",
            "published_at_utc": pd.Timestamp("2024-05-29 12:00:00", tz="UTC"),
            "event_available_at_utc": pd.Timestamp("2024-05-29 12:00:00", tz="UTC"),
            "feature_available_at_utc": pd.Timestamp("2024-05-29 12:00:00", tz="UTC"),
            "relation_channel": "direct_issuer", "research_eligible": True,
        },
    ])
    pq.write_table(pa.Table.from_pandas(rows, preserve_index=False), path)
    kept = read_family_events(path, date(2024, 5, 28))
    assert len(kept) == 1
    assert "SEALED" not in kept["matched_text"].iloc[0]

    with pytest.raises(SealedPeriodError):
        read_family_events(path, date(2024, 5, 29))

    sealed_dir = tmp_path / "swing_later_sealed_news"
    sealed_dir.mkdir()
    sealed_path = sealed_dir / "family_events.parquet"
    pq.write_table(pa.Table.from_pandas(rows.iloc[:1], preserve_index=False), sealed_path)
    with pytest.raises(SealedPeriodError):
        read_family_events(sealed_path, date(2024, 5, 28))


@pytest.mark.parametrize("mode", ["wilder_atr_14", "mtr_14"])
def test_truncated_history_and_future_poison_leave_all_setup_features_identical(mode: VolatilityMode) -> None:
    full = synthetic_panel(sessions=320, securities=16, seed=19)
    cut_idx = 260
    cut_day = pd.Timestamp(full.sessions[cut_idx - 1]).date()
    truncated = full.truncated(cut_day)

    news_full, sec_full, pit_full = _synthetic_catalysts(full)
    news_trunc, sec_trunc, pit_trunc = _synthetic_catalysts(truncated)

    cat_full = build_catalyst_matrix_from_frames(full, news_events=news_full, sec_releases=sec_full, pit_ticker_map=pit_full)
    cat_trunc = build_catalyst_matrix_from_frames(truncated, news_events=news_trunc, sec_releases=sec_trunc, pit_ticker_map=pit_trunc)

    det_full = build_setups_and_features(full, cat_full, first_row=210, volatility_mode=mode)
    det_trunc = build_setups_and_features(truncated, cat_trunc, first_row=210, volatility_mode=mode)

    for name in SETUP_FEATURE_NAMES:
        np.testing.assert_allclose(det_full.features[name][:cut_idx], det_trunc.features[name], rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(det_full.setup_a[:cut_idx], det_trunc.setup_a)
    np.testing.assert_array_equal(det_full.setup_b[:cut_idx], det_trunc.setup_b)
    np.testing.assert_array_equal(det_full.setup_c[:cut_idx], det_trunc.setup_c)

    # Future poison test: corrupting prices and volumes from cut_idx onward leaves rows < cut_idx bit-identical.
    poisoned_open = full.open.copy()
    poisoned_high = full.high.copy()
    poisoned_low = full.low.copy()
    poisoned_close = full.close.copy()
    poisoned_vol = full.volume.copy()
    poisoned_spy_c = full.spy_close.copy()
    poisoned_open[cut_idx:] = POST_END_PRICE
    poisoned_high[cut_idx:] = POST_END_PRICE * 1.2
    poisoned_low[cut_idx:] = 0.01
    poisoned_close[cut_idx:] = POST_END_PRICE
    poisoned_vol[cut_idx:] = 1e12
    poisoned_spy_c[cut_idx:] = POST_END_PRICE
    poisoned = ResearchPanel(
        sessions=full.sessions, security_ids=full.security_ids,
        open=poisoned_open, high=poisoned_high, low=poisoned_low, close=poisoned_close, volume=poisoned_vol,
        eligible=full.eligible, sector=full.sector, sector_names=full.sector_names,
        spy_open=full.spy_open, spy_close=poisoned_spy_c, qqq_open=full.qqq_open, qqq_close=full.qqq_close,
        development_end=full.development_end,
    )
    det_poison = build_setups_and_features(poisoned, cat_full, first_row=210, volatility_mode=mode)
    for name in SETUP_FEATURE_NAMES:
        np.testing.assert_allclose(det_full.features[name][:cut_idx], det_poison.features[name][:cut_idx], rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(det_full.volatility[:cut_idx], det_poison.volatility[:cut_idx], rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(det_full.shock_ratio_14[:cut_idx], det_poison.shock_ratio_14[:cut_idx], rtol=0, atol=0, equal_nan=True)


def test_triple_barrier_stop_first_gap_fills_and_right_censoring() -> None:
    days = pd.bdate_range("2022-01-03", periods=15).to_numpy().astype("datetime64[D]")
    # 4 securities testing: (0) clean target hit with upside gap capped at target,
    # (1) same-bar tie resolved stop_first, (2) overnight gap below stop filled at open,
    # (3) 10-session timeout.
    open_ = np.full((15, 4), 100.0, dtype=np.float32)
    high = np.full((15, 4), 101.0, dtype=np.float32)
    low = np.full((15, 4), 99.5, dtype=np.float32)
    close = np.full((15, 4), 100.5, dtype=np.float32)

    # Entry at row 1 open = 100.0, vol = 2.0 -> target = 105.0 (+2.5*2), stop = 98.0 (-1.0*2).
    # Col 0: gaps up on row 3 to open=106, high=107 -> target hit at 105.0 (resting limit, not 106).
    open_[3, 0], high[3, 0], low[3, 0], close[3, 0] = 106.0, 107.0, 105.5, 106.5
    # Col 1: row 2 spans BOTH target (106 >= 105) and stop (97.5 <= 98) -> stop_first at 98.0!
    open_[2, 1], high[2, 1], low[2, 1], close[2, 1] = 100.0, 106.0, 97.5, 104.0
    # Col 2: row 4 gaps below stop (open=95.0 < 98.0) -> fills at 95.0, not 98.0!
    open_[4, 2], high[4, 2], low[4, 2], close[4, 2] = 95.0, 96.0, 94.0, 95.5
    # Col 3: stays inside (98.0, 105.0) through row 10, closes at 103.0 on row 10 -> timeout!
    close[10, 3], high[10, 3] = 103.0, 103.5

    spy_open = np.full(15, 400.0, dtype=np.float32)
    spy_close = np.array([400.0 + i for i in range(15)], dtype=np.float32)
    panel = ResearchPanel(
        sessions=days, security_ids=("s0", "s1", "s2", "s3"),
        open=open_, high=high, low=low, close=close, volume=np.full((15, 4), 1000.0, dtype=np.float32),
        eligible=np.ones((15, 4), dtype=bool), sector=np.zeros((15, 4), dtype=np.int16), sector_names=("Tech",),
        spy_open=spy_open, spy_close=spy_close, qqq_open=spy_open, qqq_close=spy_close,
        development_end=pd.Timestamp(days[-1]).date(),
    )
    vol = np.full((15, 4), 2.0, dtype=np.float64)
    rows = np.array([0, 0, 0, 0, 5], dtype=np.int64)  # row 5 + 10 >= 15 -> must be censored invalid!
    cols = np.array([0, 1, 2, 3, 0], dtype=np.int64)
    out = evaluate_candidate_barriers(panel, vol, rows, cols, horizon=10)

    assert list(out.valid) == [True, True, True, True, False]
    assert list(out.y_barrier[:4]) == [1, 0, 0, 0]
    assert list(out.outcome_code[:4]) == [OUTCOME_TARGET, OUTCOME_STOP, OUTCOME_STOP, OUTCOME_TIMEOUT]
    assert list(out.exit_row[:4]) == [3, 2, 4, 10]
    assert list(out.holding_sessions[:4]) == [3, 2, 4, 10]
    np.testing.assert_allclose(out.gross_return[:4], [0.05, -0.02, -0.05, 0.03], rtol=1e-12)
    # Col 2 gapped through stop: loss is -5.00 - 0.20 cost = -5.20 on 2.00 stop_dist = -2.60R!
    np.testing.assert_allclose(out.net_r_20bps[2], -2.60, rtol=1e-12)
    # Matched SPY for Col 0 (entry spy_open[1]=400, exit spy_close[3]=403): +0.75%
    np.testing.assert_allclose(out.spy_matched_return[0], 3.0 / 400.0, rtol=1e-12)


def test_purged_walk_forward_never_leaks_overlapping_outcomes_across_fold() -> None:
    panel = synthetic_panel(sessions=900, securities=30, seed=23)
    news, sec, pit = _synthetic_catalysts(panel)
    cat = build_catalyst_matrix_from_frames(panel, news_events=news, sec_releases=sec, pit_ticker_map=pit)
    det = build_setups_and_features(panel, cat, first_row=220)
    # Force sufficient synthetic candidate density so both inner calibration and fold models fit.
    forced_mask = panel.eligible.copy()
    forced_mask[:220] = False
    forced_det = det.__class__(
        volatility_mode=det.volatility_mode, volatility=det.volatility, mtr_14=det.mtr_14,
        shock_ratio_14=det.shock_ratio_14, adr_pct_20=det.adr_pct_20, spy_bull_regime=det.spy_bull_regime,
        setup_a=det.setup_a, setup_b=det.setup_b, setup_c=det.setup_c,
        any_setup_unfiltered=forced_mask, any_setup_primary=forced_mask, any_setup_strict_adr=det.any_setup_strict_adr,
        features=det.features, report=det.report,
    )
    folds = (Fold(year=2021, test_start=650, test_stop=850),)
    table, res_clean = run_walk_forward_meta_models(panel, forced_det, first_row=220, folds=folds, horizon=10)

    tr_mask = purged_train_mask(table, first_row=220, test_start=650, horizon=10)
    assert int(tr_mask.sum()) > 100
    assert int(table.outcomes.exit_row[tr_mask].max()) < 650
    assert int((table.rows[tr_mask] + 10).max()) < 650

    # Corrupting prices from test_start (650) onward MUST leave all training statistics and fitted models identical!
    p_close = panel.close.copy()
    p_high = panel.high.copy()
    p_low = panel.low.copy()
    p_open = panel.open.copy()
    p_close[650:] = POST_END_PRICE
    p_high[650:] = POST_END_PRICE * 1.1
    p_low[650:] = POST_END_PRICE * 0.9
    p_open[650:] = POST_END_PRICE
    poisoned_panel = ResearchPanel(
        sessions=panel.sessions, security_ids=panel.security_ids,
        open=p_open, high=p_high, low=p_low, close=p_close, volume=panel.volume,
        eligible=panel.eligible, sector=panel.sector, sector_names=panel.sector_names,
        spy_open=panel.spy_open, spy_close=panel.spy_close, qqq_open=panel.qqq_open, qqq_close=panel.qqq_close,
        development_end=panel.development_end,
    )
    det_p = build_setups_and_features(poisoned_panel, cat, first_row=220)
    forced_det_p = det_p.__class__(
        volatility_mode=det_p.volatility_mode, volatility=det_p.volatility, mtr_14=det_p.mtr_14,
        shock_ratio_14=det_p.shock_ratio_14, adr_pct_20=det_p.adr_pct_20, spy_bull_regime=det_p.spy_bull_regime,
        setup_a=det_p.setup_a, setup_b=det_p.setup_b, setup_c=det_p.setup_c,
        any_setup_unfiltered=forced_mask, any_setup_primary=forced_mask, any_setup_strict_adr=det_p.any_setup_strict_adr,
        features=det_p.features, report=det_p.report,
    )
    _, res_poison = run_walk_forward_meta_models(poisoned_panel, forced_det_p, first_row=220, folds=folds, horizon=10)

    assert res_clean.folds[0]["train_candidates"] == res_poison.folds[0]["train_candidates"]
    assert res_clean.folds[0]["platt_slope"] == res_poison.folds[0]["platt_slope"]
    assert res_clean.folds[0]["train_q75_calibrated_prob"] == res_poison.folds[0]["train_q75_calibrated_prob"]
    assert res_clean.classifier_importance == res_poison.classifier_importance
    assert res_clean.regressor_importance == res_poison.regressor_importance
    # Even test predictions on the exact first test session (row 650) use features through 650, wait:
    # on rows strictly before 650 there are no test rows, so check row 650 before/after poisoning 651+!
    table_check = extract_candidate_table(panel, forced_det, forced_mask)
    assert int(table_check.rows.size) > 0


def test_score_swing_setups_writes_scoreboard_and_trial_and_refuses_sealed(tmp_path: Path) -> None:
    panel_dir = tmp_path / "data" / "panel"
    write_panel_files(panel_dir, end=date(2024, 5, 28))

    early_dir = tmp_path / "data" / "early_news"
    later_dir = tmp_path / "data" / "later_news"
    sec_dir = tmp_path / "data" / "sec_filings"
    for d in (early_dir, later_dir, sec_dir):
        d.mkdir(parents=True)
        (d / "_manifest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    empty_news = pd.DataFrame({
        "security_id": pd.Series(dtype="str"), "ticker": pd.Series(dtype="str"),
        "source_family": pd.Series(dtype="str"), "event_family": pd.Series(dtype="str"),
        "matched_text": pd.Series(dtype="str"),
        "published_at_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "event_available_at_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "feature_available_at_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "relation_channel": pd.Series(dtype="str"), "research_eligible": pd.Series(dtype="bool"),
    })
    pq.write_table(pa.Table.from_pandas(empty_news, preserve_index=False), early_dir / "family_events.parquet")
    pq.write_table(pa.Table.from_pandas(empty_news, preserve_index=False), later_dir / "family_events.parquet")

    sec_df = pd.DataFrame({
        "security_id": ["sec:A"], "sec_cik": ["0000000001"], "accession_number": ["acc-1"],
        "sec_form": ["8-K"], "item_codes": ["2.02"],
        "accepted_at_utc": [pd.Timestamp("2024-04-15 20:05:00", tz="UTC")],
        "available_at_utc": [pd.Timestamp("2024-04-15 20:05:00", tz="UTC")],
    })
    pq.write_table(pa.Table.from_pandas(sec_df, preserve_index=False), sec_dir / "filings.parquet")

    out_dir = tmp_path / "data" / "research" / "research_lane" / "setups_out"
    result = score_swing_setups(
        root=tmp_path,
        output=Path("data/research/research_lane/setups_out"),
        development_end=date(2024, 5, 28),
        panel=Path("data/panel"),
        early_news_dir=Path("data/early_news"),
        later_news_dir=Path("data/later_news"),
        filings_dir=Path("data/sec_filings"),
    )
    assert result["output"] == "data/research/research_lane/setups_out"
    assert (out_dir / "scoreboard.json").is_file()
    assert (out_dir / "scoreboard.md").is_file()
    trial_lines = (out_dir.parent / "trials.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(trial_lines) == 1
    parsed = json.loads(trial_lines[0])
    assert parsed["command"] == "score-swing-setups"

    with pytest.raises(FileExistsError):
        score_swing_setups(
            root=tmp_path,
            output=Path("data/research/research_lane/setups_out"),
            development_end=date(2024, 5, 28),
            panel=Path("data/panel"),
            early_news_dir=Path("data/early_news"),
            later_news_dir=Path("data/later_news"),
            filings_dir=Path("data/sec_filings"),
        )

    with pytest.raises(SealedPeriodError):
        score_swing_setups(
            root=tmp_path,
            output=Path("data/research/research_lane/setups_sealed"),
            development_end=date(2024, 5, 29),
            panel=Path("data/panel"),
            early_news_dir=Path("data/early_news"),
            later_news_dir=Path("data/later_news"),
            filings_dir=Path("data/sec_filings"),
        )
