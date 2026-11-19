"""Unit, causality, purge-isolation, and forward-recorder tests for Step 5a (`two_stage`, `free_backfill`, `forward_record`)."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from market_predictor.research_lane import forward_record, free_backfill, two_stage
from market_predictor.research_lane.panel import MARKET_TIMEZONE, PANEL_DIRECTORY, SealedPeriodError
from tests.support.research_lane_fundamentals import (
    SHARED_SECURITY,
    SyntheticFundamentals,
    cik,
    poisoned_inputs,
    synthetic_fundamental_inputs,
    truncated_inputs,
)
from tests.support.research_lane_panel import POST_END_PRICE, synthetic_panel, write_panel_files


@pytest.fixture(scope="module")
def synthetic() -> SyntheticFundamentals:
    return synthetic_fundamental_inputs()


def test_stage1_features_truncated_and_future_poison_invariance(synthetic: SyntheticFundamentals) -> None:
    inp = synthetic.inputs
    full = two_stage.build_stage1_features(inp.panel, inp)
    assert len(full.feature_names) == 48  # 24 cross-sectional + 24 sector-neutral, zero missingness flags

    cut = 430
    cut_day = pd.Timestamp(inp.panel.sessions[cut]).date()
    trunc_inp = truncated_inputs(synthetic, cut_day)
    trunc = two_stage.build_stage1_features(trunc_inp.panel, trunc_inp)

    rows = cut + 1
    assert np.allclose(trunc.equal_rank_24, full.equal_rank_24[:rows], rtol=1e-12, atol=0.0, equal_nan=True)
    for name in full.feature_names:
        assert np.allclose(trunc.features[name], full.features[name][:rows], rtol=1e-6, atol=1e-7, equal_nan=True), name

    p_inp = poisoned_inputs(synthetic, cut, seed=77)
    poisoned = two_stage.build_stage1_features(p_inp.panel, p_inp)
    assert np.allclose(poisoned.equal_rank_24[:rows], full.equal_rank_24[:rows], rtol=1e-12, atol=0.0, equal_nan=True)
    for name in full.feature_names:
        assert np.allclose(poisoned.features[name][:rows], full.features[name][:rows], rtol=1e-6, atol=1e-7, equal_nan=True), name


def test_dual_class_shares_stay_in_same_unseen_issuer_fold_and_walk_forward_runs(
    synthetic: SyntheticFundamentals,
) -> None:
    inp = synthetic.inputs
    buckets = two_stage._issuer_fold_ids(inp.panel.security_ids, inp.security_issuer)
    idx_shared = inp.panel.security_ids.index(SHARED_SECURITY)
    idx_primary = inp.panel.security_ids.index(f"cik:{cik(1)}")
    assert buckets[idx_shared] == buckets[idx_primary]

    bundle = two_stage.build_stage1_features(inp.panel, inp)
    res = two_stage.run_stage1_walk_forward(
        inp.panel, bundle, inp.security_issuer, first_row=260, horizons=(5, 21),
    )
    assert set(res) == {5, 21}
    assert res[21]["models"]["xgb_shallow_5seed"]["unseen_oos_scores"] is not None
    assert res[5]["models"]["xgb_shallow_5seed"]["unseen_oos_scores"] is None  # only primary/secondary run 5-fold unseen CV


def test_news_reaction_features_after_hours_clock_and_future_poison(tmp_path: Path) -> None:
    panel = synthetic_panel(sessions=360, securities=20, seed=21)
    end_day = pd.Timestamp(panel.sessions[-1]).date()
    panel_dir = tmp_path / "panel"
    (panel_dir / "final" / "panel" / "feature_profile=technical_market" / "year=2019").mkdir(parents=True)

    mem_rows = [
        {
            "ticker": f"T{c:02d}",
            "security_id": panel.security_ids[c],
            "sector": "Tech",
            "session_date_et": pd.Timestamp(s).date(),
            "decision_time_utc": pd.Timestamp(f"{pd.Timestamp(s).date().isoformat()} 23:00", tz="UTC"),
            "membership_available_at_utc": pd.Timestamp("2018-05-29 04:00", tz="UTC"),
            "cross_section_eligible": False,
            "pre_catalyst_feature_eligible": True,
        }
        for s in panel.sessions
        for c in range(len(panel.security_ids))
    ]
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(mem_rows), preserve_index=False),
        panel_dir / "final" / "panel" / "feature_profile=technical_market" / "year=2019" / "part.parquet",
    )

    early_dir = tmp_path / "early_news"
    later_dir = tmp_path / "later_news"
    early_dir.mkdir()
    later_dir.mkdir()

    # Place an after-hours 16:05 ET news event on session row 310 for security 0:
    # must NOT appear on row 310 (`e == 311`), and poisoning bars after row 312 must not alter rows <= 312!
    day_310 = pd.Timestamp(panel.sessions[310]).date()
    after_close_310 = pd.Timestamp(f"{day_310.isoformat()} 16:05:00", tz=MARKET_TIMEZONE).tz_convert("UTC")
    in_hours_305 = pd.Timestamp(f"{pd.Timestamp(panel.sessions[305]).date().isoformat()} 09:00:00", tz=MARKET_TIMEZONE).tz_convert("UTC")

    news_df = pd.DataFrame({
        "security_id": [panel.security_ids[0], panel.security_ids[1]],
        "ticker": ["T00", "T01"],
        "event_family": ["earnings", "guidance"],
        "feature_available_at_utc": [after_close_310, in_hours_305],
        "relation_channel": ["direct_issuer", "direct_issuer"],
        "research_eligible": [True, True],
    })
    pq.write_table(pa.Table.from_pandas(news_df, preserve_index=False), early_dir / "family_events.parquet")
    pq.write_table(pa.Table.from_pandas(news_df.iloc[:0], preserve_index=False), later_dir / "family_events.parquet")

    # Force panel.eligible True around rows 300..320 so centered_rank is active.
    elig = np.ones_like(panel.eligible)
    panel_all_elig = replace(panel, eligible=elig, development_end=end_day)
    feats_full, _, rep = two_stage.load_news_reaction_features(
        panel_all_elig, panel_root=panel_dir, early_news_dir=early_dir, later_news_dir=later_dir,
    )
    assert rep["mapped_direct_issuer_events"] == 2
    # Row 310 had no prior earnings event on col 0 because 16:05 ET maps to row 311!
    assert feats_full["news_rx_earnings"][310, 0] == pytest.approx(0.0, abs=1e-7)
    assert feats_full["news_rx_earnings"][311, 0] != 0.0

    # Poison prices strictly after row 320 -> rows <= 320 unchanged!
    p_close = panel_all_elig.close.copy()
    p_vol = panel_all_elig.volume.copy()
    p_close[321:] = POST_END_PRICE
    p_vol[321:] = 1e12
    poisoned_panel = replace(panel_all_elig, close=p_close, volume=p_vol)
    feats_poison, _, _ = two_stage.load_news_reaction_features(
        poisoned_panel, panel_root=panel_dir, early_news_dir=early_dir, later_news_dir=later_dir,
    )
    for name in two_stage.NEWS_FEATURE_NAMES:
        np.testing.assert_allclose(feats_poison[name][:321], feats_full[name][:321], rtol=0, atol=0, equal_nan=True)


def test_conservative_edgar_clock_and_trailing_warmup_counter() -> None:
    # 2019-04-25T16:30:00.000Z: as UTC = 16:30Z (12:30 EDT); as NY = 16:30 EDT (20:30Z).
    # Conservative max chooses 20:30Z (after close), preventing any premature intraday lookahead!
    ts = free_backfill._conservative_edgar_availability_utc("2019-04-25T16:30:00.000Z", "2019-04-25")
    assert ts == pd.Timestamp("2019-04-25 20:30:00+00:00")

    mat = np.ones((10, 2), dtype=np.float32)
    mat[3, 1] = np.nan
    cnt = free_backfill._trailing_finite_count(mat, window=4)
    assert cnt[:, 0].tolist() == [1, 2, 3, 4, 4, 4, 4, 4, 4, 4]
    assert cnt[:, 1].tolist() == [1, 2, 3, 3, 3, 3, 3, 4, 4, 4]


def test_forward_score_recorder_writes_snapshot_and_blocks_sealed_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    write_panel_files(tmp_path / PANEL_DIRECTORY, end=date(2024, 5, 28))

    with patch("market_predictor.research_lane.forward_record.guard"):
        manifest = forward_record.record_forward_scores(
            root=tmp_path, as_of_date=date(2024, 5, 28),
        )
    assert manifest["scored_session"] == "2024-05-28"
    snap_dir = tmp_path / forward_record.FORWARD_RECORD_DIRECTORY / "2024-05-28"
    assert (snap_dir / "scores.parquet").is_file()
    assert json.loads((snap_dir / "_manifest.json").read_text(encoding="utf-8"))["eligible_stocks_scored"] > 0

    # Duplicate session must fail closed (immutable append-only).
    with pytest.raises(FileExistsError), patch("market_predictor.research_lane.forward_record.guard"):
        forward_record.record_forward_scores(root=tmp_path, as_of_date=date(2024, 5, 28))

    # Historical sealed window (`2024-05-29..2026-07-08`) must fail closed in both panel and live_prospective modes!
    with pytest.raises(SealedPeriodError):
        forward_record.record_forward_scores(root=tmp_path, as_of_date=date(2024, 5, 29))
    with pytest.raises(SealedPeriodError):
        forward_record.record_forward_scores(root=tmp_path, as_of_date=date(2025, 8, 15), live_prospective=True)
