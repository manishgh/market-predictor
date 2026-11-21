"""Unit tests for Step 5h: prospective forward-score & OPRA option maturation evaluator."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import SecretStr

from market_predictor.config import Settings
from market_predictor.research_lane.forward_evaluate import evaluate_forward_scores
from market_predictor.research_lane.forward_record import FORWARD_RECORD_DIRECTORY
from market_predictor.research_lane.panel import SealedPeriodError


def _write_snapshot(
    base_dir: Path,
    snap_date: str,
    *,
    backwardation: bool,
    veto_last: bool = False,
) -> None:
    snap_dir = base_dir / snap_date
    snap_dir.mkdir(parents=True, exist_ok=True)
    # 12 stocks S00..S11 where S00 has highest score_beta_neutral_price_12 & score_r2_opra_conditioned
    rows = []
    for idx in range(12):
        sc = 0.5 - (idx / 11.0)
        is_veto = veto_last and idx == 11
        rows.append({
            "session_date": snap_date,
            "security_id": f"ticker:S{idx:02d}",
            "sector": "Tech",
            "close_adjusted": 100.0,
            "score_price_equal_12": sc,
            "score_beta_neutral_price_12": sc,
            "score_r2_opra_conditioned": sc,
            "score_reversal_short": -sc,
            "score_reversal_skew_vetoed": None if is_veto else -sc,
            "falling_knife_put_skew_veto": is_veto,
            "catalyst_shock_to_straddle": 1.40 if idx == 0 else 0.30,
        })
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), snap_dir / "scores.parquet")
    manifest = {
        "scored_session": snap_date,
        "options_summary": {"opra_enabled": True, "opra_backwardation": backwardation},
    }
    (snap_dir / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_forward_evaluator_maturation_vs_pending_and_sealed_guard(tmp_path: Path) -> None:
    cfg = Settings(_env_file=None, ALPACA_API_KEY_ID="k", ALPACA_API_SECRET_KEY=SecretStr("s"))  # type: ignore[call-arg]
    fdir = tmp_path / FORWARD_RECORD_DIRECTORY
    _write_snapshot(fdir, "2026-11-16", backwardation=False, veto_last=True)
    _write_snapshot(fdir, "2026-11-19", backwardation=True, veto_last=False)
    _write_snapshot(fdir, "2026-11-20", backwardation=False, veto_last=False)

    # Sealed as_of_date <= 2026-07-08 must raise SealedPeriodError immediately
    with pytest.raises(SealedPeriodError):
        evaluate_forward_scores(root=tmp_path, as_of_date=date(2025, 12, 15), settings=cfg)

    sessions = ["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]
    bar_rows = []
    for s_idx, s_day in enumerate(sessions):
        bar_rows.append({"ticker": "SPY", "session": s_day, "open": 500.0 + s_idx, "close": 501.0 + s_idx})
        for stk_i in range(12):
            # S00 rises strongly (+4%/day), S11 falls (-3%/day, confirming falling-knife veto!)
            drift = 0.04 - (stk_i * 0.006)
            px_o = 100.0 * ((1.0 + drift) ** s_idx)
            px_c = px_o * (1.0 + drift)
            bar_rows.append({"ticker": f"S{stk_i:02d}", "session": s_day, "open": px_o, "close": px_c})
    fake_bars = pd.DataFrame(bar_rows)

    with patch("market_predictor.research_lane.forward_evaluate._fetch_post_holdout_bars", return_value=fake_bars), patch(
        "market_predictor.research_lane.forward_evaluate.guard"
    ), patch.dict("os.environ", {"MARKET_PREDICTOR_RUNTIME_DIR": str(tmp_path / "rt")}):
        res = evaluate_forward_scores(
            root=tmp_path,
            as_of_date=date(2026, 11, 20),
            horizons=(1, 3, 10),
            settings=cfg,
        )

    # 2026-11-16 matures h=1, h=3 (h=10 pending); 2026-11-19 matures h=1 (h=3,10 pending); 2026-11-20 has 0 matured (pending_entry_bar)
    assert res["snapshots_total"] == 3
    assert res["matured_pairs"] == 3
    assert res["pending_pairs"] == 6

    report = json.loads((tmp_path / res["output"] / "report.json").read_text(encoding="utf-8"))
    assert report["sealed_holdout_untouched"] is True
    # Positive cross-sectional IC because S00..S11 scores match monotonic positive-to-negative returns
    sig_1d = report["pooled_signals_by_horizon"]["1"]["score_r2_opra_conditioned"]
    assert sig_1d["matured_dates"] == 2
    assert sig_1d["mean_rank_ic"] > 0.90
    assert sig_1d["mean_top10_net_minus_spy"] > 0.02

    # On 2026-11-19 (`backwardation=True`), R1 must shift 100% into SPY (`net_excess_vs_spy == 0.0`)
    r1_1d = report["pooled_candidates_by_horizon"]["1"]["90_10__spy_core__R1_backwardation_to_spy"]["by_snapshot"]
    nov19_r1 = next(item for item in r1_1d if item["snapshot_date"] == "2026-11-19")
    assert nov19_r1["opra_backwardation"] is True
    assert nov19_r1["net_excess_vs_spy"] == pytest.approx(0.0, abs=1e-12)

    # Vetoed stock S11 on 2026-11-16 underperformed SPY (`vetoed_mean_excess_vs_spy < 0`), while shock>straddle S00 beat SPY!
    diag_nov16_1d = next(
        d for d in report["opra_diagnostics"] if d["snapshot_date"] == "2026-11-16" and d["horizon_sessions"] == 1
    )
    assert diag_nov16_1d["vetoed_count"] == 1
    assert float(diag_nov16_1d["vetoed_mean_excess_vs_spy"]) < -0.02
    assert diag_nov16_1d["shock_gt_straddle_count"] == 1
    assert float(diag_nov16_1d["shock_gt_straddle_mean_excess_vs_spy"]) > 0.02
