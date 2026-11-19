"""Unit, causality, arithmetic, and end-to-end tests for Option C (`benchmark_tilt.py`).

Uses synthetic fixtures only; proves point-in-time causality under truncation and future
poisoning, Step 2a `quality_slow` parity, open-to-open telescope & post-drift turnover
accounting, rank-hysteresis turnover reduction, and exact +-2% stock / 0% sector active bounds.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from market_predictor.research_lane import benchmark_tilt, fundamentals
from market_predictor.research_lane.company_facts import (
    COMPANY_FACTS_DIRECTORY,
    COMPANY_FACTS_FILE,
    COMPANY_FACTS_MANIFEST,
    COMPANY_FACTS_SOURCES,
    IssuerFacts,
)
from market_predictor.research_lane.panel import MARKET_TIMEZONE, PANEL_DIRECTORY, SealedPeriodError
from market_predictor.research_lane.run import score_benchmark_tilt
from market_predictor.research_lane.sec_filings import FILINGS_DIRECTORY, FILINGS_FILE, FILINGS_MANIFEST
from market_predictor.research_lane.unadjusted_prices import (
    UNADJUSTED_AUTHORITY,
    UNADJUSTED_DIRECTORY,
    UNADJUSTED_MANIFEST,
)
from tests.support.research_lane_fundamentals import (
    SyntheticFundamentals,
    cik,
    issuer_facts,
    poisoned_inputs,
    synthetic_fundamental_inputs,
    tiny_panel,
    truncated_inputs,
)
from tests.support.research_lane_panel import synthetic_panel, write_panel_files

CUTS = (300, 371, 455, 530)


@pytest.fixture(scope="module")
def synthetic() -> SyntheticFundamentals:
    return synthetic_fundamental_inputs()


def test_frozen_option_c_constants() -> None:
    assert benchmark_tilt.TILT_HORIZONS == (3, 5, 10, 21)
    assert benchmark_tilt.SIGNAL_NAMES == (
        "reversal_short",
        "quality_slow",
        "quality_plus_reversal",
        "quality_gated_reversal",
    )
    assert benchmark_tilt.POLICY_NAMES == (
        "skill_top10_unbuffered",
        "skill_top10_hold33",
        "product_cap_mult_buffered",
        "product_sector_active_slice",
        "product_core_spy_80_sat_20",
    )
    assert benchmark_tilt.BUY_TOP_FRACTION == pytest.approx(0.10)
    assert benchmark_tilt.HOLD_TOP_FRACTION == pytest.approx(1.0 / 3.0)
    assert benchmark_tilt.STOCK_ACTIVE_LIMIT == pytest.approx(0.02)
    assert benchmark_tilt.SECTOR_ACTIVE_LIMIT == pytest.approx(0.05)
    assert benchmark_tilt.FROZEN_COST_PER_TURNOVER == pytest.approx(0.0020)
    assert benchmark_tilt.INFO_COST_PER_TURNOVER == pytest.approx(0.0010)


def test_quality_slow_and_split_blanking_match_step2a_bit_for_bit(synthetic: SyntheticFundamentals) -> None:
    inp = synthetic.inputs
    step2a_res = fundamentals.compute_fundamental_predictors(inp)
    step2a_composites = fundamentals.compute_composites(step2a_res.values, inp.panel.eligible)

    qc = benchmark_tilt.build_quality_and_market_cap(
        inp.panel, inp.unadjusted_close, inp.security_issuer, inp.load_issuer,
    )
    np.testing.assert_array_equal(qc.quality_slow, step2a_composites["composite_quality"])
    np.testing.assert_array_equal(qc.split_blanked, step2a_res.split_blanked)
    assert qc.report["securities_with_any_market_value"] > 15


@pytest.mark.parametrize("cut", CUTS)
def test_truncated_history_leaves_signals_market_value_and_decision_weights_identical(
    synthetic: SyntheticFundamentals, cut: int,
) -> None:
    inp = synthetic.inputs
    full_qc = benchmark_tilt.build_quality_and_market_cap(
        inp.panel, inp.unadjusted_close, inp.security_issuer, inp.load_issuer,
    )
    full_signals = benchmark_tilt.build_option_c_signals(inp.panel, full_qc.quality_slow)

    cut_day = pd.Timestamp(inp.panel.sessions[cut]).date()
    trunc_inp = truncated_inputs(synthetic, cut_day)
    trunc_qc = benchmark_tilt.build_quality_and_market_cap(
        trunc_inp.panel, trunc_inp.unadjusted_close, trunc_inp.security_issuer, trunc_inp.load_issuer,
    )
    trunc_signals = benchmark_tilt.build_option_c_signals(trunc_inp.panel, trunc_qc.quality_slow)

    rows = cut + 1
    assert trunc_qc.market_value.shape[0] == rows
    np.testing.assert_array_equal(trunc_qc.market_value, full_qc.market_value[:rows])
    for name in benchmark_tilt.SIGNAL_NAMES:
        assert np.allclose(trunc_signals[name], full_signals[name][:rows], rtol=1e-12, atol=0.0, equal_nan=True), name

    # Verify decision weights at decision_rows <= cut are identical under truncated panel.
    dec_rows = np.arange(260, cut + 1, 5, dtype=np.int64)
    for sig_name in ("reversal_short", "quality_gated_reversal"):
        w_s1_full, _ = benchmark_tilt.build_skill_weights(
            inp.panel, full_signals[sig_name], dec_rows, buffered=True, minimum_stocks=10,
        )
        w_s1_trunc, _ = benchmark_tilt.build_skill_weights(
            trunc_inp.panel, trunc_signals[sig_name], dec_rows, buffered=True, minimum_stocks=10,
        )
        assert np.allclose(w_s1_trunc, w_s1_full, rtol=1e-12, atol=0.0, equal_nan=True)

        for mode in ("mult_buffered", "sector_active_slice"):
            w_cap_full, _ = benchmark_tilt.build_cap_tilt_weights(
                inp.panel, full_signals[sig_name], full_qc.market_value, dec_rows, mode=mode, minimum_stocks=10,
            )
            w_cap_trunc, _ = benchmark_tilt.build_cap_tilt_weights(
                trunc_inp.panel, trunc_signals[sig_name], trunc_qc.market_value, dec_rows, mode=mode, minimum_stocks=10,
            )
            assert np.allclose(w_cap_trunc, w_cap_full, rtol=1e-12, atol=0.0, equal_nan=True)


@pytest.mark.parametrize("cut", (340, 480))
def test_future_poison_changes_nothing_at_or_before_cut(synthetic: SyntheticFundamentals, cut: int) -> None:
    inp = synthetic.inputs
    full_qc = benchmark_tilt.build_quality_and_market_cap(
        inp.panel, inp.unadjusted_close, inp.security_issuer, inp.load_issuer,
    )
    full_signals = benchmark_tilt.build_option_c_signals(inp.panel, full_qc.quality_slow)

    p_inp = poisoned_inputs(synthetic, cut, seed=cut)
    p_qc = benchmark_tilt.build_quality_and_market_cap(
        p_inp.panel, p_inp.unadjusted_close, p_inp.security_issuer, p_inp.load_issuer,
    )
    p_signals = benchmark_tilt.build_option_c_signals(p_inp.panel, p_qc.quality_slow)

    rows = cut + 1
    np.testing.assert_array_equal(p_qc.market_value[:rows], full_qc.market_value[:rows])
    for name in benchmark_tilt.SIGNAL_NAMES:
        assert np.allclose(p_signals[name][:rows], full_signals[name][:rows], rtol=1e-12, atol=0.0, equal_nan=True)

    # Include `cut` itself as a decision row: poisoning `open[cut + 1]` and all later bars/filings
    # must leave decision weights chosen at close[cut] strictly unchanged.
    dec_rows = np.arange(260, cut + 1, 5, dtype=np.int64)
    if dec_rows[-1] != cut:
        dec_rows = np.append(dec_rows, cut)
    for mode in ("mult_buffered", "sector_active_slice"):
        w_full, _ = benchmark_tilt.build_cap_tilt_weights(
            inp.panel, full_signals["quality_plus_reversal"], full_qc.market_value, dec_rows,
            mode=mode, minimum_stocks=10,
        )
        w_poison, _ = benchmark_tilt.build_cap_tilt_weights(
            p_inp.panel, p_signals["quality_plus_reversal"], p_qc.market_value, dec_rows,
            mode=mode, minimum_stocks=10,
        )
        assert np.allclose(w_poison, w_full, rtol=1e-12, atol=0.0, equal_nan=True)


def test_open_to_open_telescope_and_zero_drift_turnover_arithmetic() -> None:
    days = [d.date().isoformat() for d in pd.bdate_range("2023-01-02", periods=16)]
    base = tiny_panel(days, columns=3, close=100.0)
    opens = np.zeros((16, 3), dtype=np.float32)
    for col, growth in enumerate((1.01, 0.995, 1.02)):
        opens[:, col] = (100.0 * (growth ** np.arange(16))).astype(np.float32)
    spy_open = (400.0 * (1.005 ** np.arange(16))).astype(np.float32)
    panel = replace(base, open=opens, close=opens.copy(), spy_open=spy_open, spy_close=spy_open.copy())

    windows = benchmark_tilt.continuous_rebalance_windows(panel, first_row=0, horizon=3)
    # Decision rows: 0, 3, 6, 9, 12 -> entry rows 1, 4, 7, 10, 13 -> exit rows 4, 7, 10, 13, 16 ( wait: max_decision=16-4=12 -> 0,3,6,9 )
    assert windows.decision_rows.tolist() == [0, 3, 6, 9]
    assert windows.entry_rows.tolist() == [1, 4, 7, 10]
    assert windows.exit_rows.tolist() == [4, 7, 10, 13]

    # Telescope check on SPY: prod(1 + r_k) == spy_open[13] / spy_open[1]
    spy_comp = float(np.prod(1.0 + windows.spy_returns) - 1.0)
    expected_spy = float(spy_open[13]) / float(spy_open[1]) - 1.0
    assert spy_comp == pytest.approx(expected_spy, rel=1e-6)

    # Constant-share portfolio: shares = [1.0, 2.0, 1.0] valued at each period's entry open -> zero post-drift turnover!
    shares = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    w_schedule = np.zeros((len(windows.decision_rows), 3), dtype=np.float64)
    for k, e_row in enumerate(windows.entry_rows):
        vals = shares * opens[int(e_row)].astype(np.float64)
        w_schedule[k] = vals / vals.sum()

    path = benchmark_tilt.simulate_portfolio_path(w_schedule, windows.stock_returns)
    assert path.one_way_turnovers[0] == pytest.approx(0.5)  # initial entry
    assert np.isnan(path.steady_turnovers[0])
    assert np.allclose(path.steady_turnovers[1:], 0.0, atol=1e-6)

    # Compounded gross return equals buy-and-hold return of initial weights from row 1 to row 13.
    total_hold = float(np.dot(w_schedule[0], opens[13].astype(np.float64) / opens[1].astype(np.float64) - 1.0))
    comp_gross = float(np.prod(1.0 + path.gross_returns) - 1.0)
    assert comp_gross == pytest.approx(total_hold, rel=1e-6)

    # Exact cost check on full replacement (100% col 0 -> 100% col 1 -> 100% col 2).
    switch_w = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    sw_path = benchmark_tilt.simulate_portfolio_path(switch_w, windows.stock_returns)
    assert np.allclose(sw_path.steady_turnovers[1:], 1.0, atol=1e-12)
    assert sw_path.net_returns_20bps[0] == pytest.approx(sw_path.gross_returns[0] - 0.0010, abs=1e-12)
    assert sw_path.net_returns_20bps[1] == pytest.approx(sw_path.gross_returns[1] - 0.0020, abs=1e-12)


def test_rank_hysteresis_buffer_reduces_turnover_and_preserves_top10_count() -> None:
    days = [d.date().isoformat() for d in pd.bdate_range("2022-01-03", periods=45)]
    panel = tiny_panel(days, columns=60, close=50.0)
    rng = np.random.default_rng(42)
    base_quality = np.linspace(-1.0, 1.0, 60, dtype=np.float64)
    # Add cross-sectional jitter around the 90th percentile cutoff so unbuffered churns while hold-33% retains.
    signal = np.zeros((45, 60), dtype=np.float64)
    for r in range(45):
        signal[r] = base_quality + rng.normal(0.0, 0.18, size=60)

    windows = benchmark_tilt.continuous_rebalance_windows(panel, first_row=0, horizon=3)
    w_unbuf, _ = benchmark_tilt.build_skill_weights(panel, signal, windows.decision_rows, buffered=False, minimum_stocks=50)
    w_buf, _ = benchmark_tilt.build_skill_weights(panel, signal, windows.decision_rows, buffered=True, minimum_stocks=50)

    assert np.all((w_unbuf > 0.0).sum(axis=1) == 6)
    assert np.all((w_buf > 0.0).sum(axis=1) == 6)
    assert np.allclose(w_buf.sum(axis=1), 1.0)

    p_unbuf = benchmark_tilt.simulate_portfolio_path(w_unbuf, windows.stock_returns)
    p_buf = benchmark_tilt.simulate_portfolio_path(w_buf, windows.stock_returns)
    assert np.nanmean(p_buf.steady_turnovers) < np.nanmean(p_unbuf.steady_turnovers)


def test_sector_bounded_projection_enforces_exact_stock_and_sector_caps() -> None:
    rng = np.random.default_rng(99)
    n = 80
    raw_cap = np.exp(rng.normal(0.0, 1.5, size=n))
    raw_cap[0] = raw_cap.sum() * 0.18  # mega-cap ~15%
    raw_cap[1] = raw_cap.sum() * 0.10  # #2 mega-cap ~8%
    bench = raw_cap / raw_cap.sum()
    sector = np.asarray([i % 4 for i in range(n)], dtype=np.int16)

    # Alternate +1 and -1 within each of the 4 sectors (including -1 on mega-cap 0 in sector 0).
    state = np.where((np.arange(n) // 4) % 2 == 1, 1.0, -1.0)
    desired = bench * (1.0 + state)

    projected = benchmark_tilt.project_sector_bounded_weights(desired, bench, sector, stock_active_limit=0.02)
    assert np.all(projected >= -1e-15)
    assert projected.sum() == pytest.approx(1.0, abs=1e-13)
    assert float(np.max(np.abs(projected - bench))) <= 0.02 + 1e-13
    # Mega-cap 0 had bench > 0.10 and desired 0.0 -> must be floored at bench[0] - 0.02!
    assert projected[0] == pytest.approx(bench[0] - 0.02, abs=1e-12)
    for g in range(4):
        sec_diff = abs(float((projected[sector == g] - bench[sector == g]).sum()))
        assert sec_diff <= 1e-13


def _facts_to_sec_json(facts_map: IssuerFacts) -> dict[str, Any]:
    out: dict[str, Any] = {"facts": {}}
    for full_tag, items in facts_map.items():
        tax, tag = full_tag.split(":", 1)
        unit = "shares" if "Shares" in tag else "USD"
        rows = []
        for f in items:
            entry: dict[str, Any] = {
                "end": f.end.isoformat(),
                "val": f.value,
                "filed": f.filed.isoformat(),
                "accn": f.accession,
            }
            if f.start is not None:
                entry["start"] = f.start.isoformat()
            rows.append(entry)
        out["facts"].setdefault(tax, {})[tag] = {"units": {unit: rows}}
    return out


def test_evaluate_option_c_active_bounds_and_reading_rule_on_60_stocks() -> None:
    base_panel = synthetic_panel(sessions=420, securities=60, seed=11)
    cik_ids = tuple(f"cik:{cik(i)}" for i in range(60))
    panel = replace(base_panel, security_ids=cik_ids)
    unadjusted = panel.close.astype(np.float64) * 1.5
    facts_map = {cik(i): issuer_facts(i) for i in range(60)}
    mapping = {sec_id: sec_id.split(":")[1] for sec_id in cik_ids}

    qc = benchmark_tilt.build_quality_and_market_cap(panel, unadjusted, mapping, facts_map.get)
    signals = benchmark_tilt.build_option_c_signals(panel, qc.quality_slow)
    result = benchmark_tilt.evaluate_option_c(panel, signals, qc.market_value, first_row=280, horizons=(3, 5))

    assert set(result["cells"]) == set(benchmark_tilt.SIGNAL_NAMES)
    for sig in benchmark_tilt.SIGNAL_NAMES:
        for h in ("3", "5"):
            cell = result["cells"][sig][h]
            assert cell["skill_top10_hold33"]["vs_matched_benchmark"]["periods"] > 10
            assert cell["product_cap_mult_buffered"]["vs_matched_benchmark"]["periods"] > 10
            assert cell["product_cap_mult_buffered"]["max_stock_active_abs"] <= 0.02 + 1e-12
            assert cell["product_cap_mult_buffered"]["max_sector_active_abs"] <= 1e-12
            assert cell["product_sector_active_slice"]["max_stock_active_abs"] <= 0.02 + 1e-12
            assert cell["product_sector_active_slice"]["max_sector_active_abs"] <= 1e-12


def test_end_to_end_score_benchmark_tilt_and_sealed_period_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    end_day = date(2024, 5, 28)
    info = write_panel_files(tmp_path / PANEL_DIRECTORY, end=end_day)
    sec_ids = ("sec:A", "sec:B", "sec:C", "sec:D")

    # Write SEC filings parquet & manifest mapping sec:A..sec:D to CIKs 0..3.
    filings_dir = tmp_path / FILINGS_DIRECTORY
    filings_dir.mkdir(parents=True, exist_ok=True)
    accepted = pd.Timestamp("2024-04-15 12:00", tz=MARKET_TIMEZONE).tz_convert("UTC")
    filings_df = pd.DataFrame({
        "security_id": list(sec_ids),
        "sec_cik": [cik(i) for i in range(4)],
        "accession_number": [f"acc-{i}" for i in range(4)],
        "sec_form": ["10-Q"] * 4,
        "item_codes": [""] * 4,
        "accepted_at_utc": [accepted] * 4,
        "available_at_utc": [accepted] * 4,
    })
    filings_df.to_parquet(filings_dir / FILINGS_FILE, index=False)
    (filings_dir / FILINGS_MANIFEST).write_text("{}", encoding="utf-8")

    # Write primary company facts for issuers 1..3, and put issuer 0 in `extra_company_facts` with a post-end row!
    facts_root = tmp_path / COMPANY_FACTS_DIRECTORY
    (facts_root / "final").mkdir(parents=True, exist_ok=True)
    (facts_root / COMPANY_FACTS_MANIFEST).write_text("{}", encoding="utf-8")
    sources_dir = facts_root / COMPANY_FACTS_SOURCES
    for i in range(1, 4):
        c_str = cik(i)
        dest = sources_dir / c_str
        dest.mkdir(parents=True, exist_ok=True)
        (dest / COMPANY_FACTS_FILE).write_text(json.dumps(_facts_to_sec_json(issuer_facts(i))), encoding="utf-8")

    extra_dir = tmp_path / benchmark_tilt.EXTRA_COMPANY_FACTS_DIRECTORY / cik(0) / "20261106"
    extra_dir.mkdir(parents=True, exist_ok=True)
    extra_json = _facts_to_sec_json(issuer_facts(0))
    # Inject a post-development-end filing that Rule 2 must strip cleanly without error or leakage.
    extra_json["facts"]["us-gaap"]["Assets"]["units"]["USD"].append(
        {"end": "2025-06-30", "val": 9.99e15, "filed": "2025-08-01", "accn": "sealed-future"}
    )
    (extra_dir / COMPANY_FACTS_FILE).write_text(json.dumps(extra_json), encoding="utf-8")

    # Write unadjusted daily bars for sec:A..sec:D.
    unadj_root = tmp_path / UNADJUSTED_DIRECTORY
    (unadj_root / "bars").mkdir(parents=True, exist_ok=True)
    (unadj_root / UNADJUSTED_AUTHORITY).write_text("{}", encoding="utf-8")
    unit_artifacts = []
    sess_strings = [d.isoformat() for d in info["days"] if d <= end_day]
    for i, sec_id in enumerate(sec_ids):
        rel = f"bars/sec_{i}.parquet"
        df = pd.DataFrame({
            "security_id": sec_id,
            "session_date": sess_strings,
            "close": np.full(len(sess_strings), 50.0 + 10.0 * i, dtype=np.float64),
            "source": "alpaca",
            "timeframe": "1Day",
            "price_feed": "sip",
            "adjustment": "raw",
        })
        df.to_parquet(unadj_root / rel, index=False)
        unit_artifacts.append({"role": "stock", "security_id": sec_id, "bars_path": rel})
    (unadj_root / UNADJUSTED_MANIFEST).write_text(json.dumps({"unit_artifacts": unit_artifacts}), encoding="utf-8")

    out_rel = Path("data/research/research_lane/benchmark_tilt_test")
    with patch("market_predictor.research_lane.run.guard"):
        res = score_benchmark_tilt(root=tmp_path, output=out_rel, development_end=end_day, horizons=(3, 5))

    assert cik(0) in [k.split("_")[-2] for k in res["inputs"] if k.startswith("extra_company_facts_")]
    board = json.loads((tmp_path / out_rel / "scoreboard.json").read_text(encoding="utf-8"))
    assert "research label" in board["notice"]
    assert board["quality_and_cap_inputs"]["extra_company_facts_ciks_read"] == [cik(0)]
    assert set(board["benchmark_tilt"]["cells"]) == set(benchmark_tilt.SIGNAL_NAMES)

    md_text = (tmp_path / out_rel / "scoreboard.md").read_text(encoding="utf-8")
    assert "Pre-registered reading rule" in md_text
    trial_lines = (tmp_path / "data/research/research_lane/trials.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trial_lines) == 1
    assert json.loads(trial_lines[0])["command"] == "score-benchmark-tilt"

    # Sealed period must fail closed without explicit override.
    with pytest.raises(SealedPeriodError):
        score_benchmark_tilt(
            root=tmp_path,
            output=Path("data/research/research_lane/should_fail"),
            development_end=date(2024, 5, 29),
        )
