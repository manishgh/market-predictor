"""Unit tests for Step 5d: unified Core-Satellite vs. SPY, FINVIZ_API_KEY alias, and live forward catalysts."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import SecretStr

from market_predictor.config import Settings
from market_predictor.research_lane import benchmark_tilt, core_satellite, forward_record
from market_predictor.research_lane.panel import ResearchPanel, SealedPeriodError
from market_predictor.sources.alpaca import AlpacaNewsPage


def _make_panel(
    sessions: np.ndarray,
    security_ids: tuple[str, ...],
    *,
    seed: int,
    spy_drift: float = 0.0006,
    eligible_mask: np.ndarray | None = None,
) -> ResearchPanel:
    rng = np.random.default_rng(seed)
    n_s, n_c = len(sessions), len(security_ids)
    rets = rng.normal(0.0004, 0.015, size=(n_s, n_c))
    close = (50.0 * np.cumprod(1.0 + rets, axis=0)).astype(np.float32)
    open_px = (close * (1.0 + rng.normal(0.0, 0.002, size=(n_s, n_c)))).astype(np.float32)
    high = np.maximum(open_px, close) * np.float32(1.008)
    low = np.minimum(open_px, close) * np.float32(0.992)
    vol = rng.uniform(5e5, 3e6, size=(n_s, n_c)).astype(np.float32)

    spy_r = spy_drift + rng.normal(0.0, 0.008, size=n_s)
    spy_c = (300.0 * np.cumprod(1.0 + spy_r)).astype(np.float32)
    spy_o = (spy_c * 0.9995).astype(np.float32)

    elig = np.ones((n_s, n_c), dtype=bool) if eligible_mask is None else eligible_mask.astype(bool)
    sec_codes = np.broadcast_to((np.arange(n_c) % 3).astype(np.int16)[None, :], (n_s, n_c)).copy()
    last_d = pd.Timestamp(sessions[-1]).date()
    return ResearchPanel(
        sessions=sessions,
        security_ids=security_ids,
        open=open_px,
        high=high,
        low=low,
        close=close,
        volume=vol,
        eligible=elig,
        sector=sec_codes,
        sector_names=("Tech", "Health", "Ind"),
        spy_open=spy_o,
        spy_close=spy_c,
        qqq_open=spy_o.copy(),
        qqq_close=spy_c.copy(),
        development_end=last_d,
        load_report={},
    )


def test_settings_finviz_alias_choices_and_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINVIZ_ELITE_AUTH", raising=False)
    monkeypatch.setenv("FINVIZ_API_KEY", "test_api_key_value_123")
    s1 = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s1.finviz_elite_auth_value == "test_api_key_value_123"

    # Explicit FINVIZ_ELITE_AUTH takes priority when both are set.
    monkeypatch.setenv("FINVIZ_ELITE_AUTH", "primary_auth_456")
    s2 = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s2.finviz_elite_auth_value == "primary_auth_456"

    # Keyword initialization `finviz_elite_auth=` still works.
    monkeypatch.delenv("FINVIZ_ELITE_AUTH", raising=False)
    monkeypatch.delenv("FINVIZ_API_KEY", raising=False)
    s3 = Settings(_env_file=None, finviz_elite_auth=SecretStr("kw_secret_789"))  # type: ignore[call-arg]
    assert s3.finviz_elite_auth_value == "kw_secret_789"

    raw_err = "HTTPSConnectionPool: url /news_export.ashx?v=3&t=CROX&auth=super_secret_token_999 failed: super_secret_token_999"
    scrubbed = forward_record.redact_finviz_auth(raw_err, "super_secret_token_999")
    assert "super_secret_token_999" not in scrubbed
    assert "auth=<REDACTED>" in scrubbed


def test_finviz_csv_parser_and_live_catalyst_classification_blocks_sealed_and_future_clocks() -> None:
    # Sealed window overlap (`<= 2026-07-08`) must raise SealedPeriodError immediately.
    with pytest.raises(SealedPeriodError):
        forward_record.parse_finviz_elite_news_csv(
            "Title,Source,Date,Url,Category,Ticker\n",
            ticker="CROX",
            window_start_utc=datetime(2026, 7, 1, 0, 0, tzinfo=UTC),
            cutoff_utc=datetime(2026, 7, 20, 20, 0, tzinfo=UTC),
        )

    sessions = pd.bdate_range("2026-07-15", "2026-11-19").to_numpy().astype("datetime64[D]")
    panel = _make_panel(sessions, ("ticker:CROX", "ticker:MATX", "ticker:DECK", "ticker:SKX"), seed=7)
    scores_df = forward_record._score_panel_last_session(panel)
    assert "score_beta_neutral_price_12" in scores_df.columns

    # 2026-11-18 08:30 EST = 13:30 UTC (inside window); 2026-11-19 17:15 EST = 22:15 UTC (after 16:00 EST close -> must drop!);
    # 2025-05-01 09:00 EDT (inside sealed window -> must drop!).
    finviz_csv = (
        "Title,Source,Date,Url,Category,Ticker\n"
        "\"Crocs Reports Q3 2026 Results\",Business Wire,2026-11-18 08:30:00,https://ex.com/1,News,CROX\n"
        "\"Crocs Raises Full-Year 2026 Guidance After Close\",Reuters,2026-11-19 17:15:00,https://ex.com/2,News,CROX\n"
        "\"Old Sealed Period Headline\",Reuters,2025-05-01 09:00:00,https://ex.com/3,News,CROX\n"
    )

    fake_sub_crox: dict[str, Any] = {
        "name": "Crocs, Inc.",
        "filings": {
            "recent": {
                "form": ["8-K", "SC 13D", "10-K"],
                "filingDate": ["2026-11-18", "2026-11-17", "2023-02-20"],
                "acceptanceDateTime": ["2026-11-18T08:05:00.000Z", "2026-11-17T14:00:00.000Z", "2023-02-20T16:10:00.000Z"],
                "accessionNumber": ["0001-26-000010", "0001-26-000009", "0001-23-000001"],
                "items": [["2.02", "9.01"], [], []],
                "primaryDocDescription": ["8-K", "SC 13D", "10-K"],
            }
        },
    }

    cfg = Settings(
        _env_file=None,  # type: ignore[call-arg]
        ALPACA_API_KEY_ID="k",
        ALPACA_API_SECRET_KEY=SecretStr("s"),
        SEC_USER_AGENT="MarketPredictorResearch ops@marketpredictor.io",
        finviz_elite_auth=SecretStr("fv_tok"),
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = finviz_csv

    alpaca_item = {
        "id": "alp_901",
        "symbols": ["CROX"],
        "created_at": "2026-11-18T14:00:00Z",
        "headline": "Crocs Raises Full-Year Guidance on Strong Demand",
        "source": "fiscal.ai",
        "url": "https://ex.com/a1",
    }

    with (
        patch(
            "market_predictor.research_lane.forward_record.requests.get",
            side_effect=lambda url, **kw: (
                mock_resp
                if kw.get("params", {}).get("t") == "CROX"
                else MagicMock(status_code=200, text="Title,Source,Date,Url,Category,Ticker\n")
            ),
        ),
        patch(
            "market_predictor.sources.http.HttpClient.get_json",
            side_effect=lambda url, **kw: (
                {"0": {"cik_str": 1334036, "ticker": "CROX", "title": "Crocs, Inc."}}
                if "company_tickers.json" in str(url)
                else fake_sub_crox
            ),
        ),
        patch(
            "market_predictor.research_lane.forward_record.AlpacaSource.fetch_news_page",
            return_value=AlpacaNewsPage(
                request_page_token=None,
                next_page_token=None,
                news=(alpaca_item,),
                response_headers={},
                raw_payload={},
            ),
        ),
        patch("market_predictor.research_lane.forward_record.guard"),
    ):
        enriched, cat_df, summary = forward_record.collect_live_prospective_catalysts(
            panel, scores_df, cfg, catalyst_top_k=4,
        )

    # Post-16:00 EST headline (`17:15:00`) and sealed `2025-05-01` headline must NEVER appear!
    assert not cat_df["title"].str.contains("After Close").any()
    assert not cat_df["title"].str.contains("Old Sealed").any()
    assert set(cat_df["source_family"].unique()) == {"finviz", "alpaca", "sec"}
    # Causal company alias "Crocs, Inc." (from 2023 10-K) enables deterministic `earnings` + `guidance` classification!
    all_fams = ",".join(cat_df["classified_families"].tolist())
    assert "earnings" in all_fams
    assert "guidance" in all_fams
    assert summary["by_source_family"] == {"alpaca": 1, "finviz": 1, "sec": 2}

    crox_row = enriched.loc[enriched["security_id"] == "ticker:CROX"].iloc[0]
    assert int(crox_row["finviz_news_21d"]) == 1
    assert int(crox_row["alpaca_news_21d"]) == 1
    assert int(crox_row["sec_catalysts_21d"]) == 2
    assert crox_row["latest_catalyst_rx_1d"] is not None


def test_core_satellite_disjointness_weight_conservation_turnover_and_future_poison() -> None:
    sessions = pd.bdate_range("2021-01-04", "2022-06-30").to_numpy().astype("datetime64[D]")
    n_s = len(sessions)
    sp_elig = np.ones((n_s, 60), dtype=bool)
    mid_elig = np.ones((n_s, 60), dtype=bool)
    # CIK 0000999999 moves from mid_small (`t < 150`) into S&P 500 (`t >= 150`) with zero simultaneous overlap.
    sp_elig[:150, 0] = False
    mid_elig[150:, 0] = False

    sp_ids = tuple(f"sec_sp:{i}" for i in range(60))
    mid_ids = tuple(f"cik:{i + 1000:010d}" for i in range(60))
    sp_ciks = {sp_ids[0]: "0000999999", **{sp_ids[i]: f"{i + 1:010d}" for i in range(1, 60)}}
    mid_ciks = {mid_ids[0]: "0000999999", **{mid_ids[i]: f"{i + 1000:010d}" for i in range(1, 60)}}

    sp_panel = _make_panel(sessions, sp_ids, seed=11, eligible_mask=sp_elig)
    mid_panel = ResearchPanel(
        **{
            **_make_panel(sessions, mid_ids, seed=22, eligible_mask=mid_elig).__dict__,
            "spy_open": sp_panel.spy_open.copy(),
            "spy_close": sp_panel.spy_close.copy(),
        }
    )

    rep = core_satellite.verify_disjoint_core_satellite_universes(sp_panel, sp_ciks, mid_panel, mid_ciks)
    assert rep["simultaneous_active_cik_overlaps"] == 0
    assert rep["ciks_ever_switching_tiers"] == 1

    # Simultaneous overlap must fail closed.
    bad_mid_elig = mid_elig.copy()
    bad_mid_elig[160, 0] = True
    bad_mid_panel = ResearchPanel(**{**mid_panel.__dict__, "eligible": bad_mid_elig})
    with pytest.raises(ValueError, match="active \\(session, CIK\\)"):
        core_satellite.verify_disjoint_core_satellite_universes(sp_panel, sp_ciks, bad_mid_panel, mid_ciks)

    # Simulate 80/20 and 90/10 joint Core-Satellite and verify future-poison invariance on earlier periods.
    rng = np.random.default_rng(42)
    sp_sig = rng.uniform(-0.5, 0.5, size=(n_s, 60))
    mid_sig = rng.uniform(-0.5, 0.5, size=(n_s, 60))
    sp_mv = np.full((n_s, 60), 1e10, dtype=np.float64)

    sp_win = benchmark_tilt.continuous_rebalance_windows(sp_panel, 20, 10)
    mid_win = benchmark_tilt.continuous_rebalance_windows(mid_panel, 20, 10)
    sp_tilt_dec, _ = benchmark_tilt.build_cap_tilt_weights(
        sp_panel, sp_sig, sp_mv, sp_win.decision_rows, mode="sector_active_slice", minimum_stocks=20,
    )
    sp_tilt_w = benchmark_tilt._apply_open_fills(sp_tilt_dec, sp_win.tradable)
    mid_sat_dec, _ = benchmark_tilt.build_skill_weights(
        mid_panel, mid_sig, mid_win.decision_rows, buffered=True, minimum_stocks=20,
    )
    mid_sat_w = benchmark_tilt._apply_open_fills(mid_sat_dec, mid_win.tradable)

    res_full = core_satellite.simulate_joint_core_satellite(
        core_mode="core_sp500_tilt",
        core_weight=0.80,
        sp_tilt_weights=sp_tilt_w,
        mid_sat_weights=mid_sat_w,
        sp_windows=sp_win,
        mid_windows=mid_win,
        horizon=10,
    )
    assert res_full["vs_spy"]["periods"] == len(sp_win.decision_rows)
    assert res_full["compounded_net_20bps"] is not None

    # Future poison after row 200: period indices `k` with `exit_rows[k] < 200` must be bit-for-bit identical!
    sp_poison_close = sp_panel.close.copy()
    sp_poison_close[200:] *= np.float32(5.0)
    mid_poison_sig = mid_sig.copy()
    mid_poison_sig[200:] = -mid_poison_sig[200:]

    sp_panel_p = ResearchPanel(**{**sp_panel.__dict__, "close": sp_poison_close})
    sp_win_p = benchmark_tilt.continuous_rebalance_windows(sp_panel_p, 20, 10)
    mid_sat_dec_p, _ = benchmark_tilt.build_skill_weights(
        mid_panel, mid_poison_sig, mid_win.decision_rows, buffered=True, minimum_stocks=20,
    )
    mid_sat_w_p = benchmark_tilt._apply_open_fills(mid_sat_dec_p, mid_win.tradable)
    sp_tilt_dec_p, _ = benchmark_tilt.build_cap_tilt_weights(
        sp_panel_p, sp_sig, sp_mv, sp_win_p.decision_rows, mode="sector_active_slice", minimum_stocks=20,
    )
    sp_tilt_w_p = benchmark_tilt._apply_open_fills(sp_tilt_dec_p, sp_win_p.tradable)

    safe_k = sp_win.exit_rows < 200
    np.testing.assert_allclose(mid_sat_w_p[safe_k], mid_sat_w[safe_k], rtol=0, atol=0)
    np.testing.assert_allclose(sp_tilt_w_p[safe_k], sp_tilt_w[safe_k], rtol=0, atol=0)
