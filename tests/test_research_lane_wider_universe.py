"""Unit, ticker-reuse quarantine, and causality tests for Step 5b (`wider_universe`)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from market_predictor.research_lane import wider_universe
from market_predictor.research_lane.panel import MARKET_TIMEZONE
from tests.support.research_lane_panel import POST_END_PRICE, synthetic_panel


def test_quarantine_ticker_spans_prevents_cross_issuer_overlap_and_merges_same_cik() -> None:
    raw = pd.DataFrame([
        # Earlier issuer on EMC (`0000790070`) overlapping later issuer (`0001999999`)
        {"cik": "0000790070", "ticker": "EMC", "valid_from": date(2016, 1, 4), "valid_to": date(2019, 12, 31), "source_priority": 3},
        {"cik": "0001999999", "ticker": "EMC", "valid_from": date(2018, 6, 1), "valid_to": date(2024, 5, 28), "source_priority": 2},
        # Duplicate adjacent spans for same CIK merge cleanly
        {"cik": "0000320193", "ticker": "AAPL", "valid_from": date(2016, 1, 4), "valid_to": date(2019, 7, 8), "source_priority": 1},
        {"cik": "0000320193", "ticker": "AAPL", "valid_from": date(2018, 5, 29), "valid_to": date(2024, 5, 28), "source_priority": 2},
    ])
    clean = wider_universe.quarantine_ticker_spans(raw)
    emc = clean.loc[clean["ticker"] == "EMC"].sort_values("valid_from").reset_index(drop=True)
    assert len(emc) == 2
    assert emc.loc[0, "cik"] == "0000790070"
    assert emc.loc[0, "valid_to"] == date(2018, 5, 31)
    assert emc.loc[1, "cik"] == "0001999999"
    assert emc.loc[1, "valid_from"] == date(2018, 6, 1)

    aapl = clean.loc[clean["ticker"] == "AAPL"]
    assert len(aapl) == 1
    assert aapl.iloc[0]["valid_from"] == date(2016, 1, 4)
    assert aapl.iloc[0]["valid_to"] == date(2024, 5, 28)


def test_pit_liquidity_eligibility_truncated_and_future_poison_invariance() -> None:
    rng = np.random.default_rng(42)
    sessions = pd.bdate_range("2017-01-03", periods=450).to_numpy().astype("datetime64[D]")
    n_sess, n_sec = len(sessions), 25
    close = rng.uniform(10.0, 120.0, size=(n_sess, n_sec)).astype(np.float32)
    volume = rng.uniform(200_000.0, 5_000_000.0, size=(n_sess, n_sec)).astype(np.float32)
    unadj = close.astype(np.float64).copy()
    # Force security 0 below $5 unadjusted on session 320 -> must not be eligible on 320!
    unadj[320, 0] = 3.50
    in_span = np.ones((n_sess, n_sec), dtype=bool)

    elig_full = wider_universe.compute_pit_liquidity_eligibility(
        close, volume, unadj, in_span, sessions, top_n=10,
    )
    first_dec = int(np.searchsorted(sessions, np.datetime64("2018-05-29", "D"), side="left"))
    assert not bool(elig_full[:first_dec].any())
    assert bool((elig_full[first_dec:].sum(axis=1) <= 10).all())
    assert not bool(elig_full[320, 0])

    cut = 380
    elig_trunc = wider_universe.compute_pit_liquidity_eligibility(
        close[: cut + 1], volume[: cut + 1], unadj[: cut + 1], in_span[: cut + 1], sessions[: cut + 1], top_n=10,
    )
    np.testing.assert_array_equal(elig_trunc, elig_full[: cut + 1])

    p_close = close.copy()
    p_vol = volume.copy()
    p_raw = unadj.copy()
    p_close[cut + 1 :] = 99999.0
    p_vol[cut + 1 :] = 1e12
    p_raw[cut + 1 :] = 0.25
    elig_poison = wider_universe.compute_pit_liquidity_eligibility(
        p_close, p_vol, p_raw, in_span, sessions, top_n=10,
    )
    np.testing.assert_array_equal(elig_poison[: cut + 1], elig_full[: cut + 1])


def test_parse_submissions_resolves_delisted_doc_prefix_otc_q_and_filters_sealed_8k() -> None:
    fake_sec = MagicMock()
    payload = {
        "name": "AKORN INC",
        "sic": "2834",
        "tickers": [],
        "exchanges": [],
        "filings": {
            "recent": {
                "form": ["10-Q", "8-K", "8-K"],
                "filingDate": ["2020-08-07", "2020-05-11", "2024-06-10"],
                "accessionNumber": ["0001-20-000001", "0001-20-000002", "0001-24-000099"],
                "acceptanceDateTime": ["2020-08-07T16:30:00.000Z", "2020-05-11T08:00:00.000Z", "2024-06-10T12:00:00.000Z"],
                "items": [[], ["2.02", "7.01"], ["2.02"]],
                "primaryDocument": ["akrx-20200630.htm", "d8k.htm", "akrx-20240610.htm"],
            }
        },
    }
    parsed = wider_universe._parse_submissions_payload(
        fake_sec, "0000003116", payload, development_end=date(2024, 5, 28),
        fetch_overflow_8k=False, allow_form4_xml=False,
    )
    assert parsed["ticker"] == "AKRX"
    assert parsed["sector"] == "Health Care"
    assert parsed["last_filing_date"] == "2020-08-07"
    assert len(parsed["events_8k"]) == 1
    assert parsed["events_8k"][0]["accession_number"] == "0001-20-000002"

    otc_payload = {
        "name": "BED BATH & BEYOND",
        "sic": "5700",
        "tickers": ["BBBYQ"],
        "exchanges": ["OTC"],
        "filings": {"recent": {"form": [], "filingDate": [], "accessionNumber": []}},
    }
    parsed_otc = wider_universe._parse_submissions_payload(
        fake_sec, "0000886158", otc_payload, development_end=date(2024, 5, 28),
        fetch_overflow_8k=False, allow_form4_xml=False,
    )
    assert parsed_otc["ticker"] == "BBBY"
    assert parsed_otc["sector"] == "Consumer Discretionary"


def test_wider_catalyst_reaction_features_after_hours_clock_and_future_poison(tmp_path: Path) -> None:
    panel = synthetic_panel(sessions=340, securities=18, seed=19)
    sec_ids = tuple(f"cik:{i + 1:010d}" for i in range(len(panel.security_ids)))
    panel_elig = replace(
        panel,
        security_ids=sec_ids,
        eligible=np.ones_like(panel.eligible),
        development_end=pd.Timestamp(panel.sessions[-1]).date(),
    )

    day_290 = pd.Timestamp(panel_elig.sessions[290]).date()
    after_close_290 = pd.Timestamp(f"{day_290.isoformat()} 16:05:00", tz=MARKET_TIMEZONE).tz_convert("UTC")
    events_8k = pd.DataFrame({
        "security_id": [sec_ids[0], sec_ids[1]],
        "accession_number": ["0001-19-000001", "0001-19-000002"],
        "item_codes": ["1.01,2.02", "5.02"],
        "accepted_at_utc": [after_close_290, after_close_290 - pd.Timedelta(days=10)],
        "available_at_utc": [after_close_290, after_close_290 - pd.Timedelta(days=10)],
    })

    empty_news = tmp_path / "empty_news"
    empty_news.mkdir()
    feats_full, rep = wider_universe.build_wider_catalyst_reaction_features(
        panel_elig, events_8k, early_news_dir=empty_news, later_news_dir=empty_news,
    )
    assert rep["total_mapped_catalyst_tuples"] == 3
    # 16:05 ET on row 290 maps to row 291 -> row 290 has 0.0 on col 0, row 291 is non-zero!
    assert feats_full["rx_8k_101_material_agreement"][290, 0] == pytest.approx(0.0, abs=1e-7)
    assert feats_full["rx_8k_101_material_agreement"][291, 0] != 0.0

    p_close = panel_elig.close.copy()
    p_vol = panel_elig.volume.copy()
    p_close[300:] = POST_END_PRICE
    p_vol[300:] = 1e12
    poisoned = replace(panel_elig, close=p_close, volume=p_vol)
    feats_poison, _ = wider_universe.build_wider_catalyst_reaction_features(
        poisoned, events_8k, early_news_dir=empty_news, later_news_dir=empty_news,
    )
    for name in wider_universe.WIDER_STAGE2_FEATURE_NAMES:
        np.testing.assert_allclose(feats_poison[name][:300], feats_full[name][:300], rtol=0, atol=0, equal_nan=True)
