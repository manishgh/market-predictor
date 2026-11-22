"""Unit, ticker-reuse quarantine, and causality tests for Step 5b (`wider_universe`)."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import SecretStr

from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.research_lane import live_cohort, wider_universe
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


def _seed_relocated_cohort_authorities(ext_data_root: Path, sessions: np.ndarray) -> None:
    """Create a complete relocated `--data-root` fixture with S&P 500 exclusion (`AAPL`) & ticker rename (`OLDT->NEWT`)."""
    panel_prof = (
        ext_data_root
        / "features/edge_rebuild_swing_panel_20180529_20260708_v2"
        / "final/panel/feature_profile=technical_market/year=2026"
    )
    panel_prof.mkdir(parents=True, exist_ok=True)
    mem_df = pd.DataFrame([
        {
            "ticker": "AAPL",
            "security_id": "sec_sp500:AAPL",
            "sector": "Information Technology",
            "session_date_et": date(2026, 7, 8),
            "decision_time_utc": pd.Timestamp("2026-07-08T22:00:00Z"),
            "membership_available_at_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
            "cross_section_eligible": True,
            "pre_catalyst_feature_eligible": True,
        }
    ])
    cast(Any, pq).write_table(pa.Table.from_pandas(mem_df, preserve_index=False), panel_prof / "part.parquet")

    # Observed S&P 500 membership authority covering through 2026-11-19 (observed BEFORE 2026-11-19 close!)
    obs_dir = ext_data_root / "raw/index_membership/sp500_observed_20261119T120000Z_v1"
    obs_dir.mkdir(parents=True, exist_ok=True)
    obs_mem = pd.DataFrame([
        {
            "ticker": "AAPL",
            "security_id": "sec_sp500:AAPL",
            "effective_from_utc": pd.Timestamp("2018-05-29T04:00:00Z"),
            "effective_to_utc": pd.NaT,
            "available_at_utc": pd.Timestamp("2018-05-29T04:00:00Z"),
        }
    ])
    obs_mem["effective_to_utc"] = pd.to_datetime(obs_mem["effective_to_utc"], utc=True)
    cast(Any, pq).write_table(pa.Table.from_pandas(obs_mem, preserve_index=False), obs_dir / "memberships.parquet")
    (obs_dir / "_manifest.json").write_text(
        json.dumps({
            "status": "complete",
            "observed_at_utc": "2026-11-19T12:00:00+00:00",
            "effective_horizon_date": "2026-11-19",
        }),
        encoding="utf-8",
    )

    sec_id_dir = ext_data_root / "canonical/sec_identity/sec_identity_20190709_20260708_v2"
    sec_id_dir.mkdir(parents=True, exist_ok=True)
    rel_df = pd.DataFrame([
        {
            "security_id": "sec_sp500:AAPL",
            "sec_cik": "0000320193",
            "available_at_utc": pd.Timestamp("2019-01-01T00:00:00Z"),
        }
    ])
    cast(Any, pq).write_table(pa.Table.from_pandas(rel_df, preserve_index=False), sec_id_dir / "sec_identity_relations.parquet")

    filings_dir = ext_data_root / "research/swing_initial_fit_sec_form_inventory_corrected_clock"
    filings_dir.mkdir(parents=True, exist_ok=True)
    filings_df = pd.DataFrame({
        "security_id": pd.Series(dtype="str"),
        "sec_cik": pd.Series(dtype="str"),
        "accession_number": pd.Series(dtype="str"),
        "sec_form": pd.Series(dtype="str"),
        "item_codes": pd.Series(dtype="str"),
        "accepted_at_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "available_at_utc": pd.Series(dtype="datetime64[ns, UTC]"),
    })
    cast(Any, pq).write_table(pa.Table.from_pandas(filings_df, preserve_index=False), filings_dir / "filings.parquet")

    for sub_p in (
        "research/investment_financial_coverage_20261107/sources",
        "research/research_lane/sp500_free_backfill/company_facts",
    ):
        (ext_data_root / sub_p).mkdir(parents=True, exist_ok=True)

    wider_dir = ext_data_root / "research/research_lane/wider_us_universe"
    (wider_dir / "company_facts").mkdir(parents=True, exist_ok=True)
    # 3 CIKs:
    # - 0000320193 (`AAPL`, S&P 500 member -> MUST BE EXCLUDED!)
    # - 0001000001 (`OLDT` before rename_day, `NEWT` from rename_day onward -> MUST STITCH & BE ELIGIBLE!)
    # - 0001000002 (`MIDB`, steady mid-cap -> MUST BE ELIGIBLE!)
    rename_day = pd.Timestamp(sessions[150]).date()
    first_day = pd.Timestamp(sessions[0]).date()
    roster_rows = [
        {"cik": "0000320193", "ticker": "AAPL", "valid_from": first_day, "valid_to": date(2024, 5, 28), "source_priority": 1},
        {"cik": "0001000001", "ticker": "OLDT", "valid_from": first_day, "valid_to": date(2024, 5, 28), "source_priority": 2},
        {"cik": "0001000002", "ticker": "MIDB", "valid_from": first_day, "valid_to": date(2024, 5, 28), "source_priority": 2},
    ]
    cast(Any, pq).write_table(pa.Table.from_pandas(pd.DataFrame(roster_rows), preserve_index=False), wider_dir / "roster_spans.parquet")
    (wider_dir / "selected_ciks.json").write_text(
        json.dumps({"selected_ciks": ["0000320193", "0001000001", "0001000002"]}),
        encoding="utf-8",
    )
    (wider_dir / "cik_sectors.json").write_text(
        json.dumps({
            "0000320193": "Information Technology",
            "0001000001": "Health Care",
            "0001000002": "Industrials",
        }),
        encoding="utf-8",
    )

    ext_dir = ext_data_root / "research/research_lane/sealed_holdout_extension_20240529_20260708"
    ext_dir.mkdir(parents=True, exist_ok=True)
    ext_roster_rows = [
        {"cik": "0000320193", "ticker": "AAPL", "valid_from": first_day, "valid_to": date(2026, 11, 19), "source_priority": 1},
        {
            "cik": "0001000001",
            "ticker": "OLDT",
            "valid_from": first_day,
            "valid_to": pd.Timestamp(sessions[149]).date(),
            "source_priority": 2,
        },
        {"cik": "0001000001", "ticker": "NEWT", "valid_from": rename_day, "valid_to": date(2026, 11, 19), "source_priority": 2},
        {"cik": "0001000002", "ticker": "MIDB", "valid_from": first_day, "valid_to": date(2026, 11, 19), "source_priority": 2},
    ]
    cast(Any, pq).write_table(
        pa.Table.from_pandas(pd.DataFrame(ext_roster_rows), preserve_index=False),
        ext_dir / "extended_roster_spans.parquet",
    )


def test_r2_relocated_data_root_cohort_matches_cik_sectors_excludes_sp500_stitches_rename_and_fails_early_coverage(
    tmp_path: Path,
) -> None:
    empty_project_root = tmp_path / "empty_checkout"
    empty_project_root.mkdir()
    ext_data_root = tmp_path / "relocated_data_root"
    ext_data_root.mkdir()

    sessions = pd.bdate_range("2025-10-15", "2026-11-19").to_numpy().astype("datetime64[D]")
    assert len(sessions) >= 260
    _seed_relocated_cohort_authorities(ext_data_root, sessions)
    paths = EvidencePaths(empty_project_root, ext_data_root)

    bar_rows: list[dict[str, Any]] = []
    for idx_s, s_np in enumerate(sessions):
        s_iso = str(s_np)
        bar_rows.extend([
            {"ticker": "SPY", "session_date": s_iso, "open": 550.0, "high": 552.0, "low": 549.0, "close": 551.0, "volume": 5e7},
            {"ticker": "QQQ", "session_date": s_iso, "open": 480.0, "high": 482.0, "low": 479.0, "close": 481.0, "volume": 4e7},
            {"ticker": "AAPL", "session_date": s_iso, "open": 220.0, "high": 222.0, "low": 219.0, "close": 221.0, "volume": 4e7},
            {"ticker": "MIDB", "session_date": s_iso, "open": 45.0, "high": 46.0, "low": 44.0, "close": 45.5, "volume": 1e6},
        ])
        tk_ren = "OLDT" if idx_s < 150 else "NEWT"
        bar_rows.append({"ticker": tk_ren, "session_date": s_iso, "open": 60.0, "high": 61.0, "low": 59.0, "close": 60.5, "volume": 1.2e6})
    all_bars_df = pd.DataFrame(bar_rows)

    cfg = Settings(_env_file=None, ALPACA_API_KEY_ID="k", ALPACA_API_SECRET_KEY=SecretStr("s"))
    with patch(
        "market_predictor.research_lane.live_cohort._fetch_alpaca_daily_batch",
        side_effect=lambda alpaca, symbols, start_day, end_day, **kw: all_bars_df.loc[
            all_bars_df["ticker"].isin(symbols)
        ].copy(),
    ):
        cohort = live_cohort.load_decision_date_mid_small_cohort(
            paths,
            as_of_date=date(2026, 11, 19),
            source_cutoff_utc=datetime(2026, 11, 19, 21, 0, tzinfo=UTC),
            live_prospective=True,
            unseal_final_test_period=True,
            settings=cfg,
        )

    # 1. Canonical cik: security IDs, S&P 500 constituent AAPL (`0000320193`) excluded, renamed issuer (`0001000001`) stitched!
    assert cohort.panel.security_ids == ("cik:0001000001", "cik:0001000002")
    assert "cik:0000320193" not in cohort.panel.security_ids
    assert bool(cohort.panel.eligible[-1].all())
    assert cohort.active_ticker_by_security_id == {
        "cik:0001000001": "NEWT",
        "cik:0001000002": "MIDB",
    }
    # 2. Exact sector assignments from cik_sectors.json (`0` Unknown!)
    sec_by_sid = {
        sid: cohort.panel.sector_names[int(cohort.panel.sector[-1, idx])]
        for idx, sid in enumerate(cohort.panel.security_ids)
    }
    assert sec_by_sid == {
        "cik:0001000001": "Health Care",
        "cik:0001000002": "Industrials",
    }

    # 3. Requesting 2026-11-20 (beyond 2026-11-19 membership & roster coverage) MUST fail closed naming missing authority & date!
    with pytest.raises(DataReadinessError, match="2026-11-20"):
        live_cohort.verify_mid_small_cohort_authorities(
            paths,
            as_of_date=date(2026, 11, 20),
            source_cutoff_utc=datetime(2026, 11, 20, 21, 0, tzinfo=UTC),
            unseal_final_test_period=True,
        )


def test_r2_real_repository_authorities_report_exact_post_august_membership_gap() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "data/research/research_lane/wider_us_universe/roster_spans.parquet").is_file():
        pytest.skip("retained wider_us_universe authorities not present in checkout")

    paths = EvidencePaths(repo_root)
    # Retained 2024-05-28 & 2026-06-03 pass authority checks cleanly using roster_spans.parquet / extended_roster_spans.parquet
    _, _, rep_dev = live_cohort.verify_mid_small_cohort_authorities(
        paths,
        as_of_date=date(2024, 5, 28),
        source_cutoff_utc=datetime(2024, 5, 28, 20, 0, tzinfo=UTC),
        unseal_final_test_period=False,
    )
    assert rep_dev["roster_authority"].endswith("roster_spans.parquet")
    assert rep_dev["roster_covered_through"] == "2024-05-28"

    _, _, rep_hold = live_cohort.verify_mid_small_cohort_authorities(
        paths,
        as_of_date=date(2026, 6, 3),
        source_cutoff_utc=datetime(2026, 6, 3, 20, 0, tzinfo=UTC),
        unseal_final_test_period=True,
    )
    assert rep_hold["roster_authority"].endswith("extended_roster_spans.parquet")
    assert rep_hold["sp500_covered_through"] == "2026-06-03"

    # Live 2026-11-20 fails closed naming the exact retained 2026-08-21 S&P 500 & 2026-07-08 roster gap!
    with pytest.raises(DataReadinessError, match="2026-08-21.*2026-11-20"):
        live_cohort.verify_mid_small_cohort_authorities(
            paths,
            as_of_date=date(2026, 11, 20),
            source_cutoff_utc=datetime(2026, 11, 20, 21, 0, tzinfo=UTC),
            unseal_final_test_period=True,
        )
