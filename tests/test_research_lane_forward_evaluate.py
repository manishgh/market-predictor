"""Unit tests for Step 5h: prospective forward-score & OPRA option maturation evaluator."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import SecretStr

from market_predictor.config import Settings
from market_predictor.core.errors import ArtifactIntegrityError
from market_predictor.research_lane import forward_record
from market_predictor.research_lane.forward_evaluate import evaluate_forward_scores
from market_predictor.research_lane.forward_record import FORWARD_RECORD_DIRECTORY
from market_predictor.research_lane.panel import ResearchPanel, SealedPeriodError, file_sha256


def _write_snapshot(
    base_dir: Path,
    snap_date: str,
    *,
    backwardation: bool,
    veto_last: bool = False,
    published_at_utc: str | None = None,
    record_mode: str = "prospective",
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
    cast(Any, pq).write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), snap_dir / "scores.parquet")
    default_pub = f"{snap_date}T22:00:00+00:00"
    pub = published_at_utc if published_at_utc is not None else default_pub
    manifest = {
        "scored_session": snap_date,
        "record_mode": record_mode,
        "published_at_utc": pub,
        "recorded_at_utc": pub,
        "options_summary": {
            "opra_enabled": True,
            "benchmark_observation_kind": "synchronized_close",
            "opra_regime": "backwardation" if backwardation else "contango",
            "opra_backwardation": backwardation,
            "spy_iv_term_30_90": 1.08 if backwardation else 0.94,
            "qqq_iv_term_30_90": 1.05 if backwardation else 0.95,
        },
    }
    (snap_dir / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_forward_evaluator_maturation_vs_pending_and_sealed_guard(tmp_path: Path) -> None:
    cfg = Settings(_env_file=None, ALPACA_API_KEY_ID="k", ALPACA_API_SECRET_KEY=SecretStr("s"))
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


def test_r1_post_entry_and_retrospective_replay_snapshots_rejected_from_prospective_pool(tmp_path: Path) -> None:
    cfg = Settings(_env_file=None, ALPACA_API_KEY_ID="k", ALPACA_API_SECRET_KEY=SecretStr("s"))
    fdir = tmp_path / FORWARD_RECORD_DIRECTORY
    # 2026-11-18 recorded AFTER 2026-11-19 09:30 ET (14:30 UTC) open -> must be flagged recorded_after_entry & excluded!
    _write_snapshot(
        fdir,
        "2026-11-18",
        backwardation=False,
        published_at_utc="2026-11-19T17:13:57+00:00",
        record_mode="prospective",
    )
    # 2026-11-19 recorded as retrospective_replay -> excluded from prospective pool even if timestamp precedes open!
    _write_snapshot(
        fdir,
        "2026-11-19",
        backwardation=False,
        published_at_utc="2026-11-19T22:00:00+00:00",
        record_mode="retrospective_replay",
    )

    with patch("market_predictor.research_lane.forward_evaluate._fetch_post_holdout_bars") as mock_bars, patch(
        "market_predictor.research_lane.forward_evaluate.guard"
    ), patch.dict("os.environ", {"MARKET_PREDICTOR_RUNTIME_DIR": str(tmp_path / "rt")}):
        res = evaluate_forward_scores(
            root=tmp_path,
            as_of_date=date(2026, 11, 20),
            horizons=(1,),
            settings=cfg,
        )
        # Zero admitted prospective snapshots -> zero bar fetches & zero matured pairs!
        assert mock_bars.call_count == 0
        assert res["snapshots_recorded_after_entry_dates"] == ["2026-11-18"]
        assert res["matured_pairs"] == 0
        assert res["rejected_pairs"] == 2
        doc = json.loads((tmp_path / res["output"] / "report.json").read_text(encoding="utf-8"))
        assert doc["pooled_signals_by_horizon"] == {}


def test_r1_retained_november_snapshots_stay_untouched_and_zero_qualify_for_sec_model(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    real_records = repo_root / FORWARD_RECORD_DIRECTORY
    if not real_records.is_dir():
        pytest.skip("retained forward_records not present in checkout")

    expected_days = ("2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20")
    pre_hashes: dict[str, str] = {}
    for d_str in expected_days:
        for fname in ("_manifest.json", "scores.parquet"):
            fpath = real_records / d_str / fname
            assert fpath.is_file(), f"missing retained snapshot file {fpath}"
            pre_hashes[f"{d_str}/{fname}"] = file_sha256(fpath)

    # Evaluate retained snapshots into a scratch output folder under tmp_path without touching repo data!
    # Copy retained bytes read-only into tmp_path's forward_records to avoid writing anything under repo_root,
    # while ALSO running audit_snapshot_admission directly on `real_records` to verify real paths & preserve bytes.
    from market_predictor.research_lane.prospective_contracts import audit_snapshot_admission

    audits = []
    for d_str in expected_days:
        s_dir = real_records / d_str
        s_df = cast(Any, pq).read_table(s_dir / "scores.parquet", use_threads=False).to_pandas()
        m_js = json.loads((s_dir / "_manifest.json").read_text(encoding="utf-8"))
        audits.append(audit_snapshot_admission(s_dir, s_df, m_js))

    after_entry = [a["snapshot_date"] for a in audits if a["clock_status"] == "recorded_after_entry"]
    before_entry = [a["snapshot_date"] for a in audits if a["published_before_entry"]]
    sec_qualified = [a["snapshot_date"] for a in audits if a["qualifies_for_persisted_sec_model"]]

    assert after_entry == ["2026-11-16", "2026-11-17", "2026-11-18"]
    assert before_entry == ["2026-11-19", "2026-11-20"]
    assert sec_qualified == []
    assert all(a["unknown_sector_rows"] == 1000 for a in audits)

    # Verify original retained bytes are 100% untouched
    for d_str in expected_days:
        for fname in ("_manifest.json", "scores.parquet"):
            assert file_sha256(real_records / d_str / fname) == pre_hashes[f"{d_str}/{fname}"]
    _ = datetime(2026, 11, 20, tzinfo=UTC)


def test_r9_edited_score_file_rejected_and_swapped_model_cannot_be_graded_under_original_identity(
    tmp_path: Path,
) -> None:
    import numpy as np

    cfg = Settings(_env_file=None, ALPACA_API_KEY_ID="k", ALPACA_API_SECRET_KEY=SecretStr("s"))
    sessions = pd.bdate_range("2026-07-15", "2026-11-19").to_numpy().astype("datetime64[D]")
    n_s, n_c = len(sessions), 6
    close = np.full((n_s, n_c), 60.0, dtype=np.float32) * np.linspace(1.0, 1.12, n_s, dtype=np.float32)[:, None]
    spy_c = np.full(n_s, 520.0, dtype=np.float32)
    panel = ResearchPanel(
        sessions=sessions,
        security_ids=tuple(f"cik:{i + 1:010d}" for i in range(n_c)),
        open=close,
        high=close * np.float32(1.01),
        low=close * np.float32(0.99),
        close=close,
        volume=np.full((n_s, n_c), 8e5, dtype=np.float32),
        eligible=np.ones((n_s, n_c), dtype=bool),
        sector=np.zeros((n_s, n_c), dtype=np.int16),
        sector_names=("Industrials",),
        spy_open=spy_c,
        spy_close=spy_c,
        qqq_open=spy_c,
        qqq_close=spy_c,
        development_end=date(2026, 11, 19),
        load_report={},
    )
    with patch(
        "market_predictor.research_lane.forward_record._fetch_live_prospective_panel",
        return_value=panel,
    ), patch("market_predictor.research_lane.forward_record.guard"), patch(
        "market_predictor.research_lane.forward_evaluate.guard"
    ), patch.dict("os.environ", {"MARKET_PREDICTOR_RUNTIME_DIR": str(tmp_path / "rt")}):
        m_first = forward_record.record_forward_scores(
            root=tmp_path,
            as_of_date=date(2026, 11, 19),
            live_prospective=True,
            include_live_catalysts=False,
            include_live_options=False,
            skip_existing=True,
            settings=cfg,
            clock_fn=lambda: datetime(2026, 11, 19, 22, 0, tzinfo=UTC),
        )
        assert m_first["status"] == "recorded"
        assert "scores.parquet" in m_first["payload_artifacts"]

        # 1. Identical --skip-existing retry succeeds once cleanly
        m_retry = forward_record.record_forward_scores(
            root=tmp_path,
            as_of_date=date(2026, 11, 19),
            live_prospective=True,
            include_live_catalysts=False,
            include_live_options=False,
            skip_existing=True,
            settings=cfg,
            clock_fn=lambda: datetime(2026, 11, 19, 22, 5, tzinfo=UTC),
        )
        assert m_retry["status"] == "already_recorded"

        # 2. Editing scores.parquet (valid Parquet, altered score value) MUST raise ArtifactIntegrityError on BOTH
        #    record_forward_scores(--skip-existing) AND evaluate_forward_scores!
        scores_pq = tmp_path / FORWARD_RECORD_DIRECTORY / "2026-11-19" / "scores.parquet"
        orig_pq_bytes = scores_pq.read_bytes()
        tampered_df = cast(Any, pq).read_table(scores_pq, use_threads=False).to_pandas()
        tampered_df.loc[0, "score_price_equal_12"] = 0.999999
        cast(Any, pq).write_table(pa.Table.from_pandas(tampered_df, preserve_index=False), scores_pq)

        with pytest.raises(ArtifactIntegrityError):
            forward_record.record_forward_scores(
                root=tmp_path,
                as_of_date=date(2026, 11, 19),
                live_prospective=True,
                include_live_catalysts=False,
                include_live_options=False,
                skip_existing=True,
                settings=cfg,
                clock_fn=lambda: datetime(2026, 11, 19, 22, 10, tzinfo=UTC),
            )
        with pytest.raises(ArtifactIntegrityError):
            evaluate_forward_scores(
                root=tmp_path,
                as_of_date=date(2026, 11, 20),
                horizons=(1,),
                settings=cfg,
            )

        # Restore valid scores.parquet, then simulate a snapshot claiming a DIFFERENT valid model manifest SHA-256:
        # audit_snapshot_admission must reject it (`model_manifest_sha256_mismatch`) under original candidate identity!
        scores_pq.write_bytes(orig_pq_bytes)
        from market_predictor.research_lane.prospective_contracts import audit_snapshot_admission

        snap_dir = tmp_path / FORWARD_RECORD_DIRECTORY / "2026-11-19"
        man_doc = json.loads((snap_dir / "_manifest.json").read_text(encoding="utf-8"))
        man_doc["catalyst_summary"] = {
            "sec_xgb_model": {
                "model_scored": True,
                "manifest_sha256": "f" * 64,
                "feature_contract_sha256": "d244cea55bde7b79c22167da6bc83ad315f5165b33fbb3a39826b32d022b868e",
            }
        }
        valid_df = cast(Any, pq).read_table(scores_pq, use_threads=False).to_pandas()
        valid_df["score_mid_small_bn_sec_xgb_10d"] = 0.15
        aud = audit_snapshot_admission(snap_dir, valid_df, man_doc)
        assert aud["qualifies_for_persisted_sec_model"] is False
        reasons = aud["candidate_admissions"]["mid_small_bn_sec_xgb_10d"]["rejection_reasons"]
        assert any("model_manifest_sha256_mismatch" in r for r in reasons)


def test_r5_removing_highest_ranked_exit_preserves_selected_holdings_and_marks_incomplete(tmp_path: Path) -> None:
    """Reproduce the 20-stock `-50%` vs `+1%` example: missing top exit never replaces holding or fabricates `+0.8%` (`R5`)."""
    import numpy as np

    from market_predictor.research_lane.forward_evaluate import _score_single_signal_cohort
    from market_predictor.research_lane.prospective_contracts import (
        PERSISTED_SEC_FEATURE_CONTRACT_SHA256,
        PERSISTED_SEC_MODEL_MANIFEST_SHA256,
    )

    sids = tuple(f"cik:{i + 1:010d}" for i in range(30))
    # First test exact 20-stock cohort: S00 = -50%, S01..S19 = +1%
    sids_20 = sids[:20]
    sig_20 = np.linspace(0.5, -0.5, 20, dtype=np.float64)
    rets_complete = np.full(20, 0.01, dtype=np.float64)
    rets_complete[0] = -0.50

    res_ok = _score_single_signal_cohort(sig_20, rets_complete, 0.005, security_ids=sids_20)
    assert res_ok is not None
    assert res_ok["top10_complete"] is True
    assert res_ok["selected_security_ids"] == ["cik:0000000001", "cik:0000000002"]
    assert res_ok["top10_gross_return"] == pytest.approx(-0.245, abs=1e-12)
    assert res_ok["top10_net_20bps_return"] == pytest.approx(-0.247, abs=1e-12)

    # Remove ONLY the highest-scored stock's future exit (`NaN`):
    # Selected identities MUST stay `["cik:0000000001", "cik:0000000002"]` and cohort return MUST be `None` (`incomplete`)!
    rets_missing_top = rets_complete.copy()
    rets_missing_top[0] = np.nan
    res_inc = _score_single_signal_cohort(sig_20, rets_missing_top, 0.005, security_ids=sids_20)
    assert res_inc is not None
    assert res_inc["selected_security_ids"] == ["cik:0000000001", "cik:0000000002"]
    assert res_inc["top10_complete"] is False
    assert res_inc["selected_unresolved_outcomes"] == 1
    assert res_inc["top10_gross_return"] is None
    assert res_inc["top10_net_20bps_return"] is None
    assert res_inc["rank_ic_scope"] == "observed_subset_diagnostic"
    assert res_inc["observed_subset_rank_ic"] is not None

    # Also verify `build_snapshot_decision_holdings` freezes target weights (`3` stocks of `30` = `1/3` each) from decision inputs only!
    scores_df = pd.DataFrame({
        "session_date": ["2026-11-20"] * 30,
        "security_id": list(sids),
        "ticker": [f"T{i:02d}" for i in range(30)],
        "sector": ["Industrials"] * 30,
        "score_mid_small_bn_sec_xgb_10d": np.linspace(0.5, -0.5, 30),
        "score_mid_small_bn_sec_xgb_21d": np.linspace(0.5, -0.5, 30),
        "score_mid_small_bn_sec_xgb_opra_10d": [None] * 30,
    })
    h_df, h_sum = forward_record.build_snapshot_decision_holdings(
        scores_df,
        panel_sessions=np.asarray(["2026-11-20"], dtype="datetime64[D]"),
        dest_parent=tmp_path,
        options_summary=None,
        catalyst_summary={
            "sec_xgb_model": {
                "model_scored": True,
                "manifest_sha256": PERSISTED_SEC_MODEL_MANIFEST_SHA256,
                "feature_contract_sha256": PERSISTED_SEC_FEATURE_CONTRACT_SHA256,
            }
        },
    )
    c10 = h_df.loc[h_df["candidate_id"] == "mid_small_bn_sec_xgb_10d"].reset_index(drop=True)
    assert not bool(c10["abstained"].any())
    assert c10["security_id"].tolist() == ["cik:0000000001", "cik:0000000002", "cik:0000000003"]
    assert np.allclose(c10["target_weight"].to_numpy(dtype=np.float64), 1.0 / 3.0)
    # OPRA candidate abstains with explicit reason because OPRA was not collected
    copra = h_df.loc[h_df["candidate_id"] == "mid_small_bn_sec_xgb_opra_10d"].iloc[0]
    assert bool(copra["abstained"]) is True
    assert "opra_observation_not_synchronized_close" in str(copra["abstain_reason"])
    assert h_sum["mid_small_bn_sec_xgb_10d"]["selected_equities"] == 3
