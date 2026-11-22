"""Unit tests for Step 7: persisted `mid_small_beta_neutral_sec` XGBoost bundle, batch/live parity, tamper, & OPRA."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_predictor.core.errors import ArtifactIntegrityError
from market_predictor.research_lane import beta_neutral, sec_catalyst_model, two_stage
from market_predictor.research_lane.panel import ResearchPanel, SealedPeriodError
from market_predictor.research_lane.predictors import centered_rank


def _make_panel(n_sess: int = 380, n_sec: int = 36, seed: int = 77) -> ResearchPanel:
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2019-06-03", periods=n_sess).to_numpy().astype("datetime64[D]")
    close = (40.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.015, size=(n_sess, n_sec)), axis=0)).astype(np.float32)
    open_px = (close * np.float32(0.999)).astype(np.float32)
    spy_c = (320.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.009, size=n_sess))).astype(np.float32)
    sec_codes = np.broadcast_to((np.arange(n_sec) % 3).astype(np.int16)[None, :], (n_sess, n_sec)).copy()
    return ResearchPanel(
        sessions=sessions,
        security_ids=tuple(f"ticker:S{i:03d}" for i in range(n_sec)),
        open=open_px,
        high=close * np.float32(1.01),
        low=close * np.float32(0.99),
        close=close,
        volume=rng.uniform(4e5, 2e6, size=(n_sess, n_sec)).astype(np.float32),
        eligible=np.ones((n_sess, n_sec), dtype=bool),
        sector=sec_codes,
        sector_names=("Tech", "Health", "Ind"),
        spy_open=spy_c * np.float32(0.9995),
        spy_close=spy_c,
        qqq_open=spy_c * np.float32(0.9995),
        qqq_close=spy_c,
        development_end=pd.Timestamp(sessions[-1]).date(),
        load_report={},
    )


def test_train_command_fails_closed_without_unseal_flag(tmp_path: Path) -> None:
    with pytest.raises(SealedPeriodError, match="unseal-final-test-period"):
        sec_catalyst_model.train_and_persist_sec_catalyst_model(
            root=tmp_path,
            unseal_final_test_period=False,
        )


def test_bundle_persistence_batch_live_parity_missingness_and_tamper_detection(tmp_path: Path) -> None:
    panel = _make_panel(n_sess=380, n_sec=36, seed=19)
    rng = np.random.default_rng(19)
    n_s, n_c = panel.close.shape

    ev_8k = pd.DataFrame([
        {
            "security_id": f"ticker:S{i:03d}",
            "sec_cik": f"{i + 1:010d}",
            "accession_number": f"0001-20-{i:06d}",
            "sec_form": "8-K",
            "item_codes": "2.02" if i % 2 == 0 else "1.01",
            "accepted_at_utc": pd.Timestamp(panel.sessions[-4], tz="UTC") + pd.Timedelta(hours=12),
            "available_at_utc": pd.Timestamp(panel.sessions[-4], tz="UTC") + pd.Timedelta(hours=12),
        }
        for i in range(18)
    ])
    ev_non8k = pd.DataFrame([
        {
            "security_id": f"ticker:S{i:03d}",
            "sec_cik": f"{i + 1:010d}",
            "accession_number": f"0002-20-{i:06d}",
            "sec_form": "SC 13D",
            "catalyst_family": "rx_activist_13d",
            "available_at_utc": pd.Timestamp(panel.sessions[-8], tz="UTC") + pd.Timedelta(hours=13),
        }
        for i in range(6)
    ])

    cat_feats, cat_rep = beta_neutral.build_precise_sec_catalyst_features(
        panel, ev_8k, ev_non8k, unseal_final_test_period=True,
    )
    beta_rk = rng.uniform(-0.5, 0.5, size=(n_s, n_c)).astype(np.float32)
    ivol_rk = rng.uniform(-0.5, 0.5, size=(n_s, n_c)).astype(np.float32)
    base_bn = beta_neutral.cross_sectional_neutralize(
        rng.uniform(-0.5, 0.5, size=(n_s, n_c)), panel.eligible, panel.sector, beta_rk, ivol_rk, rerank=True,
    )

    bundle_dir = tmp_path / "models" / "mid_small_beta_neutral_sec_xgb"
    manifest = sec_catalyst_model.fit_and_save_sec_catalyst_bundle_from_panel(
        panel,
        base_bn,
        beta_rank=beta_rk,
        idio_vol_rank=ivol_rk,
        catalyst_features=cat_feats,
        destination=bundle_dir,
        root=tmp_path,
        input_sha256={"synthetic": "abc123"},
        catalyst_report=cat_rep,
        horizons=(10, 21),
    )
    assert manifest["research_only"] is True
    assert manifest["actionable"] is False
    # Verify strict maturation purging: last_matured_decision_session + h <= last session
    for h in (10, 21):
        h_rep = manifest["horizons"][str(h)]
        assert date.fromisoformat(h_rep["last_outcome_session_used"]) <= panel.development_end

    loaded = sec_catalyst_model.verify_and_load_sec_catalyst_bundle(bundle_dir)
    assert loaded.horizons == (10, 21)

    # 1. Exact batch/live parity when all 36 stocks are queried:
    scores_df = pd.DataFrame({
        "session_date": [str(panel.sessions[-1])] * n_c,
        "security_id": list(panel.security_ids),
        "score_beta_neutral_price_12": base_bn[-1].tolist(),
    })
    live_full, rep_full = sec_catalyst_model.predict_live_sec_catalyst_scores(
        panel,
        scores_df,
        all_8k_events=ev_8k,
        non8k_events=ev_non8k,
        queried_security_ids=set(panel.security_ids),
        bundle=loaded,
    )
    assert rep_full["model_scored"] is True
    x_batch_last = two_stage._design_matrix(
        cat_feats,
        beta_neutral.PRECISE_STAGE2_FEATURE_NAMES,
        np.asarray([n_s - 1], dtype=np.int64),
        panel.eligible[[n_s - 1]],
    )
    for h in (10, 21):
        batch_res = loaded.predict_residual(h, x_batch_last)
        batch_comb = centered_rank((base_bn[-1] + batch_res)[None, :], np.ones((1, n_c), dtype=bool))[0]
        np.testing.assert_allclose(
            live_full[f"sec_xgb_residual_{h}d"].to_numpy(dtype=np.float64),
            batch_res,
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            live_full[f"score_mid_small_bn_sec_xgb_{h}d"].to_numpy(dtype=np.float64),
            batch_comb,
            rtol=0,
            atol=0,
        )

    # 2. Missingness check: unqueried stocks MUST stay `None`/`NaN`, never `0.0`!
    queried_subset = set(panel.security_ids[:24])
    live_sub, _ = sec_catalyst_model.predict_live_sec_catalyst_scores(
        panel,
        scores_df,
        all_8k_events=ev_8k,
        non8k_events=ev_non8k,
        queried_security_ids=queried_subset,
        bundle=loaded,
    )
    assert live_sub["score_mid_small_bn_sec_xgb_10d"].iloc[:24].notna().all()
    assert live_sub["score_mid_small_bn_sec_xgb_10d"].iloc[24:].isna().all()
    assert live_sub["sec_xgb_residual_10d"].iloc[24:].isna().all()

    # 3. Artifact tamper checks (`ArtifactIntegrityError` on `.ubj` byte edit or feature-order edit)
    ubj_file = bundle_dir / "xgb_h10_seed0.ubj"
    orig_bytes = ubj_file.read_bytes()
    ubj_file.write_bytes(orig_bytes[:-1] + bytes([orig_bytes[-1] ^ 0xFF]))
    with pytest.raises(ArtifactIntegrityError, match="SHA-256 tamper mismatch"):
        sec_catalyst_model.verify_and_load_sec_catalyst_bundle(bundle_dir)
    ubj_file.write_bytes(orig_bytes)

    man_path = bundle_dir / "_manifest.json"
    orig_man_bytes = man_path.read_bytes()
    man_doc = json.loads(man_path.read_text(encoding="utf-8"))
    man_doc["feature_names"] = list(reversed(man_doc["feature_names"]))
    man_path.write_text(json.dumps(man_doc), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="feature_names mismatch"):
        sec_catalyst_model.verify_and_load_sec_catalyst_bundle(bundle_dir)
    man_path.write_bytes(orig_man_bytes)

    # 4. R9: A DIFFERENT internally valid model bundle at the same path cannot pass expected_manifest_sha256!
    alt_dir = tmp_path / "models" / "alt_valid_bundle"
    sec_catalyst_model.fit_and_save_sec_catalyst_bundle_from_panel(
        panel,
        base_bn,
        beta_rank=beta_rk,
        idio_vol_rank=ivol_rk,
        catalyst_features=cat_feats,
        destination=alt_dir,
        root=tmp_path,
        input_sha256={"synthetic": "different_valid_bundle"},
        catalyst_report=cat_rep,
        horizons=(10, 21),
        seeds=(7, 8, 9),
    )
    alt_loaded = sec_catalyst_model.verify_and_load_sec_catalyst_bundle(alt_dir)
    assert alt_loaded.manifest_sha256 != loaded.manifest_sha256
    with pytest.raises(ArtifactIntegrityError, match="manifest SHA-256 mismatch"):
        sec_catalyst_model.verify_and_load_sec_catalyst_bundle(
            alt_dir,
            expected_manifest_sha256=loaded.manifest_sha256,
        )


def test_opra_straddle_skew_overlay_vetoes_knives_and_boosts_shock_breakouts() -> None:
    df = pd.DataFrame({
        "security_id": ["ticker:A", "ticker:B", "ticker:C", "ticker:D"],
        "score_mid_small_bn_sec_xgb_10d": [0.40, 0.10, 0.12, -0.30],
        "falling_knife_put_skew_veto": [True, False, False, False],
        "catalyst_shock_to_straddle": [0.80, 1.80, 0.30, None],
        "latest_catalyst_rx_1d": [-0.04, 0.06, 0.04, None],
    })
    out = sec_catalyst_model.apply_opra_straddle_skew_overlay(df)
    # Stock A vetoed by 25d put skew -> None!
    assert out[0] is None
    # Stock B (shock 1.8x straddle, +6% rx) boosted above Stock C (0.3x under-reaction fade on +4% gap)!
    assert out[1] is not None and out[2] is not None and out[3] is not None
    assert out[1] > out[2] > out[3]


def test_extract_live_sec_batch_tables_strips_post_cutoff_filings() -> None:
    payload = {
        "filings": {
            "recent": {
                "form": ["8-K", "8-K", "SC 13D"],
                "filingDate": ["2026-11-18", "2026-11-20", "2026-11-17"],
                "acceptanceDateTime": [
                    "2026-11-18T13:00:00.000Z",
                    "2026-11-20T21:30:00.000Z",  # after 16:00 ET (21:00 UTC) close -> must drop!
                    "2026-11-17T14:00:00.000Z",
                ],
                "accessionNumber": ["acc-1", "acc-after-close", "acc-13d"],
                "items": [["2.02", "1.01"], ["2.02"], []],
            }
        }
    }
    ev_8k, ev_n8 = sec_catalyst_model.extract_live_sec_batch_tables_for_cik(
        payload,
        security_id="ticker:TEST",
        cik="0000123456",
        window_start_utc=datetime(2026, 8, 15, 0, 0, tzinfo=UTC),
        cutoff_utc=datetime(2026, 11, 20, 21, 0, tzinfo=UTC),
    )
    assert [r["accession_number"] for r in ev_8k] == ["acc-1"]
    assert ev_8k[0]["item_codes"] == "1.01,2.02"
    assert [r["accession_number"] for r in ev_n8] == ["acc-13d"]
    assert ev_n8[0]["catalyst_family"] == "rx_activist_13d"
