"""Unit tests for Step 7: persisted `mid_small_beta_neutral_sec` XGBoost bundle, batch/live parity, tamper, & OPRA."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

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
        "score_equal_rank_24_beta_neutral": base_bn[-1].tolist(),
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
    assert rep_full["sec_query_status_counts"] == {
        "queried_with_filings": 18,
        "queried_zero_filings": 18,
        "unqueried": 0,
    }
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

    # 2. R4: Partial SEC peer coverage (`24/36` queried) MUST distinguish `queried_zero_filings` vs `unqueried`
    #    and ABSTAIN (`model_scored=False`) rather than silently re-ranking among the 24-stock subset!
    queried_subset = set(panel.security_ids[:24])
    live_sub, rep_sub = sec_catalyst_model.predict_live_sec_catalyst_scores(
        panel,
        scores_df,
        all_8k_events=ev_8k,
        non8k_events=ev_non8k,
        queried_security_ids=queried_subset,
        bundle=loaded,
    )
    assert rep_sub["model_scored"] is False
    assert rep_sub["abstained"] is True
    assert "incomplete_sec_peer_coverage" in str(rep_sub["reason"])
    assert live_sub["sec_query_status"].iloc[:18].eq("queried_with_filings").all()
    assert live_sub["sec_query_status"].iloc[18:24].eq("queried_zero_filings").all()
    assert live_sub["sec_query_status"].iloc[24:].eq("unqueried").all()
    assert live_sub["score_mid_small_bn_sec_xgb_10d"].isna().all()
    assert live_sub["sec_xgb_residual_10d"].isna().all()

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
    assert [r["catalyst_family"] for r in ev_n8] == ["rx_activist_13d"]


def _issuer_facts_to_sec_json(facts_map: dict[str, Any]) -> dict[str, Any]:
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


def test_r3_r4_end_to_end_raw_source_batch_vs_live_parity_short_beta_and_future_poison(tmp_path: Path) -> None:
    """Feed raw source inputs (bars, unadjusted closes, company_facts JSONs, SEC filings) to both batch & live."""
    from dataclasses import replace as dc_replace

    from market_predictor.research_lane import forward_record, live_cohort
    from tests.support.research_lane_fundamentals import (
        poisoned_inputs,
        synthetic_fundamental_inputs,
        truncated_inputs,
    )

    syn = synthetic_fundamental_inputs()
    # Map synthetic 40-security panel onto canonical distinct `cik:00000000XX` IDs
    keep_idx = [
        idx for idx, sid in enumerate(syn.inputs.panel.security_ids)
        if sid in syn.inputs.security_issuer and idx < 34
    ]
    ciks = [syn.inputs.security_issuer[syn.inputs.panel.security_ids[i]] for i in keep_idx]
    cik_sids = tuple(f"cik:{c}" for c in ciks)
    base_p = syn.inputs.panel
    raw_panel = ResearchPanel(
        sessions=base_p.sessions.copy(),
        security_ids=cik_sids,
        open=np.ascontiguousarray(base_p.open[:, keep_idx]),
        high=np.ascontiguousarray(base_p.high[:, keep_idx]),
        low=np.ascontiguousarray(base_p.low[:, keep_idx]),
        close=np.ascontiguousarray(base_p.close[:, keep_idx]),
        volume=np.ascontiguousarray(base_p.volume[:, keep_idx]),
        eligible=np.ones((len(base_p.sessions), len(keep_idx)), dtype=bool),
        sector=np.ascontiguousarray(base_p.sector[:, keep_idx]),
        sector_names=base_p.sector_names,
        spy_open=base_p.spy_open.copy(),
        spy_close=base_p.spy_close.copy(),
        qqq_open=base_p.qqq_open.copy(),
        qqq_close=base_p.qqq_close.copy(),
        development_end=base_p.development_end,
        load_report={},
    )
    raw_unadj = np.ascontiguousarray(syn.inputs.unadjusted_close[:, keep_idx])
    sec_cik_map = {f"cik:{c}": c for c in ciks}

    facts_root = tmp_path / "company_facts_json"
    facts_paths: dict[str, Path] = {}
    for c in ciks:
        f_obj = syn.facts.get(c)
        if f_obj is not None:
            fp = facts_root / c / "company_facts.json"
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(json.dumps(_issuer_facts_to_sec_json(dict(f_obj))), encoding="utf-8")
            facts_paths[c] = fp

    # Remap releases & build SEC 8-K + non-8-K filings at session `cut = 460`
    cut = 460
    as_of_day = pd.Timestamp(raw_panel.sessions[cut]).date()
    old_to_new_sid = {
        syn.inputs.panel.security_ids[i]: f"cik:{syn.inputs.security_issuer[syn.inputs.panel.security_ids[i]]}"
        for i in keep_idx
    }
    rel_202 = syn.inputs.releases.loc[syn.inputs.releases["security_id"].isin(old_to_new_sid)].copy()
    rel_202["security_id"] = rel_202["security_id"].map(old_to_new_sid)
    cutoff_ts = pd.Timestamp(f"{as_of_day.isoformat()} 16:00:00", tz="America/New_York").tz_convert("UTC")
    rel_cut = rel_202.loc[pd.to_datetime(rel_202["available_at_utc"], utc=True) <= cutoff_ts].copy()

    ev_8k = pd.DataFrame([
        {
            "security_id": cik_sids[i],
            "sec_cik": ciks[i],
            "accession_number": f"0001-20-8k-{i:04d}",
            "sec_form": "8-K",
            "item_codes": "2.02" if i % 2 == 0 else "1.01,7.01",
            "accepted_at_utc": cutoff_ts - pd.Timedelta(days=(i % 7) + 1),
            "available_at_utc": cutoff_ts - pd.Timedelta(days=(i % 7) + 1),
        }
        for i in range(16)
    ])
    ev_non8k = pd.DataFrame([
        {
            "security_id": cik_sids[i],
            "sec_cik": ciks[i],
            "accession_number": f"0002-20-13d-{i:04d}",
            "sec_form": "SC 13D",
            "catalyst_family": "rx_activist_13d",
            "available_at_utc": cutoff_ts - pd.Timedelta(days=4),
        }
        for i in range(5)
    ])

    panel_at_cut = raw_panel.truncated(as_of_day)
    unadj_at_cut = raw_unadj[: cut + 1].copy()

    # --- Path A: Batch training & batch scoring directly from raw source inputs ---
    batch_bn_eq24, batch_beta_rk, batch_ivol_rk, batch_base_rep = sec_catalyst_model.compute_mid_small_stage1_baseline(
        panel_at_cut,
        unadjusted_close=unadj_at_cut,
        releases_202=rel_cut,
        security_cik_map=sec_cik_map,
        company_facts_paths=facts_paths,
        as_of_date=as_of_day,
    )
    assert batch_base_rep["baseline_scored"] is True
    batch_sec_feats, batch_cat_rep = beta_neutral.build_precise_sec_catalyst_features(
        panel_at_cut, ev_8k, ev_non8k, unseal_final_test_period=True,
    )
    bundle_dir = tmp_path / "models" / "r3_r4_bundle"
    sec_catalyst_model.fit_and_save_sec_catalyst_bundle_from_panel(
        panel_at_cut,
        batch_bn_eq24,
        beta_rank=batch_beta_rk,
        idio_vol_rank=batch_ivol_rk,
        catalyst_features=batch_sec_feats,
        destination=bundle_dir,
        root=tmp_path,
        input_sha256={"raw": "parity"},
        catalyst_report=batch_cat_rep,
        horizons=(10, 21),
    )
    loaded = sec_catalyst_model.verify_and_load_sec_catalyst_bundle(bundle_dir)

    # --- Path B: Live / Replay entry point fed ONLY raw source inputs (never injecting batch_bn_eq24!) ---
    cohort_obj = live_cohort.DecisionDateMidSmallCohort(
        panel=panel_at_cut,
        unadjusted_close=unadj_at_cut,
        releases_202=rel_cut,
        all_8k_events=ev_8k,
        non8k_events=ev_non8k,
        security_cik_map=sec_cik_map,
        company_facts_paths=facts_paths,
        active_ticker_by_security_id={sid: f"T{idx:02d}" for idx, sid in enumerate(cik_sids)},
        security_id_by_active_ticker={f"T{idx:02d}": sid for idx, sid in enumerate(cik_sids)},
        authority_report={},
    )
    live_scores = forward_record._score_panel_last_session(
        cohort_obj.panel,
        active_ticker_by_security_id=cohort_obj.active_ticker_by_security_id,
    )
    live_scores, live_stage1_rep = forward_record._attach_stage1_equal_rank_24_baseline(
        live_scores, cohort_obj, as_of_day,
    )
    assert live_stage1_rep["baseline_scored"] is True
    # Confirm 24-predictor baseline differs from 12-predictor price-only baseline (catching R3!)
    assert not np.allclose(
        live_scores["score_equal_rank_24_beta_neutral"].to_numpy(dtype=np.float64),
        live_scores["score_beta_neutral_price_12"].to_numpy(dtype=np.float64),
        equal_nan=True,
    )
    live_scored_df, live_model_rep = sec_catalyst_model.predict_live_sec_catalyst_scores(
        cohort_obj.panel,
        live_scores,
        all_8k_events=cohort_obj.all_8k_events,
        non8k_events=cohort_obj.non8k_events,
        queried_security_ids=set(cik_sids),
        bundle=loaded,
    )
    assert live_model_rep["model_scored"] is True

    # Verify exact agreement across IDs, all 12 ordered Stage-2 features, missingness, baseline, residuals, & ranks!
    assert tuple(live_scored_df["security_id"].tolist()) == cik_sids
    np.testing.assert_allclose(
        live_scored_df["score_equal_rank_24_beta_neutral"].to_numpy(dtype=np.float64),
        batch_bn_eq24[-1],
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    )
    x_batch_last = two_stage._design_matrix(
        batch_sec_feats,
        beta_neutral.PRECISE_STAGE2_FEATURE_NAMES,
        np.asarray([len(panel_at_cut.sessions) - 1], dtype=np.int64),
        panel_at_cut.eligible[[-1]],
    )
    for f_i, fn in enumerate(beta_neutral.PRECISE_STAGE2_FEATURE_NAMES):
        np.testing.assert_allclose(
            pd.to_numeric(live_scored_df[f"s2__{fn}"], errors="coerce").to_numpy(dtype=np.float64),
            x_batch_last[:, f_i].astype(np.float64),
            rtol=0,
            atol=1e-6,
            equal_nan=True,
        )
    for h in (10, 21):
        b_res = loaded.predict_residual(h, x_batch_last)
        b_comb = centered_rank((batch_bn_eq24[-1] + b_res)[None, :], np.isfinite(batch_bn_eq24[-1] + b_res)[None, :])[0]
        np.testing.assert_allclose(
            live_scored_df[f"sec_xgb_residual_{h}d"].to_numpy(dtype=np.float64),
            b_res,
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            live_scored_df[f"score_mid_small_bn_sec_xgb_{h}d"].to_numpy(dtype=np.float64),
            b_comb,
            rtol=0,
            atol=1e-6,
        )

    # --- R3 Short Beta History Abstention (`120` sessions < `253` required for `beta_252d`) ---
    short_end = pd.Timestamp(raw_panel.sessions[119]).date()
    short_panel = raw_panel.truncated(short_end)
    short_scores = forward_record._score_panel_last_session(short_panel)
    # Price-only beta_neutral_12 must NOT substitute 60d beta -> all None!
    assert short_scores["score_beta_neutral_price_12"].isna().all()
    _, _, _, short_rep = sec_catalyst_model.compute_mid_small_stage1_baseline(
        short_panel,
        unadjusted_close=raw_unadj[:120],
        releases_202=rel_cut,
        security_cik_map=sec_cik_map,
        company_facts_paths=facts_paths,
        as_of_date=short_end,
    )
    assert short_rep["abstained"] is True
    assert "insufficient_beta_252d_history" in str(short_rep["reason"])

    # --- Future-Data Poison Invariance (post-`cut` bars, post-`as_of_day` XBRL facts, post-cutoff 8-Ks) ---
    _ = truncated_inputs, poisoned_inputs
    poison_close = raw_panel.close.copy()
    poison_vol = raw_panel.volume.copy()
    poison_unadj = raw_unadj.copy()
    poison_close[cut + 1 :] *= np.float32(50.0)
    poison_vol[cut + 1 :] *= np.float32(100.0)
    poison_unadj[cut + 1 :] *= 50.0
    poison_panel_full = dc_replace(raw_panel, close=poison_close, volume=poison_vol)

    # Also add a post-`as_of_day` XBRL fact (`filed = as_of_day + 5d`) into company_facts JSON on disk
    poison_facts_root = tmp_path / "poison_facts"
    poison_facts_paths: dict[str, Path] = {}
    for c in ciks:
        f_obj = syn.facts.get(c)
        if f_obj is not None:
            js = _issuer_facts_to_sec_json(dict(f_obj))
            js["facts"].setdefault("us-gaap", {}).setdefault("Assets", {"units": {"USD": []}})["units"]["USD"].append({
                "end": as_of_day.isoformat(),
                "val": 9.99e14,
                "filed": (as_of_day + pd.Timedelta(days=5)).isoformat(),
                "accn": "9999-99-poison",
            })
            pfp = poison_facts_root / c / "company_facts.json"
            pfp.parent.mkdir(parents=True, exist_ok=True)
            pfp.write_text(json.dumps(js), encoding="utf-8")
            poison_facts_paths[c] = pfp

    poison_at_cut = poison_panel_full.truncated(as_of_day)
    p_bn_eq24, _, _, _ = sec_catalyst_model.compute_mid_small_stage1_baseline(
        poison_at_cut,
        unadjusted_close=poison_unadj[: cut + 1],
        releases_202=rel_cut,
        security_cik_map=sec_cik_map,
        company_facts_paths=poison_facts_paths,
        as_of_date=as_of_day,
    )
    np.testing.assert_allclose(p_bn_eq24, batch_bn_eq24, rtol=0, atol=0, equal_nan=True)
