import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_predictor.primary_v2.contracts import INTRADAY_V2_ID, SWING_V2_ID
from market_predictor.primary_v2.experiments import _json_sha256
from market_predictor.primary_v2.failure_attribution import (
    _join_validation_source,
    _load_complete_audit,
    _publish_audit,
    build_cohort_evidence,
    build_replicated_viability,
)
from market_predictor.primary_v2.failure_attribution_contracts import (
    load_failure_attribution_config,
)
from market_predictor.v3.errors import DataReadinessError

CONFIG = load_failure_attribution_config(
    Path("configs/primary_v2_failure_attribution.toml")
)


def test_swing_cohorts_require_and_find_replication_in_both_scopes() -> None:
    rows = _swing_rows()

    evidence = build_cohort_evidence(
        rows,
        strategy_id=SWING_V2_ID,
        config=CONFIG,
    )
    replicated = build_replicated_viability(
        evidence,
        strategy_id=SWING_V2_ID,
        config=CONFIG,
    )

    assert set(evidence["dimension"]) == set(CONFIG.swing.dimensions)
    assert set(evidence["evaluation_phase"]) == {0, 1, 2, 3, 4}
    assert len(replicated) == 4
    assert replicated["replicated_viable"].all()
    overall = replicated.loc[replicated["dimension"].eq("overall")].iloc[0]
    assert overall["walk_forward_rows"] == 240
    assert overall["ticker_holdout_rows"] == 240
    assert overall["walk_forward_average_net_return_ci_low"] > 0
    assert overall["ticker_holdout_average_excess_return_vs_spy_ci_low"] > 0


def test_intraday_cohorts_include_fixed_time_cap_and_liquidity_dimensions() -> None:
    rows = _intraday_rows()

    evidence = build_cohort_evidence(
        rows,
        strategy_id=INTRADAY_V2_ID,
        config=CONFIG,
    )

    assert set(evidence["dimension"]) == set(CONFIG.intraday.dimensions)
    time_values = set(
        evidence.loc[evidence["dimension"].eq("time_of_day"), "cohort_value"]
    )
    assert time_values == {"midday_11_to_14_et"}
    overall = evidence.loc[evidence["dimension"].eq("overall")]
    assert np.allclose(overall["target_first_rate"], 0.75)
    assert np.allclose(overall["stop_first_rate"], 0.125)
    assert np.allclose(overall["timeout_rate"], 0.125)


def test_one_scope_failure_cannot_be_reported_as_replicated() -> None:
    rows = _swing_rows()
    holdout = rows["validation_scope"].eq("ticker_holdout")
    rows.loc[holdout, "strategy_net_return"] = -0.002
    rows.loc[holdout, "strategy_gross_return"] = -0.001
    rows.loc[holdout, "strategy_excess_return_vs_spy"] = -0.001

    evidence = build_cohort_evidence(
        rows,
        strategy_id=SWING_V2_ID,
        config=CONFIG,
    )
    replicated = build_replicated_viability(
        evidence,
        strategy_id=SWING_V2_ID,
        config=CONFIG,
    )

    assert not replicated["replicated_viable"].any()
    reasons = json.loads(
        replicated.loc[
            replicated["dimension"].eq("overall"),
            "failure_reasons_json",
        ].iloc[0]
    )
    assert any(
        reason.startswith("ticker_holdout/phase-") for reason in reasons
    )


def test_exact_identity_join_rejects_negative_or_inconsistent_cost() -> None:
    strategy = CONFIG.swing
    source = _swing_rows().iloc[:2].copy()
    source["strategy_id"] = "SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1"
    source["strategy_execution_cost_fraction"] = 0.001
    validation = source[
        [
            "strategy_dataset_row_id",
            "validation_scope",
            "fold",
            "market_regime",
        ]
    ].copy()
    validation = validation.rename(
        columns={"market_regime": "validation_market_regime"}
    )
    source = source.drop(
        columns=["validation_scope", "fold", "market_regime"]
    )
    source.loc[source.index[0], "strategy_net_return"] = 0.004

    with pytest.raises(DataReadinessError, match="gross return is below net"):
        _join_validation_source(
            source,
            validation,
            strategy=strategy,
            strategy_id=SWING_V2_ID,
            minimum_cost_bps=10,
        )


def test_immutable_audit_replay_detects_tampering(tmp_path: Path) -> None:
    request = {
        "schema": "primary_v2.failure_attribution.run.v1",
        "strategy_id": SWING_V2_ID,
    }
    request_sha256 = _json_sha256(request)
    root = tmp_path / "audit"
    manifest = _publish_audit(
        root,
        request=request,
        request_sha256=request_sha256,
        summary={"status": "no_replicated_viable_cohorts"},
        cohort_evidence=pd.DataFrame({"rows": [1]}),
        replicated=pd.DataFrame({"replicated_viable": [False]}),
    )

    assert manifest["request_sha256"] == request_sha256
    assert (
        _load_complete_audit(
            root,
            expected_request_sha256=request_sha256,
        )["status"]
        == "no_replicated_viable_cohorts"
    )
    (root / "summary.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="does not verify"):
        _load_complete_audit(
            root,
            expected_request_sha256=request_sha256,
        )


def _swing_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scope in CONFIG.validation_scopes:
        for index in range(1_200):
            net = -0.001 if index % 10 == 0 else 0.002
            records.append(
                {
                    "validation_scope": scope,
                    "fold": index % 4,
                    "market_regime": "neutral",
                    "sector": "technology",
                    "ticker": f"T{index % 20:02d}",
                    "session_date_et": (
                        pd.Timestamp("2025-01-02")
                        + pd.Timedelta(days=index // 4)
                    ).date(),
                    "strategy_dataset_row_id": f"{scope}-{index}",
                    "atr_pct_14": 0.02,
                    "strategy_gross_return": net + 0.001,
                    "strategy_net_return": net,
                    "strategy_excess_return_vs_spy": 0.0005,
                    "strategy_mfe": 0.004,
                    "strategy_mae": -0.001,
                    "stamped_round_trip_cost_fraction": 0.001,
                }
            )
    return pd.DataFrame(records)


def _intraday_rows() -> pd.DataFrame:
    swing = (
        _swing_rows()
        .groupby("validation_scope", sort=False)
        .head(240)
        .reset_index(drop=True)
        .rename(
        columns={
            "strategy_dataset_row_id": "setup_id",
            "atr_pct_14": "atr_pct",
            "strategy_gross_return": "path_realized_return_gross_30m",
            "strategy_net_return": "path_realized_return_net_30m",
            "strategy_excess_return_vs_spy": (
                "path_excess_return_30m_vs_spy"
            ),
            "strategy_mfe": "path_mfe_30m",
            "strategy_mae": "path_mae_30m",
            }
        )
    )
    swing["market_cap_bucket"] = "large"
    swing["liquidity_bucket"] = "liquid"
    swing["decision_time_utc"] = pd.Timestamp(
        "2025-01-02 17:00:00",
        tz="UTC",
    )
    pattern = np.arange(len(swing)) % 8
    swing["target_before_stop_30m"] = pattern < 6
    swing["stop_before_target_30m"] = pattern == 6
    swing["path_timeout_30m"] = pattern == 7
    swing["path_outcome_bar"] = 12
    return swing
