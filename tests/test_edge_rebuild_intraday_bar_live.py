from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import market_predictor.intraday.datasets.bar_live as module
from market_predictor.intraday.features.bar_features import (
    INTRADAY_BAR_FEATURE_SCHEMA_VERSION,
    INTRADAY_BAR_MODEL_FEATURE_COLUMNS,
    INTRADAY_BAR_MODEL_FEATURES_SHA256,
)
from market_predictor.intraday.datasets.bar_live import (
    INTRADAY_BAR_LIVE_ABSTENTION_COLUMNS,
    INTRADAY_BAR_LIVE_SCHEMA_VERSION,
    build_live_intraday_bar_features,
)
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.core.errors import DataReadinessError

CONTRACT_PATH = Path("configs/edge_rebuild_strategy_contract.toml")
CUTOFF = pd.Timestamp("2026-07-08T14:01:00Z")


def test_live_adapter_returns_exact_shared_ordered_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_strategy_contract(CONTRACT_PATH)
    built = _built_rows(contract.sha256())
    monkeypatch.setattr(
        module,
        "build_causal_intraday_bar_features",
        lambda *_args, **_kwargs: built.copy(),
    )
    inputs = _input_frames()

    result = build_live_intraday_bar_features(
        *inputs,
        contract=contract,
        as_of_utc=CUTOFF,
    )

    expected = built.loc[
        built["decision_time_utc"].eq(CUTOFF),
        INTRADAY_BAR_MODEL_FEATURE_COLUMNS,
    ].reset_index(drop=True)
    pdt.assert_frame_equal(result.model_features, expected)
    assert tuple(result.model_features.columns) == INTRADAY_BAR_MODEL_FEATURE_COLUMNS
    assert result.audit_identity["as_of_utc"].eq(CUTOFF).all()
    assert result.audit_identity["live_schema_version"].eq(
        INTRADAY_BAR_LIVE_SCHEMA_VERSION
    ).all()
    assert result.abstention_identity.empty
    assert tuple(result.abstention_identity.columns) == (
        INTRADAY_BAR_LIVE_ABSTENTION_COLUMNS
    )


def test_live_adapter_rejects_future_input_and_cohort_without_eligible_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_strategy_contract(CONTRACT_PATH)
    inputs = list(_input_frames())
    inputs[0] = inputs[0].copy()
    inputs[0]["available_at_utc"] = CUTOFF + pd.Timedelta(seconds=1)
    with pytest.raises(DataReadinessError, match="evidence after as_of_utc"):
        build_live_intraday_bar_features(
            *inputs,
            contract=contract,
            as_of_utc=CUTOFF,
        )

    built = _built_rows(contract.sha256())
    built["feature_eligible"] = False
    built["feature_ineligible_reason"] = "missing_exact_five_minute_bar"
    monkeypatch.setattr(
        module,
        "build_causal_intraday_bar_features",
        lambda *_args, **_kwargs: built.copy(),
    )
    with pytest.raises(DataReadinessError, match="has no eligible rows"):
        build_live_intraday_bar_features(
            *_input_frames(),
            contract=contract,
            as_of_utc=CUTOFF,
        )


def test_live_adapter_abstains_only_the_ineligible_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_strategy_contract(CONTRACT_PATH)
    built = _built_rows(contract.sha256()).iloc[::-1].reset_index(drop=True)
    rejected = built["ticker"].eq("BBB")
    built.loc[rejected, "feature_eligible"] = False
    built.loc[rejected, "feature_ineligible_reason"] = (
        "missing_exact_five_minute_bar"
    )
    monkeypatch.setattr(
        module,
        "build_causal_intraday_bar_features",
        lambda *_args, **_kwargs: built.copy(),
    )

    result = build_live_intraday_bar_features(
        *_input_frames(),
        contract=contract,
        as_of_utc=CUTOFF,
    )

    assert result.audit_identity["ticker"].tolist() == ["AAA"]
    assert len(result.model_features) == 1
    assert result.abstention_identity["ticker"].tolist() == ["BBB"]
    assert result.abstention_identity["decision_id"].tolist() == ["decision-BBB"]
    assert result.abstention_identity["feature_ineligible_reason"].tolist() == [
        "missing_exact_five_minute_bar"
    ]
    assert result.abstention_identity["as_of_utc"].eq(CUTOFF).all()
    assert result.abstention_identity["live_schema_version"].eq(
        INTRADAY_BAR_LIVE_SCHEMA_VERSION
    ).all()


def test_live_adapter_requires_exact_fixed_cohort() -> None:
    contract = load_strategy_contract(CONTRACT_PATH)
    with pytest.raises(DataReadinessError, match="not a fixed five-minute cohort"):
        build_live_intraday_bar_features(
            *_input_frames(),
            contract=contract,
            as_of_utc=CUTOFF + pd.Timedelta(minutes=1),
        )


def _built_rows(contract_sha256: str) -> pd.DataFrame:
    rows = []
    for ticker, offset in (("AAA", 0.0), ("BBB", 1.0)):
        row: dict[str, object] = {
            column: np.float32(index + offset)
            for index, column in enumerate(INTRADAY_BAR_MODEL_FEATURE_COLUMNS)
        }
        row.update(
            {
                "decision_id": f"decision-{ticker}",
                "decision_cohort_id": "cohort",
                "ticker": ticker,
                "security_id": f"security-{ticker}",
                "session_date_et": pd.Timestamp("2026-07-08").date(),
                "decision_time_utc": CUTOFF,
                "source_feature_available_at_utc": CUTOFF,
                "feature_available_at_utc": CUTOFF,
                "primary_benchmark": "XLK",
                "universe_snapshot_id": "snapshot",
                "strategy_contract_sha256": contract_sha256,
                "feature_schema_version": INTRADAY_BAR_FEATURE_SCHEMA_VERSION,
                "ordered_feature_sha256": INTRADAY_BAR_MODEL_FEATURES_SHA256,
                "feature_eligible": True,
                "feature_ineligible_reason": pd.NA,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _input_frames() -> tuple[pd.DataFrame, ...]:
    minute = pd.Timestamp("2026-07-08T14:00:00Z")
    common = pd.DataFrame(
        {
            "bar_start_utc": [minute],
            "bar_end_utc": [minute + pd.Timedelta(minutes=1)],
            "available_at_utc": [CUTOFF],
        }
    )
    activations = pd.DataFrame({"activation_time_utc": [CUTOFF]})
    memberships = pd.DataFrame({"available_at_utc": [minute]})
    return (
        common.copy(),
        common.copy(),
        common.copy(),
        common.copy(),
        memberships,
        activations,
    )
