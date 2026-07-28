from pathlib import Path

import pandas as pd
import pytest

from market_predictor.primary_v2.contracts import (
    INTRADAY_V2_ID,
    SWING_V2_ID,
    load_primary_v2_research_config,
)
from market_predictor.primary_v2.model import (
    primary_v2_experiment_specs,
    validate_primary_v2_source_rows,
)
from market_predictor.v3.errors import DataReadinessError

CONFIG = load_primary_v2_research_config(
    Path("configs/primary_strategy_v2.toml")
)


def test_primary_v2_experiment_matrix_is_frozen_and_bounded() -> None:
    swing = primary_v2_experiment_specs(SWING_V2_ID)
    intraday = primary_v2_experiment_specs(INTRADAY_V2_ID)

    assert len(swing) == 4
    assert len(intraday) == 4
    assert len({spec.candidate_id for spec in swing}) == 4
    assert len({spec.candidate_id for spec in intraday}) == 4
    assert all("catalyst" not in spec.candidate_id for spec in swing + intraday)


def test_swing_source_validation_rejects_duplicate_identity() -> None:
    frame = _valid_source_frame(SWING_V2_ID)
    validate_primary_v2_source_rows(
        frame,
        strategy_id=SWING_V2_ID,
        config=CONFIG,
    )

    duplicated = pd.concat([frame, frame], ignore_index=True)
    with pytest.raises(DataReadinessError, match="duplicated"):
        validate_primary_v2_source_rows(
            duplicated,
            strategy_id=SWING_V2_ID,
            config=CONFIG,
        )


def test_intraday_source_validation_requires_sip_and_no_overnight_path() -> None:
    frame = _valid_source_frame(INTRADAY_V2_ID)
    validate_primary_v2_source_rows(
        frame,
        strategy_id=INTRADAY_V2_ID,
        config=CONFIG,
    )

    partial_feed = frame.assign(price_feed="iex")
    with pytest.raises(DataReadinessError, match="SIP"):
        validate_primary_v2_source_rows(
            partial_feed,
            strategy_id=INTRADAY_V2_ID,
            config=CONFIG,
        )

    overnight = frame.assign(exit_time_utc="2026-01-06T15:31:00Z")
    overnight["label_available_at_utc"] = "2026-01-06T15:31:00Z"
    with pytest.raises(DataReadinessError, match="overnight"):
        validate_primary_v2_source_rows(
            overnight,
            strategy_id=INTRADAY_V2_ID,
            config=CONFIG,
        )


def test_source_validation_rejects_future_entry_and_non_finite_label() -> None:
    frame = _valid_source_frame(SWING_V2_ID)
    future_entry = frame.assign(entry_time_utc="2026-01-04T15:30:00Z")
    with pytest.raises(DataReadinessError, match="timestamps"):
        validate_primary_v2_source_rows(
            future_entry,
            strategy_id=SWING_V2_ID,
            config=CONFIG,
        )

    strategy = CONFIG.strategies[SWING_V2_ID]
    non_finite = frame.assign(**{strategy.source_target: float("nan")})
    with pytest.raises(DataReadinessError, match="non-finite"):
        validate_primary_v2_source_rows(
            non_finite,
            strategy_id=SWING_V2_ID,
            config=CONFIG,
        )


def _valid_source_frame(strategy_id: str) -> pd.DataFrame:
    strategy = CONFIG.strategies[strategy_id]
    values: dict[str, object] = {
        column: 0.0 for column in strategy.required_source_columns
    }
    values.update(
        {
            "ticker": "TEST",
            "primary_benchmark": "XLK",
            strategy.period_column: "2026-01-05",
            strategy.row_id_column: "row-1",
            strategy.eligibility_column: True,
            "decision_time_utc": "2026-01-05T15:30:00Z",
            "entry_time_utc": "2026-01-05T15:31:00Z",
            "exit_time_utc": "2026-01-05T16:00:00Z",
            "label_available_at_utc": "2026-01-05T16:00:00Z",
            strategy.source_target: 0.01,
            strategy.spy_excess_target: 0.005,
            strategy.sector_excess_target: 0.004,
            strategy.mfe_target: 0.02,
            strategy.mae_target: -0.005,
        }
    )
    if strategy_id == INTRADAY_V2_ID:
        assert strategy.competing_risk_targets is not None
        values.update(
            {
                "price_feed": "sip",
                "adjustment": "all",
                strategy.competing_risk_targets.target_first: 1,
                strategy.competing_risk_targets.stop_first: 0,
                strategy.competing_risk_targets.timeout: 0,
                strategy.competing_risk_targets.outcome: "target_first",
                strategy.competing_risk_targets.time_to_resolution: 12,
            }
        )
    return pd.DataFrame([values])
