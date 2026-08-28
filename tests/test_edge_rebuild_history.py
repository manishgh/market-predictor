import json
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.contracts.history_collection import (
    INTRADAY_HISTORY_SCHEMA,
    IntradayHistoryConfig,
    load_intraday_history_config,
)
from market_predictor.intraday.datasets.history import (
    build_intraday_history_plan,
    load_complete_intraday_history_plan,
)

POLICY_PATH = Path("configs/edge_rebuild_intraday_history.toml")


def test_intraday_history_contract_freezes_two_tier_acquisition() -> None:
    config = load_intraday_history_config(POLICY_PATH)

    assert config.schema_version == INTRADAY_HISTORY_SCHEMA
    assert config.feature_timeframe == "5Min"
    assert config.exact_path_timeframe == "1Min"
    assert config.required_price_feed == "sip"
    assert config.required_adjustment == "all"
    assert config.target_usable_sessions == 1_250
    assert config.minimum_usable_sessions == 750
    assert config.collection_workers == 2
    assert config.maximum_symbols_per_unit == 50
    assert config.maximum_failures_before_stop == 5
    assert config.maximum_process_memory_gib == 4


def test_intraday_history_contract_rejects_full_path_downgrade() -> None:
    raw = load_intraday_history_config(POLICY_PATH).model_dump()
    raw["exact_path_timeframe"] = "5Min"

    with pytest.raises(ValueError, match="one-minute"):
        IntradayHistoryConfig.model_validate(raw)


def test_plan_is_hash_bound_point_in_time_and_selective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_intraday_history_config(POLICY_PATH).model_copy(
        update={
            "target_usable_sessions": 1_000,
            "minimum_usable_sessions": 750,
            "feature_warmup_sessions": 20,
            "minimum_session_cross_section": 300,
        }
    )
    audit_dir = tmp_path / "readiness"
    audit_dir.mkdir()
    (audit_dir / "session_calendar.csv").write_text(
        "strategy_id,session_date_et\n"
        + "\n".join(
            "INTRADAY.VWAP_EXHAUSTION_REVERSAL.30M.V1,"
            f"{date.date().isoformat()}"
            for date in pd.bdate_range("2022-01-03", periods=980)
        ),
        encoding="utf-8",
    )
    (audit_dir / "_request.json").write_text(
        json.dumps({"request_sha256": "audit-sha"}),
        encoding="utf-8",
    )
    (audit_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "blocked_pending_targeted_acquisition",
                "er2_authorized": False,
                "acquisition_plan": {
                    "authorized_by_audit": True,
                    "feed": "sip",
                    "adjustment": "all",
                },
            }
        ),
        encoding="utf-8",
    )
    (audit_dir / "_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "market_predictor.intraday.datasets.history."
        "load_complete_readiness_audit",
        lambda *_args, **_kwargs: {},
    )
    memberships_path = tmp_path / "memberships.parquet"
    tickers = [f"T{i:03d}" for i in range(300)]
    memberships = pd.DataFrame(
        {
            "ticker": tickers + ["OLD"],
            "security_id": [f"security:{value}" for value in tickers]
            + ["security:old"],
            "effective_from_utc": [
                pd.Timestamp("2019-01-01", tz="UTC")
            ]
            * 301,
            "effective_to_utc": [pd.NaT] * 300
            + [pd.Timestamp("2020-01-01", tz="UTC")],
            "sector": ["Information Technology"] * 301,
            "primary_benchmark": ["XLK"] * 301,
            "universe_snapshot_id": ["snapshot-1"] * 301,
            "membership_source_urls": ['["primary"]'] * 301,
        }
    )
    memberships.to_parquet(memberships_path, index=False)
    membership_audit = tmp_path / "membership_audit.json"
    membership_audit.write_text(
        json.dumps(
            {
                "universe_snapshot_id": "snapshot-1",
                "historical_tickers": 301,
                "membership_intervals": 301,
                "contradictions": [],
            }
        ),
        encoding="utf-8",
    )
    stock_dir = _write_ohlcv_identity(tmp_path / "stocks", 300)
    benchmark_dir = _write_ohlcv_identity(tmp_path / "benchmarks", 13)
    output = tmp_path / "plan"

    result = build_intraday_history_plan(
        readiness_audit_directory=audit_dir,
        memberships_path=memberships_path,
        membership_audit_path=membership_audit,
        existing_stock_bars_directory=stock_dir,
        existing_benchmark_bars_directory=benchmark_dir,
        policy_path=POLICY_PATH,
        output_directory=output,
        config=config,
    )
    verified = load_complete_intraday_history_plan(output)

    assert result["summary"]["existing_usable_sessions"] == 980
    assert result["summary"]["planned_history_sessions"] == 40
    assert result["summary"]["historical_tickers"] == 300
    assert result["summary"]["memory"]["hard_budget_gib"] == 4
    assert result["acquisition"]["feature_discovery"]["timeframe"] == "5Min"
    assert (
        result["acquisition"]["exact_path_labels"]["timeframe"]
        == "1Min"
    )
    assert (
        result["acquisition"]["exact_path_labels"][
            "planned_in_this_artifact"
        ]
        is False
    )
    assert result["research_only"] is True
    assert verified["plan_fingerprint"] == result["plan_fingerprint"]
    units = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted((output / "units" / "5Min").glob("*.parquet"))
        ],
        ignore_index=True,
    )
    assert set(units["price_feed"]) == {"sip"}
    assert set(units["adjustment"]) == {"all"}
    assert set(units["timeframe"]) == {"5Min"}
    assert int(units["maximum_expected_rows"].max()) <= 10_000
    assert int(units["symbol_count"].max()) <= 50
    assert "OLD" not in "".join(units["canonical_symbols_json"].astype(str))


def test_plan_rejects_static_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "market_predictor.intraday.datasets.history."
        "load_complete_readiness_audit",
        lambda *_args, **_kwargs: {},
    )
    memberships = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 450,
            "security_id": ["security:aapl"] * 450,
            "effective_from_utc": [
                pd.Timestamp("2019-01-01", tz="UTC")
            ]
            * 450,
            "effective_to_utc": [pd.NaT] * 450,
            "sector": ["Technology"] * 450,
            "primary_benchmark": ["XLK"] * 450,
            "universe_snapshot_id": ["snapshot"] * 450,
            "membership_source_urls": ["[]"] * 450,
        }
    )
    path = tmp_path / "memberships.parquet"
    memberships.to_parquet(path, index=False)
    audit = tmp_path / "audit.json"
    audit.write_text("{}", encoding="utf-8")

    with pytest.raises(DataReadinessError, match="point-in-time"):
        from market_predictor.intraday.datasets.history import (
            verify_point_in_time_memberships,
        )

        verify_point_in_time_memberships(
            path,
            audit,
            minimum_cross_section=300,
        )


def test_plan_detects_mutated_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_plan_is_hash_bound_point_in_time_and_selective(tmp_path, monkeypatch)
    output = tmp_path / "plan"
    unit = next((output / "units" / "5Min").glob("*.parquet"))
    unit.write_bytes(unit.read_bytes() + b"changed")

    with pytest.raises(DataReadinessError, match="does not verify"):
        load_complete_intraday_history_plan(output)


def _write_ohlcv_identity(path: Path, symbols: int) -> Path:
    path.mkdir()
    schema = {
        "schema_version": "ohlcv.v1",
        "source": "alpaca",
        "price_feed": "sip",
        "adjustment": "all",
        "timeframes": ["5m"],
        "start_utc": "2024-07-08T23:59:59+00:00",
        "end_utc": "2026-07-08T23:59:59+00:00",
    }
    (path / "_schema.json").write_text(
        json.dumps(schema),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(symbols)],
            "timeframe": ["5m"] * symbols,
            "rows": [1] * symbols,
            "path": [f"T{i}.parquet" for i in range(symbols)],
        }
    ).to_csv(path / "_ohlcv_manifest.csv", index=False)
    assert file_sha256(path / "_schema.json")
    return path
