"""Clock-grid feature and executable label construction for KS4."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.intraday.contracts import IntradayDatasetConfig
from market_predictor.intraday.dataset import _one_minute_ticker_features
from market_predictor.intraday.labels import add_exact_one_minute_labels
from market_predictor.intraday.specialist_contracts import (
    INTRADAY_SPECIALIST_IDS,
    IntradaySpecialistResearchConfig,
    intraday_specialist_policy_identity,
    load_intraday_specialist_research_config,
)
from market_predictor.intraday.specialist_dataset import (
    verify_specialist_collection_plan,
    verify_specialist_setup_bundle,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError

SPECIALIST_TRAINING_DATASET_SCHEMA = "intraday.specialist_training_dataset.v1"
SPECIALIST_TRAINING_ROW_SCHEMA = "intraday.specialist_training_row.v1"
_BAR_COLUMNS = [
    "ticker",
    "bar_start_utc",
    "bar_end_utc",
    "available_at_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_feed",
    "adjustment",
]


def build_intraday_specialist_training_dataset(
    *,
    setup_directory: Path,
    collection_plan_directory: Path,
    collection_directory: Path,
    policy_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Build monthly KS4 training rows with executable one-minute labels."""

    if output_directory.exists():
        raise DataReadinessError(
            f"KS4 training output must be new: {output_directory}"
        )
    setup_manifest = verify_specialist_setup_bundle(setup_directory)
    plan_manifest = verify_specialist_collection_plan(
        collection_plan_directory
    )
    config = load_intraday_specialist_research_config(policy_path)
    _verify_lineage(setup_manifest, plan_manifest, config)
    collection_path = collection_directory / "_manifest.json"
    collection = _load_json(collection_path)
    if (
        collection.get("status") != "transport_complete"
        or collection.get("failed_units") != {}
    ):
        raise DataReadinessError(
            "KS4 training requires transport-complete collection"
        )
    artifact_records = _artifact_records(collection)
    requirement_paths = sorted(
        (collection_plan_directory / "requirements").glob("*.parquet")
    )
    if not requirement_paths:
        raise DataReadinessError("KS4 training plan has no requirements")
    staging = output_directory.with_name(
        f".{output_directory.name}.staging"
    )
    request_identity = {
        "schema": SPECIALIST_TRAINING_DATASET_SCHEMA,
        "setup_fingerprint": str(setup_manifest["bundle_fingerprint"]),
        "plan_fingerprint": str(plan_manifest["plan_fingerprint"]),
        "collection_manifest_sha256": file_sha256(collection_path),
        "policy_sha256": config.policy_sha256(),
    }
    staging.mkdir(parents=True, exist_ok=True)
    request_path = staging / "_request.json"
    if request_path.exists():
        if _load_json(request_path) != request_identity:
            raise DataReadinessError(
                "KS4 training staging identity differs"
            )
    else:
        request_path.write_text(
            json.dumps(request_identity, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    files: list[dict[str, Any]] = []
    strategy_rows = {strategy: 0 for strategy in INTRADAY_SPECIALIST_IDS}
    strategy_eligible = {
        strategy: 0 for strategy in INTRADAY_SPECIALIST_IDS
    }
    reason_counts: dict[str, int] = {}
    try:
        for requirement_path in requirement_paths:
            month = requirement_path.stem
            month_setups = _load_setup_month(setup_directory, month)
            if month_setups.empty:
                continue
            month_requirements = pd.read_parquet(requirement_path)
            sessions = sorted(set(month_setups["session_date_et"]))
            for batch_number, offset in enumerate(
                range(0, len(sessions), config.cross_section_batch_sessions)
            ):
                batch_sessions = set(
                    sessions[
                        offset : offset
                        + config.cross_section_batch_sessions
                    ]
                )
                setups = month_setups[
                    month_setups["session_date_et"].isin(batch_sessions)
                ]
                requirements = month_requirements[
                    month_requirements["setup_id"].isin(setups["setup_id"])
                ]
                bars = load_clock_grid_for_requirements(
                    requirements,
                    artifact_records=artifact_records,
                    finalization_delay_seconds=(
                        config.intraday_finalization_delay_seconds
                    ),
                )
                features = build_clock_grid_features(
                    bars,
                    minimum_warmup_bars=(
                        config.minimum_one_minute_warmup_bars
                    ),
                )
                for strategy_id, strategy_setups in setups.groupby(
                    "strategy_id",
                    sort=False,
                ):
                    output_path = (
                        staging
                        / "strategies"
                        / _strategy_slug(str(strategy_id))
                        / f"{month}-part-{batch_number:03d}.parquet"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    if output_path.exists():
                        training = pd.read_parquet(output_path)
                    else:
                        strategy = config.strategies[str(strategy_id)]
                        training = build_strategy_training_rows(
                            strategy_setups,
                            bars=bars,
                            one_minute_features=features,
                            horizon_minutes=strategy.horizon_minutes,
                            config=config,
                        )
                        _validate_strategy_training_shard(
                            training,
                            expected_setups=strategy_setups,
                            strategy_id=str(strategy_id),
                            path=output_path,
                        )
                        _atomic_parquet(training, output_path)
                    _validate_strategy_training_shard(
                        training,
                        expected_setups=strategy_setups,
                        strategy_id=str(strategy_id),
                        path=output_path,
                    )
                    files.append(
                        _file_record(
                            output_path,
                            staging,
                            rows=len(training),
                        )
                    )
                    strategy_rows[str(strategy_id)] += len(training)
                    strategy_eligible[str(strategy_id)] += int(
                        training["label_eligible"].sum()
                    )
                    for reason, count in training[
                        "label_ineligible_reason"
                    ].value_counts().items():
                        reason_counts[str(reason)] = (
                            reason_counts.get(str(reason), 0) + int(count)
                        )
                del bars, features, setups, requirements
                release_process_memory()
                _guard_memory(
                    config,
                    f"KS4 training data {month} batch {batch_number}",
                )
            del month_setups, month_requirements
        fingerprint = _dataset_fingerprint(
            files=files,
            setup_fingerprint=str(setup_manifest["bundle_fingerprint"]),
            plan_fingerprint=str(plan_manifest["plan_fingerprint"]),
            collection_manifest_sha256=file_sha256(collection_path),
            policy_sha256=config.policy_sha256(),
        )
        report: dict[str, Any] = {
            "schema": SPECIALIST_TRAINING_DATASET_SCHEMA,
            "row_schema": SPECIALIST_TRAINING_ROW_SCHEMA,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_fingerprint": fingerprint,
            "setup_bundle": {
                "path": str(setup_directory),
                "bundle_fingerprint": str(
                    setup_manifest["bundle_fingerprint"]
                ),
                "manifest_sha256": file_sha256(
                    setup_directory / "_manifest.json"
                ),
            },
            "collection_plan": {
                "path": str(collection_plan_directory),
                "plan_fingerprint": str(
                    plan_manifest["plan_fingerprint"]
                ),
                "manifest_sha256": file_sha256(
                    collection_plan_directory / "_manifest.json"
                ),
            },
            "collection": {
                "path": str(collection_directory),
                "manifest_sha256": file_sha256(collection_path),
                "request_sha256": str(collection["request_sha256"]),
                "rows": int(collection["total_rows"]),
            },
            "policy": intraday_specialist_policy_identity(policy_path),
            "clock_grid_policy": {
                "missing_provider_bar": "no_eligible_trade",
                "feature_price": "causal_last_trade_carry_forward",
                "feature_volume": 0,
                "entry_requires_observed_bar": True,
                "trigger_requires_observed_bar": True,
                "timeout_requires_observed_bar": True,
                "benchmark_entry_exit_require_observed_bars": True,
                "minimum_observed_fraction_130": 0.5,
            },
            "summary": {
                "rows": sum(strategy_rows.values()),
                "eligible_rows": sum(strategy_eligible.values()),
                "strategy_rows": strategy_rows,
                "strategy_eligible_rows": strategy_eligible,
                "ineligible_reason_counts": dict(
                    sorted(reason_counts.items())
                ),
            },
            "memory": memory_audit(
                hard_budget_gib=config.maximum_process_memory_gib,
                headroom_gib=config.memory_guard_headroom_gib,
            ).to_record(),
            "files": sorted(files, key=lambda item: str(item["path"])),
        }
        request_path.unlink()
        (staging / "_manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_directory)
        return report
    except Exception:
        raise


def load_clock_grid_for_requirements(
    requirements: pd.DataFrame,
    *,
    artifact_records: Mapping[tuple[str, str], Mapping[str, Any]],
    finalization_delay_seconds: int,
) -> pd.DataFrame:
    """Load required full sessions and materialize causal no-trade slots."""

    keys = {
        (
            str(row.ticker).upper().strip(),
            pd.Timestamp(row.session_date_et).date().isoformat(),
        )
        for row in requirements[
            ["ticker", "session_date_et"]
        ].itertuples(index=False)
    }
    records_by_path: dict[Path, list[tuple[str, str]]] = {}
    for key in keys:
        record = artifact_records.get(key)
        if record is None:
            continue
        records_by_path.setdefault(Path(str(record["path"])), []).append(key)
    observed_parts: list[pd.DataFrame] = []
    grid_parts: list[pd.DataFrame] = []
    for path, path_keys in records_by_path.items():
        observed = pd.read_parquet(path, columns=_BAR_COLUMNS)
        selected_symbols = {ticker for ticker, _ in path_keys}
        observed = observed[
            observed["ticker"].astype(str).str.upper().isin(selected_symbols)
        ].copy()
        observed_parts.append(observed)
        for ticker, session in path_keys:
            record = artifact_records[(ticker, session)]
            start = pd.Timestamp(record["requested_start_utc"])
            end = pd.Timestamp(record["requested_end_utc"])
            grid_parts.append(
                pd.DataFrame(
                    {
                        "ticker": ticker,
                        "bar_start_utc": pd.date_range(
                            start,
                            end,
                            freq="1min",
                            inclusive="left",
                        ),
                    }
                )
            )
    if not observed_parts or not grid_parts:
        raise DataReadinessError(
            "KS4 requirements have no collected one-minute sessions"
        )
    observed = pd.concat(observed_parts, ignore_index=True)
    observed["ticker"] = observed["ticker"].astype(str).str.upper().str.strip()
    observed["observed_eligible_trade"] = True
    grid = pd.concat(grid_parts, ignore_index=True)
    if bool(grid.duplicated(["ticker", "bar_start_utc"]).any()):
        grid = grid.drop_duplicates(["ticker", "bar_start_utc"])
    dense = grid.merge(
        observed,
        on=["ticker", "bar_start_utc"],
        how="left",
        validate="one_to_one",
    ).sort_values(["ticker", "bar_start_utc"], kind="stable")
    dense["observed_eligible_trade"] = (
        dense["observed_eligible_trade"].fillna(False).astype(bool)
    )
    prior_close = dense.groupby("ticker", sort=False)["close"].ffill()
    for column in ("open", "high", "low", "close"):
        dense[column] = pd.to_numeric(
            dense[column],
            errors="coerce",
        ).fillna(prior_close)
    dense["volume"] = pd.to_numeric(
        dense["volume"],
        errors="coerce",
    ).fillna(0.0)
    dense["timeframe"] = "1m"
    dense["price_feed"] = dense["price_feed"].fillna("sip")
    dense["adjustment"] = dense["adjustment"].fillna("all")
    dense["bar_end_utc"] = dense["bar_start_utc"] + pd.Timedelta(minutes=1)
    dense["available_at_utc"] = dense["bar_end_utc"] + pd.Timedelta(
        seconds=finalization_delay_seconds
    )
    eastern = dense["bar_start_utc"].dt.tz_convert("America/New_York")
    dense["session_date_et"] = eastern.dt.date
    dense["session_minute_et"] = (
        eastern.dt.hour * 60 + eastern.dt.minute
    ).astype("int16")
    dense["session_slot"] = (
        dense["session_minute_et"] - (9 * 60 + 30)
    ).astype("int16")
    return dense.reset_index(drop=True)


def build_clock_grid_features(
    bars: pd.DataFrame,
    *,
    minimum_warmup_bars: int,
) -> pd.DataFrame:
    """Build the existing causal one-minute feature family on a clock grid."""

    feature_config = IntradayDatasetConfig(
        min_one_minute_bars=minimum_warmup_bars
    )
    parts: list[pd.DataFrame] = []
    for _, ticker_bars in bars.groupby("ticker", sort=False):
        part = _one_minute_ticker_features(ticker_bars, feature_config)
        observed = part["observed_eligible_trade"].astype(float)
        part["observed_fraction_130"] = observed.rolling(
            minimum_warmup_bars,
            min_periods=minimum_warmup_bars,
        ).mean()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_strategy_training_rows(
    setups: pd.DataFrame,
    *,
    bars: pd.DataFrame,
    one_minute_features: pd.DataFrame,
    horizon_minutes: int,
    config: IntradaySpecialistResearchConfig,
) -> pd.DataFrame:
    """Join causal confirmations and build executable path labels."""

    decisions = setups.copy()
    feature_columns = [
        "ticker",
        "available_at_utc",
        "bar_start_utc",
        "one_minute_history_exact",
        "observed_fraction_130",
        *[
            column
            for column in one_minute_features
            if column.endswith("_1m")
        ],
    ]
    joined_parts: list[pd.DataFrame] = []
    for ticker, left in decisions.groupby("ticker", sort=False):
        right = one_minute_features[
            one_minute_features["ticker"].eq(ticker)
        ].sort_values("available_at_utc")
        joined_parts.append(
            pd.merge_asof(
                left.sort_values("decision_time_utc"),
                right[feature_columns].drop(columns=["ticker"]),
                left_on="decision_time_utc",
                right_on="available_at_utc",
                direction="backward",
                allow_exact_matches=True,
            )
        )
    decisions = pd.concat(joined_parts, ignore_index=True)
    decisions["feature_eligible"] = (
        decisions["one_minute_history_exact"].fillna(False).astype(bool)
        & decisions["observed_fraction_130"].ge(0.5)
        & decisions["available_at_utc"].le(decisions["decision_time_utc"])
    )
    label_config = IntradayDatasetConfig(
        horizon_minutes=horizon_minutes,
        target_atr=config.target_atr,
        stop_atr=config.stop_atr,
        round_trip_cost_bps=config.minimum_round_trip_cost_bps,
        min_one_minute_bars=config.minimum_one_minute_warmup_bars,
    )
    labeled = add_exact_one_minute_labels(
        decisions,
        bars,
        label_config,
    )
    labeled = _apply_execution_evidence(labeled, bars)
    labeled.insert(0, "training_schema_version", SPECIALIST_TRAINING_ROW_SCHEMA)
    return labeled


def _apply_execution_evidence(
    labeled: pd.DataFrame,
    bars: pd.DataFrame,
) -> pd.DataFrame:
    output = labeled.copy()
    observed = bars.set_index(["ticker", "bar_start_utc"])[
        "observed_eligible_trade"
    ]
    entry_time = pd.to_datetime(output["entry_time_utc"], utc=True)
    exit_start = (
        pd.to_datetime(output["exit_time_utc"], utc=True)
        - pd.Timedelta(minutes=1)
    )
    entry_observed = _observed_lookup(
        observed,
        output["ticker"],
        entry_time,
    )
    exit_observed = _observed_lookup(
        observed,
        output["ticker"],
        exit_start,
    )
    benchmark_observed = np.ones(len(output), dtype=bool)
    for ticker in (
        pd.Series("SPY", index=output.index),
        pd.Series("QQQ", index=output.index),
        output["primary_benchmark"].astype(str).str.upper(),
    ):
        benchmark_observed &= _observed_lookup(
            observed,
            ticker,
            entry_time,
        )
        benchmark_observed &= _observed_lookup(
            observed,
            ticker,
            exit_start,
        )
    output["entry_observed"] = entry_observed
    output["exit_observed"] = exit_observed
    output["benchmark_entry_exit_observed"] = benchmark_observed
    output["label_eligible"] = (
        output["label_eligible"].fillna(False).astype(bool)
        & entry_observed
        & exit_observed
        & benchmark_observed
    )
    reason = np.full(len(output), "eligible", dtype=object)
    reason[~output["feature_eligible"].to_numpy(bool)] = (
        "insufficient_feature_history"
    )
    reason[~entry_observed] = "missing_observed_entry"
    reason[entry_observed & ~exit_observed] = "missing_observed_exit"
    reason[
        entry_observed & exit_observed & ~benchmark_observed
    ] = "missing_observed_benchmark"
    reason[
        output["feature_eligible"].to_numpy(bool)
        & entry_observed
        & exit_observed
        & benchmark_observed
        & ~output["label_path_exact"].fillna(False).to_numpy(bool)
    ] = "inexact_clock_path"
    reason[
        (reason == "eligible")
        & ~output["label_eligible"].fillna(False).to_numpy(bool)
    ] = "label_contract_ineligible"
    output["label_ineligible_reason"] = pd.Series(reason, dtype="string")
    return output


def _observed_lookup(
    observed: pd.Series,
    tickers: pd.Series,
    timestamps: pd.Series,
) -> np.ndarray:
    keys = pd.MultiIndex.from_arrays(
        [tickers.astype(str).str.upper(), timestamps]
    )
    return np.asarray(
        observed.reindex(keys).fillna(False).astype(bool).to_numpy(),
        dtype=bool,
    )


def _validate_strategy_training_shard(
    training: pd.DataFrame,
    *,
    expected_setups: pd.DataFrame,
    strategy_id: str,
    path: Path,
) -> None:
    required = {
        "training_schema_version",
        "setup_id",
        "strategy_id",
        "label_eligible",
        "label_ineligible_reason",
    }
    missing = required.difference(training.columns)
    if missing:
        raise DataReadinessError(
            f"KS4 training shard {path} is missing columns: "
            + ", ".join(sorted(missing))
        )
    if bool(training["setup_id"].duplicated().any()):
        raise DataReadinessError(
            f"KS4 training shard {path} has duplicate setup IDs"
        )
    actual_ids = set(training["setup_id"].astype(str))
    expected_ids = set(expected_setups["setup_id"].astype(str))
    if actual_ids != expected_ids:
        raise DataReadinessError(
            f"KS4 training shard {path} does not match expected setups"
        )
    if set(training["strategy_id"].astype(str)) != {strategy_id}:
        raise DataReadinessError(
            f"KS4 training shard {path} has the wrong strategy"
        )
    if set(training["training_schema_version"].astype(str)) != {
        SPECIALIST_TRAINING_ROW_SCHEMA
    }:
        raise DataReadinessError(
            f"KS4 training shard {path} has the wrong row schema"
        )
    eligible = training["label_eligible"].fillna(False).astype(bool)
    reasons = training["label_ineligible_reason"].astype("string")
    invalid_reason = (eligible & reasons.ne("eligible")) | (
        ~eligible & reasons.eq("eligible")
    )
    if bool(invalid_reason.any()):
        raise DataReadinessError(
            f"KS4 training shard {path} has inconsistent eligibility reasons"
        )


def _artifact_records(
    collection: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    artifacts = collection.get("artifacts")
    if not isinstance(artifacts, list):
        raise DataReadinessError("KS4 collection artifacts are malformed")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in artifacts:
        if not isinstance(raw, Mapping):
            raise DataReadinessError(
                "KS4 collection artifact record is malformed"
            )
        symbols = raw.get("symbol_rows")
        if not isinstance(symbols, Mapping):
            raise DataReadinessError(
                "KS4 collection artifact has no symbol rows"
            )
        session = str(raw["asof_date"])
        for symbol in symbols:
            key = (str(symbol).upper().strip(), session)
            if key in result:
                raise DataReadinessError(
                    f"KS4 duplicate artifact ticker-session: {key}"
                )
            result[key] = raw
    return result


def _load_setup_month(directory: Path, month: str) -> pd.DataFrame:
    parts = []
    for strategy_id in INTRADAY_SPECIALIST_IDS:
        path = (
            directory
            / "setups"
            / _strategy_slug(strategy_id)
            / f"{month}.parquet"
        )
        if path.is_file():
            parts.append(pd.read_parquet(path))
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame()
    )


def _verify_lineage(
    setup: Mapping[str, Any],
    plan: Mapping[str, Any],
    config: IntradaySpecialistResearchConfig,
) -> None:
    if (
        str(cast(Mapping[str, Any], setup["policy"])["policy_sha256"])
        != config.policy_sha256()
        or str(cast(Mapping[str, Any], plan["policy"])["policy_sha256"])
        != config.policy_sha256()
        or str(
            cast(Mapping[str, Any], plan["setup_bundle"])[
                "bundle_fingerprint"
            ]
        )
        != str(setup["bundle_fingerprint"])
    ):
        raise DataReadinessError("KS4 training lineage differs")


def _dataset_fingerprint(
    *,
    files: Sequence[Mapping[str, Any]],
    setup_fingerprint: str,
    plan_fingerprint: str,
    collection_manifest_sha256: str,
    policy_sha256: str,
) -> str:
    payload = {
        "schema": SPECIALIST_TRAINING_DATASET_SCHEMA,
        "setup_fingerprint": setup_fingerprint,
        "plan_fingerprint": plan_fingerprint,
        "collection_manifest_sha256": collection_manifest_sha256,
        "policy_sha256": policy_sha256,
        "files": [
            {
                "path": str(record["path"]),
                "sha256": str(record["sha256"]),
                "rows": int(record["rows"]),
            }
            for record in sorted(
                files,
                key=lambda item: str(item["path"]),
            )
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_record(
    path: Path,
    root: Path,
    *,
    rows: int,
) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"unreadable KS4 JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"KS4 JSON is not an object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _strategy_slug(strategy_id: str) -> str:
    return strategy_id.lower().replace(".", "_")


def _guard_memory(
    config: IntradaySpecialistResearchConfig,
    stage: str,
) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
    assert_peak_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
