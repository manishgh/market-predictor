"""Read-only, hash-bound ER1 data-readiness audit."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild import contracts
from market_predictor.edge_rebuild.contracts import (
    EdgeRebuildReadinessConfig,
)
from market_predictor.intraday.specialist_contracts import (
    IntradaySpecialistResearchConfig,
)
from market_predictor.intraday.specialist_experiments import (
    VerifiedTrainingBundle,
    verify_intraday_specialist_training_bundle,
)
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError

READINESS_RUN_SCHEMA = "edge_rebuild.readiness.run.v1"
READINESS_AUTHORITY_SCHEMA = "edge_rebuild.readiness.authority.v1"
_OUTPUT_NAMES = (
    "_request.json",
    "blockers.csv",
    "catalyst_readiness.csv",
    "cost_readiness.csv",
    "dimension_coverage.csv",
    "exclusion_reasons.csv",
    "fold_capacity.csv",
    "phase_capacity.csv",
    "session_calendar.csv",
    "source_inventory.csv",
    "summary.json",
)


@dataclass(frozen=True)
class VerifiedSwingSources:
    technical: pd.DataFrame
    proxy: pd.DataFrame
    identity: dict[str, object]
    bundle_request: dict[str, Any]


@dataclass(frozen=True)
class VerifiedIntradaySources:
    proxy: pd.DataFrame
    identity: dict[str, object]
    training_manifest: dict[str, Any]
    collection_manifest: dict[str, Any]
    collection_request: dict[str, Any]
    coverage_manifest: dict[str, Any]


@dataclass(frozen=True)
class VerifiedCatalystSources:
    identity: dict[str, object]
    lineage_manifest: dict[str, Any]
    news_manifest: dict[str, Any]
    coverage: pd.DataFrame


def run_edge_rebuild_readiness_audit(
    *,
    swing_bundle_dir: Path,
    swing_technical_path: Path,
    intraday_training_dir: Path,
    intraday_collection_dir: Path,
    intraday_coverage_dir: Path,
    catalyst_lineage_dir: Path,
    news_source_dir: Path,
    out_dir: Path,
    config: EdgeRebuildReadinessConfig,
    policy_path: Path,
    swing_policy_path: Path,
    intraday_config: IntradaySpecialistResearchConfig,
    intraday_policy_path: Path,
) -> dict[str, object]:
    """Verify reusable sources and publish the immutable ER1 audit."""

    _assert_memory(config, "ER1 source verification")
    swing = _verify_swing_sources(
        swing_bundle_dir=swing_bundle_dir,
        technical_path=swing_technical_path,
        policy_path=swing_policy_path,
        config=config,
    )
    _assert_memory(config, "ER1 swing verification")
    intraday = _verify_intraday_sources(
        training_dir=intraday_training_dir,
        collection_dir=intraday_collection_dir,
        coverage_dir=intraday_coverage_dir,
        policy_path=intraday_policy_path,
        intraday_config=intraday_config,
        config=config,
    )
    _assert_memory(config, "ER1 intraday verification")
    catalyst = _verify_catalyst_sources(
        lineage_dir=catalyst_lineage_dir,
        news_dir=news_source_dir,
        expected_lineage_sha256=_required_input_hash(
            swing.bundle_request,
            suffix="_manifest.json",
            contains="alpaca_catalyst_lineage_",
        ),
    )
    _assert_memory(config, "ER1 catalyst verification")

    evidence = _build_evidence(
        swing=swing,
        intraday=intraday,
        catalyst=catalyst,
        config=config,
    )
    request: dict[str, object] = {
        "schema": READINESS_RUN_SCHEMA,
        "policy_sha256": config.sha256(),
        "policy_file_sha256": file_sha256(policy_path),
        "swing_policy_file_sha256": file_sha256(swing_policy_path),
        "intraday_policy_file_sha256": file_sha256(intraday_policy_path),
        "sources": {
            "swing": swing.identity,
            "intraday": intraday.identity,
            "catalyst": catalyst.identity,
        },
        "implementation": readiness_implementation_identity(),
        "training_performed": False,
        "download_performed": False,
    }
    request_sha256 = _json_sha256_without_self_hash(request)
    summary = _build_summary(
        evidence,
        config=config,
        request_sha256=request_sha256,
    )
    result = _publish_audit(
        out_dir.resolve(),
        request=request,
        request_sha256=request_sha256,
        summary=summary,
        evidence=evidence,
    )
    del swing, intraday, catalyst, evidence
    release_process_memory()
    _assert_memory(config, "ER1 publication")
    return result


def readiness_implementation_identity() -> dict[str, object]:
    files = {
        "contracts": Path(contracts.__file__).resolve(),
        "readiness": Path(__file__).resolve(),
    }
    return {
        name: {"path": path.name, "sha256": file_sha256(path)}
        for name, path in sorted(files.items())
    }


def _verify_swing_sources(
    *,
    swing_bundle_dir: Path,
    technical_path: Path,
    policy_path: Path,
    config: EdgeRebuildReadinessConfig,
) -> VerifiedSwingSources:
    root = swing_bundle_dir.resolve()
    request = _load_json(root / "_request.json")
    manifest_path = root / "_manifest.json"
    manifest = _load_json(manifest_path)
    authority = _load_json(root / "_authority.json")
    request_sha256 = _json_sha256_without_self_hash(request)
    if (
        request.get("schema") != "swing.specialist_dataset_bundle.v4"
        or manifest.get("schema") != "swing.specialist_dataset_bundle.v4"
        or request.get("request_sha256") != request_sha256
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema")
        != "swing.specialist_dataset_authority.v1"
        or authority.get("state") != "complete"
        or authority.get("request_sha256") != request_sha256
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
    ):
        raise DataReadinessError("ER1 swing bundle authority does not verify")
    policy_hash = _required_input_hash(
        request,
        suffix="swing_specialist_research.toml",
    )
    if file_sha256(policy_path) != policy_hash:
        raise DataReadinessError("ER1 swing policy identity differs")
    technical = technical_path.resolve()
    technical_hash = file_sha256(technical)
    if technical_hash != _required_input_hash(
        request,
        suffix=".parquet",
        contains="swing_technical_",
    ):
        raise DataReadinessError("ER1 swing technical source identity differs")
    technical_manifest_path = Path(f"{technical}.manifest.json")
    if file_sha256(technical_manifest_path) != _required_input_hash(
        request,
        suffix=".parquet.manifest.json",
        contains="swing_technical_",
    ):
        raise DataReadinessError("ER1 swing technical manifest identity differs")
    technical_manifest = _load_json(technical_manifest_path)
    if (
        technical_manifest.get("artifact_type") != "swing_dataset"
        or technical_manifest.get("artifact_sha256") != technical_hash
    ):
        raise DataReadinessError("ER1 swing technical artifact does not verify")

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise DataReadinessError("ER1 swing bundle has no strategy artifacts")
    matches = [
        cast(dict[str, Any], raw)
        for raw in raw_artifacts
        if isinstance(raw, dict)
        and raw.get("strategy_id") == config.swing.proxy_strategy_id
    ]
    if len(matches) != 1:
        raise DataReadinessError("ER1 swing proxy source is ambiguous")
    proxy_record = matches[0]
    proxy_path = _resolve_inside(root, str(proxy_record.get("path", "")))
    if (
        file_sha256(proxy_path) != proxy_record.get("sha256")
        or file_sha256(Path(f"{proxy_path}.manifest.json"))
        != proxy_record.get("manifest_sha256")
    ):
        raise DataReadinessError("ER1 swing proxy artifact integrity failed")
    technical_frame = _read_required_columns(
        technical,
        {
            "adjustment",
            "cross_section_eligible",
            "daily_bar_count",
            "decision_group_id",
            "decision_time_utc",
            "dollar_volume_log",
            "feature_available_at_utc",
            "feature_eligible",
            "label_eligible",
            "label_path_exact",
            "market_regime",
            "price_feed",
            "round_trip_cost_bps",
            "sector",
            "session_date_et",
            "ticker",
        },
    )
    proxy_frame = _read_required_columns(
        proxy_path,
        {
            "adjustment",
            "catalyst_source_complete",
            "decision_time_utc",
            "event_count_3d",
            "latest_event_feature_available_at_utc",
            "price_feed",
            "setup_eligible",
            "strategy_decision_group_id",
            "strategy_execution_cost_fraction",
            "strategy_gross_return",
            "strategy_label_eligible",
            "strategy_net_return",
            "ticker",
        },
    )
    return VerifiedSwingSources(
        technical=technical_frame,
        proxy=proxy_frame,
        identity={
            "type": "verified_swing_research_sources",
            "bundle_manifest_sha256": file_sha256(manifest_path),
            "bundle_request_sha256": request_sha256,
            "technical_artifact_sha256": technical_hash,
            "technical_manifest_sha256": file_sha256(
                technical_manifest_path
            ),
            "proxy_artifact_sha256": str(proxy_record["sha256"]),
            "technical_rows": len(technical_frame),
            "proxy_rows": len(proxy_frame),
        },
        bundle_request=request,
    )


def _verify_intraday_sources(
    *,
    training_dir: Path,
    collection_dir: Path,
    coverage_dir: Path,
    policy_path: Path,
    intraday_config: IntradaySpecialistResearchConfig,
    config: EdgeRebuildReadinessConfig,
) -> VerifiedIntradaySources:
    verified = verify_intraday_specialist_training_bundle(
        training_dir,
        config=intraday_config,
        policy_path=policy_path,
    )
    collection_root = collection_dir.resolve()
    collection_manifest_path = collection_root / "_manifest.json"
    collection_request_path = collection_root / "_request.json"
    collection_manifest = _load_json(collection_manifest_path)
    collection_request = _load_json(collection_request_path)
    expected_collection = _mapping(verified.manifest.get("collection"))
    if (
        file_sha256(collection_manifest_path)
        != expected_collection.get("manifest_sha256")
        or collection_manifest.get("request_sha256")
        != expected_collection.get("request_sha256")
        or collection_request.get("request_sha256")
        != expected_collection.get("request_sha256")
        or _json_sha256_without_self_hash(collection_request)
        != expected_collection.get("request_sha256")
        or collection_request.get("price_feed")
        != config.required_price_feed
        or collection_request.get("adjustment")
        != config.required_adjustment
        or collection_request.get("timeframe")
        != config.intraday.required_timeframe
    ):
        raise DataReadinessError("ER1 intraday collection identity differs")
    coverage_manifest_path = coverage_dir.resolve() / "_manifest.json"
    coverage_manifest = _verify_intraday_coverage(
        coverage_dir.resolve(),
        expected_collection_manifest_sha256=file_sha256(
            collection_manifest_path
        ),
    )
    records = verified.strategy_files[config.intraday.proxy_strategy_id]
    columns = {
        "adjustment",
        "decision_time_utc",
        "entry_time_utc",
        "feature_available_at_utc",
        "feature_eligible",
        "label_eligible",
        "label_ineligible_reason",
        "liquidity_bucket",
        "market_cap_bucket",
        "one_minute_history_exact",
        "observed_fraction_130",
        "path_excess_return_30m_vs_sector",
        "path_excess_return_30m_vs_spy",
        "path_realized_return_gross_30m",
        "path_realized_return_net_30m",
        "price_feed",
        "regime_risk_off",
        "regime_risk_on",
        "sector",
        "session_date_et",
        "session_segment",
        "setup_id",
        "ticker",
    }
    proxy = _read_intraday_proxy(verified, records=records, columns=columns)
    return VerifiedIntradaySources(
        proxy=proxy,
        identity={
            "type": "verified_intraday_specialist_training_sources",
            "training_manifest_sha256": verified.manifest_sha256,
            "dataset_fingerprint": verified.dataset_fingerprint,
            "proxy_dataset_sha256": verified.strategy_dataset_sha256[
                config.intraday.proxy_strategy_id
            ],
            "collection_manifest_sha256": file_sha256(
                collection_manifest_path
            ),
            "collection_request_sha256": str(
                collection_request["request_sha256"]
            ),
            "coverage_manifest_sha256": file_sha256(
                coverage_manifest_path
            ),
            "coverage_fingerprint": str(
                coverage_manifest["coverage_fingerprint"]
            ),
            "collection_rows": int(
                collection_manifest.get("total_rows", 0)
            ),
            "proxy_rows": len(proxy),
        },
        training_manifest=verified.manifest,
        collection_manifest=collection_manifest,
        collection_request=collection_request,
        coverage_manifest=coverage_manifest,
    )


def _verify_catalyst_sources(
    *,
    lineage_dir: Path,
    news_dir: Path,
    expected_lineage_sha256: str,
) -> VerifiedCatalystSources:
    lineage_root = lineage_dir.resolve()
    lineage_manifest_path = lineage_root / "_manifest.json"
    lineage_request_path = lineage_root / "_request.json"
    lineage_manifest = _load_json(lineage_manifest_path)
    lineage_request = _load_json(lineage_request_path)
    lineage_request_sha256 = _json_sha256_without_self_hash(lineage_request)
    if (
        file_sha256(lineage_manifest_path) != expected_lineage_sha256
        or lineage_manifest.get("schema")
        != "swing.catalyst_lineage_manifest.v1"
        or lineage_manifest.get("status") != "complete"
        or lineage_manifest.get("request_sha256")
        != lineage_request_sha256
        or lineage_request.get("request_sha256")
        != lineage_request_sha256
    ):
        raise DataReadinessError("ER1 catalyst lineage does not verify")
    news_root = news_dir.resolve()
    news_manifest_path = news_root / "_manifest.json"
    news_manifest = _load_json(news_manifest_path)
    if (
        file_sha256(news_manifest_path)
        != lineage_request.get("collection_manifest_sha256")
        or news_manifest.get("status") != "complete"
        or news_manifest.get("request_sha256") is None
    ):
        raise DataReadinessError("ER1 catalyst news source does not verify")
    coverage_record = _mapping(lineage_manifest.get("coverage"))
    coverage_path = lineage_root / "source_coverage.parquet"
    if file_sha256(coverage_path) != coverage_record.get("sha256"):
        raise DataReadinessError("ER1 catalyst coverage artifact changed")
    coverage = _read_required_columns(
        coverage_path,
        {
            "coverage_state",
            "missingness_known",
            "requested_end_utc",
            "requested_start_utc",
            "source_family",
            "status",
            "ticker",
            "training_eligible",
        },
    )
    return VerifiedCatalystSources(
        identity={
            "type": "verified_catalyst_lineage_sources",
            "lineage_manifest_sha256": file_sha256(
                lineage_manifest_path
            ),
            "lineage_request_sha256": lineage_request_sha256,
            "news_manifest_sha256": file_sha256(news_manifest_path),
            "news_request_sha256": str(news_manifest["request_sha256"]),
            "coverage_sha256": str(coverage_record["sha256"]),
            "source_event_rows": int(
                lineage_manifest.get("source_event_rows", 0)
            ),
        },
        lineage_manifest=lineage_manifest,
        news_manifest=news_manifest,
        coverage=coverage,
    )


def _build_evidence(
    *,
    swing: VerifiedSwingSources,
    intraday: VerifiedIntradaySources,
    catalyst: VerifiedCatalystSources,
    config: EdgeRebuildReadinessConfig,
) -> dict[str, pd.DataFrame]:
    swing_rows, swing_exclusions = _prepare_swing_rows(
        swing.technical,
        config=config,
    )
    intraday_rows, intraday_exclusions = _prepare_intraday_rows(
        intraday.proxy,
        config=config,
    )
    session_calendar = pd.concat(
        [
            _session_calendar(
                swing_rows,
                strategy_id=config.swing.strategy_id,
                proxy_strategy_id=config.swing.proxy_strategy_id,
                decision_id_column="decision_group_id",
            ),
            _session_calendar(
                intraday_rows,
                strategy_id=config.intraday.strategy_id,
                proxy_strategy_id=config.intraday.proxy_strategy_id,
                decision_id_column="setup_id",
            ),
        ],
        ignore_index=True,
    )
    phases = _swing_phase_capacity(
        swing_rows,
        phases=config.swing.non_overlapping_phases,
        minimum_sessions=config.swing.minimum_sessions_per_phase,
        strategy_id=config.swing.strategy_id,
    )
    folds = _intraday_fold_capacity(
        intraday_rows,
        folds=config.intraday.required_purged_folds,
        minimum_test_sessions=(
            config.intraday.minimum_test_sessions_per_fold
        ),
        strategy_id=config.intraday.strategy_id,
    )
    dimensions = pd.concat(
        [
            _dimension_coverage(
                swing_rows,
                strategy_id=config.swing.strategy_id,
                dimensions=("year", "market_regime", "sector", "price_feed"),
            ),
            _dimension_coverage(
                intraday_rows,
                strategy_id=config.intraday.strategy_id,
                dimensions=(
                    "year",
                    "session_segment",
                    "market_regime",
                    "sector",
                    "market_cap_bucket",
                    "liquidity_bucket",
                    "price_feed",
                ),
            ),
        ],
        ignore_index=True,
    )
    costs = _cost_readiness(
        swing_proxy=swing.proxy,
        intraday_proxy=intraday.proxy,
        config=config,
    )
    catalyst_readiness = _catalyst_readiness(
        swing=swing,
        catalyst=catalyst,
        config=config,
    )
    source_inventory = _source_inventory(
        swing_rows=swing_rows,
        swing_proxy=swing.proxy,
        intraday_rows=intraday_rows,
        intraday=intraday,
        catalyst=catalyst,
        config=config,
    )
    exclusions = pd.concat(
        [swing_exclusions, intraday_exclusions],
        ignore_index=True,
    )
    blockers = _blockers(
        source_inventory=source_inventory,
        phase_capacity=phases,
        costs=costs,
        catalyst_readiness=catalyst_readiness,
        intraday=intraday,
        config=config,
    )
    return {
        "source_inventory.csv": source_inventory,
        "session_calendar.csv": session_calendar,
        "phase_capacity.csv": phases,
        "fold_capacity.csv": folds,
        "dimension_coverage.csv": dimensions,
        "cost_readiness.csv": costs,
        "catalyst_readiness.csv": catalyst_readiness,
        "exclusion_reasons.csv": exclusions,
        "blockers.csv": blockers,
    }


def _prepare_swing_rows(
    frame: pd.DataFrame,
    *,
    config: EdgeRebuildReadinessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = frame.copy()
    rows["session_date_et"] = pd.to_datetime(
        rows["session_date_et"], errors="coerce"
    ).dt.date
    rows["year"] = pd.to_datetime(
        rows["session_date_et"], errors="coerce"
    ).dt.year.astype("Int64").astype(str)
    decision = pd.to_datetime(rows["decision_time_utc"], utc=True, errors="coerce")
    available = pd.to_datetime(
        rows["feature_available_at_utc"], utc=True, errors="coerce"
    )
    masks = {
        "feature_ineligible": ~_bool_series(rows["feature_eligible"]),
        "cross_section_ineligible": ~_bool_series(
            rows["cross_section_eligible"]
        ),
        "daily_warmup_incomplete": pd.to_numeric(
            rows["daily_bar_count"], errors="coerce"
        ).lt(config.swing.minimum_daily_warmup_bars),
        "price_feed_not_sip": _normalized(rows["price_feed"]).ne(
            config.required_price_feed
        ),
        "adjustment_identity_differs": _normalized(rows["adjustment"]).ne(
            config.required_adjustment
        ),
        "feature_available_after_decision": (
            decision.isna() | available.isna() | available.gt(decision)
        ),
        "missing_identity": (
            rows["ticker"].isna()
            | rows["decision_group_id"].isna()
            | rows["session_date_et"].isna()
        ),
    }
    usable = pd.Series(True, index=rows.index)
    for mask in masks.values():
        usable &= ~mask
    rows["source_usable"] = usable
    rows["proxy_setup_eligible"] = False
    exclusions = _exclusion_frame(
        masks,
        strategy_id=config.swing.strategy_id,
        total_rows=len(rows),
    )
    return rows, exclusions


def _prepare_intraday_rows(
    frame: pd.DataFrame,
    *,
    config: EdgeRebuildReadinessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = frame.copy()
    rows["session_date_et"] = pd.to_datetime(
        rows["session_date_et"], errors="coerce"
    ).dt.date
    rows["year"] = pd.to_datetime(
        rows["session_date_et"], errors="coerce"
    ).dt.year.astype("Int64").astype(str)
    risk_on = _bool_series(rows["regime_risk_on"])
    risk_off = _bool_series(rows["regime_risk_off"])
    rows["market_regime"] = np.select(
        [risk_on & ~risk_off, risk_off & ~risk_on],
        ["risk_on", "risk_off"],
        default="neutral",
    )
    decision = pd.to_datetime(rows["decision_time_utc"], utc=True, errors="coerce")
    available = pd.to_datetime(
        rows["feature_available_at_utc"], utc=True, errors="coerce"
    )
    entry = pd.to_datetime(rows["entry_time_utc"], utc=True, errors="coerce")
    masks = {
        "feature_ineligible": ~_bool_series(rows["feature_eligible"]),
        "one_minute_history_not_exact": ~_bool_series(
            rows["one_minute_history_exact"]
        ),
        "observed_fraction_below_half": pd.to_numeric(
            rows["observed_fraction_130"], errors="coerce"
        ).lt(0.5),
        "price_feed_not_sip": _normalized(rows["price_feed"]).ne(
            config.required_price_feed
        ),
        "adjustment_identity_differs": _normalized(rows["adjustment"]).ne(
            config.required_adjustment
        ),
        "feature_available_after_decision": (
            decision.isna() | available.isna() | available.gt(decision)
        ),
        "entry_before_decision": entry.isna() | entry.lt(decision),
        "missing_identity": (
            rows["ticker"].isna()
            | rows["setup_id"].isna()
            | rows["session_date_et"].isna()
        ),
        "conflicting_market_regime": risk_on & risk_off,
    }
    usable = pd.Series(True, index=rows.index)
    for mask in masks.values():
        usable &= ~mask
    rows["source_usable"] = usable
    rows["proxy_setup_eligible"] = _bool_series(rows["label_eligible"])
    exclusions = _exclusion_frame(
        masks,
        strategy_id=config.intraday.strategy_id,
        total_rows=len(rows),
    )
    reason_counts = (
        rows.loc[
            ~_bool_series(rows["label_eligible"]),
            "label_ineligible_reason",
        ]
        .fillna("unspecified")
        .astype(str)
        .value_counts()
    )
    label_rows = pd.DataFrame(
        {
            "strategy_id": config.intraday.strategy_id,
            "reason": "proxy_label:" + reason_counts.index.astype(str),
            "excluded_rows": reason_counts.to_numpy(dtype=int),
            "total_rows": len(rows),
        }
    )
    return rows, pd.concat([exclusions, label_rows], ignore_index=True)


def _session_calendar(
    rows: pd.DataFrame,
    *,
    strategy_id: str,
    proxy_strategy_id: str,
    decision_id_column: str,
) -> pd.DataFrame:
    usable = rows.loc[rows["source_usable"]].copy()
    grouped = usable.groupby("session_date_et", sort=True, observed=True)
    calendar = grouped.agg(
        source_rows=(decision_id_column, "size"),
        unique_decision_groups=(decision_id_column, "nunique"),
        unique_tickers=("ticker", "nunique"),
        proxy_eligible_opportunities=("proxy_setup_eligible", "sum"),
    ).reset_index()
    calendar.insert(0, "strategy_id", strategy_id)
    calendar.insert(1, "proxy_strategy_id", proxy_strategy_id)
    calendar["year"] = pd.to_datetime(
        calendar["session_date_et"], errors="coerce"
    ).dt.year
    calendar["er_setup_opportunities"] = pd.NA
    calendar["er_setup_status"] = "not_built_until_ER3"
    return calendar


def _swing_phase_capacity(
    rows: pd.DataFrame,
    *,
    phases: int,
    minimum_sessions: int,
    strategy_id: str,
) -> pd.DataFrame:
    usable = rows.loc[rows["source_usable"]].copy()
    sessions = sorted(usable["session_date_et"].dropna().unique())
    phase_by_session = {
        session: index % phases for index, session in enumerate(sessions)
    }
    usable["phase"] = usable["session_date_et"].map(phase_by_session)
    grouped = usable.groupby("phase", sort=True, observed=True).agg(
        sessions=("session_date_et", "nunique"),
        source_rows=("decision_group_id", "size"),
        unique_decision_groups=("decision_group_id", "nunique"),
        unique_tickers=("ticker", "nunique"),
    )
    result = grouped.reindex(range(phases), fill_value=0).reset_index()
    result.insert(0, "strategy_id", strategy_id)
    result["minimum_sessions_required"] = minimum_sessions
    result["status"] = np.where(
        result["sessions"].ge(minimum_sessions), "pass", "blocked"
    )
    return result


def _dimension_coverage(
    rows: pd.DataFrame,
    *,
    strategy_id: str,
    dimensions: tuple[str, ...],
) -> pd.DataFrame:
    usable = rows.loc[rows["source_usable"]].copy()
    records: list[dict[str, object]] = []
    for dimension in dimensions:
        values = usable[dimension].fillna("unknown").astype(str)
        for value, indices in values.groupby(values, sort=True).groups.items():
            group = usable.loc[indices]
            records.append(
                {
                    "strategy_id": strategy_id,
                    "dimension": dimension,
                    "value": value,
                    "source_rows": len(group),
                    "sessions": int(group["session_date_et"].nunique()),
                    "tickers": int(group["ticker"].nunique()),
                    "proxy_eligible_opportunities": int(
                        group["proxy_setup_eligible"].sum()
                    ),
                }
            )
    return pd.DataFrame(records)


def _intraday_fold_capacity(
    rows: pd.DataFrame,
    *,
    folds: int,
    minimum_test_sessions: int,
    strategy_id: str,
) -> pd.DataFrame:
    sessions = np.array(
        sorted(
            rows.loc[rows["source_usable"], "session_date_et"]
            .dropna()
            .unique()
        ),
        dtype=object,
    )
    chunks = np.array_split(sessions, folds)
    records: list[dict[str, object]] = []
    for fold, chunk in enumerate(chunks):
        session_count = len(chunk)
        records.append(
            {
                "strategy_id": strategy_id,
                "fold": fold,
                "capacity_type": (
                    "chronological_test_capacity_before_ER2_purge_freeze"
                ),
                "first_test_session": str(chunk[0]) if session_count else "",
                "last_test_session": str(chunk[-1]) if session_count else "",
                "test_sessions": session_count,
                "minimum_test_sessions_required": minimum_test_sessions,
                "status": (
                    "pass"
                    if session_count >= minimum_test_sessions
                    else "blocked"
                ),
            }
        )
    return pd.DataFrame(records)


def _cost_readiness(
    *,
    swing_proxy: pd.DataFrame,
    intraday_proxy: pd.DataFrame,
    config: EdgeRebuildReadinessConfig,
) -> pd.DataFrame:
    swing_eligible = _bool_series(swing_proxy["strategy_label_eligible"])
    swing_cost = (
        pd.to_numeric(
            swing_proxy.loc[
                swing_eligible, "strategy_execution_cost_fraction"
            ],
            errors="coerce",
        )
        * 10_000
    )
    intra_eligible = _bool_series(intraday_proxy["label_eligible"])
    intra_cost = (
        pd.to_numeric(
            intraday_proxy.loc[
                intra_eligible, "path_realized_return_gross_30m"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            intraday_proxy.loc[
                intra_eligible, "path_realized_return_net_30m"
            ],
            errors="coerce",
        )
    ) * 10_000
    return pd.DataFrame(
        [
            _cost_record(
                strategy_id=config.swing.strategy_id,
                values=swing_cost,
                exact_cost_available=True,
                adverse_fill_stress_available=False,
            ),
            _cost_record(
                strategy_id=config.intraday.strategy_id,
                values=intra_cost,
                exact_cost_available=True,
                adverse_fill_stress_available=False,
            ),
        ]
    )


def _cost_record(
    *,
    strategy_id: str,
    values: pd.Series,
    exact_cost_available: bool,
    adverse_fill_stress_available: bool,
) -> dict[str, object]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "strategy_id": strategy_id,
        "rows": len(clean),
        "minimum_cost_bps": float(clean.min()) if not clean.empty else None,
        "median_cost_bps": float(clean.median()) if not clean.empty else None,
        "mean_cost_bps": float(clean.mean()) if not clean.empty else None,
        "maximum_cost_bps": float(clean.max()) if not clean.empty else None,
        "exact_cost_available": exact_cost_available,
        "adverse_fill_stress_available": adverse_fill_stress_available,
    }


def _catalyst_readiness(
    *,
    swing: VerifiedSwingSources,
    catalyst: VerifiedCatalystSources,
    config: EdgeRebuildReadinessConfig,
) -> pd.DataFrame:
    manifest = catalyst.lineage_manifest
    channel_counts = _mapping(manifest.get("channel_counts"))
    coverage_states = _mapping(_mapping(manifest.get("coverage")).get("states"))
    availability = str(
        catalyst.news_manifest.get("availability_policy", "")
    ).strip().lower()
    swing_event_time = pd.to_datetime(
        swing.proxy["latest_event_feature_available_at_utc"],
        utc=True,
        errors="coerce",
    )
    swing_decision = pd.to_datetime(
        swing.proxy["decision_time_utc"], utc=True, errors="coerce"
    )
    swing_join_violations = int(
        (swing_event_time.notna() & swing_event_time.gt(swing_decision)).sum()
    )
    records: list[dict[str, object]] = []
    for source in config.catalyst.required_source_families:
        observed = int(
            catalyst.coverage["source_family"]
            .astype(str)
            .str.lower()
            .eq(source)
            .sum()
        )
        records.append(
            _catalyst_record(
                evidence_type="source_family",
                evidence_value=source,
                observed_count=observed,
                research_ready=observed > 0,
                promotion_ready=observed > 0,
                detail=f"coverage states={dict(coverage_states)}",
            )
        )
    for channel in config.catalyst.required_relation_channels:
        observed = int(channel_counts.get(channel, 0))
        records.append(
            _catalyst_record(
                evidence_type="relation_channel",
                evidence_value=channel,
                observed_count=observed,
                research_ready=observed > 0,
                promotion_ready=observed > 0,
                detail="causal relation rows in verified catalyst lineage",
            )
        )
    records.extend(
        [
            _catalyst_record(
                evidence_type="availability",
                evidence_value=availability or "unknown",
                observed_count=int(manifest.get("source_event_rows", 0)),
                research_ready=(
                    availability
                    in config.catalyst.research_availability_policies
                ),
                promotion_ready=(
                    availability
                    in config.catalyst.promotion_availability_policies
                ),
                detail=(
                    "provider publication is a research proxy; promotion "
                    "requires observed ingestion time"
                ),
            ),
            _catalyst_record(
                evidence_type="field",
                evidence_value="sentiment",
                observed_count=int(manifest.get("training_eligible_rows", 0)),
                research_ready=bool(
                    manifest.get("training_eligible_rows", 0)
                ),
                promotion_ready=bool(
                    manifest.get("training_eligible_rows", 0)
                ),
                detail="sentiment lineage is hash-bound by catalyst request",
            ),
            _catalyst_record(
                evidence_type="decision_join",
                evidence_value=config.swing.strategy_id,
                observed_count=int(
                    pd.to_numeric(
                        swing.proxy["event_count_3d"], errors="coerce"
                    )
                    .fillna(0)
                    .gt(0)
                    .sum()
                ),
                research_ready=swing_join_violations == 0,
                promotion_ready=(
                    swing_join_violations == 0
                    and availability
                    in config.catalyst.promotion_availability_policies
                ),
                detail=f"event-time causality violations={swing_join_violations}",
            ),
            _catalyst_record(
                evidence_type="decision_join",
                evidence_value=config.intraday.strategy_id,
                observed_count=0,
                research_ready=False,
                promotion_ready=False,
                detail=(
                    "verified intraday training rows contain no catalyst "
                    "identity or availability columns"
                ),
            ),
        ]
    )
    return pd.DataFrame(records)


def _catalyst_record(
    *,
    evidence_type: str,
    evidence_value: str,
    observed_count: int,
    research_ready: bool,
    promotion_ready: bool,
    detail: str,
) -> dict[str, object]:
    return {
        "evidence_type": evidence_type,
        "evidence_value": evidence_value,
        "observed_count": observed_count,
        "research_status": "pass" if research_ready else "blocked",
        "promotion_status": "pass" if promotion_ready else "blocked",
        "detail": detail,
    }


def _source_inventory(
    *,
    swing_rows: pd.DataFrame,
    swing_proxy: pd.DataFrame,
    intraday_rows: pd.DataFrame,
    intraday: VerifiedIntradaySources,
    catalyst: VerifiedCatalystSources,
    config: EdgeRebuildReadinessConfig,
) -> pd.DataFrame:
    swing_usable = swing_rows.loc[swing_rows["source_usable"]]
    intraday_usable = intraday_rows.loc[intraday_rows["source_usable"]]
    swing_sessions = int(swing_usable["session_date_et"].nunique())
    intraday_sessions = int(
        intraday_usable["session_date_et"].nunique()
    )
    records = [
        {
            "strategy_id": config.swing.strategy_id,
            "proxy_strategy_id": config.swing.proxy_strategy_id,
            "source_role": "technical_daily_panel",
            "raw_rows": len(swing_rows),
            "source_usable_rows": len(swing_usable),
            "proxy_setup_rows": int(
                _bool_series(swing_proxy["setup_eligible"]).sum()
            ),
            "proxy_label_eligible_rows": int(
                _bool_series(
                    swing_proxy["strategy_label_eligible"]
                ).sum()
            ),
            "er_setup_opportunities": pd.NA,
            "unique_decision_groups": int(
                swing_usable["decision_group_id"].nunique()
            ),
            "unique_tickers": int(swing_usable["ticker"].nunique()),
            "valid_sessions": swing_sessions,
            "effective_session_blocks": swing_sessions
            // config.swing.proposed_horizon_sessions,
            "first_usable_decision_time_utc": _timestamp_min(
                swing_usable["decision_time_utc"]
            ),
            "last_usable_decision_time_utc": _timestamp_max(
                swing_usable["decision_time_utc"]
            ),
            "price_feed": _single_identity(swing_usable["price_feed"]),
            "adjustment": _single_identity(swing_usable["adjustment"]),
            "session_gate": (
                "pass"
                if swing_sessions >= config.swing.minimum_valid_sessions
                else "blocked"
            ),
            "exact_new_horizon_labels": False,
            "source_authority": "complete",
            "coverage_exact_rate": 1.0,
            "collection_model_data_ready": True,
        },
        {
            "strategy_id": config.intraday.strategy_id,
            "proxy_strategy_id": config.intraday.proxy_strategy_id,
            "source_role": "exact_one_minute_proxy_training",
            "raw_rows": len(intraday_rows),
            "source_usable_rows": len(intraday_usable),
            "proxy_setup_rows": len(intraday_rows),
            "proxy_label_eligible_rows": int(
                _bool_series(intraday_rows["label_eligible"]).sum()
            ),
            "er_setup_opportunities": pd.NA,
            "unique_decision_groups": int(
                intraday_usable["setup_id"].nunique()
            ),
            "unique_tickers": int(intraday_usable["ticker"].nunique()),
            "valid_sessions": intraday_sessions,
            "effective_session_blocks": intraday_sessions,
            "first_usable_decision_time_utc": _timestamp_min(
                intraday_usable["decision_time_utc"]
            ),
            "last_usable_decision_time_utc": _timestamp_max(
                intraday_usable["decision_time_utc"]
            ),
            "price_feed": str(
                intraday.collection_request.get("price_feed", "unknown")
            ),
            "adjustment": str(
                intraday.collection_request.get("adjustment", "unknown")
            ),
            "session_gate": (
                "pass"
                if intraday_sessions
                >= config.intraday.minimum_causal_sessions
                else "blocked"
            ),
            "exact_new_horizon_labels": False,
            "source_authority": "training_complete_coverage_audited",
            "coverage_exact_rate": float(
                _mapping(
                    intraday.coverage_manifest.get("summary")
                ).get("requirement_exact_rate", 0.0)
            ),
            "collection_model_data_ready": bool(
                _mapping(
                    intraday.coverage_manifest.get("summary")
                ).get("model_data_ready")
            ),
        },
        {
            "strategy_id": "CATALYST.OVERLAY",
            "proxy_strategy_id": "",
            "source_role": "alpaca_news_catalyst_lineage",
            "raw_rows": int(
                catalyst.lineage_manifest.get("source_event_rows", 0)
            ),
            "source_usable_rows": int(
                catalyst.lineage_manifest.get("training_eligible_rows", 0)
            ),
            "proxy_setup_rows": 0,
            "proxy_label_eligible_rows": 0,
            "er_setup_opportunities": pd.NA,
            "unique_decision_groups": int(
                catalyst.lineage_manifest.get("assignment_rows", 0)
            ),
            "unique_tickers": int(catalyst.coverage["ticker"].nunique()),
            "valid_sessions": 0,
            "effective_session_blocks": 0,
            "first_usable_decision_time_utc": _timestamp_min(
                catalyst.coverage["requested_start_utc"]
            ),
            "last_usable_decision_time_utc": _timestamp_max(
                catalyst.coverage["requested_end_utc"]
            ),
            "price_feed": "not_applicable",
            "adjustment": "not_applicable",
            "session_gate": "not_applicable",
            "exact_new_horizon_labels": False,
            "source_authority": "complete_research_not_promotion_ready",
            "coverage_exact_rate": float(
                catalyst.coverage["training_eligible"].astype(bool).mean()
            ),
            "collection_model_data_ready": False,
        },
    ]
    return pd.DataFrame(records)


def _blockers(
    *,
    source_inventory: pd.DataFrame,
    phase_capacity: pd.DataFrame,
    costs: pd.DataFrame,
    catalyst_readiness: pd.DataFrame,
    intraday: VerifiedIntradaySources,
    config: EdgeRebuildReadinessConfig,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    def add(
        code: str,
        scope: str,
        blocks_er2: bool,
        required_action: str,
        detail: str,
    ) -> None:
        records.append(
            {
                "blocker_code": code,
                "scope": scope,
                "blocks_er2": blocks_er2,
                "required_action": required_action,
                "detail": detail,
            }
        )

    swing = source_inventory.loc[
        source_inventory["strategy_id"].eq(config.swing.strategy_id)
    ].iloc[0]
    intra = source_inventory.loc[
        source_inventory["strategy_id"].eq(config.intraday.strategy_id)
    ].iloc[0]
    if swing["session_gate"] != "pass":
        add(
            "swing_session_history_below_gate",
            config.swing.strategy_id,
            True,
            "acquire or recover verified daily history",
            f"valid_sessions={swing['valid_sessions']}; required={config.swing.minimum_valid_sessions}",
        )
    if phase_capacity["status"].ne("pass").any():
        add(
            "swing_ten_phase_capacity_below_gate",
            config.swing.strategy_id,
            True,
            "increase verified daily session coverage",
            "one or more ten-session overlap phases have fewer than 60 sessions",
        )
    if intra["session_gate"] != "pass":
        missing = config.intraday.minimum_causal_sessions - int(
            intra["valid_sessions"]
        )
        add(
            "intraday_session_history_below_gate",
            config.intraday.strategy_id,
            True,
            (
                "collect Alpaca SIP 1Min adjustment=all for the existing "
                "ticker/benchmark universe before the first usable session"
            ),
            f"valid_sessions={intra['valid_sessions']}; required={config.intraday.minimum_causal_sessions}; missing_at_least={missing}",
        )
    coverage_summary = _mapping(
        intraday.coverage_manifest.get("summary")
    )
    if not bool(coverage_summary.get("model_data_ready")):
        add(
            "intraday_dense_clock_grid_incomplete",
            config.intraday.strategy_id,
            False,
            (
                "retain the verified causal sparse-clock policy and require "
                "observed trigger, entry, benchmark, and exit bars in ER3"
            ),
            (
                "the older all-minutes-exact gate is false; "
                f"requirement_exact_rate={coverage_summary.get('requirement_exact_rate')}; "
                "new setup rows must not impute missing trades"
            ),
        )
    for _, row in costs.loc[
        ~costs["adverse_fill_stress_available"].astype(bool)
    ].iterrows():
        add(
            "adverse_fill_stress_missing",
            str(row["strategy_id"]),
            False,
            "freeze and build adverse-fill stress labels in ER2/ER3",
            "exact stamped base costs exist; adverse-fill stress does not",
        )
    for _, row in catalyst_readiness.loc[
        catalyst_readiness["research_status"].eq("blocked")
    ].iterrows():
        add(
            "catalyst_research_evidence_missing",
            str(row["evidence_value"]),
            False,
            "build the missing causal catalyst relation or decision join in ER4",
            f"{row['evidence_type']}: {row['detail']}",
        )
    if catalyst_readiness["promotion_status"].eq("blocked").any():
        add(
            "catalyst_not_promotion_ready",
            "CATALYST.OVERLAY",
            False,
            "collect prospective first-observed timestamps before promotion",
            "provider publication proxies are research-only",
        )
    add(
        "new_strategy_setup_not_built",
        config.swing.strategy_id,
        False,
        "freeze ER2 setup then build deterministic ER3 rows and exact ten-session labels",
        "V1 proxy opportunities are reported but are not ER setup opportunities",
    )
    add(
        "new_strategy_setup_not_built",
        config.intraday.strategy_id,
        False,
        "freeze ER2 exhaustion/reclaim setup then build exact ER3 rows",
        "VWAP Reversion V1 is only a source-capacity proxy",
    )
    return pd.DataFrame(records)


def _build_summary(
    evidence: dict[str, pd.DataFrame],
    *,
    config: EdgeRebuildReadinessConfig,
    request_sha256: str,
) -> dict[str, object]:
    blockers = evidence["blockers.csv"]
    blocking = blockers.loc[blockers["blocks_er2"].astype(bool)]
    inventory = evidence["source_inventory.csv"]
    intraday = inventory.loc[
        inventory["strategy_id"].eq(config.intraday.strategy_id)
    ].iloc[0]
    return {
        "schema": contracts.READINESS_SCHEMA,
        "request_sha256": request_sha256,
        "status": (
            "blocked_pending_targeted_acquisition"
            if not blocking.empty
            else "ready_for_ER2"
        ),
        "er2_authorized": blocking.empty,
        "training_performed": False,
        "download_performed": False,
        "models_created": 0,
        "blocking_findings": len(blocking),
        "nonblocking_required_work": len(blockers) - len(blocking),
        "acquisition_plan": {
            "authorized_by_audit": not blocking.empty,
            "scope": "intraday_only",
            "provider": "alpaca",
            "feed": config.required_price_feed,
            "timeframe": config.intraday.required_timeframe,
            "adjustment": config.required_adjustment,
            "minimum_additional_sessions": max(
                0,
                config.intraday.minimum_causal_sessions
                - int(intraday["valid_sessions"]),
            ),
            "target_history_sessions": config.target_history_sessions,
            "end_before": str(intraday["first_usable_decision_time_utc"]),
            "reuse_existing_daily_and_catalyst_sources": True,
        },
        "memory": memory_audit(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
        ).to_record(),
    }


def _publish_audit(
    root: Path,
    *,
    request: dict[str, object],
    request_sha256: str,
    summary: dict[str, object],
    evidence: dict[str, pd.DataFrame],
) -> dict[str, object]:
    if root.exists():
        return load_complete_readiness_audit(
            root,
            expected_request_sha256=request_sha256,
        )
    temporary = root.with_name(f".{root.name}.{uuid4().hex}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        _write_json(temporary / "_request.json", request)
        for name, frame in evidence.items():
            frame.to_csv(temporary / name, index=False)
        _write_json(temporary / "summary.json", summary)
        manifest = {
            "schema": READINESS_RUN_SCHEMA,
            "request_sha256": request_sha256,
            "status": summary["status"],
            "artifacts": [
                {
                    "path": name,
                    "bytes": (temporary / name).stat().st_size,
                    "sha256": file_sha256(temporary / name),
                }
                for name in _OUTPUT_NAMES
            ],
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_json(temporary / "_manifest.json", manifest)
        _write_json(
            temporary / "_authority.json",
            {
                "schema": READINESS_AUTHORITY_SCHEMA,
                "state": "complete",
                "request_sha256": request_sha256,
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(
                    temporary / "_manifest.json"
                ),
            },
        )
        temporary.replace(root)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_complete_readiness_audit(
    root: Path,
    *,
    expected_request_sha256: str,
) -> dict[str, object]:
    request = _load_json(root / "_request.json")
    manifest = _load_json(root / "_manifest.json")
    authority = _load_json(root / "_authority.json")
    if (
        _json_sha256(request) != expected_request_sha256
        or manifest.get("schema") != READINESS_RUN_SCHEMA
        or manifest.get("request_sha256") != expected_request_sha256
        or authority.get("schema") != READINESS_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("request_sha256") != expected_request_sha256
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256")
        != file_sha256(root / "_manifest.json")
    ):
        raise DataReadinessError(
            "ER1 readiness output lacks matching complete authority"
        )
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise DataReadinessError("ER1 readiness manifest has no artifacts")
    expected_files = {"_authority.json", "_manifest.json"}
    for raw in raw_artifacts:
        record = _mapping(raw)
        name = str(record.get("path", ""))
        path = root / name
        expected_files.add(name)
        if (
            Path(name).name != name
            or not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or file_sha256(path) != record.get("sha256")
        ):
            raise DataReadinessError(
                f"ER1 readiness artifact does not verify: {path}"
            )
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise DataReadinessError(
            "ER1 readiness artifact file set differs from manifest"
        )
    return cast(dict[str, object], manifest)


def _read_intraday_proxy(
    verified: VerifiedTrainingBundle,
    *,
    records: tuple[dict[str, object], ...],
    columns: set[str],
) -> pd.DataFrame:
    frames = [
        _read_required_columns(
            _resolve_inside(verified.directory, str(record["path"])),
            columns,
        )
        for record in records
    ]
    frame = pd.concat(frames, ignore_index=True)
    del frames
    return frame


def _verify_intraday_coverage(
    root: Path,
    *,
    expected_collection_manifest_sha256: str,
) -> dict[str, Any]:
    manifest = _load_json(root / "_manifest.json")
    collection = _mapping(manifest.get("collection"))
    summary = _mapping(manifest.get("summary"))
    if (
        manifest.get("schema")
        != "intraday.specialist_coverage_audit.v1"
        or collection.get("manifest_sha256")
        != expected_collection_manifest_sha256
        or not manifest.get("coverage_fingerprint")
        or int(summary.get("requirements", 0)) <= 0
    ):
        raise DataReadinessError(
            "ER1 intraday coverage audit identity differs"
        )
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise DataReadinessError(
            "ER1 intraday coverage audit has no artifacts"
        )
    observed: set[str] = set()
    for raw in raw_files:
        record = _mapping(raw)
        name = str(record.get("path", ""))
        path = _resolve_inside(root, name)
        observed.add(Path(name).as_posix())
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or file_sha256(path) != record.get("sha256")
        ):
            raise DataReadinessError(
                f"ER1 intraday coverage artifact changed: {path}"
            )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.parquet")
    }
    if actual != observed:
        raise DataReadinessError(
            "ER1 intraday coverage parquet set differs from manifest"
        )
    return manifest


def _read_required_columns(path: Path, columns: set[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=sorted(columns))
    except (OSError, KeyError, ValueError) as exc:
        raise DataReadinessError(
            f"ER1 required source columns are unavailable: {path}"
        ) from exc


def _required_input_hash(
    request: dict[str, Any],
    *,
    suffix: str,
    contains: str | None = None,
) -> str:
    inputs = _mapping(request.get("inputs"))
    matches = [
        str(value)
        for key, value in inputs.items()
        if str(key).replace("\\", "/").endswith(suffix)
        and (contains is None or contains in str(key).replace("\\", "/"))
    ]
    if len(matches) != 1:
        raise DataReadinessError(
            f"ER1 expected one source input ending {suffix}; found {len(matches)}"
        )
    return matches[0]


def _resolve_inside(root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise DataReadinessError(f"unsafe ER1 artifact path: {relative}")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise DataReadinessError(f"ER1 artifact escapes source bundle: {path}")
    return path


def _exclusion_frame(
    masks: dict[str, pd.Series],
    *,
    strategy_id: str,
    total_rows: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy_id": strategy_id,
                "reason": reason,
                "excluded_rows": int(mask.fillna(True).sum()),
                "total_rows": total_rows,
            }
            for reason, mask in masks.items()
        ]
    )


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    normalized = values.fillna(False)
    if pd.api.types.is_numeric_dtype(normalized):
        return pd.to_numeric(normalized, errors="coerce").fillna(0).ne(0)
    return (
        normalized.astype(str).str.strip().str.lower().isin(
            {"1", "true", "yes"}
        )
    )


def _normalized(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.lower()


def _single_identity(values: pd.Series) -> str:
    observed = sorted(set(_normalized(values)) - {""})
    if not observed:
        return "unknown"
    if len(observed) == 1:
        return str(observed[0])
    return "mixed:" + ",".join(observed)


def _timestamp_min(values: pd.Series) -> str:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return "" if parsed.dropna().empty else parsed.min().isoformat()


def _timestamp_max(values: pd.Series) -> str:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return "" if parsed.dropna().empty else parsed.max().isoformat()


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataReadinessError("ER1 manifest field is not a mapping")
    return cast(dict[str, Any], value)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"ER1 JSON is unreadable: {path}") from exc
    return _mapping(value)


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_sha256_without_self_hash(value: dict[str, Any]) -> str:
    material = {
        key: item for key, item in value.items() if key != "request_sha256"
    }
    return _json_sha256(material)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _assert_memory(
    config: EdgeRebuildReadinessConfig,
    stage: str,
) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
