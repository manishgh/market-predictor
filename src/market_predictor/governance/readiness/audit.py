"""Read-only, hash-bound ER1 data-readiness audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.store import file_sha256
from market_predictor.core import path_integrity
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild import swing_training as swing_source_verification
from market_predictor.edge_rebuild.swing_training import (
    load_swing_candidate_authority,
    load_swing_training_config,
)
from market_predictor.edge_rebuild.training.data_io import load_swing_panel_binding
from market_predictor.edge_rebuild.training.swing_types import SwingPanelBinding
from market_predictor.evidence import readiness_authority
from market_predictor.evidence.readiness_authority import (
    CURRENT_RUN_SCHEMA,
    current_exchange_calendar_identity,
    publish_readiness_authority,
)
from market_predictor.governance.promotion import (
    bundle_verification as promotion_verification,
)
from market_predictor.governance.promotion.bundle_contracts import (
    PromotedSwingBundle,
)
from market_predictor.governance.promotion.bundle_verification import (
    validate_file_backed_promoted_bundle,
)
from market_predictor.governance.readiness.contracts import (
    PredictionDataReadinessConfig,
)
from market_predictor.intraday import (
    specialist_experiments as intraday_source_verification,
)
from market_predictor.intraday.specialist_contracts import (
    IntradaySpecialistResearchConfig,
)
from market_predictor.intraday.specialist_experiments import (
    VerifiedTrainingBundle,
    verify_intraday_specialist_training_bundle,
)
from market_predictor.modeling.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    process_memory_snapshot,
    release_process_memory,
)
from market_predictor.swing import catalyst_lineage as catalyst_source_verification
from market_predictor.swing.features.panel import (
    SWING_FEATURE_PROFILE,
)

_PROMOTED_BUNDLE_NAME = "bundle.json"
_SWING_TECHNICAL_READINESS_COLUMNS = (
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
    "membership_available_at_utc",
    "membership_effective_from_utc",
    "membership_effective_to_utc",
    "price_feed",
    "round_trip_cost_bps",
    "sector",
    "session_date_et",
    "ticker",
    "universe_snapshot_id",
)
_SWING_PANEL_READINESS_COLUMNS = tuple(
    dict.fromkeys(
        (
            *_SWING_TECHNICAL_READINESS_COLUMNS,
            "barrier_cost",
            "barrier_gross_return",
            "barrier_net_return",
            "entry_time_utc",
            "future_excess_return_10d_vs_sector",
            "future_excess_return_10d_vs_spy",
            "future_sector_return_10d",
            "future_spy_return_10d",
            "label_available_at_utc",
            "barrier_label_available_at_utc",
        )
    )
)


@dataclass(frozen=True)
class VerifiedSwingSources:
    technical: pd.DataFrame
    proxy: pd.DataFrame
    identity: dict[str, object]
    panel_manifest: dict[str, Any]
    candidate: dict[str, Any]
    promoted_bundle: PromotedSwingBundle | None


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
    feature_inventory: dict[str, Any]
    news_manifest: dict[str, Any]
    coverage: pd.DataFrame


@dataclass(frozen=True)
class SwingReadinessSlice:
    identity: dict[str, object]
    candidate_status: str
    promoted_bundle_available: bool
    session_calendar: pd.DataFrame
    phase_capacity: pd.DataFrame
    dimension_coverage: pd.DataFrame
    cost_readiness: pd.DataFrame
    exclusions: pd.DataFrame
    source_inventory: pd.DataFrame


@dataclass(frozen=True)
class IntradayReadinessSlice:
    identity: dict[str, object]
    coverage_summary: dict[str, Any]
    session_calendar: pd.DataFrame
    fold_capacity: pd.DataFrame
    dimension_coverage: pd.DataFrame
    cost_readiness: pd.DataFrame
    exclusions: pd.DataFrame
    source_inventory: pd.DataFrame


@dataclass(frozen=True)
class CatalystReadinessSlice:
    identity: dict[str, object]
    readiness: pd.DataFrame
    source_inventory: pd.DataFrame


def run_prediction_data_readiness_audit(
    *,
    swing_panel_dir: Path,
    swing_candidate_dir: Path,
    swing_promoted_bundle_dir: Path | None,
    promotion_attestation_trust_store_path: Path | None,
    promotion_gate_policy_sha256: str | None,
    intraday_training_dir: Path,
    intraday_collection_dir: Path,
    intraday_coverage_dir: Path,
    catalyst_lineage_dir: Path,
    news_source_dir: Path,
    out_dir: Path,
    config: PredictionDataReadinessConfig,
    policy_path: Path,
    swing_training_policy_path: Path,
    strategy_contract_path: Path,
    intraday_config: IntradaySpecialistResearchConfig,
    intraday_policy_path: Path,
) -> dict[str, object]:
    """Verify reusable sources and publish the immutable ER1 audit."""

    _assert_memory(config, "ER1 source verification")
    swing = _verify_swing_sources(
        panel_dir=swing_panel_dir,
        candidate_dir=swing_candidate_dir,
        promoted_bundle_dir=swing_promoted_bundle_dir,
        promotion_attestation_trust_store_path=(promotion_attestation_trust_store_path),
        promotion_gate_policy_sha256=promotion_gate_policy_sha256,
        training_policy_path=swing_training_policy_path,
        strategy_contract_path=strategy_contract_path,
        config=config,
    )
    _assert_memory(config, "ER1 swing verification")
    swing_slice = _summarize_swing_readiness(swing, config=config)
    del swing
    release_process_memory()
    _assert_memory(config, "ER1 swing evidence reduction")
    intraday = _verify_intraday_sources(
        training_dir=intraday_training_dir,
        collection_dir=intraday_collection_dir,
        coverage_dir=intraday_coverage_dir,
        policy_path=intraday_policy_path,
        intraday_config=intraday_config,
        config=config,
    )
    _assert_memory(config, "ER1 intraday verification")
    intraday_slice = _summarize_intraday_readiness(intraday, config=config)
    del intraday
    release_process_memory()
    _assert_memory(config, "ER1 intraday evidence reduction")
    catalyst = _verify_catalyst_sources(
        lineage_dir=catalyst_lineage_dir,
        news_dir=news_source_dir,
    )
    _assert_memory(config, "ER1 catalyst verification")
    catalyst_slice = _summarize_catalyst_readiness(catalyst, config=config)
    del catalyst
    release_process_memory()
    _assert_memory(config, "ER1 catalyst evidence reduction")

    evidence = _combine_readiness_evidence(
        swing=swing_slice,
        intraday=intraday_slice,
        catalyst=catalyst_slice,
        config=config,
    )
    assert_peak_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage="readiness evidence construction",
    )
    request: dict[str, object] = {
        "schema": CURRENT_RUN_SCHEMA,
        "policy_sha256": config.sha256(),
        "policy_file_sha256": file_sha256(policy_path),
        "swing_training_policy_file_sha256": file_sha256(swing_training_policy_path),
        "strategy_contract_file_sha256": file_sha256(strategy_contract_path),
        "intraday_policy_file_sha256": file_sha256(intraday_policy_path),
        "sources": {
            "swing": swing_slice.identity,
            "intraday": intraday_slice.identity,
            "catalyst": catalyst_slice.identity,
        },
        "implementation": readiness_implementation_identity(),
        "exchange_calendar": current_exchange_calendar_identity(),
        "training_performed": False,
        "download_performed": False,
    }
    request_sha256 = _json_sha256_without_self_hash(request)
    summary = _build_summary(
        evidence,
        config=config,
        request_sha256=request_sha256,
    )
    published = publish_readiness_authority(
        output_directory=out_dir,
        request=request,
        request_sha256=request_sha256,
        summary=summary,
        evidence=evidence,
    )
    del swing_slice, intraday_slice, catalyst_slice, evidence
    release_process_memory()
    _assert_memory(config, "ER1 publication")
    assert_peak_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage="readiness publication",
    )
    return dict(published.manifest)


def readiness_implementation_identity() -> dict[str, object]:
    files = {
        "authority": Path(readiness_authority.__file__).resolve(),
        "catalyst_source_verification": Path(catalyst_source_verification.__file__).resolve(),
        "contracts": Path(__file__).with_name("contracts.py").resolve(),
        "intraday_source_verification": Path(intraday_source_verification.__file__).resolve(),
        "promotion_verification": Path(promotion_verification.__file__).resolve(),
        "readiness": Path(__file__).resolve(),
        "swing_source_verification": Path(swing_source_verification.__file__).resolve(),
    }
    return {name: {"path": path.name, "sha256": file_sha256(path)} for name, path in sorted(files.items())}


def _verify_swing_sources(
    *,
    panel_dir: Path,
    candidate_dir: Path,
    promoted_bundle_dir: Path | None,
    promotion_attestation_trust_store_path: Path | None,
    promotion_gate_policy_sha256: str | None,
    training_policy_path: Path,
    strategy_contract_path: Path,
    config: PredictionDataReadinessConfig,
) -> VerifiedSwingSources:
    strategy_contract = load_strategy_contract(strategy_contract_path)
    training_config = load_swing_training_config(training_policy_path)
    binding = load_swing_panel_binding(
        panel_dir,
        strategy_contract=strategy_contract,
        config=training_config,
    )
    if strategy_contract.swing.strategy_id != config.swing.strategy_id:
        raise DataReadinessError("ER1 swing strategy identity differs")
    candidate_root = path_integrity.verify_tree_containment(
        candidate_dir,
        label="swing candidate authority",
    )
    candidate = load_swing_candidate_authority(candidate_root)
    _verify_candidate_panel_binding(candidate, binding)
    promoted = _load_optional_promoted_swing_bundle(
        promoted_bundle_dir,
        strategy_contract=strategy_contract,
        candidate=candidate,
        attestation_trust_store_path=promotion_attestation_trust_store_path,
        promotion_gate_policy_sha256=promotion_gate_policy_sha256,
    )
    panel = _read_current_swing_panel(binding, config=config)
    technical_frame = panel.loc[:, _SWING_TECHNICAL_READINESS_COLUMNS].copy()
    proxy_frame = _current_swing_outcome_proxy(panel)
    del panel
    return VerifiedSwingSources(
        technical=technical_frame,
        proxy=proxy_frame,
        identity={
            "type": "verified_edge_rebuild_swing_sources",
            "strategy_id": config.swing.strategy_id,
            "horizon_sessions": 10,
            "panel_manifest_sha256": binding.manifest_sha256,
            "panel_authority_sha256": binding.authority_sha256,
            "panel_request_sha256": binding.request_sha256,
            "candidate_status": str(candidate["status"]),
            "candidate_id": candidate.get("candidate_id"),
            "candidate_authority_sha256": file_sha256(
                path_integrity.resolve_existing_file_inside(
                    candidate_root,
                    "_authority.json",
                    label="swing candidate authority artifact",
                )
            ),
            "promoted_bundle_status": ("verified" if promoted is not None else "unavailable"),
            "promoted_bundle_sha256": (promoted.sha256() if promoted is not None else None),
            "technical_rows": len(technical_frame),
            "proxy_rows": len(proxy_frame),
        },
        panel_manifest=dict(binding.manifest),
        candidate=candidate,
        promoted_bundle=promoted,
    )


def _verify_candidate_panel_binding(
    candidate: Mapping[str, Any],
    binding: SwingPanelBinding,
) -> None:
    model_card = _mapping(candidate.get("model_card"))
    dataset = _mapping(model_card.get("dataset"))
    expected = {
        "panel_manifest_sha256": binding.manifest_sha256,
        "panel_authority_sha256": binding.authority_sha256,
        "panel_request_sha256": binding.request_sha256,
        "strategy_contract_sha256": binding.strategy_contract_sha256,
    }
    mismatches = sorted(name for name, value in expected.items() if dataset.get(name) != value)
    if mismatches:
        raise DataReadinessError("swing candidate does not bind the active panel: " + ", ".join(mismatches))


def _load_optional_promoted_swing_bundle(
    directory: Path | None,
    *,
    strategy_contract: StrategyContract,
    candidate: Mapping[str, Any],
    attestation_trust_store_path: Path | None,
    promotion_gate_policy_sha256: str | None,
) -> PromotedSwingBundle | None:
    if directory is None:
        if attestation_trust_store_path is not None or promotion_gate_policy_sha256 is not None:
            raise DataReadinessError("promotion verification inputs require a promoted swing bundle")
        return None
    if attestation_trust_store_path is None or promotion_gate_policy_sha256 is None:
        raise DataReadinessError("a promoted swing bundle requires its attestation trust store and promotion gate-policy hash")
    root = path_integrity.verify_tree_containment(
        directory,
        label="promoted swing bundle",
    )
    payload = _load_json(
        path_integrity.resolve_existing_file_inside(
            root,
            _PROMOTED_BUNDLE_NAME,
            label="promoted swing bundle metadata",
        )
    )
    bundle = validate_file_backed_promoted_bundle(
        payload,
        bundle_root=root,
        strategy_contract=strategy_contract,
        attestation_trust_store_path=attestation_trust_store_path,
        promotion_gate_policy_sha256=promotion_gate_policy_sha256,
        expected_mode="swing",
    )
    if candidate.get("status") != "candidate":
        raise DataReadinessError("a promoted swing bundle requires a verified training candidate")
    candidate_files = _mapping(_mapping(candidate.get("manifest")).get("files"))
    candidate_model = _mapping(candidate_files.get("candidate.joblib"))
    if bundle.model_artifact_sha256 != candidate_model.get("sha256"):
        raise DataReadinessError("promoted swing bundle does not bind the verified candidate model")
    return bundle


def _read_current_swing_panel(
    binding: SwingPanelBinding,
    *,
    config: PredictionDataReadinessConfig,
) -> pd.DataFrame:
    files_by_profile = _mapping(binding.manifest.get("files_by_profile"))
    records = files_by_profile.get(SWING_FEATURE_PROFILE)
    if not isinstance(records, list) or not records:
        raise DataReadinessError("swing panel has no technical profile for readiness")
    paths: list[Path] = []
    normalized_records: list[dict[str, Any]] = []
    for raw in records:
        record = _mapping(raw)
        path = _resolve_inside(
            binding.root / "final",
            str(record.get("path", "")),
        )
        paths.append(path)
        normalized_records.append(record)
    _assert_projected_parquet_load(
        paths,
        config=config,
        stage="swing readiness panel",
    )
    parts: list[pd.DataFrame] = []
    for record, path in zip(normalized_records, paths, strict=True):
        _assert_memory(config, f"swing readiness partition {path.name} before load")
        part = _read_required_columns(
            path,
            set(_SWING_PANEL_READINESS_COLUMNS),
        )
        if len(part) != int(record.get("rows", -1)):
            raise DataReadinessError("swing panel readiness partition row count differs")
        parts.append(part)
        _assert_memory(config, f"swing readiness partition {path.name} after load")
    _assert_projected_dataframe_concat(parts, config=config, stage="swing readiness panel concat")
    frame = pd.concat(parts, ignore_index=True)
    _assert_memory(config, "swing readiness panel concat")
    expected_rows = int(binding.manifest.get("rows", -1))
    if len(frame) != expected_rows:
        raise DataReadinessError("swing panel readiness row count differs")
    return frame


def _current_swing_outcome_proxy(panel: pd.DataFrame) -> pd.DataFrame:
    feature_eligible = _bool_series(panel["feature_eligible"])
    cross_section_eligible = _bool_series(panel["cross_section_eligible"])
    label_eligible = _bool_series(panel["label_eligible"])
    return pd.DataFrame(
        {
            "adjustment": panel["adjustment"],
            "decision_time_utc": panel["decision_time_utc"],
            "entry_time_utc": panel["entry_time_utc"],
            "exit_time_utc": panel["label_available_at_utc"],
            "price_feed": panel["price_feed"],
            "setup_eligible": feature_eligible & cross_section_eligible,
            "strategy_decision_group_id": panel["decision_group_id"],
            "strategy_execution_cost_fraction": panel["barrier_cost"],
            "strategy_excess_return_vs_sector": panel["future_excess_return_10d_vs_sector"],
            "strategy_excess_return_vs_spy": panel["future_excess_return_10d_vs_spy"],
            "strategy_gross_return": panel["barrier_gross_return"],
            "strategy_label_eligible": label_eligible,
            "strategy_net_return": panel["barrier_net_return"],
            "strategy_sector_return": panel["future_sector_return_10d"],
            "strategy_spy_return": panel["future_spy_return_10d"],
            "ticker": panel["ticker"],
        }
    )


def _verify_intraday_sources(
    *,
    training_dir: Path,
    collection_dir: Path,
    coverage_dir: Path,
    policy_path: Path,
    intraday_config: IntradaySpecialistResearchConfig,
    config: PredictionDataReadinessConfig,
) -> VerifiedIntradaySources:
    verified = verify_intraday_specialist_training_bundle(
        training_dir,
        config=intraday_config,
        policy_path=policy_path,
    )
    collection_root = path_integrity.verify_no_reparse_ancestry(
        collection_dir,
        label="intraday collection",
    )
    collection_manifest_path = path_integrity.resolve_existing_file_inside(
        collection_root,
        "_manifest.json",
        label="intraday collection manifest",
    )
    collection_request_path = path_integrity.resolve_existing_file_inside(
        collection_root,
        "_request.json",
        label="intraday collection request",
    )
    collection_manifest = _load_json(collection_manifest_path)
    collection_request = _load_json(collection_request_path)
    expected_collection = _mapping(verified.manifest.get("collection"))
    if (
        file_sha256(collection_manifest_path) != expected_collection.get("manifest_sha256")
        or collection_manifest.get("request_sha256") != expected_collection.get("request_sha256")
        or collection_request.get("request_sha256") != expected_collection.get("request_sha256")
        or _json_sha256_without_self_hash(collection_request) != expected_collection.get("request_sha256")
        or collection_request.get("price_feed") != config.required_price_feed
        or collection_request.get("adjustment") != config.required_adjustment
        or collection_request.get("timeframe") != config.intraday.required_timeframe
    ):
        raise DataReadinessError("ER1 intraday collection identity differs")
    coverage_root = path_integrity.verify_tree_containment(
        coverage_dir,
        label="intraday coverage authority",
    )
    coverage_manifest_path = path_integrity.resolve_existing_file_inside(
        coverage_root,
        "_manifest.json",
        label="intraday coverage manifest",
    )
    coverage_manifest = _verify_intraday_coverage(
        coverage_root,
        expected_collection_manifest_sha256=file_sha256(collection_manifest_path),
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
        "universe_snapshot_id",
        "exit_time_utc",
        "path_sector_return_30m",
        "path_spy_return_30m",
    }
    proxy = _read_intraday_proxy(
        verified,
        records=records,
        columns=columns,
        config=config,
    )
    return VerifiedIntradaySources(
        proxy=proxy,
        identity={
            "type": "verified_intraday_specialist_training_sources",
            "strategy_id": config.intraday.strategy_id,
            "proxy_strategy_id": config.intraday.proxy_strategy_id,
            "training_manifest_sha256": verified.manifest_sha256,
            "dataset_fingerprint": verified.dataset_fingerprint,
            "proxy_dataset_sha256": verified.strategy_dataset_sha256[config.intraday.proxy_strategy_id],
            "collection_manifest_sha256": file_sha256(collection_manifest_path),
            "collection_request_sha256": str(collection_request["request_sha256"]),
            "coverage_manifest_sha256": file_sha256(coverage_manifest_path),
            "coverage_fingerprint": str(coverage_manifest["coverage_fingerprint"]),
            "collection_rows": int(collection_manifest.get("total_rows", 0)),
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
) -> VerifiedCatalystSources:
    verified = catalyst_source_verification.verify_completed_catalyst_lineage(
        lineage_dir,
    )
    lineage_manifest = dict(verified.manifest)
    lineage_request = dict(verified.request)
    news_root = path_integrity.verify_no_reparse_ancestry(
        news_dir,
        label="catalyst news source",
    ).resolve(strict=True)
    news_manifest_path = path_integrity.resolve_existing_file_inside(
        news_root,
        "_manifest.json",
        label="catalyst news manifest",
    )
    news_manifest = _load_json(news_manifest_path)
    lineage_availability = str(
        verified.feature_inventory.get("availability_policy", "")
    ).strip()
    if (
        file_sha256(news_manifest_path) != lineage_request.get("collection_manifest_sha256")
        or news_manifest.get("status") != "complete"
        or news_manifest.get("request_sha256") is None
        or news_manifest.get("availability_policy") != lineage_availability
    ):
        raise DataReadinessError("ER1 catalyst news source does not verify")
    source_event_rows = lineage_manifest.get("source_event_rows")
    if not isinstance(source_event_rows, int) or isinstance(source_event_rows, bool):
        raise DataReadinessError("ER1 catalyst source event count is invalid")
    return VerifiedCatalystSources(
        identity={
            "type": "verified_catalyst_lineage_sources",
            "lineage_manifest_sha256": verified.manifest_sha256,
            "lineage_request_sha256": verified.request_sha256,
            "news_manifest_sha256": file_sha256(news_manifest_path),
            "news_request_sha256": str(news_manifest["request_sha256"]),
            "coverage_sha256": verified.coverage_sha256,
            "source_event_rows": source_event_rows,
        },
        lineage_manifest=lineage_manifest,
        feature_inventory=dict(verified.feature_inventory),
        news_manifest=news_manifest,
        coverage=verified.coverage,
    )


def _summarize_swing_readiness(
    swing: VerifiedSwingSources,
    *,
    config: PredictionDataReadinessConfig,
) -> SwingReadinessSlice:
    rows, exclusions = _prepare_swing_rows(swing.technical, config=config)
    _assert_memory(config, "swing readiness row preparation")
    _assert_projected_dataframe_concat(
        [rows],
        config=config,
        stage="swing readiness aggregation",
    )
    eligible = _bool_series(swing.proxy["strategy_label_eligible"])
    cost_bps = pd.to_numeric(
        swing.proxy.loc[eligible, "strategy_execution_cost_fraction"],
        errors="coerce",
    ) * 10_000
    result = SwingReadinessSlice(
        identity=swing.identity,
        candidate_status=str(swing.candidate.get("status", "")),
        promoted_bundle_available=swing.promoted_bundle is not None,
        session_calendar=_session_calendar(
            rows,
            strategy_id=config.swing.strategy_id,
            proxy_strategy_id=config.swing.strategy_id,
            decision_id_column="decision_group_id",
        ),
        phase_capacity=_swing_phase_capacity(
            rows,
            phases=config.swing.non_overlapping_phases,
            minimum_sessions=config.swing.minimum_sessions_per_phase,
            strategy_id=config.swing.strategy_id,
        ),
        dimension_coverage=_dimension_coverage(
            rows,
            strategy_id=config.swing.strategy_id,
            dimensions=("year", "market_regime", "sector", "price_feed"),
        ),
        cost_readiness=pd.DataFrame(
            [
                _cost_record(
                    strategy_id=config.swing.strategy_id,
                    values=cost_bps,
                    exact_cost_available=True,
                    adverse_fill_stress_available=False,
                )
            ]
        ),
        exclusions=exclusions,
        source_inventory=pd.DataFrame(
            [_swing_source_inventory_record(rows, swing.proxy, config=config)]
        ),
    )
    _assert_memory(config, "swing readiness evidence aggregation")
    return result


def _summarize_intraday_readiness(
    intraday: VerifiedIntradaySources,
    *,
    config: PredictionDataReadinessConfig,
) -> IntradayReadinessSlice:
    rows, exclusions = _prepare_intraday_rows(intraday.proxy, config=config)
    _assert_memory(config, "intraday readiness row preparation")
    _assert_projected_dataframe_concat(
        [rows],
        config=config,
        stage="intraday readiness aggregation",
    )
    eligible = _bool_series(intraday.proxy["label_eligible"])
    cost_bps = (
        pd.to_numeric(
            intraday.proxy.loc[eligible, "path_realized_return_gross_30m"],
            errors="coerce",
        )
        - pd.to_numeric(
            intraday.proxy.loc[eligible, "path_realized_return_net_30m"],
            errors="coerce",
        )
    ) * 10_000
    result = IntradayReadinessSlice(
        identity=intraday.identity,
        coverage_summary=_mapping(intraday.coverage_manifest.get("summary")),
        session_calendar=_session_calendar(
            rows,
            strategy_id=config.intraday.strategy_id,
            proxy_strategy_id=config.intraday.proxy_strategy_id,
            decision_id_column="setup_id",
        ),
        fold_capacity=_intraday_fold_capacity(
            rows,
            folds=config.intraday.required_purged_folds,
            minimum_test_sessions=config.intraday.minimum_test_sessions_per_fold,
            strategy_id=config.intraday.strategy_id,
        ),
        dimension_coverage=_dimension_coverage(
            rows,
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
        cost_readiness=pd.DataFrame(
            [
                _cost_record(
                    strategy_id=config.intraday.strategy_id,
                    values=cost_bps,
                    exact_cost_available=True,
                    adverse_fill_stress_available=False,
                )
            ]
        ),
        exclusions=exclusions,
        source_inventory=pd.DataFrame(
            [_intraday_source_inventory_record(rows, intraday, config=config)]
        ),
    )
    _assert_memory(config, "intraday readiness evidence aggregation")
    return result


def _summarize_catalyst_readiness(
    catalyst: VerifiedCatalystSources,
    *,
    config: PredictionDataReadinessConfig,
) -> CatalystReadinessSlice:
    return CatalystReadinessSlice(
        identity=catalyst.identity,
        readiness=_catalyst_readiness(catalyst=catalyst, config=config),
        source_inventory=pd.DataFrame([_catalyst_source_inventory_record(catalyst)]),
    )


def _combine_readiness_evidence(
    *,
    swing: SwingReadinessSlice,
    intraday: IntradayReadinessSlice,
    catalyst: CatalystReadinessSlice,
    config: PredictionDataReadinessConfig,
) -> dict[str, pd.DataFrame]:
    groups = (
        [swing.session_calendar, intraday.session_calendar],
        [swing.dimension_coverage, intraday.dimension_coverage],
        [swing.cost_readiness, intraday.cost_readiness],
        [swing.exclusions, intraday.exclusions],
        [swing.source_inventory, intraday.source_inventory, catalyst.source_inventory],
    )
    for frames in groups:
        _assert_projected_dataframe_concat(
            frames,
            config=config,
            stage="readiness evidence concat",
        )
    session_calendar, dimensions, costs, exclusions, source_inventory = (
        pd.concat(frames, ignore_index=True) for frames in groups
    )
    _assert_memory(config, "readiness evidence concat")
    blockers = _blockers(
        source_inventory=source_inventory,
        phase_capacity=swing.phase_capacity,
        costs=costs,
        catalyst_readiness=catalyst.readiness,
        swing_candidate_status=swing.candidate_status,
        promoted_bundle_available=swing.promoted_bundle_available,
        intraday_coverage_summary=intraday.coverage_summary,
        config=config,
    )
    return {
        "source_inventory.csv": source_inventory,
        "session_calendar.csv": session_calendar,
        "phase_capacity.csv": swing.phase_capacity,
        "fold_capacity.csv": intraday.fold_capacity,
        "dimension_coverage.csv": dimensions,
        "cost_readiness.csv": costs,
        "catalyst_readiness.csv": catalyst.readiness,
        "exclusion_reasons.csv": exclusions,
        "blockers.csv": blockers,
    }


def _prepare_swing_rows(
    frame: pd.DataFrame,
    *,
    config: PredictionDataReadinessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = frame.copy(deep=False)
    rows["session_date_et"] = pd.to_datetime(rows["session_date_et"], errors="coerce").dt.date
    rows["year"] = pd.to_datetime(rows["session_date_et"], errors="coerce").dt.year.astype("Int64").astype(str)
    decision = pd.to_datetime(rows["decision_time_utc"], utc=True, errors="coerce")
    available = pd.to_datetime(rows["feature_available_at_utc"], utc=True, errors="coerce")
    membership_available = pd.to_datetime(rows["membership_available_at_utc"], utc=True, errors="coerce")
    membership_from = pd.to_datetime(rows["membership_effective_from_utc"], utc=True, errors="coerce")
    membership_to = pd.to_datetime(rows["membership_effective_to_utc"], utc=True, errors="coerce")
    masks = {
        "feature_ineligible": ~_bool_series(rows["feature_eligible"]),
        "cross_section_ineligible": ~_bool_series(rows["cross_section_eligible"]),
        "daily_warmup_incomplete": pd.to_numeric(rows["daily_bar_count"], errors="coerce").lt(config.swing.minimum_daily_warmup_bars),
        "price_feed_not_sip": _normalized(rows["price_feed"]).ne(config.required_price_feed),
        "adjustment_identity_differs": _normalized(rows["adjustment"]).ne(config.required_adjustment),
        "feature_available_after_decision": (decision.isna() | available.isna() | available.gt(decision)),
        "membership_unavailable_at_decision": (membership_available.isna() | membership_available.gt(decision)),
        "membership_interval_invalid": (
            membership_from.isna() | membership_from.gt(decision) | (membership_to.notna() & membership_to.lt(decision))
        ),
        "missing_identity": (
            rows["ticker"].isna() | rows["decision_group_id"].isna() | rows["session_date_et"].isna() | rows["universe_snapshot_id"].isna()
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
    config: PredictionDataReadinessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = frame.copy(deep=False)
    rows["session_date_et"] = pd.to_datetime(rows["session_date_et"], errors="coerce").dt.date
    rows["year"] = pd.to_datetime(rows["session_date_et"], errors="coerce").dt.year.astype("Int64").astype(str)
    risk_on = _bool_series(rows["regime_risk_on"])
    risk_off = _bool_series(rows["regime_risk_off"])
    rows["market_regime"] = np.select(
        [risk_on & ~risk_off, risk_off & ~risk_on],
        ["risk_on", "risk_off"],
        default="neutral",
    )
    decision = pd.to_datetime(rows["decision_time_utc"], utc=True, errors="coerce")
    available = pd.to_datetime(rows["feature_available_at_utc"], utc=True, errors="coerce")
    entry = pd.to_datetime(rows["entry_time_utc"], utc=True, errors="coerce")
    masks = {
        "feature_ineligible": ~_bool_series(rows["feature_eligible"]),
        "one_minute_history_not_exact": ~_bool_series(rows["one_minute_history_exact"]),
        "observed_fraction_below_half": pd.to_numeric(rows["observed_fraction_130"], errors="coerce").lt(0.5),
        "price_feed_not_sip": _normalized(rows["price_feed"]).ne(config.required_price_feed),
        "adjustment_identity_differs": _normalized(rows["adjustment"]).ne(config.required_adjustment),
        "feature_available_after_decision": (decision.isna() | available.isna() | available.gt(decision)),
        "entry_before_decision": entry.isna() | entry.lt(decision),
        "missing_identity": (
            rows["ticker"].isna() | rows["setup_id"].isna() | rows["session_date_et"].isna() | rows["universe_snapshot_id"].isna()
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
    calendar["year"] = pd.to_datetime(calendar["session_date_et"], errors="coerce").dt.year
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
    phase_by_session = {session: index % phases for index, session in enumerate(sessions)}
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
    result["status"] = np.where(result["sessions"].ge(minimum_sessions), "pass", "blocked")
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
                    "proxy_eligible_opportunities": int(group["proxy_setup_eligible"].sum()),
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
        sorted(rows.loc[rows["source_usable"], "session_date_et"].dropna().unique()),
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
                "capacity_type": ("chronological_test_capacity_before_ER2_purge_freeze"),
                "first_test_session": str(chunk[0]) if session_count else "",
                "last_test_session": str(chunk[-1]) if session_count else "",
                "test_sessions": session_count,
                "minimum_test_sessions_required": minimum_test_sessions,
                "status": ("pass" if session_count >= minimum_test_sessions else "blocked"),
            }
        )
    return pd.DataFrame(records)


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


def _swing_benchmark_status(frame: pd.DataFrame) -> str:
    eligible = _bool_series(frame["strategy_label_eligible"])
    return _benchmark_status(
        frame.loc[eligible],
        entry_column="entry_time_utc",
        exit_column="exit_time_utc",
        numeric_columns=(
            "strategy_spy_return",
            "strategy_sector_return",
            "strategy_excess_return_vs_spy",
            "strategy_excess_return_vs_sector",
        ),
    )


def _intraday_benchmark_status(frame: pd.DataFrame) -> str:
    eligible = _bool_series(frame["label_eligible"])
    return _benchmark_status(
        frame.loc[eligible],
        entry_column="entry_time_utc",
        exit_column="exit_time_utc",
        numeric_columns=(
            "path_spy_return_30m",
            "path_sector_return_30m",
            "path_excess_return_30m_vs_spy",
            "path_excess_return_30m_vs_sector",
        ),
    )


def _benchmark_status(
    frame: pd.DataFrame,
    *,
    entry_column: str,
    exit_column: str,
    numeric_columns: tuple[str, ...],
) -> str:
    if frame.empty:
        return "blocked:no_eligible_proxy_labels"
    entry = pd.to_datetime(frame[entry_column], utc=True, errors="coerce")
    exit_time = pd.to_datetime(frame[exit_column], utc=True, errors="coerce")
    if entry.isna().any() or exit_time.isna().any() or exit_time.le(entry).any():
        return "blocked:invalid_entry_exit_interval"
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            return f"blocked:non_finite_{column}"
    return "verified_proxy_labels"


def _catalyst_readiness(
    *,
    catalyst: VerifiedCatalystSources,
    config: PredictionDataReadinessConfig,
) -> pd.DataFrame:
    manifest = catalyst.lineage_manifest
    channel_counts = _mapping(manifest.get("channel_counts"))
    coverage_states = _mapping(_mapping(manifest.get("coverage")).get("states"))
    availability = str(
        catalyst.feature_inventory.get("availability_policy", "")
    ).strip().lower()
    training_channels = {
        str(value)
        for value in catalyst.feature_inventory.get("training_eligible_channels", [])
    }
    promotion_availability = availability in config.catalyst.promotion_availability_policies
    coverage = catalyst.coverage
    eligible_coverage = coverage.loc[
        _bool_series(coverage["training_eligible"])
        & _bool_series(coverage["missingness_known"])
        & _normalized(coverage["coverage_state"]).isin(("observed_complete", "observed_empty"))
        & _normalized(coverage["status"]).isin(("observed", "observed_empty"))
    ]
    records: list[dict[str, object]] = []
    for source in config.catalyst.required_source_families:
        observed = int(eligible_coverage["source_family"].astype(str).str.lower().eq(source).sum())
        records.append(
            _catalyst_record(
                evidence_type="source_family",
                evidence_value=source,
                observed_count=observed,
                research_ready=observed > 0,
                promotion_ready=observed > 0 and promotion_availability,
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
                promotion_ready=(
                    observed > 0
                    and channel in training_channels
                    and promotion_availability
                ),
                detail="causal relation rows in verified catalyst lineage",
            )
        )
    records.extend(
        [
            _catalyst_record(
                evidence_type="availability",
                evidence_value=availability or "unknown",
                observed_count=int(manifest.get("source_event_rows", 0)),
                research_ready=(availability in config.catalyst.research_availability_policies),
                promotion_ready=(availability in config.catalyst.promotion_availability_policies),
                detail=("provider publication is a research proxy; promotion requires observed ingestion time"),
            ),
            _catalyst_record(
                evidence_type="field",
                evidence_value="sentiment",
                observed_count=int(manifest.get("training_eligible_rows", 0)),
                research_ready=bool(manifest.get("training_eligible_rows", 0)),
                promotion_ready=(
                    bool(manifest.get("training_eligible_rows", 0))
                    and promotion_availability
                ),
                detail="sentiment lineage is hash-bound by catalyst request",
            ),
            _catalyst_record(
                evidence_type="decision_join",
                evidence_value=config.swing.strategy_id,
                observed_count=0,
                research_ready=False,
                promotion_ready=False,
                detail=("technical swing panel intentionally has no event attachment; a verified event-specialist authority is required"),
            ),
            _catalyst_record(
                evidence_type="decision_join",
                evidence_value=config.intraday.strategy_id,
                observed_count=0,
                research_ready=False,
                promotion_ready=False,
                detail=("verified intraday training rows contain no catalyst identity or availability columns"),
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


def _swing_source_inventory_record(
    swing_rows: pd.DataFrame,
    swing_proxy: pd.DataFrame,
    *,
    config: PredictionDataReadinessConfig,
) -> dict[str, object]:
    swing_usable = swing_rows.loc[swing_rows["source_usable"]]
    swing_sessions = int(swing_usable["session_date_et"].nunique())
    return {
        "strategy_id": config.swing.strategy_id,
        "proxy_strategy_id": config.swing.strategy_id,
        "source_role": "edge_rebuild_ten_session_swing_panel",
        "raw_rows": len(swing_rows),
        "source_usable_rows": len(swing_usable),
        "proxy_setup_rows": int(_bool_series(swing_proxy["setup_eligible"]).sum()),
        "proxy_label_eligible_rows": int(_bool_series(swing_proxy["strategy_label_eligible"]).sum()),
        "er_setup_opportunities": pd.NA,
        "unique_decision_groups": int(swing_usable["decision_group_id"].nunique()),
        "unique_tickers": int(swing_usable["ticker"].nunique()),
        "valid_sessions": swing_sessions,
        "effective_session_blocks": swing_sessions // config.swing.proposed_horizon_sessions,
        "first_usable_decision_time_utc": _timestamp_min(swing_usable["decision_time_utc"]),
        "last_usable_decision_time_utc": _timestamp_max(swing_usable["decision_time_utc"]),
        "price_feed": _single_identity(swing_usable["price_feed"]),
        "adjustment": _single_identity(swing_usable["adjustment"]),
        "session_gate": "pass" if swing_sessions >= config.swing.minimum_valid_sessions else "blocked",
        "exact_new_horizon_labels": True,
        "source_authority": "hash_bound_panel_and_candidate",
        "coverage_exact_rate": 1.0,
        "collection_model_data_ready": True,
        "point_in_time_membership_status": (
            "verified_row_intervals" if swing_usable["universe_snapshot_id"].notna().all() else "blocked"
        ),
        "benchmark_interval_status": _swing_benchmark_status(swing_proxy),
    }


def _intraday_source_inventory_record(
    intraday_rows: pd.DataFrame,
    intraday: VerifiedIntradaySources,
    *,
    config: PredictionDataReadinessConfig,
) -> dict[str, object]:
    usable = intraday_rows.loc[intraday_rows["source_usable"]]
    sessions = int(usable["session_date_et"].nunique())
    summary = _mapping(intraday.coverage_manifest.get("summary"))
    return {
        "strategy_id": config.intraday.strategy_id,
        "proxy_strategy_id": config.intraday.proxy_strategy_id,
        "source_role": "exact_one_minute_proxy_training",
        "raw_rows": len(intraday_rows),
        "source_usable_rows": len(usable),
        "proxy_setup_rows": len(intraday_rows),
        "proxy_label_eligible_rows": int(_bool_series(intraday_rows["label_eligible"]).sum()),
        "er_setup_opportunities": pd.NA,
        "unique_decision_groups": int(usable["setup_id"].nunique()),
        "unique_tickers": int(usable["ticker"].nunique()),
        "valid_sessions": sessions,
        "effective_session_blocks": sessions,
        "first_usable_decision_time_utc": _timestamp_min(usable["decision_time_utc"]),
        "last_usable_decision_time_utc": _timestamp_max(usable["decision_time_utc"]),
        "price_feed": str(intraday.collection_request.get("price_feed", "unknown")),
        "adjustment": str(intraday.collection_request.get("adjustment", "unknown")),
        "session_gate": "pass" if sessions >= config.intraday.minimum_causal_sessions else "blocked",
        "exact_new_horizon_labels": False,
        "source_authority": "training_complete_coverage_audited",
        "coverage_exact_rate": float(summary.get("requirement_exact_rate", 0.0)),
        "collection_model_data_ready": bool(summary.get("model_data_ready")),
        "point_in_time_membership_status": (
            "verified_snapshot_identity" if usable["universe_snapshot_id"].notna().all() else "blocked"
        ),
        "benchmark_interval_status": _intraday_benchmark_status(intraday_rows),
    }


def _catalyst_source_inventory_record(
    catalyst: VerifiedCatalystSources,
) -> dict[str, object]:
    return {
        "strategy_id": "CATALYST.OVERLAY",
        "proxy_strategy_id": "",
        "source_role": "alpaca_news_catalyst_lineage",
        "raw_rows": int(catalyst.lineage_manifest.get("source_event_rows", 0)),
        "source_usable_rows": int(catalyst.lineage_manifest.get("training_eligible_rows", 0)),
        "proxy_setup_rows": 0,
        "proxy_label_eligible_rows": 0,
        "er_setup_opportunities": pd.NA,
        "unique_decision_groups": int(catalyst.lineage_manifest.get("assignment_rows", 0)),
        "unique_tickers": int(catalyst.coverage["ticker"].nunique()),
        "valid_sessions": 0,
        "effective_session_blocks": 0,
        "first_usable_decision_time_utc": _timestamp_min(catalyst.coverage["requested_start_utc"]),
        "last_usable_decision_time_utc": _timestamp_max(catalyst.coverage["requested_end_utc"]),
        "price_feed": "not_applicable",
        "adjustment": "not_applicable",
        "session_gate": "not_applicable",
        "exact_new_horizon_labels": False,
        "source_authority": "complete_research_not_promotion_ready",
        "coverage_exact_rate": float(_bool_series(catalyst.coverage["training_eligible"]).mean()),
        "collection_model_data_ready": False,
        "point_in_time_membership_status": "not_applicable",
        "benchmark_interval_status": "not_applicable",
    }


def _blockers(
    *,
    source_inventory: pd.DataFrame,
    phase_capacity: pd.DataFrame,
    costs: pd.DataFrame,
    catalyst_readiness: pd.DataFrame,
    swing_candidate_status: str,
    promoted_bundle_available: bool,
    intraday_coverage_summary: Mapping[str, object],
    config: PredictionDataReadinessConfig,
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

    swing_inventory = source_inventory.loc[source_inventory["strategy_id"].eq(config.swing.strategy_id)].iloc[0]
    intra = source_inventory.loc[source_inventory["strategy_id"].eq(config.intraday.strategy_id)].iloc[0]
    if swing_inventory["session_gate"] != "pass":
        add(
            "swing_session_history_below_gate",
            config.swing.strategy_id,
            True,
            "acquire or recover verified daily history",
            f"valid_sessions={swing_inventory['valid_sessions']}; required={config.swing.minimum_valid_sessions}",
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
        missing = config.intraday.minimum_causal_sessions - int(intra["valid_sessions"])
        add(
            "intraday_session_history_below_gate",
            config.intraday.strategy_id,
            True,
            ("collect Alpaca SIP 1Min adjustment=all for the existing ticker/benchmark universe before the first usable session"),
            f"valid_sessions={intra['valid_sessions']}; required={config.intraday.minimum_causal_sessions}; missing_at_least={missing}",
        )
    for _, row in source_inventory.loc[
        source_inventory["strategy_id"].isin([config.swing.strategy_id, config.intraday.strategy_id])
    ].iterrows():
        if not str(row["point_in_time_membership_status"]).startswith("verified"):
            add(
                "point_in_time_membership_incomplete",
                str(row["strategy_id"]),
                True,
                "repair causal universe membership before ER2",
                str(row["point_in_time_membership_status"]),
            )
        if row["benchmark_interval_status"] != "verified_proxy_labels":
            add(
                "benchmark_interval_incomplete",
                str(row["strategy_id"]),
                True,
                "repair exact SPY/sector interval evidence before ER2",
                str(row["benchmark_interval_status"]),
            )
    if not bool(intraday_coverage_summary.get("model_data_ready")):
        add(
            "intraday_dense_clock_grid_incomplete",
            config.intraday.strategy_id,
            False,
            ("retain the verified causal sparse-clock policy and require observed trigger, entry, benchmark, and exit bars in ER3"),
            (
                "the older all-minutes-exact gate is false; "
                f"requirement_exact_rate={intraday_coverage_summary.get('requirement_exact_rate')}; "
                "new setup rows must not impute missing trades"
            ),
        )
    for _, row in costs.loc[~_bool_series(costs["adverse_fill_stress_available"])].iterrows():
        add(
            "adverse_fill_stress_missing",
            str(row["strategy_id"]),
            False,
            "freeze and build adverse-fill stress labels in ER2/ER3",
            "exact stamped base costs exist; adverse-fill stress does not",
        )
    for _, row in catalyst_readiness.loc[catalyst_readiness["research_status"].eq("blocked")].iterrows():
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
    if swing_candidate_status != "candidate":
        add(
            "swing_training_candidate_unavailable",
            config.swing.strategy_id,
            False,
            "train a ten-session edge-rebuild candidate that passes frozen validation gates",
            "the verified training run produced no eligible candidate",
        )
    if not promoted_bundle_available:
        add(
            "swing_promoted_bundle_unavailable",
            config.swing.strategy_id,
            False,
            "promote an accepted ten-session candidate and publish its hash-bound bundle",
            "public swing inference remains fail-closed until this artifact exists",
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
    config: PredictionDataReadinessConfig,
    request_sha256: str,
) -> dict[str, object]:
    blockers = evidence["blockers.csv"]
    blocking = blockers.loc[_bool_series(blockers["blocks_er2"])]
    inventory = evidence["source_inventory.csv"]
    intraday = inventory.loc[inventory["strategy_id"].eq(config.intraday.strategy_id)].iloc[0]
    intraday_history_blockers = blocking.loc[
        blocking["blocker_code"].eq("intraday_session_history_below_gate")
        & blocking["scope"].eq(config.intraday.strategy_id)
    ]
    acquisition_authorized = len(blocking) == 1 and len(intraday_history_blockers) == 1
    return {
        "schema": config.schema_version,
        "request_sha256": request_sha256,
        "status": ("blocked_pending_targeted_acquisition" if not blocking.empty else "ready_for_ER2"),
        "er2_authorized": blocking.empty,
        "training_performed": False,
        "download_performed": False,
        "models_created": 0,
        "blocking_findings": len(blocking),
        "nonblocking_required_work": len(blockers) - len(blocking),
        "acquisition_plan": {
            "authorized_by_audit": acquisition_authorized,
            "scope": "intraday_only",
            "provider": "alpaca",
            "feed": config.required_price_feed,
            "timeframe": config.intraday.required_timeframe,
            "adjustment": config.required_adjustment,
            "minimum_additional_sessions": max(
                0,
                config.intraday.minimum_causal_sessions - int(intraday["valid_sessions"]),
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


def _read_intraday_proxy(
    verified: VerifiedTrainingBundle,
    *,
    records: tuple[dict[str, object], ...],
    columns: set[str],
    config: PredictionDataReadinessConfig,
) -> pd.DataFrame:
    paths = [_resolve_inside(verified.directory, str(record["path"])) for record in records]
    _assert_projected_parquet_load(
        paths,
        config=config,
        stage="intraday readiness proxy",
    )
    frames: list[pd.DataFrame] = []
    for path in paths:
        _assert_memory(config, f"intraday readiness partition {path.name} before load")
        frames.append(_read_required_columns(path, columns))
        _assert_memory(config, f"intraday readiness partition {path.name} after load")
    _assert_projected_dataframe_concat(frames, config=config, stage="intraday readiness proxy concat")
    frame = pd.concat(frames, ignore_index=True)
    _assert_memory(config, "intraday readiness proxy concat")
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
        manifest.get("schema") != "intraday.specialist_coverage_audit.v1"
        or collection.get("manifest_sha256") != expected_collection_manifest_sha256
        or not manifest.get("coverage_fingerprint")
        or int(summary.get("requirements", 0)) <= 0
    ):
        raise DataReadinessError("ER1 intraday coverage audit identity differs")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise DataReadinessError("ER1 intraday coverage audit has no artifacts")
    observed: set[str] = set()
    for raw in raw_files:
        record = _mapping(raw)
        name = str(record.get("path", ""))
        path = _resolve_inside(root, name)
        observed.add(Path(name).as_posix())
        if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)) or file_sha256(path) != record.get("sha256"):
            raise DataReadinessError(f"ER1 intraday coverage artifact changed: {path}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*.parquet")}
    if actual != observed:
        raise DataReadinessError("ER1 intraday coverage parquet set differs from manifest")
    return manifest


def _read_required_columns(path: Path, columns: set[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=sorted(columns))
    except (OSError, KeyError, ValueError) as exc:
        raise DataReadinessError(f"ER1 required source columns are unavailable: {path}") from exc


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
        if str(key).replace("\\", "/").endswith(suffix) and (contains is None or contains in str(key).replace("\\", "/"))
    ]
    if len(matches) != 1:
        raise DataReadinessError(f"ER1 expected one source input ending {suffix}; found {len(matches)}")
    return matches[0]


def _resolve_inside(root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise DataReadinessError(f"unsafe ER1 artifact path: {relative}")
    return path_integrity.resolve_existing_file_inside(
        root,
        relative.as_posix(),
        label="ER1 source artifact",
    )


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
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        invalid = numeric.notna() & ~numeric.isin((0, 1))
        if invalid.any():
            raise DataReadinessError("readiness Boolean column contains values other than 0 or 1")
        return numeric.fillna(0).eq(1)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(("", "0", "1", "false", "true"))
    if invalid.any():
        raise DataReadinessError("readiness Boolean column contains a non-canonical value")
    return normalized.isin(("1", "true"))


def _assert_projected_parquet_load(
    paths: list[Path],
    *,
    config: PredictionDataReadinessConfig,
    stage: str,
) -> None:
    snapshot = process_memory_snapshot()
    if snapshot is None:
        raise DataReadinessError(f"memory guard stopped {stage}: process memory accounting is unavailable")
    uncompressed_bytes = 0
    for path in paths:
        parquet = pq.ParquetFile(path, memory_map=True)  # type: ignore[no-untyped-call]
        uncompressed_bytes += sum(parquet.metadata.row_group(index).total_byte_size for index in range(parquet.metadata.num_row_groups))
    projected_bytes = uncompressed_bytes * 8
    threshold = int((config.maximum_process_memory_gib - config.memory_guard_headroom_gib) * 1024**3)
    if snapshot[0] + projected_bytes > threshold:
        raise DataReadinessError(
            f"memory guard stopped {stage}: projected resident data exceeds the {threshold / 1024**3:.2f} GiB safety threshold"
        )


def _assert_projected_dataframe_concat(
    frames: list[pd.DataFrame],
    *,
    config: PredictionDataReadinessConfig,
    stage: str,
) -> None:
    snapshot = process_memory_snapshot()
    if snapshot is None:
        raise DataReadinessError(f"memory guard stopped {stage}: process memory accounting is unavailable")
    retained_bytes = sum(int(frame.memory_usage(index=True, deep=True).sum()) for frame in frames)
    projected_copy_bytes = int(retained_bytes * 1.5)
    threshold = int((config.maximum_process_memory_gib - config.memory_guard_headroom_gib) * 1024**3)
    if snapshot[0] + projected_copy_bytes > threshold:
        raise DataReadinessError(
            f"memory guard stopped {stage}: projected concat exceeds the {threshold / 1024**3:.2f} GiB safety threshold"
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
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_sha256_without_self_hash(value: dict[str, Any]) -> str:
    material = {key: item for key, item in value.items() if key != "request_sha256"}
    return _json_sha256(material)


def _assert_memory(
    config: PredictionDataReadinessConfig,
    stage: str,
) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
