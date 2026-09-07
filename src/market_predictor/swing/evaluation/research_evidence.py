"""Bounded metadata inspection, never source replay or an economic evaluation."""

from __future__ import annotations

import tomllib
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256, sequence_sha256
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts import SwingPromotionConfig
from market_predictor.swing.contracts.research import (
    EXPOSED_TEST_END,
    EXPOSED_TEST_EVIDENCE,
    EXPOSED_TEST_START,
    load_swing_research_contract,
)
from market_predictor.swing.features.panel import (
    MOMENTUM_FEATURES,
    PULLBACK_FEATURES,
    TREND_FEATURES,
    VOLUME_FEATURES,
    swing_model_feature_columns,
)

_MAX_FILE_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_BYTES = 32 * 1024 * 1024
_REQUIRED_IDS = {
    "strategy", "temporal", "legacy_training", "panel_request", "panel", "matched",
    "sec", "issuer_early", "issuer_late", "broker_v1", "broker_v2", "broker_v3",
    "broker_v4", "directional_v1", "exposed_evaluation",
}
_COUNT_FIELDS = {
    "rows", "securities", "modeled_security_count", "sessions", "rows_per_profile",
    "unique_latest_announcement_count", "event_rows", "decision_rows",
    "classified_event_rows", "assignment_rows", "coverage_rows",
    "production_eligible_event_rows", "recorded_trials",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _Artifact(_StrictModel):
    id: str
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: int = Field(gt=0, le=_MAX_FILE_BYTES)
    kind: Literal["metadata", "trials", "config", "access_hash_only"]
    counts: dict[str, int] = Field(default_factory=dict)


class _Inventory(_StrictModel):
    schema_version: Literal["market_predictor.swing_research_evidence.v1"]
    feature_count: Literal[120]
    feature_order_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    retained_trial_manifests: Literal[5]
    retained_recorded_trials: Literal[60]
    promotion_defaults_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifacts: list[_Artifact] = Field(min_length=15, max_length=15)


def _inside(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or "\\" in relative or ":" in relative or any(p in {".", ".."} for p in parts.parts):
        raise DataReadinessError("research evidence path must be canonical and relative")
    if parts.as_posix() != relative:
        raise DataReadinessError("research evidence path is not canonical")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise DataReadinessError("research evidence path escapes root")
    return path


def _trial_count(metadata: dict[str, object]) -> int:
    specialists = metadata.get("specialists")
    if not isinstance(specialists, list) or not specialists:
        raise DataReadinessError("trial manifest has no specialist metadata")
    count = 0
    for specialist in specialists:
        if not isinstance(specialist, dict) or not isinstance(specialist.get("experiments"), list):
            raise DataReadinessError("trial manifest has malformed experiment metadata")
        experiments = specialist["experiments"]
        if not all(isinstance(item, dict) for item in experiments):
            raise DataReadinessError("trial manifest contains malformed experiment entries")
        count += len(experiments)
    return count


def _inspect(root: Path, artifact: _Artifact) -> tuple[dict[str, object], dict[str, object]]:
    path = _inside(root, artifact.path)
    if artifact.kind == "access_hash_only":
        if artifact.id != "exposed_evaluation" or artifact.path != EXPOSED_TEST_EVIDENCE or artifact.counts:
            raise DataReadinessError("only the known exposed evaluation may be stream-hashed")
    elif artifact.kind == "config":
        if not artifact.path.startswith("configs/") or path.suffix != ".toml" or artifact.counts:
            raise DataReadinessError("invalid inventory config path")
    elif not artifact.path.startswith(("data/features/", "data/canonical/", "data/research/", "data/models/")) or path.name not in {
        "_manifest.json", "_request.json",
    }:
        raise DataReadinessError("inventory may parse only named metadata, never raw or outcome payloads")
    if not set(artifact.counts).issubset(_COUNT_FIELDS) or any(v < 0 for v in artifact.counts.values()):
        raise DataReadinessError("inventory contains unsupported metadata counts")
    if path.stat().st_size != artifact.bytes or file_sha256(path) != artifact.sha256:
        raise DataReadinessError(f"research evidence hash/size mismatch: {artifact.id}")
    metadata: dict[str, object] = {}
    counts: dict[str, int] = {}
    if artifact.kind in {"metadata", "trials"}:
        with path.open("rb") as handle:
            payload = handle.read(_MAX_FILE_BYTES + 1)
        if len(payload) != artifact.bytes:
            raise DataReadinessError("metadata changed during bounded read")
        metadata = parse_strict_json_object(payload, label=artifact.id)
        if file_sha256(path) != artifact.sha256:
            raise DataReadinessError("metadata changed during inspection")
        for field, expected in artifact.counts.items():
            actual = _trial_count(metadata) if field == "recorded_trials" else metadata.get(field)
            if type(actual) is not int or actual != expected:
                raise DataReadinessError(f"research metadata count mismatch: {artifact.id}.{field}")
            counts[field] = actual
    return {
        "id": artifact.id, "path": artifact.path, "sha256": artifact.sha256,
        "bytes": artifact.bytes, "inspection": artifact.kind, "counts": counts,
    }, metadata


def _feature_mapping(features: tuple[str, ...]) -> dict[str, object]:
    families = {
        "momentum_volatility": MOMENTUM_FEATURES, "trend_confirmation": TREND_FEATURES,
        "pullback_timing": PULLBACK_FEATURES, "volume_liquidity": VOLUME_FEATURES,
    }
    groups = {
        name: [column for column in features if any(column == base + suffix for base in bases for suffix in (
            "_xs_z", "_xs_rank", "_sector_z",
        ))]
        for name, bases in families.items()
    }
    if sum(map(len, groups.values())) != len(features) or set().union(*map(set, groups.values())) != set(features):
        raise DataReadinessError("feature inventory does not cover the canonical ordered inputs exactly once")
    proposals = (
        ("medium_term_relative_strength", ("return_20d", "return_60d", "rel_return_20d_vs_spy", "rel_return_20d_vs_sector"),
         "Six/twelve-month momentum excluding the latest month is absent; warm-up and backfill unverified."),
        ("short_term_reaction", ("return_1d", "return_5d", "gap_return", "intraday_return", "close_location"),
         "Price-window context is not verified post-release reaction; new residual windows need admission."),
        ("volume_liquidity", VOLUME_FEATURES,
         "Existing proxies are not dollar capacity. Lagged-baseline and new interaction acceptance require causal batch/live tests."),
        ("regime_interactions", ("realized_vol_20d", "residual_return_20d_vs_spy", "residual_return_60d_vs_spy"),
         "Direct SPY/QQQ regime levels, breadth, beta and new interactions are not distinct inputs in this order."),
        ("issuer_news", (),
         "No technical_market issuer-news input. Broker evidence does not admit earnings/guidance or prove novelty/recall."),
        ("event_response", (),
         "No verified post-release interval; both availability-anchored boundaries and source coverage need verification."),
        ("sec_issuer_content", (),
         "Form counts do not prove exhibits, guidance, surprise or first-disclosure timing. No such technical estimator input."),
    )
    return {
        "ordered_features": list(features), "existing_families": groups,
        "proposed_relationships": [
            {"name": name, "existing_columns": [f for f in features if any(f == base + suffix for base in bases for suffix in (
                "_xs_z", "_xs_rank", "_sector_z",
            ))], "status": "partial_existing_inputs" if bases else "gap", "remaining_evidence": gap}
            for name, bases, gap in proposals
        ],
        "mapping_basis": "canonical_estimator_names_only_not_transformation_or_source_replay",
    }


def audit_swing_research_evidence(
    root: Path, inventory_path: Path, research_contract_path: Path, strategy_contract_path: Path,
) -> dict[str, object]:
    """Verify pinned metadata only; do not follow any manifest payload references."""
    root = root.resolve()
    try:
        if inventory_path.stat().st_size > 64 * 1024:
            raise DataReadinessError("research evidence inventory exceeds size bound")
        inventory_sha256 = file_sha256(inventory_path)
        inventory = _Inventory.model_validate(tomllib.loads(inventory_path.read_text(encoding="utf-8")))
        artifacts = inventory.artifacts
        if {a.id for a in artifacts} != _REQUIRED_IDS or len({a.path for a in artifacts}) != len(artifacts):
            raise DataReadinessError("research evidence inventory must contain each known identity/path once")
        if sum(a.bytes for a in artifacts) > _MAX_TOTAL_BYTES:
            raise DataReadinessError("research evidence inventory exceeds total byte bound")
        by_id = {a.id: a for a in artifacts}
        if _inside(root, by_id["strategy"].path) != strategy_contract_path.resolve():
            raise DataReadinessError("strategy argument does not match pinned evidence")
        if by_id["exposed_evaluation"].kind != "access_hash_only":
            raise DataReadinessError("exposed evaluation must never be parsed")
        if research_contract_path.stat().st_size > 64 * 1024:
            raise DataReadinessError("research contract exceeds size bound")
        records: list[dict[str, object]] = []
        panel: dict[str, object] = {}
        trial_manifests = recorded_trials = 0
        for artifact in artifacts:
            record, metadata = _inspect(root, artifact)
            records.append(record)
            if artifact.id == "panel":
                panel = metadata
            if artifact.kind == "trials":
                trial_manifests += 1
                recorded_trials += _trial_count(metadata)
        if (trial_manifests, recorded_trials) != (inventory.retained_trial_manifests, inventory.retained_recorded_trials):
            raise DataReadinessError("retained trial inventory does not match the frozen counts")
        strategy = load_strategy_contract(strategy_contract_path)
        research = load_swing_research_contract(research_contract_path)
        research.assert_strategy_matches(strategy)
        promotion = SwingPromotionConfig()
        if json_sha256(promotion.model_dump(mode="json")) != inventory.promotion_defaults_sha256:
            raise DataReadinessError("inherited promotion defaults changed")
        if research.maximum_drawdown != promotion.max_drawdown:
            raise DataReadinessError("research drawdown differs from inherited promotion limit")
        features = swing_model_feature_columns(contract=strategy, catalyst=False)
        if len(features) != inventory.feature_count or sequence_sha256(features) != inventory.feature_order_sha256:
            raise DataReadinessError("canonical technical feature order differs from frozen research inventory")
        columns = panel.get("columns_by_profile")
        panel_columns = columns.get("technical_market") if isinstance(columns, dict) else None
        if not isinstance(panel_columns, list) or tuple(c for c in panel_columns if c in features) != features:
            raise DataReadinessError("retained panel metadata disagrees with canonical feature order")
        if file_sha256(inventory_path) != inventory_sha256 or file_sha256(strategy_contract_path) != by_id["strategy"].sha256:
            raise DataReadinessError("inventory or strategy changed during audit")
        report: dict[str, object] = {
            "schema_version": inventory.schema_version, "status": "metadata_verified_only",
            "inventory_sha256": inventory_sha256, "research_contract_sha256": research.sha256(),
            "strategy_contract_sha256": strategy.sha256(), "artifacts": records,
            "inherited_risk": {"maximum_drawdown": promotion.max_drawdown,
                               "drawdown_source": "SwingPromotionConfig.max_drawdown",
                               "promotion_defaults_sha256": inventory.promotion_defaults_sha256,
                               "hard_maximum_sector_weight": strategy.swing.hard_maximum_sector_weight},
            "feature_count": len(features), "feature_order_sha256": sequence_sha256(features),
            "features": _feature_mapping(features),
            "historical_trials": {"retained_manifest_count": trial_manifests, "recorded_trials": recorded_trials,
                                  "duplicates_possible": True, "complete_lifetime_trial_history": False},
            "holdout_access": {"status": "already_exposed_research_only", "start": EXPOSED_TEST_START.isoformat(),
                               "end": EXPOSED_TEST_END.isoformat(), "evidence_path": EXPOSED_TEST_EVIDENCE,
                               "evidence_sha256": by_id["exposed_evaluation"].sha256,
                               "basis": "previously_reviewed_access_disclosure_not_reparsed_outcomes"},
            "source_payloads_read": False, "outcome_payloads_parsed": False, "full_replay_performed": False,
            "counts_basis": "retained_manifest_metadata_only",
            "training_authorized": False, "promotion_authorized": False,
            "total_return_accounting_verified": False,
        }
        report["audit_sha256"] = json_sha256(report)
        return report
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise DataReadinessError(f"swing research metadata audit failed: {exc}") from exc
