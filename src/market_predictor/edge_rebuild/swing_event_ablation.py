"""Immutable analyst-revision swing ablations on one matched event cohort."""
from __future__ import annotations



import hashlib
import json
import os
import shutil
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.issuer_event_family_authority import (
    IssuerEventFamilyAuthority,
)
from market_predictor.edge_rebuild.issuer_event_precision_audit import (
    IssuerEventPrecisionAudit,
    issuer_event_rule_variant,
    load_issuer_event_precision_audit,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.swing_features import (
    SWING_FEATURE_PROFILE,
    swing_model_feature_columns,
)
from market_predictor.edge_rebuild.swing_materialization import (
    load_complete_swing_feature_panel,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.swing.event_families import EVENT_FAMILIES
from market_predictor.core.errors import DataReadinessError

POLICY_SCHEMA: Final = "market_predictor.swing_analyst_revision_ablation.v1"
REQUEST_SCHEMA: Final = "edge_rebuild.swing_analyst_revision_ablation_request.v2"
MANIFEST_SCHEMA: Final = "edge_rebuild.swing_analyst_revision_ablation_manifest.v2"
AUTHORITY_SCHEMA: Final = "edge_rebuild.swing_analyst_revision_ablation_authority.v2"
TECHNICAL_PROFILE: Final = "analyst_revision_technical_only"
EVENT_PROFILE: Final = "analyst_revision_event_only"
COMBINED_PROFILE: Final = "analyst_revision_technical_plus_event"
PROFILES: Final = (TECHNICAL_PROFILE, EVENT_PROFILE, COMBINED_PROFILE)
SUBTYPE_POLICY: Final = {
    "implementation": "issuer_event_precision_audit.issuer_event_rule_variant.v1",
    "admitted": ("bare_upgrade", "bare_downgrade", "coverage"),
    "diagnostic_only": (
        "price_target_up",
        "price_target_down",
        "analyst_rating_or_target_revision",
    ),
}
SUBTYPE_POLICY_SHA256: Final = hashlib.sha256(
    json.dumps(SUBTYPE_POLICY, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
EVENT_FEATURE_COLUMNS: Final = (
    "analyst_revision_present_1d",
    "analyst_revision_latest_age_fraction_3d",
    "analyst_revision_latest_is_upgrade",
    "analyst_revision_latest_is_downgrade",
    "analyst_revision_latest_is_coverage",
    "analyst_revision_latest_direction_unverified",
    "analyst_revision_any_upgrade_3d",
    "analyst_revision_any_downgrade_3d",
    "analyst_revision_any_coverage_3d",
    "analyst_revision_conflicting_direction_3d",
    "analyst_revision_direction_available",
    "analyst_revision_latest_premarket",
    "analyst_revision_latest_regular_session",
    "analyst_revision_latest_after_close",
)
_IDENTITY_COLUMNS: Final = (
    "decision_id",
    "decision_group_id",
    "ticker",
    "security_id",
    "sector",
    "primary_benchmark",
    "market_regime",
    "session_date_et",
    "decision_time_utc",
    "feature_available_at_utc",
    "label_available_at_utc",
    "membership_effective_from_utc",
    "membership_effective_to_utc",
    "membership_available_at_utc",
    "entry_time_utc",
    "exit_time_utc",
    "entry_session_date_et",
    "exit_session_date_et",
    "entry_price",
    "exit_price",
    "label_window_expected",
    "label_path_exact",
    "decision_start_date",
    "decision_end_date",
    "horizon_sessions",
    "round_trip_cost_bps",
    "minimum_daily_bars",
    "swing_feature_schema_version",
    "dataset_label_config_sha256",
    "dataset_label_policy_json",
    "execution_policy_sha256",
    "feature_eligible",
    "label_eligible",
    "cross_section_eligible",
    "managed_path_eligible",
    "sector_benchmark_feature_eligible",
    "sector_benchmark_label_eligible",
    "sector_benchmark_abstention_reason",
    "sparse_gap_feature_eligible",
    "sparse_gap_label_eligible",
    "sparse_gap_abstention_reason",
    "swing_feature_panel_schema",
    "strategy_contract_sha256",
)
_OUTCOME_EXACT_COLUMNS: Final = frozenset(
    {
        "forward_return",
        "target_excess_rank",
        "rank_label",
        "rank_percentile",
        "ranking_group_size",
        "ranking_reliability_weight",
        "sector_peer_count",
        "sector_rank_eligible",
        "sector_rank_target_met",
    }
)
_OUTCOME_PREFIXES: Final = (
    "future_",
    "target_net_positive_",
    "barrier_",
    "managed_",
    "approx_managed_",
)
_EVENT_AUDIT_COLUMNS: Final = (
    "analyst_revision_episode_id",
    "analyst_revision_episode_sample_weight",
    "analyst_revision_source_decision_id",
    "analyst_revision_source_security_id",
    "analyst_revision_identity_alignment",
    "analyst_revision_source_coverage_known_3d",
    "analyst_revision_latest_feature_available_at_utc",
)
_ALIGNMENT_COLUMNS: Final = (
    "analyst_revision_source_decision_id",
    "analyst_revision_source_security_id",
    "ticker",
    "decision_time_utc",
    "target_decision_id",
    "target_security_id",
    "direct_decision_id_match",
    "identity_alignment",
    "inclusion_status",
    "exclusion_reason",
    "feature_eligible",
    "label_eligible",
    "cross_section_eligible",
    "managed_path_eligible",
    "rank_label_available",
)
_XNYS: Final = xcals.get_calendar("XNYS")


@dataclass(frozen=True, slots=True)
class AnalystRevisionAblationPolicy:
    event_family: str
    source_family: str
    cohort_window: str
    near_window: str
    profiles: tuple[str, ...]
    admitted_subtypes: tuple[str, ...]
    directional_subtypes: tuple[str, ...]
    diagnostic_only_subtypes: tuple[str, ...]
    maximum_process_memory_gib: float
    memory_guard_headroom_gib: float


def load_analyst_revision_ablation_policy(
    path: Path,
) -> AnalystRevisionAblationPolicy:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    expected = {
        "schema_version",
        "event_family",
        "source_family",
        "cohort_window",
        "near_window",
        "unknown_coverage_policy",
        "no_event_policy",
        "maximum_process_memory_gib",
        "memory_guard_headroom_gib",
        "profiles",
        "admitted_subtypes",
        "directional_subtypes",
        "diagnostic_only_subtypes",
    }
    if set(raw) != expected or raw.get("schema_version") != POLICY_SCHEMA:
        raise DataReadinessError("analyst-revision ablation policy fields differ")
    if (
        raw.get("event_family") != "analyst_revision"
        or raw.get("source_family") != "alpaca"
        or raw.get("cohort_window") != "3d"
        or raw.get("near_window") != "1d"
        or raw.get("unknown_coverage_policy") != "abstain"
        or raw.get("no_event_policy") != "abstain"
        or tuple(raw.get("profiles", ())) != PROFILES
        or tuple(raw.get("admitted_subtypes", ()))
        != ("bare_upgrade", "bare_downgrade", "coverage")
        or tuple(raw.get("directional_subtypes", ()))
        != ("bare_upgrade", "bare_downgrade")
    ):
        raise DataReadinessError("analyst-revision ablation policy is not frozen")
    diagnostic = tuple(raw.get("diagnostic_only_subtypes", ()))
    if diagnostic != SUBTYPE_POLICY["diagnostic_only"]:
        raise DataReadinessError("diagnostic analyst subtypes are not frozen")
    maximum = float(raw.get("maximum_process_memory_gib", 0))
    headroom = float(raw.get("memory_guard_headroom_gib", 0))
    if not 0 < maximum <= 5 or not 0 < headroom < maximum:
        raise DataReadinessError("analyst-revision memory policy is invalid")
    return AnalystRevisionAblationPolicy(
        event_family="analyst_revision",
        source_family="alpaca",
        cohort_window="3d",
        near_window="1d",
        profiles=PROFILES,
        admitted_subtypes=("bare_upgrade", "bare_downgrade", "coverage"),
        directional_subtypes=("bare_upgrade", "bare_downgrade"),
        diagnostic_only_subtypes=diagnostic,
        maximum_process_memory_gib=maximum,
        memory_guard_headroom_gib=headroom,
    )


def publish_swing_analyst_revision_ablation(
    *,
    technical_panel_directory: Path,
    event_authority_directories: Sequence[Path],
    precision_audit_directories: Sequence[Path],
    policy_path: Path,
    strategy_contract: StrategyContract,
    output_directory: Path,
) -> Mapping[str, Any]:
    """Publish three column-disjoint profiles from one verified event cohort."""

    policy = load_analyst_revision_ablation_policy(policy_path)
    if output_directory.exists():
        raise DataReadinessError(
            f"analyst-revision ablation is immutable: {output_directory}"
        )
    panel = load_complete_swing_feature_panel(technical_panel_directory)
    if (
        panel.get("feature_profiles") != [SWING_FEATURE_PROFILE]
        or panel.get("strategy_contract_sha256") != strategy_contract.sha256()
    ):
        raise DataReadinessError("analyst ablation requires the current technical panel")
    sources = _load_event_sources(
        event_authority_directories,
        precision_audit_directories,
        policy=policy,
    )
    event_features = _build_event_features(sources, policy=policy)
    if event_features.empty:
        raise DataReadinessError("verified analyst-revision cohort is empty")
    base_records = _base_records(panel)
    selected_features, alignment_audit, alignment_metrics = (
        _align_event_features_to_panel(
            technical_panel_directory,
            base_records,
            event_features,
            policy=policy,
        )
    )
    if selected_features.empty:
        raise DataReadinessError("analyst-revision cohort has no eligible panel rows")
    technical_features = swing_model_feature_columns(
        contract=strategy_contract,
        catalyst=False,
    )
    request = _build_request(
        technical_panel_directory=technical_panel_directory,
        panel=panel,
        sources=sources,
        policy_path=policy_path,
        policy=policy,
        strategy_contract=strategy_contract,
        technical_features=technical_features,
    )
    request_sha256 = _json_sha256(request)
    staging = output_directory.with_name(
        f".{output_directory.name}.{uuid4().hex}.tmp"
    )
    try:
        staging.mkdir(parents=True, exist_ok=False)
        _atomic_json(staging / "_request.json", {**request, "request_sha256": request_sha256})
        alignment_path = staging / "identity_alignment_audit.parquet"
        alignment_audit.to_parquet(alignment_path, index=False)
        alignment_record = {
            "path": alignment_path.name,
            "sha256": file_sha256(alignment_path),
            "rows": len(alignment_audit),
            "columns": list(_ALIGNMENT_COLUMNS),
        }
        eligible_ids = set(selected_features["decision_id"].astype(str))
        files: list[dict[str, Any]] = []
        shared_columns: tuple[str, ...] | None = None
        for record in base_records:
            base_path = technical_panel_directory / "final" / str(record["path"])
            base = pd.read_parquet(base_path)
            base = base.loc[base["decision_id"].astype(str).isin(eligible_ids)].copy()
            if base.empty:
                continue
            base["decision_id"] = base["decision_id"].astype(str)
            attached = _attach_event_features(base, selected_features)
            current_shared = _shared_columns(attached)
            if shared_columns is None:
                shared_columns = current_shared
            elif shared_columns != current_shared:
                raise DataReadinessError("analyst ablation shared schema changed by month")
            month = str(record["partition_month"])
            reference_decisions: str | None = None
            reference_shared: str | None = None
            for profile in PROFILES:
                model_features = _profile_features(
                    profile,
                    technical_features=technical_features,
                )
                columns = [*current_shared, *model_features]
                projected = attached.loc[:, columns].copy()
                projected["feature_profile"] = profile
                projected = projected.sort_values("decision_id", kind="stable").reset_index(drop=True)
                decision_hash = _sequence_sha256(projected["decision_id"].astype(str))
                shared_hash = _frame_sha256(projected, current_shared)
                if reference_decisions is None:
                    reference_decisions = decision_hash
                    reference_shared = shared_hash
                elif decision_hash != reference_decisions or shared_hash != reference_shared:
                    raise DataReadinessError("analyst ablation profiles are not identical")
                path = (
                    staging
                    / "panel"
                    / f"feature_profile={profile}"
                    / f"month={month}"
                    / "part.parquet"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                projected.to_parquet(path, index=False)
                files.append(
                    {
                        "path": str(path.relative_to(staging)).replace("\\", "/"),
                        "sha256": file_sha256(path),
                        "rows": len(projected),
                        "feature_profile": profile,
                        "partition_month": month,
                        "decision_ids_sha256": decision_hash,
                        "shared_content_sha256": shared_hash,
                        "model_feature_columns": list(model_features),
                    }
                )
            del base, attached
            release_process_memory()
            _guard(policy, f"analyst ablation publish {month}")
        if shared_columns is None or not files:
            raise DataReadinessError("analyst ablation produced no partitions")
        rows_per_profile = sum(
            int(item["rows"])
            for item in files
            if item["feature_profile"] == TECHNICAL_PROFILE
        )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "state": "complete",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "request_sha256": request_sha256,
            "profiles": list(PROFILES),
            "shared_columns": list(shared_columns),
            "technical_feature_columns": list(technical_features),
            "event_feature_columns": list(EVENT_FEATURE_COLUMNS),
            "rows_per_profile": rows_per_profile,
            "total_rows": rows_per_profile * len(PROFILES),
            "episode_count": int(selected_features["analyst_revision_episode_id"].nunique()),
            "unique_latest_announcement_count": int(
                selected_features["analyst_revision_episode_id"].nunique()
            ),
            "alignment_metrics": alignment_metrics,
            "alignment_audit": alignment_record,
            "files": files,
            "memory": memory_audit(
                hard_budget_gib=policy.maximum_process_memory_gib,
                headroom_gib=policy.memory_guard_headroom_gib,
            ).to_record(),
            "production_ready": False,
            "training_eligible": False,
            "research_training_eligible": True,
            "serving_eligible": False,
            "research_only_reason": "historical event authorities are research-only",
        }
        _atomic_json(staging / "_manifest.json", manifest)
        _atomic_json(
            staging / "_authority.json",
            {
                "schema": AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(staging / "_manifest.json"),
                "request_sha256": request_sha256,
                "production_ready": False,
                "training_eligible": False,
                "research_training_eligible": True,
                "serving_eligible": False,
            },
        )
        verified = _verify_published_ablation(
            staging,
            strategy_contract=strategy_contract,
            preverified_panel=panel,
            preverified_sources=sources,
            precomputed_event_features=event_features,
        )
        del sources, event_features, selected_features
        release_process_memory()
        os.replace(staging, output_directory)
        return verified
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_swing_analyst_revision_ablation(
    directory: Path,
    *,
    strategy_contract: StrategyContract,
) -> Mapping[str, Any]:
    """Strictly replay an A3.4 artifact and every upstream authority."""

    return _verify_published_ablation(
        directory,
        strategy_contract=strategy_contract,
    )


@dataclass(frozen=True, slots=True)
class _EventSources:
    authorities: tuple[IssuerEventFamilyAuthority, ...]
    audits: tuple[IssuerEventPrecisionAudit, ...]
    events: pd.DataFrame
    assignments: pd.DataFrame
    coverage: pd.DataFrame


def _load_event_sources(
    authority_directories: Sequence[Path],
    audit_directories: Sequence[Path],
    *,
    policy: AnalystRevisionAblationPolicy,
) -> _EventSources:
    if len(authority_directories) != 2 or len(audit_directories) != 2:
        raise DataReadinessError("analyst ablation requires exactly two historical eras")
    loaded_audits: list[IssuerEventPrecisionAudit] = []
    for path in audit_directories:
        loaded_audits.append(
            load_issuer_event_precision_audit(
                path,
                retain_source_authority=True,
            )
        )
        release_process_memory()
        _guard(policy, f"precision audit replay {path.name}")
    audits = tuple(loaded_audits)
    bound_authority_hashes: set[str] = set()
    for audit in audits:
        raw_admitted = audit.manifest.get("admitted_families")
        raw_blocked = audit.manifest.get("blocked_families")
        raw_request = audit.manifest.get("request")
        if (
            not isinstance(raw_admitted, list)
            or not isinstance(raw_blocked, list)
            or not isinstance(raw_request, Mapping)
        ):
            raise DataReadinessError("precision audit family binding is malformed")
        admitted = [str(value) for value in raw_admitted]
        blocked = {str(value) for value in raw_blocked}
        if admitted != [policy.event_family] or blocked != set(EVENT_FAMILIES) - {policy.event_family}:
            raise DataReadinessError("precision audit does not admit only analyst revisions")
        sample_dir = Path(str(raw_request["sample_directory"]))
        sample_manifest = _json_object(sample_dir / "_manifest.json")
        sample_request = sample_manifest.get("request")
        if not isinstance(sample_request, Mapping):
            raise DataReadinessError("precision sample authority binding is malformed")
        bound_authority_hashes.add(
            str(sample_request["issuer_event_authority_sha256"])
        )
        variants = audit.rule_variant_metrics.loc[
            audit.rule_variant_metrics["event_family"].eq(policy.event_family)
        ]
        admitted_variants = set(
            variants.loc[variants["status"].eq("admitted"), "rule_variant"].astype(str)
        )
        if admitted_variants != set(policy.admitted_subtypes):
            raise DataReadinessError("analyst subtype precision gates differ")
    if any(audit.source_authority is None for audit in audits):
        raise DataReadinessError("precision audit did not retain its verified event authority")
    authorities = tuple(
        audit.source_authority
        for audit in audits
        if audit.source_authority is not None
    )
    expected_directories = tuple(path.resolve() for path in authority_directories)
    observed_directories = tuple(authority.directory for authority in authorities)
    if observed_directories != expected_directories:
        raise DataReadinessError(
            "precision audits do not bind the supplied event authority directories"
        )
    observed_hashes = {
        file_sha256(authority.directory / "_authority.json")
        for authority in authorities
    }
    if bound_authority_hashes != observed_hashes:
        raise DataReadinessError("precision audits do not bind the supplied event authorities")
    events = pd.concat([authority.events for authority in authorities], ignore_index=True)
    assignments = pd.concat([authority.assignments for authority in authorities], ignore_index=True)
    coverage = pd.concat([authority.coverage for authority in authorities], ignore_index=True)
    eligible_events = events.loc[
        events["event_family"].eq(policy.event_family)
        & events["source_family"].eq(policy.source_family)
        & events["relation_channel"].eq("direct_issuer")
        & events["research_eligible"].astype(bool)
        & ~events["production_eligible"].astype(bool)
    ].copy()
    duplicate_source = eligible_events.duplicated(
        ["source_event_id", "security_id"], keep=False
    )
    if bool(duplicate_source.any()) or bool(eligible_events["family_event_id"].duplicated().any()):
        raise DataReadinessError("analyst events overlap or duplicate across historical eras")
    assignments = assignments.loc[
        assignments["event_family"].eq(policy.event_family)
        & assignments["original_source_family"].eq(policy.source_family)
    ].copy()
    eligible_ids = set(eligible_events["family_event_id"].astype(str))
    assigned_ids = set(assignments.loc[assignments["status"].eq("assigned"), "event_id"].astype(str))
    if not assigned_ids.issubset(eligible_ids):
        raise DataReadinessError("analyst assignments reference ineligible events")
    coverage = coverage.loc[
        coverage["event_family"].eq(policy.event_family)
        & coverage["source_family"].eq(policy.source_family)
    ].copy()
    return _EventSources(authorities, audits, eligible_events, assignments, coverage)


def _build_event_features(
    sources: _EventSources,
    *,
    policy: AnalystRevisionAblationPolicy,
) -> pd.DataFrame:
    event_frame = sources.events.copy()
    subtype_by_event_id = {
        str(row.family_event_id): _analyst_subtype(
            str(row.matched_text),
            classification_rule_id=str(row.classification_rule_id),
        )
        for row in event_frame[
            [
                "family_event_id",
                "classification_rule_id",
                "matched_text",
            ]
        ].itertuples(index=False)
    }
    events = event_frame.set_index("family_event_id")
    if not events.index.is_unique:
        raise DataReadinessError("analyst family event identity is duplicated")
    assigned = sources.assignments.loc[
        sources.assignments["status"].eq("assigned")
        & sources.assignments["window_name"].isin([policy.near_window, policy.cohort_window])
    ].copy()
    assigned["decision_time_utc"] = pd.to_datetime(assigned["decision_time_utc"], utc=True, errors="raise")
    assigned["feature_available_at_utc"] = pd.to_datetime(assigned["feature_available_at_utc"], utc=True, errors="raise")
    assigned = assigned.join(
        events[
            [
                "security_id",
                "classification_rule_id",
                "matched_text",
                "feature_available_at_utc",
            ]
        ].rename(
            columns={
                "security_id": "event_security_id",
                "feature_available_at_utc": "event_feature_available_at_utc",
            }
        ),
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    assigned["analyst_subtype"] = assigned["event_id"].astype(str).map(
        subtype_by_event_id
    )
    event_available = pd.to_datetime(assigned["event_feature_available_at_utc"], utc=True, errors="coerce")
    assignment_available = assigned["feature_available_at_utc"]
    lower = assigned["decision_time_utc"] - pd.to_timedelta(
        assigned["window_name"].map({"1d": "1D", "3d": "3D"})
    )
    if (
        assigned["event_security_id"].isna().any()
        or assigned["analyst_subtype"].isna().any()
        or bool(assigned["security_id"].astype(str).ne(assigned["event_security_id"].astype(str)).any())
        or bool(event_available.ne(assignment_available).any())
        or bool((assignment_available < lower).any())
        or bool((assignment_available > assigned["decision_time_utc"]).any())
    ):
        raise DataReadinessError("analyst assignment issuer or causal window fails")
    intervals = _merged_coverage_intervals(sources.coverage)
    cohort = assigned.loc[assigned["window_name"].eq(policy.cohort_window)].copy()
    cohort = cohort.sort_values(
        ["decision_id", "feature_available_at_utc", "event_id"],
        kind="stable",
    )
    near_ids = set(
        assigned.loc[assigned["window_name"].eq(policy.near_window), "decision_id"].astype(str)
    )
    rows: list[dict[str, Any]] = []
    for decision_id, part in cohort.groupby("decision_id", sort=False):
        security_ids = set(part["security_id"].astype(str))
        tickers = set(part["ticker"].astype(str).str.upper().str.strip())
        decision_times = set(part["decision_time_utc"])
        if len(security_ids) != 1 or len(tickers) != 1 or len(decision_times) != 1:
            raise DataReadinessError("analyst event decision identity is ambiguous")
        security_id = next(iter(security_ids))
        ticker = next(iter(tickers))
        decision_time = pd.Timestamp(next(iter(decision_times)))
        if not _window_is_covered(
            intervals.get(security_id, ()),
            start=decision_time - pd.Timedelta("3D"),
            end=decision_time,
        ):
            continue
        ordered = part
        subtypes = ordered["analyst_subtype"].astype(str)
        latest = ordered.iloc[-1]
        latest_subtype = str(subtypes.iloc[-1])
        latest_time = pd.Timestamp(latest["feature_available_at_utc"])
        age_fraction = (decision_time - latest_time) / pd.Timedelta("3D")
        if not 0 <= float(age_fraction) <= 1:
            raise DataReadinessError("analyst latest-event age is outside three days")
        event_id = str(latest["event_id"])
        publication_regime = _publication_regime(latest_time)
        premarket = publication_regime == "premarket"
        regular = publication_regime == "regular_session"
        upgrade = subtypes.eq("bare_upgrade")
        downgrade = subtypes.eq("bare_downgrade")
        coverage = subtypes.eq("coverage")
        rows.append(
            {
                "decision_id": str(decision_id),
                "security_id": security_id,
                "ticker": ticker,
                "decision_time_utc": decision_time,
                "analyst_revision_episode_id": _json_sha256([security_id, event_id]),
                "analyst_revision_source_coverage_known_3d": True,
                "analyst_revision_latest_feature_available_at_utc": latest_time,
                "analyst_revision_present_1d": int(str(decision_id) in near_ids),
                "analyst_revision_latest_age_fraction_3d": float(age_fraction),
                "analyst_revision_latest_is_upgrade": int(latest_subtype == "bare_upgrade"),
                "analyst_revision_latest_is_downgrade": int(latest_subtype == "bare_downgrade"),
                "analyst_revision_latest_is_coverage": int(latest_subtype == "coverage"),
                "analyst_revision_latest_direction_unverified": int(
                    latest_subtype not in policy.directional_subtypes
                ),
                "analyst_revision_any_upgrade_3d": int(upgrade.any()),
                "analyst_revision_any_downgrade_3d": int(downgrade.any()),
                "analyst_revision_any_coverage_3d": int(coverage.any()),
                "analyst_revision_conflicting_direction_3d": int(upgrade.any() and downgrade.any()),
                "analyst_revision_direction_available": int(
                    latest_subtype in policy.directional_subtypes
                ),
                "analyst_revision_latest_premarket": int(premarket),
                "analyst_revision_latest_regular_session": int(regular),
                "analyst_revision_latest_after_close": int(not premarket and not regular),
            }
        )
    output = pd.DataFrame.from_records(rows)
    if output.empty or output["decision_id"].duplicated().any():
        raise DataReadinessError("analyst event feature identity is empty or duplicated")
    return output


def _merged_coverage_intervals(
    coverage: pd.DataFrame,
) -> Mapping[str, tuple[tuple[pd.Timestamp, pd.Timestamp], ...]]:
    known = coverage.loc[
        coverage["research_eligible"].astype(bool)
        & coverage["missingness_known"].astype(bool)
        & coverage["coverage_state"].isin(["observed_complete", "observed_empty"])
    ].copy()
    known["start"] = pd.to_datetime(known["requested_start_utc"], utc=True, errors="raise")
    known["end"] = pd.to_datetime(known["requested_end_utc"], utc=True, errors="raise")
    if bool((known["start"] >= known["end"]).any()):
        raise DataReadinessError("analyst coverage interval is invalid")
    result: dict[str, tuple[tuple[pd.Timestamp, pd.Timestamp], ...]] = {}
    for security_id, part in known.groupby("security_id", sort=True):
        merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for row in part.sort_values(["start", "end"], kind="stable").itertuples(index=False):
            start = pd.Timestamp(row.start)
            end = pd.Timestamp(row.end)
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        result[str(security_id)] = tuple(merged)
    return result


def _window_is_covered(
    intervals: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    return any(left <= start and right >= end for left, right in intervals)


def _analyst_subtype(
    text: str,
    *,
    classification_rule_id: str = "analyst_rating_or_target_revision",
) -> str:
    variant = issuer_event_rule_variant(
        {
            "event_family": "analyst_revision",
            "classification_rule_id": classification_rule_id,
            "matched_text": text,
        }
    )
    return variant if variant in SUBTYPE_POLICY["admitted"] else "direction_unverified"


@cache
def _publication_regime(timestamp: pd.Timestamp) -> str:
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        raise DataReadinessError("analyst publication timestamp is timezone-naive")
    local_date = value.tz_convert("America/New_York").date()
    if not _XNYS.is_session(local_date):
        return "after_close"
    session = pd.Timestamp(local_date)
    market_open = _XNYS.session_open(session)
    market_close = _XNYS.session_close(session)
    if value < market_open:
        return "premarket"
    if value < market_close:
        return "regular_session"
    return "after_close"


def _build_request(
    *,
    technical_panel_directory: Path,
    panel: Mapping[str, Any],
    sources: _EventSources,
    policy_path: Path,
    policy: AnalystRevisionAblationPolicy,
    strategy_contract: StrategyContract,
    technical_features: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": REQUEST_SCHEMA,
        "technical_panel_directory": str(technical_panel_directory.resolve()),
        "technical_panel_manifest_sha256": file_sha256(
            technical_panel_directory / "final" / "_manifest.json"
        ),
        "technical_panel_authority_sha256": file_sha256(
            technical_panel_directory / "final" / "_authority.json"
        ),
        "strategy_contract_sha256": strategy_contract.sha256(),
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": file_sha256(policy_path),
        "subtype_policy_sha256": SUBTYPE_POLICY_SHA256,
        "profiles": list(policy.profiles),
        "technical_feature_columns": list(technical_features),
        "event_feature_columns": list(EVENT_FEATURE_COLUMNS),
        "event_authorities": [
            {
                "directory": str(authority.directory.resolve()),
                "authority_sha256": file_sha256(authority.directory / "_authority.json"),
            }
            for authority in sources.authorities
        ],
        "precision_audits": [
            {
                "directory": str(audit.directory.resolve()),
                "authority_sha256": file_sha256(audit.directory / "_authority.json"),
            }
            for audit in sources.audits
        ],
        "blocked_family_policy": "absent_not_zero",
        "unknown_coverage_policy": "abstain",
        "no_event_policy": "abstain",
        "identity_alignment_policy": (
            "exact_ticker_and_decision_time_with_no_conflicting_cik"
        ),
        "label_policy": "copy_full_population_labels_before_event_filtering",
        "production_ready": False,
    }


def _target_decision_spine(
    panel_directory: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    policy: AnalystRevisionAblationPolicy,
) -> pd.DataFrame:
    columns = [
        "decision_id",
        "security_id",
        "ticker",
        "decision_time_utc",
        "feature_eligible",
        "label_eligible",
        "cross_section_eligible",
        "managed_path_eligible",
        "rank_label",
    ]
    frames: list[pd.DataFrame] = []
    for record in records:
        path = panel_directory / "final" / str(record["path"])
        frames.append(pd.read_parquet(path, columns=columns))
        release_process_memory()
        _guard(policy, "analyst ablation cohort scan")
    spine = pd.concat(frames, ignore_index=True)
    spine["ticker"] = spine["ticker"].astype(str).str.upper().str.strip()
    spine["security_id"] = spine["security_id"].astype(str).str.strip()
    spine["decision_id"] = spine["decision_id"].astype(str)
    spine["decision_time_utc"] = pd.to_datetime(
        spine["decision_time_utc"], utc=True, errors="raise"
    )
    if bool(
        spine["decision_id"].duplicated().any()
        or spine.duplicated(["ticker", "decision_time_utc"]).any()
        or spine["ticker"].eq("").any()
        or spine["security_id"].eq("").any()
    ):
        raise DataReadinessError(
            "technical panel identity spine is ambiguous for event alignment"
        )
    return spine


def _align_event_features_to_panel(
    panel_directory: Path,
    records: Sequence[Mapping[str, Any]],
    event_features: pd.DataFrame,
    *,
    policy: AnalystRevisionAblationPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    source = event_features.copy()
    required = {"decision_id", "security_id", "ticker", "decision_time_utc"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise DataReadinessError(
            f"analyst event alignment fields are missing: {missing}"
        )
    source["ticker"] = source["ticker"].astype(str).str.upper().str.strip()
    source["security_id"] = source["security_id"].astype(str).str.strip()
    source["decision_id"] = source["decision_id"].astype(str)
    source["decision_time_utc"] = pd.to_datetime(
        source["decision_time_utc"], utc=True, errors="raise"
    )
    if bool(
        source["decision_id"].duplicated().any()
        or source.duplicated(["ticker", "decision_time_utc"]).any()
    ):
        raise DataReadinessError(
            "analyst event decisions are ambiguous by ticker and decision time"
        )
    spine = _target_decision_spine(panel_directory, records, policy=policy)
    target = spine.rename(
        columns={
            "decision_id": "target_decision_id",
            "security_id": "target_security_id",
        }
    )
    source = source.rename(
        columns={
            "decision_id": "analyst_revision_source_decision_id",
            "security_id": "analyst_revision_source_security_id",
        }
    )
    merged = source.merge(
        target,
        on=["ticker", "decision_time_utc"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    matched = merged["target_decision_id"].notna()
    source_cik = merged["analyst_revision_source_security_id"].map(_embedded_cik)
    target_cik = merged["target_security_id"].map(_embedded_cik)
    conflicting_cik = matched & source_cik.notna() & target_cik.notna() & source_cik.ne(target_cik)
    if bool(conflicting_cik.any()):
        raise DataReadinessError(
            "exact ticker/time event alignment found conflicting company identifiers"
        )
    merged["direct_decision_id_match"] = (
        matched
        & merged["analyst_revision_source_decision_id"]
        .astype(str)
        .eq(merged["target_decision_id"].astype(str))
    )
    merged["identity_alignment"] = "unmatched"
    merged.loc[matched, "identity_alignment"] = "exact_ticker_and_decision_time"
    rank_available = merged["rank_label"].notna()
    eligible = (
        matched
        & merged["feature_eligible"].fillna(False).astype(bool)
        & merged["label_eligible"].fillna(False).astype(bool)
        & merged["cross_section_eligible"].fillna(False).astype(bool)
        & merged["managed_path_eligible"].fillna(False).astype(bool)
        & rank_available
    )
    merged["inclusion_status"] = "excluded"
    merged.loc[eligible, "inclusion_status"] = "included"
    merged["exclusion_reason"] = ""
    reason_masks = (
        ("no_exact_ticker_and_decision_time_in_technical_panel", ~matched),
        ("technical_features_not_eligible", matched & ~merged["feature_eligible"].fillna(False).astype(bool)),
        ("outcome_label_not_eligible", matched & ~merged["label_eligible"].fillna(False).astype(bool)),
        ("cross_section_not_eligible", matched & ~merged["cross_section_eligible"].fillna(False).astype(bool)),
        ("managed_price_path_not_eligible", matched & ~merged["managed_path_eligible"].fillna(False).astype(bool)),
        ("rank_label_missing", matched & ~rank_available),
    )
    for reason, mask in reason_masks:
        empty = merged["exclusion_reason"].eq("")
        merged.loc[mask & empty, "exclusion_reason"] = reason
    merged.loc[eligible, "exclusion_reason"] = ""
    selected = merged.loc[eligible].copy()
    selected["analyst_revision_identity_alignment"] = selected[
        "identity_alignment"
    ].astype(str)
    selected["decision_id"] = selected["target_decision_id"].astype(str)
    selected["security_id"] = selected["target_security_id"].astype(str)
    selected["analyst_revision_source_decision_id"] = selected[
        "analyst_revision_source_decision_id"
    ].astype(str)
    selected["analyst_revision_source_security_id"] = selected[
        "analyst_revision_source_security_id"
    ].astype(str)
    selected = selected.drop(
        columns=[
            "target_decision_id",
            "target_security_id",
            "direct_decision_id_match",
            "identity_alignment",
            "inclusion_status",
            "exclusion_reason",
            "feature_eligible",
            "label_eligible",
            "cross_section_eligible",
            "managed_path_eligible",
            "rank_label",
            "ticker",
            "decision_time_utc",
        ]
    )
    episode_sizes = selected.groupby(
        "analyst_revision_episode_id", sort=False
    )["decision_id"].transform("size")
    selected["analyst_revision_episode_sample_weight"] = (
        1.0 / episode_sizes.astype("float64")
    )
    if selected["decision_id"].astype(str).duplicated().any():
        raise DataReadinessError("selected analyst decision identity is duplicated")
    audit = merged.copy()
    audit["rank_label_available"] = rank_available
    for column in ("feature_eligible", "label_eligible", "cross_section_eligible", "managed_path_eligible"):
        audit[column] = audit[column].fillna(False).astype(bool)
    audit["target_decision_id"] = audit["target_decision_id"].astype("string")
    audit["target_security_id"] = audit["target_security_id"].astype("string")
    audit = audit.loc[:, list(_ALIGNMENT_COLUMNS)].sort_values(
        ["decision_time_utc", "ticker", "analyst_revision_source_decision_id"],
        kind="stable",
    ).reset_index(drop=True)
    reason_counts = audit.loc[audit["inclusion_status"].eq("excluded"), "exclusion_reason"].value_counts()
    metrics = {
        "valid_news_linked_prediction_timestamps": len(source),
        "unique_latest_announcements_before_panel_alignment": int(
            source["analyst_revision_episode_id"].nunique()
        ),
        "exact_ticker_and_decision_time_matches": int(matched.sum()),
        "direct_old_decision_id_matches": int(merged["direct_decision_id_match"].sum()),
        "included_prediction_rows": len(selected),
        "included_unique_latest_announcements": int(
            selected["analyst_revision_episode_id"].nunique()
        ),
        **{f"excluded_{reason}": int(count) for reason, count in reason_counts.items()},
    }
    return selected.reset_index(drop=True), audit, metrics


def _embedded_cik(value: object) -> str | None:
    text = str(value).strip()
    if not text.startswith("cik:"):
        return None
    return text.split(":ticker:", maxsplit=1)[0]


def _attach_event_features(
    base: pd.DataFrame,
    selected_features: pd.DataFrame,
) -> pd.DataFrame:
    feature_index = selected_features.rename(
        columns={"security_id": "analyst_revision_security_id"}
    ).set_index("decision_id")
    attached = base.join(
        feature_index,
        on="decision_id",
        how="inner",
        validate="one_to_one",
    )
    if len(attached) != len(base):
        raise DataReadinessError("event feature join dropped eligible decisions")
    if bool(
        attached["security_id"]
        .astype(str)
        .ne(attached["analyst_revision_security_id"].astype(str))
        .any()
    ):
        raise DataReadinessError("event feature join changed issuer identity")
    return attached.drop(columns="analyst_revision_security_id")


def _shared_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    outcomes = sorted(
        column
        for column in frame.columns
        if column in _OUTCOME_EXACT_COLUMNS
        or column.startswith(_OUTCOME_PREFIXES)
    )
    required = tuple(dict.fromkeys((*_IDENTITY_COLUMNS, *outcomes, *_EVENT_AUDIT_COLUMNS)))
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise DataReadinessError(f"analyst ablation shared columns are missing: {missing}")
    return required


def _profile_features(
    profile: str,
    *,
    technical_features: Sequence[str],
) -> tuple[str, ...]:
    if profile == TECHNICAL_PROFILE:
        return tuple(technical_features)
    if profile == EVENT_PROFILE:
        return EVENT_FEATURE_COLUMNS
    if profile == COMBINED_PROFILE:
        return tuple((*technical_features, *EVENT_FEATURE_COLUMNS))
    raise DataReadinessError(f"unknown analyst ablation profile: {profile}")


def _base_records(panel: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    profiles = panel.get("files_by_profile")
    if not isinstance(profiles, Mapping):
        raise DataReadinessError("technical panel profile inventory is malformed")
    records = profiles.get(SWING_FEATURE_PROFILE)
    if not isinstance(records, list) or not records:
        raise DataReadinessError("technical panel has no partition inventory")
    if any(not isinstance(record, Mapping) for record in records):
        raise DataReadinessError("technical panel partition record is malformed")
    typed = [record for record in records if isinstance(record, Mapping)]
    months = [str(record.get("partition_month", "")) for record in typed]
    if not all(months) or len(months) != len(set(months)):
        raise DataReadinessError("technical panel month inventory is duplicated")
    return typed


def _verify_published_ablation(
    directory: Path,
    *,
    strategy_contract: StrategyContract,
    preverified_panel: Mapping[str, Any] | None = None,
    preverified_sources: _EventSources | None = None,
    precomputed_event_features: pd.DataFrame | None = None,
) -> Mapping[str, Any]:
    request = _json_object(directory / "_request.json")
    manifest = _json_object(directory / "_manifest.json")
    authority = _json_object(directory / "_authority.json")
    embedded = request.pop("request_sha256", None)
    if (
        request.get("schema") != REQUEST_SCHEMA
        or embedded != _json_sha256(request)
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("request_sha256") != embedded
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact_sha256") != file_sha256(directory / "_manifest.json")
        or authority.get("request_sha256") != embedded
        or authority.get("production_ready") is not False
        or authority.get("training_eligible") is not False
        or authority.get("research_training_eligible") is not True
        or authority.get("serving_eligible") is not False
        or manifest.get("production_ready") is not False
        or manifest.get("training_eligible") is not False
        or manifest.get("research_training_eligible") is not True
        or manifest.get("serving_eligible") is not False
        or request.get("strategy_contract_sha256") != strategy_contract.sha256()
    ):
        raise DataReadinessError("analyst-revision ablation root does not verify")
    policy_path = Path(str(request["policy_path"]))
    if file_sha256(policy_path) != request.get("policy_sha256"):
        raise DataReadinessError("analyst ablation policy lineage changed")
    policy = load_analyst_revision_ablation_policy(policy_path)
    if request.get("subtype_policy_sha256") != SUBTYPE_POLICY_SHA256:
        raise DataReadinessError("analyst subtype policy changed")
    panel_directory = Path(str(request["technical_panel_directory"]))
    panel = (
        preverified_panel
        if preverified_panel is not None
        else load_complete_swing_feature_panel(panel_directory)
    )
    if (
        file_sha256(panel_directory / "final" / "_manifest.json")
        != request.get("technical_panel_manifest_sha256")
        or file_sha256(panel_directory / "final" / "_authority.json")
        != request.get("technical_panel_authority_sha256")
    ):
        raise DataReadinessError("technical panel binding changed")
    source_authorities = [Path(str(item["directory"])) for item in request["event_authorities"]]
    source_audits = [Path(str(item["directory"])) for item in request["precision_audits"]]
    for item, path in zip(request["event_authorities"], source_authorities, strict=True):
        if file_sha256(path / "_authority.json") != item.get("authority_sha256"):
            raise DataReadinessError("event authority binding changed")
    for item, path in zip(request["precision_audits"], source_audits, strict=True):
        if file_sha256(path / "_authority.json") != item.get("authority_sha256"):
            raise DataReadinessError("precision audit binding changed")
    if preverified_sources is None:
        sources = _load_event_sources(source_authorities, source_audits, policy=policy)
    else:
        sources = preverified_sources
        if (
            tuple(authority.directory for authority in sources.authorities)
            != tuple(path.resolve() for path in source_authorities)
            or tuple(audit.directory for audit in sources.audits)
            != tuple(path.resolve() for path in source_audits)
        ):
            raise DataReadinessError("preverified analyst source identity differs")
    event_features = (
        precomputed_event_features
        if precomputed_event_features is not None
        else _build_event_features(sources, policy=policy)
    )
    base_records = _base_records(panel)
    selected_features, expected_alignment_audit, expected_alignment_metrics = (
        _align_event_features_to_panel(
            panel_directory,
            base_records,
            event_features,
            policy=policy,
        )
    )
    eligible_ids = set(selected_features["decision_id"].astype(str))
    if not eligible_ids:
        raise DataReadinessError("analyst ablation replay cohort is empty")
    raw_alignment = manifest.get("alignment_audit")
    if not isinstance(raw_alignment, Mapping):
        raise DataReadinessError("analyst identity-alignment audit record is missing")
    alignment_relative = str(raw_alignment.get("path", ""))
    alignment_path = directory / alignment_relative
    if (
        alignment_relative != "identity_alignment_audit.parquet"
        or not alignment_path.is_file()
        or file_sha256(alignment_path) != raw_alignment.get("sha256")
        or int(raw_alignment.get("rows", -1)) != len(expected_alignment_audit)
        or tuple(raw_alignment.get("columns", ())) != _ALIGNMENT_COLUMNS
        or manifest.get("alignment_metrics") != expected_alignment_metrics
    ):
        raise DataReadinessError("analyst identity-alignment audit binding differs")
    observed_alignment_audit = pd.read_parquet(alignment_path)
    try:
        pd.testing.assert_frame_equal(
            observed_alignment_audit,
            expected_alignment_audit,
            check_exact=True,
        )
    except AssertionError as exc:
        raise DataReadinessError(
            "analyst identity-alignment audit content differs"
        ) from exc
    expected_by_month: dict[str, pd.DataFrame] = {}
    for base_record in base_records:
        month = str(base_record["partition_month"])
        base_path = panel_directory / "final" / str(base_record["path"])
        base = pd.read_parquet(base_path)
        base["decision_id"] = base["decision_id"].astype(str)
        base = base.loc[base["decision_id"].isin(eligible_ids)].copy()
        if not base.empty:
            expected_by_month[month] = _attach_event_features(
                base,
                selected_features,
            )
        del base
        release_process_memory()
        _guard(policy, f"analyst ablation base replay {month}")
    if not expected_by_month:
        raise DataReadinessError("analyst ablation replay cohort is empty")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise DataReadinessError("analyst ablation has no files")
    technical_columns = swing_model_feature_columns(
        contract=strategy_contract,
        catalyst=False,
    )
    if (
        manifest.get("profiles") != list(PROFILES)
        or request.get("profiles") != list(PROFILES)
        or tuple(request.get("technical_feature_columns", ())) != technical_columns
        or tuple(manifest.get("technical_feature_columns", ())) != technical_columns
        or tuple(manifest.get("event_feature_columns", ())) != EVENT_FEATURE_COLUMNS
        or tuple(request.get("event_feature_columns", ())) != EVENT_FEATURE_COLUMNS
        or set(technical_columns).intersection(EVENT_FEATURE_COLUMNS)
    ):
        raise DataReadinessError("analyst ablation feature contract differs")
    shared_columns = tuple(manifest.get("shared_columns", ()))
    expected_shared_columns = _shared_columns(next(iter(expected_by_month.values())))
    if not shared_columns or shared_columns != expected_shared_columns:
        raise DataReadinessError("analyst ablation shared-column contract differs")
    expected_files = {
        "_request.json",
        "_manifest.json",
        "_authority.json",
        "identity_alignment_audit.parquet",
    }
    by_month: dict[str, list[Mapping[str, Any]]] = {}
    total_by_profile = {profile: 0 for profile in PROFILES}
    observed_profile_months: set[tuple[str, str]] = set()
    observed_technical_decisions: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("analyst ablation file record is malformed")
        profile = str(raw.get("feature_profile"))
        month = str(raw.get("partition_month"))
        relative = str(raw.get("path", ""))
        expected_relative = f"panel/feature_profile={profile}/month={month}/part.parquet"
        profile_month = (profile, month)
        if (
            profile not in PROFILES
            or relative != expected_relative
            or profile_month in observed_profile_months
            or month not in expected_by_month
        ):
            raise DataReadinessError("analyst ablation partition identity is invalid")
        observed_profile_months.add(profile_month)
        path = directory / relative
        expected_files.add(relative)
        if not path.is_file() or file_sha256(path) != raw.get("sha256"):
            raise DataReadinessError("analyst ablation partition hash changed")
        frame = pd.read_parquet(path)
        expected_features = _profile_features(
            profile,
            technical_features=technical_columns,
        )
        expected_columns = [*shared_columns, *expected_features, "feature_profile"]
        expected = expected_by_month[month].loc[
            :, [*shared_columns, *expected_features]
        ].copy()
        expected["feature_profile"] = profile
        expected = expected.sort_values("decision_id", kind="stable").reset_index(drop=True)
        if (
            list(frame.columns) != expected_columns
            or len(frame) != int(raw.get("rows", -1))
            or set(frame["feature_profile"].astype(str)) != {profile}
            or _sequence_sha256(frame["decision_id"].astype(str))
            != raw.get("decision_ids_sha256")
            or _frame_sha256(frame, shared_columns)
            != raw.get("shared_content_sha256")
            or list(raw.get("model_feature_columns", [])) != list(expected_features)
        ):
            raise DataReadinessError("analyst ablation partition content differs")
        try:
            pd.testing.assert_frame_equal(
                frame.reset_index(drop=True),
                expected,
                check_exact=True,
            )
        except AssertionError as exc:
            raise DataReadinessError(
                "analyst ablation differs from its technical panel or event authority"
            ) from exc
        if any(family in frame.columns for family in EVENT_FAMILIES if family != "analyst_revision"):
            raise DataReadinessError("blocked event family entered analyst ablation")
        latest = pd.to_datetime(
            frame["analyst_revision_latest_feature_available_at_utc"], utc=True, errors="coerce"
        )
        decision = pd.to_datetime(frame["decision_time_utc"], utc=True, errors="coerce")
        if latest.isna().any() or decision.isna().any() or bool((latest > decision).any()):
            raise DataReadinessError("analyst ablation contains future event evidence")
        if not frame["analyst_revision_source_coverage_known_3d"].astype(bool).all():
            raise DataReadinessError("unknown analyst coverage entered the cohort")
        if profile == TECHNICAL_PROFILE:
            observed_technical_decisions.update(frame["decision_id"].astype(str))
        by_month.setdefault(month, []).append(raw)
        total_by_profile[profile] += len(frame)
    observed_files = {
        str(path.relative_to(directory)).replace("\\", "/")
        for path in directory.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise DataReadinessError("analyst ablation file inventory differs")
    if observed_technical_decisions != eligible_ids:
        raise DataReadinessError("analyst ablation omitted or added eligible decisions")
    if set(by_month) != set(expected_by_month):
        raise DataReadinessError("analyst ablation omitted or added calendar months")
    for month, month_records in by_month.items():
        if {str(item["feature_profile"]) for item in month_records} != set(PROFILES):
            raise DataReadinessError(f"analyst ablation month is incomplete: {month}")
        if len({str(item["decision_ids_sha256"]) for item in month_records}) != 1 or len(
            {str(item["shared_content_sha256"]) for item in month_records}
        ) != 1:
            raise DataReadinessError("analyst ablation profiles differ in decisions or labels")
    rows = set(total_by_profile.values())
    if (
        len(rows) != 1
        or rows != {int(manifest.get("rows_per_profile", -1))}
        or sum(total_by_profile.values()) != int(manifest.get("total_rows", -1))
        or int(manifest.get("episode_count", -1))
        != int(selected_features["analyst_revision_episode_id"].nunique())
        or int(manifest.get("unique_latest_announcement_count", -1))
        != int(selected_features["analyst_revision_episode_id"].nunique())
    ):
        raise DataReadinessError("analyst ablation profile row totals differ")
    _guard(policy, "analyst ablation replay")
    return {"status": "complete", **manifest, "technical_panel": panel}


def _frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    data = frame.loc[:, list(columns)]
    hashes = pd.util.hash_pandas_object(data, index=False, categorize=False)
    header = json.dumps(
        [(column, str(data[column].dtype)) for column in columns],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + hashes.to_numpy(dtype="uint64").tobytes()).hexdigest()


def _sequence_sha256(values: Sequence[str] | pd.Series) -> str:
    return _json_sha256([str(value) for value in values])


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"analyst ablation JSON is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"analyst ablation JSON is not an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _guard(policy: AnalystRevisionAblationPolicy, stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=policy.maximum_process_memory_gib,
        headroom_gib=policy.memory_guard_headroom_gib,
        stage=stage,
    )
    assert_peak_memory_budget(
        hard_budget_gib=policy.maximum_process_memory_gib,
        headroom_gib=policy.memory_guard_headroom_gib,
        stage=stage,
    )
