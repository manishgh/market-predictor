"""Immutable source-side horizon for prospectively observed analyst revisions."""
from __future__ import annotations



import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, cast
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.intraday.datasets.event_preflight import (
    load_intraday_event_preflight_config,
)
from market_predictor.intraday.datasets.history import json_sha256
from market_predictor.edge_rebuild.prospective_broker_actions import (
    ProspectiveGeneration,
    ProspectivePoll,
    load_prospective_broker_action_generation,
    load_prospective_broker_action_poll,
)
from market_predictor.edge_rebuild.sp500_observed_memberships import (
    AUTHORITY_SCHEMA as OBSERVED_MEMBERSHIP_AUTHORITY_SCHEMA,
)
from market_predictor.edge_rebuild.sp500_observed_memberships import (
    load_observed_sp500_membership_authority,
)
from market_predictor.locking import LockTimeout, file_lock
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
)
from market_predictor.swing.event_families import (
    EVENT_FAMILY_POLICY_SHA256,
    EVENT_FAMILY_POLICY_VERSION,
    classify_event_families,
)
from market_predictor.core.errors import DataReadinessError

REQUEST_SCHEMA: Final = "edge_rebuild.prospective_analyst_revision_horizon_request.v1"
MANIFEST_SCHEMA: Final = "edge_rebuild.prospective_analyst_revision_horizon_manifest.v1"
AUTHORITY_SCHEMA: Final = "edge_rebuild.prospective_analyst_revision_horizon_authority.v1"
CLASSIFIED_ARTIFACT_TYPE: Final = "prospective_analyst_revision_classified_revisions"
EPISODE_ARTIFACT_TYPE: Final = "prospective_analyst_revision_episodes"
COVERAGE_ARTIFACT_TYPE: Final = "prospective_analyst_revision_collection_coverage"
CAPACITY_ARTIFACT_TYPE: Final = "prospective_analyst_revision_source_capacity"
_ARTIFACTS: Final = {
    "classified_revisions": ("classified_revisions.parquet", CLASSIFIED_ARTIFACT_TYPE),
    "episodes": ("episodes.parquet", EPISODE_ARTIFACT_TYPE),
    "coverage": ("coverage.parquet", COVERAGE_ARTIFACT_TYPE),
    "capacity_audit": ("capacity_audit.parquet", CAPACITY_ARTIFACT_TYPE),
}
_METADATA_FILES: Final = frozenset({"_request.json", "_manifest.json", "_authority.json"})


@dataclass(frozen=True, slots=True)
class ProspectiveAnalystRevisionHorizon:
    directory: Path
    classified_revisions: pd.DataFrame
    episodes: pd.DataFrame
    coverage: pd.DataFrame
    capacity_audit: pd.DataFrame
    manifest: Mapping[str, object]
    authority: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Inputs:
    generations: tuple[ProspectiveGeneration, ...]
    polls: tuple[ProspectivePoll, ...]
    generation_inventory: tuple[Mapping[str, object], ...]
    poll_inventory_sha256: str
    namespace_sha256: str
    registry_directory: str


def publish_prospective_analyst_revision_horizon(
    *,
    generation_directories: Sequence[Path],
    output_directory: Path,
    preflight_policy_path: Path,
    memory_hard_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.75,
) -> ProspectiveAnalystRevisionHorizon:
    """Publish classified prospective source evidence without authorizing training."""

    output = output_directory.resolve()
    try:
        with file_lock(output.with_name(f".{output.name}.publisher"), timeout=0.0):
            return _publish(
                generation_directories=generation_directories,
                output=output,
                preflight_policy_path=preflight_policy_path.resolve(),
                memory_hard_budget_gib=memory_hard_budget_gib,
                memory_headroom_gib=memory_headroom_gib,
            )
    except LockTimeout as exc:
        raise DataReadinessError(f"another process owns prospective horizon {output}") from exc


def _publish(
    *,
    generation_directories: Sequence[Path],
    output: Path,
    preflight_policy_path: Path,
    memory_hard_budget_gib: float,
    memory_headroom_gib: float,
) -> ProspectiveAnalystRevisionHorizon:
    policy = load_intraday_event_preflight_config(preflight_policy_path)
    if policy.source_family != "alpaca" or policy.event_family != "analyst_revision":
        raise DataReadinessError("prospective horizon requires the frozen Alpaca analyst policy")
    if memory_hard_budget_gib > 4.0 or memory_hard_budget_gib <= memory_headroom_gib:
        raise ValueError("prospective horizon memory policy must fit the 4 GiB hard limit")
    inputs = _load_inputs(generation_directories, output=output)
    assert_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective analyst horizon parent replay",
    )
    frames = _build_frames(inputs, policy=policy)
    request_payload: dict[str, object] = {
        "schema": REQUEST_SCHEMA,
        "generations": list(inputs.generation_inventory),
        "generation_inventory_sha256": json_sha256(list(inputs.generation_inventory)),
        "flattened_poll_inventory_sha256": inputs.poll_inventory_sha256,
        "security_identity_namespace_sha256": inputs.namespace_sha256,
        "registry_directory": inputs.registry_directory,
        "preflight_policy_path": str(preflight_policy_path),
        "preflight_policy_sha256": file_sha256(preflight_policy_path),
        "preflight_policy": asdict(policy),
        "event_family_policy_version": EVENT_FAMILY_POLICY_VERSION,
        "event_family_policy_sha256": EVENT_FAMILY_POLICY_SHA256,
        "episode_identity": "sha256(alpaca|provider_event_id|security_id)",
        "availability_policy": "earliest_observed_provider_response",
        "memory_hard_budget_gib": memory_hard_budget_gib,
        "memory_headroom_gib": memory_headroom_gib,
    }
    request_sha256 = json_sha256(request_payload)
    request = {**request_payload, "request_sha256": request_sha256}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise DataReadinessError(f"prospective horizon output must be new and empty: {output}")
    _atomic_json(output / "_request.json", request)
    child_inputs: dict[str, str] = {
        "request_sha256": request_sha256,
        "generation_inventory_sha256": str(request_payload["generation_inventory_sha256"]),
        "flattened_poll_inventory_sha256": inputs.poll_inventory_sha256,
        "event_family_policy_sha256": EVENT_FAMILY_POLICY_SHA256,
        "preflight_policy_sha256": str(request_payload["preflight_policy_sha256"]),
    }
    artifacts: dict[str, dict[str, object]] = {}
    for role, (filename, artifact_type) in _ARTIFACTS.items():
        path = output / filename
        frame = frames[role]
        write_canonical_artifact(
            frame,
            path,
            artifact_type=artifact_type,
            audit=_passing_audit(role, len(frame)),
            inputs=child_inputs,
            production_ready=False,
        )
        artifacts[role] = _artifact_record(path)
    capacity = frames["capacity_audit"].iloc[0]
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "request_sha256": request_sha256,
        "generation_count": len(inputs.generations),
        "poll_count": len(inputs.polls),
        "classified_revision_count": len(frames["classified_revisions"]),
        "analyst_episode_count": len(frames["episodes"]),
        "eligible_security_count": int(capacity["eligible_security_count"]),
        "source_capacity_status": str(capacity["source_capacity_status"]),
        "artifacts": artifacts,
        "artifact_manifest_hashes": {
            role: file_sha256(manifest_path_for(output / filename))
            for role, (filename, _) in _ARTIFACTS.items()
        },
        "training_eligible": False,
        "serving_eligible": False,
        "future_holdout_opened": False,
        "memory": memory_audit(
            hard_budget_gib=memory_hard_budget_gib,
            headroom_gib=memory_headroom_gib,
        ).to_record(),
    }
    assert_peak_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective analyst horizon publication",
    )
    _atomic_json(output / "_manifest.json", manifest)
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "state": "complete",
        "artifact": "_manifest.json",
        "artifact_sha256": file_sha256(output / "_manifest.json"),
        "request_sha256": request_sha256,
        "source_capacity_status": manifest["source_capacity_status"],
        "training_eligible": False,
        "serving_eligible": False,
        "future_holdout_opened": False,
    }
    _atomic_json(output / "_authority.json", authority)
    return load_prospective_analyst_revision_horizon(output)


def load_prospective_analyst_revision_horizon(
    directory: Path,
) -> ProspectiveAnalystRevisionHorizon:
    """Strictly replay a prospective analyst-event source horizon."""

    root = directory.resolve()
    expected_files = set(_METADATA_FILES)
    for filename, _ in _ARTIFACTS.values():
        expected_files.update({filename, f"{filename}.manifest.json", f"{filename}.lock"})
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected_files:
        raise DataReadinessError("prospective analyst horizon root inventory does not verify")
    if any(path.is_symlink() for path in root.iterdir()):
        raise DataReadinessError("prospective analyst horizon cannot contain symlinks")
    request = _json_object(root / "_request.json")
    request_payload = {str(key): value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = request.get("request_sha256")
    generation_records = request_payload.get("generations")
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request_sha256 != json_sha256(request_payload)
        or not isinstance(generation_records, list)
        or not generation_records
        or request_payload.get("generation_inventory_sha256") != json_sha256(generation_records)
        or request_payload.get("event_family_policy_version") != EVENT_FAMILY_POLICY_VERSION
        or request_payload.get("event_family_policy_sha256") != EVENT_FAMILY_POLICY_SHA256
    ):
        raise DataReadinessError("prospective analyst horizon request does not verify")
    policy_path = Path(str(request_payload.get("preflight_policy_path", ""))).resolve()
    if not policy_path.is_file() or file_sha256(policy_path) != request_payload.get("preflight_policy_sha256"):
        raise DataReadinessError("prospective analyst horizon preflight policy changed")
    policy = load_intraday_event_preflight_config(policy_path)
    if asdict(policy) != request_payload.get("preflight_policy"):
        raise DataReadinessError("prospective analyst horizon preflight values changed")
    directories = [Path(str(cast(Mapping[str, object], record).get("directory", ""))) for record in generation_records]
    inputs = _load_inputs(directories, output=root)
    if (
        list(inputs.generation_inventory) != generation_records
        or inputs.poll_inventory_sha256 != request_payload.get("flattened_poll_inventory_sha256")
        or inputs.namespace_sha256 != request_payload.get("security_identity_namespace_sha256")
        or inputs.registry_directory != request_payload.get("registry_directory")
    ):
        raise DataReadinessError("prospective analyst horizon parent lineage changed")
    manifest = _json_object(root / "_manifest.json")
    authority = _json_object(root / "_authority.json")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(root / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
        or manifest.get("training_eligible") is not False
        or manifest.get("serving_eligible") is not False
        or manifest.get("future_holdout_opened") is not False
        or authority.get("training_eligible") is not False
        or authority.get("serving_eligible") is not False
        or authority.get("future_holdout_opened") is not False
    ):
        raise DataReadinessError("prospective analyst horizon authority does not verify")
    records = manifest.get("artifacts")
    sidecars = manifest.get("artifact_manifest_hashes")
    if not isinstance(records, Mapping) or not isinstance(sidecars, Mapping):
        raise DataReadinessError("prospective analyst horizon artifact inventory is malformed")
    child_inputs: dict[str, str] = {
        "request_sha256": str(request_sha256),
        "generation_inventory_sha256": str(request_payload["generation_inventory_sha256"]),
        "flattened_poll_inventory_sha256": inputs.poll_inventory_sha256,
        "event_family_policy_sha256": EVENT_FAMILY_POLICY_SHA256,
        "preflight_policy_sha256": str(request_payload["preflight_policy_sha256"]),
    }
    loaded: dict[str, pd.DataFrame] = {}
    for role, (filename, artifact_type) in _ARTIFACTS.items():
        record = records.get(role)
        if not isinstance(record, Mapping):
            raise DataReadinessError(f"prospective analyst horizon {role} is missing")
        path = root / filename
        if (
            record.get("path") != filename
            or file_sha256(path) != record.get("sha256")
            or file_sha256(manifest_path_for(path)) != sidecars.get(role)
        ):
            raise DataReadinessError(f"prospective analyst horizon {role} hash changed")
        frame, child = load_canonical_artifact(path, expected_type=artifact_type, allow_research=True)
        if child.get("inputs") != child_inputs or child.get("production_ready") is not False:
            raise DataReadinessError(f"prospective analyst horizon {role} lineage changed")
        loaded[role] = frame
    expected = _build_frames(inputs, policy=policy)
    if any(not _frames_equal(loaded[role], expected[role]) for role in _ARTIFACTS):
        raise DataReadinessError("prospective analyst horizon does not replay from parents")
    capacity = loaded["capacity_audit"].iloc[0]
    if (
        int(manifest.get("generation_count", -1)) != len(inputs.generations)
        or int(manifest.get("poll_count", -1)) != len(inputs.polls)
        or int(manifest.get("classified_revision_count", -1)) != len(loaded["classified_revisions"])
        or int(manifest.get("analyst_episode_count", -1)) != len(loaded["episodes"])
        or int(manifest.get("eligible_security_count", -1)) != int(capacity["eligible_security_count"])
        or manifest.get("source_capacity_status") != capacity["source_capacity_status"]
        or authority.get("source_capacity_status") != capacity["source_capacity_status"]
    ):
        raise DataReadinessError("prospective analyst horizon counts do not verify")
    return ProspectiveAnalystRevisionHorizon(
        directory=root,
        classified_revisions=loaded["classified_revisions"],
        episodes=loaded["episodes"],
        coverage=loaded["coverage"],
        capacity_audit=loaded["capacity_audit"],
        manifest=manifest,
        authority=authority,
    )


def _load_inputs(generation_directories: Sequence[Path], *, output: Path) -> _Inputs:
    if not generation_directories:
        raise ValueError("generation_directories must not be empty")
    roots = tuple(path.resolve() for path in generation_directories)
    if len(roots) != len(set(roots)):
        raise DataReadinessError("prospective horizon contains duplicate generations")
    if any(output == root or output in root.parents or root in output.parents for root in roots):
        raise DataReadinessError("prospective horizon output and parents must be disjoint")
    generations = tuple(load_prospective_broker_action_generation(root) for root in roots)
    first_cutoffs = [pd.Timestamp(item.manifest["first_poll_at_utc"]) for item in generations]
    if first_cutoffs != sorted(first_cutoffs):
        raise DataReadinessError("prospective horizon generations must be chronological")
    generation_inventory = tuple(
        {
            "directory": str(item.directory),
            "request_sha256": item.request["request_sha256"],
            "manifest_sha256": file_sha256(item.directory / "_manifest.json"),
            "authority_sha256": file_sha256(item.directory / "_authority.json"),
            "poll_inventory_sha256": item.request["poll_inventory_sha256"],
        }
        for item in generations
    )
    poll_records: list[Mapping[str, object]] = []
    polls: list[ProspectivePoll] = []
    seen_poll_roots: set[Path] = set()
    for generation in generations:
        raw_records = generation.request.get("polls")
        if not isinstance(raw_records, list):
            raise DataReadinessError("prospective generation poll inventory is malformed")
        for raw in raw_records:
            if not isinstance(raw, Mapping):
                raise DataReadinessError("prospective generation poll record is malformed")
            poll_root = Path(str(raw.get("directory", ""))).resolve()
            if poll_root in seen_poll_roots:
                raise DataReadinessError("prospective horizon contains overlapping polls")
            seen_poll_roots.add(poll_root)
            poll = load_prospective_broker_action_poll(poll_root)
            polls.append(poll)
            poll_records.append(dict(raw))
    cutoffs = [pd.Timestamp(item.manifest["observed_at_utc"]) for item in polls]
    if cutoffs != sorted(cutoffs) or len(cutoffs) != len(set(cutoffs)):
        raise DataReadinessError("prospective horizon poll cutoffs are not unique and chronological")
    for previous, current in zip(polls, polls[1:], strict=False):
        parent = current.request.get("previous_poll")
        if not isinstance(parent, Mapping) or Path(str(parent.get("directory", ""))).resolve() != previous.directory:
            raise DataReadinessError("prospective horizon poll chain is not contiguous")
        if parent.get("authority_sha256") != file_sha256(previous.directory / "_authority.json"):
            raise DataReadinessError("prospective horizon previous-poll authority changed")
    namespace_values = {str(item.request["security_identity_namespace_sha256"]) for item in polls}
    registry_values = {str(item.request["registry_directory"]) for item in polls}
    if len(namespace_values) != 1 or len(registry_values) != 1:
        raise DataReadinessError("prospective horizon namespace or registry changed")
    return _Inputs(
        generations=generations,
        polls=tuple(polls),
        generation_inventory=generation_inventory,
        poll_inventory_sha256=json_sha256(poll_records),
        namespace_sha256=next(iter(namespace_values)),
        registry_directory=next(iter(registry_values)),
    )


def _build_frames(inputs: _Inputs, *, policy: Any) -> dict[str, pd.DataFrame]:
    revisions = _merge_revisions(inputs.generations)
    first_poll_by_revision: dict[tuple[str, str], ProspectivePoll] = {}
    for poll in inputs.polls:
        for row in poll.observations.to_dict("records"):
            key = (str(row["revision_id"]), str(row["ticker"]))
            first_poll_by_revision.setdefault(key, poll)
    company_maps = {poll.directory: _causal_company_map(poll) for poll in inputs.polls}
    security_conflicts = (
        revisions.loc[revisions["candidate_security_id"].astype(str).ne("")]
        .groupby(["provider_event_id", "ticker"], sort=False)["candidate_security_id"]
        .nunique()
        .gt(1)
    )
    rows: list[dict[str, object]] = []
    classifier_inputs: list[dict[str, object]] = []
    for row in revisions.to_dict("records"):
        key = (str(row["revision_id"]), str(row["ticker"]))
        first_seen_poll = first_poll_by_revision.get(key)
        security_id = str(row["candidate_security_id"])
        conflict = bool(security_conflicts.get((str(row["provider_event_id"]), str(row["ticker"])), False))
        eligible = bool(row["identity_eligible"]) and not bool(row["provider_timestamp_anomaly"]) and not conflict
        revision_event_id = hashlib.sha256(
            f"{row['revision_id']}|{row['ticker']}|{security_id}".encode()
        ).hexdigest()
        company = ""
        company_available: object = pd.NaT
        if first_seen_poll is not None:
            company, company_available = company_maps[first_seen_poll.directory].get(
                (str(row["ticker"]), security_id),
                ("", pd.NaT),
            )
        reason = ""
        if conflict:
            reason = "security_identity_changed_across_horizon"
        elif not bool(row["identity_eligible"]):
            reason = str(row["identity_ineligible_reason"] or "identity_ineligible")
        elif bool(row["provider_timestamp_anomaly"]):
            reason = "provider_timestamp_anomaly"
        record = {
            "revision_event_id": revision_event_id,
            "revision_id": str(row["revision_id"]),
            "provider_event_id": str(row["provider_event_id"]),
            "ticker": str(row["ticker"]),
            "security_id": security_id,
            "source_family": "alpaca",
            "relation_channel": "direct_issuer",
            "title": str(row["title"]),
            "published_at_utc": row["published_at_utc"],
            "provider_updated_at_utc": row["provider_updated_at_utc"],
            "revision_first_seen_at_utc": row["revision_first_seen_at_utc"],
            "event_first_seen_at_utc": row["event_first_seen_at_utc"],
            "production_available_at_utc": row["production_available_at_utc"],
            "identity_eligible": eligible,
            "issuer_company": company,
            "issuer_company_available_at_utc": company_available,
            "classified_analyst_revision": False,
            "classification_rule_id": "",
            "classification_basis": "",
            "matched_text": "",
            "eligibility_reason": reason,
        }
        rows.append(record)
        if eligible:
            classifier_inputs.append(
                {
                    "event_id": revision_event_id,
                    "security_id": security_id,
                    "ticker": str(row["ticker"]),
                    "source_family": "alpaca",
                    "feature_available_at_utc": row["production_available_at_utc"],
                    "title": str(row["title"]),
                    "issuer_company": company,
                    "issuer_company_available_at_utc": company_available,
                }
            )
    classified = classify_event_families(pd.DataFrame(classifier_inputs)) if classifier_inputs else pd.DataFrame()
    analyst = classified[classified["event_family"].eq("analyst_revision")] if not classified.empty else classified
    by_event = {str(item["event_id"]): item for item in analyst.to_dict("records")}
    for record in rows:
        match = by_event.get(str(record["revision_event_id"]))
        if match is not None:
            record["classified_analyst_revision"] = True
            record["classification_rule_id"] = str(match["classification_rule_id"])
            record["classification_basis"] = str(match["classification_basis"])
            record["matched_text"] = str(match["matched_text"])
    classified_revisions = pd.DataFrame(rows).sort_values(
        ["revision_first_seen_at_utc", "provider_event_id", "ticker", "revision_id"], kind="stable"
    ).reset_index(drop=True)
    classified_revisions = _normalize_classified_revisions(classified_revisions)
    episodes = _episodes(classified_revisions)
    coverage_parts: list[pd.DataFrame] = []
    generation_by_poll = {
        Path(str(record["directory"])).resolve(): generation.directory
        for generation in inputs.generations
        for record in cast(list[Mapping[str, object]], generation.request["polls"])
    }
    for poll in inputs.polls:
        part = poll.source_collections.copy()
        identity = poll.identity_audit.loc[
            :,
            [
                "ticker",
                "candidate_security_id",
                "identity_eligible",
                "identity_ineligible_reason",
            ],
        ].rename(columns={"candidate_security_id": "security_id"})
        if bool(identity["ticker"].duplicated().any()):
            raise DataReadinessError("prospective poll identity contains duplicate tickers")
        part = part.merge(identity, on="ticker", how="left", validate="one_to_one")
        if bool(part["security_id"].isna().any()):
            raise DataReadinessError("prospective source coverage lacks exact poll identity")
        part["poll_observed_at_utc"] = pd.Timestamp(poll.manifest["observed_at_utc"])
        part["poll_authority_sha256"] = file_sha256(poll.directory / "_authority.json")
        part["generation_directory"] = str(generation_by_poll[poll.directory])
        coverage_parts.append(part)
    coverage = pd.concat(coverage_parts, ignore_index=True).sort_values(
        ["scheduled_poll_at_utc", "ticker"], kind="stable"
    ).reset_index(drop=True)
    for column in (
        "requested_start_utc",
        "requested_end_utc",
        "started_at_utc",
        "completed_at_utc",
        "scheduled_poll_at_utc",
        "previous_poll_at_utc",
        "poll_observed_at_utc",
    ):
        coverage[column] = pd.Series(
            pd.to_datetime(coverage[column], utc=True),
            dtype="datetime64[us, UTC]",
        )
    observation_dates = pd.to_datetime(coverage["scheduled_poll_at_utc"], utc=True).dt.date.nunique()
    securities = int(episodes["security_id"].nunique()) if not episodes.empty else 0
    episode_count = len(episodes)
    source_ready = (
        episode_count >= int(policy.minimum_unique_event_episodes)
        and securities >= int(policy.minimum_securities)
        and observation_dates >= int(policy.minimum_fit_sessions)
    )
    capacity = pd.DataFrame(
        [
            {
                "generation_count": len(inputs.generations),
                "poll_count": len(inputs.polls),
                "observation_date_count": observation_dates,
                "retained_revision_count": len(classified_revisions),
                "classified_analyst_revision_count": int(classified_revisions["classified_analyst_revision"].sum()),
                "unique_analyst_episode_count": episode_count,
                "eligible_security_count": securities,
                "minimum_unique_event_episodes": int(policy.minimum_unique_event_episodes),
                "minimum_securities": int(policy.minimum_securities),
                "minimum_fit_sessions": int(policy.minimum_fit_sessions),
                "source_capacity_status": "ready_for_matched_preflight" if source_ready else "blocked",
                "matched_decision_capacity_evaluated": False,
                "training_eligible": False,
                "serving_eligible": False,
                "future_holdout_opened": False,
            }
        ]
    )
    return {
        "classified_revisions": classified_revisions,
        "episodes": episodes,
        "coverage": coverage,
        "capacity_audit": capacity,
    }


def _merge_revisions(generations: Sequence[ProspectiveGeneration]) -> pd.DataFrame:
    data = pd.concat([item.revisions for item in generations], ignore_index=True)
    invariant = [
        "provider_event_id",
        "ticker",
        "published_at_utc",
        "provider_updated_at_utc",
        "source",
        "title",
        "url",
        "summary",
        "text",
        "raw_sha256",
    ]
    rows: list[dict[str, object]] = []
    for (_, _), group in data.groupby(["revision_id", "ticker"], sort=True):
        for column in invariant:
            if group[column].fillna("").astype(str).nunique(dropna=False) != 1:
                raise DataReadinessError(f"prospective revision changed invariant field {column}")
        first = group.sort_values("revision_first_seen_at_utc", kind="stable").iloc[0].to_dict()
        security_ids = sorted(value for value in set(group["candidate_security_id"].astype(str)) if value)
        eligible = group[group["identity_eligible"].astype(bool)]
        identity_conflict = len(security_ids) != 1
        first["candidate_security_id"] = security_ids[0] if len(security_ids) == 1 else ""
        first["identity_eligible"] = not eligible.empty and not identity_conflict
        first["identity_ineligible_reason"] = (
            "" if first["identity_eligible"] else "identity_changed_across_generations" if identity_conflict else "identity_never_eligible"
        )
        first_seen = pd.to_datetime(group["revision_first_seen_at_utc"], utc=True).min()
        identity_first = pd.to_datetime(eligible["identity_first_eligible_at_utc"], utc=True, errors="coerce").min()
        first["revision_first_seen_at_utc"] = first_seen
        first["event_first_seen_at_utc"] = pd.to_datetime(group["event_first_seen_at_utc"], utc=True).min()
        first["last_seen_at_utc"] = pd.to_datetime(group["last_seen_at_utc"], utc=True).max()
        first["identity_first_eligible_at_utc"] = identity_first
        first["production_available_at_utc"] = max(first_seen, identity_first) if first["identity_eligible"] else pd.NaT
        first["observation_count"] = int(group["observation_count"].sum())
        first["provider_timestamp_anomaly"] = bool(group["provider_timestamp_anomaly"].astype(bool).any())
        rows.append(first)
    return pd.DataFrame(rows).sort_values(
        ["revision_first_seen_at_utc", "provider_event_id", "ticker"], kind="stable"
    ).reset_index(drop=True)


def _episodes(classified: pd.DataFrame) -> pd.DataFrame:
    admitted = classified[
        classified["identity_eligible"].astype(bool)
        & classified["classified_analyst_revision"].astype(bool)
    ]
    columns = [
        "source_episode_id",
        "family_event_id",
        "provider_event_id",
        "ticker",
        "security_id",
        "first_qualifying_revision_event_id",
        "feature_available_at_utc",
        "title",
        "classification_rule_id",
        "classification_basis",
        "revision_count",
        "event_family_policy_sha256",
    ]
    rows: list[dict[str, object]] = []
    for (provider_event_id, security_id), group in admitted.groupby(
        ["provider_event_id", "security_id"], sort=True
    ):
        first = group.sort_values("production_available_at_utc", kind="stable").iloc[0]
        source_episode_id = hashlib.sha256(
            f"alpaca|{provider_event_id}|{security_id}".encode()
        ).hexdigest()
        rows.append(
            {
                "source_episode_id": source_episode_id,
                "family_event_id": hashlib.sha256(
                    f"{source_episode_id}|analyst_revision|{EVENT_FAMILY_POLICY_SHA256}".encode()
                ).hexdigest(),
                "provider_event_id": str(provider_event_id),
                "ticker": str(first["ticker"]),
                "security_id": str(security_id),
                "first_qualifying_revision_event_id": str(first["revision_event_id"]),
                "feature_available_at_utc": first["production_available_at_utc"],
                "title": str(first["title"]),
                "classification_rule_id": str(first["classification_rule_id"]),
                "classification_basis": str(first["classification_basis"]),
                "revision_count": len(group),
                "event_family_policy_sha256": EVENT_FAMILY_POLICY_SHA256,
            }
        )
    result = pd.DataFrame(rows, columns=columns).sort_values(
        ["feature_available_at_utc", "provider_event_id", "security_id"], kind="stable"
    ).reset_index(drop=True)
    for column in (
        "source_episode_id",
        "family_event_id",
        "provider_event_id",
        "ticker",
        "security_id",
        "first_qualifying_revision_event_id",
        "title",
        "classification_rule_id",
        "classification_basis",
        "event_family_policy_sha256",
    ):
        result[column] = result[column].astype("string")
    result["feature_available_at_utc"] = pd.Series(
        pd.to_datetime(result["feature_available_at_utc"], utc=True),
        dtype="datetime64[us, UTC]",
    )
    result["revision_count"] = result["revision_count"].astype("int64")
    return result


def _normalize_classified_revisions(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp_columns = (
        "published_at_utc",
        "provider_updated_at_utc",
        "revision_first_seen_at_utc",
        "event_first_seen_at_utc",
        "production_available_at_utc",
        "issuer_company_available_at_utc",
    )
    for column in timestamp_columns:
        frame[column] = pd.Series(
            pd.to_datetime(frame[column], utc=True),
            dtype="datetime64[us, UTC]",
        )
    string_columns = set(frame.columns).difference(
        {*timestamp_columns, "identity_eligible", "classified_analyst_revision"}
    )
    for column in sorted(string_columns):
        frame[column] = frame[column].astype("string")
    frame["identity_eligible"] = frame["identity_eligible"].astype(bool)
    frame["classified_analyst_revision"] = frame[
        "classified_analyst_revision"
    ].astype(bool)
    return frame


def _causal_company_map(
    poll: ProspectivePoll,
) -> dict[tuple[str, str], tuple[str, object]]:
    membership_root = Path(str(poll.request["membership_authority_directory"])).resolve()
    authority = _json_object(membership_root / "_authority.json")
    if authority.get("schema") != OBSERVED_MEMBERSHIP_AUTHORITY_SCHEMA:
        return {}
    observed = load_observed_sp500_membership_authority(membership_root)
    observed_at = pd.Timestamp(observed.manifest["observed_at_utc"])
    if observed_at > pd.Timestamp(poll.manifest["observed_at_utc"]):
        raise DataReadinessError("issuer-company anchor was observed after the poll")
    anchor = pd.read_csv(membership_root / "current_anchor.csv", dtype=str, keep_default_na=False)
    if not {"ticker", "company", "cik"}.issubset(anchor.columns):
        raise DataReadinessError("observed membership issuer anchor is incomplete")
    return {
        (str(row["ticker"]).upper(), f"cik:{str(row['cik']).zfill(10)}"): (
            str(row["company"]),
            observed_at,
        )
        for row in anchor.to_dict("records")
        if str(row["company"]).strip()
    }


def _artifact_record(path: Path) -> dict[str, object]:
    return {"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}


def _passing_audit(name: str, rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name=name,
                status="pass",
                failures=0,
                rows_checked=rows,
                detail=f"{rows} rows replayed",
            ),
        )
    )


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=True,
            check_like=False,
        )
    except AssertionError:
        return False
    return True


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
