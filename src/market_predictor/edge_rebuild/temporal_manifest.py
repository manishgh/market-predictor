"""Frozen, outcome-blind temporal assignments for the edge-rebuild program."""
from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Self

import exchange_calendars as xcals
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
)

TEMPORAL_MANIFEST_SCHEMA = "edge_rebuild.temporal_manifest.v2"
TEMPORAL_AUTHORITY_SCHEMA = "edge_rebuild.temporal_manifest_authority.v2"
SWING_PANEL_MANIFEST_SCHEMA = "edge_rebuild.swing_panel_materialization.v1"
SWING_PANEL_AUTHORITY_SCHEMA = (
    "edge_rebuild.swing_panel_materialization_authority.v1"
)
CAUSAL_MODELED_DECISION_START = date(2019, 7, 9)


class TemporalManifestConfig(BaseModel):
    """One immutable schedule contract for final swing-model evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    calendar: str
    modeled_decision_start: date
    initial_fit_start: date
    initial_fit_end: date
    initial_fit_expected_sessions: int = Field(ge=1_231, le=1_231)
    validation_embargo_start: date
    validation_embargo_end: date
    validation_embargo_expected_sessions: int = Field(ge=10, le=10)
    validation_start: date
    validation_end: date
    validation_expected_sessions: int = Field(ge=252, le=252)
    final_refit_start: date
    final_refit_end: date
    final_refit_expected_sessions: int = Field(ge=1_493, le=1_493)
    final_embargo_start: date
    final_embargo_end: date
    final_embargo_expected_sessions: int = Field(ge=10, le=10)
    locked_test_start: date
    locked_test_end: date
    locked_test_expected_sessions: int = Field(ge=251, le=251)
    warmup_sessions: int = Field(ge=250, le=250)
    label_horizon_sessions: int = Field(ge=10, le=10)
    unseen_security_holdout_fraction: float = Field(gt=0.0, lt=0.5)
    unseen_security_hash_seed: int
    unseen_security_assignment: str
    maximum_process_memory_gib: float = Field(ge=1.0, le=5.0)
    memory_guard_headroom_gib: float = Field(ge=0.5, le=2.0)

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.schema_version != TEMPORAL_MANIFEST_SCHEMA:
            raise ValueError("unsupported temporal-manifest schema")
        if self.calendar != "XNYS":
            raise ValueError("temporal manifest must use the XNYS calendar")
        governed_dates = (
            (self.modeled_decision_start, CAUSAL_MODELED_DECISION_START, "modeled decision start"),
            (self.initial_fit_start, CAUSAL_MODELED_DECISION_START, "initial fit start"),
            (self.initial_fit_end, date(2024, 5, 28), "initial fit end"),
            (self.validation_embargo_start, date(2024, 5, 29), "validation embargo start"),
            (self.validation_embargo_end, date(2024, 6, 11), "validation embargo end"),
            (self.validation_start, date(2024, 6, 12), "validation start"),
            (self.validation_end, date(2025, 6, 13), "validation end"),
            (self.final_refit_start, CAUSAL_MODELED_DECISION_START, "final refit start"),
            (self.final_refit_end, date(2025, 6, 13), "final refit end"),
            (self.final_embargo_start, date(2025, 6, 16), "final embargo start"),
            (self.final_embargo_end, date(2025, 6, 30), "final embargo end"),
            (self.locked_test_start, date(2025, 7, 1), "locked test start"),
            (self.locked_test_end, date(2026, 6, 30), "locked test end"),
        )
        changed = [label for observed, expected, label in governed_dates if observed != expected]
        if changed:
            raise ValueError("governed temporal dates changed: " + ", ".join(changed))
        if self.modeled_decision_start != self.initial_fit_start:
            raise ValueError("initial fit must start at the causal decision cutoff")
        ranges = (
            (self.initial_fit_start, self.initial_fit_end, "initial fit"),
            (self.validation_embargo_start, self.validation_embargo_end, "validation embargo"),
            (self.validation_start, self.validation_end, "validation"),
            (self.final_refit_start, self.final_refit_end, "final refit"),
            (self.final_embargo_start, self.final_embargo_end, "final embargo"),
            (self.locked_test_start, self.locked_test_end, "locked test"),
        )
        if any(start > end for start, end, _label in ranges):
            raise ValueError("temporal range start must not follow its end")
        if self.final_refit_start != self.modeled_decision_start:
            raise ValueError("final refit must include every post-cutoff development session")
        if self.final_refit_end != self.validation_end:
            raise ValueError("final refit must end with the validation window")
        if self.validation_embargo_expected_sessions < self.label_horizon_sessions:
            raise ValueError("validation embargo must cover the complete label horizon")
        if self.final_embargo_expected_sessions < self.label_horizon_sessions:
            raise ValueError("final embargo must cover the complete label horizon")
        if self.unseen_security_holdout_fraction != 0.20:
            raise ValueError("unseen-security holdout fraction must remain 20%")
        if self.unseen_security_assignment != "sha256_threshold_security_id_v1":
            raise ValueError("unseen-security assignment algorithm is not frozen")
        if self.memory_guard_headroom_gib >= self.maximum_process_memory_gib:
            raise ValueError("memory guard headroom must be below the hard budget")
        return self

    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TemporalFold:
    fold: int
    train_sessions: tuple[date, ...]
    embargo_sessions: tuple[date, ...]
    validation_sessions: tuple[date, ...]

    def record(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "train": _session_block(self.train_sessions),
            "embargo": _session_block(self.embargo_sessions),
            "validation": _session_block(self.validation_sessions),
        }


@dataclass(frozen=True)
class TemporalSchedule:
    target_sessions: tuple[date, ...]
    warmup_sessions: tuple[date, ...]
    folds: tuple[TemporalFold, ...]
    final_refit_sessions: tuple[date, ...]
    final_embargo_sessions: tuple[date, ...]
    locked_test_sessions: tuple[date, ...]

    @property
    def first_session(self) -> date:
        return self.target_sessions[0]

    @property
    def last_session(self) -> date:
        return self.target_sessions[-1]


@dataclass(frozen=True)
class PanelCoverage:
    root: Path
    authority_sha256: str
    manifest_sha256: str
    manifest: dict[str, Any]
    sessions: frozenset[date]


def load_temporal_manifest_config(path: Path) -> TemporalManifestConfig:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(
            f"temporal-manifest policy is unreadable: {path}"
        ) from exc
    try:
        return TemporalManifestConfig.model_validate(payload)
    except ValueError as exc:
        raise DataReadinessError(
            f"temporal-manifest policy is invalid: {path}"
        ) from exc


def build_temporal_schedule(config: TemporalManifestConfig) -> TemporalSchedule:
    """Build and verify explicit XNYS ranges without consulting market outcomes."""

    calendar = xcals.get_calendar(config.calendar)
    initial_fit = _verified_calendar_range(
        calendar,
        config.initial_fit_start,
        config.initial_fit_end,
        config.initial_fit_expected_sessions,
        "initial fit",
    )
    validation_embargo = _verified_calendar_range(
        calendar,
        config.validation_embargo_start,
        config.validation_embargo_end,
        config.validation_embargo_expected_sessions,
        "validation embargo",
    )
    validation = _verified_calendar_range(
        calendar,
        config.validation_start,
        config.validation_end,
        config.validation_expected_sessions,
        "validation",
    )
    final_refit = _verified_calendar_range(
        calendar,
        config.final_refit_start,
        config.final_refit_end,
        config.final_refit_expected_sessions,
        "final refit",
    )
    final_embargo = _verified_calendar_range(
        calendar,
        config.final_embargo_start,
        config.final_embargo_end,
        config.final_embargo_expected_sessions,
        "final embargo",
    )
    locked = _verified_calendar_range(
        calendar,
        config.locked_test_start,
        config.locked_test_end,
        config.locked_test_expected_sessions,
        "locked test",
    )
    _require_adjacent_ranges(calendar, initial_fit, validation_embargo, "initial fit", "validation embargo")
    _require_adjacent_ranges(calendar, validation_embargo, validation, "validation embargo", "validation")
    if final_refit != (*initial_fit, *validation_embargo, *validation):
        raise DataReadinessError("final refit is not the complete post-cutoff development range")
    _require_adjacent_ranges(calendar, final_refit, final_embargo, "final refit", "final embargo")
    _require_adjacent_ranges(calendar, final_embargo, locked, "final embargo", "locked test")

    decision_position = calendar.sessions.get_loc(pd.Timestamp(config.modeled_decision_start))
    warmup_values = calendar.sessions[
        decision_position - config.warmup_sessions : decision_position
    ]
    warmup = tuple(pd.Timestamp(value).date() for value in warmup_values)
    if len(warmup) != config.warmup_sessions:
        raise DataReadinessError("calendar cannot provide the governed warm-up")
    target = (*warmup, *final_refit, *final_embargo, *locked)
    return TemporalSchedule(
        target_sessions=target,
        warmup_sessions=warmup,
        folds=(
            TemporalFold(
                fold=1,
                train_sessions=initial_fit,
                embargo_sessions=validation_embargo,
                validation_sessions=validation,
            ),
        ),
        final_refit_sessions=final_refit,
        final_embargo_sessions=final_embargo,
        locked_test_sessions=locked,
    )


def _verified_calendar_range(
    calendar: Any,
    start: date,
    end: date,
    expected_sessions: int,
    label: str,
) -> tuple[date, ...]:
    sessions = tuple(
        pd.Timestamp(value).date()
        for value in calendar.sessions_in_range(start, end)
    )
    if not sessions or sessions[0] != start or sessions[-1] != end:
        raise DataReadinessError(label + " boundaries must be exact XNYS sessions")
    if len(sessions) != expected_sessions:
        raise DataReadinessError(
            label
            + " XNYS count differs from governance: expected="
            + str(expected_sessions)
            + ", observed="
            + str(len(sessions))
        )
    return sessions


def _require_adjacent_ranges(
    calendar: Any,
    left: tuple[date, ...],
    right: tuple[date, ...],
    left_label: str,
    right_label: str,
) -> None:
    boundary = tuple(
        pd.Timestamp(value).date()
        for value in calendar.sessions_in_range(left[-1], right[0])
    )
    if boundary != (left[-1], right[0]):
        raise DataReadinessError(
            left_label + " and " + right_label + " are not adjacent XNYS ranges"
        )


def publish_temporal_manifest(
    *,
    panel_directory: Path,
    policy_path: Path,
    strategy_contract: StrategyContract,
    output_directory: Path,
    config: TemporalManifestConfig | None = None,
) -> dict[str, Any]:
    """Publish immutable schedule and coverage evidence without opening outcomes."""

    resolved = config or load_temporal_manifest_config(policy_path)
    _validate_strategy_alignment(resolved, strategy_contract)
    if output_directory.exists():
        raise DataReadinessError(
            f"temporal-manifest output must be new: {output_directory}"
        )
    _guard(resolved, "temporal-manifest start")
    schedule = build_temporal_schedule(resolved)
    coverage = _load_panel_coverage(panel_directory, resolved)
    _guard(resolved, "temporal-manifest panel coverage")

    target = set(schedule.target_sessions)
    missing = tuple(session for session in schedule.target_sessions if session not in coverage.sessions)
    assignments = _assignment_frame(schedule, coverage.sessions)
    status = "complete" if not missing else "insufficient_history"
    request = {
        "schema": TEMPORAL_MANIFEST_SCHEMA,
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": file_sha256(policy_path),
        "config_sha256": resolved.sha256(),
        "strategy_contract_sha256": strategy_contract.sha256(),
        "panel_authority_sha256": coverage.authority_sha256,
        "panel_manifest_sha256": coverage.manifest_sha256,
    }

    staging = output_directory.with_name(
        f".{output_directory.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        staging.mkdir(parents=True)
        _write_json(staging / "_request.json", request)
        assignments_path = staging / "session_assignments.csv"
        assignments.to_csv(assignments_path, index=False, lineterminator="\n")
        folds_path = staging / "folds.json"
        _write_json(
            folds_path,
            {
                "schema": TEMPORAL_MANIFEST_SCHEMA,
                "folds": [fold.record() for fold in schedule.folds],
                "final_refit": _session_block(schedule.final_refit_sessions),
                "final_embargo": _session_block(schedule.final_embargo_sessions),
                "locked_test": _session_block(schedule.locked_test_sessions),
            },
        )
        assert_peak_memory_budget(
            hard_budget_gib=resolved.maximum_process_memory_gib,
            headroom_gib=resolved.memory_guard_headroom_gib,
            stage="temporal-manifest publication",
        )
        resources = memory_audit(
            hard_budget_gib=resolved.maximum_process_memory_gib,
            headroom_gib=resolved.memory_guard_headroom_gib,
        )
        manifest: dict[str, Any] = {
            "schema": TEMPORAL_MANIFEST_SCHEMA,
            "status": status,
            "request_sha256": file_sha256(staging / "_request.json"),
            "config_sha256": resolved.sha256(),
            "strategy_contract_sha256": strategy_contract.sha256(),
            "panel_strategy_contract_sha256": str(
                coverage.manifest["strategy_contract_sha256"]
            ),
            "calendar": resolved.calendar,
            "modeled_decision_start": resolved.modeled_decision_start.isoformat(),
            "window_rationale": (
                "approximately 4.9 years initial fit plus one validation year plus "
                "one locked test year; causal-news cutoff is authoritative"
            ),
            "target": _session_block(schedule.target_sessions),
            "warmup": _session_block(schedule.warmup_sessions),
            "initial_fit": _session_block(schedule.folds[0].train_sessions),
            "validation_embargo": _session_block(
                schedule.folds[0].embargo_sessions
            ),
            "validation": _session_block(schedule.folds[0].validation_sessions),
            "final_refit": _session_block(schedule.final_refit_sessions),
            "final_embargo": _session_block(schedule.final_embargo_sessions),
            "label_horizon_sessions": resolved.label_horizon_sessions,
            "locked_test": _session_block(schedule.locked_test_sessions),
            "unseen_security_scope": {
                "identity": "security_id",
                "fraction": resolved.unseen_security_holdout_fraction,
                "seed": resolved.unseen_security_hash_seed,
                "assignment": resolved.unseen_security_assignment,
            },
            "coverage": {
                "panel_first_session": str(coverage.manifest["first_session"]),
                "panel_last_session": str(coverage.manifest["last_session"]),
                "panel_sessions": len(coverage.sessions),
                "target_sessions_present": len(target & coverage.sessions),
                "target_sessions_missing": len(missing),
                "missing_ranges": _missing_ranges(
                    missing,
                    schedule.target_sessions,
                ),
                "outcomes_read": False,
                "columns_read": ["session_date_et", "decision_group_id"],
            },
            "resources": resources.to_record(),
            "artifacts": {
                "session_assignments.csv": file_sha256(assignments_path),
                "folds.json": file_sha256(folds_path),
            },
        }
        _write_json(staging / "_manifest.json", manifest)
        _write_json(
            staging / "_authority.json",
            {
                "schema": TEMPORAL_AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(staging / "_manifest.json"),
            },
        )
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_directory)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _validate_strategy_alignment(
    config: TemporalManifestConfig,
    contract: StrategyContract,
) -> None:
    mismatches = []
    if contract.validation.swing_walk_forward_folds != 1:
        mismatches.append("validation fold count")
    if (
        config.validation_embargo_expected_sessions
        != contract.validation.embargo_sessions
        or config.final_embargo_expected_sessions
        != contract.validation.embargo_sessions
    ):
        mismatches.append("embargo sessions")
    if config.label_horizon_sessions != contract.swing.horizon_sessions:
        mismatches.append("swing label horizon")
    if (
        config.unseen_security_holdout_fraction
        != contract.validation.unseen_ticker_holdout_fraction
    ):
        mismatches.append("unseen-security holdout fraction")
    if mismatches:
        raise DataReadinessError(
            "temporal and strategy contracts disagree: " + ", ".join(mismatches)
        )


def _load_panel_coverage(
    panel_directory: Path,
    config: TemporalManifestConfig,
) -> PanelCoverage:
    root = panel_directory / "final" if (panel_directory / "final").is_dir() else panel_directory
    authority_path = root / "_authority.json"
    manifest_path = root / "_manifest.json"
    authority = _read_json(authority_path)
    if (
        authority.get("schema") != SWING_PANEL_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
    ):
        raise DataReadinessError("swing panel authority is not complete or supported")
    if authority.get("artifact_sha256") != file_sha256(manifest_path):
        raise DataReadinessError("swing panel manifest hash does not match authority")
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != SWING_PANEL_MANIFEST_SCHEMA:
        raise DataReadinessError("swing panel manifest schema is unsupported")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise DataReadinessError("swing panel manifest has no partitions")

    observed_sessions: set[date] = set()
    query_sessions: dict[str, date] = {}
    for record in records:
        if not isinstance(record, dict):
            raise DataReadinessError("swing panel partition record is invalid")
        path = _bound_artifact_path(root, str(record.get("path", "")))
        if file_sha256(path) != str(record.get("sha256", "")):
            raise DataReadinessError(f"swing panel partition hash mismatch: {path}")
        frame = pd.read_parquet(
            path,
            columns=["session_date_et", "decision_group_id"],
        )
        sessions = pd.to_datetime(frame["session_date_et"], errors="coerce").dt.date
        groups = frame["decision_group_id"].astype("string").str.strip()
        if bool(sessions.isna().any()) or bool(groups.isna().any()) or bool(groups.eq("").any()):
            raise DataReadinessError("swing panel has invalid session or decision group")
        for group, session in zip(groups, sessions, strict=True):
            prior = query_sessions.setdefault(str(group), session)
            if prior != session:
                raise DataReadinessError("a decision_group_id spans multiple sessions")
        observed_sessions.update(sessions)
        del frame
        _guard(config, f"temporal-manifest partition {path.name}")
    if len(observed_sessions) != int(manifest.get("sessions", -1)):
        raise DataReadinessError("swing panel session count differs from its manifest")
    if (
        min(observed_sessions).isoformat() != str(manifest.get("first_session"))
        or max(observed_sessions).isoformat() != str(manifest.get("last_session"))
    ):
        raise DataReadinessError("swing panel session bounds differ from its manifest")
    return PanelCoverage(
        root=root,
        authority_sha256=file_sha256(authority_path),
        manifest_sha256=file_sha256(manifest_path),
        manifest=manifest,
        sessions=frozenset(observed_sessions),
    )


def _assignment_frame(
    schedule: TemporalSchedule,
    panel_sessions: frozenset[date],
) -> pd.DataFrame:
    warmup = set(schedule.warmup_sessions)
    locked = set(schedule.locked_test_sessions)
    final_refit = set(schedule.final_refit_sessions)
    final_embargo = set(schedule.final_embargo_sessions)
    fold_roles = {
        fold.fold: {
            "train": set(fold.train_sessions),
            "embargo": set(fold.embargo_sessions),
            "validation": set(fold.validation_sessions),
        }
        for fold in schedule.folds
    }
    records: list[dict[str, object]] = []
    for session in schedule.target_sessions:
        record: dict[str, object] = {
            "session": session.isoformat(),
            "global_role": (
                "warmup" if session in warmup else "locked_test" if session in locked else "development"
            ),
            "panel_available": session in panel_sessions,
        }
        for fold in schedule.folds:
            record[f"fold_{fold.fold}_role"] = _fold_role(
                session,
                fold_roles[fold.fold],
            )
        record["final_role"] = (
            "train"
            if session in final_refit
            else "embargo"
            if session in final_embargo
            else "locked_test"
            if session in locked
            else "not_used"
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _fold_role(session: date, roles: dict[str, set[date]]) -> str:
    if session in roles["train"]:
        return "train"
    if session in roles["embargo"]:
        return "embargo"
    if session in roles["validation"]:
        return "validation"
    return "not_used"


def _session_block(sessions: tuple[date, ...]) -> dict[str, object]:
    if not sessions:
        raise DataReadinessError("temporal session block is empty")
    return {
        "first_session": sessions[0].isoformat(),
        "last_session": sessions[-1].isoformat(),
        "sessions": len(sessions),
        "sha256": _session_sha256(sessions),
    }


def _session_sha256(sessions: tuple[date, ...]) -> str:
    payload = "\n".join(session.isoformat() for session in sessions).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _missing_ranges(
    missing: tuple[date, ...],
    target: tuple[date, ...],
) -> list[dict[str, object]]:
    if not missing:
        return []
    position = {session: index for index, session in enumerate(target)}
    ranges: list[dict[str, object]] = []
    start = previous = missing[0]
    count = 1
    for session in missing[1:]:
        if position[session] != position[previous] + 1:
            ranges.append(
                {
                    "first_session": start.isoformat(),
                    "last_session": previous.isoformat(),
                    "sessions": count,
                }
            )
            start = session
            count = 0
        previous = session
        count += 1
    ranges.append(
        {
            "first_session": start.isoformat(),
            "last_session": previous.isoformat(),
            "sessions": count,
        }
    )
    return ranges


def _bound_artifact_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise DataReadinessError("swing panel partition path is not repository-bound")
    path = (root / Path(*pure.parts)).resolve()
    if root.resolve() not in path.parents:
        raise DataReadinessError("swing panel partition escapes its artifact root")
    if not path.is_file():
        raise DataReadinessError(f"swing panel partition is missing: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"JSON artifact is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise DataReadinessError(f"JSON artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _guard(config: TemporalManifestConfig, stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
