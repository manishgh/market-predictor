"""One verified original research snapshot for reaction publication and replay.

Historical formats stay inside the historical inspector. This owner neither
executes archived producers nor substitutes current adjusted query units.
"""
from __future__ import annotations

import copy
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research import original_relationship_replay as replay
from market_predictor.resources import release_process_memory
from market_predictor.swing.contracts.corrected_outcomes import CorrectedOutcomePolicy
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.research_features import ResearchFeaturePolicy
from market_predictor.swing.contracts.return_feature_profiles import ReturnRelationshipSources
from market_predictor.swing.contracts.return_relationship_publication import VerifiedReturnRelationshipPublication
from market_predictor.swing.contracts.return_relationship_reuse import RelationshipReusePolicy
from market_predictor.swing.datasets.corrected_outcomes import load_corrected_outcome_policy
from market_predictor.swing.datasets.relationship_historical_evidence import historical_month, original_stock_bars
from market_predictor.swing.datasets.return_relationship_integrity import check_files, guard, pins, read_object
from market_predictor.swing.datasets.return_relationship_reuse import ReuseEvidence, _historical_spy, inspect_reuse_evidence
from market_predictor.swing.datasets.return_relationship_rows import IDENTITY_COLUMNS


class OriginalRelationshipInputPolicy(Protocol):
    @property
    def parent_publication(self) -> SourcePin: ...

    @property
    def parent_saved_row_verification(self) -> SourcePin: ...

    @property
    def original_snapshot_replay(self) -> SourcePin: ...


@dataclass(frozen=True)
class OriginalRelationshipInputPins:
    """Actual candle verification needs no invented news qualification authority."""

    parent_publication: SourcePin
    parent_saved_row_verification: SourcePin
    original_snapshot_replay: SourcePin


def implementation_files(root: Path) -> dict[str, str]:
    """Bind the new owner separately, leaving the original replay identity intact."""
    path = Path(__file__).resolve()
    if path != root / "src/market_predictor/research/original_relationship_inputs.py":
        raise DataReadinessError("original input owner must execute from its bound repository")
    # All other imported local owners already belong to the reproduced closure.
    return {path.relative_to(root).as_posix(): file_sha256(path)}


class OriginalRelationshipInputs:
    """Created only inside the verified owner; returned metadata cannot mutate it."""

    _root: Path
    _receipt_pin: SourcePin
    _evidence: ReuseEvidence
    _outcome: CorrectedOutcomePolicy
    _parent: VerifiedReturnRelationshipPublication
    _files: dict[str, str]
    _historical: dict[str, str]
    _sources: ReturnRelationshipSources
    _groups: dict[str, frozenset[str]]
    _ownership: dict[str, str]
    _history_sessions: tuple[date, ...]
    _maximum_system_used_percent: float
    _active: bool

    def __init__(self) -> None:
        raise TypeError("original inputs must be created by the verified context manager")

    def _ensure_active(self) -> None:
        if not self._active:
            raise DataReadinessError("original input context is closed")

    @property
    def parent(self) -> VerifiedReturnRelationshipPublication:
        self._ensure_active()
        return copy.deepcopy(self._parent)

    @property
    def receipt_pin(self) -> SourcePin:
        self._ensure_active()
        return self._receipt_pin

    @property
    def original_sources(self) -> ReturnRelationshipSources:
        self._ensure_active()
        return self._sources.model_copy(deep=True)

    @property
    def source_files(self) -> dict[str, str]:
        self._ensure_active()
        return dict(self._files)

    @property
    def historical_implementation_files(self) -> dict[str, str]:
        self._ensure_active()
        return dict(self._historical)

    @property
    def source_basis(self) -> dict[str, Any]:
        self._ensure_active()
        return copy.deepcopy(self._evidence.relationships.request["source_basis"])

    @property
    def history_sessions(self) -> tuple[date, ...]:
        self._ensure_active()
        return self._history_sessions

    def recheck(self) -> None:
        self._ensure_active()
        guard(self._maximum_system_used_percent)
        check_files(self._root, self._files)

    def _read_pins(self, files: dict[str, str]) -> None:
        if any(self._files.get(name) != digest for name, digest in pins(self._root, files).items()):
            raise DataReadinessError("original reader consumed a source outside the reproduced receipt")
        check_files(self._root, files)

    def read_parent_month(self, month: str) -> pd.DataFrame:
        self._ensure_active()
        if month not in self._parent.months:
            raise DataReadinessError("original parent month is outside the frozen population")
        guard(self._maximum_system_used_percent)
        frame = historical_month(self._evidence.relationships, month)
        if (not frame.feature_profile.eq("technical_relationships").all()
                or not set(frame.decision_id).issubset(self._ownership)):
            raise DataReadinessError("original monthly profile or ownership differs")
        return frame

    def iter_month_groups(self, month: str) -> Iterator[tuple[str, pd.DataFrame]]:
        """Intersect independently proved complete owners with one saved month."""
        frame = self.read_parent_month(month)
        owners = frame.decision_id.map(self._ownership)
        if owners.isna().any():
            raise DataReadinessError("original month contains unowned decisions")
        seen: set[str] = set()
        monthly_ids = set(frame.decision_id)
        for key in sorted(owners.unique()):
            self._ensure_active()
            group = frame.loc[owners.eq(key)].reset_index(drop=True)
            identifiers = set(group.decision_id)
            if (identifiers != self._groups[key].intersection(monthly_ids) or identifiers & seen
                    or len(identifiers) != len(group)):
                raise DataReadinessError("original monthly ownership overlaps or loses decisions")
            seen.update(identifiers)
            yield str(key), group
        if seen != monthly_ids:
            raise DataReadinessError("original monthly groups do not cover their parent")

    def read_stock(self, original_group_key: str) -> pd.DataFrame:
        self._ensure_active()
        if original_group_key not in self._groups:
            raise DataReadinessError("stock read requires an original verified group key")
        guard(self._maximum_system_used_percent)
        consumed: dict[str, str] = {}
        bars = original_stock_bars(self._root, self._evidence.relationships, original_group_key, consumed)
        self._read_pins(consumed)
        replay._validate_physical_history(bars, benchmark=False)
        item = self._evidence.relationships.request["stock_inventory"][original_group_key]
        if ("security_id" not in bars or not bars.security_id.eq(item["security_id"]).all()
                or not bars.ticker.eq(item["artifact"]["ticker"]).all()):
            raise DataReadinessError("original physical stock identity differs from its inspected inventory")
        if item["kind"] == "corrected":
            for rule in self._outcome.decision_corrections:
                if rule.security_id == item["security_id"]:
                    bars.loc[bars.session_date_et.between(rule.first_session, rule.last_session), "ticker"] = rule.ticker
        return bars

    def read_spy(self) -> pd.DataFrame:
        self._ensure_active()
        guard(self._maximum_system_used_percent)
        consumed: dict[str, str] = {}
        bars = _historical_spy(self._root, self._evidence, consumed)
        self._read_pins(consumed)
        replay._validate_physical_history(bars, benchmark=True)
        return bars


def _open(root: Path, policy: OriginalRelationshipInputPolicy) -> OriginalRelationshipInputs:
    guard(90.0)
    pin = policy.original_snapshot_replay
    saved = read_object(inside(root, pin.path), pin.sha256)
    if saved.get("schema") != replay.SCHEMA or saved.get("status") != "passed_original_snapshot_replay":
        raise DataReadinessError("original input receipt is not a passed original replay")
    config = SourcePin.model_validate(saved["config"])
    reuse = RelationshipReusePolicy.model_validate(read_object(inside(root, config.path), config.sha256))
    if (reuse.historical_relationship_publication != policy.parent_publication
            or reuse.historical_relationship_receipt != policy.parent_saved_row_verification):
        raise DataReadinessError("original input receipt belongs to a different parent or numerical window")
    local = implementation_files(root)
    actual = replay._replay(root=root, config=config,
        failed_comparison=SourcePin.model_validate(saved["failed_current_comparison"]))
    if json_sha256(saved) != json_sha256(actual):
        raise DataReadinessError("original input receipt failed independent reproduction")
    evidence = inspect_reuse_evidence(root, reuse)
    files = pins(root, saved["source_files"])
    if any(files.get(name) != digest for name, digest in evidence.files.items()):
        raise DataReadinessError("original receipt omits original parent evidence")
    feature = ResearchFeaturePolicy.model_validate(tomllib.loads(inside(root, reuse.feature_config.path).read_text(encoding="utf-8")))
    correction = feature.outcome_source_config
    if files.get(correction.path) != correction.sha256:
        raise DataReadinessError("original correction policy is outside the reproduced receipt")
    outcome = load_corrected_outcome_policy(root, Path(correction.path), correction.sha256)
    sources = ReturnRelationshipSources.model_validate(saved["original_sources"])
    request = evidence.relationships.request
    if (sources.model_dump(mode="json") != request["sources"]
            or sources.baseline_authority_sha256 != reuse.historical_parent_publication.sha256
            or saved["original_stock_inventory_sha256"] != json_sha256(request["stock_inventory"])):
        raise DataReadinessError("original source basis or stock ownership differs")
    pieces: list[pd.DataFrame] = []
    for month in sorted(evidence.relationships.manifest["months"]):
        guard(90.0)
        pieces.append(historical_month(evidence.relationships, month, list(IDENTITY_COLUMNS)))
    population = pd.concat(pieces, ignore_index=True)
    del pieces
    if (len(population) != saved["rows"] or population.decision_id.duplicated().any()
            or json_sha256(sorted(population.decision_id)) != saved["decision_ids_sha256"]):
        raise DataReadinessError("original ownership population differs from the replay")
    assigned = replay._assign_original_groups(population, request["stock_inventory"])
    groups = {key: frozenset(frame.decision_id) for key, frame in assigned.items()}
    ownership = {identifier: key for key, identifiers in groups.items() for identifier in identifiers}
    if set(groups) != set(saved["groups"]):
        raise DataReadinessError("original owner group inventory differs from the replay")
    for key, identifiers in groups.items():
        checked = saved["groups"][key]
        if (checked["rows"] != len(identifiers) or checked["decision_ids_sha256"] != json_sha256(sorted(identifiers))
                or checked["quarantine"] != request["stock_inventory"][key]["quarantine"]):
            raise DataReadinessError("original replay group population or abstention differs")
    del population, assigned
    release_process_memory()
    names = tuple(request["model_columns"])
    clocks = dict(request["availability_columns"])
    if len(names) != 124 or len(set(names)) != 124:
        raise DataReadinessError("original parent must retain its complete ordered 124 features")
    for record in evidence.relationships.manifest["months"].values():
        child = record["profiles"]["technical_relationships"]
        if tuple(child["model_columns"]) != names or child["availability_columns"] != clocks:
            raise DataReadinessError("original parent feature contract changes between months")
    files = pins(root, files, local, {pin.path: pin.sha256})
    parent = VerifiedReturnRelationshipPublication(request, evidence.relationships.manifest, dict(files),
        evidence.parent.manifest, evidence.parent.path, names, clocks, evidence.relationships.manifest["months"])
    context = object.__new__(OriginalRelationshipInputs)
    context._root, context._receipt_pin, context._evidence = root, pin, evidence
    context._outcome, context._parent, context._files = outcome, parent, files
    context._historical, context._sources = dict(saved["historical_implementation_files"]), sources
    context._groups, context._ownership, context._active = groups, ownership, True
    context._history_sessions = tuple(stamp.date() for stamp in xcals.get_calendar("XNYS").sessions_in_range(
        replay.SOURCE_START, replay.SOURCE_END))
    context._maximum_system_used_percent = 90.0
    context.recheck()
    return context


@contextmanager
def _verified_original_relationship_inputs(root: Path, policy: OriginalRelationshipInputPolicy
                                          ) -> Iterator[OriginalRelationshipInputs]:
    """Same full verification under the research caller's already-owned lease."""
    context = _open(root.resolve(), policy)
    try:
        yield context
    finally:
        try:
            context.recheck()
        finally:
            context._active = False


@contextmanager
def verified_original_relationship_inputs(root: Path, policy: OriginalRelationshipInputPolicy
                                         ) -> Iterator[OriginalRelationshipInputs]:
    """Standalone public owner acquires exactly one heavy lease."""
    root = root.resolve()
    runtime = heavy_job_runtime_dir()
    with heavy_job_lease("original-relationship-inputs", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        with _verified_original_relationship_inputs(root, policy) as context:
            yield context
