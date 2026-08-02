"""Fail-closed live construction for edge-rebuild swing model features."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Final, Protocol

import exchange_calendars as xcals
import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.joins import MEMBERSHIP_VALUE_COLUMNS
from market_predictor.canonical.reconciliation import stamp_canonical_decision_ids
from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.catalyst_authority import (
    REQUIRED_MODEL_SOURCE_FAMILIES,
    TRACKED_SOURCE_FAMILIES,
    CatalystDecisionAuthority,
    load_catalyst_decision_authority,
)
from market_predictor.edge_rebuild.serving import (
    canonical_payload_sha256,
    validate_ordered_feature_frame,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.swing_features import (
    SWING_CATALYST_FEATURE_PROFILE,
    SWING_FEATURE_PROFILE,
    build_swing_ablation_rows,
    build_swing_feature_rows,
    finalize_swing_feature_panel,
    swing_model_feature_columns,
)
from market_predictor.resources import assert_memory_budget
from market_predictor.v3.errors import DataReadinessError

SWING_LIVE_SCHEMA_VERSION: Final = "edge_rebuild.swing_live.v1"
SWING_LIVE_INPUT_SCHEMA_VERSION: Final = "edge_rebuild.swing_live_inputs.v2"
SWING_LIVE_INPUT_POINTER_SCHEMA: Final = "edge_rebuild.swing_live_input_pointer.v1"
SWING_LIVE_INPUT_POINTER: Final = "active_generation.json"
SWING_LIVE_INPUT_GENERATIONS: Final = "generations"
SWING_LIVE_REQUIRED_WATERMARKS: Final = (
    "stock_daily_bars_available_at_utc",
    "benchmark_daily_bars_available_at_utc",
    "membership_available_at_utc",
    "alpaca_news_available_at_utc",
)
SWING_LIVE_IDENTITY_COLUMNS: Final = (
    "decision_id",
    "security_id",
    "ticker",
    "session_date_et",
    "decision_time_utc",
)


@dataclass(frozen=True, slots=True)
class SwingLiveFeatureFrames:
    """Two row-identical, model-only frames for the current swing decision."""

    technical_market: pd.DataFrame
    catalyst_full: pd.DataFrame
    context: pd.DataFrame
    as_of_utc: pd.Timestamp
    decision_time_utc: pd.Timestamp
    session_date_et: date
    excluded_security_ids: tuple[str, ...] = ()
    schema_version: str = SWING_LIVE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class SwingLiveInputs:
    stock_daily_bars: pd.DataFrame
    benchmark_daily_bars: pd.DataFrame
    point_in_time_memberships: pd.DataFrame
    catalyst_authority_directory: Path
    catalyst_authority_sha256: str
    manifest_path: Path
    manifest_sha256: str
    generated_at_utc: datetime
    source_watermarks: Mapping[str, str]
    generation_id: str
    pointer_sha256: str


class SwingLiveInputProvider(Protocol):
    def load(
        self,
        *,
        as_of_utc: datetime,
        maximum_bytes: int,
        maximum_rows: int,
    ) -> SwingLiveInputs: ...


class FileSwingLiveInputProvider:
    """Cache one immutable, pointer-selected live input generation."""

    def __init__(
        self,
        directory: Path,
        *,
        memory_budget_gib: float = 4.0,
        memory_headroom_gib: float = 0.5,
    ) -> None:
        self.directory = directory
        self._memory_budget_gib = memory_budget_gib
        self._memory_headroom_gib = memory_headroom_gib
        self._lock = threading.Lock()
        self._cached: SwingLiveInputs | None = None

    def load(
        self,
        *,
        as_of_utc: datetime,
        maximum_bytes: int,
        maximum_rows: int,
    ) -> SwingLiveInputs:
        root = self.directory.resolve()
        pointer = _load_input_pointer(root)
        cached = self._cached
        if cached is not None and cached.pointer_sha256 == pointer["pointer_sha256"]:
            _validate_cached_input_cutoff(cached, as_of_utc)
            return cached
        with self._lock:
            pointer = _load_input_pointer(root)
            cached = self._cached
            if cached is not None and cached.pointer_sha256 == pointer["pointer_sha256"]:
                _validate_cached_input_cutoff(cached, as_of_utc)
                return cached
            loaded = self._load_generation(
                root,
                pointer=pointer,
                as_of_utc=as_of_utc,
                maximum_bytes=maximum_bytes,
                maximum_rows=maximum_rows,
            )
            after = _load_input_pointer(root)
            if after["pointer_sha256"] != pointer["pointer_sha256"]:
                raise DataReadinessError(
                    "active swing live-input generation changed during verification"
                )
            self._cached = loaded
            return loaded

    def _load_generation(
        self,
        root: Path,
        *,
        pointer: Mapping[str, str],
        as_of_utc: datetime,
        maximum_bytes: int,
        maximum_rows: int,
    ) -> SwingLiveInputs:
        assert_memory_budget(
            hard_budget_gib=self._memory_budget_gib,
            headroom_gib=self._memory_headroom_gib,
            stage="before swing live-input generation load",
        )
        generation_root = _verified_input_generation_root(
            root,
            pointer["generation_id"],
        )
        manifest_path = generation_root / "_manifest.json"
        manifest = _read_hashed_json(
            manifest_path,
            expected_sha256=pointer["manifest_file_sha256"],
            label="swing live input manifest",
        )
        if (
            manifest.get("schema") != SWING_LIVE_INPUT_SCHEMA_VERSION
            or manifest.get("state") != "complete"
            or manifest.get("market_data_provider") != "alpaca"
            or manifest.get("market_data_feed") != "sip"
            or manifest.get("market_data_adjustment") != "all"
        ):
            raise DataReadinessError("swing live input manifest contract is invalid")
        generated = _strict_utc_value(
            manifest.get("generated_at_utc"),
            "swing live input generated_at_utc",
        )
        cutoff = _utc_cutoff(as_of_utc)
        if generated > cutoff:
            raise DataReadinessError("swing live inputs were generated after as_of_utc")
        files = manifest.get("files")
        if not isinstance(files, Mapping):
            raise DataReadinessError("swing live input file inventory is invalid")
        expected_names = {
            "stock_daily_bars",
            "benchmark_daily_bars",
            "point_in_time_memberships",
        }
        if set(str(name) for name in files) != expected_names:
            raise DataReadinessError("swing live input file inventory is incomplete")
        records = {name: files[name] for name in sorted(expected_names)}
        catalyst = _verified_relative_directory(
            generation_root,
            manifest.get("catalyst_authority_directory"),
            label="catalyst authority",
        )
        catalyst_authority_path = catalyst / "_authority.json"
        if catalyst_authority_path.is_symlink() or not catalyst_authority_path.is_file():
            raise DataReadinessError("swing live catalyst authority identity is unavailable")
        expected_catalyst_sha256 = str(
            manifest.get("catalyst_authority_sha256", "")
        )
        if (
            len(expected_catalyst_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_catalyst_sha256
            )
            or file_sha256(catalyst_authority_path) != expected_catalyst_sha256
        ):
            raise DataReadinessError(
                "swing live catalyst authority hash does not verify"
            )
        watermarks_raw = manifest.get("source_watermarks")
        if not isinstance(watermarks_raw, Mapping) or set(watermarks_raw) != set(
            SWING_LIVE_REQUIRED_WATERMARKS
        ):
            raise DataReadinessError("swing live source watermarks are missing")
        watermarks: dict[str, str] = {}
        for key in SWING_LIVE_REQUIRED_WATERMARKS:
            value = _strict_utc_value(
                watermarks_raw[key], f"swing live source watermark {key}"
            )
            if value > cutoff:
                raise DataReadinessError("swing live source watermark is after as_of_utc")
            watermarks[key] = value.isoformat()
        if max(_strict_utc_value(value, key) for key, value in watermarks.items()) > generated:
            raise DataReadinessError(
                "swing live generation predates one or more source watermarks"
            )
        projections = {
            "stock_daily_bars": (
                "ticker", "timeframe", "bar_start_utc", "bar_end_utc",
                "available_at_utc", "open", "high", "low", "close",
                "volume", "price_feed", "adjustment", "schema_version",
            ),
            "benchmark_daily_bars": (
                "ticker", "timeframe", "bar_start_utc", "bar_end_utc",
                "available_at_utc", "open", "high", "low", "close",
                "volume", "price_feed", "adjustment", "schema_version",
            ),
            "point_in_time_memberships": (
                "ticker", "effective_from_utc", "effective_to_utc",
                "available_at_utc", *MEMBERSHIP_VALUE_COLUMNS,
            ),
        }
        frames: dict[str, pd.DataFrame] = {}
        total_bytes = 0
        total_rows = 0
        for name in sorted(expected_names):
            frame, size, rows = _read_projected_parquet(
                generation_root,
                records[name],
                columns=projections[name],
                maximum_bytes=maximum_bytes - total_bytes,
                maximum_rows=maximum_rows - total_rows,
                label=name,
                memory_budget_gib=self._memory_budget_gib,
                memory_headroom_gib=self._memory_headroom_gib,
            )
            frames[name] = frame
            total_bytes += size
            total_rows += rows
            assert_memory_budget(
                hard_budget_gib=self._memory_budget_gib,
                headroom_gib=self._memory_headroom_gib,
                stage=f"after projected swing input read: {name}",
            )
        _require_physical_sip_all(frames["stock_daily_bars"], "stock daily bars")
        _require_physical_sip_all(
            frames["benchmark_daily_bars"],
            "benchmark daily bars",
        )
        return SwingLiveInputs(
            stock_daily_bars=frames["stock_daily_bars"],
            benchmark_daily_bars=frames["benchmark_daily_bars"],
            point_in_time_memberships=frames["point_in_time_memberships"],
            catalyst_authority_directory=catalyst,
            catalyst_authority_sha256=expected_catalyst_sha256,
            manifest_path=manifest_path,
            manifest_sha256=pointer["manifest_file_sha256"],
            generated_at_utc=generated.to_pydatetime(),
            source_watermarks=watermarks,
            generation_id=pointer["generation_id"],
            pointer_sha256=pointer["pointer_sha256"],
        )


def build_live_swing_features(
    stock_daily_bars: pd.DataFrame,
    benchmark_daily_bars: pd.DataFrame,
    point_in_time_memberships: pd.DataFrame,
    *,
    contract: StrategyContract,
    catalyst_authority_directory: Path,
    expected_catalyst_authority_sha256: str,
    live_manifest_path: Path,
    expected_live_manifest_sha256: str,
    as_of_utc: object,
    memory_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.5,
) -> SwingLiveFeatureFrames:
    """Build the latest complete same-session swing cross-section.

    The function deliberately accepts an immutable authority directory instead
    of an in-memory catalyst object. This makes hash verification unavoidable
    at the live boundary. All feature math is delegated to the batch builders.
    """

    cutoff = _utc_cutoff(as_of_utc)
    _verify_live_feature_bindings(
        catalyst_authority_directory=catalyst_authority_directory,
        expected_catalyst_authority_sha256=expected_catalyst_authority_sha256,
        live_manifest_path=live_manifest_path,
        expected_live_manifest_sha256=expected_live_manifest_sha256,
    )
    expected = _effective_membership_security_ids(
        point_in_time_memberships,
        decision_time=_expected_swing_decision_time(cutoff),
        contract=contract,
    )
    _reject_future_evidence(
        stock_daily_bars,
        label="stock daily bars",
        timestamp_columns=("bar_start_utc", "bar_end_utc", "available_at_utc"),
        cutoff=cutoff,
    )
    _reject_future_evidence(
        benchmark_daily_bars,
        label="benchmark daily bars",
        timestamp_columns=("bar_start_utc", "bar_end_utc", "available_at_utc"),
        cutoff=cutoff,
    )
    _reject_future_evidence(
        point_in_time_memberships,
        label="point-in-time memberships",
        timestamp_columns=("available_at_utc",),
        cutoff=cutoff,
    )

    technical_history = build_swing_feature_rows(
        stock_daily_bars,
        benchmark_daily_bars,
        point_in_time_memberships,
        contract=contract,
    )
    assert_memory_budget(
        hard_budget_gib=memory_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="after live swing technical feature construction",
    )
    technical_history = stamp_canonical_decision_ids(technical_history)
    current, excluded_security_ids = _select_complete_current_cross_section(
        technical_history,
        expected_security_ids=expected,
        cutoff=cutoff,
        contract=contract,
    )

    authority = load_catalyst_decision_authority(
        catalyst_authority_directory,
        require_production_ready=True,
        expected_authority_sha256=expected_catalyst_authority_sha256,
    )
    _validate_exact_catalyst_authority(
        authority,
        current=current,
        expected_security_ids=expected,
        cutoff=cutoff,
    )
    ablations = build_swing_ablation_rows(current, authority)
    catalyst_ready = ablations[SWING_CATALYST_FEATURE_PROFILE]["catalyst_required_source_complete"].fillna(False).astype(bool)
    if not bool(catalyst_ready.all()):
        catalyst_exclusions = tuple(
            sorted(
                ablations[SWING_CATALYST_FEATURE_PROFILE].loc[
                    ~catalyst_ready,
                    "security_id",
                ].astype(str)
            )
        )
        excluded_security_ids = _validate_live_security_exclusions(
            expected,
            (*excluded_security_ids, *catalyst_exclusions),
            contract=contract,
            reason="market or catalyst evidence",
        )
        ablations = {
            profile: frame.loc[catalyst_ready].reset_index(drop=True)
            for profile, frame in ablations.items()
        }
        current = current.loc[catalyst_ready].reset_index(drop=True)
    surviving_ids = tuple(sorted(current["security_id"].astype(str)))
    finalized = {
        profile: finalize_swing_feature_panel(
            ablations[profile],
            contract=contract,
            expected_security_ids=surviving_ids,
        )
        for profile in (SWING_FEATURE_PROFILE, SWING_CATALYST_FEATURE_PROFILE)
    }
    assert_memory_budget(
        hard_budget_gib=memory_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="after live swing catalyst feature construction",
    )
    _validate_profile_identity(finalized)

    technical = _model_frame(
        finalized[SWING_FEATURE_PROFILE],
        columns=swing_model_feature_columns(contract=contract, catalyst=False),
        profile=SWING_FEATURE_PROFILE,
    )
    catalyst = _model_frame(
        finalized[SWING_CATALYST_FEATURE_PROFILE],
        columns=swing_model_feature_columns(contract=contract, catalyst=True),
        profile=SWING_CATALYST_FEATURE_PROFILE,
    )
    if not technical.index.equals(catalyst.index):
        raise DataReadinessError("technical_market and catalyst_full live row identities differ")
    context = finalized[SWING_CATALYST_FEATURE_PROFILE].copy()
    context.index = pd.MultiIndex.from_frame(
        context.loc[:, SWING_LIVE_IDENTITY_COLUMNS],
        names=SWING_LIVE_IDENTITY_COLUMNS,
    )
    context = context.loc[catalyst.index].copy()

    decision_times = current["decision_time_utc"].drop_duplicates()
    sessions = current["session_date_et"].drop_duplicates()
    result = SwingLiveFeatureFrames(
        technical_market=technical,
        catalyst_full=catalyst,
        context=context,
        as_of_utc=cutoff,
        decision_time_utc=pd.Timestamp(decision_times.iloc[0]),
        session_date_et=sessions.iloc[0],
        excluded_security_ids=excluded_security_ids,
    )
    _verify_live_feature_bindings(
        catalyst_authority_directory=catalyst_authority_directory,
        expected_catalyst_authority_sha256=expected_catalyst_authority_sha256,
        live_manifest_path=live_manifest_path,
        expected_live_manifest_sha256=expected_live_manifest_sha256,
    )
    return result


def _verify_live_feature_bindings(
    *,
    catalyst_authority_directory: Path,
    expected_catalyst_authority_sha256: str,
    live_manifest_path: Path,
    expected_live_manifest_sha256: str,
) -> None:
    files = (
        (
            catalyst_authority_directory / "_authority.json",
            expected_catalyst_authority_sha256,
            "catalyst authority",
        ),
        (live_manifest_path, expected_live_manifest_sha256, "live input manifest"),
    )
    for path, expected, label in files:
        if (
            len(expected) != 64
            or path.is_symlink()
            or not path.is_file()
            or file_sha256(path) != expected
        ):
            raise DataReadinessError(
                f"swing live {label} changed or does not match its generation"
            )


def _select_complete_current_cross_section(
    rows: pd.DataFrame,
    *,
    expected_security_ids: tuple[str, ...],
    cutoff: pd.Timestamp,
    contract: StrategyContract,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    required = {
        *SWING_LIVE_IDENTITY_COLUMNS,
        "daily_bar_count",
        "feature_eligible",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise DataReadinessError(f"swing live history is missing required columns: {missing}")
    decision_time = _strict_utc_series(rows["decision_time_utc"], "decision_time_utc")
    available = rows.loc[decision_time.le(cutoff)].copy()
    if available.empty:
        raise DataReadinessError("no completed swing decision is available at as_of_utc")
    available["decision_time_utc"] = decision_time.loc[available.index]
    latest_decision = available["decision_time_utc"].max()
    current = available.loc[available["decision_time_utc"].eq(latest_decision)].copy()
    if current.empty:
        raise DataReadinessError("latest swing decision cross-section is empty")
    if current["session_date_et"].nunique(dropna=False) != 1:
        raise DataReadinessError("latest swing decision mixes multiple market sessions")
    observed_session = pd.Timestamp(current["session_date_et"].iloc[0])
    expected_session = _latest_closed_session(cutoff)
    if observed_session != expected_session:
        raise DataReadinessError(
            f"latest swing decision is stale; expected closed XNYS session {expected_session.date()}, observed {observed_session.date()}"
        )
    if bool(current.duplicated("security_id").any()):
        raise DataReadinessError("latest swing decision has duplicate security identities")
    observed = tuple(sorted(current["security_id"].astype(str)))
    unexpected_ids = sorted(set(observed).difference(expected_security_ids))
    if unexpected_ids:
        raise DataReadinessError(
            "latest swing decision contains unexpected security identities: "
            f"{unexpected_ids[:10]}"
        )
    missing_ids = tuple(sorted(set(expected_security_ids).difference(observed)))
    warmup = pd.to_numeric(current["daily_bar_count"], errors="coerce")
    cold = warmup.lt(contract.swing.minimum_warmup_sessions) | warmup.isna()
    ineligible = ~current["feature_eligible"].fillna(False).astype(bool)
    rejected_ids = tuple(
        sorted(current.loc[cold | ineligible, "security_id"].astype(str))
    )
    exclusions = _validate_live_security_exclusions(
        expected_security_ids,
        (*missing_ids, *rejected_ids),
        contract=contract,
        reason="same-session warm-up or market features",
    )
    retained = current.loc[~current["security_id"].astype(str).isin(exclusions)]
    if retained.empty:
        raise DataReadinessError("latest swing cross-section has no eligible securities")
    return (
        retained.sort_values("security_id", kind="stable").reset_index(drop=True),
        exclusions,
    )


def _validate_live_security_exclusions(
    expected_security_ids: tuple[str, ...],
    excluded_security_ids: tuple[str, ...],
    *,
    contract: StrategyContract,
    reason: str,
) -> tuple[str, ...]:
    excluded = tuple(sorted(set(excluded_security_ids)))
    unexpected = sorted(set(excluded).difference(expected_security_ids))
    if unexpected:
        raise DataReadinessError(
            f"live swing exclusions contain unexpected identities: {unexpected[:10]}"
        )
    fraction = len(excluded) / len(expected_security_ids)
    if fraction > contract.data_quality.maximum_security_exclusion_fraction:
        raise DataReadinessError(
            "live swing security exclusions exceed the governed ceiling for "
            f"{reason}: {len(excluded)}/{len(expected_security_ids)} ({fraction:.2%})"
        )
    return excluded


def _validate_exact_catalyst_authority(
    authority: CatalystDecisionAuthority,
    *,
    current: pd.DataFrame,
    expected_security_ids: tuple[str, ...],
    cutoff: pd.Timestamp,
) -> None:
    if authority.manifest.get("production_ready") is not True:
        raise DataReadinessError("live swing requires a production-ready catalyst authority")
    authority_completed = _strict_utc_value(
        authority.manifest.get("completed_at_utc"),
        "catalyst authority completed_at_utc",
    )
    if authority_completed > cutoff:
        raise DataReadinessError("catalyst authority was not available at as_of_utc")
    tracked = _manifest_string_sequence(
        authority.manifest.get("tracked_source_families"),
        "tracked_source_families",
    )
    if tracked != tuple(TRACKED_SOURCE_FAMILIES):
        raise DataReadinessError("catalyst authority tracked source-family contract differs")
    required = _manifest_string_sequence(
        authority.manifest.get("required_model_source_families"),
        "required_model_source_families",
    )
    if required != tuple(REQUIRED_MODEL_SOURCE_FAMILIES):
        raise DataReadinessError("catalyst authority required model-source contract differs")

    target_ids = set(current["decision_id"].astype(str))
    target_evidence = authority.decisions.loc[authority.decisions["decision_id"].astype(str).isin(target_ids)]
    if not target_evidence.empty:
        evidence_decisions = _strict_utc_series(
            target_evidence["decision_time_utc"],
            "catalyst decision_time_utc",
        )
        latest_event = _strict_optional_utc_series(
            target_evidence["latest_event_feature_available_at_utc"],
            "latest_event_feature_available_at_utc",
        )
        if bool(evidence_decisions.gt(cutoff).any()):
            raise DataReadinessError("catalyst authority contains decisions after as_of_utc")
        if bool(latest_event.gt(cutoff).fillna(False).any()):
            raise DataReadinessError("catalyst authority contains evidence available after as_of_utc")

    coverage = authority.coverage.loc[authority.coverage["security_id"].astype(str).isin(expected_security_ids)]
    if not coverage.empty:
        requested_end = _strict_utc_series(
            coverage["requested_end_utc"],
            "catalyst requested_end_utc",
        )
        if bool(requested_end.gt(cutoff).any()):
            raise DataReadinessError("catalyst coverage contains evidence available after as_of_utc")
        collection_completed = _strict_utc_series(
            coverage["completed_at_utc"],
            "catalyst coverage completed_at_utc",
        )
        if bool(collection_completed.gt(cutoff).any()):
            raise DataReadinessError("catalyst collection completion is after as_of_utc")


def _latest_closed_session(cutoff: pd.Timestamp) -> pd.Timestamp:
    calendar = xcals.get_calendar("XNYS")
    candidate = pd.Timestamp(calendar.minute_to_session(cutoff, direction="previous"))
    if cutoff < pd.Timestamp(calendar.session_close(candidate)).tz_convert("UTC"):
        candidate = pd.Timestamp(calendar.previous_session(candidate))
    return candidate.tz_localize(None).normalize()


def _validate_profile_identity(finalized: dict[str, pd.DataFrame]) -> None:
    technical = finalized[SWING_FEATURE_PROFILE].loc[:, SWING_LIVE_IDENTITY_COLUMNS].reset_index(drop=True)
    catalyst = finalized[SWING_CATALYST_FEATURE_PROFILE].loc[:, SWING_LIVE_IDENTITY_COLUMNS].reset_index(drop=True)
    if not technical.equals(catalyst):
        raise DataReadinessError("technical_market and catalyst_full live row identities differ")


def _model_frame(
    rows: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    profile: str,
) -> pd.DataFrame:
    eligible = rows["feature_eligible"].fillna(False).astype(bool) & rows["cross_section_eligible"].fillna(False).astype(bool)
    if not bool(eligible.any()):
        raise DataReadinessError(
            f"{profile} has no stocks meeting the sector peer floor"
        )
    eligible_rows = rows.loc[eligible]
    identity = pd.MultiIndex.from_frame(
        eligible_rows.loc[:, SWING_LIVE_IDENTITY_COLUMNS],
        names=SWING_LIVE_IDENTITY_COLUMNS,
    )
    frame = eligible_rows.loc[:, columns].copy()
    frame.index = identity
    validate_ordered_feature_frame(frame, columns, frame_name=profile)
    return frame


def _effective_membership_security_ids(
    memberships: pd.DataFrame,
    *,
    decision_time: pd.Timestamp,
    contract: StrategyContract,
) -> tuple[str, ...]:
    required = {
        "ticker",
        "security_id",
        "effective_from_utc",
        "effective_to_utc",
        "available_at_utc",
    }
    missing = sorted(required.difference(memberships.columns))
    if missing:
        raise DataReadinessError(
            f"point-in-time membership authority is missing columns: {missing}"
        )
    data = memberships.loc[:, sorted(required)].copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["security_id"] = data["security_id"].astype(str).str.strip()
    effective_from = _strict_utc_series(
        data["effective_from_utc"],
        "membership effective_from_utc",
    )
    available = _strict_utc_series(
        data["available_at_utc"],
        "membership available_at_utc",
    )
    effective_to = _strict_optional_utc_series(
        data["effective_to_utc"],
        "membership effective_to_utc",
    )
    active = effective_from.le(decision_time) & available.le(decision_time)
    active &= effective_to.isna() | effective_to.gt(decision_time)
    current = data.loc[active].copy()
    if current.empty or bool(
        current[["ticker", "security_id"]].eq("").any(axis=None)
    ):
        raise DataReadinessError("effective point-in-time membership is empty or invalid")
    if bool(current.duplicated("ticker").any()) or bool(
        current.duplicated("security_id").any()
    ):
        raise DataReadinessError(
            "effective point-in-time membership has ambiguous ticker/security identity"
        )
    normalized = tuple(sorted(current["security_id"].astype(str)))
    minimum = contract.labels.minimum_cross_section_for_ranking
    if len(normalized) < minimum:
        raise DataReadinessError(f"expected swing cross-section is below the frozen minimum: {len(normalized)} < {minimum}")
    return normalized


def _expected_swing_decision_time(cutoff: pd.Timestamp) -> pd.Timestamp:
    session = _latest_closed_session(cutoff)
    decision = swing_prediction_cutoffs(pd.Series([session.date()])).iloc[0]
    return pd.Timestamp(decision).tz_convert("UTC")


def _utc_cutoff(value: object) -> pd.Timestamp:
    try:
        cutoff = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("as_of_utc must be a valid timezone-aware timestamp") from exc
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise DataReadinessError("as_of_utc must be a valid timezone-aware timestamp")
    return cutoff.tz_convert("UTC")


def _reject_future_evidence(
    frame: pd.DataFrame,
    *,
    label: str,
    timestamp_columns: tuple[str, ...],
    cutoff: pd.Timestamp,
) -> None:
    missing = sorted(set(timestamp_columns).difference(frame.columns))
    if missing:
        raise DataReadinessError(f"{label} are missing cutoff columns: {missing}")
    if frame.empty:
        raise DataReadinessError(f"{label} cannot be empty")
    for column in timestamp_columns:
        timestamps = _strict_utc_series(frame[column], f"{label} {column}")
        if bool(timestamps.gt(cutoff).any()):
            raise DataReadinessError(f"{label} contain {column} evidence available after as_of_utc")


def _strict_utc_series(values: pd.Series, label: str) -> pd.Series:
    if not _series_is_timezone_aware(values):
        raise DataReadinessError(f"{label} must be timezone-aware")
    try:
        parsed = pd.to_datetime(values, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataReadinessError(f"{label} contains an invalid timestamp") from exc
    if bool(parsed.isna().any()):
        raise DataReadinessError(f"{label} contains a missing timestamp")
    return parsed


def _strict_optional_utc_series(values: pd.Series, label: str) -> pd.Series:
    present = values.loc[values.notna()]
    if not present.empty and not _series_is_timezone_aware(present):
        raise DataReadinessError(f"{label} must be timezone-aware when present")
    try:
        return pd.to_datetime(values, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataReadinessError(f"{label} contains an invalid timestamp") from exc


def _strict_utc_value(value: object, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError(f"{label} is invalid") from exc
    if pd.isna(parsed) or parsed.tzinfo is None:
        raise DataReadinessError(f"{label} must be timezone-aware")
    return parsed.tz_convert("UTC")


def _series_is_timezone_aware(values: pd.Series) -> bool:
    dtype = values.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return True
    if pd.api.types.is_datetime64_dtype(dtype):
        return False
    for value in values.loc[values.notna()]:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return False
        if timestamp.tzinfo is None:
            return False
    return True


def _manifest_string_sequence(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise DataReadinessError(f"catalyst authority {label} is malformed")
    return tuple(value)


def _load_input_pointer(root: Path) -> dict[str, str]:
    pointer_path = root / SWING_LIVE_INPUT_POINTER
    payload = _read_json(pointer_path, "swing live-input generation pointer")
    expected = {
        "schema",
        "generation_id",
        "manifest_file_sha256",
        "previous_generation_id",
        "activated_at_utc",
        "pointer_sha256",
    }
    if set(payload) != expected or payload.get("schema") != SWING_LIVE_INPUT_POINTER_SCHEMA:
        raise DataReadinessError("swing live-input generation pointer schema is invalid")
    unsigned = dict(payload)
    pointer_sha = str(unsigned.pop("pointer_sha256", ""))
    if canonical_payload_sha256(unsigned) != pointer_sha:
        raise DataReadinessError("swing live-input generation pointer hash is invalid")
    for field in ("generation_id", "manifest_file_sha256", "pointer_sha256"):
        value = str(payload.get(field, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise DataReadinessError(
                f"swing live-input generation pointer {field} is invalid"
            )
    if payload.get("generation_id") != payload.get("manifest_file_sha256"):
        raise DataReadinessError(
            "swing live-input generation identity must equal its manifest hash"
        )
    _strict_utc_value(payload.get("activated_at_utc"), "live-input activation")
    previous = payload.get("previous_generation_id")
    if previous is not None and (
        not isinstance(previous, str)
        or len(previous) != 64
        or any(character not in "0123456789abcdef" for character in previous)
    ):
        raise DataReadinessError("swing live-input previous generation is invalid")
    return {
        str(key): str(value) if value is not None else ""
        for key, value in payload.items()
    }


def _verified_input_generation_root(root: Path, generation_id: str) -> Path:
    generations = root / SWING_LIVE_INPUT_GENERATIONS
    candidate = generations / generation_id
    if generations.is_symlink() or candidate.is_symlink():
        raise DataReadinessError("swing live-input generation cannot use symlinks")
    try:
        generations_root = generations.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DataReadinessError("swing live-input generation is unavailable") from exc
    if not resolved.is_dir() or not resolved.is_relative_to(generations_root):
        raise DataReadinessError("swing live-input generation escapes its repository")
    return resolved


def _read_projected_parquet(
    root: Path,
    record: object,
    *,
    columns: Sequence[str],
    maximum_bytes: int,
    maximum_rows: int,
    label: str,
    memory_budget_gib: float,
    memory_headroom_gib: float,
) -> tuple[pd.DataFrame, int, int]:
    if not isinstance(record, Mapping):
        raise DataReadinessError(f"swing live {label} file record is invalid")
    if maximum_bytes < 1 or maximum_rows < 1:
        raise DataReadinessError("swing live aggregate input limit exceeded")
    path = _verified_relative_path(root, record.get("path"), label=label)
    expected_sha = str(record.get("sha256", ""))
    expected_rows = record.get("rows")
    if len(expected_sha) != 64 or not isinstance(expected_rows, int) or expected_rows < 1:
        raise DataReadinessError(f"swing live {label} identity is invalid")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size < 1 or before.st_size > maximum_bytes:
                raise DataReadinessError(f"swing live {label} byte limit exceeded")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha:
                raise DataReadinessError(f"swing live {label} artifact hash does not verify")
            handle.seek(0)
            parquet = pq.ParquetFile(handle)  # type: ignore[no-untyped-call]
            rows = int(parquet.metadata.num_rows)
            if rows != expected_rows or rows > maximum_rows:
                raise DataReadinessError(f"swing live {label} row limit or identity failed")
            schema_columns = tuple(parquet.schema_arrow.names)
            missing = sorted(set(columns).difference(schema_columns))
            if missing:
                raise DataReadinessError(
                    f"swing live {label} is missing projected columns: {missing}"
                )
            projected_bytes = _projected_parquet_bytes(parquet, columns)
            if projected_bytes > maximum_bytes:
                raise DataReadinessError(
                    f"swing live {label} projected byte limit exceeded"
                )
            assert_memory_budget(
                hard_budget_gib=memory_budget_gib,
                headroom_gib=memory_headroom_gib,
                stage=f"before projected swing parquet read: {label}",
            )
            table = parquet.read(  # type: ignore[no-untyped-call]
                columns=list(columns),
                use_threads=False,
            )
            assert_memory_budget(
                hard_budget_gib=memory_budget_gib,
                headroom_gib=memory_headroom_gib,
                stage=f"after projected swing parquet read: {label}",
            )
            after = os.fstat(handle.fileno())
            if (
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise DataReadinessError(f"swing live {label} changed during read")
    except (OSError, ValueError) as exc:
        if isinstance(exc, DataReadinessError):
            raise
        raise DataReadinessError(f"swing live {label} is unreadable") from exc
    frame = table.to_pandas()
    assert_memory_budget(
        hard_budget_gib=memory_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage=f"after projected swing pandas conversion: {label}",
    )
    if len(frame) != rows or tuple(frame.columns) != tuple(columns):
        raise DataReadinessError(f"swing live {label} projected read changed")
    return frame, int(before.st_size), rows


def _projected_parquet_bytes(
    parquet: pq.ParquetFile,
    columns: Sequence[str],
) -> int:
    selected = set(columns)
    total = 0
    metadata = parquet.metadata
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            if column.path_in_schema in selected:
                total += int(column.total_uncompressed_size)
    return total


def _require_physical_sip_all(frame: pd.DataFrame, label: str) -> None:
    feeds = frame["price_feed"].astype(str).str.lower().str.strip()
    adjustments = frame["adjustment"].astype(str).str.lower().str.strip()
    if not bool(feeds.eq("sip").all()) or not bool(adjustments.eq("all").all()):
        raise DataReadinessError(f"{label} require physical Alpaca SIP/all rows")


def _validate_cached_input_cutoff(inputs: SwingLiveInputs, as_of_utc: datetime) -> None:
    cutoff = _utc_cutoff(as_of_utc)
    generated = _strict_utc_value(inputs.generated_at_utc, "cached live-input generation")
    if generated > cutoff:
        raise DataReadinessError("cached swing live inputs were generated after as_of_utc")
    for key, value in inputs.source_watermarks.items():
        if _strict_utc_value(value, f"cached source watermark {key}") > cutoff:
            raise DataReadinessError("cached swing source watermark is after as_of_utc")


def _read_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise DataReadinessError(f"{label} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _read_hashed_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, object]:
    if path.is_symlink():
        raise DataReadinessError(f"{label} cannot be a symlink")
    try:
        with path.open("rb") as handle:
            payload = handle.read(1024 * 1024 + 1)
    except OSError as exc:
        raise DataReadinessError(f"{label} is unreadable") from exc
    if not payload or len(payload) > 1024 * 1024:
        raise DataReadinessError(f"{label} byte limit is invalid")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise DataReadinessError(f"{label} hash does not verify")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _verified_manifest_file(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise DataReadinessError(f"swing live {label} file record is invalid")
    path = _verified_relative_path(root, value.get("path"), label=label)
    expected = str(value.get("sha256", ""))
    if len(expected) != 64 or file_sha256(path) != expected:
        raise DataReadinessError(f"swing live {label} artifact hash does not verify")
    return path


def _verified_relative_directory(root: Path, value: object, *, label: str) -> Path:
    path = _verified_relative_path(root, value, label=label, require_file=False)
    if not path.is_dir():
        raise DataReadinessError(f"swing live {label} directory is unavailable")
    return path


def _verified_relative_path(
    root: Path,
    value: object,
    *,
    label: str,
    require_file: bool = True,
) -> Path:
    raw = str(value or "")
    relative = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or relative.is_absolute()
        or relative.as_posix() != raw
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise DataReadinessError(f"swing live {label} path is invalid")
    path = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise DataReadinessError(f"swing live {label} path contains a symlink")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DataReadinessError(f"swing live {label} artifact is unavailable") from exc
    if not resolved.is_relative_to(root) or (require_file and not resolved.is_file()):
        raise DataReadinessError(f"swing live {label} path escapes its authority")
    return resolved
