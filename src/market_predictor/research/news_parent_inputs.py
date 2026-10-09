"""Load exact previously verified monthly predictors without numerical rebuilding.

The caller owns the shared heavy-job lease. Completed numerical replay is bound
as immutable evidence, not represented as work performed by this adapter.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.resources import assert_memory_budget
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.relationship_historical_evidence import (
    HistoricalFeatureEvidence,
    historical_month,
    inspect_historical_publication,
)
from market_predictor.swing.datasets.return_relationship_integrity import check_files, pins, read_object

PUBLICATION = SourcePin(path="data/features/swing_return_relationships_initial_fit_utc_clocks/_manifest.json",
                        sha256="e9ec4b628a59f7925fb66a0800194f1a712d617a819d13927860d0b9ab0ef04c")
SAVED_ROW_RECEIPT = SourcePin(path="data/reports/swing_return_relationships_initial_fit_utc_verification.json",
                              sha256="e1067950ace121d13e4b3227d7488540e03e558d205ee07668696801708d1086")
ORIGINAL_REPLAY = SourcePin(path="data/reports/swing_original_relationship_replay.json",
                            sha256="aa8000577dd9628d5edfc5557758238394ebf36aaa8a26f8fae0177fc7d9cec9")
EXPECTED_ROWS = 586305
EXPECTED_IDS = "7472fc73a5fc06ce065050f71edf71015d666d6ecb8162543461855c74ecf39f"
MONTHS = tuple(f"{year}-{month:02}" for year in range(2019, 2025) for month in range(1, 13)
               if "2019-07" <= f"{year}-{month:02}" <= "2024-05")
IDENTITY_COLUMNS = ("decision_id", "security_id", "ticker", "decision_time_utc", "session_date_et")
IMPLEMENTATION_PATHS = (
    "research/news_parent_inputs.py", "swing/datasets/relationship_historical_evidence.py",
    "swing/datasets/return_relationship_integrity.py", "canonical/store.py", "evidence/hashing.py", "evidence/io.py",
    "core/errors.py", "core/json_integrity.py", "core/system_memory.py", "resources.py", "process_memory.py",
    "swing/contracts/holding_materialization.py", "swing/contracts/holding_accounting.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _guard() -> None:
    assert_memory_budget(stage="saved news parent", hard_budget_gib=5.0, headroom_gib=0.75)
    assert_system_memory_available(minimum_available_gib=2.0, maximum_used_percent=85.0)


def _implementation(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {(package / name).relative_to(root).as_posix(): file_sha256(package / name) for name in IMPLEMENTATION_PATHS}


def _clock(values: pd.Series, label: str) -> pd.Series:
    _require(isinstance(values.dtype, pd.DatetimeTZDtype), f"saved parent {label} must retain aware timestamp dtype")
    return values.dt.tz_convert("UTC")


def _identities(frame: pd.DataFrame, month: str) -> None:
    _require(frame.columns.is_unique and set(IDENTITY_COLUMNS).issubset(frame.columns), "saved parent identity columns differ")
    for name in ("decision_id", "security_id", "ticker"):
        _require(frame[name].map(lambda value: isinstance(value, str) and bool(value) and value.strip() == value).all(),
                 f"saved parent {name} is invalid")
    _require(not frame.decision_id.duplicated().any(), "saved parent repeats monthly decisions")
    decisions = _clock(frame.decision_time_utc, "decision_time_utc")
    _require(not decisions.isna().any() and decisions.between("2019-07-09T00:00:00Z", "2024-05-28T22:00:00Z").all(),
             "saved parent decision escaped initial fit")
    _require(frame.session_date_et.astype(str).str[:7].eq(month).all() and
             decisions.dt.tz_convert("America/New_York").dt.strftime("%Y-%m").eq(month).all(),
             "saved parent decision month differs")


def _report(root: Path, evidence: HistoricalFeatureEvidence) -> tuple[dict[str, Any], dict[str, str]]:
    saved = read_object(inside(root, ORIGINAL_REPLAY.path), ORIGINAL_REPLAY.sha256)
    _require(saved.get("schema") == "market_predictor.original_relationship_replay" and
             saved.get("status") == "passed_original_snapshot_replay" and
             saved.get("rows") == EXPECTED_ROWS and saved.get("decision_ids_sha256") == EXPECTED_IDS and
             saved.get("differences") == [] and saved.get("additions_source_replayed") is True and
             saved.get("inherited_predictors_equal") is True and saved.get("usable_ohlcv_validated") is True,
             "closed original parent replay scope differs")
    for name in ("training_eligible", "promotion_eligible", "serving_eligible", "consumer_admission_changed",
                 "provider_transport_admitted", "historical_first_seen_proven", "quarantined_source_inputs_admitted",
                 "saved_rows_rewritten", "targets_read_or_replayed", "baseline_price_features_replayed",
                 "raw_news_aggregates_replayed", "current_archive_equivalence_proven"):
        _require(saved.get(name) is False, f"closed original parent replay flag differs: {name}")
    _require(saved.get("source_window_bounds") == ["2018-05-29", "2024-05-28"] and
             saved.get("historical_implementation_disposition") == "immutable_provenance_not_executed",
             "closed original parent source scope differs")
    config_pin = SourcePin.model_validate(saved["config"])
    config = read_object(inside(root, config_pin.path), config_pin.sha256)
    _require(config.get("historical_relationship_publication") == PUBLICATION.model_dump(mode="json") and
             config.get("historical_relationship_receipt") == SAVED_ROW_RECEIPT.model_dump(mode="json") and
             config.get("historical_parent_publication") == evidence.request["parent_publication"] and
             config.get("historical_parent_receipt") == evidence.request["parent_saved_row_verification"],
             "closed original replay config belongs to a different parent")
    _require(saved.get("original_sources") == evidence.request.get("sources") and
             saved.get("source_basis") == evidence.request.get("source_basis") and
             saved.get("original_stock_inventory_sha256") == json_sha256(evidence.request["stock_inventory"]),
             "closed original replay source correspondence differs")
    _require(tuple(sorted(saved["months"])) == MONTHS, "closed original replay month scope differs")
    for month, record in evidence.manifest["months"].items():
        checked = saved["months"][month]
        _require(checked.get("rows") == record["rows"] and checked.get("decision_ids_sha256") == record["decision_ids_sha256"]
                 and checked.get("inherited_predictors_equal") is True, "closed parent monthly ownership differs")
    declared = pins(root, saved["source_files"])
    for name, digest in evidence.source_files.items():
        _require(declared.get(name) == digest, "closed original replay omits exact parent source binding")
    _require(declared.get(config_pin.path) == config_pin.sha256, "closed replay omits config pin")
    # Raw source and producer inventories remain in the pinned receipt. This
    # consumer reads the proved monthly bytes only, not those old raw inputs.
    files = pins(root, evidence.source_files, {ORIGINAL_REPLAY.path: ORIGINAL_REPLAY.sha256,
                                             config_pin.path: config_pin.sha256})
    return saved, files


def _ownership(evidence: HistoricalFeatureEvidence) -> None:
    """Bounded metadata projection only; never reconstruct or concatenate features."""
    with TemporaryDirectory(prefix="news-parent-ownership-") as temporary:
        with closing(sqlite3.connect(Path(temporary) / "ownership.sqlite")) as db:
            db.execute("PRAGMA cache_size=-8192")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA mmap_size=0")
            db.execute("CREATE TABLE decisions(id TEXT PRIMARY KEY)")
            rows = 0
            for month in MONTHS:
                _guard()
                frame = historical_month(evidence, month, list(IDENTITY_COLUMNS))
                _identities(frame, month)
                try:
                    db.executemany("INSERT INTO decisions VALUES (?)", ((value,) for value in frame.decision_id))
                except sqlite3.IntegrityError as exc:
                    raise DataReadinessError("saved parent repeats global decision ownership") from exc
                rows += len(frame)
                db.commit()
                del frame
            digest = hashlib.sha256(b"[")
            for position, (identifier,) in enumerate(db.execute("SELECT id FROM decisions ORDER BY id")):
                if position % 4096 == 0:
                    _guard()
                digest.update(("," if position else "").encode())
                digest.update(json.dumps(identifier).encode())
            digest.update(b"]")
            _require(rows == EXPECTED_ROWS and digest.hexdigest() == EXPECTED_IDS, "saved parent global population differs")


class NewsParentInputs:
    """Context-bound saved parent reader. Returned frames are never modified."""

    def __init__(self, root: Path, evidence: HistoricalFeatureEvidence, report: dict[str, Any],
                 source_files: dict[str, str], implementation_files: dict[str, str]) -> None:
        self._root, self._evidence, self._report = root, evidence, report
        self._source_files, self._implementation_files = source_files, implementation_files
        self._active = True

    @property
    def request(self) -> dict[str, Any]:
        return copy.deepcopy(self._evidence.request)

    @property
    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._evidence.manifest)

    @property
    def model_columns(self) -> tuple[str, ...]:
        return tuple(self._evidence.request["model_columns"])

    @property
    def availability_columns(self) -> Mapping[str, str]:
        return dict(self._evidence.request["availability_columns"])

    @property
    def source_files(self) -> Mapping[str, str]:
        return dict(self._source_files)

    @property
    def implementation_files(self) -> Mapping[str, str]:
        return dict(self._implementation_files)

    @property
    def historical_implementation_files(self) -> Mapping[str, Any]:
        return copy.deepcopy({"original_replay": self._report["current_implementation_files"],
                              "original_producers": self._report["historical_implementation_files"]})

    def recheck(self) -> None:
        _require(self._active, "saved parent context is closed")
        _guard()
        check_files(self._root, self._source_files)
        check_files(self._root, self._implementation_files)

    def read_month(self, month: str) -> pd.DataFrame:
        _require(self._active and month in MONTHS, "saved parent month unavailable or context closed")
        _guard()
        frame = historical_month(self._evidence, month)
        _identities(frame, month)
        columns = self.model_columns
        _require(tuple(name for name in frame.columns if name in columns) == columns,
                 "saved parent physical model order differs")
        _require(frame.feature_profile.eq("technical_relationships").all(), "saved parent feature profile differs")
        decisions = _clock(frame.decision_time_utc, "decision_time_utc")
        for name in columns:
            _require(pd.api.types.is_float_dtype(frame[name].dtype), f"saved parent model dtype differs: {name}")
            numeric = frame[name].to_numpy(dtype="float64", na_value=np.nan)
            _require(not np.isinf(numeric).any(), f"saved parent model contains infinity: {name}")
            clock_name = self._evidence.request["availability_columns"][name]
            _require(clock_name in frame, "saved parent feature clock column missing")
            clock = _clock(frame[clock_name], clock_name)
            _require(not (frame[name].notna() & clock.isna()).any() and not clock.gt(decisions).any(),
                     f"saved parent feature clock unavailable or future: {name}")
        _guard()
        return frame

    def iter_months(self) -> Iterator[tuple[str, pd.DataFrame]]:
        for month in MONTHS:
            yield month, self.read_month(month)
        self.recheck()


@contextmanager
def verified_news_parent_inputs(*, root: Path) -> Iterator[NewsParentInputs]:
    """Caller owns the lease throughout entry, monthly reads and final checks."""
    root = root.resolve()
    _guard()
    implementation = _implementation(root)
    evidence = inspect_historical_publication(root, PUBLICATION, SAVED_ROW_RECEIPT, profile="technical_relationships")
    _require(tuple(sorted(evidence.manifest["months"])) == MONTHS and evidence.manifest["rows"] == EXPECTED_ROWS,
             "saved news parent population differs")
    names = tuple(evidence.request["model_columns"])
    clocks = evidence.request["availability_columns"]
    _require(len(names) == len(set(names)) == 124 and set(names).issubset(clocks), "saved news parent feature schema differs")
    for month in MONTHS:
        child = evidence.manifest["months"][month]["profiles"]["technical_relationships"]
        _require(tuple(child["model_columns"]) == names and child["availability_columns"] == clocks,
                 "saved parent monthly model schema differs")
    report, files = _report(root, evidence)
    _ownership(evidence)
    context = NewsParentInputs(root, evidence, report, files, implementation)
    try:
        context.recheck()
        yield context
        context.recheck()
    finally:
        context._active = False
