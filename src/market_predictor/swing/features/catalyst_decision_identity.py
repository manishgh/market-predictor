"""Pinned canonical decision identities for rename-spanning catalyst evidence."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.reconciliation import stamp_canonical_decision_ids
from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError

DECISION_KEYS = ("decision_id", "security_id", "ticker", "decision_time_utc")
DECISION_COLUMNS = (*DECISION_KEYS, "timeframe", "bar_start_utc", "prediction_cutoff_policy_id")
MAXIMUM_DECISION_ROWS = 250_000


def verify_decision_identity_pin(record: object) -> Path | None:
    if record is None:
        return None
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "manifest_sha256"}:
        raise DataReadinessError("canonical decision identity requires path and two exact pins")
    if not all(isinstance(record[name], str) and re.fullmatch(r"[0-9a-f]{64}", record[name])
            for name in ("sha256", "manifest_sha256")) or not isinstance(record["path"], str):
        raise DataReadinessError("canonical decision identity pins are malformed")
    path = Path(record["path"])
    if not path.is_absolute() or not path.is_file():
        raise DataReadinessError("canonical decision identity requires an existing absolute path")
    if file_sha256(path) != record["sha256"] or file_sha256(manifest_path_for(path)) != record["manifest_sha256"]:
        raise DataReadinessError("canonical decision identity pin differs")
    return path


def load_decision_identity(record: object, *, production_ready: bool) -> pd.DataFrame | None:
    path = verify_decision_identity_pin(record)
    if path is None:
        return None
    metadata = pq.ParquetFile(path).metadata  # type: ignore[no-untyped-call]
    if metadata.num_rows > MAXIMUM_DECISION_ROWS:
        raise DataReadinessError("canonical decision identity exceeds bounded partition size")
    frame, _ = load_canonical_artifact(path, expected_type="decisions", allow_research=not production_ready,
        columns=DECISION_COLUMNS)
    stamped = stamp_canonical_decision_ids(frame)
    if (frame.empty or frame.decision_id.duplicated().any()
            or not frame.decision_id.reset_index(drop=True).equals(stamped.decision_id.reset_index(drop=True))):
        raise DataReadinessError("canonical decision identity keys are not canonical and unique")
    verify_decision_identity_pin(record)
    return stamped.loc[:, DECISION_KEYS]


def verify_decision_keys(rows: pd.DataFrame, decisions: pd.DataFrame) -> None:
    if rows.empty:
        return
    joined = rows.loc[:, DECISION_KEYS].merge(decisions.loc[:, DECISION_KEYS], on=list(DECISION_KEYS),
        how="left", validate="many_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise DataReadinessError("catalyst assignment differs from pinned canonical decision identity")
