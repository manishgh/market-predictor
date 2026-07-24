from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from market_predictor.intraday.contracts import (
    downside_target_column,
    excess_return_column,
    net_return_column,
    opportunity_target_column,
)
from market_predictor.swing.contracts import (
    swing_excess_column,
    swing_net_return_column,
    swing_target_column,
)

LABEL_RECONCILIATION_SCHEMA = "label_source_reconciliation.v1"
LABEL_IDENTITY_COLUMNS = ("ticker", "decision_time_utc")
_HASH_CHUNK_ROWS = 20_000


def swing_label_material_columns(horizon: int) -> tuple[str, ...]:
    return (
        "entry_time_utc",
        "exit_time_utc",
        "label_available_at_utc",
        "entry_price",
        "exit_price",
        "entry_session_date_et",
        "exit_session_date_et",
        "label_window_expected",
        "label_path_exact",
        "label_eligible",
        f"future_gross_return_{horizon}d",
        swing_net_return_column(horizon),
        f"future_spy_return_{horizon}d",
        f"future_qqq_return_{horizon}d",
        f"future_sector_return_{horizon}d",
        swing_excess_column(horizon, "SPY"),
        swing_excess_column(horizon, "QQQ"),
        swing_excess_column(horizon, "SECTOR"),
        f"future_mfe_{horizon}d",
        f"future_mae_{horizon}d",
        swing_target_column(horizon),
        "target_excess_rank",
    )


def intraday_label_material_columns(horizon: int) -> tuple[str, ...]:
    return (
        "entry_time_utc",
        "exit_time_utc",
        "label_available_at_utc",
        "label_window_end_utc",
        "entry_price",
        "target_price",
        "stop_price",
        "path_outcome",
        "path_outcome_bar",
        "label_window_expected",
        "label_path_exact",
        "label_eligible",
        opportunity_target_column(horizon),
        downside_target_column(horizon),
        f"path_timeout_{horizon}m",
        f"path_realized_return_gross_{horizon}m",
        net_return_column(horizon),
        f"path_mfe_{horizon}m",
        f"path_mae_{horizon}m",
        f"path_spy_return_{horizon}m",
        f"path_qqq_return_{horizon}m",
        f"path_sector_return_{horizon}m",
        excess_return_column(horizon, "SPY"),
        excess_return_column(horizon, "QQQ"),
        excess_return_column(horizon, "SECTOR"),
    )


def stamp_label_reconciliation(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str],
    material_columns: Sequence[str],
    label_policy_sha256: str,
) -> pd.DataFrame:
    output = frame.copy()
    material_sha = label_material_sha256(
        output,
        identity_columns=identity_columns,
        material_columns=material_columns,
    )
    payload = {
        "schema": LABEL_RECONCILIATION_SCHEMA,
        "label_policy_sha256": label_policy_sha256,
        "label_material_sha256": material_sha,
    }
    reconciliation_sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    output["label_material_sha256"] = material_sha
    output["label_source_reconciliation_sha256"] = reconciliation_sha
    output["label_source_reconciliation_errors"] = 0
    return output


def label_material_sha256(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str],
    material_columns: Sequence[str],
) -> str:
    columns = [*identity_columns, *material_columns]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        return ""
    row_hashes = np.empty(len(frame), dtype="S32")
    for start in range(0, len(frame), _HASH_CHUNK_ROWS):
        stop = min(start + _HASH_CHUNK_ROWS, len(frame))
        chunk = _normalized_frame(frame.iloc[start:stop].loc[:, columns], columns)
        for offset, row in enumerate(chunk.itertuples(index=False, name=None)):
            encoded = json.dumps(
                row,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
            row_hashes[start + offset] = hashlib.sha256(encoded).digest()
    row_hashes.sort()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": columns,
                "row_count": len(frame),
                "schema": LABEL_RECONCILIATION_SCHEMA,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def replay_mismatch_count(
    actual: pd.DataFrame,
    reproduced: pd.DataFrame,
    *,
    identity_columns: Sequence[str],
    material_columns: Sequence[str],
) -> int:
    missing_actual = sorted(set((*identity_columns, *material_columns)).difference(actual.columns))
    missing_reproduced = sorted(set((*identity_columns, *material_columns)).difference(reproduced.columns))
    if missing_actual or missing_reproduced:
        return max(len(actual), len(reproduced), 1)
    if len(actual) != len(reproduced):
        return abs(len(actual) - len(reproduced)) + min(
            len(actual),
            len(reproduced),
        )
    left_order = _identity_sort_order(actual, identity_columns)
    right_order = _identity_sort_order(reproduced, identity_columns)
    mismatches = 0
    for column in (*identity_columns, *material_columns):
        left = actual[column].iloc[left_order].reset_index(drop=True)
        right = reproduced[column].iloc[right_order].reset_index(drop=True)
        if column.endswith("_utc"):
            left_time = pd.to_datetime(left, utc=True, errors="coerce")
            right_time = pd.to_datetime(right, utc=True, errors="coerce")
            mismatches += int((~((left_time == right_time) | (left_time.isna() & right_time.isna()))).sum())
            continue
        left_numeric = pd.to_numeric(left, errors="coerce")
        right_numeric = pd.to_numeric(right, errors="coerce")
        numeric_rows = left_numeric.notna() | right_numeric.notna()
        if bool(numeric_rows.any()):
            mismatches += int(
                (
                    ~np.isclose(
                        left_numeric[numeric_rows].to_numpy(float),
                        right_numeric[numeric_rows].to_numpy(float),
                        rtol=1e-10,
                        atol=1e-12,
                        equal_nan=True,
                    )
                ).sum()
            )
            string_rows = ~numeric_rows
        else:
            string_rows = pd.Series(True, index=left.index)
        if bool(string_rows.any()):
            mismatches += int(
                left.loc[string_rows].astype("string").fillna("").ne(right.loc[string_rows].astype("string").fillna("")).sum()
            )
    return mismatches


def _identity_sort_order(
    frame: pd.DataFrame,
    identity_columns: Sequence[str],
) -> np.ndarray:
    keys: list[np.ndarray] = []
    for column in identity_columns:
        if column.endswith("_utc"):
            values = pd.to_datetime(
                frame[column],
                utc=True,
                errors="coerce",
            )
            keys.append(values.astype("int64").to_numpy())
        else:
            keys.append(frame[column].astype("string").fillna("").to_numpy(str))
    return np.lexsort(tuple(reversed(keys)))


def _normalized_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        if column.endswith("_utc"):
            values = pd.to_datetime(
                normalized[column],
                utc=True,
                errors="coerce",
            )
            normalized[column] = values.map(lambda value: "" if pd.isna(value) else value.isoformat())
        elif pd.api.types.is_numeric_dtype(normalized[column]):
            values = pd.to_numeric(normalized[column], errors="coerce")
            normalized[column] = values.map(lambda value: "" if pd.isna(value) else format(float(value), ".17g"))
        else:
            normalized[column] = normalized[column].fillna("").astype(str)
    return normalized


def stamped_material_hash_is_valid(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str],
    material_columns: Sequence[str],
) -> bool:
    if "label_material_sha256" not in frame or frame.empty:
        return False
    values = frame["label_material_sha256"].fillna("").astype(str).unique()
    return (
        len(values) == 1
        and len(values[0]) == 64
        and values[0]
        == label_material_sha256(
            frame,
            identity_columns=identity_columns,
            material_columns=material_columns,
        )
    )
