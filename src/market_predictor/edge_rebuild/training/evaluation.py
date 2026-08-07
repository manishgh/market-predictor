from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.v3.errors import DataReadinessError


def _iso(value: object) -> str:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize('UTC')
    return cast(str, parsed.tz_convert('UTC').isoformat())


def _security_holdout_mask(
    data: pd.DataFrame,
    strategy_contract: StrategyContract,
) -> pd.Series:
    fraction = strategy_contract.validation.unseen_ticker_holdout_fraction
    threshold = int(fraction * 2**64)
    identities = data["security_id"].astype(str)
    assigned = identities.map(
        lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)
        < threshold
    )
    if not assigned.any() or assigned.all():
        raise DataReadinessError("stable security holdout produced an empty partition")
    return assigned.astype(bool)


def _calibration_bins(
    target: np.ndarray, probability: np.ndarray
) -> list[dict[str, float | int]]:
    edges = np.linspace(0.0, 1.0, 11)
    indices = np.clip(np.searchsorted(edges, probability, side="right") - 1, 0, 9)
    records: list[dict[str, float | int]] = []
    for index in range(10):
        selected = indices == index
        if not selected.any():
            continue
        records.append(
            {
                "bin": index,
                "rows": int(selected.sum()),
                "mean_probability": float(probability[selected].mean()),
                "observed_rate": float(target[selected].mean()),
            }
        )
    return records



def _expected_calibration_error(
    records: Sequence[Mapping[str, float | int]], rows: int
) -> float:
    return float(
        sum(
            int(record["rows"])
            / rows
            * abs(float(record["mean_probability"]) - float(record["observed_rate"]))
            for record in records
        )
    )



def _overlap_audit(
    data: pd.DataFrame,
    *,
    strategy_contract: StrategyContract,
    final_refit_sessions: tuple[str, ...],
    final_test_sessions: tuple[str, ...],
    final_embargo_sessions: tuple[str, ...],
) -> dict[str, Any]:
    holdout = _security_holdout_mask(data, strategy_contract)
    development = data.loc[data["session_date_et"].isin(final_refit_sessions)]
    final_test = data.loc[data["session_date_et"].isin(final_test_sessions)]
    unseen_development = development.loc[~holdout.loc[development.index]]
    unseen_final_test = final_test.loc[holdout.loc[final_test.index]]
    records = [
        _overlap_record(
            "temporal_final_refit_vs_locked_test",
            development,
            final_test,
            require_security_disjoint=False,
        ),
        _overlap_record(
            "unseen_security_final_refit_vs_locked_test",
            unseen_development,
            unseen_final_test,
            require_security_disjoint=True,
        ),
    ]
    reserved = set(final_embargo_sessions)
    reserved_overlap = len(
        reserved.intersection(final_refit_sessions)
        | reserved.intersection(final_test_sessions)
    )
    return {
        "row_identity_overlap_total": sum(record["decision_id_overlap"] for record in records),
        "security_identity_overlap_total": sum(
            record["security_id_overlap"] for record in records
        ),
        "session_overlap_total": sum(record["session_overlap"] for record in records),
        "reserved_session_overlap_total": reserved_overlap,
        "all_temporal_partitions_disjoint": all(
            record["decision_id_overlap"] == 0
            and record["security_disjoint_requirement_passed"]
            and record["session_overlap"] == 0
            and record["labels_purged_before_right_partition"]
            for record in records
        )
        and reserved_overlap == 0,
        "records": records,
    }



def _overlap_record(
    name: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    require_security_disjoint: bool,
) -> dict[str, Any]:
    if left.empty or right.empty:
        raise DataReadinessError(f"overlap audit partition is empty: {name}")
    max_label = cast(pd.Timestamp, left["label_available_at_utc"].max())
    min_decision = cast(pd.Timestamp, right["decision_time_utc"].min())
    security_overlap = len(
        set(left["security_id"].astype(str)).intersection(
            right["security_id"].astype(str)
        )
    )
    return {
        "partition": name,
        "decision_id_overlap": len(
            set(left["decision_id"].astype(str)).intersection(
                right["decision_id"].astype(str)
            )
        ),
        "session_overlap": len(
            set(left["session_date_et"].astype(str)).intersection(
                right["session_date_et"].astype(str)
            )
        ),
        "security_id_overlap": security_overlap,
        "security_disjoint_required": require_security_disjoint,
        "security_disjoint_requirement_passed": (
            security_overlap == 0 if require_security_disjoint else True
        ),
        "max_left_label_available_at_utc": _iso(max_label),
        "min_right_decision_time_utc": _iso(min_decision),
        "labels_purged_before_right_partition": max_label < min_decision,
    }




