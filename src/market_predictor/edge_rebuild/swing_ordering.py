"""Deterministic top-versus-bottom swing ordering gate before model fitting."""
from __future__ import annotations



import hashlib
import json
import shutil
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.swing_materialization import (
    load_complete_swing_feature_panel,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.core.errors import DataReadinessError

ORDERING_SCHEMA: Final = "edge_rebuild.swing_ordering_audit.v1"
ORDERING_AUTHORITY_SCHEMA: Final = "edge_rebuild.swing_ordering_audit_authority.v1"


class SwingOrderingConfig(BaseModel):
    """Preregistered score and economic acceptance criteria."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    score_features: tuple[str, ...]
    score_directions: tuple[int, ...]
    top_quantile: float = Field(gt=0.0, lt=0.5)
    bottom_quantile: float = Field(gt=0.0, lt=0.5)
    minimum_scored_securities_per_session: int = Field(ge=20)
    minimum_sessions: int = Field(ge=60)
    minimum_mean_spread_bps: float = Field(ge=0.0)
    minimum_positive_session_share: float = Field(ge=0.0, le=1.0)
    minimum_newey_west_t_stat: float = Field(ge=0.0)
    newey_west_lag_sessions: int = Field(ge=1, le=60)

    @model_validator(mode="after")
    def validate_score(self) -> Self:
        if self.schema_version != "edge_rebuild.swing_ordering.v1":
            raise ValueError("unsupported swing ordering config schema")
        if not self.score_features or len(self.score_features) != len(
            self.score_directions
        ):
            raise ValueError("score features and directions must be non-empty and aligned")
        if len(set(self.score_features)) != len(self.score_features):
            raise ValueError("swing ordering score features must be unique")
        if set(self.score_directions).difference({-1, 1}):
            raise ValueError("swing ordering directions must be -1 or 1")
        if self.top_quantile + self.bottom_quantile >= 1.0:
            raise ValueError("top and bottom quantiles must leave a middle population")
        return self


def load_swing_ordering_config(path: Path) -> SwingOrderingConfig:
    with path.open("rb") as handle:
        return SwingOrderingConfig.model_validate(tomllib.load(handle))


def audit_swing_ordering(
    *,
    panel_dir: Path,
    config_path: Path,
    output_dir: Path,
    memory_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.75,
) -> dict[str, Any]:
    """Evaluate an outcome-blind technical composite on managed net returns."""

    if output_dir.exists():
        raise DataReadinessError(f"swing ordering output must be new: {output_dir}")
    config = load_swing_ordering_config(config_path)
    panel_manifest = load_complete_swing_feature_panel(panel_dir)
    panel_manifest_path = panel_dir / "final" / "_manifest.json"
    request = {
        "schema": ORDERING_SCHEMA,
        "panel_dir": str(panel_dir.resolve()),
        "panel_manifest_sha256": file_sha256(panel_manifest_path),
        "panel_request_sha256": panel_manifest["request_sha256"],
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "score_features": list(config.score_features),
        "score_directions": list(config.score_directions),
    }
    request_sha256 = _json_sha256(request)
    session_parts: list[pd.DataFrame] = []
    required = [
        "security_id",
        "session_date_et",
        "decision_time_utc",
        "barrier_label_available_at_utc",
        "feature_eligible",
        "forward_return",
        *config.score_features,
    ]
    for record in panel_manifest["files"]:
        path = panel_dir / "final" / str(record["path"])
        frame = pd.read_parquet(path, columns=required)
        session_parts.append(_score_partition(frame, config=config))
        del frame
        release_process_memory()
        _guard(memory_budget_gib, memory_headroom_gib, "swing ordering partition")
    if not session_parts:
        raise DataReadinessError("swing ordering audit found no panel partitions")
    sessions = pd.concat(session_parts, ignore_index=True).sort_values(
        "session_date_et",
        kind="stable",
    )
    del session_parts
    if len(sessions) < config.minimum_sessions:
        raise DataReadinessError(
            f"swing ordering has {len(sessions)} sessions; "
            f"{config.minimum_sessions} required"
        )
    spread = sessions["spread_return"].to_numpy(dtype=float)
    mean_spread = float(np.mean(spread))
    t_stat = _newey_west_t_stat(spread, config.newey_west_lag_sessions)
    positive_share = float(np.mean(spread > 0.0))
    gates = {
        "minimum_sessions": len(sessions) >= config.minimum_sessions,
        "minimum_mean_spread_bps": (
            mean_spread * 10_000.0 >= config.minimum_mean_spread_bps
        ),
        "minimum_positive_session_share": (
            positive_share >= config.minimum_positive_session_share
        ),
        "minimum_newey_west_t_stat": (
            t_stat >= config.minimum_newey_west_t_stat
        ),
    }
    manifest: dict[str, Any] = {
        "schema": ORDERING_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "request_sha256": request_sha256,
        "status": "passed" if all(gates.values()) else "failed",
        "outcome": "managed_ten_session_barrier_net_return",
        "score": {
            "aggregation": "equal_weighted_directional_mean",
            "features": list(config.score_features),
            "directions": list(config.score_directions),
        },
        "sessions": int(len(sessions)),
        "first_session": str(pd.to_datetime(sessions["session_date_et"]).min().date()),
        "last_session": str(pd.to_datetime(sessions["session_date_et"]).max().date()),
        "top_rows": int(sessions["top_rows"].sum()),
        "bottom_rows": int(sessions["bottom_rows"].sum()),
        "top_mean_return": float(
            np.average(sessions["top_mean_return"], weights=sessions["top_rows"])
        ),
        "bottom_mean_return": float(
            np.average(
                sessions["bottom_mean_return"],
                weights=sessions["bottom_rows"],
            )
        ),
        "mean_session_spread": mean_spread,
        "mean_session_spread_bps": mean_spread * 10_000.0,
        "median_session_spread_bps": float(np.median(spread) * 10_000.0),
        "positive_session_share": positive_share,
        "newey_west_lag_sessions": config.newey_west_lag_sessions,
        "newey_west_t_stat": t_stat,
        "gates": gates,
        "thresholds": {
            "minimum_sessions": config.minimum_sessions,
            "minimum_mean_spread_bps": config.minimum_mean_spread_bps,
            "minimum_positive_session_share": config.minimum_positive_session_share,
            "minimum_newey_west_t_stat": config.minimum_newey_west_t_stat,
        },
        "source": request,
        "memory": memory_audit(
            hard_budget_gib=memory_budget_gib,
            headroom_gib=memory_headroom_gib,
        ).to_record(),
    }
    _publish_ordering_audit(
        sessions,
        manifest=manifest,
        request={**request, "request_sha256": request_sha256},
        output_dir=output_dir,
    )
    return manifest


def load_complete_swing_ordering_audit(output_dir: Path) -> dict[str, Any]:
    """Verify an ordering audit before any later gate consumes it."""

    request = _load_json(output_dir / "_request.json")
    manifest_path = output_dir / "_manifest.json"
    manifest = _load_json(manifest_path)
    authority = _load_json(output_dir / "_authority.json")
    request_payload = {
        key: value for key, value in request.items() if key != "request_sha256"
    }
    request_sha256 = _json_sha256(request_payload)
    sessions_path = output_dir / "session_spreads.parquet"
    if (
        request.get("schema") != ORDERING_SCHEMA
        or request.get("request_sha256") != request_sha256
        or manifest.get("schema") != ORDERING_SCHEMA
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != ORDERING_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("request_sha256") != request_sha256
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or not sessions_path.is_file()
        or manifest.get("session_spreads_sha256") != file_sha256(sessions_path)
    ):
        raise DataReadinessError(
            f"swing ordering audit authority does not verify: {output_dir}"
        )
    sessions = pd.read_parquet(sessions_path, columns=["session_date_et"])
    if len(sessions) != int(manifest.get("sessions", -1)):
        raise DataReadinessError("swing ordering session row count does not verify")
    return manifest


def _score_partition(
    frame: pd.DataFrame,
    *,
    config: SwingOrderingConfig,
) -> pd.DataFrame:
    required = set(config.score_features) | {
        "security_id",
        "session_date_et",
        "decision_time_utc",
        "barrier_label_available_at_utc",
        "feature_eligible",
        "forward_return",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataReadinessError(f"swing ordering panel columns are missing: {missing}")
    decision = pd.to_datetime(frame["decision_time_utc"], utc=True, errors="coerce")
    label_available = pd.to_datetime(
        frame["barrier_label_available_at_utc"],
        utc=True,
        errors="coerce",
    )
    numeric = frame.loc[:, list(config.score_features)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    eligible = (
        frame["feature_eligible"].fillna(False).astype(bool)
        & pd.to_numeric(frame["forward_return"], errors="coerce").notna()
        & numeric.notna().all(axis=1)
        & decision.notna()
        & label_available.notna()
        & label_available.ge(decision)
    )
    data = frame.loc[
        eligible,
        ["security_id", "session_date_et", "forward_return"],
    ].copy()
    if data.empty:
        return _empty_session_metrics()
    data["score"] = (
        numeric.loc[eligible].to_numpy(dtype=float)
        * np.asarray(config.score_directions, dtype=float)
    ).mean(axis=1)
    counts = data.groupby("session_date_et")["security_id"].transform("size")
    data = data.loc[counts.ge(config.minimum_scored_securities_per_session)].copy()
    if data.empty:
        return _empty_session_metrics()
    data["score_percentile"] = data.groupby("session_date_et")["score"].rank(
        method="average",
        pct=True,
    )
    top = data["score_percentile"].gt(1.0 - config.top_quantile)
    bottom = data["score_percentile"].le(config.bottom_quantile)
    selected = data.loc[top | bottom].assign(bucket=np.where(top[top | bottom], "top", "bottom"))
    grouped = selected.groupby(["session_date_et", "bucket"])["forward_return"]
    means = grouped.mean().unstack("bucket")
    sizes = grouped.size().unstack("bucket")
    complete = means[["top", "bottom"]].notna().all(axis=1)
    output = pd.DataFrame(
        {
            "session_date_et": means.index[complete],
            "top_mean_return": means.loc[complete, "top"].to_numpy(),
            "bottom_mean_return": means.loc[complete, "bottom"].to_numpy(),
            "top_rows": sizes.loc[complete, "top"].to_numpy(dtype=int),
            "bottom_rows": sizes.loc[complete, "bottom"].to_numpy(dtype=int),
        }
    )
    output["spread_return"] = (
        output["top_mean_return"] - output["bottom_mean_return"]
    )
    return output


def _newey_west_t_stat(values: np.ndarray, lag: int) -> float:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if len(finite) < lag + 2:
        raise DataReadinessError("too few session spreads for Newey-West inference")
    residual = finite - float(np.mean(finite))
    long_run_variance = float(np.dot(residual, residual) / len(residual))
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        covariance = float(
            np.dot(residual[offset:], residual[:-offset]) / len(residual)
        )
        long_run_variance += 2.0 * weight * covariance
    standard_error = np.sqrt(max(long_run_variance, 0.0) / len(residual))
    if standard_error == 0.0:
        return float("inf") if float(np.mean(finite)) > 0.0 else 0.0
    return float(np.mean(finite) / standard_error)


def _empty_session_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "session_date_et",
            "top_mean_return",
            "bottom_mean_return",
            "top_rows",
            "bottom_rows",
            "spread_return",
        ]
    )


def _publish_ordering_audit(
    sessions: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    request: dict[str, Any],
    output_dir: Path,
) -> None:
    staging = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    try:
        sessions_path = staging / "session_spreads.parquet"
        sessions.to_parquet(sessions_path, index=False)
        manifest["session_spreads_sha256"] = file_sha256(sessions_path)
        _write_json(staging / "_request.json", request)
        _write_json(staging / "_manifest.json", manifest)
        _write_json(
            staging / "_authority.json",
            {
                "schema": ORDERING_AUTHORITY_SCHEMA,
                "state": "complete",
                "request_sha256": manifest["request_sha256"],
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(staging / "_manifest.json"),
            },
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _guard(hard: float, headroom: float, stage: str) -> None:
    assert_memory_budget(hard_budget_gib=hard, headroom_gib=headroom, stage=stage)
    assert_peak_memory_budget(
        hard_budget_gib=hard,
        headroom_gib=headroom,
        stage=stage,
    )


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"invalid or missing JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"JSON artifact must contain an object: {path}")
    return {str(key): item for key, item in value.items()}
