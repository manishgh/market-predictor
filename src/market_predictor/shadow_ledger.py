from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import pandas as pd

from market_predictor.locking import file_lock
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.evaluation import session_block_interval

SHADOW_BUNDLE_SCHEMA = "market_predictor.shadow_evidence.v1"
SHADOW_LEDGER_ENTRY_SCHEMA = "market_predictor.shadow_ledger_entry.v1"
ShadowResult = Literal["passed", "failed"]


def write_shadow_bundle(
    root: Path,
    sessions: pd.DataFrame,
    *,
    hypothesis: dict[str, Any],
    candidate_artifact_sha256: str,
    generated_at: datetime | None = None,
    bootstrap_iterations: int = 1_000,
    bootstrap_seed: int = 42,
) -> Path:
    """Write content-addressed, immutable paired candidate/baseline shadow evidence."""

    required = {"session_date_et", "candidate_benchmark_excess_return", "baseline_benchmark_excess_return"}
    if missing := sorted(required.difference(sessions.columns)):
        raise DataReadinessError(f"shadow evidence is missing columns: {', '.join(missing)}")
    if bootstrap_iterations < 100:
        raise ValueError("shadow bootstrap requires at least 100 iterations")
    _require_sha256(candidate_artifact_sha256, "candidate_artifact_sha256")
    hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
    family = str(hypothesis.get("hypothesis_family") or "")
    baseline_id = str(hypothesis.get("baseline_id") or "")
    hypothesis_sha = str(hypothesis.get("record_sha256") or "")
    prediction_policy_sha = str(hypothesis.get("prediction_policy_sha256") or "")
    for value, name in (
        (hypothesis_sha, "hypothesis record_sha256"),
        (prediction_policy_sha, "hypothesis prediction_policy_sha256"),
    ):
        _require_sha256(value, name)
    evidence = sessions.loc[:, sorted(required)].copy()
    evidence["session_date_et"] = pd.to_datetime(evidence["session_date_et"], errors="coerce").dt.date
    for column in ("candidate_benchmark_excess_return", "baseline_benchmark_excess_return"):
        evidence[column] = pd.to_numeric(evidence[column], errors="coerce")
    if bool(evidence.isna().any(axis=1).any()) or len(evidence) < 2:
        raise DataReadinessError("shadow evidence requires at least two complete sessions")
    if bool(evidence["session_date_et"].duplicated().any()):
        raise DataReadinessError("shadow evidence requires exactly one paired row per session")
    numeric = evidence[["candidate_benchmark_excess_return", "baseline_benchmark_excess_return"]]
    if not bool(numeric.map(math.isfinite).all(axis=None)):
        raise DataReadinessError("shadow evidence returns must be finite")
    evidence = evidence.sort_values("session_date_et", kind="stable").reset_index(drop=True)
    evidence["paired_improvement"] = (
        evidence["candidate_benchmark_excess_return"] - evidence["baseline_benchmark_excess_return"]
    )
    interval = session_block_interval(
        evidence,
        metric=lambda frame: float(pd.to_numeric(frame["paired_improvement"]).mean()),
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    records = [
        {
            "session_date_et": cast(date, row.session_date_et).isoformat(),
            "candidate_benchmark_excess_return": float(row.candidate_benchmark_excess_return),
            "baseline_benchmark_excess_return": float(row.baseline_benchmark_excess_return),
        }
        for row in evidence.itertuples(index=False)
    ]
    created = _utc(generated_at or datetime.now(UTC))
    declared_at = _parse_utc(str(hypothesis.get("declared_at_utc") or ""), "hypothesis declared_at_utc")
    if created <= declared_at:
        raise DataReadinessError("shadow evidence must be generated after hypothesis declaration")
    content: dict[str, Any] = {
        "schema": SHADOW_BUNDLE_SCHEMA,
        "hypothesis_id": hypothesis_id,
        "hypothesis_family": family,
        "hypothesis_record_sha256": hypothesis_sha,
        "baseline_id": baseline_id,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "prediction_policy_sha256": prediction_policy_sha,
        "generated_at_utc": created.isoformat(),
        "first_session_date_et": records[0]["session_date_et"],
        "last_session_date_et": records[-1]["session_date_et"],
        "independent_sessions": len(records),
        "bootstrap": {"iterations": bootstrap_iterations, "seed": bootstrap_seed},
        "paired_improvement_interval": interval,
        "session_returns": records,
    }
    fingerprint = _json_sha256(content)
    payload = {**content, "shadow_fingerprint": fingerprint}
    path = root / "shadow" / f"{fingerprint}.json"
    with file_lock(root / ".shadow-bundles"):
        if path.exists():
            existing = load_shadow_bundle(path)
            if existing != payload:
                raise DataReadinessError("shadow fingerprint collision or immutable bundle mismatch")
            return path
        _write_json_atomic(path, payload)
    return path


def load_shadow_bundle(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"shadow evidence bundle is unavailable or invalid: {path}") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError("shadow evidence bundle must contain an object")
    payload = {str(key): value for key, value in loaded.items()}
    fingerprint = str(payload.pop("shadow_fingerprint", ""))
    if payload.get("schema") != SHADOW_BUNDLE_SCHEMA or _json_sha256(payload) != fingerprint:
        raise DataReadinessError("shadow evidence bundle integrity check failed")
    return {**payload, "shadow_fingerprint": fingerprint}


def shadow_gate_failures(
    bundle: dict[str, Any],
    *,
    minimum_independent_sessions: int,
    minimum_paired_improvement_ci_low: float,
) -> list[str]:
    failures: list[str] = []
    sessions = _int_value(bundle.get("independent_sessions"))
    if sessions is None or sessions < minimum_independent_sessions:
        failures.append(f"shadow independent_sessions {sessions} < {minimum_independent_sessions}")
    interval = bundle.get("paired_improvement_interval")
    low = _float_value(interval.get("low")) if isinstance(interval, dict) else None
    if low is None or low <= minimum_paired_improvement_ci_low:
        failures.append(
            f"shadow paired improvement CI low {low} must be > {minimum_paired_improvement_ci_low}"
        )
    return failures


def consume_shadow_fingerprint(
    ledger_path: Path,
    *,
    bundle: dict[str, Any],
    hypothesis: dict[str, Any],
    result: ShadowResult,
    attestation_id: str | None,
    consumed_at: datetime | None = None,
) -> dict[str, Any]:
    """Append one hash-chained decision; fingerprints are never reusable."""

    if result not in {"passed", "failed"}:
        raise ValueError("shadow result must be passed or failed")
    fingerprint = str(bundle.get("shadow_fingerprint") or "")
    family = str(hypothesis.get("hypothesis_family") or "")
    hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
    if bundle.get("hypothesis_id") != hypothesis_id or bundle.get("hypothesis_family") != family:
        raise DataReadinessError("shadow evidence does not match the predeclared hypothesis")
    _require_sha256(fingerprint, "shadow_fingerprint")
    if attestation_id is not None:
        _require_sha256(attestation_id, "attestation_id")
    with file_lock(ledger_path):
        entries = load_shadow_ledger(ledger_path)
        if any(entry.get("shadow_fingerprint") == fingerprint for entry in entries):
            raise DataReadinessError("shadow fingerprint has already been consumed")
        if any(entry.get("hypothesis_family") == family and entry.get("result") == "failed" for entry in entries):
            raise DataReadinessError("hypothesis family was retired by failed shadow evidence")
        previous_sha = str(entries[-1]["entry_sha256"]) if entries else None
        content: dict[str, Any] = {
            "schema": SHADOW_LEDGER_ENTRY_SCHEMA,
            "sequence": len(entries) + 1,
            "previous_entry_sha256": previous_sha,
            "shadow_fingerprint": fingerprint,
            "hypothesis_id": hypothesis_id,
            "hypothesis_family": family,
            "result": result,
            "attestation_id": attestation_id,
            "consumed_at_utc": _utc(consumed_at or datetime.now(UTC)).isoformat(),
        }
        entry = {**content, "entry_sha256": _json_sha256(content)}
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return entry


def load_shadow_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    previous_sha: str | None = None
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataReadinessError(f"shadow ledger is corrupt at line {line_number}") from exc
        if not isinstance(loaded, dict):
            raise DataReadinessError(f"shadow ledger is corrupt at line {line_number}")
        entry = {str(key): value for key, value in loaded.items()}
        entry_sha = str(entry.pop("entry_sha256", ""))
        if (
            entry.get("schema") != SHADOW_LEDGER_ENTRY_SCHEMA
            or entry.get("sequence") != line_number
            or entry.get("previous_entry_sha256") != previous_sha
            or _json_sha256(entry) != entry_sha
        ):
            raise DataReadinessError(f"shadow ledger integrity check failed at line {line_number}")
        entry = {**entry, "entry_sha256": entry_sha}
        entries.append(entry)
        previous_sha = entry_sha
    return entries


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{name} must be a SHA-256 digest")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("shadow timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _parse_utc(value: str, name: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise DataReadinessError(f"{name} must be timezone-aware")
    return cast(datetime, timestamp.tz_convert("UTC").to_pydatetime())


def _float_value(value: object) -> float | None:
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int_value(value: object) -> int | None:
    try:
        parsed = int(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return parsed
