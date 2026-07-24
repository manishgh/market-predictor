from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from market_predictor.locking import file_lock
from market_predictor.v3.errors import DataReadinessError

SHADOW_LEDGER_ENTRY_SCHEMA = "market_predictor.shadow_ledger_entry.v2"
ShadowResult = Literal["passed", "failed"]


def shadow_gate_failures(
    bundle: dict[str, Any],
    *,
    minimum_independent_sessions: int,
    minimum_paired_improvement_ci_low: float,
) -> list[str]:
    failures: list[str] = []
    sessions = _int_value(bundle.get("independent_sessions"))
    if sessions is None or sessions < minimum_independent_sessions:
        failures.append(
            "shadow independent_sessions "
            f"{sessions} < {minimum_independent_sessions}"
        )
    interval = bundle.get("paired_improvement_interval")
    low = (
        _float_value(interval.get("low"))
        if isinstance(interval, dict)
        else None
    )
    if low is None or low <= minimum_paired_improvement_ci_low:
        failures.append(
            "shadow paired improvement CI low "
            f"{low} must be > {minimum_paired_improvement_ci_low}"
        )
    return failures


def consume_shadow_fingerprint(
    ledger_path: Path,
    *,
    bundle: dict[str, Any],
    hypothesis: dict[str, Any],
    result: ShadowResult,
    attestation_id: str | None,
    transaction_id: str,
    consumed_at: datetime | None = None,
) -> dict[str, Any]:
    """Append one hash-chained decision; fingerprints are never reusable."""

    if result not in {"passed", "failed"}:
        raise ValueError("shadow result must be passed or failed")
    fingerprint = str(bundle.get("shadow_fingerprint") or "")
    family = str(hypothesis.get("hypothesis_family") or "")
    hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
    if (
        bundle.get("hypothesis_id") != hypothesis_id
        or bundle.get("hypothesis_family") != family
    ):
        raise DataReadinessError(
            "shadow evidence does not match the predeclared hypothesis"
        )
    _require_sha256(fingerprint, "shadow_fingerprint")
    _require_sha256(transaction_id, "transaction_id")
    if attestation_id is not None:
        _require_sha256(attestation_id, "attestation_id")
    with file_lock(ledger_path):
        entries = load_shadow_ledger(ledger_path)
        existing = next(
            (
                entry
                for entry in entries
                if entry.get("shadow_fingerprint") == fingerprint
            ),
            None,
        )
        if existing is not None:
            if (
                existing.get("transaction_id") == transaction_id
                and existing.get("result") == result
                and existing.get("attestation_id") == attestation_id
            ):
                return existing
            raise DataReadinessError(
                "shadow fingerprint has already been consumed"
            )
        if any(
            entry.get("hypothesis_family") == family
            and entry.get("result") == "failed"
            for entry in entries
        ):
            raise DataReadinessError(
                "hypothesis family was retired by failed shadow evidence"
            )
        previous_sha = (
            str(entries[-1]["entry_sha256"]) if entries else None
        )
        content: dict[str, Any] = {
            "schema": SHADOW_LEDGER_ENTRY_SCHEMA,
            "sequence": len(entries) + 1,
            "previous_entry_sha256": previous_sha,
            "shadow_fingerprint": fingerprint,
            "hypothesis_id": hypothesis_id,
            "hypothesis_family": family,
            "result": result,
            "attestation_id": attestation_id,
            "transaction_id": transaction_id,
            "consumed_at_utc": _utc(
                consumed_at or datetime.now(UTC)
            ).isoformat(),
        }
        entry = {
            **content,
            "entry_sha256": _json_sha256(content),
        }
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(
                json.dumps(
                    entry,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    return entry


def load_shadow_ledger(
    ledger_path: Path,
) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    previous_sha: str | None = None
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataReadinessError(
                f"shadow ledger is corrupt at line {line_number}"
            ) from exc
        if not isinstance(loaded, dict):
            raise DataReadinessError(
                f"shadow ledger is corrupt at line {line_number}"
            )
        entry = {str(key): value for key, value in loaded.items()}
        entry_sha = str(entry.pop("entry_sha256", ""))
        if (
            entry.get("schema") != SHADOW_LEDGER_ENTRY_SCHEMA
            or entry.get("sequence") != line_number
            or entry.get("previous_entry_sha256") != previous_sha
            or _json_sha256(entry) != entry_sha
        ):
            raise DataReadinessError(
                "shadow ledger integrity check failed at line "
                f"{line_number}"
            )
        entry = {**entry, "entry_sha256": entry_sha}
        entries.append(entry)
        previous_sha = entry_sha
    return entries


def _json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef"
        for character in value.lower()
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("shadow timestamps must be timezone-aware")
    return value.astimezone(UTC)


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
