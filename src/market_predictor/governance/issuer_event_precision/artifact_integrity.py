"""Integrity and immutable-publication helpers for issuer-event precision evidence."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import (
    file_sha256,
)
from market_predictor.core.errors import DataReadinessError

_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")

def _audit_report(name: str, rows: int, failures: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name=name,
                status="pass" if failures == 0 else "fail",
                failures=failures,
                rows_checked=rows,
                detail="deterministic issuer-event precision authority validation",
            ),
        )
    )

def _request(manifest: Mapping[str, object], authority: Mapping[str, object]) -> dict[str, object]:
    value = manifest.get("request")
    if (
        not isinstance(value, dict)
        or _json_sha256(value) != manifest.get("request_sha256")
        or authority.get("request_sha256") != manifest.get("request_sha256")
    ):
        raise DataReadinessError("precision authority request does not verify")
    return {str(key): item for key, item in value.items()}


def _artifact_records(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, dict):
        raise DataReadinessError("precision authority artifact inventory is malformed")
    output: dict[str, dict[str, object]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise DataReadinessError("precision authority artifact record is malformed")
        output[str(key)] = {str(name): item for name, item in value.items()}
    return output


def _artifact_record(path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": str(manifest["artifact_sha256"]),
        "rows": _nonnegative_int(manifest, "rows"),
    }


def _file_record(path: Path, rows: int) -> dict[str, object]:
    return {"path": path.name, "sha256": file_sha256(path), "rows": rows}


def _verify_canonical_record(
    path: Path,
    frame: pd.DataFrame,
    child: Mapping[str, object],
    records: Mapping[str, Mapping[str, object]],
    key: str,
    *,
    request_sha256: str,
    expected_artifact_path: Path | None = None,
) -> None:
    record = records.get(key)
    inputs = child.get("inputs")
    expected_path = (expected_artifact_path or path).resolve()
    if (
        record is None
        or record.get("path") != path.name
        or record.get("sha256") != child.get("artifact_sha256")
        or record.get("rows") != len(frame)
        or child.get("artifact_path") != str(expected_path)
        or not isinstance(inputs, dict)
        or inputs.get("request_sha256") != request_sha256
    ):
        raise DataReadinessError(f"precision {key} artifact lineage does not verify")


def _verify_file_record(
    path: Path,
    records: Mapping[str, Mapping[str, object]],
    key: str,
    rows: int,
) -> None:
    record = records.get(key)
    if record is None or record.get("path") != path.name or record.get("sha256") != file_sha256(path) or record.get("rows") != rows:
        raise DataReadinessError(f"precision {key} file lineage does not verify")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _read_csv(path: Path, expected_columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != list(expected_columns):
        raise DataReadinessError(f"precision ledger schema differs: {path}")
    return frame


def _verify_inventory(directory: Path, expected: set[str]) -> None:
    if directory.is_symlink():
        raise DataReadinessError("precision authority directory cannot be a symlink")
    entries = tuple(directory.iterdir())
    observed = {path.name for path in entries if path.is_file()}
    if (
        observed != expected
        or any(path.is_dir() for path in entries)
        or any(path.is_symlink() for path in entries)
    ):
        raise DataReadinessError("precision authority file inventory differs")


def _new_staging(output_directory: Path) -> Path:
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.with_name(f".{output_directory.name}.{uuid4().hex}.tmp")
    staging.mkdir()
    return staging


def _rewrite_artifact_path(path: Path, final_path: Path) -> None:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = _json_object(manifest_path)
    manifest["artifact_path"] = str(final_path.resolve())
    _atomic_json(manifest_path, manifest)


def _remove_lock(path: Path) -> None:
    path.with_name(f"{path.name}.lock").unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"expected JSON object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _json_text_object(value: str, context: str) -> dict[str, object]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise DataReadinessError(f"{context} is not valid JSON") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"{context} is malformed")
    return {str(key): item for key, item in loaded.items()}


def _json_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_title(value: str) -> str:
    return " ".join(_NORMALIZE_PATTERN.sub(" ", value.lower()).split())


def _clean_text(value: object) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _required_path(record: Mapping[str, object], key: str) -> Path:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"precision lineage has invalid {key}")
    return Path(value).resolve()


def _manifest_request(manifest: Mapping[str, object]) -> Mapping[str, object]:
    value = manifest.get("request")
    if not isinstance(value, dict):
        raise DataReadinessError("precision manifest request is malformed")
    return value


def _required_hash(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise DataReadinessError(f"precision lineage has invalid {key}")
    return value


def _required_text(record: Mapping[str, object], key: str) -> str:
    value = _clean_text(record.get(key))
    if not value:
        raise DataReadinessError(f"precision authority requires {key}")
    return value


def _positive_int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DataReadinessError(f"precision policy has invalid {key}")
    return value


def _nonnegative_int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DataReadinessError(f"precision lineage has invalid {key}")
    return value


def _bounded_float(
    record: Mapping[str, object],
    key: str,
    *,
    lower: float,
    upper: float,
    inclusive: bool = False,
) -> float:
    value = record.get(key)
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise DataReadinessError(f"precision policy has invalid {key}")
    parsed = float(value)
    valid = lower <= parsed <= upper if inclusive else lower < parsed < upper
    if not valid:
        raise DataReadinessError(f"precision policy has invalid {key}")
    return parsed


def _threshold(record: Mapping[str, object], key: str) -> float:
    return _bounded_float(record, key, lower=0.0, upper=1.0, inclusive=True)


def _timestamp(value: object, label: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise DataReadinessError(f"precision {label} is invalid")
    return pd.Timestamp(parsed)


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)

def _assert_frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    label: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_exact=True,
            check_dtype=False,
        )
    except AssertionError as exc:
        raise DataReadinessError(f"{label} failed") from exc
