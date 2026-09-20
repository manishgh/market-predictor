"""Read-only integrity checks for completed historical swing return runs.

This is not training admission, numerical replay, or evidence of causality.
Historical source pins remain opaque: no saved implementation is executed.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.training.return_artifacts import SCHEMA, verify_unit

MAX_JSON_BYTES = 1024**2
MAX_PAYLOAD_BYTES = 64 * 1024**2
MAX_SOURCE_BYTES = 512 * 1024**2
MAX_SOURCE_TOTAL_BYTES = 16 * 1024**3
MAX_SOURCE_FILES = 4096
FLAGS = ("serving_eligible", "promotion_eligible", "portfolio_evaluated", "outer_validation_opened", "historical_test_opened")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _digest(value: Any) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid SHA256 pin")
    return str(value)


def _no_links(path: Path) -> None:
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(info.st_mode)
                 and not (getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT),
                 "retained run path contains a link or reparse point")


def _path(root: Path, raw: Any) -> Path:
    _require(isinstance(raw, str) and 0 < len(raw) <= 1024, "invalid retained path")
    _require(not any(char in raw for char in ("\\", ":", "\x00")), "noncanonical retained path")
    relative = PurePosixPath(raw)
    _require(not relative.is_absolute() and relative.as_posix() == raw
             and all(part not in ("", ".", "..") and not part.endswith((" ", ".")) for part in raw.split("/")),
             "retained path escapes its root or is noncanonical")
    result = root.joinpath(*relative.parts)
    _no_links(result)
    _require(result.resolve().is_relative_to(root), "retained path escapes its root")
    return result


def _bounded_hash(path: Path, limit: int) -> str:
    _no_links(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "retained file size/type exceeds bound")
        digest = hashlib.sha256()
        count = 0
        while block := stream.read(min(1024**2, limit + 1 - count)):
            count += len(block)
            _require(count <= limit, "retained file grew beyond bound")
            digest.update(block)
        after = os.fstat(stream.fileno())
    _no_links(path)
    current = path.stat()
    _require(count == before.st_size and
             (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) ==
             (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns), "retained file changed while hashing")
    return digest.hexdigest()


def _object(path: Path, digest: str) -> dict[str, Any]:
    _digest(digest)
    _no_links(path)
    with path.open("rb") as stream:
        _require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "retained JSON is not a regular file")
        content = stream.read(MAX_JSON_BYTES + 1)
    _require(len(content) <= MAX_JSON_BYTES, "retained JSON exceeds bound")
    _require(hashlib.sha256(content).hexdigest() == digest, "retained JSON differs from external pin")
    return parse_strict_json_object(content, label=str(path))


def _names(value: Any, label: str) -> list[str]:
    _require(isinstance(value, list) and 0 < len(value) <= 4096
             and all(isinstance(item, str) and 0 < len(item) <= 256 for item in value), f"invalid {label}")
    _require(len(set(value)) == len(value), f"duplicate {label}")
    return list(value)


def _request(request: dict[str, Any]) -> ReturnTrainingPolicy:
    _require(request.get("schema") == "market_predictor.swing_return_training_request", "wrong request schema")
    _require(all(request.get(flag) is False for flag in FLAGS), "request is not exclusively historical research")
    policy = ReturnTrainingPolicy.model_validate(request["policy"])
    sources = request.get("source_files")
    if not isinstance(sources, dict) or not 0 < len(sources) <= MAX_SOURCE_FILES:
        raise DataReadinessError("invalid source inventory size")
    for pin in (policy.readiness, policy.readiness_config):
        _require(sources.get(pin.path) == pin.sha256, "policy source pin is absent or differs from source inventory")
    _names(request["feature_names"], "feature names")
    _names(request["holdout_security_ids"], "holdout identities")
    _digest(request["input_decision_ids_sha256"])
    _require(type(request["input_rows"]) is int and request["input_rows"] > 0, "invalid input row count")
    folds = request["folds"]
    _require(isinstance(folds, list) and len(folds) == policy.folds, "request must contain four folds")
    for number, fold in enumerate(folds, 1):
        _require(isinstance(fold, dict) and set(fold) == {"number", "train", "embargo", "score"}
                 and type(fold["number"]) is int and fold["number"] == number, "invalid fold identity")
        days = [_names(fold[name], "fold sessions") for name in ("train", "embargo", "score")]
        for group in days:
            _require(group == sorted(group) and all(date.fromisoformat(day).isoformat() == day for day in group),
                     "invalid fold session order")
        _require(len(days[0]) >= policy.minimum_train_sessions and len(days[1]) == policy.embargo_sessions
                 and days[0][-1] < days[1][0] and days[1][-1] < days[2][0], "invalid recorded fold boundaries")
    return policy


def _unit(manifest: dict[str, Any], request: dict[str, Any], policy: ReturnTrainingPolicy,
          family: str, scope: str, parameters: dict[str, Any]) -> dict[str, Any]:
    _require(set(manifest) == {"schema", "request_sha256", "unit", "files", "metrics", "serving_eligible", "promotion_eligible"}
             and manifest["schema"] == SCHEMA, "wrong unit schema")
    unit: dict[str, Any] = manifest["unit"]
    _require(unit["family"] == family and unit["scope"] == scope and unit["parameters"] == parameters
             and unit["feature_names"] == request["feature_names"] and unit["target"] == policy.target
             and unit["preprocessing"] == policy.missingness, "unit differs from frozen request")
    for name in ("training_rows", "training_sessions", "scoring_rows", "scope_eligible_rows"):
        _require(type(unit[name]) is int and 0 <= unit[name] <= request["input_rows"], "invalid unit counts")
    _require(0 < unit["training_sessions"] <= unit["training_rows"]
             and unit["scope_eligible_rows"] <= unit["scoring_rows"], "inconsistent unit counts")
    if scope == "final_refit":
        _require(unit["scoring_rows"] == unit["scope_eligible_rows"] == 0 and manifest["metrics"] == {},
                 "final refit must not claim scoring")
    else:
        _require(unit["scope_eligible_rows"] > 0 and isinstance(manifest["metrics"], dict), "missing scoring evidence")
    identities = _names(unit["training_security_ids"], "training identities")
    if scope == "security_transfer":
        _require(not set(identities).intersection(request["holdout_security_ids"]), "transfer training includes holdouts")
    for name in ("training_decision_ids_sha256", "training_weights_sha256", "scoring_decision_ids_sha256"):
        _digest(unit[name])
    cutoff = datetime.fromisoformat(unit["fit_cutoff_utc"])
    maturity = datetime.fromisoformat(unit["maximum_training_label_maturity"])
    _require(cutoff.utcoffset() is not None and maturity.utcoffset() is not None and maturity < cutoff,
             "invalid recorded maturity boundary")
    return unit


def verify_completed_run_integrity(*, root: Path, directory: Path, manifest_sha256: str,
                                   request_sha256: str, checkpoint_sha256: str) -> dict[str, Any]:
    """Verify one 18-unit historical run; report source drift without admitting reuse.

    Pins must come from an independent retained inventory, not this directory.
    Only explicitly named files are read. No model or parquet is deserialized.
    Source hashes may require substantial sequential I/O; no raw-data walk occurs.
    """
    try:
        return _verify(root, directory, manifest_sha256, request_sha256, checkpoint_sha256)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        raise DataReadinessError(f"invalid retained swing run: {exc}") from exc


def _verify(root: Path, directory: Path, manifest_sha256: str, request_sha256: str,
            checkpoint_sha256: str) -> dict[str, Any]:
    root = Path(os.path.abspath(root))
    _no_links(root)
    directory = _path(root, directory.as_posix())
    relative = directory.relative_to(root).as_posix()
    roots = {"_manifest.json": manifest_sha256, "_request.json": request_sha256, "_checkpoint.json": checkpoint_sha256}
    objects = {name: _object(_path(directory, name), digest) for name, digest in roots.items()}
    manifest, request, checkpoint = (objects[name] for name in roots)
    policy = _request(request)
    _require(set(manifest) == {"schema", "status", "request_sha256", "specifications", "fold_scope_fits", "final_models",
                              "units", *FLAGS}, "unexpected root manifest fields")
    _require(manifest["schema"] == "market_predictor.swing_return_training"
             and manifest["status"] == "complete_research_only"
             and all(manifest[flag] is False for flag in FLAGS), "run is not complete research-only")
    for name, expected in (("specifications", 2), ("fold_scope_fits", 16), ("final_models", 2)):
        _require(type(manifest[name]) is int and manifest[name] == expected, "wrong completed unit counts")
    _require(set(checkpoint) == {"request_sha256", "units"}
             and manifest["request_sha256"] == checkpoint["request_sha256"] == request_sha256
             and manifest["units"] == checkpoint["units"], "root/checkpoint request or inventory mismatch")
    families = {"regularized_linear_return": policy.linear.model_dump(), "shallow_boosted_return": policy.boosted.model_dump()}
    expected_units = {f"{family}/{scope}/fold-{fold['number']}": (family, scope, parameters)
                      for family, parameters in families.items() for fold in request["folds"]
                      for scope in ("temporal", "security_transfer")}
    expected_units.update({f"{family}/final_refit": (family, "final_refit", parameters) for family, parameters in families.items()})
    _require(isinstance(manifest["units"], dict) and set(manifest["units"]) == set(expected_units), "expected exactly 18 units")
    payload_bytes = 0
    for key, (family, scope, parameters) in expected_units.items():
        record = manifest["units"][key]
        _require(set(record) == {"path", "manifest_sha256", "metrics"} and record["path"] == f"{relative}/{key}",
                 "unit path or record differs from expected inventory")
        child = _path(directory, key)
        item = _object(_path(child, "_manifest.json"), record["manifest_sha256"])
        unit = _unit(item, request, policy, family, scope, parameters)
        fold = request["folds"][-1] if scope == "final_refit" else request["folds"][int(key.rsplit("-", 1)[1]) - 1]
        cutoff_day = fold["score"][-1] if scope == "final_refit" else fold["score"][0]
        _require(datetime.fromisoformat(unit["fit_cutoff_utc"]).date().isoformat() == cutoff_day,
                 "unit cutoff differs from recorded fold")
        _require(item["request_sha256"] == request_sha256 and record["metrics"] == item["metrics"], "unit request/metrics mismatch")
        files = {"model.joblib"} if scope == "final_refit" else {"model.joblib", "predictions.parquet"}
        _require(isinstance(item["files"], dict) and set(item["files"]) == files, "unexpected unit files")
        for name, digest in item["files"].items():
            path = _path(child, name)
            payload_bytes += path.stat().st_size
            _require(payload_bytes <= 1024**3, "run payload exceeds total bound")
            _require(_bounded_hash(path, MAX_PAYLOAD_BYTES) == _digest(digest), "unit payload hash mismatch")
        verify_unit(child, manifest_sha256=record["manifest_sha256"], request_sha256=request_sha256, unit=unit)
    sources = request["source_files"]
    _require(isinstance(sources, dict) and 0 < len(sources) <= MAX_SOURCE_FILES, "invalid source inventory size")
    source_results = []
    total = 0
    for name, source_pin in sources.items():
        source_pin = _digest(source_pin)
        actual = None
        try:
            path = _path(root, name)
            total += path.stat().st_size
            _require(total <= MAX_SOURCE_TOTAL_BYTES, "source inventory exceeds total bound")
            actual = _bounded_hash(path, MAX_SOURCE_BYTES)
            status = "matching" if actual == source_pin else "hash_mismatch"
        except FileNotFoundError:
            status = "missing"
        except PermissionError:
            status = "unreadable"
        source_results.append({"path": name, "expected_sha256": source_pin, "actual_sha256": actual, "status": status})
    for name, digest in roots.items():
        _object(_path(directory, name), digest)
    return {"schema": "market_predictor.swing_retained_run_integrity.v1", "scope": "historical_integrity_only",
            "status": "historical_integrity_verified", "directory": relative, "root_sha256": roots,
            "verified_units": len(expected_units), "source_files": source_results,
            "source_integrity": "matching" if all(row["status"] == "matching" for row in source_results) else "discrepancies",
            "current_replay": "unverified", "numerical_equivalence": "unverified", "causality": "unverified",
            "serving_eligible": False, "promotion_eligible": False}
