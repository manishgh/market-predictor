"""Publish independently pinned, month-bounded catalyst decision authorities."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.reconciliation import stamp_canonical_decision_ids
from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import heavy_job_lease

CONFIG_SCHEMA = "swing.monthly_catalyst_publication.v1"
INDEX_SCHEMA = "market_predictor.initial_fit_catalyst_months"
_KEYS = ["decision_id", "security_id", "ticker", "decision_time_utc"]
_DECISION_COLUMNS = [*_KEYS, "timeframe", "bar_start_utc", "prediction_cutoff_policy_id"]
_SOURCE_KEYS = (
    "collection_manifest_sha256", "collection_audit_sha256", "attribution_manifest_sha256",
    "sentiment_manifest_sha256", "source_collections_sha256", "policy_sha256",
)


def _object(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise DataReadinessError("monthly catalyst metadata must be an object")
    return result


def _pin(path: Path, digest: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or path.is_symlink() or file_sha256(path) != digest:
        raise DataReadinessError(f"monthly catalyst external pin differs: {path}")


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise DataReadinessError("monthly catalyst input escapes the workspace")
    return path.resolve()


def _check_sources(root: Path, pins: dict[str, str]) -> None:
    if not pins:
        raise DataReadinessError("monthly catalyst publication requires independent source metadata pins")
    for name, digest in pins.items():
        _pin(_path(root, name), digest)


def _lineage_source_pins(root: Path, lineage: dict[str, Any], request: Any, pins: dict[str, str]) -> None:
    paths = lineage.get("source_paths")
    if not isinstance(paths, dict) or set(paths) != set(_SOURCE_KEYS):
        raise DataReadinessError("lineage requires an explicit path for each source pin")
    for key in _SOURCE_KEYS:
        path = _path(root, paths[key])
        relative = path.relative_to(root).as_posix()
        if pins.get(relative) != request[key]:
            raise DataReadinessError(f"lineage source path/hash association differs: {key}")
        _pin(path, request[key])


def _checkpoint(output: Path, payload: dict[str, Any], progress: Callable[[dict[str, Any]], None] | None) -> None:
    path = output / "_checkpoint.json"
    temporary = output / f".checkpoint-{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if progress is not None:
        progress({"checkpoint_path": str(path), "checkpoint_sha256": file_sha256(path),
            "completed_months": sorted(payload["months"])})


def _check_completed_month(output: Path, month: str, record: dict[str, Any], expected: dict[str, Any]) -> None:
    if not re.fullmatch(re.escape(month) + r"(?:\.retry-[0-9a-f]{32})?", record["directory"]):
        raise DataReadinessError("monthly catalyst index directory differs")
    if any(record[key] != expected[key] for key in ("rows", "decision_ids_sha256", "decisions", "lineages")):
        raise DataReadinessError("monthly catalyst decision inventory differs")
    child = _path(output, record["directory"])
    _pin(child / "_authority.json", record["authority_sha256"])
    _pin(child / "_manifest.json", record["manifest_sha256"])


def _result(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {"directory": str(output), "manifest_sha256": file_sha256(output / "_manifest.json"),
        "months": manifest["months"], "rows": manifest["rows"], "cohort_sha256": manifest["cohort_sha256"]}


def _month(value: str) -> None:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value) or not "2019-07" <= value <= "2024-05":
        raise DataReadinessError("monthly catalyst partition is outside initial fit")


def _decisions(root: Path, month: str, record: dict[str, Any]) -> pd.DataFrame:
    path = _path(root, record["path"])
    _pin(path, record["sha256"])
    _pin(manifest_path_for(path), record["manifest_sha256"])
    frame, _ = load_canonical_artifact(path, expected_type="decisions", allow_research=True, columns=_DECISION_COLUMNS)
    frame = stamp_canonical_decision_ids(frame)
    times = pd.to_datetime(frame.decision_time_utc, utc=True, errors="coerce")
    if (frame.empty or frame.decision_id.duplicated().any() or times.isna().any()
            or not times.dt.tz_convert("America/New_York").dt.strftime("%Y-%m").eq(month).all()
            or not times.between(pd.Timestamp("2019-07-09T00:00:00Z"), pd.Timestamp("2024-05-28T22:00:00Z")).all()):
        raise DataReadinessError("canonical decisions do not match their bounded monthly partition")
    return frame


def _check_keys(rows: pd.DataFrame, decisions: pd.DataFrame) -> None:
    if rows.empty:
        return
    joined = rows[_KEYS].merge(decisions[_KEYS], on=_KEYS, how="left", validate="many_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise DataReadinessError("monthly catalyst evidence differs from corrected canonical decision keys")


def publish_monthly_catalyst_authorities(*, root: Path, config_path: Path,
    expected_config_sha256: str, output_directory: Path,
    expected_existing_manifest_sha256: str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """One shared lease; load and release only one monthly authority at a time."""
    root = root.resolve()
    with heavy_job_lease("publish-monthly-initial-fit-catalyst-authorities", runtime_dir=root / "data/runtime"):
        assert_system_memory_available()
        return _publish(root, config_path, expected_config_sha256, output_directory,
            expected_existing_manifest_sha256=expected_existing_manifest_sha256, progress=progress)


def _publish(root: Path, config_path: Path, digest: str, output_directory: Path, *,
    expected_existing_manifest_sha256: str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from market_predictor.swing.catalyst_lineage import verify_completed_catalyst_lineage
    from market_predictor.swing.features.catalyst_decision_authority import (
        load_catalyst_decision_authority,
        publish_catalyst_decision_authority,
    )

    config_path = _path(root, str(config_path))
    _pin(config_path, digest)
    config = _object(config_path)
    if (set(config) != {"schema", "source_files", "months", "cohort_sha256", "rows"} or config["schema"] != CONFIG_SCHEMA
            or not isinstance(config["months"], dict) or not config["months"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(config["cohort_sha256"]))
            or type(config["rows"]) is not int or config["rows"] <= 0):
        raise DataReadinessError("monthly catalyst publication config is malformed")
    pins = config["source_files"]
    if not isinstance(pins, dict):
        raise DataReadinessError("monthly catalyst source pins must be an object")
    _check_sources(root, pins)
    source_files = {_path(root, name).relative_to(root).as_posix(): pin for name, pin in pins.items()}
    source_files[config_path.relative_to(root).as_posix()] = digest
    output = _path(root, str(output_directory))
    for month in config["months"]:
        _month(month)
    request = {"config_path": str(config_path), "config_sha256": digest, "config": config}
    request_sha256 = json_sha256(request)
    months: dict[str, Any] = {}
    if output.exists():
        if expected_existing_manifest_sha256 is None:
            raise DataReadinessError("monthly catalyst resume requires an external manifest/checkpoint pin")
        completed = (output / "_manifest.json").exists()
        saved_path = output / ("_manifest.json" if completed else "_checkpoint.json")
        _pin(saved_path, expected_existing_manifest_sha256)
        saved = _object(saved_path)
        if saved.get("request") != request or saved.get("request_sha256") != request_sha256:
            raise DataReadinessError("monthly catalyst resume request differs")
        if completed:
            load_monthly_catalyst_index(output, expected_manifest_sha256=expected_existing_manifest_sha256)
        elif (saved.get("schema") != INDEX_SCHEMA or saved.get("status") != "in_progress_research_only"
                or not isinstance(saved.get("months"), dict) or not set(saved["months"]).issubset(config["months"])):
            raise DataReadinessError("monthly catalyst checkpoint contract differs")
        _check_sources(root, saved["source_files"])
        source_files.update(saved["source_files"])
        for month, record in sorted(saved["months"].items()):
            _check_completed_month(output, month, record, config["months"][month])
            decisions = _decisions(root, month, record["decisions"])
            authority = load_catalyst_decision_authority(_path(output, record["directory"]))
            _check_keys(authority.decisions, decisions)
            if (record["catalyst_decision_rows"] != len(authority.decisions) or record["rows"] != len(decisions)
                    or record["decision_ids_sha256"] != json_sha256(sorted(decisions.decision_id.astype(str)))):
                raise DataReadinessError("monthly catalyst checkpoint row inventory differs")
            del decisions, authority
        if completed:
            return _result(output, saved)
        months.update(saved["months"])
    else:
        if expected_existing_manifest_sha256 is not None:
            raise DataReadinessError("monthly catalyst resume directory is absent")
        output.mkdir(parents=True)

    def save_checkpoint() -> None:
        _checkpoint(output, {"schema": INDEX_SCHEMA, "status": "in_progress_research_only",
            "request": request, "request_sha256": request_sha256, "months": months, "source_files": source_files}, progress)

    save_checkpoint()
    for month, record in sorted(config["months"].items()):
        if month in months:
            continue
        assert_system_memory_available()
        decisions = _decisions(root, month, record["decisions"])
        decision_ids_sha256 = json_sha256(sorted(decisions.decision_id.astype(str)))
        if record["rows"] != len(decisions) or record["decision_ids_sha256"] != decision_ids_sha256:
            raise DataReadinessError("canonical monthly decision population differs from its independent inventory")
        decision_path = _path(root, record["decisions"]["path"])
        source_files[decision_path.relative_to(root).as_posix()] = record["decisions"]["sha256"]
        source_files[manifest_path_for(decision_path).relative_to(root).as_posix()] = record["decisions"]["manifest_sha256"]
        lineages = record["lineages"]
        if not isinstance(lineages, list) or not lineages:
            raise DataReadinessError("each month requires its pinned source lineages")
        directories: list[Path] = []
        for lineage in lineages:
            directory = _path(root, lineage["directory"])
            _pin(directory / "_manifest.json", lineage["manifest_sha256"])
            source_files[(directory / "_manifest.json").relative_to(root).as_posix()] = lineage["manifest_sha256"]
            verified = verify_completed_catalyst_lineage(directory)
            if verified.request["decisions_sha256"] != record["decisions"]["sha256"]:
                raise DataReadinessError("lineage is not keyed to this canonical monthly decision partition")
            _lineage_source_pins(root, lineage, verified.request, source_files)
            children = verified.manifest["artifacts"]
            if not isinstance(children, list) or any(not isinstance(child, dict) for child in children):
                raise DataReadinessError("monthly lineage artifact inventory is malformed")
            for child in children:
                assert_system_memory_available()
                event_path = _path(directory, child["event_path"])
                clocks, _ = load_canonical_artifact(event_path, expected_type="catalyst_events", allow_research=True,
                    columns=["feature_available_at_utc"])
                times = pd.to_datetime(clocks.feature_available_at_utc, utc=True, errors="coerce")
                if times.isna().any() or times.gt(pd.Timestamp("2024-05-28T22:00:00Z")).any():
                    raise DataReadinessError("lineage event numeric projection would cross initial fit")
                assignment_path = _path(directory, child["assignment_path"])
                assignments, _ = load_canonical_artifact(assignment_path, expected_type="catalyst_event_assignments",
                    allow_research=True, columns=[*_KEYS, "status"])
                _check_keys(assignments.loc[assignments.status.eq("assigned")], decisions)
            directories.append(directory)
        child_name = month if not (output / month).exists() else f"{month}.retry-{uuid4().hex}"
        child_directory = output / child_name
        authority = publish_catalyst_decision_authority(directories, child_directory,
            canonical_decisions={"path": str(decision_path.resolve()), "sha256": record["decisions"]["sha256"],
                "manifest_sha256": record["decisions"]["manifest_sha256"]})
        _check_keys(authority.decisions, decisions)
        months[month] = {"directory": child_name, "authority_sha256": file_sha256(child_directory / "_authority.json"),
            "manifest_sha256": file_sha256(child_directory / "_manifest.json"), "decisions": record["decisions"],
            "rows": len(decisions), "decision_ids_sha256": decision_ids_sha256,
            "catalyst_decision_rows": len(authority.decisions),
            "lineages": lineages}
        del authority, decisions
        _pin(config_path, digest)
        _check_sources(root, pins)
        save_checkpoint()
    rows = sum(m["rows"] for m in months.values())
    if rows != config["rows"]:
        raise DataReadinessError("monthly catalyst index lost canonical decision rows")
    _check_sources(root, source_files)
    manifest = {"schema": INDEX_SCHEMA, "status": "complete_research_only", "production_ready": False,
        "source_coverage_admitted": False, "scope_policy": "observed_articles_not_coverage_admission",
        "cohort_sha256": config["cohort_sha256"], "source_files": source_files, "rows": rows,
        "request": request,
        "months": months,
        "missing_value_policy": "existing_authority_semantics_keep_all_canonical_decisions_in_final_join"}
    manifest["request_sha256"] = json_sha256(manifest["request"])
    pending = output / f".manifest-{uuid4().hex}.pending"
    try:
        with pending.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        pending.rename(output / "_manifest.json")
    finally:
        pending.unlink(missing_ok=True)
    return _result(output, manifest)


def load_monthly_catalyst_index(directory: Path, *, expected_manifest_sha256: str) -> dict[str, Any]:
    """Read only the small index and external monthly authority pins, not Parquet."""
    _pin(directory / "_manifest.json", expected_manifest_sha256)
    manifest = _object(directory / "_manifest.json")
    request = manifest.get("request")
    if not isinstance(request, dict):
        raise DataReadinessError("monthly catalyst index request is malformed")
    if (manifest.get("schema") != INDEX_SCHEMA or manifest.get("status") != "complete_research_only"
            or manifest.get("production_ready") is not False or manifest.get("source_coverage_admitted") is not False
            or manifest.get("request_sha256") != json_sha256(request)):
        raise DataReadinessError("monthly catalyst index contract differs")
    config = manifest["request"]["config"]
    if (set(manifest["months"]) != set(config["months"]) or manifest["rows"] != config["rows"]
            or manifest["cohort_sha256"] != config["cohort_sha256"]
            or sum(row["rows"] for row in manifest["months"].values()) != manifest["rows"]):
        raise DataReadinessError("monthly catalyst index population differs")
    for month, record in manifest["months"].items():
        _month(month)
        _check_completed_month(directory, month, record, config["months"][month])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-existing-manifest-sha256")
    args = parser.parse_args()
    result = publish_monthly_catalyst_authorities(root=args.root, config_path=args.config,
        expected_config_sha256=args.expected_config_sha256, output_directory=args.out_dir,
        expected_existing_manifest_sha256=args.expected_existing_manifest_sha256,
        progress=lambda record: print(json.dumps(record, sort_keys=True), flush=True))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
