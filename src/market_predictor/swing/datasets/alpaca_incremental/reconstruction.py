"""Explicit offline reconstruction of receipts from retained exact provider pages.

This is not a collection fallback or a historical-schema reader. The retained
request is evidence of query semantics; every page passes the current validator.
Only a complete, separately staged publication can become the new archive root.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from market_predictor.canonical.store import file_sha256
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.swing.datasets.alpaca_incremental import collector
from market_predictor.swing.datasets.alpaca_incremental.config import FAMILIES, Config, load_config, read_object
from market_predictor.swing.datasets.alpaca_incremental.pages import Unit
from market_predictor.swing.datasets.alpaca_incremental.storage import IntegrityError, publish, verified

LOGGER = logging.getLogger(__name__)


def _units(config: Config, request: dict[str, Any], source: Path, through: date,
           cutoff: datetime, pins: dict[Path, str]) -> tuple[dict[str, Unit], list[tuple[Path, dict[str, Any]]]]:
    symbols = tuple(request["symbols"])
    batches = [symbols[i:i + config.batch_size] for i in range(0, len(symbols), config.batch_size)]
    units: dict[str, Unit] = {}
    captures: list[tuple[Path, dict[str, Any]]] = []

    def add(day: date, capture: str = "daily", partial: datetime | None = None) -> None:
        for batch in batches:
            for family in FAMILIES:
                unit = Unit(day, family, batch, capture, partial)
                if unit.key in units:
                    raise IntegrityError("duplicate retained capture")
                units[unit.key] = unit

    day = config.start
    while day <= through:
        add(day)
        day += timedelta(days=1)
    for kind in ("revision_captures", "captures"):
        directory = source / kind
        for path in sorted(directory.iterdir()) if directory.exists() else ():
            if not path.is_file() or path.suffix != ".json":
                raise IntegrityError("unexpected retained capture file")
            pins[path] = file_sha256(path)
            record = verified(path)
            if record.get("request_sha256") != request["request_sha256"]:
                raise IntegrityError("retained capture request differs")
            if kind == "revision_captures":
                day = date.fromisoformat(record["day"])
                capture_date = date.fromisoformat(record["capture_date"])
                if (set(record) != {"day", "capture_date", "request_sha256"}
                        or not config.start <= day <= through or not day < capture_date <= cutoff.date()
                        or (capture_date - day).days > config.revision_overlap_days
                        or path.name != f"{capture_date}-{day}.json"):
                    raise IntegrityError("invalid retained revision capture")
                add(day, f"revision-{capture_date}")
            else:
                observed = datetime.fromisoformat(record["cutoff"])
                if (set(record) != {"cutoff", "request_sha256"} or observed.utcoffset() != timedelta(0)
                        or not config.start <= observed.date() <= cutoff.date() or observed > cutoff
                        or observed.date() > through + timedelta(days=1)
                        or path.name != f"{observed.strftime('%Y%m%dT%H%M%S%fZ')}.json"):
                    raise IntegrityError("invalid retained partial capture")
                add(observed.date(), f"partial-{observed.isoformat()}", observed)
            captures.append((path, record))
    return units, captures


def retained_file_set(source: Path) -> tuple[list[str], list[str]]:
    """Snapshot all retained unit/capture evidence, including rejected attempts."""
    unit_keys = sorted(p.name for p in (source / "units").iterdir() if p.is_dir())
    files = ["request.json"]
    for name in ("units", "revision_captures", "captures"):
        files.extend(p.relative_to(source).as_posix() for p in (source / name).rglob("*") if p.is_file())
    return unit_keys, sorted(files)


def reconstruct(config_path: Path, source: Path, *, through: date, inventory: Path,
                inventory_sha256: str, root: Path | None = None,
                allow_empty_capture_intents: bool = False) -> dict[str, Any]:
    """Rebuild only an unchanged query's complete, unsplit retained archive.

Original failures/rejected attempts remain historical evidence at the source.
Output must not exist. On interruption staging is retained, without authority.
No network client is instantiated and no feature/target/model files are read.
"""
    root = (root or Path.cwd()).resolve()
    config_path = (root / config_path).resolve()
    source = (root / source).resolve()
    inventory = (root / inventory).resolve()
    with heavy_job_lease("alpaca-incremental-offline-reconstruction",
                         runtime_dir=root / heavy_job_runtime_dir(), config_path=config_path):
        collector.guard_memory()
        config_pin = file_sha256(config_path)
        config, output, request = load_config(config_path)
        cutoff = datetime.now(UTC)
        if through < config.start or through >= cutoff.date():
            raise ValueError("reconstruction through must be a completed UTC day")
        if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("reconstruction requires a separate nonexistent output")
        if file_sha256(inventory) != inventory_sha256:
            raise IntegrityError("frozen inventory hash differs")
        frozen = read_object(inventory)
        keys, files = retained_file_set(source)
        if (frozen.get("schema") != "market_predictor.retained_archive_inventory"
                or frozen.get("unit_keys") != keys or not isinstance(frozen.get("files"), dict)
                or sorted(frozen["files"]) != files):
            raise IntegrityError("retained evidence differs from the frozen file inventory")
        pins = {config_path: config_pin, inventory: inventory_sha256}
        LOGGER.info("Checking %d frozen source files", len(files))
        for relative, expected in frozen["files"].items():
            path = (source / relative).resolve()
            if not path.is_relative_to(source) or file_sha256(path) != expected:
                raise IntegrityError("retained evidence hash differs from frozen inventory")
            pins[path] = expected
        retained = verified(source / "request.json")
        retained_hash = retained.get("request_sha256")
        if retained_hash != json_sha256({k: v for k, v in retained.items() if k != "request_sha256"}):
            raise IntegrityError("retained request identity differs from its content")
        # No historical identity is accepted by the normal collection reader.
        # Query equivalence plus current page replay justify new derived receipts.
        query = {k: v for k, v in request.items() if k not in {"schema", "request_sha256"}}
        if query != {k: v for k, v in retained.items() if k not in {"schema", "request_sha256"}}:
            raise IntegrityError("retained query scope or collection policy differs")
        units, captures = _units(config, retained, source, through, cutoff, pins)
        directory = source / "units"
        observed_keys = {p.name for p in directory.iterdir() if p.is_dir()}
        missing = set(units) - observed_keys
        incomplete_captures: list[dict[str, Any]] = []
        completed_captures: list[tuple[Path, dict[str, Any]]] = []
        for path, record in captures:
            capture = (f"revision-{record['capture_date']}" if path.parent.name == "revision_captures"
                       else f"partial-{record['cutoff']}")
            capture_day = (date.fromisoformat(record["day"]) if path.parent.name == "revision_captures"
                           else datetime.fromisoformat(record["cutoff"]).date())
            scope = {key for key, unit in units.items() if unit.capture == capture and unit.day == capture_day}
            if allow_empty_capture_intents and scope and scope <= missing:
                # Collector writes capture intent before acquisition. An empty
                # attempt establishes no observed data or revision freshness.
                incomplete_captures.append({"path": str(path.relative_to(source)), "sha256": pins[path],
                                            "capture": capture, "missing_units": len(scope),
                                            "reason": "capture_intent_has_no_successful_unit_archives"})
                for key in scope:
                    del units[key]
            else:
                completed_captures.append((path, record))
        if observed_keys != set(units):
            raise IntegrityError("retained unit inventory has missing or unexpected units")
        staging = output.with_name(f".{output.name}.reconstruction-{uuid4().hex}")
        staging.mkdir(parents=True)
        publish(staging / "request.json", request)
        records: list[dict[str, Any]] = []
        pages = rows = 0
        for key, unit in sorted(units.items()):
            collector.guard_memory()
            origin = directory / key
            if (origin / "split.json").exists():
                raise IntegrityError("split trees require a separately frozen reconstruction scope")
            receipt_path = origin / "success.json"
            pins[receipt_path] = file_sha256(receipt_path)
            receipt = verified(receipt_path)
            collector._replay(origin, receipt, unit, config, retained_hash)
            archive = origin / receipt["archive"]
            pins[archive] = receipt["archive_sha256"]
            destination = staging / "units" / key
            destination.mkdir(parents=True)
            copied = destination / archive.name
            shutil.copyfile(archive, copied)
            if file_sha256(copied) != pins[archive]:
                raise IntegrityError("copied provider archive differs")
            reconstructed = {**receipt, "request_sha256": request["request_sha256"]}
            publish(destination / "success.json", reconstructed)
            records.append({"unit_key": key, "source_receipt_sha256": pins[receipt_path],
                            "archive": archive.name, "archive_sha256": pins[archive],
                            "receipt_sha256": file_sha256(destination / "success.json"),
                            "pages": receipt["pages"], "rows": receipt["rows"]})
            pages += receipt["pages"]
            rows += receipt["rows"]
            if len(records) % 500 == 0:
                LOGGER.info("Replayed and copied %d/%d archive units", len(records), len(units))
        for path, record in completed_captures:
            publish(staging / path.parent.name / path.name, {**record, "request_sha256": request["request_sha256"]})
        LOGGER.info("Verifying reconstructed daily/revision units with the current offline collector")
        summary = collector._run_loaded(config, staging, request, through, cutoff, None, True, None)
        if summary["status"] != "verified" or summary["complete_through_utc_date"] != through.isoformat():
            raise IntegrityError("reconstructed archive failed the current offline reader")
        # Detect source/config changes during the full replay, before publication.
        LOGGER.info("Rechecking all original source hashes before publication")
        if retained_file_set(source) != (keys, files):
            raise IntegrityError("retained file or capture inventory changed during replay")
        for path, expected in pins.items():
            if file_sha256(path) != expected:
                raise IntegrityError("reconstruction input changed during replay")
        _, current_output, current_request = load_config(config_path)
        if current_output != output or current_request != request:
            raise IntegrityError("reconstruction config inputs changed during replay")
        if retained_file_set(source) != (keys, files):
            raise IntegrityError("retained file or capture inventory changed during final hash checks")
        report: dict[str, Any] = {
            "schema": "market_predictor.alpaca_incremental_reconstruction",
            "status": "verified", "reconstruction_started_at_utc": cutoff.isoformat(),
            "reconstructed_at_utc": datetime.now(UTC).isoformat(),
            "source_root": str(source), "output_root": str(output),
            "source_request_file_sha256": pins[source / "request.json"],
            "source_request_sha256": retained_hash, "request_sha256": request["request_sha256"],
            "config_sha256": config_pin, "through": through.isoformat(),
            "input_inventory": {"path": str(inventory), "sha256": inventory_sha256},
            "units": records, "unit_count": len(records), "pages": pages, "rows": rows,
            "capture_sources": [{"path": str(p.relative_to(source)), "sha256": pins[p]} for p, _ in captures],
            "complete_capture_count": len(completed_captures), "incomplete_capture_intents": incomplete_captures,
            "raw_bytes_preserved": True, "retrieval_clocks_preserved": True,
            "source_only": True, "training_ready": False, "issuer_attribution": False,
            "active_membership_authority": False,
        }
        summary["status_path"] = str(output / "status.json")
        publish(staging / "status.json", summary)
        publish(staging / "_reconstruction.json", report)
        if output.exists():
            raise FileExistsError("reconstruction output appeared during replay")
        staging.rename(output)
        return {k: v for k, v in report.items() if k not in {"units", "capture_sources"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--through", type=date.fromisoformat, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--allow-empty-capture-intents", action="store_true",
                        help="Preserve completely empty revision/partial attempts as unavailable source evidence.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = reconstruct(args.config, args.source, through=args.through, inventory=args.inventory,
                         inventory_sha256=args.inventory_sha256, root=args.root,
                         allow_empty_capture_intents=args.allow_empty_capture_intents)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
