"""Rebuild daily history publications offline with the canonical collector."""
from __future__ import annotations

import argparse
import json
import logging
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.sources.alpaca import decode_bars_page_response
from market_predictor.sources.http import HttpByteResponse
from market_predictor.swing.datasets import history_archive as archive

LOGGER = logging.getLogger(__name__)
MAX_BODY = 32 * 1024**2


def _object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 8 * 1024**2:
        raise DataReadinessError("retained history metadata exceeds bounded size")
    return archive._load_json(path)


class _RetainedPages:
    """An immutable query index; consumption tracking is protected across workers."""

    def __init__(self, source: Path, pages: dict[str, tuple[dict[str, Any], dict[str, Any]]]) -> None:
        self.source, self.pages = source, pages
        self.used: set[str] = set()
        self.lock = threading.Lock()

    def fetch_daily_page(self, symbol: str, start: datetime, end_exclusive: datetime, *,
                         page_token: str | None, asof: date, adjustment: str) -> archive.SwingDailyPage:
        params: dict[str, Any] = {
            "symbols": symbol, "start": start.isoformat(),
            "end": (end_exclusive - timedelta(microseconds=1)).isoformat(),
            "asof": asof.isoformat(), "adjustment": adjustment,
            "timeframe": archive.TIMEFRAME, "feed": archive.PRICE_FEED, "limit": 10_000, "sort": "asc",
        }
        if page_token is not None:
            params["page_token"] = page_token
        key = archive._json_sha256(params)
        with self.lock:
            if key not in self.pages or key in self.used:
                raise DataReadinessError("retained query is missing, ambiguous or consumed twice")
            self.used.add(key)
            completed = len(self.used)
        if completed % 100 == 0:
            LOGGER.info("Decoded %d/%d retained history pages", completed, len(self.pages))
        _, page = self.pages[key]
        transport = page["transport"]
        metadata = transport["metadata"]
        path = archive._resolve_inside(self.source, transport["body_path"])
        if path.stat().st_size > MAX_BODY:
            raise DataReadinessError("retained provider body exceeds bounded size")
        with path.open("rb") as handle:
            body = handle.read(MAX_BODY + 1)
        if len(body) > MAX_BODY:
            raise DataReadinessError("retained provider body exceeds bounded size")
        response = HttpByteResponse(**{
            **metadata, "body": body, "retrieved_at_utc": datetime.fromisoformat(metadata["retrieved_at_utc"]),
            "redirect_chain": tuple(metadata["redirect_chain"]),
            "safe_headers": tuple(tuple(pair) for pair in metadata["safe_headers"]),
        })
        decoded = decode_bars_page_response(response, expected_params=params)
        return archive.SwingDailyPage(
            request_page_token=decoded.request_page_token, next_page_token=decoded.next_page_token,
            response_symbol=symbol, response_timeframe=archive.TIMEFRAME, response_feed=archive.PRICE_FEED,
            response_adjustment=adjustment, bars=decoded.bars.get(symbol, ()),
            response_headers=decoded.response_headers, raw_payload=decoded.raw_payload, transport_response=response,
        )


def reconstruct_history(*, root: Path, source_directory: Path, source_authority_sha256: str,
                        plan_directory: Path, plan_authority_sha256: str, output_directory: Path) -> dict[str, Any]:
    """Publish fresh derived artifacts; retain source bytes and observation clocks."""
    root = root.resolve()
    source = (root / source_directory).resolve()
    plan_directory = (root / plan_directory).resolve()
    output = (root / output_directory).resolve()
    if (not output.is_relative_to(root) or output == root or output.exists()
            or output.is_relative_to(source) or source.is_relative_to(output)
            or output.is_relative_to(plan_directory) or plan_directory.is_relative_to(output)):
        raise DataReadinessError("history reconstruction requires a separate nonexistent workspace output")
    with heavy_job_lease("daily-history-offline-reconstruction", runtime_dir=root / heavy_job_runtime_dir()):
        archive._guard("daily history reconstruction")
        authority_path = source / "_authority.json"
        if file_sha256(authority_path) != source_authority_sha256:
            raise DataReadinessError("original collection authority hash differs")
        pins = {authority_path: source_authority_sha256}

        def pinned(path: Path, expected: str | None = None) -> dict[str, Any]:
            digest = file_sha256(path)
            if expected is not None and digest != expected:
                raise DataReadinessError("original history artifact hash differs")
            pins[path] = digest
            return _object(path)

        authority = pinned(authority_path, source_authority_sha256)
        retained = pinned(source / "_request.json")
        manifest = pinned(source / "_manifest.json", authority.get("artifact_sha256"))
        retained_hash = retained.get("request_sha256")
        if (authority.get("state") != "complete" or authority.get("artifact") != "_manifest.json"
                or retained_hash != archive._json_sha256({k: v for k, v in retained.items() if k != "request_sha256"})
                or authority.get("request_sha256") != retained_hash or manifest.get("request_sha256") != retained_hash
                or manifest.get("status") not in {"complete", "complete_with_unavailable"}
                or manifest.get("failed_units") != [] or manifest.get("unattempted_units") != []):
            raise DataReadinessError("original history lacks a complete hash-bound collection")
        plan = archive._load_verified_plan(plan_directory, expected_plan_authority_sha256=plan_authority_sha256)
        units = archive._bind_provider_symbols(plan.units, lambda ticker: archive._provider_from_request(retained, ticker))
        if (retained.get("provider") != "alpaca" or retained.get("timeframe") != archive.TIMEFRAME
                or retained.get("price_feed") != archive.PRICE_FEED or retained.get("adjustment") != plan.adjustment
                or retained.get("transport_receipts_required") is not True
                or retained.get("universe_sha256") != plan.universe_sha256
                or retained.get("provider_unit_set_sha256") != archive._provider_unit_set_sha256(units)
                or plan.provider_symbols is not None and retained.get("provider_symbols") != plan.provider_symbols):
            raise DataReadinessError("retained provider query set differs from current plan")
        expected = {str(row["unit_id"]): row for row in units.to_dict(orient="records")}
        records = manifest.get("unit_artifacts")
        if (not isinstance(records, list) or len(records) != len(expected)
                or {r.get("unit_id") for r in records} != set(expected)
                or authority.get("unit_set_sha256") != archive._unit_artifact_set_sha256(records)
                or manifest.get("unit_set_sha256") != authority.get("unit_set_sha256")):
            raise DataReadinessError("retained history unit inventory differs")
        query_pages: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        source_units: list[dict[str, Any]] = []
        rows = 0
        for record in records:
            archive._guard("daily history retained page replay")
            unit = expected[record["unit_id"]]
            path = archive._resolve_inside(source, record["unit_manifest_path"])
            original = pinned(path, record["unit_manifest_sha256"])
            identity = {**archive._unit_identity_record(unit), "provider_symbol": unit["provider_symbol"],
                        "plan_unit_sha256": unit["plan_unit_sha256"], "request_sha256": retained_hash,
                        "timeframe": archive.TIMEFRAME, "price_feed": archive.PRICE_FEED, "adjustment": plan.adjustment}
            if any(original.get(k) != v for k, v in identity.items()):
                raise DataReadinessError("retained unit identity differs from current acquisition query")
            bars_path = archive._resolve_inside(source, original["bars_path"])
            if file_sha256(bars_path) != original["bars_sha256"]:
                raise DataReadinessError("original canonical bars hash differs")
            pins[bars_path] = original["bars_sha256"]
            response_rows = archive._verify_pages(source, original, adjustment=plan.adjustment, require_transport=True)
            if len(response_rows) != original["rows"] or original["rows"] != record["rows"]:
                raise DataReadinessError("retained response counters differ")
            rows += len(response_rows)
            del response_rows
            for page in original["pages"]:
                raw_path = archive._resolve_inside(source, page["raw_path"])
                body_path = archive._resolve_inside(source, page["transport"]["body_path"])
                pins[raw_path] = page["raw_sha256"]
                pins[body_path] = page["transport"]["metadata"]["sha256"]
                key = archive._json_sha256(archive._transport_params(unit, plan.adjustment, page["request_page_token"]))
                if key in query_pages:
                    raise DataReadinessError("retained query/page identity is ambiguous")
                query_pages[key] = (unit, page)
            source_units.append({"unit_id": unit["unit_id"], "manifest_sha256": pins[path]})
        if rows != manifest.get("total_rows"):
            raise DataReadinessError("original history total row count differs")
        staging = output.with_name(f".{output.name}.reconstruction-{uuid4().hex}")
        adapter = _RetainedPages(source, query_pages)
        rebuilt = archive.collect_swing_history_plan(
            plan_directory=plan_directory, output_directory=staging, source_factory=lambda: adapter,
            provider_symbol_for=lambda ticker: archive._provider_from_request(retained, ticker),
            expected_plan_authority_sha256=plan_authority_sha256,
        )
        if (rebuilt.get("status") not in {"complete", "complete_with_unavailable"}
                or rebuilt.get("total_rows") != rows or adapter.used != set(query_pages)):
            raise DataReadinessError("canonical reconstruction did not consume every complete retained page")
        archive.load_complete_swing_history_collection(
            staging, plan_directory=plan_directory, expected_adjustment=plan.adjustment,
            expected_plan_authority_sha256=plan_authority_sha256,
        )
        for path, digest in pins.items():
            if file_sha256(path) != digest:
                raise DataReadinessError("retained history evidence changed during reconstruction")
        archive._assert_plan_files_unchanged(plan_directory, plan.hashes)
        report = {
            "schema": "market_predictor.daily_history_reconstruction", "status": "verified",
            "source_root": str(source), "output_root": str(output),
            "source_authority_sha256": source_authority_sha256, "plan_authority_sha256": plan_authority_sha256,
            "source_units": source_units, "source_files": {str(p.relative_to(source)): h for p, h in pins.items()},
            "unit_count": len(expected), "pages": len(query_pages), "rows": rows,
            "provider_bytes_and_retrieval_clocks_preserved": True,
            "derived_ingestion_clock": "actual_reconstruction_time_not_original_provider_retrieval",
            "reconstructed_at_utc": datetime.now(UTC).isoformat(), "source_only": True, "training_ready": False,
        }
        archive._atomic_json(staging / "_reconstruction.json", report)
        if output.exists():
            raise FileExistsError("history reconstruction output appeared during verification")
        staging.rename(output)
        return {k: v for k, v in report.items() if k not in {"source_units", "source_files"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-authority-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-authority-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = reconstruct_history(root=args.root, source_directory=args.source,
                                 source_authority_sha256=args.source_authority_sha256, plan_directory=args.plan,
                                 plan_authority_sha256=args.plan_authority_sha256, output_directory=args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
