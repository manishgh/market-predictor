"""Immutable, bounded publication of original-query content inspection results."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.issuer_events.content_inventory import inspect_saved_alpaca_content
from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.symbols import canonical_symbol, normalized_ticker
from market_predictor.core.system_memory import system_memory_snapshot
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import assert_memory_budget
from market_predictor.swing.datasets.initial_fit_issuer_news import LAST_INITIAL_FIT_CUTOFF
from market_predictor.swing.datasets.issuer_news_preparation import FIRST

# Modules whose code decides acceptance or recorded values. The relationship
# import-closure pinner is itself hash-bound to closed evidence, so it is not reused.
IMPLEMENTATION_PATHS = (
    "research/issuer_content_inventory.py", "catalysts/issuer_events/content_inventory.py",
    "catalysts/issuer_events/news_history_contracts.py", "catalysts/issuer_events/news_query_scope.py",
    "canonical/normalize.py", "canonical/store.py", "data_quality.py", "core/json_integrity.py",
    "core/symbols.py", "evidence/io.py", "evidence/hashing.py",
    "swing/datasets/initial_fit_issuer_news.py", "swing/datasets/issuer_news_preparation.py",
)


def _guard() -> None:
    assert_memory_budget(stage="issuer content inventory", hard_budget_gib=5.0, headroom_gib=0.75)
    snapshot = system_memory_snapshot()
    if snapshot is None or snapshot.used_percent >= 90.0:
        raise MemoryBudgetError("issuer content inventory requires system memory below 90 percent")


def publish_saved_content_inventory(
    *, root: Path, event_artifact: SourcePin, event_manifest: SourcePin,
    security_id: str, ticker: str, start_utc: pd.Timestamp, cutoff_utc: pd.Timestamp, output: Path,
) -> dict[str, Any]:
    """Inspect one explicit original query, never infer cohort/source completeness.

    The configured shared lease is acquired before any source input is loaded.
    Only a new direct child of data/research may be published. Partial files are
    staged beside it and never advertised as a completed authority.
    """
    root = root.resolve()
    output = inside(root, output)
    if output.parent != root / "data/research":
        raise DataReadinessError("content inventory output must be a new direct child of data/research")
    for value in (start_utc, cutoff_utc):
        if not isinstance(value, pd.Timestamp) or pd.isna(value) or value.tzinfo is None:
            raise DataReadinessError("content inventory requires timezone-aware cutoffs")
    if not FIRST <= start_utc <= cutoff_utc <= LAST_INITIAL_FIT_CUTOFF:
        raise DataReadinessError("content inventory is restricted to the initial-fit issuer-news window")
    query_ticker = canonical_symbol(normalized_ticker(ticker))
    pins = {name: pin.model_copy(update={"path": inside(root, pin.path).relative_to(root).as_posix()})
            for name, pin in (("event_artifact", event_artifact), ("event_manifest", event_manifest))}
    if any((root / pin.path).is_relative_to(output) for pin in pins.values()):
        raise DataReadinessError("content inventory output overlaps source inputs")
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease("inspect-saved-issuer-content", runtime_dir=runtime):
        _guard()
        if output.exists():
            raise FileExistsError(f"immutable content inventory already exists: {output}")
        rows, summary = inspect_saved_alpaca_content(
            root=root, event_artifact=event_artifact, event_manifest=event_manifest,
            security_id=security_id, ticker=ticker, start_utc=start_utc, cutoff_utc=cutoff_utc, memory_check=_guard,
        )
        _guard()
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = output.with_name(f".{output.name}.{uuid4().hex}.pending")
        staging.mkdir()
        try:
            records = staging / "records.parquet"
            rows.to_parquet(records, index=False)
            # Record the implementation used for this inspection, not archived code.
            package = Path(__file__).resolve().parents[1]
            implementation = {f"market_predictor/{name}": file_sha256(package / name) for name in IMPLEMENTATION_PATHS}
            report = {
                "schema": "market_predictor.saved_issuer_content_inventory",
                "status": "complete_inventory_only",
                "scope": "single_original_query_chunk_not_target_cohort_coverage",
                **{name: pin.model_dump() for name, pin in pins.items()},
                "query_security_id": security_id, "query_ticker": query_ticker,
                "start_utc": start_utc.isoformat(), "cutoff_utc": cutoff_utc.isoformat(),
                "rows": len(rows), "records_path": "records.parquet", "records_sha256": file_sha256(records),
                "implementation_files": implementation, "inspection": summary,
                "content_qualification": "not_established", "issuer_attribution": "not_established_by_inventory",
                "training_eligible": False, "promotion_eligible": False, "serving_eligible": False,
            }
            write_json_object(staging / "_manifest.json", report)
            _guard()
            staging.rename(output)
        finally:
            if staging.exists():
                for name in ("records.parquet", "_manifest.json"):
                    (staging / name).unlink(missing_ok=True)
                staging.rmdir()
        return {**report, "manifest_sha256": file_sha256(output / "_manifest.json")}
