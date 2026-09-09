"""Single immutable writer for exact-unit daily history acquisition plans."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.locking import file_lock
from market_predictor.resources import assert_peak_memory_budget, memory_audit

PLAN_SCHEMA = "edge_rebuild.swing_history_acquisition_plan.v2"
AUTHORITY_SCHEMA = "edge_rebuild.swing_history_acquisition_plan_authority.v2"
DAILY_BAR_UNITS_FILE = "daily_bar_units.csv"
UNIT_COLUMNS = ("security_id", "ticker", "start_date", "end_date", "role")


def publish_daily_history_plan(*, output: Path, request: dict[str, Any],
    manifest: dict[str, Any], units: pd.DataFrame) -> dict[str, Any]:
    """Publish supplied verified requirements together; never replace an authority."""
    if list(units.columns) != list(UNIT_COLUMNS) or units.empty or units.duplicated().any():
        raise DataReadinessError("history plan requires unique, ordered exact units")
    result = copy.deepcopy(manifest)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(output, timeout=0.0):
        if output.exists():
            raise DataReadinessError(f"history acquisition-plan output must be new: {output}")
        staging = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
        try:
            staging.mkdir()
            units_path = staging / DAILY_BAR_UNITS_FILE
            units.to_csv(units_path, index=False, lineterminator="\n")
            result["daily_bars"]["units_artifact"] = {
                "path": DAILY_BAR_UNITS_FILE, "bytes": units_path.stat().st_size,
                "sha256": file_sha256(units_path),
            }
            _write(staging / "_request.json", request)
            result["request_sha256"] = file_sha256(staging / "_request.json")
            assert_peak_memory_budget(hard_budget_gib=4.0, headroom_gib=0.75, stage="history plan publication")
            result["resources"] = memory_audit(hard_budget_gib=4.0, headroom_gib=0.75).to_record()
            _write(staging / "_manifest.json", result)
            _write(staging / "_authority.json", {
                "schema": AUTHORITY_SCHEMA, "state": "complete", "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(staging / "_manifest.json"),
                "request_sha256": result["request_sha256"],
                "units_sha256": result["daily_bars"]["units_artifact"]["sha256"],
                "universe_sha256": result["membership"]["universe_sha256"],
            })
            staging.rename(output)
            return result
        finally:
            if staging.is_dir():
                shutil.rmtree(staging)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
