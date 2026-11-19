"""One appended JSON line per research-lane run: code identity, inputs, settings and headline numbers."""
from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRIAL_LOG_NAME = "trials.jsonl"


def git_identity(root: Path) -> dict[str, Any]:
    """Current commit and whether tracked or untracked files differ from it (None when git is unavailable)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True,
                                text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True,
                                text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def headline(scoreboard: dict[str, Any]) -> dict[str, Any]:
    """Mean rank correlation, its Newey-West t-statistic and the top-tenth spread per predictor and horizon."""
    result: dict[str, Any] = {}
    if "predictors" in scoreboard:
        for name, horizons in scoreboard["predictors"].items():
            result[name] = {
                horizon: {
                    "mean_rank_correlation": entry["rank_correlation"]["mean"],
                    "newey_west_t": entry["rank_correlation"]["t"],
                    "top_tenth_mean_spread_per_period_fraction": entry["top_tenth"]["mean_spread_per_period_fraction"],
                }
                for horizon, entry in horizons.items()
            }
    if "universe" in scoreboard:
        result["universe"] = {
            horizon: {
                "known_label_share": facts["known_label_share"],
                "average_stock_minus_spy_fraction": facts["average_stock_minus_spy"]["mean"],
                "average_stock_minus_qqq_fraction": facts["average_stock_minus_qqq"]["mean"],
            }
            for horizon, facts in scoreboard["universe"].items()
        }
    if "technical_models" in scoreboard:
        result["technical_models_summary"] = scoreboard["technical_models"]["summary"]
    if "swing_setups" in scoreboard:
        result["swing_setups_summary"] = scoreboard["swing_setups"]["reading_summary"]
    if "benchmark_tilt" in scoreboard:
        result["benchmark_tilt_summary"] = scoreboard["benchmark_tilt"]["reading_summary"]
    return result


def append_trial(log_path: Path, *, root: Path, command: str, output: Path, input_sha256: dict[str, str],
                 settings: dict[str, Any], scoreboard: dict[str, Any]) -> dict[str, Any]:
    record = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "output": output.as_posix(),
        "code": git_identity(root),
        "inputs": input_sha256,
        "settings": settings,
        "headline": headline(scoreboard),
        "label_kind": "research label (provider split- and dividend-adjusted prices)",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return record
