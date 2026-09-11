"""Standalone source-only CLI. Exit 75 means resource pressure or a busy lease."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from market_predictor.heavy_jobs import HeavyJobBusyError
from market_predictor.swing.datasets.alpaca_incremental import collect
from market_predictor.swing.datasets.alpaca_incremental.storage import IntegrityError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded Alpaca source-only incremental archives")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Workspace root for the shared lease; default CWD.")
    parser.add_argument("--through", type=date.fromisoformat, help="UTC date; default yesterday. Today is partial only.")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = collect(args.config, through=args.through, max_units=args.max_units, offline=args.offline, root=args.root)
    except HeavyJobBusyError:
        result = {"status": "busy", "source_only": True}
    except IntegrityError:
        result = {"status": "integrity_failed", "source_only": True}
    except (ValueError, OSError):
        result = {"status": "invalid_config_or_io", "source_only": True}
    compact = {key: value for key, value in result.items() if key not in ("families", "revisions", "partial_today")}
    for scope in ("families", "revisions", "partial_today"):
        if scope in result:
            compact[scope] = {family: {key: value for key, value in counts.items() if key != "symbols"}
                for family, counts in result[scope].items()}
    print(json.dumps(compact, sort_keys=True, allow_nan=False))
    status = result["status"]
    if status in ("busy", "paused_memory"):
        return 75
    if status == "invalid_config_or_io":
        return 2
    return 0 if status in ("complete", "verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
