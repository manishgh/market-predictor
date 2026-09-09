"""Offline fixed-horizon compilation using the existing raw-plan lease owner."""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer

from market_predictor.edge_rebuild.swing_history_collection import load_complete_swing_history_collection
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError
from market_predictor.swing.contracts.holding_materialization import FixedHoldingBatch
from market_predictor.swing.datasets.holding_materialization import materialize_fixed_holdings
from market_predictor.swing.datasets.initial_fit_raw_share_plan import verified_initial_fit_raw_share_plan
from market_predictor.swing.datasets.symbol_corrected_sources import reconstruct_symbol_corrected_sources
from market_predictor.swing.datasets.symbol_corrections import inside, pinned_object


@contextmanager
def verified_holding_sources(root: Path, request_path: Path, request_sha256: str,
    parent_config: Path) -> Iterator[dict[str, Any]]:
    """Bounded metadata bootstrap precedes the lease; all numeric replay is inside it."""
    batch = FixedHoldingBatch.model_validate_json(json.dumps(pinned_object(request_path, request_sha256)))
    selection_path = inside(root, Path(batch.source_selection.path))
    selected = pinned_object(selection_path, batch.source_selection.sha256)
    plan = inside(root, Path(selected["correction_plan"]))
    authority = pinned_object(plan / "_authority.json", selected["correction_plan_sha256"])
    correction = pinned_object(plan / "_request.json", authority["request_sha256"])
    policy = correction["policy"]
    with verified_initial_fit_raw_share_plan(root, parent_config, inside(root, Path(policy["parent_plan"])),
        expected_plan_sha256=policy["parent_plan_sha256"]):
        pinned_object(request_path, request_sha256)
        pinned_object(selection_path, batch.source_selection.sha256)
        yield reconstruct_symbol_corrected_sources(root=root, correction_plan=plan,
            correction_plan_sha256=selected["correction_plan_sha256"],
            correction_archive=inside(root, Path(selected["correction_archive"])),
            correction_archive_sha256=selected["correction_archive_sha256"], loader=load_complete_swing_history_collection)


def register_holding_materialization_commands(app: typer.Typer, console: Any) -> None:
    @app.command("materialize-swing-fixed-holdings")
    def command(root: Path = typer.Option(Path(".")), request_file: Path = typer.Option(...),
        expected_request_sha256: str = typer.Option(...), output: Path = typer.Option(...),
        parent_config: Path = typer.Option(Path("configs/swing_initial_fit_raw_share_plan.toml")),
        expected_output_sha256: str | None = typer.Option(None)) -> None:
        """Compile reviewed source-bound requests; never infer managed exits or settlement."""
        root = root.resolve()
        request_path, out = inside(root, request_file), inside(root, output)
        try:
            result = materialize_fixed_holdings(root=root, request_path=request_path,
                request_sha256=expected_request_sha256, output=out, expected_output_sha256=expected_output_sha256,
                selection_context=lambda: verified_holding_sources(root, request_path, expected_request_sha256, parent_config))
        except HeavyJobBusyError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from exc
        rows = result["rows"]
        console.print({"status": result["status"], "requests": len(rows),
            "specifications": sum(row["specification"] is not None for row in rows),
            "label_eligible": False, "output": str(out), "audit_sha256": result["audit_sha256"]})
