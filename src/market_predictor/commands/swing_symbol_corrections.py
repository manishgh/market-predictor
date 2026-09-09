"""CLI adapters for reviewed symbol corrections and deterministic source selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from market_predictor.canonical.store import file_sha256
from market_predictor.config import get_settings
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.swing_history_collection import (
    AlpacaSwingDailyPageSource,
    SwingDailyPageSource,
    collect_swing_history_plan,
    load_complete_swing_history_collection,
)
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.swing.datasets.initial_fit_raw_share_plan import verified_initial_fit_raw_share_plan
from market_predictor.swing.datasets.symbol_corrected_sources import (
    publish_or_verify_symbol_corrected_sources,
    reconstruct_symbol_corrected_sources,
)
from market_predictor.swing.datasets.symbol_corrections import (
    inside,
    load_symbol_correction_policy,
    pinned_object,
    publish_symbol_correction_plan,
)


def register_symbol_correction_commands(app: typer.Typer, console: Any) -> None:
    @app.command("collect-swing-symbol-corrections")
    def symbol_corrections_command(
        root: Path = typer.Option(Path(".")),
        config: Path = typer.Option(Path("configs/swing_symbol_corrections.toml")),
        parent_config: Path = typer.Option(Path("configs/swing_initial_fit_raw_share_plan.toml")),
        expected_policy_sha256: str = typer.Option(...),
        plan_dir: Path = typer.Option(...),
        out_dir: Path | None = typer.Option(None),
        expected_plan_sha256: str | None = typer.Option(None),
        publish_plan: bool = typer.Option(False),
        offline: bool = typer.Option(False),
        max_units: int | None = typer.Option(None, min=1, max=2),
        selection_file: Path | None = typer.Option(None),
        expected_archive_sha256: str | None = typer.Option(None),
        expected_selection_sha256: str | None = typer.Option(None),
    ) -> None:
        """Plan two corrections, collect/resume them, or replay their selected source segments."""
        root = root.resolve()
        # Bounded, pinned bootstrap locates the unchanged parent lease owner. All
        # source loading waits for that lease; recheck the policy inside it.
        policy = load_symbol_correction_policy(root, config, expected_policy_sha256)
        plan_path = inside(root, plan_dir)
        protected = [inside(root, value) for value in (
            policy.parent_plan, policy.parent_archive, policy.document_archive, policy.document_inventory, str(config))]
        destinations = [plan_path] + ([inside(root, out_dir)] if out_dir is not None else [])
        if selection_file is not None:
            destinations.append(inside(root, selection_file))
        for index, path in enumerate(destinations):
            if any(path.is_relative_to(other) or other.is_relative_to(path)
                for other in protected + destinations[:index]):
                raise DataReadinessError("correction outputs must not overlap inputs or one another")
        if publish_plan:
            if offline or out_dir is not None or expected_plan_sha256 is not None or selection_file is not None:
                raise typer.BadParameter("Plan publication cannot collect, replay or replace an existing plan")
        elif out_dir is None or expected_plan_sha256 is None:
            raise typer.BadParameter("Collection/replay requires output archive and independently saved plan pin")
        if selection_file is not None and (not offline or expected_archive_sha256 is None):
            raise typer.BadParameter("Source selection requires offline replay and independently saved archive pin")
        if expected_archive_sha256 is not None and not offline:
            raise typer.BadParameter("An expected archive pin applies only to completed offline replay")
        sources: list[AlpacaSource] = []
        try:
            with verified_initial_fit_raw_share_plan(root, parent_config, inside(root, policy.parent_plan),
                expected_plan_sha256=policy.parent_plan_sha256):
                if load_symbol_correction_policy(root, config, expected_policy_sha256) != policy:
                    raise DataReadinessError("symbol correction policy changed between bootstrap and lease")
                if publish_plan:
                    publish_symbol_correction_plan(root, config, expected_policy_sha256, plan_path,
                        load_complete_swing_history_collection)
                    result: dict[str, Any] = {"status": "plan_published", "planned_units": 2,
                        "plan_sha256": file_sha256(plan_path / "_authority.json")}
                else:
                    assert out_dir is not None and expected_plan_sha256 is not None
                    authority = pinned_object(plan_path / "_authority.json", expected_plan_sha256)
                    request = pinned_object(plan_path / "_request.json", authority["request_sha256"])
                    if (request["policy_sha256"] != expected_policy_sha256 or request["source_root"] != str(root)
                            or inside(root, request["policy_path"]) != inside(root, config)):
                        raise DataReadinessError("CLI policy/root differs from the independently pinned correction plan")
                    archive = inside(root, out_dir)
                    if offline:
                        if expected_archive_sha256 is not None:
                            pinned_object(archive / "_authority.json", expected_archive_sha256)
                        if selection_file is not None:
                            assert expected_archive_sha256 is not None
                            result = reconstruct_symbol_corrected_sources(root=root, correction_plan=plan_path,
                                correction_plan_sha256=expected_plan_sha256, correction_archive=archive,
                                correction_archive_sha256=expected_archive_sha256, loader=load_complete_swing_history_collection)
                            publish_or_verify_symbol_corrected_sources(inside(root, selection_file), result,
                                expected_sha256=expected_selection_sha256)
                            result = {key: value for key, value in result.items() if key != "segments"}
                        else:
                            result = load_complete_swing_history_collection(archive, plan_directory=plan_path,
                                expected_adjustment="raw", expected_plan_authority_sha256=expected_plan_sha256)
                        if expected_archive_sha256 is not None:
                            pinned_object(archive / "_authority.json", expected_archive_sha256)
                    else:
                        settings = get_settings()
                        if not settings.has_alpaca or settings.alpaca_stock_feed.strip().lower() != "sip":
                            raise typer.BadParameter("Configured Alpaca credentials and SIP are required")

                        def source_factory() -> SwingDailyPageSource:
                            source = AlpacaSource(settings)
                            sources.append(source)
                            return AlpacaSwingDailyPageSource(source)

                        mapping = {c.ticker: c.provider_symbol for c in policy.corrections}
                        result = collect_swing_history_plan(plan_directory=plan_path, output_directory=archive,
                            source_factory=source_factory, provider_symbol_for=mapping.__getitem__,
                            expected_plan_authority_sha256=expected_plan_sha256, maximum_units_this_run=max_units)
                    result = {key: value for key, value in result.items() if key not in {"unit_artifacts", "unavailable_units"}}
        except HeavyJobBusyError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from exc
        finally:
            for source in sources:
                source.client.session.close()
        console.print(result)
        if result.get("status") == "incomplete":
            raise typer.Exit(code=2)
