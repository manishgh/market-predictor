"""Guarded CLI for the two corrected, consistently adjusted predictor histories."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import typer

from market_predictor.canonical.store import file_sha256
from market_predictor.config import get_settings
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.io import inside
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError, heavy_job_lease
from market_predictor.resources import assert_memory_budget
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.swing.datasets.feature_history_plan import feature_history_requirements, publish_feature_history_plan
from market_predictor.swing.datasets.history_archive import (
    AlpacaSwingDailyPageSource,
    SwingDailyPage,
    SwingDailyPageSource,
    collect_swing_history_plan,
    load_complete_swing_history_collection,
)
from market_predictor.swing.datasets.symbol_corrections import pinned_object


def _guard() -> None:
    assert_memory_budget(stage="corrected adjusted feature history", hard_budget_gib=4.0, headroom_gib=0.75)
    assert_system_memory_available(minimum_available_gib=2.0)


class _GuardedDailySource(AlpacaSwingDailyPageSource):
    def fetch_daily_page(self, symbol: str, start: datetime, end_exclusive: datetime, *,
        page_token: str | None, asof: date, adjustment: str,
    ) -> SwingDailyPage:
        _guard()
        result = super().fetch_daily_page(symbol, start, end_exclusive,
            page_token=page_token, asof=asof, adjustment=adjustment)
        _guard()
        return result


def register_feature_history_commands(app: typer.Typer) -> None:
    @app.command("collect-swing-corrected-feature-history")
    def corrected_feature_history_command(
        root: Path = typer.Option(Path(".")),
        config: Path = typer.Option(Path("configs/swing_corrected_feature_history.toml")),
        expected_policy_sha256: str = typer.Option(...),
        plan_dir: Path = typer.Option(...),
        out_dir: Path | None = typer.Option(None),
        expected_plan_sha256: str | None = typer.Option(None),
        publish_plan: bool = typer.Option(False),
        offline: bool = typer.Option(False),
    ) -> None:
        """Plan, collect one resumable unit, or verify corrected SIP feature sources."""
        root = root.resolve()
        plan = inside(root, plan_dir)
        if not plan.is_relative_to(root / "data" / "research"):
            raise typer.BadParameter("Feature history plans must be beneath data/research")
        if publish_plan:
            if offline or out_dir is not None or expected_plan_sha256 is not None:
                raise typer.BadParameter("Plan publication cannot collect or replay an archive")
        elif out_dir is None or expected_plan_sha256 is None:
            raise typer.BadParameter("Collection/replay needs the archive path and independent plan pin")
        sources: list[AlpacaSource] = []
        try:
            with heavy_job_lease("collect-swing-corrected-feature-history", runtime_dir=root / "data/runtime"):
                _guard()
                request, _, _ = feature_history_requirements(root, config, expected_policy_sha256)
                if publish_plan:
                    publish_feature_history_plan(root, config, expected_policy_sha256, plan)
                    typer.echo({"status": "plan_published", "plan_sha256": file_sha256(plan / "_authority.json")})
                    return
                assert expected_plan_sha256 is not None and out_dir is not None
                authority = pinned_object(plan / "_authority.json", expected_plan_sha256)
                if pinned_object(plan / "_request.json", authority["request_sha256"]) != request:
                    raise DataReadinessError("Feature history plan differs from the pinned policy")
                output = inside(root, out_dir)
                protected = [plan, inside(root, config)]
                correction = request["reviewed_correction_policy"]
                protected.extend(inside(root, correction[key]) for key in (
                    "parent_plan", "parent_archive", "document_inventory", "document_archive"))
                if (not output.is_relative_to(root / "data/raw") or output == root / "data/raw"
                        or any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected)):
                    raise DataReadinessError("Feature archive must be separate from protected sources beneath data/raw")
                if offline:
                    result = load_complete_swing_history_collection(output, plan_directory=plan,
                        expected_adjustment="all", expected_plan_authority_sha256=expected_plan_sha256)
                else:
                    settings = get_settings()
                    if not settings.has_alpaca or settings.alpaca_stock_feed.strip().lower() != "sip":
                        raise typer.BadParameter("Configured Alpaca credentials and SIP are required")

                    def factory() -> SwingDailyPageSource:
                        source = AlpacaSource(settings)
                        sources.append(source)
                        return _GuardedDailySource(source)

                    result = collect_swing_history_plan(plan_directory=plan, output_directory=output,
                        source_factory=factory, provider_symbol_for=lambda ticker: str(request["provider_symbols"][ticker]),
                        maximum_units_this_run=1, expected_plan_authority_sha256=expected_plan_sha256)
                _guard()
                typer.echo({key: result[key] for key in ("status", "requested_units", "observed_units", "failed_units")
                    if key in result})
                if result.get("status") not in {"complete", "complete_with_unavailable"}:
                    raise typer.Exit(code=2)
        except HeavyJobBusyError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from exc
        finally:
            for source in sources:
                source.client.session.close()
