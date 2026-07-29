"""Edge-rebuild research audit commands."""

from pathlib import Path
from typing import Any

import typer

from market_predictor.config import get_settings
from market_predictor.edge_rebuild.contracts import (
    load_edge_rebuild_readiness_config,
)
from market_predictor.edge_rebuild.extended_session_context import (
    build_extended_session_context_plan,
)
from market_predictor.edge_rebuild.history_collection import (
    collect_intraday_history,
)
from market_predictor.edge_rebuild.history_contracts import (
    load_extended_session_context_config,
    load_intraday_history_config,
)
from market_predictor.edge_rebuild.intraday_history import (
    build_intraday_history_plan,
)
from market_predictor.edge_rebuild.readiness import (
    run_edge_rebuild_readiness_audit,
)
from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.intraday.specialist_contracts import (
    load_intraday_specialist_research_config,
)
from market_predictor.sources.alpaca import AlpacaSource


def register_edge_rebuild_commands(app: typer.Typer, console: Any) -> None:
    @app.command("plan-edge-rebuild-extended-session-context")
    @serialized_heavy_job("plan-edge-rebuild-extended-session-context")
    def plan_edge_rebuild_extended_session_context(
        intraday_plan_dir: Path = typer.Option(...),
        intraday_collection_dir: Path = typer.Option(...),
        memberships: Path = typer.Option(...),
        membership_audit: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(
            Path("configs/edge_rebuild_extended_session_context.toml")
        ),
    ) -> None:
        """Plan the separate ER1B pre/post-market five-minute context layer."""

        result = build_extended_session_context_plan(
            intraday_plan_directory=intraday_plan_dir,
            intraday_collection_directory=intraday_collection_dir,
            memberships_path=memberships,
            membership_audit_path=membership_audit,
            policy_path=policy,
            output_directory=out_dir,
            config=load_extended_session_context_config(policy),
        )
        console.print(result["summary"])

    @app.command("collect-edge-rebuild-intraday-history")
    @serialized_heavy_job("collect-edge-rebuild-intraday-history")
    def collect_edge_rebuild_intraday_history(
        plan_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        max_units: int | None = typer.Option(
            None,
            min=1,
            help="Optional resumable operational batch limit.",
        ),
        policy: Path = typer.Option(
            Path("configs/edge_rebuild_intraday_history.toml"),
            help=(
                "ER1A regular-session policy, or the ER1B extended-session "
                "context policy when collecting that plan."
            ),
        ),
        extended_session_context: bool = typer.Option(
            False,
            help="Collect an ER1B extended-session context plan.",
        ),
    ) -> None:
        """Collect resumable PIT SIP five-minute ER1A or ER1B history."""

        settings = get_settings()
        config = (
            load_extended_session_context_config(policy)
            if extended_session_context
            else load_intraday_history_config(policy)
        )
        result = collect_intraday_history(
            plan_directory=plan_dir,
            policy_path=policy,
            output_directory=out_dir,
            config=config,
            source_factory=lambda: AlpacaSource(settings),
            maximum_units_this_run=max_units,
        )
        console.print(
            {
                "status": result["status"],
                "completed_units": result["completed_units"],
                "requested_units": result["requested_units"],
            }
        )

    @app.command("plan-edge-rebuild-intraday-history")
    @serialized_heavy_job("plan-edge-rebuild-intraday-history")
    def plan_edge_rebuild_intraday_history(
        readiness_audit_dir: Path = typer.Option(...),
        memberships: Path = typer.Option(...),
        membership_audit: Path = typer.Option(...),
        existing_stock_bars_dir: Path = typer.Option(...),
        existing_benchmark_bars_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(
            Path("configs/edge_rebuild_intraday_history.toml")
        ),
    ) -> None:
        """Plan causal PIT five-minute history before selective minute labels."""

        result = build_intraday_history_plan(
            readiness_audit_directory=readiness_audit_dir,
            memberships_path=memberships,
            membership_audit_path=membership_audit,
            existing_stock_bars_directory=existing_stock_bars_dir,
            existing_benchmark_bars_directory=(
                existing_benchmark_bars_dir
            ),
            policy_path=policy,
            output_directory=out_dir,
            config=load_intraday_history_config(policy),
        )
        console.print(result["summary"])

    @app.command("audit-edge-rebuild-readiness")
    @serialized_heavy_job("audit-edge-rebuild-readiness")
    def audit_edge_rebuild_readiness(
        swing_bundle_dir: Path = typer.Option(...),
        swing_technical_path: Path = typer.Option(...),
        intraday_training_dir: Path = typer.Option(...),
        intraday_collection_dir: Path = typer.Option(...),
        intraday_coverage_dir: Path = typer.Option(...),
        catalyst_lineage_dir: Path = typer.Option(...),
        news_source_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(
            Path("configs/edge_rebuild_readiness.toml")
        ),
        swing_policy: Path = typer.Option(
            Path("configs/swing_specialist_research.toml")
        ),
        intraday_policy: Path = typer.Option(
            Path("configs/intraday_specialist_research.toml")
        ),
    ) -> None:
        """Audit independent source capacity without fitting a model."""

        result = run_edge_rebuild_readiness_audit(
            swing_bundle_dir=swing_bundle_dir,
            swing_technical_path=swing_technical_path,
            intraday_training_dir=intraday_training_dir,
            intraday_collection_dir=intraday_collection_dir,
            intraday_coverage_dir=intraday_coverage_dir,
            catalyst_lineage_dir=catalyst_lineage_dir,
            news_source_dir=news_source_dir,
            out_dir=out_dir,
            config=load_edge_rebuild_readiness_config(policy),
            policy_path=policy,
            swing_policy_path=swing_policy,
            intraday_config=load_intraday_specialist_research_config(
                intraday_policy
            ),
            intraday_policy_path=intraday_policy,
        )
        console.print(result)
