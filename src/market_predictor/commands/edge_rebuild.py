"""Edge-rebuild research audit commands."""

from pathlib import Path
from typing import Any

import typer

from market_predictor.edge_rebuild.contracts import (
    load_edge_rebuild_readiness_config,
)
from market_predictor.edge_rebuild.readiness import (
    run_edge_rebuild_readiness_audit,
)
from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.intraday.specialist_contracts import (
    load_intraday_specialist_research_config,
)


def register_edge_rebuild_commands(app: typer.Typer, console: Any) -> None:
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
