"""Primary V2 distributional research commands."""

from pathlib import Path
from typing import Any

import typer

from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.intraday.specialist_contracts import (
    load_intraday_specialist_research_config,
)
from market_predictor.primary_v2.contracts import (
    INTRADAY_V2_ID,
    SWING_V2_ID,
    load_primary_v2_research_config,
)
from market_predictor.primary_v2.experiments import (
    run_primary_v2_experiments,
)
from market_predictor.primary_v2.failure_attribution import (
    run_primary_v2_failure_attribution,
)
from market_predictor.primary_v2.failure_attribution_contracts import (
    load_failure_attribution_config,
)
from market_predictor.swing.specialist_contracts import (
    load_swing_specialist_research_config,
)


def register_primary_v2_commands(app: typer.Typer, console: Any) -> None:
    @app.command("train-primary-v2")
    @serialized_heavy_job("train-primary-v2")
    def train_primary_v2(
        strategy_id: str = typer.Option(
            ...,
            "--strategy-id",
            help=(
                "Exact V2 ID: SWING.CROSS_SECTIONAL_MOMENTUM.5D.V2 or "
                "INTRADAY.VWAP_REVERSION.30M.V2."
            ),
        ),
        source_dir: Path = typer.Option(
            ...,
            help="Verified KS3 or KS4 source dataset bundle.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="Immutable, resumable primary V2 output directory.",
        ),
        policy: Path = typer.Option(
            Path("configs/primary_strategy_v2.toml"),
            help="Frozen primary V2 research policy.",
        ),
        candidate_id: list[str] | None = typer.Option(
            None,
            "--candidate-id",
            help="Run selected exact candidate IDs; repeat for multiple.",
        ),
        swing_v1_policy: Path = typer.Option(
            Path("configs/swing_specialist_research.toml"),
            help="Frozen KS3 source policy.",
        ),
        intraday_v1_policy: Path = typer.Option(
            Path("configs/intraday_specialist_research.toml"),
            help="Frozen KS4 source policy.",
        ),
    ) -> None:
        """Evaluate one clearly named V2 strategy from exact V1 labels."""

        normalized = strategy_id.strip().upper()
        if normalized not in {SWING_V2_ID, INTRADAY_V2_ID}:
            raise typer.BadParameter(
                "strategy-id must be one of the two frozen primary V2 IDs"
            )
        result = run_primary_v2_experiments(
            strategy_id=normalized,
            source_dir=source_dir,
            out_dir=out_dir,
            config=load_primary_v2_research_config(policy),
            policy_path=policy,
            swing_v1_config=load_swing_specialist_research_config(
                swing_v1_policy
            ),
            swing_v1_policy_path=swing_v1_policy,
            intraday_v1_config=load_intraday_specialist_research_config(
                intraday_v1_policy
            ),
            intraday_v1_policy_path=intraday_v1_policy,
            candidate_ids=candidate_id,
            progress=console.print,
        )
        console.print(result)

    @app.command("audit-primary-v2-failures")
    @serialized_heavy_job("audit-primary-v2-failures")
    def audit_primary_v2_failures(
        strategy_id: str = typer.Option(
            ...,
            "--strategy-id",
            help=(
                "Exact V2 ID: SWING.CROSS_SECTIONAL_MOMENTUM.5D.V2 or "
                "INTRADAY.VWAP_REVERSION.30M.V2."
            ),
        ),
        v2_run_dir: Path = typer.Option(
            ...,
            help="Complete authoritative primary V2 run directory.",
        ),
        source_dir: Path = typer.Option(
            ...,
            help="Exact verified KS3 or KS4 source dataset bundle.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="Immutable failure-attribution output directory.",
        ),
        policy: Path = typer.Option(
            Path("configs/primary_v2_failure_attribution.toml"),
            help="Frozen failure-attribution policy.",
        ),
        primary_v2_policy: Path = typer.Option(
            Path("configs/primary_strategy_v2.toml"),
            help="Frozen primary V2 research policy.",
        ),
        swing_v1_policy: Path = typer.Option(
            Path("configs/swing_specialist_research.toml"),
            help="Frozen KS3 source policy.",
        ),
        intraday_v1_policy: Path = typer.Option(
            Path("configs/intraday_specialist_research.toml"),
            help="Frozen KS4 source policy.",
        ),
    ) -> None:
        """Attribute V2 failures and find independently replicated cohorts."""

        normalized = strategy_id.strip().upper()
        if normalized not in {SWING_V2_ID, INTRADAY_V2_ID}:
            raise typer.BadParameter(
                "strategy-id must be one of the two frozen primary V2 IDs"
            )
        result = run_primary_v2_failure_attribution(
            strategy_id=normalized,
            v2_run_dir=v2_run_dir,
            source_dir=source_dir,
            out_dir=out_dir,
            config=load_failure_attribution_config(policy),
            policy_path=policy,
            primary_v2_config=load_primary_v2_research_config(
                primary_v2_policy
            ),
            primary_v2_policy_path=primary_v2_policy,
            swing_v1_config=load_swing_specialist_research_config(
                swing_v1_policy
            ),
            swing_v1_policy_path=swing_v1_policy,
            intraday_v1_config=load_intraday_specialist_research_config(
                intraday_v1_policy
            ),
            intraday_v1_policy_path=intraday_v1_policy,
        )
        console.print(result)
