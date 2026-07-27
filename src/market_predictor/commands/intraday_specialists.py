"""KS4 intraday specialist research commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.intraday.specialist_dataset import (
    build_intraday_specialist_collection_plan,
    build_intraday_specialist_setup_bundle,
)


def register_intraday_specialist_commands(
    app: typer.Typer,
    console: Console,
) -> None:
    @app.command("build-intraday-specialist-setups")
    @serialized_heavy_job("build-intraday-specialist-setups")
    def build_intraday_specialist_setups(
        technical_dir: Path = typer.Option(
            ...,
            help="Retained complete monthly V3 five-minute technical shards.",
        ),
        benchmark_dir: Path = typer.Option(
            ...,
            help="Audited SIP SPY, QQQ, and sector ETF five-minute bars.",
        ),
        policy: Path = typer.Option(
            Path("configs/intraday_specialist_research.toml"),
            help="Frozen KS4 specialist policy.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="New immutable KS4 setup bundle directory.",
        ),
    ) -> None:
        """Extract causal setups and selective SIP one-minute requirements."""

        report = build_intraday_specialist_setup_bundle(
            technical_directory=technical_dir,
            benchmark_directory=benchmark_dir,
            policy_path=policy,
            output_directory=out_dir,
        )
        summary = report["summary"]
        console.print(
            "Built "
            f"{summary['setups']:,} KS4 setups across "
            f"{summary['tickers']:,} tickers and "
            f"{summary['sessions']:,} sessions."
        )

    @app.command("plan-intraday-specialist-one-minute")
    @serialized_heavy_job("plan-intraday-specialist-one-minute")
    def plan_intraday_specialist_one_minute(
        setup_dir: Path = typer.Option(
            ...,
            help="Completed immutable KS4 setup bundle.",
        ),
        policy: Path = typer.Option(
            Path("configs/intraday_specialist_research.toml"),
            help="Frozen KS4 specialist policy.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="New immutable selective one-minute collection plan.",
        ),
    ) -> None:
        """Expand setups into exact regular-session SIP collection windows."""

        report = build_intraday_specialist_collection_plan(
            setup_directory=setup_dir,
            policy_path=policy,
            output_directory=out_dir,
        )
        summary = report["summary"]
        console.print(
            "Planned "
            f"{summary['one_minute_requirements']:,} requirements as "
            f"{summary['merged_collection_windows']:,} merged windows "
            f"for {summary['required_tickers']:,} tickers."
        )
