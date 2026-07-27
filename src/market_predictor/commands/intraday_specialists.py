"""KS4 intraday specialist research commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from market_predictor.config import get_settings
from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.intraday.specialist_collection import (
    build_intraday_specialist_acquisition_units,
    collect_intraday_specialist_one_minute,
)
from market_predictor.intraday.specialist_dataset import (
    build_intraday_specialist_collection_plan,
    build_intraday_specialist_setup_bundle,
)
from market_predictor.sources.alpaca import AlpacaSource


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

    @app.command("build-intraday-specialist-acquisition-units")
    @serialized_heavy_job(
        "build-intraday-specialist-acquisition-units"
    )
    def build_intraday_specialist_units(
        collection_plan_dir: Path = typer.Option(
            ...,
            help="Completed immutable KS4 one-minute collection plan.",
        ),
        policy: Path = typer.Option(
            Path("configs/intraday_specialist_research.toml"),
            help="Frozen KS4 specialist policy.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="New immutable multi-symbol Alpaca unit bundle.",
        ),
    ) -> None:
        """Build bounded full-session, multi-symbol Alpaca request units."""

        report = build_intraday_specialist_acquisition_units(
            collection_plan_directory=collection_plan_dir,
            policy_path=policy,
            output_directory=out_dir,
        )
        summary = report["summary"]
        console.print(
            "Built "
            f"{summary['units']:,} Alpaca units for "
            f"{summary['ticker_sessions']:,} ticker-sessions and at most "
            f"{summary['maximum_expected_rows']:,} one-minute rows."
        )

    @app.command("collect-intraday-specialist-one-minute")
    @serialized_heavy_job("collect-intraday-specialist-one-minute")
    def collect_intraday_specialist_one_minute_command(
        acquisition_units_dir: Path = typer.Option(
            ...,
            help="Completed immutable KS4 Alpaca acquisition units.",
        ),
        policy: Path = typer.Option(
            Path("configs/intraday_specialist_research.toml"),
            help="Frozen KS4 specialist policy.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="Resumable KS4 one-minute collection directory.",
        ),
    ) -> None:
        """Collect hash-audited Alpaca SIP one-minute unit artifacts."""

        settings = get_settings()
        report = collect_intraday_specialist_one_minute(
            acquisition_units_directory=acquisition_units_dir,
            policy_path=policy,
            output_directory=out_dir,
            source_factory=lambda: AlpacaSource(settings),
        )
        console.print(
            f"KS4 one-minute collection status={report['status']} "
            f"completed={report['completed_units']:,}/"
            f"{report['requested_units']:,}."
        )
