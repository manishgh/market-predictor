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
from market_predictor.intraday.specialist_contracts import (
    load_intraday_specialist_research_config,
)
from market_predictor.intraday.specialist_coverage import (
    build_intraday_specialist_coverage_audit,
)
from market_predictor.intraday.specialist_dataset import (
    build_intraday_specialist_collection_plan,
    build_intraday_specialist_setup_bundle,
)
from market_predictor.intraday.specialist_experiments import (
    train_intraday_specialist_experiments,
)
from market_predictor.intraday.specialist_training_data import (
    build_intraday_specialist_training_dataset,
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

    @app.command("audit-intraday-specialist-one-minute-coverage")
    @serialized_heavy_job(
        "audit-intraday-specialist-one-minute-coverage"
    )
    def audit_intraday_specialist_one_minute_coverage(
        collection_plan_dir: Path = typer.Option(
            ...,
            help="Completed immutable KS4 one-minute collection plan.",
        ),
        collection_dir: Path = typer.Option(
            ...,
            help="Transport-complete KS4 one-minute collection.",
        ),
        policy: Path = typer.Option(
            Path("configs/intraday_specialist_research.toml"),
            help="Frozen KS4 specialist policy.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="New immutable KS4 coverage-audit directory.",
        ),
    ) -> None:
        """Audit every planned minute without imputing missing paths."""

        report = build_intraday_specialist_coverage_audit(
            collection_plan_directory=collection_plan_dir,
            collection_directory=collection_dir,
            policy_path=policy,
            output_directory=out_dir,
        )
        summary = report["summary"]
        console.print(
            "Audited "
            f"{summary['requirements']:,} requirements; "
            f"{summary['grid_complete_setups']:,}/"
            f"{summary['setups']:,} setups have a physically complete grid."
        )

    @app.command("build-intraday-specialist-training-data")
    @serialized_heavy_job("build-intraday-specialist-training-data")
    def build_intraday_specialist_training_data(
        setup_dir: Path = typer.Option(
            ...,
            help="Completed immutable causal KS4 setup bundle.",
        ),
        collection_plan_dir: Path = typer.Option(
            ...,
            help="Completed immutable KS4 one-minute collection plan.",
        ),
        collection_dir: Path = typer.Option(
            ...,
            help="Transport-complete KS4 one-minute collection.",
        ),
        policy: Path = typer.Option(
            Path("configs/intraday_specialist_research.toml"),
            help="Frozen KS4 specialist policy.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="New immutable KS4 training dataset directory.",
        ),
    ) -> None:
        """Build clock-grid features and executable one-minute labels."""

        report = build_intraday_specialist_training_dataset(
            setup_directory=setup_dir,
            collection_plan_directory=collection_plan_dir,
            collection_directory=collection_dir,
            policy_path=policy,
            output_directory=out_dir,
        )
        summary = report["summary"]
        console.print(
            "Built "
            f"{summary['eligible_rows']:,}/"
            f"{summary['rows']:,} executable KS4 training rows."
        )

    @app.command("train-intraday-specialists")
    @serialized_heavy_job("train-intraday-specialists")
    def train_intraday_specialists(
        dataset_dir: Path = typer.Option(
            ...,
            help="Completed immutable KS4 training dataset bundle.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="Resumable immutable KS4 experiment bundle directory.",
        ),
        policy: Path = typer.Option(
            Path("configs/intraday_specialist_research.toml"),
            help="Frozen KS4 specialist policy.",
        ),
        strategy_id: str | None = typer.Option(
            None,
            "--strategy-id",
            help="Run one frozen strategy; omit to run all sequentially.",
        ),
    ) -> None:
        """Evaluate the frozen KS4 intraday candidate matrix."""

        config = load_intraday_specialist_research_config(policy)
        result = train_intraday_specialist_experiments(
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            config=config,
            policy_path=policy,
            strategy_ids=(strategy_id,) if strategy_id is not None else None,
            progress=console.print,
        )
        console.print(result)
        if result["invocation_status"] != "complete":
            raise typer.Exit(code=2)
