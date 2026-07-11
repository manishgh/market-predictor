from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from market_predictor.v3.readiness import DevelopmentReadinessConfig, audit_development_readiness


def register_v3_readiness_commands(app: typer.Typer, console: Console) -> None:
    @app.command("audit-v3-development-readiness")
    def audit_readiness(
        bars: Path = typer.Option(..., help="Ticker 5-minute parquet file or dataset directory."),
        universe: Path = typer.Option(..., help="Point-in-time universe CSV or parquet."),
        benchmark_dir: Path = typer.Option(..., help="Directory containing per-symbol benchmark parquet files."),
        out: Path = typer.Option(Path("data/reports/v3_development_readiness_latest.json"), help="Readiness report JSON."),
        minimum_tickers: int = typer.Option(300, min=2, help="Minimum distinct development symbols."),
        minimum_sessions: int = typer.Option(252, min=2, help="Minimum development sessions."),
        required_benchmarks: str = typer.Option(
            "SPY,QQQ,XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY",
            help="Comma-separated exact-timestamp benchmark symbols.",
        ),
    ) -> None:
        """Gate C8 on history, PIT universe, SIP provenance, and benchmarks."""
        symbols = tuple(dict.fromkeys(item.strip().upper() for item in required_benchmarks.split(",") if item.strip()))
        report = audit_development_readiness(
            bars_path=bars,
            universe_path=universe,
            benchmark_dir=benchmark_dir,
            config=DevelopmentReadinessConfig(
                minimum_tickers=minimum_tickers,
                minimum_sessions=minimum_sessions,
                required_benchmarks=symbols,
            ),
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"Wrote V3 development readiness report to {out}")
        for check in report["checks"]:
            color = "green" if check["status"] == "pass" else "red"
            console.print(f"[{color}]{check['status'].upper()}[/{color}] {check['name']}: {check['observed']}")
        if not report["ready"]:
            raise typer.Exit(code=2)
