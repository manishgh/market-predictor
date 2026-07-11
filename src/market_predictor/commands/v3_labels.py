from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from market_predictor.v3.labels import V3LabelConfig, build_v3_labels


def register_v3_label_commands(app: typer.Typer, console: Console) -> None:
    @app.command("build-v3-labels")
    def build_labels(
        bars: Path = typer.Option(..., help="Audited ticker decision bars in CSV or parquet."),
        benchmarks: Path = typer.Option(..., help="Audited QQQ and sector ETF bars in CSV or parquet."),
        out: Path = typer.Option(Path("data/features/v3_labels_latest.parquet"), help="Output V3 labeled parquet."),
        horizons: str = typer.Option("6,12,24", help="Comma-separated bar horizons."),
        primary_horizon: int = typer.Option(12, min=1, help="Primary path and ranking horizon in bars."),
        bar_minutes: int = typer.Option(5, min=1, help="Minutes represented by each bar."),
        round_trip_cost_bps: float = typer.Option(10.0, min=0, help="Round-trip cost deducted from ticker returns."),
        minimum_ranking_group: int = typer.Option(20, min=2, help="Minimum simultaneous candidates for ranking grades."),
        partition: str = typer.Option("development", help="Input partition: development or shadow."),
        overwrite: bool = typer.Option(False, help="Explicitly replace an existing derived label artifact."),
    ) -> None:
        """Build exact-interval V3 return, path, rank, and overlap labels."""
        parsed_horizons = _parse_horizons(horizons)
        if partition not in {"development", "shadow"}:
            raise typer.BadParameter("partition must be development or shadow")
        if out.exists() and not overwrite:
            raise typer.BadParameter(f"Output already exists; pass --overwrite to replace it: {out}")
        config = V3LabelConfig(
            horizons_bars=parsed_horizons,
            primary_horizon_bars=primary_horizon,
            bar_minutes=bar_minutes,
            round_trip_cost_bps=round_trip_cost_bps,
            minimum_ranking_group=minimum_ranking_group,
        )
        labeled = build_v3_labels(
            _read_frame(bars),
            _read_frame(benchmarks),
            config=config,
            partition=partition,  # type: ignore[arg-type]
        )
        if labeled.empty:
            raise typer.BadParameter("No label rows were produced; verify sessions, horizons, and benchmark coverage.")
        out.parent.mkdir(parents=True, exist_ok=True)
        labeled.to_parquet(out, index=False)
        console.print(f"Wrote {len(labeled)} V3 label rows to {out}")


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter("horizons must be comma-separated positive integers") from exc
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise typer.BadParameter("horizons must be comma-separated positive integers")
    return horizons


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Missing input: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise typer.BadParameter(f"Unsupported input format: {path.suffix}")
