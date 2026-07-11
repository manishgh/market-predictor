from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from market_predictor.v3.features import build_v3_features, core_feature_columns


def register_v3_feature_commands(app: typer.Typer, console: Console) -> None:
    @app.command("build-v3-features")
    def build_features(
        bars: Path = typer.Option(..., help="Audited ticker OHLCV CSV or parquet."),
        benchmarks: Path = typer.Option(..., help="Exact-timestamp QQQ, SPY, and sector ETF bars."),
        source_availability: Path | None = typer.Option(None, help="Optional point-in-time source availability CSV or parquet."),
        out: Path = typer.Option(Path("data/features/v3_features_latest.parquet"), help="Output V3 feature parquet."),
        minimum_cross_section: int = typer.Option(20, min=2, help="Minimum simultaneous candidates for ranks and robust z-scores."),
        overwrite: bool = typer.Option(False, help="Explicitly replace an existing derived feature artifact."),
    ) -> None:
        """Build the frozen batch/live V3 feature schema."""
        if out.exists() and not overwrite:
            raise typer.BadParameter(f"Output already exists; pass --overwrite to replace it: {out}")
        availability = _read_frame(source_availability) if source_availability is not None else None
        features = build_v3_features(
            _read_frame(bars),
            _read_frame(benchmarks),
            source_availability=availability,
            minimum_cross_section=minimum_cross_section,
        )
        if features.empty:
            raise typer.BadParameter("No feature rows were produced.")
        out.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(out, index=False)
        populated = sum(features[column].notna().any() for column in core_feature_columns())
        console.print(f"Wrote {len(features)} V3 feature rows to {out}")
        console.print(f"Core feature coverage: {populated}/{len(core_feature_columns())}")


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Missing input: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise typer.BadParameter(f"Unsupported input format: {path.suffix}")
