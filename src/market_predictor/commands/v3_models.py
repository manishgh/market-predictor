from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import typer
from rich.console import Console

from market_predictor.v3.development import load_verified_development_dataset
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.models import MODEL_FAMILIES, ModelFamily, V3TrainingConfig, train_v3_model_suite


def register_v3_model_commands(app: typer.Typer, console: Console) -> None:
    @app.command("train-v3-models")
    def train_models(
        dataset: Path = typer.Option(..., help="Development-only V3 feature and label parquet."),
        output_dir: Path = typer.Option(Path("models/v3/candidates"), help="Candidate model artifact directory."),
        report_out: Path = typer.Option(Path("data/reports/v3_training_report_latest.json"), help="Training report JSON."),
        predictions_out: Path = typer.Option(Path("data/reports/v3_oof_predictions_latest.parquet"), help="OOF prediction parquet."),
        feature_audit_out: Path = typer.Option(Path("data/reports/v3_feature_fold_audit_latest.csv"), help="Fold feature audit CSV."),
        families: str = typer.Option("B0,B1,B2,R1,D1", help="Comma-separated model families."),
        n_splits: int = typer.Option(4, min=2, max=10, help="Expanding walk-forward fold count."),
        embargo_sessions: int = typer.Option(1, min=0, max=10, help="Purged sessions between train and test."),
        min_train_sessions: int = typer.Option(20, min=2, help="Minimum initial training sessions."),
        min_train_rows: int = typer.Option(500, min=1, help="Minimum training rows per fold."),
        ticker_holdout_fraction: float = typer.Option(0.2, min=0.01, max=0.99, help="Deterministic symbol holdout fraction."),
        overwrite: bool = typer.Option(False, help="Explicitly replace candidate artifacts."),
    ) -> None:
        """Train V3 baselines, ranker, and downside model with purged OOF evidence."""
        parsed = _parse_families(families)
        training_data, dataset_fingerprint = _read_dataset(dataset)
        config = V3TrainingConfig(
            families=parsed,
            n_splits=n_splits,
            embargo_sessions=embargo_sessions,
            min_train_sessions=min_train_sessions,
            min_train_rows=min_train_rows,
            ticker_holdout_fraction=ticker_holdout_fraction,
            training_dataset_fingerprint=dataset_fingerprint,
        )
        report, predictions, feature_audit = train_v3_model_suite(
            training_data,
            output_dir,
            config=config,
            overwrite=overwrite,
        )
        report_out.parent.mkdir(parents=True, exist_ok=True)
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        feature_audit_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        predictions.to_parquet(predictions_out, index=False)
        feature_audit.to_csv(feature_audit_out, index=False)
        console.print(f"Wrote V3 training report to {report_out}")
        console.print(f"Wrote {len(predictions)} OOF audit rows to {predictions_out}")
        for family, result in report["models"].items():
            if result["status"] == "complete":
                console.print(f"{family}: {result['artifact_path']}")
            else:
                console.print(f"[red]{family}: {result['error']}[/red]")
        if report["failed_families"]:
            raise typer.Exit(code=2)


def _parse_families(value: str) -> tuple[ModelFamily, ...]:
    parsed = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    invalid = sorted(set(parsed).difference(MODEL_FAMILIES))
    if not parsed or invalid:
        raise typer.BadParameter(f"families must be selected from {MODEL_FAMILIES}; invalid={invalid}")
    return cast(tuple[ModelFamily, ...], parsed)


def _read_dataset(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        raise typer.BadParameter(f"Missing dataset: {path}")
    if path.is_dir():
        try:
            dataset, manifest = load_verified_development_dataset(path)
        except DataReadinessError as exc:
            raise typer.BadParameter(str(exc)) from exc
        return dataset, str(manifest["dataset_fingerprint"])
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path), None
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), None
    raise typer.BadParameter(f"Unsupported dataset format: {path.suffix}")
