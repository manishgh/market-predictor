from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from market_predictor.canonical.store import load_canonical_artifact
from market_predictor.commands.configuration import load_typed_config
from market_predictor.drift_policy import (
    DriftPolicyV1,
    DriftStateStore,
    evaluate_drift,
)
from market_predictor.outcome_intents import register_snapshot_intents
from market_predictor.outcome_repository import OutcomeRepository
from market_predictor.outcome_worker import mature_pending_intents
from market_predictor.performance_monitoring import (
    build_performance_cohorts,
    load_performance_report,
    write_performance_report,
)
from market_predictor.prediction_snapshot import PredictionSnapshotStore


def register_outcome_commands(app: typer.Typer, console: Any) -> None:
    @app.command("register-outcome-intents")
    def register_outcome_intents(
        snapshot_id: str = typer.Option(..., help="Immutable prediction snapshot id."),
        snapshot_dir: Path = typer.Option(
            Path("data/predictions/snapshots"),
            help="Prediction snapshot repository.",
        ),
        outcome_dir: Path = typer.Option(
            Path("data/predictions/outcomes"),
            help="Durable local outcome repository.",
        ),
    ) -> None:
        """Freeze maturation intents from one identity-complete live snapshot."""

        intents = register_snapshot_intents(
            PredictionSnapshotStore(snapshot_dir),
            OutcomeRepository(outcome_dir),
            snapshot_id,
        )
        console.print(
            json.dumps(
                {
                    "snapshot_id": snapshot_id,
                    "registered_intents": len(intents),
                    "maturation_keys": [
                        intent.maturation_key for intent in intents
                    ],
                },
                sort_keys=True,
            )
        )

    @app.command("mature-outcomes")
    def mature_outcomes(
        bars: Path = typer.Option(
            ...,
            help="Hash-verified production-ready canonical OHLCV artifact.",
        ),
        outcome_dir: Path = typer.Option(
            Path("data/predictions/outcomes"),
            help="Durable local outcome repository.",
        ),
        observed_as_of: datetime | None = typer.Option(
            None,
            help="Timezone-aware observation cutoff; defaults to current UTC.",
        ),
    ) -> None:
        """Mature canonical semantic predictions at their frozen label horizon."""

        frame, manifest = load_canonical_artifact(bars)
        source_sha = str(manifest["artifact_sha256"])
        cutoff = observed_as_of or datetime.now(UTC)
        if cutoff.utcoffset() is None:
            raise typer.BadParameter("observed-as-of must be timezone-aware")
        summary = mature_pending_intents(
            OutcomeRepository(outcome_dir),
            frame,
            observed_as_of=cutoff,
            source_artifact_sha256=source_sha,
        )
        console.print(json.dumps(summary, sort_keys=True))

    @app.command("build-outcome-performance-report")
    def build_outcome_performance_report(
        outcome_dir: Path = typer.Option(
            Path("data/predictions/outcomes"),
            help="Durable local outcome repository.",
        ),
        report_out: Path = typer.Option(
            Path("data/monitoring/performance/latest.json"),
            help="Atomic output path for the validated performance report.",
        ),
        minimum_samples: int = typer.Option(
            30,
            min=1,
            help="Minimum matured outcomes required for sufficient evidence.",
        ),
        generated_at: datetime | None = typer.Option(
            None,
            help="Timezone-aware report timestamp; defaults to current UTC.",
        ),
    ) -> None:
        """Build immutable release/view/horizon performance cohorts."""

        timestamp = generated_at or datetime.now(UTC)
        if timestamp.utcoffset() is None:
            raise typer.BadParameter("generated-at must be timezone-aware")
        report = build_performance_cohorts(
            OutcomeRepository(outcome_dir),
            generated_at=timestamp,
            minimum_samples=minimum_samples,
        )
        persisted = write_performance_report(report_out, report)
        source_ids = persisted.get("source_outcome_ids")
        cohorts = persisted.get("rows")
        if not isinstance(source_ids, list) or not isinstance(cohorts, list):
            raise RuntimeError("validated performance report shape is invalid")
        console.print(
            json.dumps(
                {
                    "report_id": persisted["report_id"],
                    "report_path": str(report_out),
                    "source_outcomes": len(source_ids),
                    "cohorts": len(cohorts),
                },
                sort_keys=True,
            )
        )

    @app.command("publish-drift-assessment")
    def publish_drift_assessment(
        mode: str = typer.Option(..., help="Prediction view: swing or intraday."),
        horizon: str = typer.Option(..., help="Canonical route horizon, such as 5d or 60m."),
        model_release_id: str = typer.Option(
            ...,
            help="Active model release SHA-256 identity.",
        ),
        feature_drift_report: Path = typer.Option(
            ...,
            help="Feature-drift JSON produced from the active model reference.",
        ),
        performance_report: Path = typer.Option(
            Path("data/monitoring/performance/latest.json"),
            help="Validated matured-outcome performance report.",
        ),
        policy_config: Path = typer.Option(
            Path("configs/drift_policy.toml"),
            help="Versioned drift policy TOML or JSON.",
        ),
        drift_dir: Path = typer.Option(
            Path("data/monitoring/drift"),
            help="Persisted route drift-state repository.",
        ),
        evaluated_at: datetime | None = typer.Option(
            None,
            help="Timezone-aware assessment timestamp; defaults to current UTC.",
        ),
    ) -> None:
        """Evaluate and atomically publish one release-specific route drift state."""

        timestamp = evaluated_at or datetime.now(UTC)
        if timestamp.utcoffset() is None:
            raise typer.BadParameter("evaluated-at must be timezone-aware")
        feature_drift = _load_json_object(feature_drift_report)
        report = load_performance_report(performance_report)
        policy = load_typed_config(policy_config, DriftPolicyV1)
        assessment = evaluate_drift(
            mode=mode.strip().lower(),
            horizon=horizon.strip().lower(),
            model_release_id=model_release_id.strip().lower(),
            feature_drift=feature_drift,
            performance_report=report,
            policy=policy,
            evaluated_at=timestamp,
        )
        DriftStateStore(drift_dir).publish(assessment)
        console.print(assessment.model_dump_json())


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise typer.BadParameter(f"JSON input does not exist: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid JSON input: {path}") from exc
    if not isinstance(loaded, dict):
        raise typer.BadParameter(f"JSON input must contain an object: {path}")
    return {str(key): value for key, value in loaded.items()}
