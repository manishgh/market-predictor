from __future__ import annotations

import json
from dataclasses import fields
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from market_predictor.collection.outcome_bars import collect_bars, collect_corporate_actions
from market_predictor.commands.configuration import load_typed_config
from market_predictor.config import Settings
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.governance.drift.policy import (
    DriftPolicy,
    DriftStateStore,
    evaluate_drift,
)
from market_predictor.governance.outcomes.collection_plan import plan_outcome_collection
from market_predictor.governance.outcomes.evidence import EvidenceTerms
from market_predictor.governance.outcomes.performance import (
    build_performance_cohorts,
    load_performance_report,
    write_performance_report,
)
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.worker import mature_pending_intents, record_operator_resolution
from market_predictor.monitoring_lease import monitoring_lease
from market_predictor.serving.outcome_intents import register_snapshot_intents
from market_predictor.serving.routes import ServingRoute
from market_predictor.serving.snapshot_store import PredictionSnapshotStore
from market_predictor.serving.swing_features import FileSwingLiveInputProvider
from market_predictor.sources.alpaca import AlpacaSource

# Typer's default datetime formats carry no offset, which the commands refuse.
_AWARE_FORMATS = ["%Y-%m-%dT%H:%M:%S%z"]
_RECEIPT_DIR = Path("data/predictions/outcome_receipts")
_DRIFT_POLICY = Path("configs/drift_policy.toml")


def _evidence_terms(policy_path: Path) -> EvidenceTerms:
    policy = load_typed_config(policy_path, DriftPolicy)
    return EvidenceTerms(
        grace_days=policy.pending_grace_days,
        settlement_days=policy.outcome_settlement_days,
        drift_policy_sha256=policy.sha256(),
    )


def _aware(value: datetime | None, option: str) -> datetime:
    moment = value or datetime.now(UTC)
    if moment.utcoffset() is None:
        raise typer.BadParameter(f"{option} must be timezone-aware")
    return moment.astimezone(UTC)


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

        registration = register_snapshot_intents(
            PredictionSnapshotStore(snapshot_dir),
            OutcomeRepository(outcome_dir),
            snapshot_id,
        )
        console.print(
            json.dumps(
                {
                    "snapshot_id": snapshot_id,
                    "registered_intents": len(registration.intents),
                    "maturation_keys": [
                        intent.maturation_key for intent in registration.intents
                    ],
                    "unmonitored_tickers": registration.unmonitored_tickers,
                },
                sort_keys=True,
            )
        )

    @app.command("collect-outcome-bars")
    def collect_outcome_bars(
        outcome_dir: Path = typer.Option(
            Path("data/predictions/outcomes"),
            help="Durable local outcome repository.",
        ),
        receipt_dir: Path = typer.Option(_RECEIPT_DIR, help="Outcome evidence receipts and page bodies."),
        drift_policy: Path = typer.Option(_DRIFT_POLICY, help="Drift policy with the grace and settlement periods."),
        maturation_key: str | None = typer.Option(
            None, help="Collect for this one intent now, even if frozen or not yet due; needs --decision-session."
        ),
        decision_session: str | None = typer.Option(None, help="The named intent's decision session."),
    ) -> None:
        """Collect the bars and corporate actions that pending outcomes still need, with receipts."""

        if (maturation_key is None) != (decision_session is None):
            raise typer.BadParameter("--maturation-key and --decision-session go together")
        only = (
            (maturation_key.strip().lower(), date.fromisoformat(decision_session.strip()))
            if maturation_key is not None and decision_session is not None
            else None
        )
        terms = _evidence_terms(drift_policy)
        with monitoring_lease("collect-outcome-bars"):
            plan = plan_outcome_collection(
                OutcomeRepository(outcome_dir), receipts_root=receipt_dir, now=datetime.now(UTC), terms=terms, only=only
            )
            source = AlpacaSource(Settings())
            bars = collect_bars(source, plan.bar_units, root=receipt_dir)
            actions = collect_corporate_actions(source, plan.action_units, root=receipt_dir)
        receipts = [*bars, *actions]
        console.print(
            json.dumps(
                {
                    "bar_units": len(plan.bar_units),
                    "corporate_action_units": len(plan.action_units),
                    "failed_receipts": sum(receipt["status"] == "failed" for receipt in receipts),
                },
                sort_keys=True,
            )
        )

    @app.command("mature-outcomes")
    def mature_outcomes(
        outcome_dir: Path = typer.Option(
            Path("data/predictions/outcomes"),
            help="Durable local outcome repository.",
        ),
        receipt_dir: Path = typer.Option(_RECEIPT_DIR, help="Outcome evidence receipts and page bodies."),
        drift_policy: Path = typer.Option(_DRIFT_POLICY, help="Drift policy with the grace and settlement periods."),
        live_input_root: Path | None = typer.Option(
            None,
            help="Swing live-input repository whose point-in-time memberships evidence removals; none leaves them unknown.",
        ),
        observed_as_of: datetime | None = typer.Option(
            None,
            formats=_AWARE_FORMATS,
            help="Timezone-aware observation cutoff, such as 2026-08-08T12:00:00+00:00; defaults to current UTC.",
        ),
    ) -> None:
        """Mature canonical semantic predictions from collected receipts at their frozen label horizon."""

        cutoff = _aware(observed_as_of, "observed-as-of")
        terms = _evidence_terms(drift_policy)
        memberships: pd.DataFrame | None = None
        if live_input_root is not None:
            # The same read limits the serving route applies to live inputs.
            limits = {field.name: field.default for field in fields(ServingRoute)}
            memberships = FileSwingLiveInputProvider(live_input_root).load(
                as_of_utc=cutoff,
                maximum_bytes=int(str(limits["max_feature_bytes"])),
                maximum_rows=int(str(limits["max_feature_rows"])),
            ).point_in_time_memberships
        with monitoring_lease("mature-outcomes"):
            summary = mature_pending_intents(
                OutcomeRepository(outcome_dir),
                receipts_root=receipt_dir,
                observed_as_of=cutoff,
                terms=terms,
                memberships=memberships,
            )
        console.print(json.dumps(summary, sort_keys=True))
        if summary["registration_incomplete"]:
            # Each keeps its route overdue until its registration is rerun.
            console.print(
                f"{summary['registration_incomplete']} registrations stopped before their semantic record; "
                "rerun them to complete the pending index"
            )
            raise typer.Exit(code=1)

    @app.command("record-operator-outcome-resolution")
    def record_operator_outcome_resolution(
        maturation_key: str = typer.Option(..., help="The pending intent's maturation key."),
        decision_session: str = typer.Option(..., help="The intent's decision session, such as 2026-07-24."),
        operator: str = typer.Option(..., help="Who verified the finding."),
        reference: str = typer.Option(..., help="The evidence checked: a filing, an exchange notice, a URL."),
        outcome_dir: Path = typer.Option(
            Path("data/predictions/outcomes"),
            help="Durable local outcome repository.",
        ),
        receipt_dir: Path = typer.Option(_RECEIPT_DIR, help="Outcome evidence receipts and page bodies."),
        drift_policy: Path = typer.Option(_DRIFT_POLICY, help="Drift policy with the grace and settlement periods."),
    ) -> None:
        """Record an operator's verified finding that a stock stopped trading, so its outcome never matures."""

        with monitoring_lease("record-operator-outcome-resolution"):
            attempt = record_operator_resolution(
                OutcomeRepository(outcome_dir),
                maturation_key=maturation_key.strip().lower(),
                decision_session=date.fromisoformat(decision_session.strip()),
                operator_id=operator.strip(),
                reference=reference.strip(),
                receipts_root=receipt_dir,
                observed_as_of=datetime.now(UTC),
                terms=_evidence_terms(drift_policy),
            )
        console.print(attempt.model_dump_json())

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
        lookback_days: int = typer.Option(
            180,
            min=1,
            help="Calendar days of outcomes: decisions whose horizon ends in this window, or is still open.",
        ),
        generated_at: datetime | None = typer.Option(
            None,
            formats=_AWARE_FORMATS,
            help="Timezone-aware report timestamp, such as 2026-08-08T12:00:00+00:00; defaults to current UTC.",
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
            lookback_days=lookback_days,
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
        mode: str = typer.Option("swing", help="Prediction view; only swing is served (intraday is retired)."),
        horizon: str = typer.Option(..., help="Route horizon in exchange sessions, such as 10b."),
        model_release_id: str = typer.Option(
            ...,
            help="Active model release SHA-256 identity.",
        ),
        model_artifact_sha256: str = typer.Option(
            ...,
            help="Active model artifact SHA-256 identity.",
        ),
        prediction_policy_sha256: str = typer.Option(
            ...,
            help="Active prediction-policy SHA-256 identity.",
        ),
        label_policy_sha256: str = typer.Option(
            ...,
            help="Active label-policy SHA-256 identity.",
        ),
        execution_policy_sha256: str = typer.Option(
            ...,
            help="Active execution-policy SHA-256 identity.",
        ),
        feature_reference_profile_sha256: str = typer.Option(
            ...,
            help="Feature-reference SHA-256 identity bound to the active model.",
        ),
        feature_reference_names_sha256: str = typer.Option(
            ...,
            help="Feature-name-set SHA-256 identity bound to the active model.",
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
            formats=_AWARE_FORMATS,
            help="Timezone-aware assessment timestamp, such as 2026-08-08T12:00:00+00:00; defaults to current UTC.",
        ),
    ) -> None:
        """Evaluate and atomically publish one release-specific route drift state."""

        if mode.strip().lower() != "swing":
            raise typer.BadParameter("only swing routes are assessed; intraday prediction is retired")
        timestamp = evaluated_at or datetime.now(UTC)
        if timestamp.utcoffset() is None:
            raise typer.BadParameter("evaluated-at must be timezone-aware")
        feature_drift = _load_json_object(feature_drift_report)
        report = load_performance_report(performance_report)
        policy = load_typed_config(policy_config, DriftPolicy)
        assessment = evaluate_drift(
            mode="swing",
            horizon=horizon.strip().lower(),
            model_release_id=model_release_id.strip().lower(),
            model_artifact_sha256=model_artifact_sha256.strip().lower(),
            prediction_policy_sha256=prediction_policy_sha256.strip().lower(),
            label_policy_sha256=label_policy_sha256.strip().lower(),
            execution_policy_sha256=execution_policy_sha256.strip().lower(),
            feature_reference_profile_sha256=(
                feature_reference_profile_sha256.strip().lower()
            ),
            feature_reference_names_sha256=(
                feature_reference_names_sha256.strip().lower()
            ),
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
        loaded = parse_strict_json_object(path.read_bytes(), label="JSON input")
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"invalid JSON input: {path}") from exc
    return {str(key): value for key, value in loaded.items()}
