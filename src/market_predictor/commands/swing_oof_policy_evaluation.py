"""Evaluate frozen exit policies on saved scores without model fitting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import typer
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import inside
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError
from market_predictor.research.swing_oof_policy_evaluation import Learner, publish_saved_oof_policy_evaluation
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.action_evidence import load_corporate_action_evidence
from market_predictor.swing.datasets.corrected_outcomes import load_corrected_outcome_policy
from market_predictor.swing.labels.frozen_exit import ExitPolicy


def register_saved_oof_policy_command(app: typer.Typer) -> None:
    @app.command("evaluate-saved-swing-policy")
    def evaluate(
        root: Path = typer.Option(...), output: Path = typer.Option(...),
        run_manifest: Path = typer.Option(...), run_manifest_sha256: str = typer.Option(...),
        feature_publication: Path = typer.Option(...), feature_publication_sha256: str = typer.Option(...),
        target_config: Path = typer.Option(...), target_config_sha256: str = typer.Option(...),
        strategy_contract: Path = typer.Option(...), strategy_contract_sha256: str = typer.Option(...),
        research_contract: Path = typer.Option(...), research_contract_sha256: str = typer.Option(...),
        learner: str = typer.Option(...), exit_policy: str = typer.Option(...),
        historical_configuration_evidence: Path | None = typer.Option(None),
        historical_configuration_evidence_sha256: str | None = typer.Option(None),
    ) -> None:
        """Load action evidence first, then evaluate through the single source lease."""
        try:
            if learner not in ("regularized_linear_return", "shallow_boosted_return"):
                raise DataReadinessError("saved policy evaluation requires one frozen return learner")
            if exit_policy not in ("target_stop_ten_session_timeout", "stop_ten_session_timeout"):
                raise DataReadinessError("saved policy evaluation requires one frozen exit policy")
            if (historical_configuration_evidence is None) != (historical_configuration_evidence_sha256 is None):
                raise DataReadinessError("historical configuration evidence requires both path and SHA256")
            historical = (SourcePin(path=historical_configuration_evidence.as_posix(),
                                    sha256=historical_configuration_evidence_sha256)
                          if historical_configuration_evidence is not None
                          and historical_configuration_evidence_sha256 is not None else None)
            target = SourcePin(path=target_config.as_posix(), sha256=target_config_sha256)
            run = SourcePin(path=run_manifest.as_posix(), sha256=run_manifest_sha256)
            features = SourcePin(path=feature_publication.as_posix(), sha256=feature_publication_sha256)
            strategy = SourcePin(path=strategy_contract.as_posix(), sha256=strategy_contract_sha256)
            research = SourcePin(path=research_contract.as_posix(), sha256=research_contract_sha256)
            policy = load_corrected_outcome_policy(root, Path(target.path), target.sha256)
            if file_sha256(inside(root, policy.action_config.path)) != policy.action_config.sha256:
                raise DataReadinessError("saved policy action configuration differs from its target pin")
            evidence = load_corporate_action_evidence(
                root=root, config=Path(policy.action_config.path), archive=Path(policy.action_archive),
                expected_audit_sha256=policy.action_audit_sha256,
            )
            report = publish_saved_oof_policy_evaluation(
                root=root, run_manifest=run, feature_publication=features,
                target_config=target,
                strategy_contract=strategy, research_contract=research,
                learner=cast(Learner, learner), exit_policy=cast(ExitPolicy, exit_policy), evidence=evidence, output=output,
                historical_configuration_evidence=historical,
            )
        except HeavyJobBusyError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from error
        except (DataReadinessError, OSError, ValidationError) as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=2) from error
        summary = {name: report[name] for name in (
            "status", "manifest_sha256", "selected_rows", "training_eligible", "promotion_eligible", "serving_eligible",
        ) if name in report}
        summary["report_path"] = str(inside(root, output) / "report.json")
        typer.echo(json.dumps(summary, sort_keys=True))
        if report["status"] != "computed":
            raise typer.Exit(code=2)
