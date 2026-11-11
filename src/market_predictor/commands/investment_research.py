"""Thin command surfaces for immutable investment source and research artifacts."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from market_predictor.config import get_settings
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError
from market_predictor.investment.dividend_resolution import resolve_investment_dividends
from market_predictor.investment.raw_acquisition import (
    collect_investment_raw_acquisition,
    publish_investment_raw_acquisition_plan,
    verify_investment_raw_acquisition,
)
from market_predictor.research.investment_coverage import plan_investment_coverage, run_investment_coverage
from market_predictor.research.investment_currencies import collect_investment_dividend_currencies
from market_predictor.research.investment_dataset import build_investment_dataset
from market_predictor.research.investment_diagnostics import diagnose_investment_predictions
from market_predictor.research.investment_feature_ablation import run_investment_feature_ablation
from market_predictor.research.investment_features import (
    build_investment_features,
    collect_investment_company_facts,
    evaluate_investment_predictions,
)
from market_predictor.research.investment_forecast_adjustment import run_investment_forecast_adjustment
from market_predictor.research.investment_labels import materialize_investment_labels
from market_predictor.research.investment_ranking import train_investment_ranking
from market_predictor.research.investment_selection_outcomes import resolve_investment_selection_outcomes
from market_predictor.research.investment_training import train_investment_models
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.ictax import collect_ictax_archive, verify_ictax_archive
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.history_archive import AlpacaSwingDailyPageSource


def _run(action: Callable[[], dict[str, Any]]) -> None:
    try:
        result = action()
    except HeavyJobBusyError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from error
    except (DataReadinessError, MemoryBudgetError, OSError, ValueError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True, allow_nan=False))


def register_investment_collection_commands(app: typer.Typer) -> None:
    @app.command("collect-investment-raw-candles")
    def collect_raw_candles(root: Path = typer.Option(...), request: Path = typer.Option(...),
                            plan: Path = typer.Option(...), plan_authority_sha256: str = typer.Option(...),
                            output: Path = typer.Option(...)) -> None:
        """Resume exact missing raw SIP candles; preserve missing units and unreviewed stock identities."""
        _run(lambda: collect_investment_raw_acquisition(root, request, plan, output,
            expected_plan_authority_sha256=plan_authority_sha256,
            source_factory=lambda: AlpacaSwingDailyPageSource(AlpacaSource(get_settings()))))

    @app.command("collect-ictax-distributions")
    def collect_tax_distributions(root: Path = typer.Option(...), request: Path = typer.Option(...),
                                  request_sha256: str = typer.Option(...), output: Path = typer.Option(...)) -> None:
        """Archive official fund distribution responses with resumable source checkpoints."""
        _run(lambda: collect_ictax_archive(root=root, request=request, request_sha256=request_sha256, output=output))

    @app.command("collect-investment-dividend-currencies")
    def collect_currencies(root: Path = typer.Option(...), action_config: Path = typer.Option(...),
                           action_config_sha256: str = typer.Option(...), action_archive: Path = typer.Option(...),
                           action_audit_sha256: str = typer.Option(...), output: Path = typer.Option(...),
                           required_events: Path | None = typer.Option(None),
                           required_events_sha256: str | None = typer.Option(None)) -> None:
        """Corroborate dividend currency with archived provider responses for exact action IDs."""
        if (required_events is None) != (required_events_sha256 is None):
            raise typer.BadParameter("required-events and required-events-sha256 must be supplied together")
        event_scope = None
        if required_events is not None and required_events_sha256 is not None:
            event_scope = SourcePin(path=required_events.as_posix(), sha256=required_events_sha256)
        _run(lambda: collect_investment_dividend_currencies(
            root=root, action_config=SourcePin(path=action_config.as_posix(), sha256=action_config_sha256),
            action_archive=action_archive, action_audit_sha256=action_audit_sha256, output=output, required_events=event_scope))

    @app.command("collect-investment-coverage")
    def collect_coverage(root: Path = typer.Option(...), roster_manifest: Path = typer.Option(...),
                         roster_manifest_sha256: str = typer.Option(...), output: Path = typer.Option(...)) -> None:
        """Resume serial financial collection and feature coverage over the frozen investment roster."""
        _run(lambda: run_investment_coverage(root=root, roster_manifest=roster_manifest,
            roster_manifest_sha256=roster_manifest_sha256, output=output))

    @app.command("collect-investment-company-facts")
    def collect(root: Path = typer.Option(...), cik: str = typer.Option(...), security_id: str = typer.Option(...),
                output: Path = typer.Option(...)) -> None:
        """Archive one issuer's SEC financial facts; requires configured SEC_USER_AGENT."""
        _run(lambda: collect_investment_company_facts(root=root, cik=cik, security_id=security_id, output=output))


def register_investment_research_commands(app: typer.Typer) -> None:
    @app.command("resolve-investment-dividends")
    def resolve_dividends(root: Path = typer.Option(...), request: Path = typer.Option(...),
                          request_sha256: str = typer.Option(...), output: Path = typer.Option(...)) -> None:
        """Resolve exact required cash events from retained sources and report local evidence gaps."""
        _run(lambda: resolve_investment_dividends(root=root,
            request=SourcePin(path=request.as_posix(), sha256=request_sha256), output=output))

    @app.command("plan-investment-raw-candles")
    def plan_raw_candles(root: Path = typer.Option(...), request: Path = typer.Option(...),
                        output: Path = typer.Option(...)) -> None:
        """Reconstruct exact missing sessions from pinned roster and retained sources."""
        _run(lambda: publish_investment_raw_acquisition_plan(root, request, output))

    @app.command("verify-investment-raw-candles")
    def verify_raw_candles(root: Path = typer.Option(...), request: Path = typer.Option(...),
                          plan: Path = typer.Option(...), plan_authority_sha256: str = typer.Option(...),
                          archive: Path = typer.Option(...)) -> None:
        """Replay raw transport and report observations, missing sessions and identity quarantine."""
        _run(lambda: verify_investment_raw_acquisition(root, request, plan, archive,
            expected_plan_authority_sha256=plan_authority_sha256))

    @app.command("verify-ictax-distributions")
    def verify_tax_distributions(root: Path = typer.Option(...), manifest: Path = typer.Option(...),
                                 manifest_sha256: str = typer.Option(...)) -> None:
        """Replay archived fund-year source bytes and receipts without admitting training data."""
        _run(lambda: verify_ictax_archive(root=root, manifest=manifest, manifest_sha256=manifest_sha256))

    @app.command("materialize-investment-labels")
    def materialize_labels(root: Path = typer.Option(...), request: Path = typer.Option(...),
                           request_sha256: str = typer.Option(...), output: Path = typer.Option(...),
                           parent_receipt: Path | None = typer.Option(None),
                           parent_receipt_sha256: str | None = typer.Option(None)) -> None:
        """Calculate monthly future-return targets and report missing price or action data."""
        if (parent_receipt is None) != (parent_receipt_sha256 is None):
            raise typer.BadParameter("parent-receipt and parent-receipt-sha256 must be supplied together")
        receipt = None if parent_receipt is None or parent_receipt_sha256 is None else SourcePin(
            path=parent_receipt.as_posix(), sha256=parent_receipt_sha256)
        _run(lambda: materialize_investment_labels(root=root, request=request,
            request_sha256=request_sha256, output=output, parent_receipt=receipt))

    @app.command("plan-investment-coverage")
    def plan_coverage(root: Path = typer.Option(...), request: Path = typer.Option(...),
                      request_sha256: str = typer.Option(...), output: Path = typer.Option(...)) -> None:
        """Freeze the full investment population and historical monthly decisions before collection."""
        _run(lambda: plan_investment_coverage(root=root, request=request, request_sha256=request_sha256, output=output))

    @app.command("build-investment-features")
    def build(root: Path = typer.Option(...), request: Path = typer.Option(...), request_sha256: str = typer.Option(...),
              output: Path = typer.Option(...)) -> None:
        """Build research-only financial snapshots from pinned SEC source and corrected clocks."""
        _run(lambda: build_investment_features(root=root, request=request, request_sha256=request_sha256, output=output))

    @app.command("evaluate-investment-predictions")
    def evaluate(root: Path = typer.Option(...), request: Path = typer.Option(...), request_sha256: str = typer.Option(...),
                 output: Path = typer.Option(...)) -> None:
        """Compare complete paired saved forecasts using mature canonical investment outcomes."""
        _run(lambda: evaluate_investment_predictions(root=root, request=request,
                                                   request_sha256=request_sha256, output=output))

    @app.command("build-investment-dataset")
    def build_dataset(root: Path = typer.Option(...), request: Path = typer.Option(...),
                      request_sha256: str = typer.Option(...), output: Path = typer.Option(...)) -> None:
        """Join verified targets to prior-session features and exact-close financial snapshots."""
        _run(lambda: build_investment_dataset(root=root, request=request, request_sha256=request_sha256, output=output))

    @app.command("train-investment-models")
    def train_models(root: Path = typer.Option(...), request: Path = typer.Option(...),
                     request_sha256: str = typer.Option(...), output: Path = typer.Option(...)) -> None:
        """Fit fixed retrospective research models and evaluate matched SPY-relative outcomes."""
        _run(lambda: train_investment_models(root=root, request=request, request_sha256=request_sha256, output=output))

    @app.command("diagnose-investment-predictions")
    def diagnose_predictions(root: Path = typer.Option(...), request: Path = typer.Option(...),
                             request_sha256: str = typer.Option(...), output: Path = typer.Option(...)) -> None:
        """Explain all saved validation errors without refitting or changing forecasts."""
        _run(lambda: diagnose_investment_predictions(root=root, request=request,
            request_sha256=request_sha256, output=output))

    @app.command("ablate-investment-features")
    def ablate_features(root: Path = typer.Option(...), request: Path = typer.Option(...),
                        request_sha256: str = typer.Option(...), output: Path = typer.Option(...)) -> None:
        """Measure separate news and financial effects on matched validation rows."""
        _run(lambda: run_investment_feature_ablation(root=root, request=request,
            request_sha256=request_sha256, output=output))

    @app.command("adjust-investment-forecasts")
    def adjust_forecasts(root: Path = typer.Option(...), request: Path = typer.Option(...),
                         request_sha256: str = typer.Option(...), output: Path = typer.Option(...)) -> None:
        """Fit clipped financial inputs and measure fixed score-only shrinkage."""
        _run(lambda: run_investment_forecast_adjustment(root=root, request=request,
            request_sha256=request_sha256, output=output))

    @app.command("train-investment-ranking")
    def train_ranking(root: Path = typer.Option(...), request: Path = typer.Option(...),
                      request_sha256: str = typer.Option(...), output: Path = typer.Option(...)) -> None:
        """Fit the fixed within-date ranking target and compare saved return-model controls."""
        _run(lambda: train_investment_ranking(root=root, request=request, request_sha256=request_sha256, output=output))

    @app.command("resolve-investment-selection-outcomes")
    def resolve_selection_outcomes(root: Path = typer.Option(...), request: Path = typer.Option(...),
                                   request_sha256: str = typer.Option(...), output: Path = typer.Option(...)) -> None:
        """Recover fixed selected stock/SPY outcomes from retained verified sources."""
        _run(lambda: resolve_investment_selection_outcomes(root=root, request=request,
            request_sha256=request_sha256, output=output))
