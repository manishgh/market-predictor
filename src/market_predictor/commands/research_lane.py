"""Thin command surface for exploratory research-lane scoreboards."""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError
from market_predictor.research_lane.panel import DEFAULT_DEVELOPMENT_END, PANEL_DIRECTORY
from market_predictor.research_lane.run import (
    score_benchmark_tilt,
    score_fundamental_predictors,
    score_price_predictors,
    score_short_reversal,
    score_swing_setups,
    score_technical_models,
)
from market_predictor.research_lane.setups import VolatilityMode


def _invoke(function: Callable[..., dict[str, Any]], **arguments: Any) -> None:
    try:
        result = function(**arguments)
    except HeavyJobBusyError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from error
    except (DataReadinessError, MemoryBudgetError, OSError, ValueError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True, allow_nan=False))


def register_research_lane_commands(app: typer.Typer) -> None:
    @app.command("score-price-predictors")
    def score(
        root: Path = typer.Option(...),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        development_end: datetime = typer.Option(DEFAULT_DEVELOPMENT_END.isoformat(), formats=["%Y-%m-%d"],
                                                 help="Last session loaded; later rows are never read."),
        output: Path | None = typer.Option(None, help="New directory; default data/research/research_lane/"
                                                      "price_predictors_<UTC timestamp>."),
        panel: Path = typer.Option(PANEL_DIRECTORY),
        unseal_final_test_period: bool = typer.Option(
            False, help="Permit a development end inside the sealed final-test period (2024-05-29 onward)."),
    ) -> None:
        """Score frozen price predictors by later research-label return on development history (no model fit)."""
        _invoke(score_price_predictors, root=root, output=output, development_end=development_end.date(),
                data_root=data_root, panel=panel, unseal_final_test_period=unseal_final_test_period)

    @app.command("score-fundamental-predictors")
    def score_fundamental(
        root: Path = typer.Option(...),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        development_end: datetime = typer.Option(DEFAULT_DEVELOPMENT_END.isoformat(), formats=["%Y-%m-%d"],
                                                 help="Last session, filing date and acceptance day read."),
        output: Path | None = typer.Option(None, help="New directory; default data/research/research_lane/"
                                                      "fundamental_predictors_<UTC timestamp>."),
        panel: Path = typer.Option(PANEL_DIRECTORY),
        unseal_final_test_period: bool = typer.Option(
            False, help="Permit a development end inside the sealed final-test period (2024-05-29 onward)."),
    ) -> None:
        """Score frozen earnings-reaction and financial-statement inputs by later research-label return (no fit)."""
        _invoke(score_fundamental_predictors, root=root, output=output, development_end=development_end.date(),
                data_root=data_root, panel=panel, unseal_final_test_period=unseal_final_test_period)

    @app.command("score-technical-models")
    def score_technical(
        root: Path = typer.Option(...),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        development_end: datetime = typer.Option(DEFAULT_DEVELOPMENT_END.isoformat(), formats=["%Y-%m-%d"],
                                                 help="Last session loaded; later rows are never read."),
        output: Path | None = typer.Option(None, help="New directory; default data/research/research_lane/"
                                                      "technical_models_<UTC timestamp>."),
        panel: Path = typer.Option(PANEL_DIRECTORY),
        unseal_final_test_period: bool = typer.Option(
            False, help="Permit a development end inside the sealed final-test period (2024-05-29 onward)."),
    ) -> None:
        """Score additional technical indicators alone and walk-forward ridge/boosted-tree combinations."""
        _invoke(score_technical_models, root=root, output=output, development_end=development_end.date(),
                data_root=data_root, panel=panel, unseal_final_test_period=unseal_final_test_period)

    @app.command("score-short-reversal")
    def score_reversal(
        root: Path = typer.Option(...),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        development_end: datetime = typer.Option(DEFAULT_DEVELOPMENT_END.isoformat(), formats=["%Y-%m-%d"],
                                                 help="Last session loaded; later rows are never read."),
        output: Path | None = typer.Option(None, help="New directory; default data/research/research_lane/"
                                                      "short_reversal_<UTC timestamp>."),
        panel: Path = typer.Option(PANEL_DIRECTORY),
        unseal_final_test_period: bool = typer.Option(
            False, help="Permit a development end inside the sealed final-test period (2024-05-29 onward)."),
    ) -> None:
        """Score frozen 1-3 session reversal inputs at 1, 2, 3 and 5 sessions with a pre-registered reading rule."""
        _invoke(score_short_reversal, root=root, output=output, development_end=development_end.date(),
                data_root=data_root, panel=panel, unseal_final_test_period=unseal_final_test_period)

    @app.command("score-swing-setups")
    def score_setups(
        root: Path = typer.Option(...),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        development_end: datetime = typer.Option(DEFAULT_DEVELOPMENT_END.isoformat(), formats=["%Y-%m-%d"],
                                                 help="Last session and catalyst day read; later rows never read."),
        output: Path | None = typer.Option(None, help="New directory; default data/research/research_lane/"
                                                      "swing_setups_<UTC timestamp>."),
        panel: Path = typer.Option(PANEL_DIRECTORY),
        volatility_mode: str = typer.Option("wilder_atr_14", help="wilder_atr_14 or mtr_14 (Hypothesis V2-01)."),
        unseal_final_test_period: bool = typer.Option(
            False, help="Permit a development end inside the sealed final-test period (2024-05-29 onward)."),
    ) -> None:
        """Score Stage-1 deterministic swing setups and Stage-2 purged walk-forward meta-models."""
        if volatility_mode not in ("wilder_atr_14", "mtr_14"):
            typer.echo(f"unsupported volatility_mode: {volatility_mode}", err=True)
            raise typer.Exit(code=2)
        mode: VolatilityMode = "wilder_atr_14" if volatility_mode == "wilder_atr_14" else "mtr_14"
        _invoke(score_swing_setups, root=root, output=output, development_end=development_end.date(),
                data_root=data_root, panel=panel, volatility_mode=mode,
                unseal_final_test_period=unseal_final_test_period)

    @app.command("score-benchmark-tilt")
    def score_tilt(
        root: Path = typer.Option(...),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        development_end: datetime = typer.Option(DEFAULT_DEVELOPMENT_END.isoformat(), formats=["%Y-%m-%d"],
                                                 help="Last session and filing day read; later rows never read."),
        output: Path | None = typer.Option(None, help="New directory; default data/research/research_lane/"
                                                      "benchmark_tilt_<UTC timestamp>."),
        panel: Path = typer.Option(PANEL_DIRECTORY),
        unseal_final_test_period: bool = typer.Option(
            False, help="Permit a development end inside the sealed final-test period (2024-05-29 onward)."),
    ) -> None:
        """Score Option C cap-weighted benchmark tilts and turnover-buffered reversal/quality baskets."""
        _invoke(score_benchmark_tilt, root=root, output=output, development_end=development_end.date(),
                data_root=data_root, panel=panel, unseal_final_test_period=unseal_final_test_period)
