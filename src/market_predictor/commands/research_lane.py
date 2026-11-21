"""Thin command surface for exploratory research-lane scoreboards."""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import typer

from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError
from market_predictor.research_lane.beta_neutral import collect_wider_sec_catalysts
from market_predictor.research_lane.forward_evaluate import evaluate_forward_scores
from market_predictor.research_lane.forward_record import FORWARD_RECORD_DIRECTORY, record_forward_scores
from market_predictor.research_lane.free_backfill import BACKFILL_DIRECTORY, collect_free_sp500_backfill
from market_predictor.research_lane.panel import DEFAULT_DEVELOPMENT_END, PANEL_DIRECTORY
from market_predictor.research_lane.run import (
    score_benchmark_tilt,
    score_core_satellite_spy,
    score_fundamental_predictors,
    score_price_predictors,
    score_short_reversal,
    score_vol_regime_core_satellite,
    score_wider_beta_neutral_models,
)
from market_predictor.research_lane.vol_regime import CBOE_DIRECTORY, collect_cboe_vol_indices
from market_predictor.research_lane.wider_universe import (
    WIDER_UNIVERSE_DIRECTORY,
    UniverseSubset,
    collect_wider_us_universe,
)


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

    @app.command("collect-research-lane-free-sp500")
    def collect_free_sp500(
        root: Path = typer.Option(...),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        development_end: datetime = typer.Option(DEFAULT_DEVELOPMENT_END.isoformat(), formats=["%Y-%m-%d"],
                                                 help="Last session and filing day read; sealed rows never requested."),
        panel: Path = typer.Option(PANEL_DIRECTORY),
        output: Path = typer.Option(BACKFILL_DIRECTORY),
    ) -> None:
        """Resumably collect free SEC company_facts, 2016-2019 daily SIP bars, and pre-2019 8-K releases."""
        _invoke(collect_free_sp500_backfill, root=root, development_end=development_end.date(),
                data_root=data_root, panel=panel, output=output)

    @app.command("record-forward-research-scores")
    def record_forward(
        root: Path = typer.Option(...),
        as_of_date: str = typer.Option(
            "latest",
            help="Decision date YYYY-MM-DD to score prospectively, or 'latest' for latest completed session.",
        ),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        panel: Path = typer.Option(PANEL_DIRECTORY),
        live_prospective: bool = typer.Option(False, help="Fetch trailing live Alpaca SIP bars (> 2026-07-08)."),
        include_live_catalysts: bool = typer.Option(
            True, help="Capture live Finviz Elite, Alpaca, and SEC catalysts in live_prospective mode."
        ),
        include_live_options: bool = typer.Option(
            True, help="Capture zero-fit OPRA option straddle, 25d IV skew, and SPY/QQQ term structure."
        ),
        catalyst_top_k: int = typer.Option(100, help="Top-ranked stocks queried for per-symbol live catalysts/options."),
        skip_existing: bool = typer.Option(
            False, "--skip-existing", help="Return existing snapshot cleanly instead of raising FileExistsError."
        ),
        unseal_final_test_period: bool = typer.Option(False),
    ) -> None:
        """Publish one append-only dated forward research score snapshot without reading future outcomes."""
        raw_day = as_of_date.strip()
        if raw_day.lower() == "latest":
            parsed_day: date | None = None
        else:
            try:
                parsed_day = date.fromisoformat(raw_day)
            except ValueError as err:
                raise typer.BadParameter(f"expected YYYY-MM-DD or 'latest', got {as_of_date!r}") from err
        _invoke(
            record_forward_scores,
            root=root,
            as_of_date=parsed_day,
            data_root=data_root,
            panel=panel,
            live_prospective=live_prospective,
            include_live_catalysts=include_live_catalysts,
            include_live_options=include_live_options,
            catalyst_top_k=catalyst_top_k,
            skip_existing=skip_existing,
            unseal_final_test_period=unseal_final_test_period,
        )

    @app.command("evaluate-forward-research-scores")
    def evaluate_forward(
        root: Path = typer.Option(...),
        as_of_date: str = typer.Option(
            "latest",
            help="Completed session YYYY-MM-DD through which forward outcomes are evaluated, or 'latest'.",
        ),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        forward_dir: Path = typer.Option(FORWARD_RECORD_DIRECTORY),
        output: Path | None = typer.Option(None),
        unseal_final_test_period: bool = typer.Option(False),
    ) -> None:
        """Evaluate recorded prospective forward snapshots against matured open[t+1]->close[t+h] returns and SPY."""
        raw_day = as_of_date.strip()
        if raw_day.lower() == "latest":
            parsed_day: date | None = None
        else:
            try:
                parsed_day = date.fromisoformat(raw_day)
            except ValueError as err:
                raise typer.BadParameter(f"expected YYYY-MM-DD or 'latest', got {as_of_date!r}") from err
        _invoke(
            evaluate_forward_scores,
            root=root,
            as_of_date=parsed_day,
            data_root=data_root,
            forward_dir=forward_dir,
            output=output,
            unseal_final_test_period=unseal_final_test_period,
        )

    @app.command("collect-research-lane-wider-universe")
    def collect_wider(
        root: Path = typer.Option(...),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        development_end: datetime = typer.Option(DEFAULT_DEVELOPMENT_END.isoformat(), formats=["%Y-%m-%d"]),
        panel: Path = typer.Option(PANEL_DIRECTORY),
        sp500_backfill: Path = typer.Option(BACKFILL_DIRECTORY),
        output: Path = typer.Option(WIDER_UNIVERSE_DIRECTORY),
    ) -> None:
        """Resumably collect the $0 survivorship-checked wider US common-stock universe (<= development_end)."""
        _invoke(collect_wider_us_universe, root=root, development_end=development_end.date(),
                data_root=data_root, panel=panel, sp500_backfill=sp500_backfill, output=output)

    @app.command("collect-research-lane-sec-catalysts")
    def collect_sec_catalysts(
        root: Path = typer.Option(...),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        development_end: datetime = typer.Option(DEFAULT_DEVELOPMENT_END.isoformat(), formats=["%Y-%m-%d"]),
        wider_dir: Path = typer.Option(WIDER_UNIVERSE_DIRECTORY),
    ) -> None:
        """Resumably collect $0 SEC SC 13D/A, S-3/424B, and NT 10-K/Q catalysts (<= development_end)."""
        _invoke(collect_wider_sec_catalysts, root=root, development_end=development_end.date(),
                data_root=data_root, wider_dir=wider_dir)

    @app.command("score-wider-beta-neutral-models")
    def score_wider_beta_neutral(
        root: Path = typer.Option(...),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        development_end: datetime = typer.Option(DEFAULT_DEVELOPMENT_END.isoformat(), formats=["%Y-%m-%d"]),
        subset: UniverseSubset = typer.Option("mid_small_ex_sp500", help="mid_small_ex_sp500 or all_top1200."),
        output: Path | None = typer.Option(None),
        panel: Path = typer.Option(PANEL_DIRECTORY),
        sp500_backfill_dir: Path = typer.Option(BACKFILL_DIRECTORY),
        wider_dir: Path = typer.Option(WIDER_UNIVERSE_DIRECTORY),
        unseal_final_test_period: bool = typer.Option(False),
    ) -> None:
        """Score Step 5c beta/vol/sector-neutral Stage-1 + precise SEC catalyst Stage-2 models."""
        _invoke(score_wider_beta_neutral_models, root=root, output=output, development_end=development_end.date(),
                subset=subset, data_root=data_root, panel=panel, sp500_backfill_dir=sp500_backfill_dir,
                wider_dir=wider_dir, unseal_final_test_period=unseal_final_test_period)

    @app.command("score-core-satellite-spy")
    def score_core_satellite(
        root: Path = typer.Option(...),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        development_end: datetime = typer.Option(DEFAULT_DEVELOPMENT_END.isoformat(), formats=["%Y-%m-%d"]),
        output: Path | None = typer.Option(None),
        panel: Path = typer.Option(PANEL_DIRECTORY),
        sp500_backfill_dir: Path = typer.Option(BACKFILL_DIRECTORY),
        wider_dir: Path = typer.Option(WIDER_UNIVERSE_DIRECTORY),
        unseal_final_test_period: bool = typer.Option(False),
    ) -> None:
        """Score Step 5d unified Core SPY / S&P 500 tilt + mid/small SEC-catalyst satellite vs. SPY."""
        _invoke(
            score_core_satellite_spy,
            root=root,
            output=output,
            development_end=development_end.date(),
            data_root=data_root,
            panel=panel,
            sp500_backfill_dir=sp500_backfill_dir,
            wider_dir=wider_dir,
            unseal_final_test_period=unseal_final_test_period,
        )

    @app.command("collect-research-lane-cboe-term-structure")
    def collect_cboe_term(
        root: Path = typer.Option(...),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        development_end: datetime = typer.Option(DEFAULT_DEVELOPMENT_END.isoformat(), formats=["%Y-%m-%d"]),
        output: Path = typer.Option(CBOE_DIRECTORY),
        unseal_final_test_period: bool = typer.Option(False),
    ) -> None:
        """Collect $0 public CBOE VIX/VIX3M/VIX9D/VXN/SKEW daily history truncated <= development_end."""
        _invoke(
            collect_cboe_vol_indices,
            root=root,
            development_end=development_end.date(),
            data_root=data_root,
            output=output,
            unseal_final_test_period=unseal_final_test_period,
        )

    @app.command("score-vol-regime-core-satellite")
    def score_vol_regime(
        root: Path = typer.Option(...),
        data_root: Path | None = typer.Option(None, help="Directory containing the relocated data namespaces."),
        development_end: datetime = typer.Option(DEFAULT_DEVELOPMENT_END.isoformat(), formats=["%Y-%m-%d"]),
        output: Path | None = typer.Option(None),
        panel: Path = typer.Option(PANEL_DIRECTORY),
        sp500_backfill_dir: Path = typer.Option(BACKFILL_DIRECTORY),
        wider_dir: Path = typer.Option(WIDER_UNIVERSE_DIRECTORY),
        cboe_dir: Path = typer.Option(CBOE_DIRECTORY),
        unseal_final_test_period: bool = typer.Option(False),
    ) -> None:
        """Score Step 5e CBOE volatility term-structure & skew regime Core-Satellite vs. SPY."""
        _invoke(
            score_vol_regime_core_satellite,
            root=root,
            output=output,
            development_end=development_end.date(),
            data_root=data_root,
            panel=panel,
            sp500_backfill_dir=sp500_backfill_dir,
            wider_dir=wider_dir,
            cboe_dir=cboe_dir,
            unseal_final_test_period=unseal_final_test_period,
        )
