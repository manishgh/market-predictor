from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import typer

from market_predictor.catalysts.issuer_events.alpaca_news_collection import (
    collect_alpaca_news_history,
)
from market_predictor.config import get_settings
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.swing_history_collection import (
    AlpacaSwingDailyPageSource,
    SwingDailyPageSource,
    collect_swing_history_plan,
    load_complete_swing_history_collection,
)
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError, serialized_heavy_job
from market_predictor.sources.alpaca import AlpacaNewsPage, AlpacaSource
from market_predictor.sources.alpaca_corporate_actions import fetch_corporate_actions_page
from market_predictor.sources.official_documents import (
    OfficialDocumentSource,
    collect_official_documents,
    load_official_document_inventory,
    verify_official_document_collection,
)
from market_predictor.sources.provider_symbols import PROVIDER_ALPACA, provider_symbol
from market_predictor.swing.datasets.corporate_action_collection import collect_holding_corporate_actions
from market_predictor.swing.datasets.initial_fit_raw_share_plan import verified_initial_fit_raw_share_plan


def register_swing_collection_commands(app: typer.Typer, console: Any) -> None:
    @app.command("collect-swing-initial-fit-raw-prices")
    def collect_swing_initial_fit_raw_prices_command(
        root: Path = typer.Option(Path(".")),
        config: Path = typer.Option(Path("configs/swing_initial_fit_raw_share_plan.toml")),
        plan_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        expected_plan_sha256: str = typer.Option(..., help="Independently saved acquisition authority-file hash."),
        max_units: int | None = typer.Option(None, min=1),
        offline: bool = typer.Option(False, help="Replay a completed archive without network calls."),
    ) -> None:
        """Verify exact cohort/session requirements, then collect raw SIP prices."""
        root = root.resolve()
        output = (root / out_dir).resolve()
        plan_path = (root / plan_dir).resolve()
        sources: list[AlpacaSource] = []
        try:
            with verified_initial_fit_raw_share_plan(root, config, plan_path, expected_plan_sha256=expected_plan_sha256) as plan:
                if (output == root or not output.is_relative_to(root) or output.is_relative_to(plan_path)
                        or plan_path.is_relative_to(output)):
                    raise DataReadinessError("raw collection output must be separate from its plan and inside repository")
                if offline:
                    result = load_complete_swing_history_collection(output, plan_directory=plan_path,
                        expected_adjustment="raw", expected_plan_authority_sha256=expected_plan_sha256)
                else:
                    settings = get_settings()
                    if not settings.has_alpaca or settings.alpaca_stock_feed.strip().lower() != "sip":
                        raise typer.BadParameter("Configured Alpaca credentials and SIP are required")

                    def source_factory() -> SwingDailyPageSource:
                        source = AlpacaSource(settings)
                        sources.append(source)
                        return AlpacaSwingDailyPageSource(source)

                    result = collect_swing_history_plan(plan_directory=plan_path, output_directory=output,
                        source_factory=source_factory, provider_symbol_for=lambda ticker: str(plan["provider_symbols"][ticker]),
                        maximum_units_this_run=max_units, expected_plan_authority_sha256=expected_plan_sha256)
        except HeavyJobBusyError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from exc
        finally:
            for source in sources:
                source.client.session.close()
        console.print({key: result[key] for key in (
            "status", "requested_units", "terminal_units", "observed_units", "unavailable_units", "failed_units", "stop_reason",
        )})
        if result["status"] not in {"complete", "complete_with_unavailable"}:
            raise typer.Exit(code=2)

    @app.command("collect-swing-holding-corporate-actions")
    def collect_swing_holding_corporate_actions_command(
        root: Path = typer.Option(Path(".")),
        config: Path = typer.Option(Path("configs/swing_holding_corporate_actions.toml")),
        out_dir: Path = typer.Option(..., help="New or matching resumable corporate-action response archive."),
        offline: bool = typer.Option(False),
        expected_audit_sha256: str | None = typer.Option(None, help="Independent report hash, required for offline replay."),
    ) -> None:
        """Collect historical accounting evidence without treating retrieval as announcement time."""
        source: AlpacaSource | None = None

        def fetch(params: dict[str, Any], maximum_bytes: int) -> Any:
            nonlocal source
            if source is None:
                source = AlpacaSource(get_settings())
            return fetch_corporate_actions_page(source, ticker=params["symbols"],
                start=date.fromisoformat(params["start"]), end=date.fromisoformat(params["end"]),
                page_token=params.get("page_token"), limit=params["limit"], maximum_body_bytes=maximum_bytes)

        try:
            result = collect_holding_corporate_actions(root, config, out_dir,
                fetch=None if offline else fetch, expected_audit_sha256=expected_audit_sha256)
        except HeavyJobBusyError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from exc
        finally:
            if source is not None:
                source.client.session.close()
        console.print({key: result[key] for key in (
            "status", "requested_tickers", "acquired_tickers", "action_counts", "audit_sha256", "accounting_eligible",
        )})
        if result["status"] != "collected_unreviewed":
            raise typer.Exit(code=2)

    @app.command("collect-swing-holding-source-documents")
    def collect_swing_holding_source_documents_command(
        inventory: Path = typer.Option(Path("configs/swing_holding_source_documents.toml")),
        out_dir: Path = typer.Option(..., help="New or matching resumable official response archive."),
        offline: bool = typer.Option(False, help="Verify existing bytes without a network request."),
    ) -> None:
        """Retain official evidence without approving its accounting interpretation."""
        policy = load_official_document_inventory(inventory)
        if offline:
            result = verify_official_document_collection(out_dir, policy)
        else:
            source = OfficialDocumentSource(get_settings())
            try:
                result = collect_official_documents(inventory=policy, output_directory=out_dir, fetch=source.fetch)
            finally:
                source.close()
        console.print({key: result[key] for key in (
            "status", "requested_documents", "archived_documents", "interpretation_status", "accounting_eligible",
        )})
        if result["status"] != "collected_unreviewed":
            raise typer.Exit(code=2)

    @app.command("collect-edge-rebuild-swing-history")
    @serialized_heavy_job("collect-edge-rebuild-swing-history")
    def collect_edge_rebuild_swing_history_command(
        plan_dir: Path = typer.Option(
            ...,
            help="Complete swing_history_acquisition_plan.v2 authority directory.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="New or matching resumable exact-unit collection directory.",
        ),
        max_units: int | None = typer.Option(
            None,
            min=1,
            help="Optional resumable operational batch limit.",
        ),
    ) -> None:
        """Collect exact authority-bound swing daily units from Alpaca SIP."""

        settings = get_settings()
        if not settings.has_alpaca:
            raise typer.BadParameter(
                "ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY are required"
            )
        if settings.alpaca_stock_feed.strip().lower() != "sip":
            raise typer.BadParameter("ALPACA_STOCK_FEED must be sip")

        def source_factory() -> SwingDailyPageSource:
            return AlpacaSwingDailyPageSource(AlpacaSource(settings))

        result = collect_swing_history_plan(
            plan_directory=plan_dir,
            output_directory=out_dir,
            source_factory=source_factory,
            provider_symbol_for=lambda ticker: provider_symbol(
                ticker,
                PROVIDER_ALPACA,
            ),
            maximum_units_this_run=max_units,
        )
        console.print(
            {
                key: result[key]
                for key in (
                    "status",
                    "requested_units",
                    "terminal_units",
                    "observed_units",
                    "unavailable_units",
                    "failed_units",
                    "unattempted_units",
                    "resumed_units",
                    "stop_reason",
                )
            }
        )
        if result["status"] not in {"complete", "complete_with_unavailable"}:
            raise typer.Exit(code=2)

    @app.command("collect-alpaca-news-history")
    @serialized_heavy_job("collect-alpaca-news-history")
    def collect_alpaca_news_history_command(
        memberships: Path = typer.Option(
            ...,
            help="Hash-verified point-in-time membership artifact with security IDs.",
        ),
        start_date: str = typer.Option(
            ...,
            help="Inclusive first publication date YYYY-MM-DD.",
        ),
        end_date: str = typer.Option(
            ...,
            help="Inclusive frozen final publication date YYYY-MM-DD.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="Resumable raw-page and research-event collection directory.",
        ),
        workers: int = typer.Option(
            2,
            min=1,
            max=4,
            help="Bounded independent network workers; no model work is started.",
        ),
        chunk_days: int = typer.Option(
            92,
            min=7,
            max=366,
            help="Half-open provider request chunk length.",
        ),
    ) -> None:
        """Collect publication-time-proxy Alpaca/Benzinga history immutably."""

        settings = get_settings()
        if not settings.has_alpaca:
            raise typer.BadParameter(
                "ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY are required"
            )
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
        except ValueError as exc:
            raise typer.BadParameter(
                "start-date and end-date must be YYYY-MM-DD"
            ) from exc
        source = AlpacaSource(settings)

        def fetch_page(
            symbol: str,
            start_at: datetime,
            end_at: datetime,
            page_token: str | None,
        ) -> AlpacaNewsPage:
            return source.fetch_news_page(
                symbol,
                start_at,
                end_at,
                page_token=page_token,
                include_content=True,
                limit=50,
            )

        result = collect_alpaca_news_history(
            memberships_path=memberships,
            start_date=start,
            end_date=end,
            out_dir=out_dir,
            fetch_page=fetch_page,
            provider_symbol_for=lambda ticker: provider_symbol(
                ticker,
                PROVIDER_ALPACA,
            ),
            workers=workers,
            chunk_days=chunk_days,
        )
        console.print(
            {
                "status": result.status,
                "requested_chunks": result.requested_chunks,
                "observed_chunks": result.observed_chunks,
                "empty_chunks": result.empty_chunks,
                "failed_chunks": list(result.failed_chunks),
                "skipped_chunks": result.skipped_chunks,
                "manifest": (
                    str(result.manifest_path)
                    if result.manifest_path is not None
                    else None
                ),
                "status_path": str(result.status_path),
                "production_ready": False,
                "availability_policy": "provider_publication_proxy",
            }
        )
        if result.status == "incomplete":
            raise typer.Exit(code=2)
