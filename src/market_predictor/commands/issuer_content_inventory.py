"""Research CLI adapters for original saved-news content inspection."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError
from market_predictor.research.issuer_candidate_derivative import (
    publish_issuer_candidate_derivative,
    verify_issuer_candidate_derivative,
)
from market_predictor.research.issuer_content_cohort_inventory import publish_cohort_content_inventory
from market_predictor.research.issuer_content_inventory import publish_saved_content_inventory
from market_predictor.research.issuer_content_qualification_authority import publish_issuer_content_qualification
from market_predictor.research.issuer_content_review_population import publish_content_review_population
from market_predictor.research.issuer_review_packets import publish_issuer_review_packets, verify_issuer_review_packets
from market_predictor.research.legacy_query_identity_proofs import publish_legacy_query_identity_proofs
from market_predictor.research.sec_acceptance_clock import collect_sec_clock_pages, publish_sec_acceptance_clock
from market_predictor.research.sec_filing_documents import collect_sec_filing_documents
from market_predictor.research.sec_form_inventory import MODES, Mode, publish_sec_form_inventory
from market_predictor.swing.contracts.holding_materialization import SourcePin as HoldingSourcePin
from market_predictor.swing.datasets.initial_fit_issuer_news import LAST_INITIAL_FIT_CUTOFF
from market_predictor.swing.datasets.issuer_news_preparation import FIRST

_COLLECTION_OUTCOME = ("stop_status_code", "stopped_unit", "phase", "units_without_final_outcome", "manifest_sha256",
                       "totals")


def _publish(call: Callable[[], dict[str, Any]], keys: tuple[str, ...], optional: tuple[str, ...] = ()) -> None:
    try:
        report = call()
    except HeavyJobBusyError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from error
    except (DataReadinessError, OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({key: report[key] for key in (*keys, *(key for key in optional if key in report))}, sort_keys=True))


def register_issuer_content_commands(app: typer.Typer) -> None:
    @app.command("publish-issuer-review-packets")
    def publish_review_packets(
        config: Path = typer.Option(...), expected_config_sha256: str = typer.Option(...),
        output: Path = typer.Option(...), root: Path = typer.Option(Path(".")),
    ) -> None:
        """Publish complete blind source packets and a frozen fresh review frame."""
        _publish(lambda: publish_issuer_review_packets(root=root, config=config,
            expected_config_sha256=expected_config_sha256, output=output),
            ("status", "counts", "manifest_sha256", "qualification_established",
             "training_eligible", "serving_eligible", "promotion_eligible", "economic_eligible"))

    @app.command("verify-issuer-review-packets")
    def verify_review_packets(
        publication: Path = typer.Option(...), publication_sha256: str = typer.Option(...),
        root: Path = typer.Option(Path(".")),
    ) -> None:
        """Reproduce source packets and exclusions without approving events or models."""
        def verify() -> dict[str, Any]:
            verified = verify_issuer_review_packets(root=root,
                publication=HoldingSourcePin(path=publication.as_posix(), sha256=publication_sha256))
            return {"status": "passed_blind_review_packet_replay_only", "counts": verified.counts,
                    "manifest_sha256": verified.publication.sha256, "qualification_established": False,
                    "training_eligible": False, "serving_eligible": False, "promotion_eligible": False,
                    "economic_eligible": False}

        _publish(verify, ("status", "counts", "manifest_sha256", "qualification_established",
                         "training_eligible", "serving_eligible", "promotion_eligible", "economic_eligible"))

    @app.command("derive-saved-issuer-candidates")
    def derive_candidates(
        parent_population: Path = typer.Option(...), parent_population_sha256: str = typer.Option(...),
        output: Path = typer.Option(...), root: Path = typer.Option(Path(".")),
    ) -> None:
        """Replay revised event statements from the original text without recollecting news."""
        _publish(lambda: publish_issuer_candidate_derivative(root=root,
            parent_population=HoldingSourcePin(path=parent_population.as_posix(), sha256=parent_population_sha256),
            output=output), ("status", "counts", "manifest_sha256", "qualification_established",
                            "training_eligible", "serving_eligible", "promotion_eligible"))

    @app.command("verify-saved-issuer-candidates")
    def verify_candidates(
        publication: Path = typer.Option(...), publication_sha256: str = typer.Option(...),
        root: Path = typer.Option(Path(".")),
    ) -> None:
        """Independently replay every candidate derivative row; never qualify news for training."""
        def verify() -> dict[str, Any]:
            verified = verify_issuer_candidate_derivative(root=root,
                publication=HoldingSourcePin(path=publication.as_posix(), sha256=publication_sha256))
            return {"status": "passed_candidate_replay_only", "counts": verified.counts,
                    "manifest_sha256": verified.publication.sha256, "qualification_established": False,
                    "training_eligible": False, "serving_eligible": False, "promotion_eligible": False}

        _publish(verify, ("status", "counts", "manifest_sha256", "qualification_established",
                         "training_eligible", "serving_eligible", "promotion_eligible"))

    @app.command("qualify-issuer-content")
    def qualify_content(
        population_authority: Path = typer.Option(...), population_sha256: str = typer.Option(...),
        reviewer_one: Path = typer.Option(...), reviewer_one_sha256: str = typer.Option(...),
        reviewer_two: Path = typer.Option(...), reviewer_two_sha256: str = typer.Option(...),
        output: Path = typer.Option(...), root: Path = typer.Option(Path(".")),
    ) -> None:
        """Measure two independent source reviews and publish historical feature evidence."""
        _publish(lambda: publish_issuer_content_qualification(
            root=root,
            population_authority=HoldingSourcePin(path=population_authority.as_posix(), sha256=population_sha256),
            reviewer_files=(
                HoldingSourcePin(path=reviewer_one.as_posix(), sha256=reviewer_one_sha256),
                HoldingSourcePin(path=reviewer_two.as_posix(), sha256=reviewer_two_sha256),
            ), output=output,
        ), ("status", "counts", "research_feature_eligible", "training_eligible", "serving_eligible", "manifest_sha256"))

    @app.command("prepare-issuer-content-review")
    def prepare_review(
        config: Path = typer.Option(...), config_sha256: str = typer.Option(...), output: Path = typer.Option(...),
        root: Path = typer.Option(Path(".")), resume_checkpoint_sha256: str | None = typer.Option(None),
    ) -> None:
        """Prepare initial-fit source-only candidate/negative review evidence; never feature admission."""
        _publish(lambda: publish_content_review_population(root=root, config=config, config_sha256=config_sha256,
            output=output, resume_checkpoint_sha256=resume_checkpoint_sha256),
            ("status", "totals", "manifest_sha256", "training_eligible", "serving_eligible"))

    @app.command("inspect-saved-issuer-content")
    def inspect(
        event_artifact: Path = typer.Option(...), event_sha256: str = typer.Option(...),
        event_manifest: Path = typer.Option(...), manifest_sha256: str = typer.Option(...),
        security_id: str = typer.Option(...), ticker: str = typer.Option(...),
        output: Path = typer.Option(...), root: Path = typer.Option(Path(".")),
        start_utc: str = typer.Option(FIRST.isoformat(), help="Not before the initial-fit issuer-news start."),
        cutoff_utc: str = typer.Option(LAST_INITIAL_FIT_CUTOFF.isoformat(), help="Not after the initial-fit cutoff."),
    ) -> None:
        """Verify one original query's content fields without admitting a model feature."""
        _publish(lambda: publish_saved_content_inventory(
            root=root, event_artifact=SourcePin(path=event_artifact.as_posix(), sha256=event_sha256),
            event_manifest=SourcePin(path=event_manifest.as_posix(), sha256=manifest_sha256),
            security_id=security_id, ticker=ticker, output=output,
            start_utc=pd.Timestamp(start_utc), cutoff_utc=pd.Timestamp(cutoff_utc),
        ), ("status", "rows", "manifest_sha256", "training_eligible", "serving_eligible"))

    @app.command("inspect-issuer-content-cohort")
    def inspect_cohort(
        config: Path = typer.Option(...), config_sha256: str = typer.Option(...), output: Path = typer.Option(...),
        root: Path = typer.Option(Path(".")), resume_checkpoint_sha256: str | None = typer.Option(None),
    ) -> None:
        """Inventory every pinned initial-fit news query unit without admitting a model feature."""
        _publish(lambda: publish_cohort_content_inventory(root=root, config=config, config_sha256=config_sha256,
            output=output, resume_checkpoint_sha256=resume_checkpoint_sha256),
            ("status", "records_rows", "manifest_sha256", "training_eligible", "serving_eligible"))

    @app.command("prove-legacy-query-identities")
    def prove(
        config: Path = typer.Option(...), config_sha256: str = typer.Option(...), output: Path = typer.Option(...),
        root: Path = typer.Option(Path(".")),
    ) -> None:
        """Prove or reject legacy news-query identities the CIK bridge left unconverted; never admission."""
        _publish(lambda: publish_legacy_query_identity_proofs(root=root, config=config, config_sha256=config_sha256,
            output=output), ("status", "totals", "manifest_sha256", "training_eligible", "serving_eligible"))

    @app.command("collect-sec-clock-pages")
    def collect_clock_pages(
        collection_authority: Path = typer.Option(...), collection_sha256: str = typer.Option(...),
        output: Path = typer.Option(...), root: Path = typer.Option(Path(".")),
        resume_checkpoint_sha256: str | None = typer.Option(None),
    ) -> None:
        """Collect EDGAR detail pages of each issuer's first and last archive filing in every form group."""
        _publish(lambda: collect_sec_clock_pages(
            root=root, collection={"path": collection_authority.as_posix(), "sha256": collection_sha256}, output=output,
            resume_checkpoint_sha256=resume_checkpoint_sha256),
            ("status", "checkpoint_sha256"), _COLLECTION_OUTCOME)

    @app.command("publish-sec-acceptance-clock")
    def publish_clock(
        collection_authority: Path = typer.Option(...), collection_sha256: str = typer.Option(...),
        pages_manifest: Path = typer.Option(...), pages_sha256: str = typer.Option(...), output: Path = typer.Option(...),
        root: Path = typer.Option(Path(".")),
    ) -> None:
        """Decide each SEC issuer's acceptance-clock convention from EDGAR's own pages; never a guess."""
        _publish(lambda: publish_sec_acceptance_clock(
            root=root, collection={"path": collection_authority.as_posix(), "sha256": collection_sha256},
            pages={"path": pages_manifest.as_posix(), "sha256": pages_sha256}, output=output),
            ("status", "manifest_sha256", "totals"))

    @app.command("inspect-sec-form-inventory")
    def inspect_sec(
        config: Path = typer.Option(...), config_sha256: str = typer.Option(...), output: Path = typer.Option(...),
        mode: str = typer.Option("initial_fit", help=f"One of {', '.join(MODES)}."), root: Path = typer.Option(Path(".")),
    ) -> None:
        """Inventory saved SEC filing metadata and documents for the approved cohort; never content or admission."""
        if mode not in MODES:
            typer.echo(f"mode must be one of {MODES}", err=True)
            raise typer.Exit(code=2)
        selected: Mode = "initial_fit" if mode == "initial_fit" else "later_sealed"
        _publish(lambda: publish_sec_form_inventory(root=root, config=config, config_sha256=config_sha256, output=output,
            mode=selected), ("status", "mode", "manifest_sha256", "training_eligible", "serving_eligible"))

    @app.command("collect-sec-filing-documents")
    def collect_sec_documents(
        inventory_manifest: Path = typer.Option(...), inventory_sha256: str = typer.Option(...), output: Path = typer.Option(...),
        root: Path = typer.Option(Path(".")), resume_checkpoint_sha256: str | None = typer.Option(None),
        pilot_accession: list[str] = typer.Option([], help="Restrict to these selected accessions (pilot runs only)."),
    ) -> None:
        """Collect detail pages, primary documents and EX-99 exhibits of selected SEC 8-Ks; never content review."""
        _publish(lambda: collect_sec_filing_documents(
            root=root, inventory={"path": inventory_manifest.as_posix(), "sha256": inventory_sha256}, output=output,
            resume_checkpoint_sha256=resume_checkpoint_sha256, pilot_accessions=tuple(pilot_accession)),
            ("status", "checkpoint_sha256"), _COLLECTION_OUTCOME)
