from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import typer

from market_predictor.strategy_governance import (
    validate_strategy_execution_ledger,
)
from market_predictor.strategy_research_contracts import (
    validate_strategy_research_contracts,
)


def register_strategy_governance_commands(app: typer.Typer, console: Any) -> None:
    @app.command("validate-strategy-execution-ledger")
    def validate_strategy_execution_ledger_command(
        ledger: Path = typer.Option(
            Path("docs/strategy_execution_ledger.json"),
            help="Authoritative strategy execution ledger.",
        ),
        repository_root: Path = typer.Option(
            Path("."),
            help="Repository root used to resolve evidence and Git identities.",
        ),
        verify_git: bool = typer.Option(
            True,
            help="Verify bound commits against local remote-tracking refs.",
        ),
        report_out: Path | None = typer.Option(
            None,
            help="Optional deterministic JSON validation report.",
        ),
    ) -> None:
        """Validate plan coverage, closure evidence, hashes, and Git bindings."""
        report = validate_strategy_execution_ledger(
            ledger,
            repository_root=repository_root,
            verify_git=verify_git,
        )
        if report_out is not None:
            _write_json_atomic(report_out, report)
        console.print(json.dumps(report, indent=2, sort_keys=True))

    @app.command("validate-strategy-research-contracts")
    def validate_strategy_research_contracts_command(
        ledger: Path = typer.Option(
            Path("docs/strategy_execution_ledger.json"),
            help="Authoritative strategy execution ledger.",
        ),
        hypotheses: Path = typer.Option(
            Path("docs/strategy_hypothesis_registry.json"),
            help="Bounded strategy research hypothesis registry.",
        ),
        policy: Path = typer.Option(
            Path("configs/strategy_research_governance.toml"),
            help="Shared research, budget, and retirement policy.",
        ),
        reference_models: Path = typer.Option(
            Path("docs/reference_model_inventory.json"),
            help="Non-serving generic reference-model inventory.",
        ),
        repository_root: Path = typer.Option(
            Path("."),
            help="Repository root used to resolve contracts and Git identities.",
        ),
        verify_git: bool = typer.Option(
            True,
            help="Verify bound commits against local remote-tracking refs.",
        ),
        report_out: Path | None = typer.Option(
            None,
            help="Optional deterministic JSON validation report.",
        ),
    ) -> None:
        """Validate KS0 strategy hypotheses and frozen shared assumptions."""
        report = validate_strategy_research_contracts(
            ledger_path=ledger,
            hypothesis_registry_path=hypotheses,
            policy_path=policy,
            reference_inventory_path=reference_models,
            repository_root=repository_root,
            verify_git=verify_git,
        )
        if report_out is not None:
            _write_json_atomic(report_out, report)
        console.print(json.dumps(report, indent=2, sort_keys=True))


def _write_json_atomic(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
