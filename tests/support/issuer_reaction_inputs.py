"""Explicit unit-only replacement for the original snapshot input owner.

This adapts the small CURRENT synthetic relationship publication. It never proves
that an actual original research receipt or historical source is admissible.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.research import issuer_reaction_publication as publisher
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.issuer_reaction_publication import IssuerReactionPublicationPolicy
from market_predictor.swing.contracts.return_feature_profiles import ReturnRelationshipSources
from market_predictor.swing.contracts.return_relationship_publication import ReturnRelationshipPublicationPolicy
from market_predictor.swing.datasets.adjusted_history_bindings import bind_adjusted_history_decisions
from market_predictor.swing.datasets.return_relationship_integrity import check_files, read_object
from market_predictor.swing.datasets.return_relationship_parent import verify_parent
from market_predictor.swing.datasets.return_relationship_rows import read_spy, read_stock
from market_predictor.swing.datasets.return_relationship_sources import verify_source_context
from market_predictor.swing.datasets.return_relationship_verification import validate_return_relationship_receipt


class SyntheticOriginalInputs:
    """Scoped adapter around genuine synthetic query ownership and physical reads."""

    def __init__(self, root: Path, policy: IssuerReactionPublicationPolicy) -> None:
        self.root = root
        self.active = True
        self.receipt_pin = policy.original_snapshot_replay
        receipt = read_object(root / self.receipt_pin.path, self.receipt_pin.sha256)
        assert receipt == {"synthetic_unit_test_only": True, "original_snapshot_replayed": False}
        parent_receipt = read_object(root / policy.parent_saved_row_verification.path, policy.parent_saved_row_verification.sha256)
        self.parent = validate_return_relationship_receipt(root, policy.parent_publication, parent_receipt)
        relationship_policy = ReturnRelationshipPublicationPolicy.model_validate(self.parent.request["policy"])
        self.context = verify_source_context(root, relationship_policy, verify_parent(root, relationship_policy))
        self.original_sources = ReturnRelationshipSources.model_validate(self.parent.request["sources"])
        self.source_files = {
            **self.parent.source_files, self.receipt_pin.path: self.receipt_pin.sha256,
            policy.parent_saved_row_verification.path: policy.parent_saved_row_verification.sha256,
        }
        self.historical_implementation_files = dict(self.parent.request["historical_implementation_files"])
        self.source_basis = self.parent.request["source_basis"]
        self.history_sessions = tuple(day.date() for day in xcals.get_calendar("XNYS").sessions_in_range(
            relationship_policy.source_start, relationship_policy.source_end))
        self.directory = (root / policy.parent_publication.path).parent

    def recheck(self) -> None:
        self._ensure_active()
        check_files(self.root, self.source_files)

    def _ensure_active(self) -> None:
        if not self.active:
            raise DataReadinessError("synthetic original input context is closed")

    def read_parent_month(self, month: str) -> pd.DataFrame:
        self._ensure_active()
        child = self.parent.months[month]["profiles"]["technical_relationships"]
        path = self.directory / child["path"]
        check_files(self.root, {path.relative_to(self.root).as_posix(): child["sha256"],
            manifest_path_for(path).relative_to(self.root).as_posix(): child["manifest_sha256"]})
        rows, _ = load_canonical_artifact(path, allow_research=True)
        return rows

    def iter_month_groups(self, month: str) -> Iterator[tuple[str, pd.DataFrame]]:
        rows = self.read_parent_month(month)
        bound = bind_adjusted_history_decisions(rows, self.context.bindings)
        for (identity, unit), group in bound.groupby(["security_id", "source_group"], sort=True):
            key = json_sha256([identity, unit])
            assert key in self.parent.request["stock_inventory"]
            yield key, rows.loc[rows.decision_id.isin(group.decision_id)].reset_index(drop=True)

    def read_stock(self, key: str) -> pd.DataFrame:
        self._ensure_active()
        return read_stock(self.root, self.context, self.parent.request["stock_inventory"][key])

    def read_spy(self) -> pd.DataFrame:
        self._ensure_active()
        return read_spy(self.context)


def install_synthetic_original_inputs(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, Any]:
    """Patch only the owner boundary; qualifier/hash/feature kernels stay native."""
    lifecycle: dict[str, Any] = {"entered": 0, "exited": 0, "contexts": []}

    @contextmanager
    def synthetic_inputs(actual_root: Path, policy: IssuerReactionPublicationPolicy) -> Iterator[SyntheticOriginalInputs]:
        assert actual_root == root
        context = SyntheticOriginalInputs(root, policy)
        context.recheck()
        lifecycle["entered"] += 1
        lifecycle["contexts"].append(context)
        try:
            yield context
            context.recheck()
        finally:
            context.active = False
            lifecycle["exited"] += 1

    monkeypatch.setattr(publisher, "_verified_original_relationship_inputs", synthetic_inputs)
    return lifecycle


def synthetic_replay_pin(root: Path) -> SourcePin:
    path = root / "data/reports/synthetic_original_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"synthetic_unit_test_only": true, "original_snapshot_replayed": false}\n', encoding="utf-8")
    return SourcePin(path=path.relative_to(root).as_posix(), sha256=file_sha256(path))
