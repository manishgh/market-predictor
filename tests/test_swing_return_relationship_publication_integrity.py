"""Fail-closed distinctions between historical provenance and current execution."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.system_memory import SystemMemory
from market_predictor.swing.contracts.return_relationship_publication import ReturnRelationshipPublicationPolicy
from market_predictor.swing.datasets import return_relationship_integrity as integrity
from market_predictor.swing.datasets.return_relationship_parent import _historical_sources
from market_predictor.swing.datasets.return_relationship_rows import ADDITIONS, assert_parent_parity
from market_predictor.swing.datasets.return_relationship_storage import stage_baseline
from tests.test_swing_training_readiness import _json


def _historical(root: Path, name: str, *, declared: bool = True) -> tuple[dict[str, str], Any]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"original source bytes\n")
    pins = {name: file_sha256(path)}
    if declared:
        request = root / "data/replay/_request.json"
        _json(request, dict(schema="market_predictor.predictor_implementation_replay.v1.request",
            current_implementation_files=dict(pins)))
        pins[request.relative_to(root).as_posix()] = file_sha256(request)
    policy = SimpleNamespace(parent_publication=SimpleNamespace(sha256="1" * 64))
    return pins, policy


def test_declared_ancestral_code_is_not_reexecuted_or_claimed_equivalent(tmp_path: Path) -> None:
    name = "src/market_predictor/canonical/normalize.py"
    declared, policy = _historical(tmp_path, name)
    (tmp_path / name).write_bytes(b"current implementation has changed\n")
    live, historical = _historical_sources(tmp_path, declared, policy)
    assert historical == {name: declared[name]}
    assert name not in live
    assert historical[name] != file_sha256(tmp_path / name)


def test_historical_disposition_does_not_exempt_current_executed_file(tmp_path: Path) -> None:
    name = "src/market_predictor/canonical/normalize.py"
    declared, policy = _historical(tmp_path, name)
    path = tmp_path / name
    path.write_bytes(b"fresh executable implementation\n")
    live, historical = _historical_sources(tmp_path, declared, policy)
    current = {name: file_sha256(path)}
    admitted = integrity.pins(tmp_path, live, current)
    assert admitted[name] != historical[name]
    path.write_bytes(b"tampered after new publication\n")
    with pytest.raises(DataReadinessError, match="source changed"):
        integrity.check_files(tmp_path, admitted)


@pytest.mark.parametrize("name", ["src/market_predictor/unclassified.py", "data/prices.parquet", "configs/settings.json",
    "src/market_predictor/swing/catalyst_lineage.py"])
def test_unknown_or_wrong_parent_paths_never_leave_byte_checks(tmp_path: Path, name: str) -> None:
    declared, policy = _historical(tmp_path, name, declared=False)
    (tmp_path / name).write_bytes(b"changed\n")
    with pytest.raises(DataReadinessError, match="source changed"):
        _historical_sources(tmp_path, declared, policy)


def test_forged_implementation_authority_cannot_reclassify_data(tmp_path: Path) -> None:
    declared, policy = _historical(tmp_path, "data/source.py")
    with pytest.raises(DataReadinessError, match="non-code"):
        _historical_sources(tmp_path, declared, policy)


def test_mismatched_authority_digest_is_not_a_historical_exemption(tmp_path: Path) -> None:
    name = "src/market_predictor/canonical/normalize.py"
    declared, policy = _historical(tmp_path, name)
    authority = tmp_path / "data/replay/_request.json"
    declared[authority.relative_to(tmp_path).as_posix()] = _json(authority,
        dict(schema="market_predictor.predictor_implementation_replay.v1.request", current_implementation_files={name: "f" * 64}))
    (tmp_path / name).write_bytes(b"changed\n")
    with pytest.raises(DataReadinessError, match="source changed"):
        _historical_sources(tmp_path, declared, policy)


def test_memory_limit_is_percentage_only_with_five_gib_process_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(integrity, "assert_memory_budget", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(integrity, "system_memory_snapshot", lambda: SystemMemory(8 * 1024**3, 1024**3))
    integrity.guard(90.0)
    assert calls[0]["hard_budget_gib"] == 5.0
    monkeypatch.setattr(integrity, "system_memory_snapshot", lambda: SystemMemory(10_000, 1_000))
    with pytest.raises(MemoryBudgetError):
        integrity.guard(90.0)


def test_staging_cannot_be_adopted_using_its_own_recomputed_hash(tmp_path: Path) -> None:
    (tmp_path / "_baseline_stage").mkdir()
    _json(tmp_path / "_baseline_stage/_manifest.json", dict(request_sha256="1" * 64, inventory_sha256="2" * 64, files={}))
    with pytest.raises(DataReadinessError, match="independently checkpointed"):
        stage_baseline(tmp_path, None, {}, "1" * 64, lambda: None)  # type: ignore[arg-type]


def test_only_profile_identity_may_change_parent_columns() -> None:
    baseline = pd.DataFrame(dict(decision_id=["a"], feature_profile=["technical_market"], target=[0.25], eligible=[False]))
    result = baseline.assign(feature_profile="technical_relationships")
    for name in ADDITIONS:
        result[name] = None
    assert_parent_parity(baseline, result)
    result.loc[0, "target"] = 0.5
    with pytest.raises(DataReadinessError, match="inherited"):
        assert_parent_parity(baseline, result)


def test_config_rejects_scope_widening_unknown_fields_and_snapshot_shortcuts() -> None:
    pin = dict(path="data/a.json", sha256="a" * 64)
    config = dict(schema_version="market_predictor.return_relationship_publication_config",
        parent_publication=pin, parent_saved_row_verification=pin, feature_config=pin, predictor_failure_facts=pin, strategy_contract=pin)
    assert ReturnRelationshipPublicationPolicy.model_validate(config).maximum_system_used_percent == 90.0
    for extra in ({"source_start": "2019-01-01"}, {"decision_end": "2025-01-01"},
            {"maximum_system_used_percent": 95.0}, {"historical_implementation_snapshot": pin}):
        with pytest.raises(ValidationError):
            ReturnRelationshipPublicationPolicy.model_validate({**config, **extra})


def test_current_closure_includes_ancestor_initializers_and_their_imports(tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "src/market_predictor"
    modules = {
        "__init__.py": "from . import root_dependency\n",
        "root_dependency.py": "VALUE = 1\n",
        "swing/__init__.py": "import market_predictor\n",
        "swing/datasets/__init__.py": "from . import leaf\n",
        "swing/datasets/leaf.py": "VALUE = 2\n",
        "swing/datasets/return_relationship_publication.py": "from market_predictor.swing.datasets import leaf\n",
    }
    for name, content in modules.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(integrity, "__file__", str(package / "swing/datasets/return_relationship_integrity.py"))
    bound = integrity.current_implementation(tmp_path)
    assert set(bound) == {"src/market_predictor/" + name for name in modules}
    (package / "__init__.py").write_text("from . import root_dependency\nCHANGED = True\n", encoding="utf-8")
    assert integrity.current_implementation(tmp_path) != bound
    with pytest.raises(DataReadinessError, match="source changed"):
        integrity.check_files(tmp_path, bound)
