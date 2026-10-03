"""Price-source orchestration with independent archive verifiers as fixture doubles."""
from __future__ import annotations

import json
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets import corrected_outcomes as module

REPO = Path(__file__).resolve().parents[1]


def _write(root: Path, relative: str, value: str | dict[str, Any]) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return path


def _pin(root: Path, path: Path) -> SourcePin:
    return SourcePin(path=path.relative_to(root).as_posix(), sha256=file_sha256(path))


def _authority(root: Path, directory: str, request: dict[str, Any], manifest: dict[str, Any]) -> SourcePin:
    request_path = _write(root, f"{directory}/_request.json", request)
    manifest_path = _write(root, f"{directory}/_manifest.json", manifest)
    path = _write(root, f"{directory}/_authority.json", {
        "request_sha256": file_sha256(request_path), "artifact_sha256": file_sha256(manifest_path)})
    return _pin(root, path)


@pytest.fixture
def prices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path
    config_text = (REPO / "configs/swing_corrected_outcomes.toml").read_text(encoding="utf-8")
    original = tomllib.loads(config_text)
    membership_path = root / "metadata/memberships.parquet"
    membership_path.parent.mkdir()
    membership = {name: "fixture" for name in module.MEMBERSHIP_COLUMNS}
    membership.update(security_id="cik:0001415404", ticker="ECHO", sector="technology", primary_benchmark="XLK",
        effective_from_utc=pd.Timestamp("2019-01-01", tz="UTC"), effective_to_utc=pd.NaT,
        available_at_utc=pd.Timestamp("2019-01-01", tz="UTC"))
    pd.DataFrame([membership]).to_parquet(membership_path, index=False)
    manifest_path = _write(root, "metadata/panel.json", {"files": []})
    preflight = _write(root, "metadata/preflight.json", {"request": {
        "membership_path": "metadata/memberships.parquet", "parent_manifest_path": "metadata/panel.json"}})
    parent_config = _write(root, "configs/parent.toml", 'preflight_path = "metadata/preflight.json"\n')
    parent_request = {"source_files": {path.relative_to(root).as_posix(): file_sha256(path)
        for path in (membership_path, manifest_path, preflight)},
        "retained_security_ids": [f"security:{index}" for index in range(585)] + ["cik:0001415404"],
        "excluded_security_ids": [f"excluded:{index}" for index in range(45)],
        "decision_start": "2019-07-09", "initial_fit_end": "2024-05-28"}
    parent = _authority(root, "plans/parent", parent_request, {})
    raw_records = []
    archive_pins = []
    for name in ("parent", "corrected"):
        directory = f"archives/{name}"
        bars = _write(root, f"{directory}/bars.bin", "synthetic raw observations, independent verifier owns parsing")
        page = _write(root, f"{directory}/page.json", {})
        body = _write(root, f"{directory}/body.bin", "synthetic provider bytes")
        unit = _write(root, f"{directory}/unit.json", {"pages": [{
            "raw_path": "page.json", "raw_sha256": file_sha256(page),
            "transport": {"body_path": "body.bin", "metadata": {"sha256": file_sha256(body)}}}]})
        record = {"bars_path": "bars.bin", "bars_sha256": file_sha256(bars),
            "unit_manifest_path": "unit.json", "unit_manifest_sha256": file_sha256(unit), "provider_symbol": "SATS"}
        raw_records.append(record)
        archive_pins.append(_authority(root, directory, {}, {"unit_artifacts": [record]}))
    mapping = tomllib.loads((REPO / "configs/swing_symbol_corrections.toml").read_text(encoding="utf-8"))
    documents = {"status": "collected_unreviewed", "documents": []}
    mapping.update(parent_plan="plans/parent", parent_plan_sha256=parent.sha256,
        parent_archive="archives/parent", parent_archive_sha256=archive_pins[0].sha256,
        document_inventory="documents/inventory.toml", document_archive="documents/archive",
        document_report_sha256=json_sha256(documents))
    mapping_text = (REPO / "configs/swing_symbol_corrections.toml").read_text(encoding="utf-8")
    old_mapping = tomllib.loads(mapping_text)
    for key, value in mapping.items():
        if isinstance(value, str) and value != old_mapping[key]:
            mapping_text = mapping_text.replace(json.dumps(old_mapping[key]), json.dumps(value))
    mapping_path = _write(root, "configs/corrections.toml", mapping_text)
    _write(root, mapping["document_inventory"], "fixture inventory")
    _write(root, "documents/archive/_request.json", {})
    _write(root, "documents/archive/receipt.json", {"fixture": "reviewed"})
    correction = _authority(root, "plans/corrected", {"policy": mapping}, {})
    selected = {"correction_plan": "plans/corrected", "correction_plan_sha256": correction.sha256,
        "correction_archive": "archives/corrected", "correction_archive_sha256": archive_pins[1].sha256,
        "policy_sha256": file_sha256(mapping_path),
        "segments": [{"archive": "archives/corrected", "artifact": raw_records[1]}]}
    selection_path = _write(root, "metadata/selection.json", selected)
    for key, path in (("source_selection", selection_path), ("parent_config", parent_config),
            ("symbol_corrections", mapping_path)):
        config_text = config_text.replace(json.dumps(original[key]["path"]), json.dumps(path.relative_to(root).as_posix()))
        config_text = config_text.replace(original[key]["sha256"], file_sha256(path))
    config = _write(root, "configs/outcomes.toml", config_text)
    config_sha = file_sha256(config)
    policy = module.load_corrected_outcome_policy(root, config, config_sha)
    calls: list[str] = []

    @contextmanager
    def verified_plan(actual_root: Path, actual_config: Path, directory: Path, *, expected_plan_sha256: str) -> Iterator[dict[str, Any]]:
        assert actual_root == root and actual_config == parent_config and directory == root / "plans/parent"
        assert expected_plan_sha256 == parent.sha256
        calls.append("lease_enter")
        try:
            yield {"requirements": {"in_window_decisions": 1}}
        finally:
            calls.append("lease_exit")

    def reconstruct(**kwargs: Any) -> dict[str, Any]:
        assert kwargs == {"root": root, "correction_plan": root / "plans/corrected",
            "correction_plan_sha256": correction.sha256, "correction_archive": root / "archives/corrected",
            "correction_archive_sha256": archive_pins[1].sha256, "loader": module.load_complete_swing_history_collection}
        calls.append("reconstruct")
        return json.loads(json.dumps(selected))

    monkeypatch.setattr(module, "verified_initial_fit_raw_share_plan", verified_plan)
    monkeypatch.setattr(module, "reconstruct_symbol_corrected_sources", reconstruct)
    monkeypatch.setattr(module, "load_official_document_inventory", lambda path: object())
    monkeypatch.setattr(module, "verify_official_document_collection", lambda *args: documents)
    monkeypatch.setattr(module, "_guard", lambda: None)
    return {"root": root, "config": config, "config_sha256": config_sha, "policy": policy,
        "calls": calls, "selected": selected, "parent_request": parent_request, "documents": documents}


def _arguments(prices: dict[str, Any]) -> dict[str, Any]:
    return {key: prices[key] for key in ("root", "config", "config_sha256", "policy")}


def test_price_context_binds_sources_and_holds_lease_without_target_dependencies(
    prices: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("price context touched a target-admission dependency")

    for name in ("load_corporate_action_evidence", "prepare_corporate_action_scope", "verify_action_scope_documents",
            "reconcile_actions", "load_trade_simulation_context", "load_swing_research_contract", "build_ordinary_holding"):
        monkeypatch.setattr(module, name, forbidden)
    with module.verified_corrected_price_sources(**_arguments(prices)) as source:
        assert prices["calls"] == ["lease_enter", "reconstruct"]
        assert "action_index" not in source
        assert len(source["request"]["retained_security_ids"]) == 586
        assert set(source["memberships"].ticker) == {"ECHO", "SATS"}
        assert {"archives/parent/body.bin", "archives/corrected/page.json", "documents/archive/receipt.json",
            "metadata/memberships.parquet"}.issubset(source["source_files"])
        assert not any("action" in path or "labels/" in path for path in source["source_files"])
    assert prices["calls"][-1] == "lease_exit"


@pytest.mark.parametrize("path", ["archives/parent/body.bin", "archives/corrected/page.json",
    "documents/archive/receipt.json", "metadata/memberships.parquet"])
def test_price_context_rechecks_every_source_after_yield(prices: dict[str, Any], path: str) -> None:
    with pytest.raises(DataReadinessError, match="source changed"):
        with module.verified_corrected_price_sources(**_arguments(prices)) as source:
            source["source_files"].clear()
            (prices["root"] / path).write_bytes(b"poisoned after verification")
    assert prices["calls"][-1] == "lease_exit"


def test_price_context_requires_exact_selection_reconstruction(prices: dict[str, Any]) -> None:
    prices["selected"]["unexpected"] = True
    with pytest.raises(DataReadinessError, match="selection differs from independent reconstruction"):
        with module.verified_corrected_price_sources(**_arguments(prices)):
            pytest.fail("poisoned reconstruction yielded")


def test_price_context_rejects_population_drift(prices: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    pinned_object = module.pinned_object

    def poisoned(path: Path, expected: str | None = None) -> dict[str, Any]:
        result = pinned_object(path, expected)
        if path == prices["root"] / "plans/parent/_request.json":
            result["retained_security_ids"] = result["retained_security_ids"][:-1]
        return result

    monkeypatch.setattr(module, "pinned_object", poisoned)
    with pytest.raises(DataReadinessError, match="586-security"):
        with module.verified_corrected_price_sources(**_arguments(prices)):
            pytest.fail("changed population yielded")


def test_price_context_rejects_unreviewed_decision_document(prices: dict[str, Any]) -> None:
    path = prices["config"]
    text = path.read_text(encoding="utf-8").replace('"echostar_pre_window_symbol"', '"unreviewed_document"')
    path.write_text(text, encoding="utf-8")
    prices["config_sha256"] = file_sha256(path)
    prices["policy"] = module.load_corrected_outcome_policy(prices["root"], path, prices["config_sha256"])
    with pytest.raises(DataReadinessError, match="unreviewed source document"):
        with module.verified_corrected_price_sources(**_arguments(prices)):
            pytest.fail("unreviewed correction yielded")


def test_price_context_rejects_policy_substitution(prices: dict[str, Any]) -> None:
    prices["policy"] = prices["policy"].model_copy(update={"decision_corrections": ()})
    with pytest.raises(DataReadinessError, match="pinned configuration"):
        with module.verified_corrected_price_sources(**_arguments(prices)):
            pytest.fail("substituted policy yielded")


def _target_arguments(prices: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = prices["root"]
    action = _write(root, "target/action.toml", 'schema_version = "fixture"\n')
    research = _write(root, "target/research.toml", "research")
    simulation = _write(root, "target/simulation.toml", "simulation")
    policy = prices["policy"].model_copy(update={"action_config": _pin(root, action),
        "research_contract": _pin(root, research), "simulation_policy": _pin(root, simulation),
        "official_windows": (), "reviewed_cash_distribution_scopes": ()})
    symbols = {f"SYM{index}" for index in range(565)} | module.SUCCESSORS
    source = {"source_files": {}, "selection": {"segments": [{"artifact": {"provider_symbol": s}} for s in symbols]}}

    @contextmanager
    def price_source(*args: Any) -> Iterator[dict[str, Any]]:
        yield source

    monkeypatch.setattr(module, "verified_corrected_price_sources", price_source)
    scope = {"policy": {"source_selection": policy.source_selection.model_dump(mode="json")}, "tickers": sorted(symbols)}
    monkeypatch.setattr(module, "prepare_corporate_action_scope", lambda *args: scope)
    request = _write(root, f"{policy.action_archive}/_request.json", {"request_sha256": "a" * 64,
        "policy": scope["policy"], "tickers": sorted(symbols)})
    evidence = SimpleNamespace(request_sha256="a" * 64,
        source_files={request.relative_to(root).as_posix(): file_sha256(request)}, recheck=lambda root: None)
    return {**_arguments(prices), "policy": policy, "evidence": evidence, "scope": scope}


def test_outcome_wrapper_rejects_old_selection_even_with_identical_tickers(
    prices: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _target_arguments(prices, monkeypatch)
    args.pop("scope")["policy"]["source_selection"]["sha256"] = "0" * 64
    with pytest.raises(DataReadinessError, match="scope does not bind"):
        with module.verified_corrected_research_sources(**args):
            pytest.fail("old action selection yielded")


@pytest.mark.parametrize("poison", ["missing_action", "mismatched_action", "missing_research", "missing_simulation"])
def test_outcome_wrapper_keeps_target_dependency_gates(
    prices: dict[str, Any], monkeypatch: pytest.MonkeyPatch, poison: str,
) -> None:
    args = _target_arguments(prices, monkeypatch)
    args.pop("scope")
    if poison == "mismatched_action":
        args["evidence"].request_sha256 = "b" * 64
    else:
        path = {"missing_action": f"{args['policy'].action_archive}/_request.json",
            "missing_research": "target/research.toml", "missing_simulation": "target/simulation.toml"}[poison]
        (prices["root"] / path).unlink()
    with pytest.raises((DataReadinessError, FileNotFoundError)):
        with module.verified_corrected_research_sources(**args):
            pytest.fail("invalid target authority yielded")


def test_outcome_wrapper_admits_actions_only_after_all_target_checks(
    prices: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _target_arguments(prices, monkeypatch)
    args.pop("scope")
    calls: list[str] = []
    args["evidence"].recheck = lambda root: calls.append("evidence")
    document = _write(prices["root"], "target/entitlement.txt", "reviewed scope")

    def scopes(root: Path, scopes: Any) -> dict[str, str]:
        calls.append("scopes")
        return {document.relative_to(root).as_posix(): file_sha256(document)}

    def reconcile(*args: Any) -> tuple[dict[str, Any], tuple[Any, ...]]:
        calls.append("reconcile")
        return {}, ()

    monkeypatch.setattr(module, "verify_action_scope_documents", scopes)
    monkeypatch.setattr(module, "reconcile_actions", reconcile)
    with module.verified_corrected_research_sources(**args) as source:
        assert calls == ["evidence", "scopes", "reconcile"]
        assert source["action_index"] == ({}, ())
        assert "target/entitlement.txt" in source["source_files"]
    assert calls == ["evidence", "scopes", "reconcile", "evidence"]
