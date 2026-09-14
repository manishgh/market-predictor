"""Synthetic pinned sources only; no actual market collection or admission."""
from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.implementation_snapshot import create_implementation_snapshot
from market_predictor.sources.official_documents import collect_official_documents, verify_official_document_collection
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets import feature_history_plan as owner
from market_predictor.swing.datasets import history_archive as collector
from market_predictor.swing.datasets.history_plan_publication import PLAN_SCHEMA, UNIT_COLUMNS
from tests.test_official_document_collection import _fetch, _inventory
from tests.test_swing_history_collection import _FakeSource, _json, _plan, _write_json
from tests.test_swing_symbol_corrections import _write_toml

CONFIG = Path(__file__).resolve().parents[1] / "configs/swing_corrected_feature_history.toml"


@pytest.fixture
def evidence(tmp_path: Path) -> dict[str, Any]:
    parent = _plan(tmp_path, adjustment="raw")
    request = _json(parent / "_request.json")
    request.update(scope="initial_fit_raw_share_acquisition", retained_security_ids=list(owner._ENTITIES))
    _write_json(parent / "_request.json", request)
    authority = _json(parent / "_authority.json")
    authority["request_sha256"] = file_sha256(parent / "_request.json")
    _write_json(parent / "_authority.json", authority)
    inventory = _inventory(4)
    _write_toml(tmp_path / "documents.toml", inventory.model_dump(mode="json"), "documents")
    collect_official_documents(inventory=inventory, output_directory=tmp_path / "documents", fetch=_fetch)
    correction = {"schema_version": "market_predictor.swing_symbol_correction_policy",
        "parent_plan": "plan", "parent_plan_sha256": file_sha256(parent / "_authority.json"),
        "parent_archive": "unused-original-raw", "parent_archive_sha256": "a" * 64,
        "document_inventory": "documents.toml", "document_archive": "documents",
        "document_report_sha256": json_sha256(verify_official_document_collection(tmp_path / "documents", inventory)),
        "corrections": [{"security_id": identity, "ticker": values[0], "provider_symbol": values[1],
            "start_date": str(values[2]), "end_date": "2024-05-28", "parent_unit_id": "swing-daily-" + str(index) * 24,
            "document_ids": [f"filing_{index * 2}", f"filing_{index * 2 + 1}"],
            "record_locators": ["Synthetic identity evidence", "Synthetic transition evidence"],
            "interpretation": "Synthetic test-only reviewed source mapping, not actual issuer evidence."}
            for index, (identity, values) in enumerate(owner._ENTITIES.items())]}
    _write_toml(tmp_path / "corrections.toml", correction, "corrections")
    config = tmp_path / "feature-history.toml"
    config.write_text(CONFIG.read_text(encoding="utf-8").replace("configs/swing_symbol_corrections.toml", "corrections.toml")
        .replace("be63ad9460f80cf87a377b903971c5415bcfefdf5d457308a9a9028d2b5488e3",
            file_sha256(tmp_path / "corrections.toml")), encoding="utf-8")
    return {"root": tmp_path, "config": config, "policy_pin": file_sha256(config), "correction": correction}


def _requirements(evidence: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    return owner.feature_history_requirements(evidence["root"], evidence["config"], evidence["policy_pin"])


def _publish(evidence: dict[str, Any]) -> tuple[Path, str]:
    output = evidence["root"] / "adjusted-plan"
    owner.publish_feature_history_plan(evidence["root"], evidence["config"], evidence["policy_pin"], output)
    return output, file_sha256(output / "_authority.json")


@pytest.fixture
def migrated_plan(evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = evidence["root"]
    package = root / "src/market_predictor"
    for name in ("swing/datasets/feature_history_plan.py", "swing/datasets/history_plan_publication.py",
        "swing/labels/holding_paths.py"):
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# Synthetic historical implementation.\n")
    producer = package / "swing/datasets/feature_history_plan.py"
    monkeypatch.setattr(owner, "__file__", str(producer))
    plan, pin = _publish(evidence)
    archive = root / "adjusted-archive"
    result = collector.collect_swing_history_plan(plan_directory=plan, output_directory=archive,
        source_factory=_FakeSource, provider_symbol_for=lambda ticker: ticker, expected_plan_authority_sha256=pin)
    snapshot = create_implementation_snapshot(root,
        {producer.relative_to(root).as_posix(): file_sha256(producer)}, root / "data/evidence/plan")
    producer.write_bytes(b"# Synthetic current implementation.\n")
    return dict(evidence, plan=plan, archive=archive, result=result, authority=SourcePin(
        path=(plan / "_authority.json").relative_to(root).as_posix(), sha256=pin),
        snapshot=SourcePin(path=snapshot["manifest_path"], sha256=snapshot["manifest_sha256"]))


def test_explicit_snapshot_reconstructs_plan_without_rewriting_archive(migrated_plan: dict[str, Any]) -> None:
    state = migrated_plan
    before = {p.name: p.read_bytes() for p in state["plan"].iterdir() if p.is_file()}
    kwargs = dict(plan_directory=state["plan"], expected_adjustment="all",
        expected_plan_authority_sha256=state["authority"].sha256)
    with pytest.raises(DataReadinessError, match="differs from pinned"):
        collector.load_complete_swing_history_collection(state["archive"], **kwargs)
    assert collector.load_complete_swing_history_collection(state["archive"], **kwargs,
        feature_plan_snapshot=state["snapshot"]) == state["result"]
    proof = owner.verify_feature_plan_replay(root=state["root"], authority=state["authority"],
        implementation_snapshot=state["snapshot"])
    assert state["snapshot"].path in proof
    assert any(name.endswith(".bin") for name in proof)
    assert {p.name: p.read_bytes() for p in state["plan"].iterdir() if p.is_file()} == before


@pytest.mark.parametrize("poison", ["keys", "digest", "request", "units", "blob"])
def test_snapshot_cannot_hide_changed_plan_semantics(migrated_plan: dict[str, Any], poison: str) -> None:
    state = migrated_plan
    request = _json(state["plan"] / "_request.json")
    manifest = _json(state["plan"] / "_manifest.json")
    units = pd.read_csv(state["plan"] / "daily_bar_units.csv", dtype=str)
    if poison == "keys":
        request["implementation_files"].pop("swing/labels/holding_paths.py")
    elif poison == "digest":
        request["implementation_files"]["swing/datasets/feature_history_plan.py"] = "f" * 64
    elif poison == "request":
        request["required_sessions"] += 1
    elif poison == "units":
        units.loc[0, "ticker"] = "OTHER"
    else:
        snapshot_path = state["root"] / state["snapshot"].path
        entry = next(iter(_json(snapshot_path)["files"].values()))
        blob = snapshot_path.parent / entry["blob"]
        blob.write_bytes(b"Changed archived bytes")
    with pytest.raises(DataReadinessError):
        owner.validate_feature_history_collection_plan(directory=state["plan"], request=request,
            manifest=manifest, units=units, implementation_snapshot=state["snapshot"])


def test_feature_snapshot_cannot_authorize_another_plan_scope(migrated_plan: dict[str, Any]) -> None:
    with pytest.raises(DataReadinessError, match="another acquisition scope"):
        collector._load_verified_plan(migrated_plan["root"] / "plan", feature_plan_snapshot=migrated_plan["snapshot"])


def test_two_full_adjusted_streams_not_historical_ticker_or_membership(evidence: dict[str, Any]) -> None:
    request, manifest, units = _requirements(evidence)
    assert request["schema"] == manifest["schema"] == PLAN_SCHEMA
    assert request["scope"] == manifest["scope"] == owner.FEATURE_HISTORY_SCOPE
    assert list(units.columns) == list(UNIT_COLUMNS)
    assert units.ticker.tolist() == ["FI", "SATS"]
    assert units.security_id.tolist() == ["cik:0000798354", "cik:0001415404"]
    assert units.start_date.eq("2018-05-29").all() and units.end_date.eq("2024-05-28").all()
    assert units.role.eq("stock").all() and len(units) == 2
    assert request["provider_symbols"] == {"FI": "FI", "SATS": "SATS"}
    assert request["asof_policy"] == "inclusive_unit_end_date_entity_mapping_not_ownership"
    assert request["decision_start"] == "2019-07-09"
    assert request["query_symbol_policy"] == "asof_entity_stream_label_not_historical_exchange_ticker"
    assert request["feature_basis"] == "single_full_adjusted_stream_per_entity_no_raw_splice"
    assert request["required_sessions"] > 1500
    assert request["membership_authority"] == _json(evidence["root"] / "plan/_request.json")["membership_authority"]
    assert manifest["daily_bars"]["adjustment"] == "all" and manifest["daily_bars"]["benchmark_units"] == 0
    for name in ("outcomes_read", "ownership_admitted", "bar_coverage_verified", "feature_eligible", "label_eligible",
            "accounting_eligible", "promotion_eligible", "historical_availability_proven"):
        assert manifest[name] is False


def test_reconstruction_opens_no_price_or_feature_parquet(evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("plan attempted numerical input access")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    assert _requirements(evidence)[2].shape == (2, 5)


def test_immutable_shared_publication_and_owner_replay(evidence: dict[str, Any]) -> None:
    output, pin = _publish(evidence)
    request, manifest = _json(output / "_request.json"), _json(output / "_manifest.json")
    units = pd.read_csv(output / "daily_bar_units.csv", dtype=str)
    owner.validate_feature_history_collection_plan(directory=output, request=request, manifest=manifest, units=units)
    assert file_sha256(output / "_authority.json") == pin
    with pytest.raises(DataReadinessError, match="must be new"):
        owner.publish_feature_history_plan(evidence["root"], evidence["config"], evidence["policy_pin"], output)


@pytest.mark.parametrize("poison", ["provider", "asof", "adjustment", "identity", "warmup", "heldout", "decision", "benchmark"])
def test_owner_rejects_rehashed_plan_changes(evidence: dict[str, Any], poison: str) -> None:
    request, manifest, units = _requirements(evidence)
    request, manifest, units = copy.deepcopy(request), copy.deepcopy(manifest), units.copy()
    if poison == "provider":
        request["provider_symbols"]["SATS"] = "ECHO"
    elif poison == "asof":
        request["asof_policy"] = "current_symbol"
    elif poison == "adjustment":
        manifest["daily_bars"]["adjustment"] = "raw"
    elif poison == "identity":
        units.loc[0, "security_id"] = "wrong-owner"
    elif poison == "warmup":
        units.loc[0, "start_date"] = "2019-07-09"
    elif poison == "heldout":
        units.loc[0, "end_date"] = "2024-05-29"
    elif poison == "decision":
        request["decision_start"] = "2018-05-29"
    else:
        manifest["daily_bars"]["benchmark_units"] = 1
    with pytest.raises(DataReadinessError, match="differs from pinned"):
        owner.validate_feature_history_collection_plan(directory=evidence["root"], request=request, manifest=manifest, units=units)


@pytest.mark.parametrize(("old", "new"), [("2018-05-29", "2018-05-28"), ("2019-07-09", "2018-05-29"),
    ("2024-05-28", "2024-05-29"), ('adjustment = "all"', 'adjustment = "raw"')])
def test_config_cannot_change_frozen_boundaries(evidence: dict[str, Any], old: str, new: str) -> None:
    config = evidence["config"]
    config.write_text(config.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    evidence["policy_pin"] = file_sha256(config)
    with pytest.raises((DataReadinessError, ValidationError)):
        _requirements(evidence)


def test_configuration_requires_external_pin(evidence: dict[str, Any]) -> None:
    evidence["policy_pin"] = "f" * 64
    with pytest.raises(DataReadinessError, match="independent pin"):
        _requirements(evidence)


@pytest.mark.parametrize("poison", ["provider", "transition", "document", "document_pin"])
def test_reviewed_correction_source_binding(evidence: dict[str, Any], poison: str) -> None:
    correction = evidence["correction"]
    if poison == "provider":
        correction["corrections"][0]["provider_symbol"] = "ECHO"
    elif poison == "transition":
        correction["corrections"][1]["start_date"] = "2023-06-06"
    elif poison == "document":
        correction["corrections"][0]["document_ids"][0] = "absent"
    else:
        correction["document_report_sha256"] = "f" * 64
    path = evidence["root"] / "corrections.toml"
    old = file_sha256(path)
    _write_toml(path, correction, "corrections")
    config = evidence["config"]
    config.write_text(config.read_text(encoding="utf-8").replace(old, file_sha256(path)), encoding="utf-8")
    evidence["policy_pin"] = file_sha256(config)
    with pytest.raises(DataReadinessError):
        _requirements(evidence)


def test_parent_request_mutation_does_not_rebind_retained_population(evidence: dict[str, Any]) -> None:
    path = evidence["root"] / "plan/_request.json"
    changed = _json(path)
    changed["retained_security_ids"] = []
    _write_json(path, changed)
    with pytest.raises(DataReadinessError, match="file pin"):
        _requirements(evidence)


def test_output_cannot_overlap_preserved_raw_input(evidence: dict[str, Any]) -> None:
    with pytest.raises(DataReadinessError, match="overlaps"):
        owner.publish_feature_history_plan(evidence["root"], evidence["config"], evidence["policy_pin"],
            evidence["root"] / "unused-original-raw/adjusted-plan")


def test_general_collector_uses_pinned_full_history_asof_and_replays(evidence: dict[str, Any]) -> None:
    output, pin = _publish(evidence)
    source = _FakeSource()
    archive = evidence["root"] / "adjusted-archive"
    result = collector.collect_swing_history_plan(plan_directory=output, output_directory=archive,
        source_factory=lambda: source, provider_symbol_for=lambda ticker: ticker, expected_plan_authority_sha256=pin)
    assert result["status"] == "complete"
    assert len(source.calls) == 2
    assert {call["symbol"] for call in source.calls} == {"FI", "SATS"}
    assert all(call["asof"] == date(2024, 5, 28) and call["adjustment"] == "all" for call in source.calls)
    assert all(call["start"].date() == date(2018, 5, 29) for call in source.calls)
    source.calls.clear()
    assert collector.load_complete_swing_history_collection(archive, plan_directory=output,
        expected_adjustment="all", expected_plan_authority_sha256=pin) == result
    assert source.calls == []
    # Transport completion is deliberately not an exact-session or feature pass.
    assert result["total_rows"] == 2
    assert _json(output / "_manifest.json")["bar_coverage_verified"] is False
