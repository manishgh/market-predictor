"""Rebuild compact issuer-news views against a proven research identity bridge."""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.heavy_jobs import heavy_job_lease
from market_predictor.swing.datasets.initial_fit_issuer_news import _records, _write_frame, _write_json
from market_predictor.swing.datasets.issuer_news_preparation import _build_compact_lineage, _guard, _verified_projection
from market_predictor.swing.datasets.issuer_news_publication import CONFIG_SCHEMA, _check_sources, _decisions, _object, _pin
from market_predictor.swing.datasets.issuer_news_source_clocks import restore_compact_identity_clocks
from market_predictor.universe.issuer_news_identity import build_issuer_news_bridge, map_news_coverage, map_news_relations
from market_predictor.universe.sp500.membership_authority import load_sp500_membership_authority_envelope

SCHEMA = "market_predictor.issuer_news_identity_alignment"


def _immutable_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _object(path) != value:
            raise DataReadinessError("completed identity alignment artifact changed")
    else:
        _write_json(path, value)


def _record_path(root: Path, record: dict[str, Any], pins: dict[str, str]) -> Path:
    if set(record) != {"path", "sha256"}:
        raise DataReadinessError("identity alignment requires explicit path/hash pins")
    path = inside(root, record["path"])
    _pin(path, record["sha256"])
    pins[path.relative_to(root).as_posix()] = record["sha256"]
    return path


def _canonical(root: Path, record: dict[str, Any], kind: str, pins: dict[str, str]) -> pd.DataFrame:
    if set(record) != {"path", "sha256", "manifest_sha256"}:
        raise DataReadinessError("identity alignment requires canonical artifact and manifest pins")
    path = _record_path(root, {key: record[key] for key in ("path", "sha256")}, pins)
    sidecar = manifest_path_for(path)
    _pin(sidecar, record["manifest_sha256"])
    pins[sidecar.relative_to(root).as_posix()] = record["manifest_sha256"]
    frame, _ = load_canonical_artifact(path, expected_type=kind, allow_research=True)
    return frame


def _sec_envelope(root: Path, record: dict[str, Any], pins: dict[str, str]) -> pd.DataFrame:
    authority_path = _record_path(root, record, pins)
    authority = _object(authority_path)
    directory = authority_path.parent
    if (authority.get("schema") != "edge_rebuild.sec_identity_authority.v2"
            or authority.get("state") != "identity_complete" or authority.get("artifact") != "_manifest.json"):
        raise DataReadinessError("identity alignment requires a completed SEC identity authority")
    manifest_path = directory / "_manifest.json"
    _pin(manifest_path, authority["artifact_sha256"])
    manifest = _object(manifest_path)
    request_path = directory / "_request.json"
    request = _object(request_path)
    if (manifest.get("schema") != "edge_rebuild.sec_identity_manifest.v2"
            or request.get("schema") != "edge_rebuild.sec_identity_request.v2"
            or manifest.get("status") != "complete" or manifest.get("request_sha256") != authority["request_sha256"]
            or request.get("request_sha256") != authority["request_sha256"]
            or json_sha256({k: v for k, v in request.items() if k != "request_sha256"}) != authority["request_sha256"]
            or manifest.get("parent_lineage") != request.get("parent_lineage")):
        raise DataReadinessError("SEC identity envelope request/parent binding differs")
    child = manifest["relation_artifact"]
    relation_path = inside(directory, child["path"])
    _pin(relation_path, child["sha256"])
    for path in (manifest_path, request_path, relation_path):
        pins[path.relative_to(root).as_posix()] = file_sha256(path)
    return pd.read_parquet(relation_path)


def _bridge(root: Path, config: dict[str, Any], pins: dict[str, str]) -> pd.DataFrame:
    registry = _canonical(root, config["source_registry"], "security_business_label_coverage", pins)
    legacy = _canonical(root, config["source_memberships"], "memberships", pins)
    target_path = _record_path(root, config["target_membership_authority"], pins)
    memberships, _ = load_sp500_membership_authority_envelope(target_path.parent)
    for name in ("_manifest.json", "_request.json", "memberships.parquet", "memberships.parquet.manifest.json"):
        path = target_path.parent / name
        pins[path.relative_to(root).as_posix()] = file_sha256(path)
    sec = _sec_envelope(root, config["target_sec_identity_authority"], pins)
    sec_request = _object(inside(root, config["target_sec_identity_authority"]["path"]).parent / "_request.json")
    if sec_request["parent_lineage"]["membership_authority"]["authority_sha256"] != config["target_membership_authority"]["sha256"]:
        raise DataReadinessError("SEC identity belongs to another target membership authority")
    correction_path = _record_path(root, config["symbol_corrections"], pins)
    corrections = tomllib.loads(correction_path.read_text(encoding="utf-8"))
    corrected = {row["security_id"] for row in corrections["corrections"]}
    bridge = build_issuer_news_bridge(registry, legacy, memberships, sec, evidence_sha256=json_sha256(pins))
    # Those source queries were independently proven wrong. Only their separately
    # corrected collection may supply these target identities, never an alias bridge.
    return bridge.loc[~bridge.target_security_id.isin(corrected)].reset_index(drop=True)


def _implementation(root: Path) -> dict[str, str]:
    names = ("swing/datasets/issuer_news_identity_alignment.py", "swing/datasets/issuer_news_source_clocks.py",
        "universe/issuer_news_identity.py",
        "swing/datasets/issuer_news_preparation.py", "swing/datasets/issuer_news_publication.py",
        "swing/catalyst_lineage.py", "canonical/reconciliation.py", "canonical/store.py")
    return {f"src/market_predictor/{name}": file_sha256(root / "src/market_predictor" / name) for name in names}


def _mapped_view(view: Path, target: Path, bridge: pd.DataFrame, request_pin: str) -> dict[str, int]:
    manifests = {name: _object(view / name / "_manifest.json") for name in ("collection", "attribution", "sentiment")}
    original_audit = _object(view / "collection/_audit.json")
    records = {name: _records(value) for name, value in manifests.items()}
    ledger, _ = load_canonical_artifact(view / "collection/source_collections.parquet",
        expected_type="source_collections", allow_research=True)
    mapped = map_news_coverage(ledger, bridge)
    coverage = _write_frame(mapped, target / "collection/source_collections.parquet", "source_collections",
        {"identity_alignment_request_sha256": request_pin,
            "original_source_ledger_sha256": manifests["collection"]["source_collections_sha256"]})
    source_blind = set(original_audit["coverage_blindspot_security_ids"])
    blind = set(mapped.loc[mapped.identity_original_security_id.isin(source_blind), "security_id"]) | source_blind
    old_logical = set(manifests["collection"]["verified_logical_chunk_ids"])
    logical = sorted(set(mapped.loc[mapped.identity_original_chunk_id.isin(old_logical), "chunk_id"]))
    new_relations = []
    counts = {"relation_rows": 0, "mapped_relation_rows": 0, "unmapped_relation_rows": 0}
    for chunk, record in records["attribution"].items():
        _guard()
        relations, _ = load_canonical_artifact(Path(record["path"]), expected_type="event_security_relations", allow_research=True)
        restored = restore_compact_identity_clocks(relations, record["original_source_children"])
        aligned = map_news_relations(restored, bridge)
        path = target / "attribution/relations" / f"{chunk}.parquet"
        derived = _write_frame(aligned, path, "event_security_relations", {
            "identity_alignment_request_sha256": request_pin, "original_relations_sha256": record["sha256"],
            "source_event_artifact_sha256": record["source_event_sha256"]})
        derived.update(chunk_id=chunk, original_source_children=record["original_source_children"],
            source_event_sha256=record["source_event_sha256"])
        new_relations.append(derived)
        changed = aligned.target_security_id.ne(relations.target_security_id).sum()
        counts["relation_rows"] += len(aligned)
        counts["mapped_relation_rows"] += int(changed)
        counts["unmapped_relation_rows"] += int(len(aligned) - changed)
    for name, manifest in manifests.items():
        updated = {**manifest, "identity_alignment_request_sha256": request_pin,
            "coverage_blindspot_security_ids": sorted(blind), "verified_logical_chunk_ids": logical}
        if name == "collection":
            updated.update(source_collections_path=coverage["path"], source_collections_sha256=coverage["sha256"])
        elif name == "attribution":
            updated["artifacts"] = new_relations
        _write_json(target / name / "_manifest.json", updated)
    _write_json(target / "collection/_audit.json", {**original_audit,
        "identity_alignment_request_sha256": request_pin, "coverage_blindspot_security_ids": sorted(blind),
        "audit_scope": "verified_research_identity_translation_not_provider_coverage_admission"})
    return counts


def align_monthly_news(*, root: Path, config_path: Path, expected_config_sha256: str,
    output: Path, expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Reuse immutable scores and compact source batches; no downloads or fitting."""
    root = root.resolve()
    output = inside(root, output)
    if not output.is_relative_to(root / "data/research") or output == root / "data/research":
        raise DataReadinessError("aligned news output must be below data/research")
    with heavy_job_lease("align-initial-fit-issuer-news-identities", runtime_dir=root / "data/runtime"):
        _guard()
        pins: dict[str, str] = {}
        config_path = _record_path(root, {"path": str(config_path), "sha256": expected_config_sha256}, pins)
        config = _object(config_path)
        expected = {"schema", "preparation", "source_registry", "source_memberships", "target_membership_authority",
            "target_sec_identity_authority", "symbol_corrections"}
        if set(config) != expected or config["schema"] != SCHEMA:
            raise DataReadinessError("news identity alignment configuration differs")
        preparation_path = _record_path(root, config["preparation"], pins)
        parent = _object(preparation_path)
        if parent.get("status") != "complete_research_only":
            raise DataReadinessError("news alignment requires completed compact preparation")
        if output.is_relative_to(preparation_path.parent) or preparation_path.parent.is_relative_to(output):
            raise DataReadinessError("news alignment output overlaps original preparation")
        publication_path = preparation_path.parent / "monthly-publication.json"
        _pin(publication_path, parent["publication_config_sha256"])
        pins[publication_path.relative_to(root).as_posix()] = parent["publication_config_sha256"]
        publication = _object(publication_path)
        if (publication.get("schema") != CONFIG_SCHEMA or publication.get("cohort_sha256") != parent["request"]["cohort_sha256"]
                or publication.get("rows") != parent["request"]["rows"]
                or set(publication["months"]) != set(parent["months"])):
            raise DataReadinessError("prepared publication population differs from parent")
        pins.update(publication["source_files"])
        bridge = _bridge(root, config, pins)
        pins.update(_implementation(root))
        request = {"schema": SCHEMA, "source_files": pins, "config": config,
            "production_ready": False, "translation_policy": "proven_cik_exact_ticker_interval_only",
            "unproved_identity_policy": "preserve_original_as_unavailable_never_guess"}
        request_pin = json_sha256(request)
        if output.exists():
            if expected_checkpoint_sha256 is None:
                raise DataReadinessError("news alignment resume requires independently pinned checkpoint")
            _pin(output / "_checkpoint.json", expected_checkpoint_sha256)
            state = _object(output / "_checkpoint.json")
            if state.get("request_sha256") != request_pin or _object(output / "_request.json") != request:
                raise DataReadinessError("news alignment resume inputs changed")
        else:
            if expected_checkpoint_sha256 is not None:
                raise DataReadinessError("news alignment resume output is absent")
            output.mkdir(parents=True)
            _write_json(output / "_request.json", request)
            state = {"schema": SCHEMA, "request_sha256": request_pin, "lineages": {}, "source_files": {}}
            _write_json(output / "_checkpoint.json", state)
        _check_sources(root, pins)
        bridge_path = output / "identity_bridge.parquet"
        if bridge_path.exists():
            actual, _ = load_canonical_artifact(bridge_path, expected_type="issuer_news_identity_bridge", allow_research=True)
            pd.testing.assert_frame_equal(actual, bridge)
        else:
            _write_frame(bridge, bridge_path, "issuer_news_identity_bridge", {"identity_alignment_request_sha256": request_pin})
        source_files = {**pins, bridge_path.relative_to(root).as_posix(): file_sha256(bridge_path),
            manifest_path_for(bridge_path).relative_to(root).as_posix(): file_sha256(manifest_path_for(bridge_path)),
            (output / "_request.json").relative_to(root).as_posix(): file_sha256(output / "_request.json")}
        for source in parent["sources"].values():
            _verified_projection(root, source, parent["request_sha256"])
        for month, record in sorted(publication["months"].items()):
            for lineage in record["lineages"]:
                generation = Path(lineage["directory"]).name
                key = f"{month}/{generation}"
                if generation == "corrected":
                    continue
                if key in state["lineages"]:
                    _check_sources(root, state["lineages"][key]["files"])
                    continue
                _guard()
                source_view = inside(root, lineage["source_paths"]["collection_manifest_sha256"]).parent.parent
                target = output / "views" / month / f"{generation}-{uuid4().hex}"
                counts = _mapped_view(source_view, target, bridge, request_pin)
                target_lineage = output / "lineages" / month / target.name
                policy = inside(root, lineage["source_paths"]["policy_sha256"])
                decision_path = inside(root, record["decisions"]["path"])
                _decisions(root, month, record["decisions"])
                _build_compact_lineage(target, decision_path, policy, target_lineage)
                files = {p.relative_to(root).as_posix(): file_sha256(p) for directory in (target, target_lineage)
                    for p in directory.rglob("*") if p.is_file() and not p.name.endswith(".lock")}
                paths = {name: str((target / inside(root, path).relative_to(source_view)).relative_to(root).as_posix())
                    for name, path in lineage["source_paths"].items() if name != "policy_sha256"}
                paths["policy_sha256"] = lineage["source_paths"]["policy_sha256"]
                replacement = {"directory": target_lineage.relative_to(root).as_posix(),
                    "manifest_sha256": file_sha256(target_lineage / "_manifest.json"), "source_paths": paths}
                state["lineages"][key] = {"record": replacement, "files": files, "counts": counts}
                temporary = output / "_checkpoint.pending"
                _write_json(temporary, state)
                temporary.replace(output / "_checkpoint.json")
                print(json.dumps({"completed_lineages": len(state["lineages"]), "latest": key, **counts}), flush=True)
        months = {}
        for month, record in publication["months"].items():
            lineages = []
            for lineage in record["lineages"]:
                key = f"{month}/{Path(lineage['directory']).name}"
                if key in state["lineages"]:
                    updated = state["lineages"][key]
                    source_files.update(updated["files"])
                    lineages.append(updated["record"])
                else:
                    lineages.append(lineage)
            months[month] = {**record, "lineages": lineages}
        _check_sources(root, source_files)
        result = {"schema": CONFIG_SCHEMA, "cohort_sha256": publication["cohort_sha256"],
            "rows": publication["rows"], "source_files": source_files, "months": months}
        final_config = output / "monthly-publication.json"
        _immutable_json(final_config, result)
        manifest = {**state, "status": "complete_research_only", "source_files": source_files,
            "bridge_rows": len(bridge), "rows": publication["rows"], "months": len(months),
            "publication_config_sha256": file_sha256(final_config), "training_eligible": False, "promotion_eligible": False}
        _check_sources(root, source_files)
        _immutable_json(output / "_manifest.json", manifest)
        return {"manifest_sha256": file_sha256(output / "_manifest.json"), "months": len(months),
            "bridge_rows": len(bridge), "publication_config_sha256": manifest["publication_config_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args()
    print(json.dumps(align_monthly_news(root=args.root, config_path=args.config,
        expected_config_sha256=args.expected_config_sha256, output=args.output,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
