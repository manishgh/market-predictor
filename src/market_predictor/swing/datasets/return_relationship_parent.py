"""Saved-parent integrity separated from freshly executed implementation identity."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import inside
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts.return_relationship_publication import ReturnRelationshipPublicationPolicy
from market_predictor.swing.datasets.return_relationship_integrity import check_files, pins, read_object
from market_predictor.swing.features.panel import swing_model_feature_columns

_REVIEWED_PARENT = "63574b3b55cd30adaebf62b4040f92a2030e6dbb17772014afb3e3c168158256"
_REVIEWED_REQUEST = "4c184267b7923cb30e33ff967221bfe1769dd803a3f6879f3ba98527846260a5"
_REVIEWED_LOCAL_CODE = (
    "swing/datasets/research_dataset.py", "swing/datasets/predictor_abstention_derivation.py",
    "swing/features/research_ablation.py", "swing/features/research_partition.py", "swing/features/research_join.py",
    "swing/features/catalyst_aggregates.py", "swing/features/catalyst_decision_authority.py",
    "swing/features/catalyst_decision_identity.py", "swing/features/cross_sectional.py", "swing/features/pipeline.py",
    "swing/catalyst_lineage.py",
)

@dataclass(frozen=True)
class VerifiedRelationshipParent:
    path: Path
    manifest: dict[str, Any]
    request: dict[str, Any]
    source_files: dict[str, str]
    historical_implementation_files: dict[str, str]
    model_columns: tuple[str, ...]
    availability_columns: dict[str, str]


def _historical_sources(root: Path, declared: dict[str, str], policy: ReturnRelationshipPublicationPolicy,
) -> tuple[dict[str, str], dict[str, str]]:
    source = pins(root, declared)
    declarations: dict[str, str] = {}
    authorities = {
        "market_predictor.research_predictor_request": "implementation_files",
        "market_predictor.outcome_implementation_replay_request": "current_implementation_files",
        "market_predictor.predictor_implementation_replay.v1.request": "current_implementation_files",
    }
    for name, digest in source.items():
        if not name.endswith("/_request.json"):
            continue
        document = read_object(inside(root, name), digest)
        field = authorities.get(document.get("schema", ""))
        if field is None:
            continue
        for code, old_digest in pins(root, document[field]).items():
            if not code.startswith("src/market_predictor/") or not code.endswith(".py"):
                raise DataReadinessError("historical implementation declaration contains a non-code source")
            if source.get(code) == old_digest:
                declarations[code] = old_digest
    if policy.parent_publication.sha256 == _REVIEWED_PARENT:
        parent = read_object(inside(root, policy.parent_publication.path), _REVIEWED_PARENT)
        if parent.get("request_sha256") != _REVIEWED_REQUEST:
            raise DataReadinessError("historical classification is bound to another parent request")
        for relative in _REVIEWED_LOCAL_CODE:
            name = "src/market_predictor/" + relative
            if name not in source:
                raise DataReadinessError("reviewed historical declaration is missing from frozen parent")
            declarations[name] = source[name]
    # Only independently declared historical code leaves the current-byte checks.
    # No equivalence or numerical replay claim follows from this disposition.
    live = pins(root, {name: digest for name, digest in source.items() if name not in declarations})
    check_files(root, live)
    return live, declarations


def verify_parent(root: Path, policy: ReturnRelationshipPublicationPolicy) -> VerifiedRelationshipParent:
    path = inside(root, policy.parent_publication.path)
    if path.name != "_manifest.json":
        raise DataReadinessError("relationship parent must be a completed manifest")
    manifest = read_object(path, policy.parent_publication.sha256)
    request_path = path.parent / "_request.json"
    request = read_object(request_path, manifest["request_sha256"])
    receipt = read_object(inside(root, policy.parent_saved_row_verification.path), policy.parent_saved_row_verification.sha256)
    months = manifest.get("months")
    expected_months = [f"{year:04d}-{month:02d}" for year in range(2019, 2025) for month in range(1, 13)
        if "2019-07" <= f"{year:04d}-{month:02d}" <= "2024-05"]
    if (manifest.get("schema") != "market_predictor.research_join"
            or manifest.get("status") != "complete_research_only" or manifest.get("exclusions_added") != []
            or manifest.get("training_eligible") is not False or manifest.get("promotion_eligible") is not False
            or request.get("schema") != "market_predictor.research_join_request"
            or request.get("rows") != manifest.get("rows") or request.get("historical_first_seen_proven") is not False
            or not isinstance(months, dict) or sorted(months) != expected_months
            or type(manifest.get("rows")) is not int or manifest["rows"] < 1
            or receipt.get("status") != "passed" or receipt.get("manifest_sha256") != policy.parent_publication.sha256
            or receipt.get("scope") != "published_join_population_clocks_original_targets"
            or receipt.get("unique_decisions") != manifest["rows"] or receipt.get("months") != len(months)
            or receipt.get("matched_profile_population") is not True or receipt.get("original_outcome_values_exact") is not True
            or receipt.get("outcome_filtered_rows") != 0 or receipt.get("training_eligible") is not False
            or receipt.get("promotion_eligible") is not False):
        raise DataReadinessError("relationship parent or original saved-row receipt differs")
    declared = pins(root, request["source_files"], request["decision_source_files"])
    for pin in (policy.feature_config, policy.predictor_failure_facts, policy.strategy_contract):
        if declared.get(inside(root, pin.path).relative_to(root).as_posix()) != pin.sha256:
            raise DataReadinessError("relationship configuration substitutes a parent source authority")
    live, historical = _historical_sources(root, declared, policy)
    names = tuple(swing_model_feature_columns(contract=load_strategy_contract(inside(root, policy.strategy_contract.path)),
        catalyst=False))
    if len(names) != 120:
        raise DataReadinessError("relationship baseline must have exactly 120 ordered features")
    clocks: dict[str, str] | None = None
    rows = 0
    children: dict[str, str] = {}
    for month, record in sorted(months.items()):
        if set(record["profiles"]) != {"technical_market", "catalyst_full"} or type(record["rows"]) is not int:
            raise DataReadinessError("relationship parent monthly inventory differs")
        rows += record["rows"]
        for profile, child in record["profiles"].items():
            if child["path"] != f"{month}/{profile}.parquet":
                raise DataReadinessError("relationship parent child path differs")
            child_path = inside(path.parent, child["path"])
            children.update({child_path.relative_to(root).as_posix(): child["sha256"],
                manifest_path_for(child_path).relative_to(root).as_posix(): child["manifest_sha256"]})
            sidecar = read_object(manifest_path_for(child_path), child["manifest_sha256"])
            if (sidecar.get("artifact_type") != "swing_research_join" or sidecar.get("artifact_sha256") != child["sha256"]
                    or sidecar.get("rows") != record["rows"] or sidecar.get("production_ready") is not False
                    or sidecar.get("inputs") != {"request_sha256": manifest["request_sha256"]}):
                raise DataReadinessError("relationship parent child lineage differs")
            if profile == "technical_market":
                actual = child["availability_columns"]
                if (child["model_columns"] != list(names) or not isinstance(actual, dict)
                        or not set(names).issubset(actual) or (clocks is not None and actual != clocks)
                        or not set(actual.values()).issubset(sidecar["columns"])):
                    raise DataReadinessError("relationship parent feature ordering or clocks differ")
                clocks = dict(actual)
    if rows != manifest["rows"] or clocks is None:
        raise DataReadinessError("relationship parent monthly totals differ")
    live = pins(root, live, children, {policy.parent_publication.path: policy.parent_publication.sha256,
        request_path.relative_to(root).as_posix(): manifest["request_sha256"],
        policy.parent_saved_row_verification.path: policy.parent_saved_row_verification.sha256})
    check_files(root, live)
    return VerifiedRelationshipParent(path, manifest, request, live, historical, names, clocks)


def load_parent_month(parent: VerifiedRelationshipParent, month: str, columns: list[str] | None = None) -> Any:
    child = parent.manifest["months"][month]["profiles"]["technical_market"]
    path = inside(parent.path.parent, child["path"])
    if file_sha256(path) != child["sha256"] or file_sha256(manifest_path_for(path)) != child["manifest_sha256"]:
        raise DataReadinessError("relationship parent child changed before read")
    frame, _ = load_canonical_artifact(path, expected_type="swing_research_join", allow_research=True, columns=columns)
    if file_sha256(path) != child["sha256"]:
        raise DataReadinessError("relationship parent changed during read")
    return frame
